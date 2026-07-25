from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from numpy import ndarray
from PIL import Image


def iter_point_batches(points, points_per_batch):
    """Yield consecutive slices of `points` of length <= points_per_batch."""
    for start in range(0, len(points), points_per_batch):
        yield points[start : start + points_per_batch]


class AutomaticMaskGeneration:
    """Generate and convert prompt-free SAM masks."""

    def __init__(
        self,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.95,
        stability_score_offset=1.0,
        box_nms_thresh=0.7,
        mask_threshold=0.0,
        points_per_batch=64,
        min_mask_region_area=0,
    ):
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.mask_threshold = mask_threshold
        self.points_per_batch = points_per_batch
        self.min_mask_region_area = min_mask_region_area

    @staticmethod
    def build_point_grid(points_per_side, height, width):
        """Return an (N, 2) float32 grid of (x, y) pixel coordinates."""
        fraction = (np.arange(points_per_side) + 0.5) / points_per_side
        grid_x, grid_y = np.meshgrid(fraction * width, fraction * height)
        grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        return grid.astype(np.float32)

    @staticmethod
    def _stability_score(logits, mask_threshold, offset):
        """Calculate mask stability under high and low logit thresholds."""
        high = logits > (mask_threshold + offset)
        low = logits > (mask_threshold - offset)
        intersection = high.sum(axis=-1, dtype=np.int16).sum(
            axis=-1, dtype=np.int32
        )
        union = low.sum(axis=-1, dtype=np.int16).sum(axis=-1, dtype=np.int32)
        return np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float64),
            where=union > 0,
        )

    @staticmethod
    def _mask_to_box(mask):
        """Return an XYXY bounding box, or None for an empty mask."""
        rows = np.flatnonzero(np.any(mask, axis=1))
        columns = np.flatnonzero(np.any(mask, axis=0))
        if not len(rows) or not len(columns):
            return None
        return np.array(
            [columns[0], rows[0], columns[-1], rows[-1]],
            dtype=np.float32,
        )

    @staticmethod
    def mask_to_rle(mask):
        """Encode a row-major boolean mask as an uncompressed RLE."""
        flattened = np.asarray(mask, dtype=np.bool_).reshape(-1)
        changes = np.flatnonzero(flattened[1:] != flattened[:-1]) + 1
        boundaries = np.concatenate(([0], changes, [flattened.size]))
        counts = np.diff(boundaries).astype(np.int32)
        if flattened[0]:
            counts = np.concatenate((np.zeros(1, dtype=np.int32), counts))
        return {"size": mask.shape, "counts": counts}

    @staticmethod
    def rle_to_mask(rle):
        """Decode an RLE produced by `mask_to_rle` into a boolean mask."""
        counts = np.asarray(rle["counts"])
        values = np.arange(len(counts), dtype=np.int64) % 2 == 1
        mask = np.repeat(values, counts)
        return mask.reshape(rle["size"])

    @staticmethod
    def _remove_small_regions(mask, area_threshold, mode):
        """Remove small disconnected regions or holes from a mask."""
        correct_holes = mode == "holes"
        working_mask = np.logical_xor(correct_holes, mask).astype(np.uint8)
        label_count, regions, stats, _ = cv2.connectedComponentsWithStats(
            working_mask, 8
        )
        sizes = stats[1:, cv2.CC_STAT_AREA]
        small_regions = [
            index + 1
            for index, size in enumerate(sizes)
            if size < area_threshold
        ]
        if not small_regions:
            return mask, False

        fill_labels = [0, *small_regions]
        if not correct_holes:
            fill_labels = [
                index
                for index in range(label_count)
                if index not in fill_labels
            ]
            if not fill_labels:
                fill_labels = [int(np.argmax(sizes)) + 1]
        return np.isin(regions, fill_labels), True

    @staticmethod
    def _box_nms(boxes, scores, iou_threshold, should_stop=None):
        """Return indices kept by score-ordered box IoU NMS."""
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        x_min, y_min, x_max, y_max = boxes.T
        areas = np.maximum(0.0, x_max - x_min) * np.maximum(0.0, y_max - y_min)
        order = np.argsort(-scores, kind="stable")
        keep = []

        while order.size:
            if should_stop is not None and should_stop():
                return []
            current = order[0]
            keep.append(current)
            remaining = order[1:]
            if not remaining.size:
                break

            intersection_x_min = np.maximum(x_min[current], x_min[remaining])
            intersection_y_min = np.maximum(y_min[current], y_min[remaining])
            intersection_x_max = np.minimum(x_max[current], x_max[remaining])
            intersection_y_max = np.minimum(y_max[current], y_max[remaining])
            intersection = np.maximum(
                0.0, intersection_x_max - intersection_x_min
            ) * np.maximum(0.0, intersection_y_max - intersection_y_min)
            union = areas[current] + areas[remaining] - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            order = remaining[iou <= iou_threshold]

        return keep

    def _postprocess_small_regions(
        self,
        rles,
        should_stop=None,
    ):
        """Clean small mask regions and rerun box NMS."""
        if self.min_mask_region_area <= 0 or not rles:
            return rles

        processed_rles = []
        boxes = []
        scores = []
        for rle in rles:
            if should_stop is not None and should_stop():
                return []
            mask = self.rle_to_mask(rle)
            mask, holes_changed = self._remove_small_regions(
                mask,
                self.min_mask_region_area,
                mode="holes",
            )
            mask, islands_changed = self._remove_small_regions(
                mask,
                self.min_mask_region_area,
                mode="islands",
            )
            changed = holes_changed or islands_changed
            box = self._mask_to_box(mask)
            if box is None:
                continue
            processed_rles.append(self.mask_to_rle(mask) if changed else rle)
            boxes.append(box)
            scores.append(float(not changed))

        keep = self._box_nms(
            boxes,
            scores,
            self.box_nms_thresh,
            should_stop=should_stop,
        )
        return [processed_rles[index] for index in keep]

    @staticmethod
    def _normalize_decoder_output(logits, ious):
        """Normalize decoder outputs to (B, C, H, W) and (B, C)."""
        logits = np.asarray(logits, dtype=np.float32)
        ious = np.asarray(ious, dtype=np.float32)
        if logits.ndim == 3:
            logits = logits[:, None, :, :]
        if ious.ndim == 1:
            ious = ious[:, None]
        if logits.ndim != 4 or ious.ndim != 2:
            raise ValueError("Unexpected SAM decoder output dimensions")
        if logits.shape[:2] != ious.shape:
            raise ValueError("SAM mask and score dimensions do not match")
        return logits, ious

    def generate(
        self,
        decode_batch,
        image_hw,
        should_stop=None,
    ):
        """Generate quality-filtered full-resolution masks as RLE records."""
        height, width = image_hw
        grid = self.build_point_grid(
            self.points_per_side,
            height,
            width,
        )
        rles = []
        boxes = []
        scores = []

        for start in range(0, len(grid), self.points_per_batch):
            if should_stop is not None and should_stop():
                return []
            chunk = grid[start : start + self.points_per_batch]
            logits, ious = self._normalize_decoder_output(*decode_batch(chunk))
            candidate_logits = logits.reshape(-1, *logits.shape[-2:])
            candidate_scores = ious.reshape(-1)
            eligible = (
                np.flatnonzero(candidate_scores > self.pred_iou_thresh)
                if self.pred_iou_thresh > 0.0
                else np.arange(len(candidate_scores))
            )

            for index in eligible:
                if should_stop is not None and should_stop():
                    return []
                full_logits = cv2.resize(
                    candidate_logits[index],
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
                if self.stability_score_thresh > 0.0:
                    stability = self._stability_score(
                        full_logits[None],
                        self.mask_threshold,
                        self.stability_score_offset,
                    )[0]
                    if stability < self.stability_score_thresh:
                        continue
                mask = full_logits > self.mask_threshold
                box = self._mask_to_box(mask)
                if box is None:
                    continue
                rles.append(self.mask_to_rle(mask))
                boxes.append(box)
                scores.append(candidate_scores[index])

        if not boxes:
            return []

        keep = self._box_nms(
            boxes,
            scores,
            self.box_nms_thresh,
            should_stop=should_stop,
        )
        selected_rles = [rles[index] for index in keep]
        return self._postprocess_small_regions(
            selected_rles,
            should_stop=should_stop,
        )

    @staticmethod
    def masks_to_shapes(
        mask,
        output_mode,
        epsilon=0.001,
        min_area=0,
        label="AUTOLABEL_OBJECT",
    ):
        """Convert every external mask contour to an AnyLabeling shape."""
        from PyQt6 import QtCore

        from anylabeling.views.labeling.shape import Shape

        binary = np.asarray(mask, dtype=np.uint8)
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        shapes = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            approx = cv2.approxPolyDP(
                contour,
                epsilon * cv2.arcLength(contour, True),
                True,
            )
            points = approx.reshape(-1, 2)
            min_points = 2 if output_mode == "contour" else 3
            if len(points) < min_points:
                continue

            shape = Shape(flags={})
            for x, y in points:
                shape.add_point(QtCore.QPointF(int(x), int(y)))
            if output_mode == "contour":
                shape.shape_type = "linestrip"
                shape.closed = False
            else:
                shape.shape_type = "polygon"
                shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.label = label
            shape.selected = False
            shapes.append(shape)
        return shapes

    def generate_shapes(
        self,
        decode_batch,
        image_hw,
        output_mode,
        epsilon=0.001,
        label_prefix="object",
        should_stop=None,
    ):
        """Generate AnyLabeling shapes with incrementing object labels."""
        mask_rles = self.generate(
            decode_batch,
            image_hw,
            should_stop=should_stop,
        )
        shapes = []
        for mask_rle in mask_rles:
            if should_stop is not None and should_stop():
                return []
            mask_shapes = self.masks_to_shapes(
                self.rle_to_mask(mask_rle),
                output_mode,
                epsilon=epsilon,
                min_area=self.min_mask_region_area,
            )
            for shape in mask_shapes:
                shape.label = f"{label_prefix}{len(shapes) + 1}"
                shapes.append(shape)
        return shapes


class SegmentAnything2ONNX:
    """Segmentation model using Segment Anything 2 (SAM2)"""

    def __init__(self, encoder_model_path, decoder_model_path, device) -> None:
        self.encoder = SAM2ImageEncoder(encoder_model_path, device)
        self.decoder = SAM2ImageDecoder(
            decoder_model_path, device, self.encoder.input_shape[2:]
        )

    def encode(self, cv_image: np.ndarray) -> List[np.ndarray]:
        original_size = cv_image.shape[:2]
        high_res_feats_0, high_res_feats_1, image_embed = self.encoder(
            cv_image
        )
        return {
            "high_res_feats_0": high_res_feats_0,
            "high_res_feats_1": high_res_feats_1,
            "image_embedding": image_embed,
            "original_size": original_size,
        }

    def predict_masks(self, embedding, prompt) -> List[np.ndarray]:
        points = []
        labels = []
        for mark in prompt:
            if mark["type"] == "point":
                points.append(mark["data"])
                labels.append(mark["label"])
            elif mark["type"] == "rectangle":
                points.append([mark["data"][0], mark["data"][1]])  # top left
                points.append(
                    [mark["data"][2], mark["data"][3]]
                )  # bottom right
                labels.append(2)
                labels.append(3)
        points, labels = np.array(points), np.array(labels)

        image_embedding = embedding["image_embedding"]
        high_res_feats_0 = embedding["high_res_feats_0"]
        high_res_feats_1 = embedding["high_res_feats_1"]
        original_size = embedding["original_size"]
        self.decoder.set_image_size(original_size)
        masks, _ = self.decoder(
            image_embedding,
            high_res_feats_0,
            high_res_feats_1,
            points,
            labels,
        )

        return masks

    def predict_masks_batch(self, embedding, points_xy, points_per_batch=64):
        """Decode batches of single foreground point prompts.

        Returns all mask candidates as logits with shape (K, C, H, W) and
        predicted IoU scores with shape (K, C).
        """
        image_embedding = embedding["image_embedding"]
        high_res_feats_0 = embedding["high_res_feats_0"]
        high_res_feats_1 = embedding["high_res_feats_1"]
        original_size = embedding["original_size"]
        self.decoder.set_image_size(original_size)

        all_logits = []
        all_scores = []
        for batch in iter_point_batches(points_xy, points_per_batch):
            coords = [
                np.asarray(p, dtype=np.float32).reshape(1, 2) for p in batch
            ]
            labels = [np.array([1], dtype=np.float32) for _ in batch]
            masks, scores = self.decoder.predict_batch(
                image_embedding,
                high_res_feats_0,
                high_res_feats_1,
                coords,
                labels,
            )
            scores = np.asarray(scores)
            masks = np.asarray(masks)
            if scores.ndim == 1:  # single-mask decoders: (B,) -> (B, 1)
                scores = scores[:, None]
                masks = masks[:, None]
            all_logits.append(masks)
            all_scores.append(scores)
        return (
            np.concatenate(all_logits, axis=0),
            np.concatenate(all_scores, axis=0),
        )

    def transform_masks(self, masks, original_size, transform_matrix):
        """Transform the masks back to the original image size."""
        output_masks = []
        for batch in range(masks.shape[0]):
            batch_masks = []
            for mask_id in range(masks.shape[1]):
                mask = masks[batch, mask_id]
                mask = cv2.warpAffine(
                    mask,
                    transform_matrix[:2],
                    (original_size[1], original_size[0]),
                    flags=cv2.INTER_LINEAR,
                )
                batch_masks.append(mask)
            output_masks.append(batch_masks)
        return np.array(output_masks)


def _create_session(path: str, device: str) -> ort.InferenceSession:
    """Create an ONNX Runtime inference session for SAM2.

    The CUDA provider options keep arena growth conservative and reduce
    steady-state VRAM usage during repeated interactive inference.

    Args:
        path (str): Path to the ONNX model file.
        device (str): Device to run inference on ('gpu' or 'cpu').

    Returns:
        ort.InferenceSession: Configured inference session.
    """
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    ort.set_default_logger_severity(sess_options.log_severity_level)
    sess_options.enable_mem_pattern = True
    sess_options.enable_mem_reuse = True

    if device.lower() == "gpu":
        cuda_provider_options = {
            "device_id": 0,
            "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "DEFAULT",
        }
        providers = [
            ("CUDAExecutionProvider", cuda_provider_options),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        path, providers=providers, sess_options=sess_options
    )
    return session


class SAM2ImageEncoder:
    def __init__(self, path: str, device: str) -> None:
        # Initialize model
        self.session = _create_session(path, device)

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.encode_image(image)

    def encode_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = self.prepare_input(image)

        outputs = self.forward_encoder(input_tensor)

        return self.process_output(outputs)

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]

        input_img = image.astype(np.float32) / 255.0
        bilinear = getattr(Image, "Resampling", Image).BILINEAR
        input_tensor = np.stack(
            [
                np.asarray(
                    Image.fromarray(input_img[:, :, channel]).resize(
                        (self.input_width, self.input_height),
                        bilinear,
                    ),
                    dtype=np.float32,
                )
                for channel in range(3)
            ]
        )
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_tensor -= mean[:, None, None]
        input_tensor /= std[:, None, None]
        return input_tensor[None]

    def forward_encoder(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )

        return outputs

    def process_output(
        self, outputs: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return outputs[0], outputs[1], outputs[2]

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [
            model_inputs[i].name for i in range(len(model_inputs))
        ]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [
            model_outputs[i].name for i in range(len(model_outputs))
        ]


class SAM2ImageDecoder:
    def __init__(
        self,
        path: str,
        device: str,
        encoder_input_size: Tuple[int, int],
        orig_im_size: Tuple[int, int] = None,
        mask_threshold: float = 0.0,
    ) -> None:
        # Initialize model
        self.session = _create_session(path, device)

        self.orig_im_size = (
            orig_im_size if orig_im_size is not None else encoder_input_size
        )
        self.encoder_input_size = encoder_input_size
        self.mask_threshold = mask_threshold
        self.scale_factor = 4

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: Union[List[np.ndarray], np.ndarray],
        point_labels: Union[List[np.ndarray], np.ndarray],
    ) -> Tuple[List[np.ndarray], ndarray]:
        return self.predict(
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            point_coords,
            point_labels,
        )

    def predict(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: Union[List[np.ndarray], np.ndarray],
        point_labels: Union[List[np.ndarray], np.ndarray],
    ) -> Tuple[List[np.ndarray], ndarray]:
        inputs = self.prepare_inputs(
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            point_coords,
            point_labels,
        )

        outputs = self.forward_decoder(inputs)

        return self.process_output(outputs)

    def predict_batch(
        self,
        image_embed,
        high_res_feats_0,
        high_res_feats_1,
        point_coords,
        point_labels,
    ):
        """Run the decoder on a batch of prompts and return raw masks + scores.

        Returns (masks[B, num_masks, h, w], scores[B, num_masks]) without
        selecting the best mask or resizing to the original image size.
        """
        inputs = self.prepare_inputs(
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            point_coords,
            point_labels,
        )
        outputs = self.forward_decoder(inputs)
        return outputs[0], outputs[1]

    def prepare_inputs(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: Union[List[np.ndarray], np.ndarray],
        point_labels: Union[List[np.ndarray], np.ndarray],
    ):
        input_point_coords, input_point_labels = self.prepare_points(
            point_coords, point_labels
        )

        num_labels = input_point_labels.shape[0]
        mask_input = np.zeros(
            (
                num_labels,
                1,
                self.encoder_input_size[0] // self.scale_factor,
                self.encoder_input_size[1] // self.scale_factor,
            ),
            dtype=np.float32,
        )
        has_mask_input = np.array([0], dtype=np.float32)

        return (
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            input_point_coords,
            input_point_labels,
            mask_input,
            has_mask_input,
        )

    def prepare_points(
        self,
        point_coords: Union[List[np.ndarray], np.ndarray],
        point_labels: Union[List[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(point_coords, np.ndarray):
            input_point_coords = point_coords[np.newaxis, ...]
            input_point_labels = point_labels[np.newaxis, ...]
        else:
            max_num_points = max([coords.shape[0] for coords in point_coords])
            # We need to make sure that all inputs have the same number of points
            # Add invalid points to pad the input (0, 0) with -1 value for labels
            input_point_coords = np.zeros(
                (len(point_coords), max_num_points, 2), dtype=np.float32
            )
            input_point_labels = (
                np.ones((len(point_coords), max_num_points), dtype=np.float32)
                * -1
            )

            for i, (coords, labels) in enumerate(
                zip(point_coords, point_labels)
            ):
                input_point_coords[i, : coords.shape[0], :] = coords
                input_point_labels[i, : labels.shape[0]] = labels

        input_point_coords[..., 0] = (
            input_point_coords[..., 0]
            / self.orig_im_size[1]
            * self.encoder_input_size[1]
        )  # Normalize x
        input_point_coords[..., 1] = (
            input_point_coords[..., 1]
            / self.orig_im_size[0]
            * self.encoder_input_size[0]
        )  # Normalize y

        return input_point_coords.astype(
            np.float32
        ), input_point_labels.astype(np.float32)

    def forward_decoder(self, inputs) -> List[np.ndarray]:
        outputs = self.session.run(
            self.output_names,
            {
                self.input_names[i]: inputs[i]
                for i in range(len(self.input_names))
            },
        )
        return outputs

    def process_output(
        self, outputs: List[np.ndarray]
    ) -> Tuple[List[Union[np.ndarray, Any]], np.ndarray]:
        scores = outputs[1].squeeze()
        masks = outputs[0][0]

        # Select the best masks based on the scores
        best_mask = masks[np.argmax(scores)]
        best_mask = cv2.resize(
            best_mask, (self.orig_im_size[1], self.orig_im_size[0])
        )
        return (
            np.array([[best_mask]]),
            scores,
        )

    def set_image_size(self, orig_im_size: Tuple[int, int]) -> None:
        self.orig_im_size = orig_im_size

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [
            model_inputs[i].name for i in range(len(model_inputs))
        ]

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [
            model_outputs[i].name for i in range(len(model_outputs))
        ]
