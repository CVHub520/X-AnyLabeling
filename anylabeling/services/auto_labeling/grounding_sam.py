import os
import traceback
import onnxruntime
import numpy as np

from copy import deepcopy

import cv2
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QCoreApplication

from anylabeling.utils import GenericWorker
from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import (
    get_bounding_boxes,
    qt_img_to_rgb_cv_img,
)
from anylabeling.services.auto_labeling.utils import calculate_rotation_theta

from .model import Model
from .types import AutoLabelingResult
from .lru_cache import LRUCache
from .engines.build_onnx_engine import OnnxBaseModel
from .__base__.grounding_dino import GroundingDINOBase


class SegmentAnythingONNX:
    """Segmentation model using SAM-HQ"""

    def __init__(self, encoder_model_path, decoder_model_path) -> None:
        self.target_size = 1024
        self.input_size = (684, 1024)

        # Load models
        providers = onnxruntime.get_available_providers()

        # Pop TensorRT Runtime due to crashing issues
        # TODO: Add back when TensorRT backend is stable
        providers = [p for p in providers if p != "TensorrtExecutionProvider"]

        self.encoder_session = onnxruntime.InferenceSession(
            encoder_model_path, providers=providers
        )
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_model_path, providers=providers
        )

    def run_encoder(self, encoder_inputs):
        """Run encoder"""
        features = self.encoder_session.run(None, encoder_inputs)
        image_embeddings, interm_embeddings = features[0], np.stack(
            features[1:]
        )
        return image_embeddings, interm_embeddings

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def apply_coords(self, coords: np.ndarray, original_size, target_length):
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def run_decoder(
        self,
        image_embeddings,
        interm_embeddings,
        original_size,
        transform_matrix,
        input_points,
        input_labels,
    ):
        """Run decoder"""

        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate(
            [input_points, np.array([[0.0, 0.0]])], axis=0
        )[None, :, :]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[
            None, :
        ].astype(np.float32)
        onnx_coord = self.apply_coords(
            onnx_coord, self.input_size, self.target_size
        ).astype(np.float32)

        # Apply the transformation matrix to the coordinates.
        onnx_coord = np.concatenate(
            [
                onnx_coord,
                np.ones((1, onnx_coord.shape[1], 1), dtype=np.float32),
            ],
            axis=2,
        )
        onnx_coord = np.matmul(onnx_coord, transform_matrix.T)
        onnx_coord = onnx_coord[:, :, :2].astype(np.float32)

        # Create an empty mask input and an indicator for no mask.
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": image_embeddings,
            "interm_embeddings": interm_embeddings,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(self.input_size, dtype=np.float32),
        }
        masks, _, _ = self.decoder_session.run(None, decoder_inputs)

        # Transform the masks back to the original image size.
        inv_transform_matrix = np.linalg.inv(transform_matrix)
        transformed_masks = self.transform_masks(
            masks, original_size, inv_transform_matrix
        )

        return transformed_masks

    def transform_masks(self, masks, original_size, transform_matrix):
        """Transform masks
        Transform the masks back to the original image size.
        """
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

    def encode(self, cv_image):
        """
        Calculate embedding and metadata for a single image.
        """
        original_size = cv_image.shape[:2]

        # Calculate a transformation matrix to convert to self.input_size
        scale_x = self.input_size[1] / cv_image.shape[1]
        scale_y = self.input_size[0] / cv_image.shape[0]
        scale = min(scale_x, scale_y)
        transform_matrix = np.array(
            [
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1],
            ]
        )
        cv_image = cv2.warpAffine(
            cv_image,
            transform_matrix[:2],
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_LINEAR,
        )

        encoder_inputs = {
            self.encoder_input_name: cv_image.astype(np.float32),
        }
        image_embeddings, interm_embeddings = self.run_encoder(encoder_inputs)
        return {
            "image_embeddings": image_embeddings,
            "interm_embeddings": interm_embeddings,
            "original_size": original_size,
            "transform_matrix": transform_matrix,
        }

    def predict_masks(self, embedding, input_points, input_labels):
        """
        Predict masks for a single image.
        """
        masks = self.run_decoder(
            embedding["image_embeddings"],
            embedding["interm_embeddings"],
            embedding["original_size"],
            embedding["transform_matrix"],
            input_points,
            input_labels,
        )

        return masks


class GroundingSAM(Model):
    """Open-Set instance segmentation model using GroundingSAM"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_type",
            "model_path",
            "input_width",
            "input_height",
            "box_threshold",
            "text_threshold",
            "encoder_model_path",
            "decoder_model_path",
        ]
        widgets = [
            "edit_text",
            "button_send",
            "output_label",
            "output_select_combobox",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
            "button_auto_decode",
            "mask_fineness_slider",
            "mask_fineness_value_label",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            "rotation": QCoreApplication.translate("Model", "Rotation"),
        }
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        # ----------- Grounding-DINO ---------- #
        model_type = self.config["model_type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_type} model.",
                )
            )

        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.model_configs = GroundingDINOBase.get_configs(
            self.config["model_type"]
        )
        self.net.max_text_len = self.model_configs.max_text_len
        self.net.tokenizer = GroundingDINOBase.get_tokenlizer(
            self.model_configs.text_encoder_type
        )
        self.box_threshold = self.config["box_threshold"]
        self.text_threshold = self.config["text_threshold"]
        self.target_size = (
            self.config["input_width"],
            self.config["input_height"],
        )

        # ----------- HQ-SAM ---------- #
        self.input_size = self.config["input_size"]
        self.max_width = self.config["max_width"]
        self.max_height = self.config["max_height"]

        # Get encoder and decoder model paths
        encoder_model_abs_path = self.get_model_abs_path(
            self.config, "encoder_model_path"
        )
        if not encoder_model_abs_path or not os.path.isfile(
            encoder_model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize encoder of SAM_HQ.",
                )
            )
        decoder_model_abs_path = self.get_model_abs_path(
            self.config, "decoder_model_path"
        )
        if not decoder_model_abs_path or not os.path.isfile(
            decoder_model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize decoder of SAM_HQ.",
                )
            )

        # Load models
        self.model = SegmentAnythingONNX(
            encoder_model_abs_path, decoder_model_abs_path
        )

        # Mark for auto labeling
        # points, rectangles
        self.marks = []

        # Cache for image embedding
        self.cache_size = 10
        self.preloaded_size = self.cache_size - 3
        self.image_embedding_cache = LRUCache(self.cache_size)
        self.current_image_embedding_cache = {}

        # Pre-inference worker
        self.pre_inference_thread = None
        self.pre_inference_worker = None
        self.stop_inference = False

        self.epsilon = 0.001

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def set_mask_fineness(self, epsilon):
        """Set mask fineness epsilon value"""
        self.epsilon = epsilon

    def preprocess(self, image, text_prompt, img_mask=None):
        blob = GroundingDINOBase.preprocess_image(image, self.target_size)
        tokenized, text_self_attention_masks, position_ids, caption = (
            GroundingDINOBase.encode_text(
                text_prompt, self.net.tokenizer, self.net.max_text_len
            )
        )
        inputs = {
            "img": blob,
            "input_ids": np.array(tokenized["input_ids"], dtype=np.int64),
            "attention_mask": np.array(
                tokenized["attention_mask"], dtype=bool
            ),
            "token_type_ids": np.array(
                tokenized["token_type_ids"], dtype=np.int64
            ),
            "position_ids": np.array(position_ids, dtype=np.int64),
            "text_token_mask": np.array(text_self_attention_masks, dtype=bool),
        }
        if img_mask is None:
            inputs["img_mask"] = np.zeros(
                (1, blob.shape[0], blob.shape[2], blob.shape[3]),
                dtype=np.float32,
            )
        else:
            inputs["img_mask"] = img_mask
        return blob, inputs, caption

    def postprocess(
        self, outputs, caption, with_logits=True, token_spans=None
    ):
        if token_spans is not None:
            # TODO: Using token_spans.
            raise NotImplementedError
        logits, boxes = outputs
        boxes_filt, pred_phrases = GroundingDINOBase.decode_predictions(
            logits,
            boxes,
            caption,
            self.net.tokenizer,
            self.box_threshold,
            self.text_threshold,
            apply_sigmoid=False,
            with_logits=with_logits,
        )
        return boxes_filt, pred_phrases

    def post_process(self, masks, label=None):
        """
        Post process masks
        """
        # Find contours
        masks[masks > 0.0] = 255
        masks[masks <= 0.0] = 0
        masks = masks.astype(np.uint8)
        contours, _ = cv2.findContours(
            masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Refine contours
        approx_contours = []
        for contour in contours:
            # Approximate contour using configurable epsilon
            epsilon = self.epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_contours.append(approx)

        # Remove too big contours ( >90% of image size)
        if len(approx_contours) > 1:
            image_size = masks.shape[0] * masks.shape[1]
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area < image_size * 0.9
            ]
            approx_contours = filtered_approx_contours

        # Remove small contours (area < 20% of average area)
        if len(approx_contours) > 1:
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            avg_area = np.mean(areas)

            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area > avg_area * 0.2
            ]
            approx_contours = filtered_approx_contours

        # Contours to shapes
        shapes = []
        if self.output_mode == "polygon":
            for approx in approx_contours:
                # Scale points
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue
                points.append(points[0])

                # Create shape
                shape = Shape(flags={})
                for point in points:
                    point[0] = int(point[0])
                    point[1] = int(point[1])
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shape.shape_type = "polygon"
                shape.closed = True
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.label = "AUTOLABEL_OBJECT" if label is None else label
                shape.selected = False
                shapes.append(shape)
        elif self.output_mode in ["rectangle", "rotation"]:
            shape = Shape(flags={})
            rectangle_box, rotation_box = get_bounding_boxes(
                approx_contours[0]
            )
            xmin, ymin, xmax, ymax = rectangle_box
            if self.output_mode == "rectangle":
                shape.add_point(QtCore.QPointF(int(xmin), int(ymin)))
                shape.add_point(QtCore.QPointF(int(xmax), int(ymin)))
                shape.add_point(QtCore.QPointF(int(xmax), int(ymax)))
                shape.add_point(QtCore.QPointF(int(xmin), int(ymax)))
            else:
                for point in rotation_box:
                    shape.add_point(
                        QtCore.QPointF(int(point[0]), int(point[1]))
                    )
                shape.direction = calculate_rotation_theta(rotation_box)
            shape.shape_type = self.output_mode
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.label = "AUTOLABEL_OBJECT" if label is None else label
            shape.selected = False
            shapes.append(shape)

        return shapes if label is None else shapes[0]

    def predict_shapes(self, image, image_path=None, text_prompt=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            cv_image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        try:
            cached_data = self.image_embedding_cache.get(image_path)
            if cached_data is not None:
                image_embedding = cached_data
            else:
                if self.stop_inference:
                    return AutoLabelingResult([], replace=False)
                image_embedding = self.model.encode(cv_image)
                self.image_embedding_cache.put(
                    image_path,
                    image_embedding,
                )
                if self.stop_inference:
                    return AutoLabelingResult([], replace=False)

            if text_prompt:
                blob, inputs, caption = self.preprocess(
                    cv_image, text_prompt, None
                )
                outputs = self.net.get_ort_inference(
                    blob, inputs=inputs, extract=False
                )
                boxes_filt, pred_phrases = self.postprocess(outputs, caption)
                img_h, img_w, _ = cv_image.shape
                boxes = GroundingDINOBase.rescale_boxes(
                    boxes_filt, img_h, img_w
                )
                shapes = []
                for box, label_info in zip(boxes, pred_phrases):
                    x1, y1, x2, y2 = box
                    label, _ = label_info
                    point_coords = np.array(
                        [[x1, y1], [x2, y2]], dtype=np.float32
                    )
                    point_labels = np.array([2, 3], dtype=np.float32)
                    masks = self.model.predict_masks(
                        image_embedding, point_coords, point_labels
                    )
                    if len(masks.shape) == 4:
                        masks = masks[0][0]
                    else:
                        masks = masks[0]
                    results = self.post_process(masks, label=label)
                    shapes.append(results)
                result = AutoLabelingResult(shapes, replace=False)
            else:
                point_coords, point_labels = self.get_input_points()
                masks = self.model.predict_masks(
                    image_embedding, point_coords, point_labels
                )
                if len(masks.shape) == 4:
                    masks = masks[0][0]
                else:
                    masks = masks[0]
                shapes = self.post_process(masks)
                result = AutoLabelingResult(shapes, replace=False)
            return result
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

    def get_input_points(self):
        """Get input points"""
        points = []
        labels = []
        for mark in self.marks:
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
        points, labels = np.array(points).astype(np.float32), np.array(
            labels
        ).astype(np.float32)
        return points, labels

    def unload(self):
        del self.net
        self.stop_inference = True
        if self.pre_inference_thread:
            self.pre_inference_thread.quit

    def preload_worker(self, files):
        """
        Preload next files, run inference and cache results
        """
        files = files[: self.preloaded_size]
        for filename in files:
            if self.image_embedding_cache.find(filename):
                continue
            image = self.load_image_from_filename(filename)
            if image is None:
                continue
            if self.stop_inference:
                return
            cv_image = qt_img_to_rgb_cv_img(image)
            image_embedding = self.model.encode(cv_image)
            self.image_embedding_cache.put(
                filename,
                image_embedding,
            )

    def on_next_files_changed(self, next_files):
        """
        Handle next files changed. This function can preload next files
        and run inference to save time for user.
        """
        if (
            self.pre_inference_thread is None
            or not self.pre_inference_thread.isRunning()
        ):
            self.pre_inference_thread = QThread()
            self.pre_inference_worker = GenericWorker(
                self.preload_worker, next_files
            )
            self.pre_inference_worker.finished.connect(
                self.pre_inference_thread.quit
            )
            self.pre_inference_worker.moveToThread(self.pre_inference_thread)
            self.pre_inference_thread.started.connect(
                self.pre_inference_worker.run
            )
            self.pre_inference_thread.start()
