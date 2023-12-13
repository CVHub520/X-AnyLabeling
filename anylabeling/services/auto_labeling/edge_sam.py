import logging
import os
import traceback

import cv2
import numpy as np
import onnxruntime as ort
from copy import deepcopy
from typing import Tuple
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QCoreApplication

from anylabeling.utils import GenericWorker
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .lru_cache import LRUCache
from .model import Model
from .types import AutoLabelingResult


class EdgeSAMONNX(object):
    def __init__(
        self, encoder_model_path, decoder_model_path, target_length
    ) -> None:
        # Load models
        providers = ort.get_available_providers()

        # Pop TensorRT Runtime due to crashing issues
        # TODO: Add back when TensorRT backend is stable
        providers = [p for p in providers if p != "TensorrtExecutionProvider"]

        self.encoder_session = ort.InferenceSession(
            encoder_model_path, providers=providers
        )
        self.decoder_session = ort.InferenceSession(
            decoder_model_path, providers=providers
        )

        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.target_length = target_length

    def run_encoder(self, encoder_inputs):
        """Run encoder"""
        image_embeddings = self.encoder_session.run(None, encoder_inputs)
        return image_embeddings[0]

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def transform(self, input_image: np.ndarray) -> np.ndarray:
        """image transform

        This function can convert the input image to the required input format for vit.

        Args:
            input_image (np.ndarray): input image, the image type should be RGB.

        Returns:
            np.ndarray: transformed image.
        """
        # Resize
        h, w, _ = input_image.shape
        target_size = self.get_preprocess_shape(h, w, self.target_length)
        input_image = cv2.resize(input_image, target_size[::-1])

        # HWC -> CHW
        input_image = input_image.transpose((2, 0, 1))

        # CHW -> NCHW
        input_image = np.ascontiguousarray(input_image)
        input_image = np.expand_dims(input_image, 0)

        return input_image

    def encode(self, cv_image):
        """
        Calculate embedding and metadata for a single image.
        """
        original_size = cv_image.shape[:2]
        encoder_inputs = {
            self.encoder_input_name: self.transform(cv_image),
        }

        image_embeddings = self.run_encoder(encoder_inputs)
        return {
            "image_embeddings": image_embeddings,
            "original_size": original_size,
        }

    def get_input_points(self, prompt):
        """Get input points"""
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
        points, labels = np.array(points).astype(np.float32), np.array(
            labels
        ).astype(np.float32)
        return points, labels

    @staticmethod
    def calculate_stability_score(
        masks: np.ndarray, mask_threshold: float, threshold_offset: float
    ) -> np.ndarray:
        """
        Computes the stability score for a batch of masks. The stability
        score is the IoU between the binary masks obtained by thresholding
        the predicted mask logits at high and low values.
        """
        high_threshold_mask = masks > (mask_threshold + threshold_offset)
        low_threshold_mask = masks > (mask_threshold - threshold_offset)

        intersections = np.sum(
            high_threshold_mask & low_threshold_mask,
            axis=(-1, -2),
            dtype=np.int32,
        )
        unions = np.sum(
            high_threshold_mask | low_threshold_mask,
            axis=(-1, -2),
            dtype=np.int32,
        )

        return intersections / unions

    def apply_coords(
        self, coords: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(
            boxes.reshape(-1, 2, 2), original_size, new_size
        )
        return boxes

    def postprocess_masks(
        self,
        mask: np.ndarray,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> np.ndarray:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
        mask (np.ndarray): mask from the mask_decoder, in 1xHxW format.
        input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
        original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
        (np.ndarray): Mask in 1xHxW format, where (H, W)
            is given by original_size.
        """
        img_size = self.target_length

        # Upscale masks to the intermediate size
        mask = cv2.resize(mask, (img_size, img_size))

        # Remove padding
        mask = mask[..., : input_size[0], : input_size[1]]

        # Upscale masks to the original size
        new_size = original_size[::-1]
        mask = cv2.resize(mask, new_size)

        return mask

    def run_decoder(self, image_embeddings, original_size, prompt):
        """Run decoder"""
        point_coords, point_labels = self.get_input_points(prompt)

        if point_coords is None or point_labels is None:
            raise ValueError(
                "Unable to segment, please input at least one box or point."
            )

        if point_coords is not None:
            if isinstance(point_coords, list):
                point_coords = np.array(point_coords, dtype=np.float32)
            if isinstance(point_labels, list):
                point_labels = np.array(point_labels, dtype=np.float32)

        if point_coords is not None:
            point_coords = self.apply_coords(
                point_coords, original_size
            ).astype(np.float32)
            point_coords = np.expand_dims(point_coords, axis=0)
            point_labels = np.expand_dims(point_labels, axis=0)

        assert point_coords.shape[0] == 1 and point_coords.shape[-1] == 2
        assert point_labels.shape[0] == 1

        input_dict = {
            "image_embeddings": image_embeddings,
            "point_coords": point_coords,
            "point_labels": point_labels,
        }
        scores, masks = self.decoder_session.run(None, input_dict)
        mask_threshold = 0.0
        stability_score_offset = 1.0
        scores = self.calculate_stability_score(
            masks[0], mask_threshold, stability_score_offset
        )
        max_score_index = np.argmax(scores)
        masks = masks[0, max_score_index]
        input_size = self.get_preprocess_shape(
            *original_size, self.target_length
        )
        masks = self.postprocess_masks(masks, input_size, original_size)
        masks = masks > 0.0
        return masks

    def predict_masks(self, embedding, prompt):
        """
        Predict masks for a single image.
        """
        masks = self.run_decoder(
            embedding["image_embeddings"],
            embedding["original_size"],
            prompt,
        )

        return masks


class EdgeSAM(Model):
    """Segmentation model using EdgeSAM"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "encoder_model_path",
            "decoder_model_path",
        ]
        widgets = [
            "output_label",
            "output_select_combobox",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "polygon"

    def __init__(self, config_path, on_message) -> None:
        # Run the parent class's init method
        super().__init__(config_path, on_message)

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
                    "Could not download or initialize encoder of EdgeSAM.",
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
                    "Could not download or initialize decoder of EdgeSAM.",
                )
            )

        # Load models
        self.target_length = self.config.get("target_length", 1024)
        self.model = EdgeSAMONNX(
            encoder_model_abs_path, decoder_model_abs_path, self.target_length
        )

        # Mark for auto labeling
        # points, rectangles
        self.marks = []

        # Cache for image embedding
        self.cache_size = 10
        self.preloaded_size = self.cache_size - 3
        self.image_embedding_cache = LRUCache(self.cache_size)

        # Pre-inference worker
        self.pre_inference_thread = None
        self.pre_inference_worker = None
        self.stop_inference = False

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def post_process(self, masks):
        """
        Post process masks
        """
        # Find contours
        masks = masks.astype(np.uint8)
        contours, _ = cv2.findContours(
            masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Refine contours
        approx_contours = []
        for contour in contours:
            # Approximate contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
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
                shape.line_width = 1
                shape.label = "AUTOLABEL_OBJECT"
                shape.selected = False
                shapes.append(shape)
        elif self.output_mode == "rectangle":
            x_min = 100000000
            y_min = 100000000
            x_max = 0
            y_max = 0
            for approx in approx_contours:
                # Scale points
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue

                # Get min/max
                for point in points:
                    x_min = min(x_min, point[0])
                    y_min = min(y_min, point[1])
                    x_max = max(x_max, point[0])
                    y_max = max(y_max, point[1])

            # Create shape
            shape = Shape(flags={})
            shape.add_point(QtCore.QPointF(x_min, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_max))
            shape.add_point(QtCore.QPointF(x_min, y_max))
            shape.shape_type = "rectangle"
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.line_width = 1
            shape.label = "AUTOLABEL_OBJECT"
            shape.selected = False
            shapes.append(shape)

        return shapes

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict shapes from image
        """

        if image is None or not self.marks:
            return AutoLabelingResult([], replace=False)

        shapes = []
        try:
            # Use cached image embedding if possible
            cached_data = self.image_embedding_cache.get(filename)
            if cached_data is not None:
                image_embedding = cached_data
            else:
                cv_image = qt_img_to_rgb_cv_img(image, filename)
                if self.stop_inference:
                    return AutoLabelingResult([], replace=False)
                image_embedding = self.model.encode(cv_image)
                self.image_embedding_cache.put(
                    filename,
                    image_embedding,
                )
            if self.stop_inference:
                return AutoLabelingResult([], replace=False)
            masks = self.model.predict_masks(image_embedding, self.marks)
            shapes = self.post_process(masks)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

        result = AutoLabelingResult(shapes, replace=False)
        return result

    def unload(self):
        self.stop_inference = True
        if self.pre_inference_thread:
            self.pre_inference_thread.quit()

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
