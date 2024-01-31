import logging
import os
import traceback

import cv2
import numpy as np
import onnxruntime as ort
from copy import deepcopy
from typing import Any, Union, Tuple
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QCoreApplication

from anylabeling.utils import GenericWorker
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .lru_cache import LRUCache
from .model import Model
from .types import AutoLabelingResult


class SamEncoder:
    """Sam encoder model.

    In this class, encoder model will encoder the input image.

    Args:
        model_path (str): sam encoder onnx model path.
    """

    def __init__(self, model_path: str):
        # Load models
        providers = ort.get_available_providers()

        # Pop TensorRT Runtime due to crashing issues
        # TODO: Add back when TensorRT backend is stable
        providers = [p for p in providers if p != "TensorrtExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape
        self.input_size = (self.input_shape[-1], self.input_shape[-2])

    def _extract_feature(self, cv_image: np.ndarray) -> np.ndarray:
        """extract image feature

        this function can use vit to extract feature from transformed image.

        Args:
            cv_image (np.ndarray): input image with RGB format.

        Returns:
            image_embedding: image`s feature.
            origin_image_size: image`s size.
        """
        image_embedding = self.session.run(None, {self.input_name: cv_image})[
            0
        ]
        return {
            "image_embedding": image_embedding,
            "origin_image_size": cv_image.shape[:2],
        }

    def __call__(self, img: np.array) -> Any:
        return self._extract_feature(img)


class SamDecoder:
    """Sam decoder model.

    This class is the sam prompt encoder and lightweight mask decoder.

    Args:
        model_path (str): decoder model path.
    """

    def __init__(self, model_path: str, target_size: int):
        # Load models
        providers = ort.get_available_providers()

        # Pop TensorRT Runtime due to crashing issues
        # TODO: Add back when TensorRT backend is stable
        providers = [p for p in providers if p != "TensorrtExecutionProvider"]

        self.target_size = target_size
        self.session = ort.InferenceSession(model_path, providers=providers)

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

    def run(
        self,
        img_embeddings: np.ndarray,
        origin_image_size: Union[list, tuple],
        point_coords: Union[list, np.ndarray] = None,
        point_labels: Union[list, np.ndarray] = None,
        boxes: Union[list, np.ndarray] = None,
        mask_input: np.ndarray = None,
    ):
        """decoder forward function

        This function can use image feature and prompt to generate mask. Must input
        at least one box or point.

        Args:
            img_embeddings (np.ndarray): the image feature from vit encoder.
            origin_image_size (list or tuple): the input image size.
            point_coords (list or np.ndarray): the input points.
            point_labels (list or np.ndarray): the input points label, 1 indicates
                a foreground point and 0 indicates a background point.
            boxes (list or np.ndarray): A length 4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray): A low resolution mask input to the model,
                typically coming from a previous prediction iteration. Has form
                1xHxW, where for SAM, H=W=4 * embedding.size.

        Returns:
            the segment results.
        """
        input_size = self.get_preprocess_shape(
            *origin_image_size, long_side_length=self.target_size
        )

        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError(
                "Unable to segment, please input at least one box or point."
            )

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")
        if mask_input is None:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.zeros(1, dtype=np.float32)
        else:
            mask_input = np.expand_dims(mask_input, axis=0)
            has_mask_input = np.ones(1, dtype=np.float32)
            if mask_input.shape != (1, 1, 256, 256):
                raise ValueError("Got wrong mask!")
        if point_coords is not None:
            if isinstance(point_coords, list):
                point_coords = np.array(point_coords, dtype=np.float32)
            if isinstance(point_labels, list):
                point_labels = np.array(point_labels, dtype=np.float32)

        if point_coords is not None:
            point_coords = self.apply_coords(
                point_coords, origin_image_size, input_size
            ).astype(np.float32)
            point_coords = np.expand_dims(point_coords, axis=0)
            point_labels = np.expand_dims(point_labels, axis=0)

        assert point_coords.shape[0] == 1 and point_coords.shape[-1] == 2
        assert point_labels.shape[0] == 1
        input_dict = {
            "image_embeddings": img_embeddings,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": np.array(origin_image_size, dtype=np.float32),
        }
        masks, _, _ = self.session.run(None, input_dict)

        return masks[0]

    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(
            boxes.reshape(-1, 2, 2), original_size, new_size
        )
        return boxes.reshape(-1, 4)


class EfficientViT_SAM(Model):
    """Segmentation model using EfficientViT_SAM"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "target_size",
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
            "rotation": QCoreApplication.translate("Model", "Rotation"),
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
                    "Could not download or initialize encoder of Segment Anything.",
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
                    "Could not download or initialize decoder of Segment Anything.",
                )
            )

        # Load models
        self.target_size = self.config["target_size"]
        self.encoder_model = SamEncoder(encoder_model_abs_path)
        self.decoder_model = SamDecoder(
            decoder_model_abs_path, self.target_size
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
        masks[masks > 0.0] = 255
        masks[masks <= 0.0] = 0
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
        elif self.output_mode in ["rectangle", "rotation"]:
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
            shape.shape_type = (
                "rectangle" if self.output_mode == "rectangle" else "rotation"
            )
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.line_width = 1
            shape.label = "AUTOLABEL_OBJECT"
            shape.selected = False
            shapes.append(shape)

        return shapes

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
                origin_image_size = cv_image.shape[:2]
                if self.stop_inference:
                    return AutoLabelingResult([], replace=False)
                image_embedding = self.encoder_model(cv_image)
                self.image_embedding_cache.put(
                    filename,
                    image_embedding,
                )
            if self.stop_inference:
                return AutoLabelingResult([], replace=False)

            point_coords, point_labels = self.get_input_points()
            masks = self.decoder_model.run(
                img_embeddings=image_embedding["image_embedding"],
                origin_image_size=image_embedding["origin_image_size"],
                point_coords=point_coords,
                point_labels=point_labels,
            )
            if len(masks.shape) == 4:
                masks = masks[0][0]
            else:
                masks = masks[0]
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
            image_embedding = self.encoder_model(cv_image)
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
