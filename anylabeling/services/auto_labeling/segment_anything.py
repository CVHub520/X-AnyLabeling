import logging
import os
from copy import deepcopy

import cv2
import numpy as np
import onnxruntime
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QCoreApplication

from anylabeling.utils import GenericWorker
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .lru_cache import LRUCache
from .model import Model
from .types import AutoLabelingResult


class SegmentAnything(Model):
    """Segmentation model using SegmentAnything"""

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
        providers = onnxruntime.get_available_providers()

        # Pop TensorRT Runtime due to crashing issues
        # TODO: Add back when TensorRT backend is stable
        providers = [p for p in providers if p != "TensorrtExecutionProvider"]

        if providers:
            logging.info(
                "Available providers for ONNXRuntime: %s", ", ".join(providers)
            )
        else:
            logging.warning("No available providers for ONNXRuntime")
        self.encoder_session = onnxruntime.InferenceSession(
            encoder_model_abs_path, providers=providers
        )
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_model_abs_path, providers=providers
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

    def get_input_points(self, resized_ratio):
        """Get input points"""
        points = []
        labels = []
        for mark in self.marks:
            if mark["type"] == "point":
                points.append(mark["data"])
                labels.append(mark["label"])
            elif mark["type"] == "rectangle":
                points.append([mark["data"][0], mark["data"][1]])  # top left
                points.append([mark["data"][2], mark["data"][3]])  # top right
                labels.append(2)
                labels.append(3)
        points, labels = np.array(points), np.array(labels)

        # Resize points based on scales
        points[:, 0] = points[:, 0] * resized_ratio[0]
        points[:, 1] = points[:, 1] * resized_ratio[1]
        return points, labels

    def pre_process(self, image):
        # Resize by max width and max height
        # In the original code, the image is resized to long side 1024
        # However, there is a positional deviation when the image does not
        # have the same aspect ratio as in the exported ONNX model (2250x1500)
        # => Resize by max width and max height
        max_width = self.max_width
        max_height = self.max_height
        original_size = image.shape[:2]
        h, w = image.shape[:2]
        if w > max_width:
            h = int(h * max_width / w)
            w = max_width
        if h > max_height:
            w = int(w * max_height / h)
            h = max_height
        image = cv2.resize(image, (w, h))
        resized_ratio = (
            w / original_size[1],
            h / original_size[0],
        )

        # Pad to have size at least max_width x max_height
        h, w = image.shape[:2]
        padh = max_height - h
        padw = max_width - w
        image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode="constant")
        size_after_apply_max_width_height = image.shape[:2]

        # Normalize
        pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, -1)
        pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, -1)
        x = (image - pixel_mean) / pixel_std

        # Padding to square
        h, w = x.shape[:2]
        padh = self.input_size - h
        padw = self.input_size - w
        x = np.pad(x, ((0, padh), (0, padw), (0, 0)), mode="constant")
        x = x.astype(np.float32)

        # Transpose
        x = x.transpose(2, 0, 1)[None, :, :, :]

        encoder_inputs = {
            "x": x,
        }
        return encoder_inputs, resized_ratio, size_after_apply_max_width_height

    def run_encoder(self, encoder_inputs):
        output = self.encoder_session.run(None, encoder_inputs)
        image_embedding = output[0]
        return image_embedding

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

    def apply_coords(
        self, coords: np.ndarray, original_size, target_length
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = SegmentAnything.get_preprocess_shape(
            original_size[0], original_size[1], target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def run_decoder(
        self, image_embedding, resized_ratio, size_after_apply_max_width_height
    ):
        input_points, input_labels = self.get_input_points(resized_ratio)

        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate(
            [input_points, np.array([[0.0, 0.0]])], axis=0
        )[None, :, :]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[
            None, :
        ].astype(np.float32)
        onnx_coord = self.apply_coords(
            onnx_coord, size_after_apply_max_width_height, self.input_size
        ).astype(np.float32)

        # Create an empty mask input and an indicator for no mask.
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(
                size_after_apply_max_width_height, dtype=np.float32
            ),
        }
        masks, _, _ = self.decoder_session.run(None, decoder_inputs)
        masks = masks[0, 0, :, :]  # Only get 1 mask
        masks = masks > 0.0
        masks = masks.reshape(size_after_apply_max_width_height)
        return masks

    def post_process(self, masks, resized_ratio):
        """
        Post process masks
        """
        # Find contours
        contours, _ = cv2.findContours(
            masks.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Refine contours
        approx_contours = []
        for contour in contours:
            # Approximate contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_contours.append(approx)

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
                points[:, 0] = points[:, 0] / resized_ratio[0]
                points[:, 1] = points[:, 1] / resized_ratio[1]
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
                points[:, 0] = points[:, 0] / resized_ratio[0]
                points[:, 1] = points[:, 1] / resized_ratio[1]
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
            shape.add_point(QtCore.QPointF(x_max, y_max))
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
                (
                    resized_ratio,
                    size_after_apply_max_width_height,
                    image_embedding,
                ) = cached_data
            else:
                cv_image = qt_img_to_rgb_cv_img(image, filename)
                (
                    encoder_inputs,
                    resized_ratio,
                    size_after_apply_max_width_height,
                ) = self.pre_process(cv_image)
                if self.stop_inference:
                    return AutoLabelingResult([], replace=False)
                image_embedding = self.run_encoder(encoder_inputs)
                self.image_embedding_cache.put(
                    filename,
                    (
                        resized_ratio,
                        size_after_apply_max_width_height,
                        image_embedding,
                    ),
                )
            if self.stop_inference:
                return AutoLabelingResult([], replace=False)
            masks = self.run_decoder(
                image_embedding,
                resized_ratio,
                size_after_apply_max_width_height,
            )
            shapes = self.post_process(masks, resized_ratio)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return AutoLabelingResult([], replace=False)

        result = AutoLabelingResult(shapes, replace=False)
        return result

    def unload(self):
        self.stop_inference = True
        if self.pre_inference_thread:
            self.pre_inference_thread.quit()
            self.pre_inference_thread.wait()
        if self.encoder_session:
            self.encoder_session = None
        if self.decoder_session:
            self.decoder_session = None

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
            (
                encoder_inputs,
                resized_ratio,
                size_after_apply_max_width_height,
            ) = self.pre_process(cv_image)
            image_embedding = self.run_encoder(encoder_inputs)
            self.image_embedding_cache.put(
                filename,
                (
                    resized_ratio,
                    size_after_apply_max_width_height,
                    image_embedding,
                ),
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
