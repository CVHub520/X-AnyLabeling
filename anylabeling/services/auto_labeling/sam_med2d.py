import logging
import os
import traceback

import cv2
import numpy as np
import onnxruntime as ort
from copy import deepcopy
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.utils import GenericWorker
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .lru_cache import LRUCache
from .model import Model
from .types import AutoLabelingResult
from .__base__.clip import ChineseClipONNX


class SegmentAnythingONNX:
    def __init__(self, encoder_model_path, decoder_model_path) -> None:
        # Basic hyp-parameters
        self.pixel_mean = np.array([123.675, 116.28, 103.53])
        self.pixel_std = np.array([58.395, 57.12, 57.375])

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
        self.encoder_input_shape = self.encoder_session.get_inputs()[0].shape
        self.encoder_input_size = self.encoder_input_shape[-2:]

    def run_encoder(self, encoder_inputs):
        """Run encoder"""
        output = self.encoder_session.run(None, encoder_inputs)
        image_embedding = output[0]
        return image_embedding

    def transform(self, input_image: np.ndarray) -> np.ndarray:
        """image transform

        This function can convert the input image to the required input format for vit.

        Args:
            input_image (np.ndarray): input image, the image type should be RGB.

        Returns:
            np.ndarray: transformed image.
        """
        # Normalization
        input_image = (input_image - self.pixel_mean) / self.pixel_std

        # Resize
        input_image = cv2.resize(
            input_image, self.encoder_input_size, cv2.INTER_NEAREST
        )

        # HWC -> CHW
        input_image = input_image.transpose((2, 0, 1))

        # CHW -> NCHW
        input_image = np.expand_dims(input_image, 0).astype(np.float32)

        return input_image

    def encode(self, cv_image):
        """
        Calculate embedding and metadata for a single image.
        """
        original_size = cv_image.shape[:2]
        encoder_inputs = {
            self.encoder_input_name: self.transform(cv_image),
        }

        image_embedding = self.run_encoder(encoder_inputs)
        return {
            "image_embedding": image_embedding,
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

    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def run_decoder(self, image_embedding, original_size, prompt):
        """Run decoder"""
        point_coords, point_labels = self.get_input_points(prompt)

        if point_coords is None or point_labels is None:
            raise ValueError(
                "Unable to segment, please input at least one box or point."
            )

        h, w = self.encoder_input_size
        if image_embedding.shape != (1, 256, int(h / 16), int(w / 16)):
            raise ValueError("Got wrong embedding shape!")

        has_mask_input = np.zeros(1, dtype=np.float32)
        mask_input = np.zeros((1, 1, int(h / 4), int(w / 4)), dtype=np.float32)

        if point_coords is not None:
            if isinstance(point_coords, list):
                point_coords = np.array(point_coords, dtype=np.float32)
            if isinstance(point_labels, list):
                point_labels = np.array(point_labels, dtype=np.float32)

        if point_coords is not None:
            point_coords = self.apply_coords(
                point_coords, original_size, self.encoder_input_size
            ).astype(np.float32)
            point_coords = np.expand_dims(point_coords, axis=0)
            point_labels = np.expand_dims(point_labels, axis=0)

        assert point_coords.shape[0] == 1 and point_coords.shape[-1] == 2
        assert point_labels.shape[0] == 1
        input_dict = {
            "image_embeddings": image_embedding,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": np.array(original_size, dtype=np.float32),
        }
        masks, _, _ = self.decoder_session.run(None, input_dict)

        return masks

    def predict_masks(self, embedding, prompt):
        """
        Predict masks for a single image.
        """
        masks = self.run_decoder(
            embedding["image_embedding"],
            embedding["original_size"],
            prompt,
        )

        return masks


class SAM_Med2D(Model):
    """Segmentation model using SAM_Med2D"""

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
            "rotation": QCoreApplication.translate("Model", "Rotation"),
        }
        default_output_mode = "polygon"

    def __init__(self, config_path, on_message) -> None:
        # Run the parent class's init method
        super().__init__(config_path, on_message)
        self.input_size = self.config["input_size"]

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

        # Pre-inference worker
        self.pre_inference_thread = None
        self.pre_inference_worker = None
        self.stop_inference = False

        # CLIP models
        self.clip_net = None
        clip_txt_model_path = self.config.get("txt_model_path", "")
        clip_img_model_path = self.config.get("img_model_path", "")
        if clip_txt_model_path and clip_img_model_path:
            if self.config["model_type"] == "cn_clip":
                model_arch = self.config["model_arch"]
                self.clip_net = ChineseClipONNX(
                    clip_txt_model_path,
                    clip_img_model_path,
                    model_arch,
                    device=__preferred_device__,
                )
            self.classes = self.config.get("classes", [])

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def post_process(self, masks, image=None):
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
            if self.clip_net is not None and self.classes:
                img = image[y_min:y_max, x_min:x_max]
                out = self.clip_net(img, self.classes)
                shape.cache_label = self.classes[int(np.argmax(out))]
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
        cv_image = qt_img_to_rgb_cv_img(image, filename)
        try:
            # Use cached image embedding if possible
            cached_data = self.image_embedding_cache.get(filename)
            if cached_data is not None:
                image_embedding = cached_data
            else:
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
            if len(masks.shape) == 4:
                masks = masks[0][0]
            else:
                masks = masks[0]
            shapes = self.post_process(masks, cv_image)
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
