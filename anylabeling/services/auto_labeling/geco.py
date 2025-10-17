import os
import cv2
import traceback
import numpy as np
import onnxruntime as ort

from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QCoreApplication

from anylabeling.utils import GenericWorker
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .lru_cache import LRUCache
from .model import Model
from .types import AutoLabelingResult


class GeCoONNX:
    """zero shot count model using GeCo"""

    def __init__(
        self, encoder_model_path, decoder_model_path, input_size, box_threshold
    ) -> None:
        self.input_size = input_size
        self.box_threshold = box_threshold

        # Load models
        providers = ort.get_available_providers()

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3

        # Pop TensorRT Runtime due to crashing issues
        # TODO: Add back when TensorRT backend is stable
        providers = [p for p in providers if p != "TensorrtExecutionProvider"]

        self.encoder_session = ort.InferenceSession(
            encoder_model_path,
            providers=providers,
            sess_options=sess_options,
        )
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.decoder_session = ort.InferenceSession(
            decoder_model_path,
            providers=providers,
            sess_options=sess_options,
        )

    def get_input_points(self, prompt):
        """Get input points"""
        points = []
        for mark in prompt:
            points.append(
                [
                    mark["data"][0],
                    mark["data"][1],
                    mark["data"][2],
                    mark["data"][3],
                ]
            )  # top left
        points = np.array(points)  # (n, 4)
        return points

    def run_encoder(self, encoder_inputs):
        """Run encoder"""
        outputs = self.encoder_session.run(None, encoder_inputs)
        image_embeddings, hq_features = outputs[0], outputs[1]
        return image_embeddings, hq_features

    def run_decoder(
        self,
        image_embeddings,
        hq_features,
        scale_factor,
        prompt,
    ):
        """Run decoder"""
        prompt_bboxes = self.get_input_points(prompt)  # (n, 4)
        prompt_bboxes = prompt_bboxes * np.array(
            [scale_factor, scale_factor, scale_factor, scale_factor]
        )
        prompt_bboxes = prompt_bboxes[None].astype(np.float32)

        decoder_inputs = {
            "image_embeddings": image_embeddings,
            "hq_features": hq_features,
            "bboxes": prompt_bboxes,
        }

        outputs = self.decoder_session.run(None, decoder_inputs)

        pred_bboxes, box_v = outputs[0][0], outputs[1][0]

        keep = box_v > box_v.max() / self.box_threshold

        pred_bboxes = pred_bboxes[keep]
        box_v = box_v[keep]

        xywh = pred_bboxes.copy()  # xyxy
        xywh[:, :2] = pred_bboxes[:, :2]
        xywh[:, 2:] = pred_bboxes[:, 2:] - pred_bboxes[:, :2]

        keep = cv2.dnn.NMSBoxes(xywh.tolist(), box_v.tolist(), 0, 0.5)

        pred_bboxes = pred_bboxes[keep]
        box_v = box_v[keep]

        pred_bboxes = np.clip(pred_bboxes, 0, 1)

        pred_bboxes = (
            pred_bboxes
            / np.array(
                [scale_factor, scale_factor, scale_factor, scale_factor]
            )
            * self.input_size
        )

        return pred_bboxes

    def encode(self, cv_image):
        """
        Calculate embedding and metadata for a single image.
        """
        # Calculate a transformation matrix to convert to self.input_size
        scale_x = self.input_size / cv_image.shape[1]
        scale_y = self.input_size / cv_image.shape[0]
        scale_factor = min(scale_x, scale_y)

        new_H = int(scale_factor * cv_image.shape[0])
        new_W = int(scale_factor * cv_image.shape[1])

        cv_image = cv2.resize(
            cv_image, (new_W, new_H), interpolation=cv2.INTER_LINEAR
        )

        top = 0
        bottom = self.input_size - new_H
        left = 0
        right = self.input_size - new_W

        cv_image = cv2.copyMakeBorder(
            cv_image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )  # add border
        cv_image = np.transpose(cv_image / 255.0, axes=(2, 0, 1))[None]

        encoder_inputs = {
            self.encoder_input_name: cv_image.astype(np.float32),
        }
        image_embeddings, hq_features = self.run_encoder(encoder_inputs)
        return {
            "image_embeddings": image_embeddings,
            "hq_features": hq_features,
            "scale_factor": scale_factor,
        }

    def predict_bboxes(self, embedding, prompt):
        """
        Predict masks for a single image.
        """
        bboxes = self.run_decoder(
            embedding["image_embeddings"],
            embedding["hq_features"],
            embedding["scale_factor"],
            prompt,
        )

        return bboxes


class GeCo(Model):
    """zero shot count model using GeCo"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "encoder_model_path",
            "decoder_model_path",
            "encoder_data_path",
        ]
        widgets = [
            "button_add_rect",
            "button_clear",
            "button_finish_object",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, config_path, on_message) -> None:
        # Run the parent class's init method
        super().__init__(config_path, on_message)
        self.input_size = self.config.get("input_size", 1024)
        self.box_threshold = self.config.get("box_threshold", 4)

        # Get encoder and decoder model paths
        encoder_data_abs_path = self.get_model_abs_path(
            self.config, "encoder_data_path"
        )
        if not encoder_data_abs_path or not os.path.isfile(
            encoder_data_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize encoder data of GeCo.",
                )
            )
        encoder_model_abs_path = self.get_model_abs_path(
            self.config, "encoder_model_path"
        )
        if not encoder_model_abs_path or not os.path.isfile(
            encoder_model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize encoder of GeCo.",
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
                    "Could not download or initialize decoder of GeCo.",
                )
            )

        # Load models
        self.model = GeCoONNX(
            encoder_model_abs_path,
            decoder_model_abs_path,
            self.input_size,
            self.box_threshold,
        )

        # Mark for auto labeling
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

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict shapes from image
        """
        if image is None or not self.marks:
            return AutoLabelingResult([], replace=False)

        shapes = []
        cv_image = qt_img_to_rgb_cv_img(image, filename)
        img_h, img_w = cv_image.shape[:2]

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

            bboxes = self.model.predict_bboxes(image_embedding, self.marks)
            for det_bbox in bboxes:
                shapes.append(self.post_process(det_bbox, img_h, img_w))

        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
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

    def post_process(self, bbox, img_h, img_w, label="AUTOLABEL_OBJECT"):
        """
        Post process masks
        """
        x_min = min(max(bbox[0], 0), img_w - 1)
        y_min = min(max(bbox[1], 0), img_h - 1)
        x_max = min(max(bbox[2], 0), img_w - 1)
        y_max = min(max(bbox[3], 0), img_h - 1)

        # Create shape
        shape = Shape(flags={})
        shape.add_point(QtCore.QPointF(x_min, y_min))
        shape.add_point(QtCore.QPointF(x_max, y_min))
        shape.add_point(QtCore.QPointF(x_max, y_max))
        shape.add_point(QtCore.QPointF(x_min, y_max))
        shape.shape_type = "rectangle"
        shape.closed = True
        shape.label = label
        shape.fill_color = "#000000"
        shape.line_color = "#000000"
        shape.selected = False
        shape.group_id = None

        return shape
