import os
import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel
from .utils.points_conversion import cxywh2xyxy


class RTDETR(Model):
    """Object detection model using RTDETR"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "conf_threshold",
            "classes",
        ]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "toggle_preserve_existing_annotations",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_name = self.config["type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_name} model.",
                )
            )
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]
        self.input_shape = self.net.get_input_shape()[-2:]
        self.conf_thres = self.config["conf_threshold"]
        self.replace = True

    def set_auto_labeling_conf(self, value):
        """set auto labeling confidence threshold"""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def preprocess(self, input_image):
        """
        Pre-processes the input image before feeding it to the network.

        Args:
            input_image (numpy.ndarray): The input image to be processed.

        Returns:
            numpy.ndarray: The pre-processed output.
        """
        # Get the image width and height
        image_h, image_w = input_image.shape[:2]
        input_h, input_w = self.input_shape

        # Compute the scaling factors
        ratio_h = input_h / image_h
        ratio_w = input_w / image_w

        # Perform the pre-processing steps
        image = cv2.resize(
            input_image, (0, 0), fx=ratio_w, fy=ratio_h, interpolation=2
        )
        image = image.transpose((2, 0, 1))  # HWC to CHW
        image = np.ascontiguousarray(image).astype("float32")
        image /= 255  # 0 - 255 to 0.0 - 1.0
        if len(image.shape) == 3:
            image = image[None]
        return image

    def postprocess(self, input_image, outputs):
        """
        Post-processes the network's output.

        Args:
            input_image (numpy.ndarray): The input image.
            outputs (numpy.ndarray): The output from the network.

        Returns:
            list: List of dictionaries containing the output
                    bounding boxes, labels, and scores.
        """
        image_height, image_width = input_image.shape[:2]

        boxes, scores = outputs[:, :4], outputs[:, 4:]

        # Normalize scores if they are not already in the range (0, 1)
        if not (np.all((scores > 0) & (scores < 1))):
            scores = 1 / (1 + np.exp(-scores))

        boxes = cxywh2xyxy(boxes)
        _max = scores.max(-1)
        _mask = _max > self.conf_thres
        boxes, scores = boxes[_mask], scores[_mask]
        indexs, scores = scores.argmax(-1), scores.max(-1)

        # Normalize the bounding box coordinates
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = np.floor(
            np.minimum(np.maximum(1, x1 * image_width), image_width - 1)
        ).astype(int)
        y1 = np.floor(
            np.minimum(np.maximum(1, y1 * image_height), image_height - 1)
        ).astype(int)
        x2 = np.ceil(
            np.minimum(np.maximum(1, x2 * image_width), image_width - 1)
        ).astype(int)
        y2 = np.ceil(
            np.minimum(np.maximum(1, y2 * image_height), image_height - 1)
        ).astype(int)
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        results = []
        for box, index, score in zip(boxes, indexs, scores):
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            label = str(self.classes[index])

            result = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": label,
                "score": float(score),
            }

            results.append(result)

        return results

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        blob = self.preprocess(image)
        detections = self.net.get_ort_inference(
            blob, extract=True, squeeze=True
        )
        results = self.postprocess(image, detections)
        shapes = []

        for result in results:
            xmin = result["x1"]
            ymin = result["y1"]
            xmax = result["x2"]
            ymax = result["y2"]
            shape = Shape(
                label=result["label"],
                score=result["score"],
                shape_type="rectangle",
            )
            shape.add_point(QtCore.QPointF(xmin, ymin))
            shape.add_point(QtCore.QPointF(xmax, ymin))
            shape.add_point(QtCore.QPointF(xmax, ymax))
            shape.add_point(QtCore.QPointF(xmin, ymax))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=self.replace)
        return result

    def unload(self):
        del self.net
