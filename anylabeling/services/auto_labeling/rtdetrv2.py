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


class RTDETRv2(Model):
    """Object detection model using RTDETRv2"""

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
        # Perform the pre-processing steps
        image = cv2.resize(input_image, (input_w, input_h))
        image = image.transpose((2, 0, 1))  # HWC to CHW
        image = np.ascontiguousarray(image).astype("float32")
        image /= 255  # 0 - 255 to 0.0 - 1.0
        if len(image.shape) == 3:
            image = image[None]
        orig_size = np.array([image_w, image_h], np.int64)[None, :]
        blob = {"images": image, "orig_target_sizes": orig_size}
        return blob

    def postprocess(self, outputs):
        """
        Post-processes the network's output.

        Args:
            outputs (numpy.ndarray): The output from the network.

        Returns:
            scores (List[float]): prediction score
            indexs (List[int]): category index
            bboxes (List[list[int]]): xyxy format
        """
        indexs, boxes, scores = outputs
        scores = scores[0]
        indexs = indexs[0][scores > self.conf_thres]
        bboxes = boxes[0][scores > self.conf_thres]

        return scores, indexs, bboxes

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
            None, inputs=blob, extract=False
        )
        scores, indexs, bboxes = self.postprocess(detections)
        shapes = []

        for score, index, box in zip(scores, indexs, bboxes):
            xmin, ymin, xmax, ymax = box
            label = self.classes[int(index)]
            shape = Shape(
                label=str(label), score=float(score), shape_type="rectangle"
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
