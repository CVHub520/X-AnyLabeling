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


class PULC_Attribute(Model):
    """Multi-label Attribute Classification using PULC"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "classes",
            "attributes",
        ]
        widgets = ["button_run"]
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
        self.label = self.config["classes"]
        self.attributes = self.config["attributes"]
        self.input_shape = self.net.get_input_shape()[-2:][::-1]

    def preprocess(self, input_image):
        """
        Post-processes the network's output.
        """
        image = cv2.resize(input_image, self.input_shape, interpolation=1)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = np.array(mean).reshape((1, 1, 3)).astype("float32")
        std = np.array(std).reshape((1, 1, 3)).astype("float32")
        image = (
            image.astype("float32") * np.float32(1.0 / 255.0) - mean
        ) / std
        image = image.transpose(2, 0, 1).astype("float32")
        image = np.expand_dims(image, axis=0)
        return image

    def postprocess(self, outs):
        """
        Predict shapes from image
        """
        outs = outs.tolist()

        interval = 0
        results = {}
        for property, infos in self.attributes.items():
            options, threshold = infos
            if threshold == -1:
                num_classes = len(options)
                current_class = outs[interval : interval + num_classes]
                current_index = np.argmax(current_class)
                results[property] = options[current_index]
                interval += num_classes
            elif 0.0 <= threshold <= 1.0:
                current_score = outs[interval]
                current_class = (
                    options[0] if current_score > threshold else options[1]
                )
                results[property] = current_class
                interval += 1

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
        outputs = self.net.get_ort_inference(blob, squeeze=True)
        attributes = self.postprocess(outputs)
        shapes = []
        shape = Shape(
            label=self.label,
            attributes=attributes,
            shape_type="rectangle",
        )
        h, w = image.shape[:2]
        shape.add_point(QtCore.QPointF(0, 0))
        shape.add_point(QtCore.QPointF(w, 0))
        shape.add_point(QtCore.QPointF(w, h))
        shape.add_point(QtCore.QPointF(0, h))
        shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.net
