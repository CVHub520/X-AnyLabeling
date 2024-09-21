import os
import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .utils import softmax
from .engines.build_onnx_engine import OnnxBaseModel


class InternImage_CLS(Model):
    """Image Classification model using InternImage"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize InternImage model.",
                )
            )
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]
        self.input_shape = self.net.get_input_shape()[-2:]

    def preprocess(self, input_image, mean=None, std=None):
        """
        Classification:
            Pre-processes the input image before feeding it to the network.

        Args:
            input_image (numpy.ndarray): The input image to be processed.
            mean (numpy.ndarray): Mean values for normalization.
                If not provided, default values are used.
            std (numpy.ndarray): Standard deviation values for normalization.
                If not provided, default values are used.

        Returns:
            numpy.ndarray: The processed input image.
        """
        h, w = self.input_shape
        # Resize the input image
        input_data = cv2.resize(input_image, (w, h))
        # Transpose the dimensions of the image
        input_data = input_data.transpose((2, 0, 1))
        if not mean:
            mean = np.array([0.485, 0.456, 0.406])
        if not std:
            std = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(input_data.shape).astype("float32")
        # Normalize the image data
        for channel in range(input_data.shape[0]):
            norm_img_data[channel, :, :] = (
                input_data[channel, :, :] / 255 - mean[channel]
            ) / std[channel]
        blob = norm_img_data.reshape(1, 3, h, w).astype("float32")
        return blob

    def postprocess(self, outs, topk=1):
        """
        Classification:
            Post-processes the output of the network.

        Args:
            outs (list): Output predictions from the network.
            topk (int): Number of top predictions to consider.

        Returns:
            str: Predicted label.
        """
        res = softmax(np.array(outs)).tolist()
        index = np.argmax(res)
        label = str(self.classes[index])

        return label

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
        predictions = self.net.get_ort_inference(blob, extract=False)
        label = self.postprocess(predictions)
        result = AutoLabelingResult(
            shapes=[], replace=False, description=label
        )
        return result

    def unload(self):
        del self.net
