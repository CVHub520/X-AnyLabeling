import os
import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .types import AutoLabelingResult
from .utils import softmax
from .__base__.yolo import YOLO
from .engines.build_onnx_engine import OnnxBaseModel


class YOLOv5_ResNet(YOLO):
    class Meta:
        required_config_names = [
            "cls_model_path",
            "det_classes",
            "cls_classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        """Classify"""
        model_abs_path = self.get_model_abs_path(self.config, "cls_model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {self.config['type']} model.",
                )
            )
        self.cls_net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.cls_classes = self.config["cls_classes"]
        self.cls_input_shape = self.cls_net.get_input_shape()[-2:]

        """Detection"""
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {self.config['type']} model.",
                )
            )
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        _, _, self.input_height, self.input_width = self.net.get_input_shape()
        if not isinstance(self.input_width, int):
            self.input_width = self.config.get("input_width", -1)
        if not isinstance(self.input_height, int):
            self.input_height = self.config.get("input_height", -1)

        self.model_type = self.config["type"]
        self.classes = self.config["det_classes"]
        self.anchors = self.config.get("anchors", None)
        self.agnostic = self.config.get("agnostic", False)
        self.show_boxes = self.config.get("show_boxes", False)
        self.strategy = self.config.get("strategy", "largest")
        self.iou_thres = self.config.get("nms_threshold", 0.45)
        self.conf_thres = self.config.get("confidence_threshold", 0.25)
        self.filter_classes = self.config.get("filter_classes", None)

        self.task = "det"
        self.nc = len(self.classes)
        self.input_shape = (self.input_height, self.input_width)
        if self.anchors:
            self.nl = len(self.anchors)
            self.na = len(self.anchors[0]) // 2
            self.grid = [np.zeros(1)] * self.nl
            self.stride = (
                np.array([self.stride // 4, self.stride // 2, self.stride])
                if not isinstance(self.stride, list)
                else np.array(self.stride)
            )
            self.anchor_grid = np.asarray(
                self.anchors, dtype=np.float32
            ).reshape(self.nl, -1, 2)
        if self.filter_classes:
            self.filter_classes = [
                i
                for i, item in enumerate(self.classes)
                if item in self.filter_classes
            ]

    def cls_preprocess(self, input_image, mean=None, std=None):
        """
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
        h, w = self.cls_input_shape
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

    def cls_postprocess(self, outs, topk=1):
        """
        Classification: Post-processes the output of the network.

        Args:
            outs (list): Output predictions from the network.
            topk (int): Number of top predictions to consider. Default is 1.

        Returns:
            str: Predicted label.
        """
        res = softmax(np.array(outs)).tolist()
        index = np.argmax(res)
        label = str(self.cls_classes[index])

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

        blob = self.preprocess(image, upsample_mode="letterbox")
        outputs = self.net.get_ort_inference(blob=blob, extract=False)
        boxes, _, _, _, _ = self.postprocess(outputs)

        shapes = []
        for box in boxes:
            x1, y1, x2, y2 = list(map(int, box))
            img = image[y1:y2, x1:x2]

            blob = self.cls_preprocess(img)
            predictions = self.cls_net.get_ort_inference(blob, extract=False)
            label = self.cls_postprocess(predictions)

            shape = Shape(label=label, shape_type="rectangle")
            shape.add_point(QtCore.QPointF(x1, y1))
            shape.add_point(QtCore.QPointF(x2, y1))
            shape.add_point(QtCore.QPointF(x2, y2))
            shape.add_point(QtCore.QPointF(x1, y2))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)

        return result

    def unload(self):
        del self.net
        del self.cls_net
