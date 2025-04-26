import os
import numpy as np
from PIL import Image

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel
from .utils.general import sigmoid
from .utils.points_conversion import cxcywh2xyxy


class RFDETR(Model):
    """Object detection model using RF-DETR"""

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

        input_width = self.config.get("input_width", 560)
        input_height = self.config.get("input_height", 560)
        self.input_shape = (input_height, input_width)

        self.conf_thres = self.config.get("conf_threshold", 0.50)
        self.num_select = self.config.get("num_select", 300)
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
            input_image (PIL.Image.Image): The input image to be processed.

        Returns:
            numpy.ndarray: The pre-processed output.
        """
        # resize with bilinear interpolation
        image = input_image.resize(self.input_shape, Image.BILINEAR)

        # convert to numpy array
        image = np.array(image)

        # div 255
        image = image.astype(np.float32) / 255.0

        # transpose to CHW format last
        image = image.transpose((2, 0, 1))

        # normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(-1, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(-1, 1, 1)
        image = (image - mean) / std

        # add batch dimension
        image = np.expand_dims(image, axis=0)

        # convert to contiguous array
        image = np.ascontiguousarray(image)

        return image

    def postprocess(self, outs, image_shape):
        """
        Post-processes the network's output.

        Args:
            input_image (numpy.ndarray): The input image.
            outputs (numpy.ndarray): The output from the network.

        Returns:
            list: List of dictionaries containing the output
                    bounding boxes, labels, and scores.
        """
        out_bbox = outs[0]
        out_logits = outs[1]

        prob = sigmoid(out_logits)
        prob_reshaped = prob.reshape(out_logits.shape[0], -1)

        topk_indexes = np.argpartition(-prob_reshaped, self.num_select, axis=1)[:, :self.num_select]
        topk_values = np.take_along_axis(prob_reshaped, topk_indexes, axis=1)

        sort_indices = np.argsort(-topk_values, axis=1)
        topk_values = np.take_along_axis(topk_values, sort_indices, axis=1)
        topk_indexes = np.take_along_axis(topk_indexes, sort_indices, axis=1)

        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        boxes = cxcywh2xyxy(out_bbox)

        topk_boxes_expanded = np.expand_dims(topk_boxes, axis=-1)
        topk_boxes_tiled = np.tile(topk_boxes_expanded, (1, 1, 4))

        boxes = np.take_along_axis(boxes, topk_boxes_tiled, axis=1)
        img_h, img_w = image_shape
        scale_fct = np.array([[img_w, img_h, img_w, img_h]], dtype=np.float32)
        boxes = boxes * scale_fct[:, None, :]

        keep = scores[0] > self.conf_thres
        scores = scores[0][keep].tolist()
        labels = labels[0][keep].tolist()
        boxes = boxes[0][keep].tolist()

        return boxes, scores, labels

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = Image.open(image_path)
            image_shape = image.size[::-1]  # (height, width)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        blob = self.preprocess(image)
        detections = self.net.get_ort_inference(
            blob, extract=False, squeeze=False
        )
        boxes, scores, labels = self.postprocess(detections, image_shape)
        shapes = []

        for box, score, label in zip(boxes, scores, labels):
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
            shape = Shape(
                label=self.classes[label],
                score=score,
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
