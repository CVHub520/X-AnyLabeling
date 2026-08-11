import os

import cv2
import numpy as np
from PIL import Image
from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape

from .engines.build_onnx_engine import OnnxBaseModel
from .model import Model
from .types import AutoLabelingResult
from .utils.box import numpy_nms
from .utils.points_conversion import masks2segments


class DFINESeg(Model):
    """Instance segmentation model using D-FINE-seg."""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "classes",
        ]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "input_iou",
            "edit_iou",
            "toggle_preserve_existing_annotations",
            "button_classes_filter",
            "mask_fineness_slider",
            "mask_fineness_value_label",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {self.config['type']} model.",
                )
            )

        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]
        input_shape = self.net.get_input_shape()
        self.input_height = (
            input_shape[-2]
            if isinstance(input_shape[-2], int)
            else self.config.get("input_height", 640)
        )
        self.input_width = (
            input_shape[-1]
            if isinstance(input_shape[-1], int)
            else self.config.get("input_width", 640)
        )

        if len(self.net.get_output_name()) != 4:
            raise ValueError(
                "D-FINE-seg expects labels, boxes, scores, and masks outputs"
            )

        self.conf_thres = self.config.get("conf_threshold", 0.5)
        self.iou_thres = self.config.get("iou_threshold", 0.7)
        self.mask_thres = self.config.get("mask_threshold", 0.5)
        self.epsilon = self.config.get("epsilon_factor", 0.001)
        self.filter_classes = None
        self.replace = True

    def set_auto_labeling_conf(self, value):
        """Set the confidence threshold."""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_iou(self, value):
        """Set the IoU threshold."""
        if value > 0:
            self.iou_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle preservation of existing annotations."""
        self.replace = not state

    def set_auto_labeling_filter_classes(self, class_names):
        """Set filter classes by name."""
        if not class_names or len(class_names) == len(self.classes):
            self.filter_classes = None
        else:
            self.filter_classes = class_names

    def set_mask_fineness(self, epsilon):
        """Set the contour approximation factor."""
        self.epsilon = epsilon

    def preprocess(self, image):
        """Resize and normalize an RGB image for inference."""
        image = image.resize(
            (self.input_width, self.input_height), Image.BILINEAR
        )
        blob = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        return np.ascontiguousarray(blob[None] / 255.0)

    def postprocess(self, outputs, image_shape):
        """Scale model outputs to the original image size."""
        image_height, image_width = image_shape
        labels, boxes, scores, masks = outputs
        keep = scores[0] >= self.conf_thres
        labels = labels[0][keep].astype(int)
        boxes = boxes[0][keep].astype(np.float32)
        scores = scores[0][keep]
        masks = masks[0][keep]

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.input_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.input_height)
        valid = (
            (labels >= 0)
            & (labels < len(self.classes))
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        labels, boxes, scores = labels[valid], boxes[valid], scores[valid]
        masks = masks[valid]

        if len(boxes):
            max_coordinate = boxes.max() + 1
            nms_boxes = boxes + labels[:, None] * max_coordinate
            keep = numpy_nms(nms_boxes, scores, self.iou_thres)
            labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
            masks = masks[keep]

        boxes[:, [0, 2]] *= image_width / self.input_width
        boxes[:, [1, 3]] *= image_height / self.input_height

        masks = (
            np.stack(
                [
                    cv2.resize(
                        mask,
                        (image_width, image_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    >= self.mask_thres
                    for mask in masks
                ]
            )
            if len(masks)
            else np.empty((0, image_height, image_width), dtype=bool)
        )
        for mask, box in zip(masks, boxes):
            x1, y1, x2, y2 = box.astype(int)
            mask[:y1] = False
            mask[y2:] = False
            mask[:, :x1] = False
            mask[:, x2:] = False

        return boxes, scores, labels, masks

    def predict_shapes(self, image, image_path=None):
        """Predict shapes from an image."""
        if image is None:
            return []

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        blob = self.preprocess(image)
        outputs = self.net.get_ort_inference(blob, extract=False)
        result = self.postprocess(outputs, (image.height, image.width))

        boxes, scores, labels, masks = result
        segments = masks2segments(masks, self.epsilon)
        shapes = []
        for box, score, label_index, segment in zip(
            boxes, scores, labels, segments
        ):
            label = self.classes[int(label_index)]
            if self.filter_classes and label not in self.filter_classes:
                continue

            if self.output_mode == "polygon":
                if len(segment) < 3:
                    continue
                shape = Shape(
                    label=label,
                    score=float(score),
                    shape_type="polygon",
                )
                for x, y in segment:
                    shape.add_point(QtCore.QPointF(float(x), float(y)))
                shape.closed = True
            else:
                x1, y1, x2, y2 = box
                shape = Shape(
                    label=label,
                    score=float(score),
                    shape_type="rectangle",
                )
                shape.add_point(QtCore.QPointF(float(x1), float(y1)))
                shape.add_point(QtCore.QPointF(float(x2), float(y1)))
                shape.add_point(QtCore.QPointF(float(x2), float(y2)))
                shape.add_point(QtCore.QPointF(float(x1), float(y2)))
            shapes.append(shape)

        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        del self.net
