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
from .utils.points_conversion import cxcywh2xyxy, masks2segments


class RFDETR(Model):
    """Object detection and instance segmentation model using RF-DETR"""

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
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        self.model_type = self.config["type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {self.model_type} model.",
                )
            )
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]

        _, _, input_height, input_width = self.net.get_input_shape()
        if not isinstance(input_width, int):
            default_input_width = 432 if "seg" in self.model_type else 560
            input_width = self.config.get("input_width", default_input_width)
        if not isinstance(input_height, int):
            default_input_height = 432 if "seg" in self.model_type else 560
            input_height = self.config.get(
                "input_height", default_input_height
            )
        self.input_shape = (input_height, input_width)

        self.num_outputs = len(self.net.ort_session.get_outputs())
        self.has_mask = self.num_outputs == 3

        self.conf_thres = self.config.get("conf_threshold", 0.50)
        self.num_select = self.config.get("num_select", 300)
        self.show_boxes = self.config.get("show_boxes", False)
        self.epsilon = self.config.get("epsilon", 0.001)
        self.replace = True

    def set_auto_labeling_conf(self, value):
        """set auto labeling confidence threshold"""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def set_mask_fineness(self, epsilon):
        """Set mask fineness epsilon value"""
        self.epsilon = epsilon

    def preprocess(self, input_image):
        """
        Pre-processes the input image before feeding it to the network.

        Args:
            input_image (PIL.Image.Image): The input image to be processed.

        Returns:
            numpy.ndarray: The pre-processed output.
        """
        # Convert grayscale to RGB if needed
        if input_image.mode == "L":
            input_image = input_image.convert("RGB")

        # resize with bilinear interpolation
        image = input_image.resize(self.input_shape, Image.BILINEAR)

        # convert to numpy array
        image = np.array(image)

        # div 255
        image = image.astype(np.float32) / 255.0

        # transpose to CHW format
        image = image.transpose((2, 0, 1))

        # normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(
            -1, 1, 1
        )
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(
            -1, 1, 1
        )
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
            outs (list): The output from the network.
            image_shape (tuple): The shape of the input image (height, width).

        Returns:
            tuple: Tuple containing bounding boxes, scores, labels, and masks.
        """
        out_bbox = outs[0]
        out_logits = outs[1]
        out_masks = outs[2] if len(outs) == 3 else None

        prob = sigmoid(out_logits)
        prob_reshaped = prob.reshape(out_logits.shape[0], -1)

        topk_indexes = np.argpartition(
            -prob_reshaped, self.num_select, axis=1
        )[:, : self.num_select]
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

        if out_masks is not None:
            masks = np.take_along_axis(
                out_masks, topk_boxes[:, :, None, None], axis=1
            )
            masks = masks[0]
            resized_masks = np.stack(
                [
                    np.array(Image.fromarray(mask).resize((img_w, img_h)))
                    for mask in masks
                ],
                axis=0,
            )
            masks = (resized_masks > 0).astype(np.uint8) * 255
        else:
            masks = None

        keep = scores[0] > self.conf_thres
        scores = scores[0][keep]
        labels = labels[0][keep]
        boxes = boxes[0][keep]
        if masks is not None:
            masks = masks[keep]

        return boxes, scores, labels, masks

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = Image.open(image_path)
            image_shape = image.size[::-1]
        except Exception as e:
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        blob = self.preprocess(image)
        detections = self.net.get_ort_inference(
            blob, extract=False, squeeze=False
        )
        boxes, scores, labels, masks = self.postprocess(
            detections, image_shape
        )
        shapes = []

        if self.has_mask and masks is not None:
            segments = masks2segments(masks, self.epsilon)
            for i, (segment, box, score, label) in enumerate(
                zip(segments, boxes, scores, labels)
            ):
                shape = Shape(
                    label=self.classes[int(label)],
                    score=float(score),
                    shape_type="polygon",
                )
                for point in segment:
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shape.closed = True
                shapes.append(shape)

                if self.show_boxes:
                    box_shape = Shape(
                        label=self.classes[int(label)],
                        score=float(score),
                        shape_type="rectangle",
                    )
                    box_shape.add_point(QtCore.QPointF(box[0], box[1]))
                    box_shape.add_point(QtCore.QPointF(box[2], box[1]))
                    box_shape.add_point(QtCore.QPointF(box[2], box[3]))
                    box_shape.add_point(QtCore.QPointF(box[0], box[3]))
                    shapes.append(box_shape)
        else:
            for box, score, label in zip(boxes, scores, labels):
                shape = Shape(
                    label=self.classes[int(label)],
                    score=float(score),
                    shape_type="rectangle",
                )
                shape.add_point(QtCore.QPointF(box[0], box[1]))
                shape.add_point(QtCore.QPointF(box[2], box[1]))
                shape.add_point(QtCore.QPointF(box[2], box[3]))
                shape.add_point(QtCore.QPointF(box[0], box[3]))
                shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=self.replace)
        return result

    def unload(self):
        del self.net
