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


class YOLOX(Model):
    """Object detection model using YOLOX v0.3.0"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "p6",
            "iou_threshold",
            "conf_threshold",
            "classes",
        ]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "input_iou",
            "edit_iou",
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
        self.p6 = self.config["p6"]
        self.classes = self.config["classes"]
        self.input_shape = self.net.get_input_shape()[-2:]
        self.nms_thres = self.config["iou_threshold"]
        self.conf_thres = self.config["conf_threshold"]
        self.replace = True

    def set_auto_labeling_conf(self, value):
        """set auto labeling confidence threshold"""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_iou(self, value):
        """set auto labeling iou threshold"""
        if value > 0:
            self.nms_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def preprocess(self, input_image):
        """
        Pre-process the input RGB image before feeding it to the network.
        """
        if len(input_image.shape) == 3:
            padded_img = (
                np.ones(
                    (self.input_shape[0], self.input_shape[1], 3),
                    dtype=np.uint8,
                )
                * 114
            )
        else:
            padded_img = np.ones(self.input_shape, dtype=np.uint8) * 114

        ratio_hw = min(
            self.input_shape[0] / input_image.shape[0],
            self.input_shape[1] / input_image.shape[1],
        )
        resized_img = cv2.resize(
            input_image,
            (
                int(input_image.shape[1] * ratio_hw),
                int(input_image.shape[0] * ratio_hw),
            ),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[
            : int(input_image.shape[0] * ratio_hw),
            : int(input_image.shape[1] * ratio_hw),
        ] = resized_img

        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img[None, :, :, :], ratio_hw

    def postprocess(self, outputs):
        """
        Post-process the network's output.
        """
        grids = []
        expanded_strides = []
        p6 = self.p6
        img_size = self.input_shape
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

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

        blob, ratio_hw = self.preprocess(image)
        outputs = self.net.get_ort_inference(blob)
        predictions = self.postprocess(outputs)[0]
        results = self.rescale(predictions, ratio_hw)

        shapes = []
        final_boxes, final_scores, final_cls_inds = (
            results[:, :4],
            results[:, 4],
            results[:, 5],
        )
        for box, score, cls_inds in zip(
            final_boxes, final_scores, final_cls_inds
        ):
            if score < self.conf_thres:
                continue
            x1, y1, x2, y2 = box
            score = float(score)
            label = str(self.classes[int(cls_inds)])
            rectangle_shape = Shape(
                label=label, score=score, shape_type="rectangle"
            )
            rectangle_shape.add_point(QtCore.QPointF(x1, y1))
            rectangle_shape.add_point(QtCore.QPointF(x2, y1))
            rectangle_shape.add_point(QtCore.QPointF(x2, y2))
            rectangle_shape.add_point(QtCore.QPointF(x1, y2))
            shapes.append(rectangle_shape)

        result = AutoLabelingResult(shapes, replace=self.replace)

        return result

    def rescale(self, predictions, ratio):
        """Rescale the output to the original image shape"""

        nms_thr = self.nms_thres
        score_thr = self.conf_thres

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio
        dets = self.multiclass_nms_class_agnostic(
            boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr
        )

        return dets

    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [
                    valid_boxes[keep],
                    valid_scores[keep, None],
                    valid_cls_inds[keep, None],
                ],
                1,
            )
        return dets

    @staticmethod
    def nms(boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def unload(self):
        del self.net
