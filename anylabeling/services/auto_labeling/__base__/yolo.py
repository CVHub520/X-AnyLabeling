import logging
import os

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from ..model import Model
from ..types import AutoLabelingResult
from ..utils import (
    numpy_nms,
    letterbox,
    xywh2xyxy,
    rescale_box,
)
from ..engines.build_onnx_engine import OnnxBaseModel


class YOLO(Model):

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "nms_threshold",
            "confidence_threshold",
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
        model_name = self.config['type']
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", 
                    f"Could not download or initialize {model_name} model."
                )
            )
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]
        self.input_shape = self.net.get_input_shape()[-2:]
        self.hide_box = self.config.get("hide_box", True)
        self.keypoints = self.config.get("keypoints", [])
        self.nms_thres = self.config["nms_threshold"]
        self.conf_thres = self.config["confidence_threshold"]
        self.stride = self.config.get("stride", 32)
        self.anchors = self.config.get("anchors", None)
        self.agnostic = self.config.get("agnostic", False)
        self.filter_classes = self.config.get("filter_classes", None)

        if self.anchors:
            self.nl = len(self.anchors)
            self.na = len(self.anchors[0]) // 2
            self.grid = [np.zeros(1)] * self.nl
            self.stride = np.array(
                [self.stride//4, self.stride//2, self.stride]
            ) if not isinstance(self.stride, list) else \
            np.array(self.stride)
            self.anchor_grid = np.asarray(
                self.anchors, dtype=np.float32
            ).reshape(self.nl, -1, 2)
        if self.filter_classes:
            self.filter_classes = [
                i for i, item in enumerate(self.classes) 
                if item in self.filter_classes
            ]

    def preprocess(self, input_image):
        """
        Pre-process the input RGB image before feeding it to the network.
        """
        image = letterbox(input_image, self.input_shape, stride=self.stride)[0]
        image = image.transpose((2, 0, 1)) # HWC to CHW
        image = np.ascontiguousarray(image).astype('float32')
        image /= 255  # 0 - 255 to 0.0 - 1.0
        if len(image.shape) == 3:
            image = image[None]
        return image

    def postprocess(
            self, 
            prediction, 
            multi_label=False, 
            max_det=1000,
        ):

        if self.anchors:
            prediction = self.scale_coords(prediction)

        num_classes = prediction.shape[2] - 5
        pred_candidates = np.logical_and(
            prediction[..., 4] > self.conf_thres, 
            np.max(prediction[..., 5:], axis=-1) > self.conf_thres
        )

        max_wh = 4096
        max_nms = 30000
        multi_label &= num_classes > 1

        output = [np.zeros((0, 6))] * prediction.shape[0]
        for img_idx, x in enumerate(prediction):  # image index, image inference
            x = x[pred_candidates[img_idx]]  # confidence

            # If no box remains, skip the next process.
            if not x.shape[0]:
                continue

            # confidence multiply the objectness
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
            if multi_label:
                box_idx, class_idx = np.nonzero(x[:, 5:] > self.conf_thres)
                box = box[box_idx]
                conf = x[box_idx, class_idx + 5][:, None]
                class_idx = class_idx[:, None].astype(float)
                x = np.concatenate((box, conf, class_idx), axis=1)
            else:
                conf = np.max(x[:, 5:], axis=1, keepdims=True)
                class_idx = np.argmax(x[:, 5:], axis=1)
                x = np.concatenate(
                    (box, conf, class_idx[:, None].astype(float)), axis=1
                )[conf.flatten() > self.conf_thres]

            # Filter by class, only keep boxes whose category is in classes.
            if self.filter_classes:
                x = x[(x[:, 5:6] == np.array(self.filter_classes)).any(1)]

            # Check shape
            num_box = x.shape[0]  # number of boxes
            if not num_box:  # no boxes kept.
                continue
            elif num_box > max_nms:  # excess max boxes' number.
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            class_offset = x[:, 5:6] * (0 if self.agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
            keep_box_idx = numpy_nms(boxes, scores, self.nms_thres)  # NMS
            if keep_box_idx.shape[0] > max_det:  # limit detections
                keep_box_idx = keep_box_idx[:max_det]

            output[img_idx] = x[keep_box_idx]

        return output

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        blob = self.preprocess(image)
        predictions = self.net.get_ort_inference(blob)
        results = self.postprocess(predictions)[0]

        if len(results) == 0: 
            return AutoLabelingResult([], replace=True)
        results[:, :4] = rescale_box(self.input_shape, results[:, :4], image.shape).round()
        shapes = []
        for *xyxy, _, cls_id in reversed(results):
            rectangle_shape = Shape(
                label=str(self.classes[int(cls_id)]), 
                shape_type="rectangle",
            )
            rectangle_shape.add_point(QtCore.QPointF(xyxy[0], xyxy[1]))
            rectangle_shape.add_point(QtCore.QPointF(xyxy[2], xyxy[3]))
            shapes.append(rectangle_shape)
        result = AutoLabelingResult(shapes, replace=True)

        return result

    @staticmethod
    def make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def scale_coords(self, outs):
        outs = outs[0]
        row_ind = 0
        for i in range(self.nl):
            h = int(self.input_shape[0] / self.stride[i])
            w = int(self.input_shape[1] / self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2: 4] != (h, w):
                self.grid[i] = self.make_grid(w, h)
            outs[row_ind:row_ind + length, 0:2] = \
                (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + \
                    np.tile(self.grid[i], (self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = \
                (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * \
                    np.repeat(self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        return outs[np.newaxis, :]

    def unload(self):
        del self.net
