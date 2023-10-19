import logging
import os

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .utils import (
    numpy_nms,
    letterbox,
    xywh2xyxy,
    rescale_box_and_landmark
)

from .engines.build_onnx_engine import OnnxBaseModel


class YOLOv6Face(Model):
    """Object detection model using YOLOv6Face v4.0"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "stride",
            "nms_threshold",
            "confidence_threshold",
            "classes",
            "five_key_points_classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            "point": QCoreApplication.translate("Model", "Point"),
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
        self.nms_thres = self.config["nms_threshold"]
        self.conf_thres = self.config["confidence_threshold"]
        self.stride = self.config.get("stride", 32)
        self.anchors = self.config.get("anchors", None)
        self.agnostic = self.config.get("agnostic", False)
        self.filter_classes = self.config.get("filter_classes", None)
        self.kps_classes = self.config["five_key_points_classes"]

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
        """
        Post-process the network's output, to get the 
        bounding boxes, key-points and their confidence scores.
        """

        """Runs Non-Maximum Suppression (NMS) on inference results.
        Args:
            prediction: (tensor), with shape [N, 15 + num_classes], N is the number of bboxes.
            multi_label: (bool), when it is set to True, one box can have multi labels, 
                                                otherwise, one box only huave one label.
            max_det:(int), max number of output bboxes.
        Returns:
            list of detections, echo item is one tensor with shape (num_boxes, 16), 
                                                16 is for [xyxy, ldmks, conf, cls].
        """
        num_classes = prediction.shape[2] - 15  # number of classes
        pred_candidates = np.logical_and(
            prediction[..., 14] > self.conf_thres, 
            np.max(prediction[..., 15:], axis=-1) > self.conf_thres
        )

        # Function settings.
        max_wh = 4096  # maximum box width and height
        max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
        multi_label &= num_classes > 1  # multiple labels per box

        output = [np.zeros((0, 16))] * prediction.shape[0]

        for img_idx, x in enumerate(prediction):  # image index, image inference
            x = x[pred_candidates[img_idx]]  # confidence

            # If no box remains, skip the next process.
            if not x.shape[0]:
                continue

            # confidence multiply the objectness
            x[:, 15:] *= x[:, 14:15]  # conf = obj_conf * cls_conf

            # (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix's shape is  (n,16), each row represents (xyxy, conf, cls, lmdks)
            if multi_label:
                box_idx, class_idx = np.nonzero(x[:, 15:] > self.conf_thres).T
                x = np.concatenate((
                    box[box_idx], 
                    x[box_idx, class_idx + 15, None], 
                    class_idx[:, None].astype(np.float32), 
                    x[box_idx, 4:14],
                ), 1)
            else:
                conf = np.max(x[:, 15:], axis=1, keepdims=True)
                class_idx = np.argmax(x[:, 15:], axis=1, keepdims=True)
                x = np.concatenate((
                    box, conf, class_idx.astype(np.float32), x[:, 4:14]
                ), 1)[conf.ravel() > self.conf_thres]

            # Filter by class, only keep boxes whose category is in classes.
            if self.filter_classes:
                fc = [i for i, item in enumerate(self.classes) 
                      if item in self.filter_classes
                     ]
                x = x[(x[:, 5:6] == np.array(fc)).any(1)]

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
        results[:, :4], results[:, -10:] = rescale_box_and_landmark(
            self.input_shape, results[:, :4], results[:, -10:], image.shape
        )

        shapes = []
        for r in reversed(results):
            xyxy, _, cls_id, lmdks = r[:4], r[4], r[5], r[6:]
            x1, y1, x2, y2 = list(map(int, xyxy))
            lmdks = list(map(int, lmdks))
            label = self.classes[int(cls_id)]
            rectangle_shape = Shape(label=label, shape_type="rectangle")
            rectangle_shape.add_point(QtCore.QPointF(x1, y1))
            rectangle_shape.add_point(QtCore.QPointF(x2, y2))
            for i in range(0, len(lmdks), 2):
                x, y = lmdks[i], lmdks[i + 1]
                point_shape = Shape(label=self.kps_classes[i//2], shape_type="point")
                point_shape.add_point(QtCore.QPointF(x, y))
                shapes.append(point_shape)
        result = AutoLabelingResult(shapes, replace=True)

        return result

    def unload(self):
        del self.net
