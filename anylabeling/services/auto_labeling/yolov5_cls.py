import logging
import os

import cv2
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
    rescale_box,
)
from .engines.build_onnx_engine import OnnxBaseModel


class YOLOv5_CLS(Model):
    """Object detection with Classify model using YOLOv5_CLS"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "det_model_path",
            "cls_model_path",
            "cls_score_threshold",
            "stride",
            "nms_threshold",
            "confidence_threshold",
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

        model_name = self.config['type']
        det_model_abs_path = self.get_model_abs_path(self.config, "det_model_path")
        if not det_model_abs_path or not os.path.isfile(det_model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", 
                    f"Could not download or initialize {model_name} model."
                )
            )
        cls_model_abs_path = self.get_model_abs_path(self.config, "cls_model_path")
        if not cls_model_abs_path or not os.path.isfile(cls_model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", 
                    f"Could not download or initialize {model_name} model."
                )
            )
        self.net = OnnxBaseModel(det_model_abs_path, __preferred_device__)
        self.classes = self.config["det_classes"]
        self.input_shape = self.net.get_input_shape()[-2:]
        self.nms_thres = self.config["nms_threshold"]
        self.conf_thres = self.config["confidence_threshold"]
        self.stride = self.config["stride"]
        self.anchors = self.config.get("anchors", None)
        self.agnostic = self.config.get("agnostic", False)
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

        self.cls_net = OnnxBaseModel(cls_model_abs_path, __preferred_device__)
        self.cls_classes = self.config["cls_classes"]
        self.cls_input_shape = self.cls_net.get_input_shape()[-2:]

    def det_preprocess(self, input_image):
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

    def det_postprocess(
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
        
        norm_img_data = np.zeros(input_data.shape).astype('float32')
        
        # Normalize the image data
        for channel in range(input_data.shape[0]):
            norm_img_data[channel, :, :] = (
                input_data[channel, :, :] / 255 - mean[channel]
            ) / std[channel]
        
        blob = norm_img_data.reshape(1, 3, h, w).astype('float32')
        
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
        res = self._softmax(np.array(outs)).tolist()
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
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        blob = self.det_preprocess(image)
        predictions = self.net.get_ort_inference(blob)
        results = self.det_postprocess(predictions)[0]

        if len(results) == 0: 
            return AutoLabelingResult([], replace=True)
        results[:, :4] = rescale_box(self.input_shape, results[:, :4], image.shape).round()
        shapes = []

        for *xyxy, _, _ in reversed(results):
            x1, y1, x2, y2 = [int(i) for i in xyxy]
            img = image[y1: y2, x1: x2]

            blob = self.cls_preprocess(img)
            predictions = self.cls_net.get_ort_inference(blob, extract=False)
            label = self.cls_postprocess(predictions)

            shape = Shape(label=label, shape_type="rectangle")
            shape.add_point(QtCore.QPointF(x1, y1))
            shape.add_point(QtCore.QPointF(x2, y2))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    @staticmethod
    def _softmax(x):
        """
        Applies the softmax function to the input array.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output array after applying softmax.
        """
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

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
        del self.cls_net
