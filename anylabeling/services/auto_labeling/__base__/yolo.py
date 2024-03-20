import logging
import os

import cv2
import math
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from ..model import Model
from ..engines import OnnxBaseModel
from ..types import AutoLabelingResult
from ..trackers import ByteTrack, OcSort
from ..utils import (
    letterbox,
    scale_boxes,
    scale_coords,
    masks2segments,
    xywhr2xyxyxyxy,
    non_max_suppression_v5,
    non_max_suppression_v8,
)


class YOLO(Model):
    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
        ]
        widgets = ["button_run"]
        output_modes = {
            "point": QCoreApplication.translate("Model", "Point"),
            "polygon": QCoreApplication.translate("Model", "Polygon"),
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
                    f"Could not download or initialize {self.config['type']} model.",
                )
            )

        self.engine = self.config.get("engine", "ort")
        if self.engine.lower() == "dnn":
            from ..engines import DnnBaseModel

            self.net = DnnBaseModel(model_abs_path, __preferred_device__)
            self.input_width = self.config.get("input_width", 640)
            self.input_height = self.config.get("input_height", 640)
        else:
            self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
            (
                _,
                _,
                self.input_height,
                self.input_width,
            ) = self.net.get_input_shape()
            if not isinstance(self.input_width, int):
                self.input_width = self.config.get("input_width", -1)
            if not isinstance(self.input_height, int):
                self.input_height = self.config.get("input_height", -1)

        self.model_type = self.config["type"]
        self.classes = self.config.get("classes", [])
        if isinstance(self.classes, dict):
            self.classes = list(self.classes.values())
        self.stride = self.config.get("stride", 32)
        self.anchors = self.config.get("anchors", None)
        self.agnostic = self.config.get("agnostic", False)
        self.show_boxes = self.config.get("show_boxes", False)
        self.epsilon_factor = self.config.get("epsilon_factor", 0.005)
        self.iou_thres = self.config.get("nms_threshold", 0.45)
        self.conf_thres = self.config.get("confidence_threshold", 0.25)
        self.filter_classes = self.config.get("filter_classes", None)
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

        """Tracker"""
        tracker = self.config.get("tracker", None)
        if tracker == "ocsort":
            self.tracker = OcSort(self.input_shape)
        elif tracker == "bytetrack":
            self.tracker = ByteTrack(self.input_shape)
        else:
            self.tracker = None

        """Keypoints"""
        self.keypoints = self.config.get("keypoints", [])
        self.five_key_points_classes = self.config.get(
            "five_key_points_classes", []
        )

        if self.model_type in [
            "yolov5",
            "yolov6",
            "yolov7",
            "yolov8",
            "yolov9",
            "gold_yolo",
        ]:
            self.task = "det"
        elif self.model_type in ["yolov5_seg", "yolov8_seg"]:
            self.task = "seg"
        elif self.model_type in [
            "yolov5_track",
            "yolov8_track",
        ]:
            self.task = "track"
        elif self.model_type in [
            "yolov8_obb",
        ]:
            self.task = "obb"

    def inference(self, blob):
        if self.engine == "dnn" and self.task in ["det", "seg", "track"]:
            outputs = self.net.get_dnn_inference(blob=blob, extract=False)
            if self.task == "det" and not isinstance(outputs, (tuple, list)):
                outputs = [outputs]
        else:
            outputs = self.net.get_ort_inference(blob=blob, extract=False)
        return outputs

    def preprocess(self, image, upsample_mode="letterbox"):
        self.img_height, self.img_width = image.shape[:2]
        # Upsample
        if upsample_mode == "resize":
            input_img = cv2.resize(
                image, (self.input_width, self.input_height)
            )
        elif upsample_mode == "letterbox":
            input_img = letterbox(image, self.input_shape)[0]
        elif upsample_mode == "centercrop":
            m = min(self.img_height, self.img_width)
            top = (self.img_height - m) // 2
            left = (self.img_width - m) // 2
            cropped_img = image[top : top + m, left : left + m]
            input_img = cv2.resize(
                cropped_img, (self.input_width, self.input_height)
            )
        # Transpose
        input_img = input_img.transpose(2, 0, 1)
        # Expand
        input_img = input_img[np.newaxis, :, :, :].astype(np.float32)
        # Contiguous
        input_img = np.ascontiguousarray(input_img)
        # Norm
        blob = input_img / 255.0
        return blob

    def postprocess(self, preds):
        if self.model_type in [
            "yolov5",
            "yolov5_resnet",
            "yolov5_ram",
            "yolov5_sam",
            "yolov5_seg",
            "yolov5_track",
            "yolov6",
            "yolov7",
            "gold_yolo",
        ]:
            # Only support YOLOv5 version 5.0 and earlier versions
            if self.model_type == "yolov5" and self.anchors:
                preds = self.scale_grid(preds)
            p = non_max_suppression_v5(
                preds[0],
                task=self.task,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                classes=self.filter_classes,
                agnostic=self.agnostic,
                multi_label=False,
                nc=self.nc,
            )
        elif self.model_type in [
            "yolov8",
            "yolov8_efficientvit_sam",
            "yolov8_seg",
            "yolov8_track",
            "yolov8_obb",
            "yolov9",
        ]:
            p = non_max_suppression_v8(
                preds[0],
                task=self.task,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                classes=self.filter_classes,
                agnostic=self.agnostic,
                multi_label=False,
                nc=self.nc,
            )

        masks = None
        img_shape = (self.img_height, self.img_width)
        if self.task == "seg":
            proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
            self.mask_height, self.mask_width = proto.shape[2:]
        for i, pred in enumerate(p):
            if self.task == "seg":
                if np.size(pred) == 0:
                    continue
                masks = self.process_mask(
                    proto[i],
                    pred[:, 6:],
                    pred[:, :4],
                    self.input_shape,
                    upsample=True,
                )  # HWC
            if self.task == "obb":
                pred[:, :4] = scale_boxes(
                    self.input_shape, pred[:, :4], img_shape, xywh=True
                )
            else:
                pred[:, :4] = scale_boxes(
                    self.input_shape, pred[:, :4], img_shape
                )

        if self.task == "obb":
            pred = np.concatenate(
                [pred[:, :4], pred[:, -1:], pred[:, 4:6]], axis=-1
            )
            bbox = pred[:, :5]
            conf = pred[:, -2]
            clas = pred[:, -1]
        else:
            bbox = pred[:, :4]
            conf = pred[:, 4:5]
            clas = pred[:, 5:6]
        return (bbox, masks, clas, conf)

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

        blob = self.preprocess(image, upsample_mode="letterbox")
        outputs = self.inference(blob)
        boxes, masks, class_ids, scores = self.postprocess(outputs)
        points = [[] for _ in range(len(boxes))]
        if self.task == "seg" and masks is not None:
            points = [
                scale_coords(self.input_shape, x, image.shape, normalize=False)
                for x in masks2segments(masks, self.epsilon_factor)
            ]
        track_ids = [[] for _ in range(len(boxes))]
        if self.task == "track":
            image_shape = image.shape[:2][::-1]
            results = np.concatenate((boxes, scores, class_ids), axis=1)
            boxes, track_ids, _, class_ids = self.tracker.track(
                results, image_shape
            )

        shapes = []
        for box, class_id, point, track_id in zip(
            boxes, class_ids, points, track_ids
        ):
            if (
                self.show_boxes and self.task != "track"
            ) or self.task == "det":
                x1, y1, x2, y2 = box.astype(float)
                shape = Shape(flags={})
                shape.add_point(QtCore.QPointF(x1, y1))
                shape.add_point(QtCore.QPointF(x2, y1))
                shape.add_point(QtCore.QPointF(x2, y2))
                shape.add_point(QtCore.QPointF(x1, y2))
                shape.shape_type = "rectangle"
                shape.closed = True
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.line_width = 1
                shape.label = str(self.classes[int(class_id)])
                shape.selected = False
                shapes.append(shape)
            if self.task == "seg":
                shape = Shape(flags={})
                for p in point:
                    shape.add_point(QtCore.QPointF(int(p[0]), int(p[1])))
                shape.shape_type = "polygon"
                shape.closed = True
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.line_width = 1
                shape.label = str(self.classes[int(class_id)])
                shape.selected = False
                shapes.append(shape)
            if self.task == "track":
                x1, y1, x2, y2 = list(map(float, box))
                shape = Shape(flags={})
                shape.add_point(QtCore.QPointF(x1, y1))
                shape.add_point(QtCore.QPointF(x2, y1))
                shape.add_point(QtCore.QPointF(x2, y2))
                shape.add_point(QtCore.QPointF(x1, y2))
                shape.shape_type = "rectangle"
                shape.group_id = int(track_id)
                shape.closed = True
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.line_width = 1
                shape.label = str(self.classes[int(class_id)])
                shape.selected = False
                shapes.append(shape)
            if self.task == "obb":
                poly = xywhr2xyxyxyxy(box)
                x0, y0 = poly[0]
                x1, y1 = poly[1]
                x2, y2 = poly[2]
                x3, y3 = poly[3]
                direction = self.calculate_rotation_theta(poly)
                shape = Shape(flags={})
                shape.add_point(QtCore.QPointF(x0, y0))
                shape.add_point(QtCore.QPointF(x1, y1))
                shape.add_point(QtCore.QPointF(x2, y2))
                shape.add_point(QtCore.QPointF(x3, y3))
                shape.shape_type = "rotation"
                shape.closed = True
                shape.direction = direction
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.line_width = 1
                shape.label = str(self.classes[int(class_id)])
                shape.selected = False
                shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)

        return result

    @staticmethod
    def make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    @staticmethod
    def calculate_rotation_theta(poly):
        x1, y1 = poly[0]
        x2, y2 = poly[1]

        # Calculate one of the diagonal vectors (after rotation)
        diagonal_vector_x = x2 - x1
        diagonal_vector_y = y2 - y1

        # Calculate the rotation angle in radians
        rotation_angle = math.atan2(diagonal_vector_y, diagonal_vector_x)

        # Convert radians to degrees
        rotation_angle_degrees = math.degrees(rotation_angle)

        if rotation_angle_degrees < 0:
            rotation_angle_degrees += 360

        return rotation_angle_degrees / 360 * (2 * math.pi)

    def scale_grid(self, outs):
        outs = outs[0]
        row_ind = 0
        for i in range(self.nl):
            h = int(self.input_shape[0] / self.stride[i])
            w = int(self.input_shape[1] / self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h, w):
                self.grid[i] = self.make_grid(w, h)
            outs[row_ind : row_ind + length, 0:2] = (
                outs[row_ind : row_ind + length, 0:2] * 2.0
                - 0.5
                + np.tile(self.grid[i], (self.na, 1))
            ) * int(self.stride[i])
            outs[row_ind : row_ind + length, 2:4] = (
                outs[row_ind : row_ind + length, 2:4] * 2
            ) ** 2 * np.repeat(self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        return outs[np.newaxis, :]

    def process_mask(self, protos, masks_in, bboxes, shape, upsample=False):
        """
        Apply masks to bounding boxes using the output of the mask head.

        Args:
            protos (np.ndarray): A tensor of shape [mask_dim, mask_h, mask_w].
            masks_in (np.ndarray): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
            bboxes (np.ndarray): A tensor of shape [n, 4], where n is the number of masks after NMS.
            shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
            upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

        Returns:
            (np.ndarray): A binary mask tensor of shape [n, h, w],
            where n is the number of masks after NMS, and h and w
            are the height and width of the input image.
            The mask is applied to the bounding boxes.
        """
        c, mh, mw = protos.shape
        ih, iw = shape
        masks = 1 / (
            1
            + np.exp(
                -np.dot(masks_in, protos.reshape(c, -1).astype(float)).astype(
                    float
                )
            )
        )
        masks = masks.reshape(-1, mh, mw)

        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih
        masks = self.crop_mask_np(masks, downsampled_bboxes)  # CHW
        if upsample:
            if masks.shape[0] == 1:
                masks_np = np.squeeze(masks, axis=0)
                masks_resized = cv2.resize(
                    masks_np,
                    (shape[1], shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                masks = np.expand_dims(masks_resized, axis=0)
            else:
                masks_np = np.transpose(masks, (1, 2, 0))
                masks_resized = cv2.resize(
                    masks_np,
                    (shape[1], shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                masks = np.transpose(masks_resized, (2, 0, 1))
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0

        return masks

    @staticmethod
    def crop_mask_np(masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

        Args:
        masks (np.ndarray): [n, h, w] array of masks.
        boxes (np.ndarray): [n, 4] array of bbox coordinates in relative point form.

        Returns:
        (np.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.hsplit(boxes[:, :, None], 4)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
        c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

        return masks * ((r >= x1) & (r < x2) & (c >= y1) & (c < y2))

    def unload(self):
        del self.net
