import os

import cv2
import numpy as np

from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .engines import OnnxBaseModel
from .types import AutoLabelingResult


def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFD(Model):
    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
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
        self.input_name = self.net.get_input_name()
        self.output_names = self.net.get_output_name()
        self.input_width = self.config.get("input_width", 640)
        self.input_height = self.config.get("input_height", 640)
        self.conf_thres = self.config.get("conf_threshold", 0.5)
        self.iou_thres = self.config.get("iou_threshold", 0.4)
        self.max_det = self.config.get("max_det", 0)
        self.replace = True
        self.center_cache = {}
        self.classes = self.config.get("classes", {})
        self.keypoint_name = {
            class_name: keypoints
            for class_name, keypoints in self.classes.items()
        }
        self.label = next(iter(self.classes), "face")
        self._init_vars()

    def _init_vars(self):
        output_count = len(self.output_names)
        self.use_kps = False
        self.fmc = 3
        self.feat_stride_fpn = [8, 16, 32]
        self.num_anchors = 2
        if output_count == 9:
            self.use_kps = True
        elif output_count == 10:
            self.fmc = 5
            self.feat_stride_fpn = [8, 16, 32, 64, 128]
            self.num_anchors = 1
        elif output_count == 15:
            self.fmc = 5
            self.feat_stride_fpn = [8, 16, 32, 64, 128]
            self.num_anchors = 1
            self.use_kps = True
        elif output_count != 6:
            raise ValueError(
                QCoreApplication.translate(
                    "Model",
                    "Unsupported SCRFD output count: {output_count}",
                ).format(output_count=output_count)
            )

    def set_auto_labeling_conf(self, value):
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_iou(self, value):
        if value > 0:
            self.iou_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        self.replace = not state

    def preprocess(self, image):
        input_size = (self.input_width, self.input_height)
        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized_img = cv2.resize(image, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        blob = det_img.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, :, :, :]
        return np.ascontiguousarray(blob), det_scale

    def get_anchor_centers(self, height, width, stride):
        key = (height, width, stride)
        if key in self.center_cache:
            return self.center_cache[key]
        anchor_centers = np.stack(
            np.mgrid[:height, :width][::-1], axis=-1
        ).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if self.num_anchors > 1:
            anchor_centers = np.stack(
                [anchor_centers] * self.num_anchors, axis=1
            ).reshape((-1, 2))
        if len(self.center_cache) < 100:
            self.center_cache[key] = anchor_centers
        return anchor_centers

    def forward(self, blob):
        net_outs = self.net.get_ort_inference(
            inputs={self.input_name: blob}, extract=False
        )
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_height = blob.shape[2]
        input_width = blob.shape[3]

        for idx, stride in enumerate(self.feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.fmc] * stride
            if scores.ndim == 3:
                scores = scores[0]
                bbox_preds = bbox_preds[0]
            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            if self.use_kps:
                kps_preds = net_outs[idx + self.fmc * 2] * stride
                if kps_preds.ndim == 3:
                    kps_preds = kps_preds[0]
                kps_preds = kps_preds.reshape(-1, 10)

            height = input_height // stride
            width = input_width // stride
            anchor_centers = self.get_anchor_centers(height, width, stride)
            pos_inds = np.where(scores >= self.conf_thres)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds, np.newaxis])
            bboxes_list.append(bboxes[pos_inds])
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kpss_list.append(kpss[pos_inds])

        return scores_list, bboxes_list, kpss_list

    def nms(self, dets):
        if dets.size == 0:
            return []
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
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
            overlap = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(overlap <= self.iou_thres)[0]
            order = order[inds + 1]
        return keep

    def postprocess(self, blob, det_scale):
        scores_list, bboxes_list, kpss_list = self.forward(blob)
        if not scores_list or sum(len(scores) for scores in scores_list) == 0:
            return np.empty((0, 5), dtype=np.float32), None

        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        else:
            kpss = None

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if kpss is not None:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        if self.max_det > 0 and det.shape[0] > self.max_det:
            det = det[: self.max_det, :]
            if kpss is not None:
                kpss = kpss[: self.max_det, :, :]
        return det, kpss

    def create_shapes(self, det, kpss):
        shapes = []
        kpt_names = self.keypoint_name.get(self.label, [])
        for i, bbox in enumerate(reversed(det)):
            x1, y1, x2, y2, score = bbox
            rectangle_shape = Shape(
                label=self.label,
                shape_type="rectangle",
                group_id=int(i),
                score=float(score),
            )
            rectangle_shape.add_point(QtCore.QPointF(int(x1), int(y1)))
            rectangle_shape.add_point(QtCore.QPointF(int(x2), int(y1)))
            rectangle_shape.add_point(QtCore.QPointF(int(x2), int(y2)))
            rectangle_shape.add_point(QtCore.QPointF(int(x1), int(y2)))
            shapes.append(rectangle_shape)
            if kpss is None:
                continue
            kps = kpss[len(det) - 1 - i]
            for j, point in enumerate(kps):
                if j >= len(kpt_names):
                    break
                point_shape = Shape(
                    label=kpt_names[j],
                    shape_type="point",
                    group_id=int(i),
                )
                point_shape.add_point(
                    QtCore.QPointF(int(point[0]), int(point[1]))
                )
                shapes.append(point_shape)
        return shapes

    def predict_shapes(self, image, image_path=None):
        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        blob, det_scale = self.preprocess(image)
        det, kpss = self.postprocess(blob, det_scale)
        if len(det) == 0:
            return AutoLabelingResult([], replace=self.replace)
        shapes = self.create_shapes(det, kpss)
        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        del self.net
