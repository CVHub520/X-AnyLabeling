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
from .__base__.rtmdet import RTMDet
from .pose.rtmo_onnx import RTMO


class RTMDet_Pose(Model):
    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "det_model_path",
            "pose_model_path",
            "pose",
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

        det_model_abs_path = self.get_model_abs_path(
            self.config, "det_model_path"
        )
        if not det_model_abs_path or not os.path.isfile(det_model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize RTMDet model."
                )
            )
        pose_model_abs_path = self.get_model_abs_path(
            self.config, "pose_model_path"
        )
        if not pose_model_abs_path or not os.path.isfile(pose_model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize Pose model."
                )
            )
        self.draw_det_box = self.config.get("draw_det_box", True)
        self.kpt_thr = self.config.get("kpt_threshold", 0.3)
        self.score_thr = self.config.get("score_threshold", 0.3)
        self.kpt_classes = self.config.get("keypoints", [])
        self.rtmdet = RTMDet(det_model_abs_path, score_thr=self.score_thr)
        if self.config["pose"] == "rtmo":
            self.pose = RTMO(pose_model_abs_path)
        else:
            self.pose = None

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

        det_results = self.rtmdet(image)

        shapes = []
        for i, bbox in enumerate(det_results):
            x1, y1, x2, y2 = list(map(int, bbox))

            if self.draw_det_box:
                rectangle_shape = Shape(
                    label="person", shape_type="rectangle", group_id=int(i)
                )
                rectangle_shape.add_point(QtCore.QPointF(x1, y1))
                rectangle_shape.add_point(QtCore.QPointF(x2, y1))
                rectangle_shape.add_point(QtCore.QPointF(x2, y2))
                rectangle_shape.add_point(QtCore.QPointF(x1, y2))
                shapes.append(rectangle_shape)

            img = image[y1:y2, x1:x2]
            try:
                keypoints, scores = self.pose(img)
            except:
                keypoints, scores = [], []
            if not self.pose and len(keypoints) == 0:
                continue
            for j in range(len(keypoints[0])):
                kpt_point, score = keypoints[0][j], scores[0][j]
                if score < self.kpt_thr:
                    continue
                point_shape = Shape(
                    label=str(self.kpt_classes[j]),
                    shape_type="point",
                    group_id=int(i),
                )
                x = int(kpt_point[0]) + x1
                y = int(kpt_point[1]) + y1
                point_shape.add_point(QtCore.QPointF(x, y))
                shapes.append(point_shape)

        result = AutoLabelingResult(shapes, replace=True)

        return result

    def unload(self):
        del self.rtmdet
        del self.pose
