import os
import cv2
import math
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .utils.general import letterbox
from .utils.points_conversion import rbox2poly
from .engines.build_onnx_engine import OnnxBaseModel


class YOLOv5OBB(Model):
    """Rotation model using YOLOv5OBB"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "stride",
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
            "rotation": QCoreApplication.translate("Model", "Rotation"),
        }
        default_output_mode = "rotation"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize YOLOv5OBB model.",
                )
            )

        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.stride = self.config["stride"]
        self.classes = self.config["classes"]
        self.nms_thres = self.config["iou_threshold"]
        self.conf_thres = self.config["conf_threshold"]

        _, _, h, w = self.net.get_input_shape()
        self.input_shape = (h, w)
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

    def preprocess(self, img):
        """
        Pre-process the input RGB image before feeding it to the network.
        """
        img = letterbox(img, self.input_shape, auto=False, stride=self.stride)[
            0
        ]
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img).astype("float32")
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img

    def postprocess(self, outputs, old_shape):
        """
        Post-process the network's output
        """
        nc = outputs.shape[2] - 5 - 180  # number of classes

        xc = outputs[..., 4] > self.conf_thres
        outputs = outputs[:][xc]

        generate_boxes, bboxes, scores = [], [], []
        for out in outputs:
            cx, cy, longside, shortside, obj_score = out[:5]
            class_scores = out[5 : 5 + nc]
            class_idx = np.argmax(class_scores)

            max_class_score = class_scores[class_idx] * obj_score
            if max_class_score < self.conf_thres:
                continue

            theta_scores = out[5 + nc :]
            theta_idx = np.argmax(theta_scores)
            theta_pred = (theta_idx - 90) / 180 * np.pi

            bboxes.append([[cx, cy], [longside, shortside], max_class_score])
            scores.append(max_class_score)
            generate_boxes.append(
                [
                    cx,
                    cy,
                    longside,
                    shortside,
                    theta_pred,
                    max_class_score,
                    class_idx,
                ]
            )
        indices = cv2.dnn.NMSBoxesRotated(
            bboxes, scores, self.conf_thres, self.nms_thres
        )
        try:
            det = np.array(generate_boxes)[indices.flatten()]
            pred_poly = rbox2poly(det[:, :5])
            pred_poly = self.scale_polys(pred_poly, old_shape)
            # (n, [poly conf cls])
            results = np.concatenate((pred_poly, det[:, -2:]), axis=1)
            return results
        except Exception:
            return []

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

        blob = self.preprocess(image)
        detections = self.net.get_ort_inference(blob)
        results = self.postprocess(detections, image.shape)

        shapes = []
        for *poly, _, cls_id in reversed(results):
            label = str(self.classes[int(cls_id)])
            x0, y0, x1, y1, x2, y2, x3, y3 = poly
            direction = self.calculate_rotation_theta(poly)
            shape = Shape(
                label=label, shape_type="rotation", direction=direction
            )
            shape.add_point(QtCore.QPointF(x0, y0))
            shape.add_point(QtCore.QPointF(x1, y1))
            shape.add_point(QtCore.QPointF(x2, y2))
            shape.add_point(QtCore.QPointF(x3, y3))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=self.replace)
        return result

    @staticmethod
    def calculate_rotation_theta(poly):
        x1, y1, x2, y2 = poly[:4]

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

    def scale_polys(self, polys, old_shape, ratio_pad=None):
        # ratio_pad: [(h_raw, w_raw), (hw_ratios, wh_paddings)]
        # Rescale coords (xyxyxyxy) from new_shape to old_shape
        new_shape = self.input_shape
        # calculate from old_shape
        if ratio_pad is None:
            # gain  = resized / raw
            gain = min(
                new_shape[0] / old_shape[0],
                new_shape[1] / old_shape[1],
            )
            # wh padding
            pad = (new_shape[1] - old_shape[1] * gain) / 2, (
                new_shape[0] - old_shape[0] * gain
            ) / 2
        else:
            gain = ratio_pad[0][0]  # h_ratios
            pad = ratio_pad[1]  # wh_paddings
        polys[:, [0, 2, 4, 6]] -= pad[0]  # x padding
        polys[:, [1, 3, 5, 7]] -= pad[1]  # y padding
        # Rescale poly shape to img0_shape
        polys[:, :8] /= gain
        return polys

    def unload(self):
        del self.net
