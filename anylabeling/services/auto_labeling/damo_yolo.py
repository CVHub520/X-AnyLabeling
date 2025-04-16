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


class DAMO_YOLO(Model):
    """Object detection model using DAMO_YOLO"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
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

        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize DAMO_YOLO model.",
                )
            )

        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]
        self.filter_classes = self.config.get("filter_classes", [])
        self.input_shape = self.net.get_input_shape()
        self.input_size = self.input_shape[-2:]
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
        src_h, src_w, _ = input_image.shape
        _, dst_c, dst_h, dst_w = self.input_shape
        transformed_image = np.ones((dst_h, dst_w, dst_c), dtype=np.uint8)
        ratio_hw = min(dst_h / src_h, dst_w / src_w)
        new_h, new_w = int(ratio_hw * src_h), int(ratio_hw * src_w)
        image = cv2.resize(
            input_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        transformed_image[:new_h, :new_w, :] = image
        transformed_image = transformed_image.transpose((2, 0, 1))
        transformed_image = np.ascontiguousarray(transformed_image).astype(
            "float32"
        )
        if len(transformed_image.shape) == 3:
            transformed_image = transformed_image[None]
        return transformed_image, ratio_hw

    def postprocess(self, predictions, ratio_hw):
        scores = predictions[0].squeeze(axis=0)
        bboxes = predictions[1].squeeze(axis=0)
        bboxes /= ratio_hw

        boxes, confidences, class_ids = [], [], []
        for i in range(len(bboxes)):
            score = np.max(scores[i, :])
            if score < self.conf_thres:
                continue
            class_id = np.argmax(scores[i, :])
            if (
                self.filter_classes
                and self.classes[int(class_id)] not in self.filter_classes
            ):
                continue
            xmin, ymin, xmax, ymax = bboxes[i, :].astype(np.int32)
            width = xmax - xmin
            height = ymax - ymin
            boxes.append([xmin, ymin, width, height])
            confidences.append(score)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_thres, self.nms_thres
        )
        output_infos = []
        for i in indices:
            x, y, w, h = boxes[i]
            output_info = {
                "xmin": x,
                "ymin": y,
                "xmax": x + w,
                "ymax": y + h,
                "label": str(self.classes[int(class_ids[i])]),
                "score": float(confidences[i]),
            }
            output_infos.append(output_info)

        return output_infos

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
        predictions = self.net.get_ort_inference(blob, extract=False)
        results = self.postprocess(predictions, ratio_hw)

        shapes = []
        for result in results:
            shape = Shape(
                label=result["label"],
                score=result["score"],
                shape_type="rectangle",
            )
            xmin = result["xmin"]
            ymin = result["ymin"]
            xmax = result["xmax"]
            ymax = result["ymax"]
            pt1 = QtCore.QPointF(xmin, ymin)
            pt2 = QtCore.QPointF(xmax, ymin)
            pt3 = QtCore.QPointF(xmax, ymax)
            pt4 = QtCore.QPointF(xmin, ymax)
            shape.add_point(pt1)
            shape.add_point(pt2)
            shape.add_point(pt3)
            shape.add_point(pt4)
            shapes.append(shape)
        result = AutoLabelingResult(shapes, replace=self.replace)
        return result

    def unload(self):
        del self.net
