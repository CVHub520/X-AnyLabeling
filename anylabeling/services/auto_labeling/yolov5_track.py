import logging
import os

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .types import AutoLabelingResult
from .utils import (
    rescale_box,
)
from .__base__.yolo import YOLO
from .engines.build_onnx_engine import OnnxBaseModel
from .trackers.byte_track.bytetracker import ByteTrack
from .trackers.oc_sort.ocsort import OcSort

class YOLOv5_Tracker(YOLO):
    """MOT model using YOLOv5_Tracker"""
    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "nms_threshold",
            "confidence_threshold",
            "classes",
            "tracker",
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
        if self.config["tracker"] == "ocsort":
            self.tracker = OcSort(self.input_shape)
        elif self.config["tracker"] == "bytetrack":
            self.tracker = ByteTrack(self.input_shape)
        else:
            raise NotImplementedError(
                QCoreApplication.translate(
                    "Model", "Not implemented tracker method."
                )
            )

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
        results[:, :4] = rescale_box(
            self.input_shape, results[:, :4], image.shape
        ).round()
        bboxes_xyxy, ids, _, class_ids = self.tracker.track(results, image.shape[:2][::-1])
        shapes = []
        for xyxy, id, class_id in zip(bboxes_xyxy, ids, class_ids):
            x0, y0, x1, y1 = [int(i) for i in (xyxy)]
            rectangle_shape = Shape(
                label=str(self.classes[int(class_id)]),
                shape_type="rectangle", 
                group_id=int(id)
            )
            rectangle_shape.add_point(QtCore.QPointF(x0, y0))
            rectangle_shape.add_point(QtCore.QPointF(x1, y1))
            shapes.append(rectangle_shape)
        
        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.net
