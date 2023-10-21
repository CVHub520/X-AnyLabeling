import logging
import os

import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .types import AutoLabelingResult
from .__base__.yolo import YOLO
from .__base__.sam import SegmentAnythingONNX
from .utils import rescale_box
from .engines.build_onnx_engine import OnnxBaseModel


class YOLOv5SegmentAnything(YOLO):
    """Segmentation model using YOLOv5 by SegmentAnything"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "target_size",
            "max_width",
            "max_height",
            "encoder_model_path",
            "decoder_model_path",
            "model_path",
            "stride",
            "nms_threshold",
            "confidence_threshold",
            "classes",
        ]
        widgets = [
            "button_run",
            "output_label",
            "output_select_combobox",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "polygon"

    def __init__(self, config_path, on_message) -> None:
        # Run the parent class's init method
        super().__init__(config_path, on_message)

        """YOLOv5 Model"""
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

        """Segment Anything Model"""
        # Get encoder the model paths
        encoder_model_abs_path = self.get_model_abs_path(
            self.config, "encoder_model_path"
        )
        if not encoder_model_abs_path or not os.path.isfile(
            encoder_model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize encoder of Segment Anything.",
                )
            )
        # Get decoder the model paths
        decoder_model_abs_path = self.get_model_abs_path(
            self.config, "decoder_model_path"
        )
        if not decoder_model_abs_path or not os.path.isfile(
            decoder_model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize decoder of Segment Anything.",
                )
            )

        # Load models
        self.target_size = self.config["target_size"]
        self.input_size = (self.config["max_height"], self.config["max_width"])
        self.encoder_session = OnnxBaseModel(encoder_model_abs_path, __preferred_device__)
        self.decoder_session = OnnxBaseModel(decoder_model_abs_path, __preferred_device__)
        self.model = SegmentAnythingONNX(
            self.encoder_session, self.decoder_session, self.target_size, self.input_size
        )

        # Mark for auto labeling: [points, rectangles]
        self.marks = []
        self.image_embed_cache = {}

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def get_sam_results(self, approx_contours, label=None):
        # Contours to shapes
        shapes = []
        if self.output_mode == "polygon":
            for approx in approx_contours:
                # Scale points
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue
                points.append(points[0])

                # Create shape
                shape = Shape(flags={})
                for point in points:
                    point[0] = int(point[0])
                    point[1] = int(point[1])
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shape.shape_type = "polygon"
                shape.closed = True
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.line_width = 1
                shape.label = "AUTOLABEL_OBJECT" if label is None else label
                shape.selected = False
                shapes.append(shape)
        elif self.output_mode == "rectangle":
            x_min = 100000000
            y_min = 100000000
            x_max = 0
            y_max = 0
            for approx in approx_contours:
                # Scale points
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue

                # Get min/max
                for point in points:
                    x_min = min(x_min, point[0])
                    y_min = min(y_min, point[1])
                    x_max = max(x_max, point[0])
                    y_max = max(y_max, point[1])

            # Create shape
            shape = Shape(flags={})
            shape.add_point(QtCore.QPointF(x_min, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_max))
            shape.shape_type = "rectangle"
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.line_width = 1
            shape.label = "AUTOLABEL_OBJECT" if label is None else label
            shape.selected = False
            shapes.append(shape)

        return shapes if label is None else shapes[0]

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict shapes from image
        """
        if image is None:
            return []

        try:
            cv_image = qt_img_to_rgb_cv_img(image, filename)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []
        if filename not in self.image_embed_cache:
            image_embedding = self.model.encode(cv_image)
            blob = self.preprocess(cv_image)
            predictions = self.net.get_ort_inference(blob)
            results = self.postprocess(predictions)[0]
            results[:, :4] = rescale_box(
                self.input_shape, results[:, :4], cv_image.shape
            ).round()
            shapes = []
            for *xyxy, _, cls_id in reversed(results):
                label = str(self.classes[int(cls_id)])
                x1, y1, x2, y2 = list(map(int, xyxy))
                box_prompt = [
                    np.array([[x1, y1], [x2, y2]]),
                    np.array([2, 3]),
                ]
                masks = self.model.predict_masks(image_embedding, box_prompt, transform_prompt=False)
                if len(masks.shape) == 4:
                    masks = masks[0][0]
                else:
                    masks = masks[0]
                approx_contours = self.model.get_approx_contours(masks)
                results = self.get_sam_results(approx_contours, label=label)
                shapes.append(results)
            result = AutoLabelingResult(shapes, replace=True)
            self.image_embed_cache[filename] = image_embedding
            return result
        else:
            masks = self.model.predict_masks(self.image_embed_cache[filename], self.marks)
            if len(masks.shape) == 4:
                masks = masks[0][0]
            else:
                masks = masks[0]
            approx_contours = self.model.get_approx_contours(masks)
            shapes = self.get_sam_results(approx_contours)
            result = AutoLabelingResult(shapes, replace=False)
            return result

    def unload(self):
        del self.net
        del self.encoder_session
        del self.decoder_session
