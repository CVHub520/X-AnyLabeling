import os
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import (
    get_bounding_boxes,
    qt_img_to_rgb_cv_img,
)


from .types import AutoLabelingResult
from .__base__.yolo import YOLO
from .__base__.sam import SegmentAnythingONNX
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
        ]
        widgets = [
            "button_run",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
            "button_auto_decode",
            "output_label",
            "output_select_combobox",
            "mask_fineness_slider",
            "mask_fineness_value_label",
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
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize YOLOv5 model."
                )
            )
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        _, _, self.input_height, self.input_width = self.net.get_input_shape()
        if not isinstance(self.input_width, int):
            self.input_width = self.config.get("input_width", -1)
        if not isinstance(self.input_height, int):
            self.input_height = self.config.get("input_height", -1)

        self.task = "det"
        self.model_type = self.config["type"]
        self.classes = self.config.get("classes", [])
        self.anchors = self.config.get("anchors", None)
        self.agnostic = self.config.get("agnostic", False)
        self.show_boxes = self.config.get("show_boxes", False)
        self.strategy = self.config.get("strategy", "largest")
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
        max_height = self.config["max_height"]
        max_width = self.config["max_width"]
        self.target_size = self.config["target_size"]
        self.input_size = (max_height, max_width)
        self.encoder_session = OnnxBaseModel(
            encoder_model_abs_path, __preferred_device__
        )
        self.decoder_session = OnnxBaseModel(
            decoder_model_abs_path, __preferred_device__
        )
        self.model = SegmentAnythingONNX(
            self.encoder_session,
            self.decoder_session,
            self.target_size,
            self.input_size,
        )

        # Mark for auto labeling: [points, rectangles]
        self.marks = []
        self.image_embed_cache = {}

        self.epsilon = 0.001

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def set_mask_fineness(self, epsilon):
        """Set mask fineness epsilon value"""
        self.epsilon = epsilon

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
                shape.label = "AUTOLABEL_OBJECT" if label is None else label
                shape.selected = False
                shapes.append(shape)
        elif self.output_mode == "rectangle":
            shape = Shape(flags={})
            rectangle_box, _ = get_bounding_boxes(approx_contours[0])
            xmin, ymin, xmax, ymax = rectangle_box
            shape.add_point(QtCore.QPointF(int(xmin), int(ymin)))
            shape.add_point(QtCore.QPointF(int(xmax), int(ymin)))
            shape.add_point(QtCore.QPointF(int(xmax), int(ymax)))
            shape.add_point(QtCore.QPointF(int(xmin), int(ymax)))
            shape.shape_type = self.output_mode
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
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
            logger.warning("Could not inference model")
            logger.warning(e)
            return []
        if filename not in self.image_embed_cache:
            image_embedding = self.model.encode(cv_image)
            blob = self.preprocess(cv_image, upsample_mode="letterbox")
            outputs = self.net.get_ort_inference(blob=blob, extract=False)
            boxes, class_ids, _, _, _ = self.postprocess(outputs)

            shapes = []
            for box, class_id in zip(boxes, class_ids):
                label = str(self.classes[int(class_id)])
                x1, y1, x2, y2 = list(map(int, box))
                box_prompt = [
                    np.array([[x1, y1], [x2, y2]]),
                    np.array([2, 3]),
                ]
                masks = self.model.predict_masks(
                    image_embedding, box_prompt, transform_prompt=False
                )
                if len(masks.shape) == 4:
                    masks = masks[0][0]
                else:
                    masks = masks[0]
                approx_contours = self.model.get_approx_contours(
                    masks, self.epsilon
                )
                results = self.get_sam_results(approx_contours, label=label)
                shapes.append(results)
            result = AutoLabelingResult(shapes, replace=True)
            self.image_embed_cache[filename] = image_embedding
            return result
        else:
            masks = self.model.predict_masks(
                self.image_embed_cache[filename], self.marks
            )
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
