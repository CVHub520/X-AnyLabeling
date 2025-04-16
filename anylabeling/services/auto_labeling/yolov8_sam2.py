import os
import cv2
import traceback
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QCoreApplication

from anylabeling.utils import GenericWorker
from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import (
    get_bounding_boxes,
    qt_img_to_rgb_cv_img,
)


from .engines.build_onnx_engine import OnnxBaseModel
from .lru_cache import LRUCache
from .types import AutoLabelingResult
from .__base__.sam2 import SegmentAnything2ONNX
from .__base__.yolo import YOLO


class YOLOv8SegmentAnything2(YOLO):
    """Segmentation model using YOLOv8 by SAM2"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "encoder_model_path",
            "decoder_model_path",
        ]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "input_iou",
            "edit_iou",
            "toggle_preserve_existing_annotations",
            "output_label",
            "output_select_combobox",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            "rotation": QCoreApplication.translate("Model", "Rotation"),
        }
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        # ----------- YOLOv8 ---------- #
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {self.config['type']} model.",
                )
            )
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        _, _, self.input_height, self.input_width = self.net.get_input_shape()
        if not isinstance(self.input_width, int):
            self.input_width = self.config.get("input_width", -1)
        if not isinstance(self.input_height, int):
            self.input_height = self.config.get("input_height", -1)

        self.replace = True
        self.model_type = self.config["type"]
        self.classes = self.config["classes"]
        self.anchors = self.config.get("anchors", None)
        self.agnostic = self.config.get("agnostic", False)
        self.show_boxes = self.config.get("show_boxes", False)
        self.strategy = self.config.get("strategy", "largest")
        self.iou_thres = self.config.get("iou_threshold", 0.45)
        self.conf_thres = self.config.get("conf_threshold", 0.25)
        self.filter_classes = self.config.get("filter_classes", None)

        self.task = "det"
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

        # ----------- Segment-Anything-2 ---------- #
        encoder_model_abs_path = self.get_model_abs_path(
            self.config, "encoder_model_path"
        )
        if not encoder_model_abs_path or not os.path.isfile(
            encoder_model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize encoder of SAM2.",
                )
            )
        decoder_model_abs_path = self.get_model_abs_path(
            self.config, "decoder_model_path"
        )
        if not decoder_model_abs_path or not os.path.isfile(
            decoder_model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize decoder of SAM2.",
                )
            )

        # Load models
        self.model = SegmentAnything2ONNX(
            encoder_model_abs_path,
            decoder_model_abs_path,
            __preferred_device__,
        )

        # Mark for auto labeling
        self.marks = []  # points, rectangles

        # Cache for image embedding
        self.cache_size = 10
        self.preloaded_size = self.cache_size - 3
        self.image_embedding_cache = LRUCache(self.cache_size)
        self.current_image_embedding_cache = {}

        # Pre-inference worker
        self.pre_inference_thread = None
        self.pre_inference_worker = None
        self.stop_inference = False

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def post_process(self, masks, label=None):
        """
        Post process masks
        """
        # Find contours
        masks[masks > 0.0] = 255
        masks[masks <= 0.0] = 0
        masks = masks.astype(np.uint8)
        contours, _ = cv2.findContours(
            masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Refine contours
        approx_contours = []
        for contour in contours:
            # Approximate contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_contours.append(approx)

        # Remove too big contours ( >90% of image size)
        if len(approx_contours) > 1:
            image_size = masks.shape[0] * masks.shape[1]
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area < image_size * 0.9
            ]

        # Remove small contours (area < 20% of average area)
        if len(approx_contours) > 1:
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            avg_area = np.mean(areas)

            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area > avg_area * 0.2
            ]
            approx_contours = filtered_approx_contours

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
        elif self.output_mode in ["rectangle", "rotation"]:
            shape = Shape(flags={})
            rectangle_box, rotation_box = get_bounding_boxes(
                approx_contours[0]
            )
            xmin, ymin, xmax, ymax = rectangle_box
            if self.output_mode == "rectangle":
                shape.add_point(QtCore.QPointF(int(xmin), int(ymin)))
                shape.add_point(QtCore.QPointF(int(xmax), int(ymin)))
                shape.add_point(QtCore.QPointF(int(xmax), int(ymax)))
                shape.add_point(QtCore.QPointF(int(xmin), int(ymax)))
            else:
                for point in rotation_box:
                    shape.add_point(
                        QtCore.QPointF(int(point[0]), int(point[1]))
                    )
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

        try:
            # Use cached image embedding if possible
            cached_data = self.image_embedding_cache.get(filename)
            if cached_data is not None:
                image_embedding = cached_data
            else:
                if self.stop_inference:
                    return AutoLabelingResult([], replace=False)
                image_embedding = self.model.encode(cv_image)
                self.image_embedding_cache.put(
                    filename,
                    image_embedding,
                )
            if self.stop_inference:
                return AutoLabelingResult([], replace=False)

            blob = self.preprocess(cv_image, upsample_mode="letterbox")
            outputs = self.net.get_ort_inference(blob=blob, extract=False)
            boxes, class_ids, _, _, _ = self.postprocess(outputs)

            shapes = []
            for box, class_id in zip(boxes, class_ids):
                label = str(self.classes[int(class_id)])
                marks = [
                    {
                        "data": list(map(int, box)),
                        "label": 1,
                        "type": "rectangle",
                    }
                ]
                masks = self.model.predict_masks(image_embedding, marks)
                if len(masks.shape) == 4:
                    masks = masks[0][0]
                else:
                    masks = masks[0]
                shape = self.post_process(masks, label=label)
                shapes.append(shape)
            result = AutoLabelingResult(shapes, replace=self.replace)
            return result
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

    def unload(self):
        del self.net
        self.stop_inference = True
        if self.pre_inference_thread:
            self.pre_inference_thread.quit

    def preload_worker(self, files):
        """
        Preload next files, run inference and cache results
        """
        files = files[: self.preloaded_size]
        for filename in files:
            if self.image_embedding_cache.find(filename):
                continue
            image = self.load_image_from_filename(filename)
            if image is None:
                continue
            if self.stop_inference:
                return
            cv_image = qt_img_to_rgb_cv_img(image)
            image_embedding = self.model.encode(cv_image)
            self.image_embedding_cache.put(
                filename,
                image_embedding,
            )

    def on_next_files_changed(self, next_files):
        """
        Handle next files changed. This function can preload next files
        and run inference to save time for user.
        """
        if (
            self.pre_inference_thread is None
            or not self.pre_inference_thread.isRunning()
        ):
            self.pre_inference_thread = QThread()
            self.pre_inference_worker = GenericWorker(
                self.preload_worker, next_files
            )
            self.pre_inference_worker.finished.connect(
                self.pre_inference_thread.quit
            )
            self.pre_inference_worker.moveToThread(self.pre_inference_thread)
            self.pre_inference_thread.started.connect(
                self.pre_inference_worker.run
            )
            self.pre_inference_thread.start()
