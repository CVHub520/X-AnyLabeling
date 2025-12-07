import os
import cv2
import traceback
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.utils import GenericWorker
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import (
    get_bounding_boxes,
    qt_img_to_rgb_cv_img,
)
from anylabeling.services.auto_labeling.utils import calculate_rotation_theta

from .lru_cache import LRUCache
from .model import Model
from .types import AutoLabelingResult
from .__base__.clip import ChineseClipONNX
from .__base__.sam2 import SegmentAnything2ONNX


class SegmentAnything2(Model):
    """Segmentation model using SegmentAnything2"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "encoder_model_path",
            "decoder_model_path",
        ]
        widgets = [
            "output_label",
            "output_select_combobox",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
            "button_auto_decode",
            "button_cropping",
            "mask_fineness_slider",
            "mask_fineness_value_label",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            "rotation": QCoreApplication.translate("Model", "Rotation"),
        }
        default_output_mode = "polygon"

    def __init__(self, config_path, on_message) -> None:
        # Run the parent class's init method
        super().__init__(config_path, on_message)

        # Get encoder and decoder model paths
        encoder_model_abs_path = self.get_model_abs_path(
            self.config, "encoder_model_path"
        )
        if not encoder_model_abs_path or not os.path.isfile(
            encoder_model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize encoder of Segment Anything 2.",
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
                    "Could not download or initialize decoder of Segment Anything 2.",
                )
            )

        # Load models
        self.model = SegmentAnything2ONNX(
            encoder_model_abs_path,
            decoder_model_abs_path,
            __preferred_device__,
        )

        # Mark for auto labeling
        # points, rectangles
        self.marks = []

        # Cache for image embedding
        self.cache_size = 10
        self.preloaded_size = self.cache_size - 3
        self.image_embedding_cache = LRUCache(self.cache_size)

        # Pre-inference worker
        self.pre_inference_thread = None
        self.pre_inference_worker = None
        self.stop_inference = False

        # CLIP models
        self.clip_net = None
        clip_txt_model_path = self.config.get("txt_model_path", "")
        clip_img_model_path = self.config.get("img_model_path", "")
        if clip_txt_model_path and clip_img_model_path:
            if self.config["model_type"] == "cn_clip":
                clip_txt_model_path = self.get_model_abs_path(
                    self.config, "txt_model_path"
                )
                _ = self.get_model_abs_path(self.config, "txt_extra_path")
                clip_img_model_path = self.get_model_abs_path(
                    self.config, "img_model_path"
                )
                _ = self.get_model_abs_path(self.config, "img_extra_path")
                model_arch = self.config["model_arch"]
                self.clip_net = ChineseClipONNX(
                    clip_txt_model_path,
                    clip_img_model_path,
                    model_arch,
                    device=__preferred_device__,
                )
            self.classes = self.config.get("classes", [])

        self.epsilon = self.config.get("epsilon", 0.001)
        self.padding_ratio = self.config.get("padding_ratio", 0.2)
        self.cropping_mode = False

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def set_mask_fineness(self, epsilon):
        """Set mask fineness epsilon value"""
        self.epsilon = epsilon

    def set_cropping_mode(self, enabled: bool):
        """Set cropping mode for small object detection"""
        self.cropping_mode = enabled

    def post_process(self, masks, image=None):
        """
        Post process masks with shape of [height, width]
        """
        masks[masks > 0.0] = 255
        masks[masks <= 0.0] = 0
        masks = masks.astype(np.uint8)
        contours, _ = cv2.findContours(
            masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Refine contours
        approx_contours = []
        for contour in contours:
            # Approximate contour using configurable epsilon
            epsilon = self.epsilon * cv2.arcLength(contour, True)
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
                shape.label = "AUTOLABEL_OBJECT"
                shape.selected = False
                shapes.append(shape)
        elif self.output_mode == "rectangle":
            x_min = 100000000
            y_min = 100000000
            x_max = 0
            y_max = 0
            for approx in approx_contours:
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue

                for point in points:
                    x_min = min(x_min, point[0])
                    y_min = min(y_min, point[1])
                    x_max = max(x_max, point[0])
                    y_max = max(y_max, point[1])

            shape = Shape(flags={})
            shape.add_point(QtCore.QPointF(x_min, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_max))
            shape.add_point(QtCore.QPointF(x_min, y_max))
            shape.shape_type = "rectangle"
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            if self.clip_net is not None and self.classes:
                img = image[y_min:y_max, x_min:x_max]
                out = self.clip_net(img, self.classes)
                shape.cache_label = self.classes[int(np.argmax(out))]
            shape.label = "AUTOLABEL_OBJECT"
            shape.selected = False
            shapes.append(shape)
        elif self.output_mode == "rotation":
            shape = Shape(flags={})
            rotation_box = get_bounding_boxes(approx_contours[0])[1]
            for point in rotation_box:
                shape.add_point(QtCore.QPointF(int(point[0]), int(point[1])))
            shape.direction = calculate_rotation_theta(rotation_box)
            shape.shape_type = self.output_mode
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.label = "AUTOLABEL_OBJECT"
            shape.selected = False
            shapes.append(shape)

        return shapes

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict shapes from image
        """
        if image is None or not self.marks:
            return AutoLabelingResult([], replace=False)

        shapes = []
        cv_image = qt_img_to_rgb_cv_img(image, filename)
        original_height, original_width = cv_image.shape[:2]

        try:
            crop_offset_x = 0
            crop_offset_y = 0
            cropped_image = cv_image
            cropped_marks = self.marks
            rectangle_mark = None

            if self.cropping_mode:
                for mark in self.marks:
                    if mark.get("type") == "rectangle":
                        rectangle_mark = mark
                        break

                if rectangle_mark:
                    x1, y1, x2, y2 = rectangle_mark["data"]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    box_width = x2 - x1
                    box_height = y2 - y1
                    padding_x = int(box_width * self.padding_ratio)
                    padding_y = int(box_height * self.padding_ratio)

                    crop_x1 = max(0, x1 - padding_x)
                    crop_y1 = max(0, y1 - padding_y)
                    crop_x2 = min(original_width, x2 + padding_x)
                    crop_y2 = min(original_height, y2 + padding_y)

                    crop_offset_x = crop_x1
                    crop_offset_y = crop_y1
                    cropped_image = cv_image[crop_y1:crop_y2, crop_x1:crop_x2]

                    cropped_marks = []
                    for mark in self.marks:
                        if mark.get("type") == "point":
                            px, py = mark["data"]
                            cropped_marks.append(
                                {
                                    "type": "point",
                                    "data": [
                                        px - crop_offset_x,
                                        py - crop_offset_y,
                                    ],
                                    "label": mark["label"],
                                }
                            )
                        elif mark.get("type") == "rectangle":
                            rx1, ry1, rx2, ry2 = mark["data"]
                            cropped_marks.append(
                                {
                                    "type": "rectangle",
                                    "data": [
                                        rx1 - crop_offset_x,
                                        ry1 - crop_offset_y,
                                        rx2 - crop_offset_x,
                                        ry2 - crop_offset_y,
                                    ],
                                    "label": mark["label"],
                                }
                            )

            cache_key = (
                f"{filename}_crop_{crop_offset_x}_{crop_offset_y}"
                if self.cropping_mode and rectangle_mark
                else filename
            )

            cached_data = self.image_embedding_cache.get(cache_key)
            if cached_data is not None:
                image_embedding = cached_data
            else:
                if self.stop_inference:
                    return AutoLabelingResult([], replace=False)
                image_embedding = self.model.encode(cropped_image)
                self.image_embedding_cache.put(cache_key, image_embedding)

            if self.stop_inference:
                return AutoLabelingResult([], replace=False)

            masks = self.model.predict_masks(image_embedding, cropped_marks)
            if len(masks.shape) == 4:
                masks = masks[0][0]
            else:
                masks = masks[0]

            if self.cropping_mode and rectangle_mark:
                full_mask = np.zeros(
                    (original_height, original_width), dtype=masks.dtype
                )
                mask_height, mask_width = masks.shape
                crop_height, crop_width = cropped_image.shape[:2]

                end_y = crop_offset_y + mask_height
                end_x = crop_offset_x + mask_width

                if end_y > original_height or end_x > original_width:
                    end_y = min(end_y, original_height)
                    end_x = min(end_x, original_width)
                    adjusted_mask_height = end_y - crop_offset_y
                    adjusted_mask_width = end_x - crop_offset_x
                    if adjusted_mask_height > 0 and adjusted_mask_width > 0:
                        masks = masks[
                            :adjusted_mask_height, :adjusted_mask_width
                        ]
                        mask_height = adjusted_mask_height
                        mask_width = adjusted_mask_width
                    else:
                        masks = full_mask
                        mask_height = 0
                        mask_width = 0

                if (
                    mask_height > 0
                    and mask_width > 0
                    and mask_height == crop_height
                    and mask_width == crop_width
                ):
                    full_mask[
                        crop_offset_y : crop_offset_y + mask_height,
                        crop_offset_x : crop_offset_x + mask_width,
                    ] = masks
                else:
                    target_width = min(
                        crop_width, original_width - crop_offset_x
                    )
                    target_height = min(
                        crop_height, original_height - crop_offset_y
                    )
                    resized_mask = cv2.resize(
                        masks,
                        (target_width, target_height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    full_mask[
                        crop_offset_y : crop_offset_y + target_height,
                        crop_offset_x : crop_offset_x + target_width,
                    ] = resized_mask
                masks = full_mask

            shapes = self.post_process(masks, cv_image)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

        result = AutoLabelingResult(shapes, replace=False)
        return result

    def unload(self):
        self.stop_inference = True
        if self.pre_inference_thread:
            self.pre_inference_thread.quit()

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
