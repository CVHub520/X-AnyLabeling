import os
import cv2
import time
import traceback
import numpy as np
from collections import Counter

from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import (
    get_bounding_boxes,
    qt_img_to_rgb_cv_img,
)
from anylabeling.services.auto_labeling.utils import calculate_rotation_theta

from .model import Model
from .lru_cache import LRUCache
from .types import AutoLabelingResult
from .__base__.sam3 import SegmentAnything3ONNX


class SegmentAnything3(Model):
    """Segmentation model using Segment Anything 3."""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "encoder_model_path",
            "encoder_model_data_path",
            "decoder_model_path",
            "decoder_model_data_path",
            "language_encoder_path",
            "language_encoder_data_path",
        ]
        widgets = [
            "edit_text",
            "button_send",
            "output_label",
            "output_select_combobox",
            # Visual prompt widgets are intentionally hidden for the current
            # SAM3 ONNX text-grounding UI. Keep them here as the integration
            # points if ONNX/PyTorch visual prompting is enabled later.
            # "button_add_point",
            # "button_remove_point",
            # "button_add_rect",
            # "button_clear",
            # "button_finish_object",
            "edit_conf",
            "toggle_preserve_existing_annotations",
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
        super().__init__(config_path, on_message)

        self.get_model_abs_path(self.config, "encoder_model_data_path")
        self.get_model_abs_path(self.config, "decoder_model_data_path")
        self.get_model_abs_path(self.config, "language_encoder_data_path")

        encoder_model_abs_path = self.get_model_abs_path(
            self.config, "encoder_model_path"
        )
        if not encoder_model_abs_path or not os.path.isfile(
            encoder_model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize encoder of "
                    "Segment Anything 3.",
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
                    "Could not download or initialize decoder of "
                    "Segment Anything 3.",
                )
            )

        language_encoder_abs_path = self.get_model_abs_path(
            self.config, "language_encoder_path"
        )
        if not language_encoder_abs_path or not os.path.isfile(
            language_encoder_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize language encoder of "
                    "Segment Anything 3.",
                )
            )

        self.model = SegmentAnything3ONNX(
            encoder_model_abs_path,
            decoder_model_abs_path,
            language_encoder_abs_path,
            __preferred_device__,
        )
        self.marks = []
        self.epsilon = self.config.get("epsilon", 0.001)
        self.conf_thres = self.config.get("conf_threshold", 0.5)
        self.replace = True
        self.cache_size = 10
        self.image_embedding_cache = LRUCache(self.cache_size)

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks."""
        self.marks = marks

    def set_auto_labeling_conf(self, value):
        """Set auto labeling confidence threshold."""
        self.conf_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations."""
        self.replace = not state

    def set_mask_fineness(self, epsilon):
        """Set mask fineness epsilon value."""
        self.epsilon = epsilon

    @staticmethod
    def split_text_prompts(text_prompt):
        separators = [",", "."]
        separator_used = None
        for separator in separators:
            if separator in text_prompt:
                separator_used = separator
                break

        if separator_used:
            prompts = [
                prompt.strip()
                for prompt in text_prompt.split(separator_used)
                if prompt.strip()
            ]
        else:
            prompts = [text_prompt.strip()] if text_prompt.strip() else []

        return list(dict.fromkeys(prompts))

    def post_process(self, masks, label, scores=None):
        masks[masks > 0.0] = 255
        masks[masks <= 0.0] = 0
        masks = masks.astype(np.uint8)

        shapes = []
        for index, mask in enumerate(masks):
            if mask.ndim == 3:
                mask = mask[0]
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            epsilon = self.epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue

            shape = Shape(flags={})
            if self.output_mode == "polygon":
                points = approx.reshape(-1, 2).tolist()
                points.append(points[0])
                for point in points:
                    shape.add_point(
                        QtCore.QPointF(int(point[0]), int(point[1]))
                    )
                shape.shape_type = "polygon"
            elif self.output_mode == "rectangle":
                x, y, w, h = cv2.boundingRect(approx)
                shape.add_point(QtCore.QPointF(x, y))
                shape.add_point(QtCore.QPointF(x + w, y))
                shape.add_point(QtCore.QPointF(x + w, y + h))
                shape.add_point(QtCore.QPointF(x, y + h))
                shape.shape_type = "rectangle"
            elif self.output_mode == "rotation":
                rotation_box = get_bounding_boxes(approx)[1]
                for point in rotation_box:
                    shape.add_point(
                        QtCore.QPointF(int(point[0]), int(point[1]))
                    )
                shape.direction = calculate_rotation_theta(rotation_box)
                shape.shape_type = "rotation"
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.label = label or "AUTOLABEL_OBJECT"
            if scores is not None and index < len(scores):
                shape.score = float(scores[index])
            shape.selected = False
            shapes.append(shape)

        return shapes

    def predict_shapes(
        self, image, filename=None, text_prompt=None
    ) -> AutoLabelingResult:
        """
        Predict shapes from image.
        """
        if image is None:
            return AutoLabelingResult([], replace=False)

        text_prompts = self.split_text_prompts((text_prompt or "").strip())
        if not text_prompts:
            self.on_message(
                QCoreApplication.translate(
                    "Model",
                    "SAM3 requires a text prompt.",
                )
            )
            return AutoLabelingResult([], replace=False)

        try:
            total_start = time.perf_counter()
            preprocess_start = time.perf_counter()
            cv_image = qt_img_to_rgb_cv_img(image, filename)
            preprocess_time = time.perf_counter() - preprocess_start

            image_encoder_start = time.perf_counter()
            cache_key = filename
            cache_hit = (
                cache_key is not None
                and self.image_embedding_cache.find(cache_key)
            )
            if cache_hit:
                image_embedding = self.image_embedding_cache.get(cache_key)
                image_encoder_time = 0.0
            else:
                image_embedding = self.model.encode_image(cv_image)
                image_encoder_time = time.perf_counter() - image_encoder_start
                if cache_key is not None:
                    self.image_embedding_cache.put(cache_key, image_embedding)

            shapes = []
            language_encoder_time = 0.0
            decoder_time = 0.0
            postprocess_time = 0.0
            for prompt in text_prompts:
                language_encoder_start = time.perf_counter()
                prompt_embedding = self.model.apply_text_prompt(
                    image_embedding, prompt
                )
                language_encoder_time += (
                    time.perf_counter() - language_encoder_start
                )

                decoder_start = time.perf_counter()
                masks, scores = self.model.predict_masks(
                    prompt_embedding, self.marks, self.conf_thres
                )
                decoder_time += time.perf_counter() - decoder_start

                postprocess_start = time.perf_counter()
                shapes.extend(self.post_process(masks, prompt, scores))
                postprocess_time += time.perf_counter() - postprocess_start

            output_counts = Counter(shape.label for shape in shapes)
            output_summary = ",".join(
                f"{label}:{count}"
                for label, count in sorted(output_counts.items())
            )
            if not output_summary:
                output_summary = "none"
            total_time = time.perf_counter() - total_start
            logger.info(
                "SAM3 | "
                f"image={os.path.abspath(filename) if filename else ''} | "
                f"prompt={','.join(text_prompts)} | "
                f"output_mode={self.output_mode} | "
                f"conf={self.conf_thres:.4f} | "
                f"preprocess={preprocess_time * 1000:.1f}ms | "
                f"image_encoder={image_encoder_time * 1000:.1f}ms | "
                f"cache={'hit' if cache_hit else 'miss'} | "
                f"language_encoder={language_encoder_time * 1000:.1f}ms | "
                f"decoder={decoder_time * 1000:.1f}ms | "
                f"postprocess={postprocess_time * 1000:.1f}ms | "
                f"outputs={output_summary} | "
                f"total={total_time * 1000:.1f}ms"
            )
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        self.image_embedding_cache = LRUCache(self.cache_size)
        if hasattr(self, "model"):
            del self.model
