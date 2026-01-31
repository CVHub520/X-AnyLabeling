import os

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
from .__base__.grounding_dino import GroundingDINOBase


class Grounding_DINO(Model):
    """Open-Set object detection model using Grounding_DINO"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_type",
            "model_path",
            "input_width",
            "input_height",
            "conf_threshold",
            "text_threshold",
        ]
        widgets = [
            "edit_text",
            "button_send",
            "input_box_thres",
            "edit_conf",
            "toggle_preserve_existing_annotations",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        model_type = self.config["model_type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_type} model.",
                )
            )

        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.model_configs = GroundingDINOBase.get_configs(
            self.config["model_type"]
        )
        self.net.max_text_len = self.model_configs.max_text_len
        self.net.tokenizer = GroundingDINOBase.get_tokenlizer(
            self.model_configs.text_encoder_type
        )
        self.box_threshold = self.config["conf_threshold"]
        self.text_threshold = self.config["text_threshold"]
        self.target_size = (
            self.config["input_width"],
            self.config["input_height"],
        )
        self.replace = True

    def set_auto_labeling_conf(self, value):
        """set auto labeling box threshold"""
        if value > 0:
            self.box_threshold = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def preprocess(self, image, text_prompt):
        blob = GroundingDINOBase.preprocess_image(image, self.target_size)
        tokenized, text_self_attention_masks, position_ids, caption = (
            GroundingDINOBase.encode_text(
                text_prompt, self.net.tokenizer, self.net.max_text_len
            )
        )
        inputs = {
            "img": blob,
            "input_ids": np.array(tokenized["input_ids"], dtype=np.int64),
            "attention_mask": np.array(
                tokenized["attention_mask"], dtype=bool
            ),
            "token_type_ids": np.array(
                tokenized["token_type_ids"], dtype=np.int64
            ),
            "position_ids": np.array(position_ids, dtype=np.int64),
            "text_token_mask": np.array(text_self_attention_masks, dtype=bool),
        }
        return blob, inputs, caption

    def postprocess(
        self, outputs, caption, with_logits=True, token_spans=None
    ):
        if token_spans is not None:
            # TODO: Using token_spans.
            raise NotImplementedError
        logits, boxes = outputs
        boxes_filt, pred_phrases = GroundingDINOBase.decode_predictions(
            logits,
            boxes,
            caption,
            self.net.tokenizer,
            self.box_threshold,
            self.text_threshold,
            apply_sigmoid=True,
            with_logits=with_logits,
        )
        return boxes_filt, pred_phrases

    def predict_shapes(self, image, image_path=None, text_prompt=None):
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

        blob, inputs, caption = self.preprocess(image, text_prompt)
        outputs = self.net.get_ort_inference(
            blob, inputs=inputs, extract=False
        )
        boxes_filt, pred_phrases = self.postprocess(outputs, caption)

        shapes = []
        img_h, img_w, _ = image.shape
        boxes = GroundingDINOBase.rescale_boxes(boxes_filt, img_h, img_w)
        for box, label_info in zip(boxes, pred_phrases):
            x1, y1, x2, y2 = box
            label, score = label_info
            shape = Shape(
                label=str(label), score=float(score), shape_type="rectangle"
            )
            shape.add_point(QtCore.QPointF(x1, y1))
            shape.add_point(QtCore.QPointF(x2, y1))
            shape.add_point(QtCore.QPointF(x2, y2))
            shape.add_point(QtCore.QPointF(x1, y2))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=self.replace)
        return result

    def unload(self):
        del self.net
