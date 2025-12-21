import os
import cv2
import traceback
import numpy as np

from typing import Dict
from tokenizers import Tokenizer

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
from anylabeling.services.auto_labeling.utils import calculate_rotation_theta

from .model import Model
from .types import AutoLabelingResult
from .lru_cache import LRUCache
from .utils.general import Args
from .engines.build_onnx_engine import OnnxBaseModel
from .__base__.sam2 import SegmentAnything2ONNX


class GroundingSAM2(Model):
    """Open-Set object detection and instance segmentation model using GroundingSAM2"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_type",
            "model_path",
            "box_threshold",
            "text_threshold",
            "encoder_model_path",
            "decoder_model_path",
        ]
        widgets = [
            "edit_text",
            "button_send",
            "output_label",
            "output_select_combobox",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
            "button_auto_decode",
            "button_cropping",
            "input_conf",
            "edit_conf",
            "mask_fineness_slider",
            "mask_fineness_value_label",
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

        # ----------- Grounding-DINO ---------- #
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
        self.model_configs = self.get_configs(self.config["model_type"])
        self.net.max_text_len = self.model_configs.max_text_len
        self.net.tokenizer = self.get_tokenlizer(
            self.model_configs.text_encoder_type
        )
        self.box_threshold = self.config["box_threshold"]
        self.text_threshold = self.config["text_threshold"]
        self.target_size = (
            self.config["input_width"],
            self.config["input_height"],
        )

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
        # points, rectangles
        self.marks = []

        # Cache for image embedding
        self.cache_size = 10
        self.preloaded_size = self.cache_size - 3
        self.image_embedding_cache = LRUCache(self.cache_size)
        self.current_image_embedding_cache = {}

        # Pre-inference worker
        self.pre_inference_thread = None
        self.pre_inference_worker = None
        self.stop_inference = False

        self.epsilon = self.config.get("epsilon", 0.001)
        self.padding_ratio = self.config.get("padding_ratio", 0.2)
        self.cropping_mode = False

    def set_auto_labeling_conf(self, value):
        """set auto labeling confidence threshold"""
        if value > 0:
            self.box_threshold = value

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def set_mask_fineness(self, epsilon):
        """Set mask fineness epsilon value"""
        self.epsilon = epsilon

    def set_cropping_mode(self, enabled: bool):
        """Set cropping mode for small object detection"""
        self.cropping_mode = enabled

    def preprocess(self, image, text_prompt):
        # Resize the image
        image = cv2.resize(
            image, self.target_size, interpolation=cv2.INTER_LINEAR
        )

        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0).astype(np.float32)

        # encoder texts
        captions = self.get_caption(str(text_prompt))
        tokenized_raw_results = self.net.tokenizer.encode(captions)
        tokenized = {
            "input_ids": np.array([tokenized_raw_results.ids], dtype=np.int64),
            "token_type_ids": np.array(
                [tokenized_raw_results.type_ids], dtype=np.int64
            ),
            "attention_mask": np.array([tokenized_raw_results.attention_mask]),
        }
        specical_tokens = [101, 102, 1012, 1029]
        (
            text_self_attention_masks,
            position_ids,
            _,
        ) = self.generate_masks_with_special_tokens_and_transfer_map(
            tokenized, specical_tokens
        )
        if text_self_attention_masks.shape[1] > self.net.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.net.max_text_len, : self.net.max_text_len
            ]

            position_ids = position_ids[:, : self.net.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][
                :, : self.net.max_text_len
            ]
            tokenized["attention_mask"] = tokenized["attention_mask"][
                :, : self.net.max_text_len
            ]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][
                :, : self.net.max_text_len
            ]
        inputs = {}
        inputs["img"] = image
        inputs["input_ids"] = np.array(tokenized["input_ids"], dtype=np.int64)
        inputs["attention_mask"] = np.array(
            tokenized["attention_mask"], dtype=bool
        )
        inputs["token_type_ids"] = np.array(
            tokenized["token_type_ids"], dtype=np.int64
        )
        inputs["position_ids"] = np.array(position_ids, dtype=np.int64)
        inputs["text_token_mask"] = np.array(
            text_self_attention_masks, dtype=bool
        )
        return image, inputs, captions

    def postprocess(
        self, outputs, caption, with_logits=True, token_spans=None
    ):
        logits, boxes = outputs
        logits_filt = self.sig(
            np.squeeze(logits, 0)
        )  # [0]  # prediction_logits.shape = (nq, 256)
        boxes_filt = np.squeeze(
            boxes, 0
        )  # [0]  # prediction_boxes.shape = (nq, 4)
        # filter output
        if token_spans is None:
            filt_mask = logits_filt.max(axis=1) > self.box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

            # get phrase
            tokenlizer = self.net.tokenizer
            tokenized_raw_results = tokenlizer.encode(caption)
            tokenized = {
                "input_ids": np.array(
                    tokenized_raw_results.ids, dtype=np.int64
                ),
                "token_type_ids": np.array(
                    tokenized_raw_results.type_ids, dtype=np.int64
                ),
                "attention_mask": np.array(
                    tokenized_raw_results.attention_mask
                ),
            }
            # build pred
            pred_phrases = []
            for logit in logits_filt:
                posmap = logit > self.text_threshold
                pred_phrase = self.get_phrases_from_posmap(
                    posmap, tokenized, tokenlizer
                )
                if with_logits:
                    pred_phrases.append([pred_phrase, logit.max()])
                else:
                    pred_phrases.append([pred_phrase, 1.0])
        else:
            # TODO: Using token_spans.
            raise NotImplementedError
        return boxes_filt, pred_phrases

    def post_process(self, masks, label=None, score=None):
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
            approx_contours = filtered_approx_contours

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
                shape = Shape(
                    label="AUTOLABEL_OBJECT" if label is None else label,
                    score=float(score) if score is not None else None,
                    shape_type="polygon",
                    flags={},
                )
                for point in points:
                    point[0] = int(point[0])
                    point[1] = int(point[1])
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shape.closed = True
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.selected = False
                shapes.append(shape)
        elif self.output_mode in ["rectangle", "rotation"]:
            shape = Shape(
                label="AUTOLABEL_OBJECT" if label is None else label,
                score=float(score) if score is not None else None,
                shape_type=self.output_mode,
                flags={},
            )
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
                shape.direction = calculate_rotation_theta(rotation_box)
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.selected = False
            shapes.append(shape)

        return shapes if label is None else shapes[0]

    def predict_shapes(self, image, image_path=None, text_prompt=None):
        """
        Predict shapes from image
        """
        if image is None:
            return []

        try:
            cv_image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        try:
            original_height, original_width = cv_image.shape[:2]

            cached_data = self.image_embedding_cache.get(image_path)
            if cached_data is not None:
                image_embedding = cached_data
            else:
                if self.stop_inference:
                    return AutoLabelingResult([], replace=False)
                image_embedding = self.model.encode(cv_image)
                self.image_embedding_cache.put(
                    image_path,
                    image_embedding,
                )
                if self.stop_inference:
                    return AutoLabelingResult([], replace=False)

            if text_prompt:
                blob, inputs, caption = self.preprocess(cv_image, text_prompt)
                outputs = self.net.get_ort_inference(
                    blob, inputs=inputs, extract=False
                )
                boxes_filt, pred_phrases = self.postprocess(outputs, caption)
                img_h, img_w, _ = cv_image.shape
                boxes = self.rescale_boxes(boxes_filt, img_h, img_w)
                shapes = []
                for box, label_info in zip(boxes, pred_phrases):
                    label, score = label_info
                    marks = [
                        {
                            "data": box,
                            "label": 1,
                            "type": "rectangle",
                        }
                    ]
                    masks = self.model.predict_masks(image_embedding, marks)
                    if len(masks.shape) == 4:
                        masks = masks[0][0]
                    else:
                        masks = masks[0]
                    shape = self.post_process(masks, label=label, score=score)
                    shapes.append(shape)
                result = AutoLabelingResult(shapes, replace=False)
            else:
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
                        cropped_image = cv_image[
                            crop_y1:crop_y2, crop_x1:crop_x2
                        ]

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
                    f"{image_path}_crop_{crop_offset_x}_{crop_offset_y}"
                    if self.cropping_mode and rectangle_mark
                    else image_path
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

                masks = self.model.predict_masks(
                    image_embedding, cropped_marks
                )
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
                        if (
                            adjusted_mask_height > 0
                            and adjusted_mask_width > 0
                        ):
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

                shapes = self.post_process(masks)
                result = AutoLabelingResult(shapes, replace=False)
            return result
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

    @staticmethod
    def sig(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def rescale_boxes(boxes, img_h, img_w):
        converted_boxes = []
        for box in boxes:
            # from 0..1 to 0..W, 0..H
            converted_box = box * np.array([img_w, img_h, img_w, img_h])
            # from xywh to xyxy
            converted_box[:2] -= converted_box[2:] / 2
            converted_box[2:] += converted_box[:2]
            converted_boxes.append(converted_box)
        return np.array(converted_boxes, dtype=int)

    @staticmethod
    def get_configs(model_type):
        if model_type == "groundingdino_swinb_cogcoor":
            configs = Args(
                batch_size=1,
                modelname="groundingdino",
                backbone="swin_B_384_22k",
                position_embedding="sine",
                pe_temperatureH=20,
                pe_temperatureW=20,
                return_interm_indices=[1, 2, 3],
                backbone_freeze_keywords=None,
                enc_layers=6,
                dec_layers=6,
                pre_norm=False,
                dim_feedforward=2048,
                hidden_dim=256,
                dropout=0.0,
                nheads=8,
                num_queries=900,
                query_dim=4,
                num_patterns=0,
                num_feature_levels=4,
                enc_n_points=4,
                dec_n_points=4,
                two_stage_type="standard",
                two_stage_bbox_embed_share=False,
                two_stage_class_embed_share=False,
                transformer_activation="relu",
                dec_pred_bbox_embed_share=True,
                dn_box_noise_scale=1.0,
                dn_label_noise_ratio=0.5,
                dn_label_coef=1.0,
                dn_bbox_coef=1.0,
                embed_init_tgt=True,
                dn_labelbook_size=2000,
                max_text_len=256,
                text_encoder_type="bert-base-uncased",
                use_text_enhancer=True,
                use_fusion_layer=True,
                use_checkpoint=True,
                use_transformer_ckpt=True,
                use_text_cross_attention=True,
                text_dropout=0.0,
                fusion_dropout=0.0,
                fusion_droppath=0.1,
                sub_sentence_present=True,
            )
        elif model_type == "groundingdino_swint_ogc":
            configs = Args(
                batch_size=1,
                modelname="groundingdino",
                backbone="swin_T_224_1k",
                position_embedding="sine",
                pe_temperatureH=20,
                pe_temperatureW=20,
                return_interm_indices=[1, 2, 3],
                backbone_freeze_keywords=None,
                enc_layers=6,
                dec_layers=6,
                pre_norm=False,
                dim_feedforward=2048,
                hidden_dim=256,
                dropout=0.0,
                nheads=8,
                num_queries=900,
                query_dim=4,
                num_patterns=0,
                num_feature_levels=4,
                enc_n_points=4,
                dec_n_points=4,
                two_stage_type="standard",
                two_stage_bbox_embed_share=False,
                two_stage_class_embed_share=False,
                transformer_activation="relu",
                dec_pred_bbox_embed_share=True,
                dn_box_noise_scale=1.0,
                dn_label_noise_ratio=0.5,
                dn_label_coef=1.0,
                dn_bbox_coef=1.0,
                embed_init_tgt=True,
                dn_labelbook_size=2000,
                max_text_len=256,
                text_encoder_type="bert-base-uncased",
                use_text_enhancer=True,
                use_fusion_layer=True,
                use_checkpoint=True,
                use_transformer_ckpt=True,
                use_text_cross_attention=True,
                text_dropout=0.0,
                fusion_dropout=0.0,
                fusion_droppath=0.1,
                sub_sentence_present=True,
            )
        else:
            raise ValueError(
                QCoreApplication.translate(
                    "Model", "Invalid model_type in GroundingDINO model."
                )
            )
        return configs

    @staticmethod
    def get_caption(text_prompt):
        caption = text_prompt.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        captions = caption
        return captions

    @staticmethod
    def get_tokenlizer(text_encoder_type):
        import os
        from importlib.resources import files
        from anylabeling.services.auto_labeling import configs

        cfg_name = text_encoder_type.replace("-", "_") + "_tokenizer.json"

        try:
            tokenizer_resource = files(configs.bert).joinpath(cfg_name)
            tokenizer_json = tokenizer_resource.read_text(encoding="utf-8")
            tokenizer = Tokenizer.from_str(tokenizer_json)
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")

        try:
            bert_dir = os.path.dirname(configs.bert.__file__)
            tokenizer_path = os.path.join(bert_dir, cfg_name)
            if os.path.exists(tokenizer_path):
                tokenizer = Tokenizer.from_file(tokenizer_path)
                return tokenizer
            else:
                logger.error(f"Tokenizer file not found: {tokenizer_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading tokenizer from fallback path: {e}")
            return None

    @staticmethod
    def get_phrases_from_posmap(
        posmap: np.ndarray,
        tokenized: Dict,
        tokenizer,
        left_idx: int = 0,
        right_idx: int = 255,
    ):
        assert isinstance(posmap, np.ndarray), "posmap must be numpy.ndarray"
        if posmap.ndim == 1:
            posmap[0 : left_idx + 1] = False
            posmap[right_idx:] = False
            non_zero_idx = np.where(posmap)[0]
            token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
            return tokenizer.decode(token_ids)
        else:
            raise NotImplementedError("posmap must be 1-dim")

    @staticmethod
    def generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens_list
    ):
        input_ids = tokenized["input_ids"]
        bs, num_token = input_ids.shape
        # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
        special_tokens_mask = np.zeros((bs, num_token), dtype=bool)
        for special_token in special_tokens_list:
            special_tokens_mask |= input_ids == special_token

        # idxs: each row is a list of indices of special tokens
        idxs = np.argwhere(special_tokens_mask)

        # generate attention mask and positional ids
        attention_mask = np.eye(num_token, dtype=bool).reshape(
            1, num_token, num_token
        )
        attention_mask = np.tile(attention_mask, (bs, 1, 1))
        position_ids = np.zeros((bs, num_token), dtype=int)
        cate_to_token_mask_list = [[] for _ in range(bs)]
        previous_col = 0
        for i in range(idxs.shape[0]):
            row, col = idxs[i]
            if (col == 0) or (col == num_token - 1):
                attention_mask[row, col, col] = True
                position_ids[row, col] = 0
            else:
                attention_mask[
                    row, previous_col + 1 : col + 1, previous_col + 1 : col + 1
                ] = True
                position_ids[row, previous_col + 1 : col + 1] = np.arange(
                    0, col - previous_col
                )
                c2t_maski = np.zeros((num_token), dtype=bool)
                c2t_maski[previous_col + 1 : col] = True
                cate_to_token_mask_list[row].append(c2t_maski)
            previous_col = col

        cate_to_token_mask_list = [
            np.stack(cate_to_token_mask_listi, axis=0)
            for cate_to_token_mask_listi in cate_to_token_mask_list
        ]

        return attention_mask, position_ids, cate_to_token_mask_list

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
