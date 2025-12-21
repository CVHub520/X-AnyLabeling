import os
import cv2
import numpy as np

from typing import Dict
from tokenizers import Tokenizer

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .utils.general import Args
from .engines.build_onnx_engine import OnnxBaseModel


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
        self.model_configs = self.get_configs(self.config["model_type"])
        self.net.max_text_len = self.model_configs.max_text_len
        self.net.tokenizer = self.get_tokenlizer(
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
        prediction_logits_ = np.squeeze(
            logits, 0
        )  # [0]  # prediction_logits.shape = (nq, 256)
        logits_filt = self.sig(prediction_logits_)
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
        boxes = self.rescale_boxes(boxes_filt, img_h, img_w)
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
        return caption

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
        # special_tokens_mask: bs, num_token.
        # 1 for special tokens. 0 for normal tokens
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
