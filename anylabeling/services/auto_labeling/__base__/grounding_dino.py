import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tokenizers import Tokenizer

from anylabeling.views.labeling.logger import logger
from ..utils.general import Args


class GroundingDINOBase:
    """Base class for GroundingDINO-based models with common utilities."""

    SPECIAL_TOKENS = [101, 102, 1012, 1029]
    IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGE_STD = np.array([0.229, 0.224, 0.225])

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array with sigmoid applied.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def rescale_boxes(boxes: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
        """
        Rescale normalized boxes to image dimensions.

        Args:
            boxes (np.ndarray): Normalized boxes in xywh format.
            img_h (int): Image height.
            img_w (int): Image width.

        Returns:
            np.ndarray: Rescaled boxes in xyxy format.
        """
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
    def get_caption(text_prompt: str) -> str:
        """
        Format text prompt as caption.

        Args:
            text_prompt (str): Raw text prompt.

        Returns:
            str: Formatted caption ending with period.
        """
        caption = text_prompt.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        return caption

    @staticmethod
    def get_tokenlizer(text_encoder_type: str) -> Tokenizer:
        """
        Load tokenizer for text encoder.

        Args:
            text_encoder_type (str): Type of text encoder (e.g., 'bert-base-uncased').

        Returns:
            Tokenizer: Loaded tokenizer or None if failed.
        """
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
    def get_phrase_token_ranges(
        input_ids: np.ndarray, special_token_ids: set
    ) -> List[Tuple[int, int]]:
        """
        Get token ranges for each phrase.

        Args:
            input_ids (np.ndarray): Token IDs.
            special_token_ids (set): Set of special token IDs.

        Returns:
            List[Tuple[int, int]]: List of (start, end) ranges.
        """
        ranges = []
        start_idx = None
        for i, token_id in enumerate(input_ids):
            if token_id in special_token_ids:
                if start_idx is not None:
                    ranges.append((start_idx, i))
                    start_idx = None
            else:
                if start_idx is None:
                    start_idx = i
        return ranges

    @staticmethod
    def get_phrases_from_posmap(
        posmap: np.ndarray,
        tokenized: Dict,
        tokenizer: Tokenizer,
        left_idx: int = 0,
        right_idx: int = 255,
    ) -> str:
        """
        Extract phrases from position map.

        Args:
            posmap (np.ndarray): Boolean position map.
            tokenized (Dict): Tokenized text with input_ids.
            tokenizer (Tokenizer): Tokenizer instance.
            left_idx (int): Left boundary index.
            right_idx (int): Right boundary index.

        Returns:
            str: Decoded phrase string.
        """
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
        tokenized: Dict, special_tokens_list: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Generate attention masks and position IDs with special tokens.

        Args:
            tokenized (Dict): Tokenized text with input_ids.
            special_tokens_list (List[int]): List of special token IDs.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
                - attention_mask: Shape (bs, num_token, num_token).
                - position_ids: Shape (bs, num_token).
                - cate_to_token_mask_list: List of category-to-token masks.
        """
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

    @staticmethod
    def get_configs(model_type: str) -> Args:
        """
        Get model configuration for specified model type.

        Args:
            model_type (str): Model type identifier.

        Returns:
            Args: Model configuration object.

        Raises:
            ValueError: If model_type is not supported.
        """
        base_config = dict(
            batch_size=1,
            modelname="groundingdino",
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

        if model_type == "groundingdino_swinb_cogcoor":
            base_config["backbone"] = "swin_B_384_22k"
        elif model_type == "groundingdino_swint_ogc":
            base_config["backbone"] = "swin_T_224_1k"
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

        return Args(**base_config)

    @classmethod
    def preprocess_image(
        cls,
        image: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image (np.ndarray): Input BGR image.
            target_size (Tuple[int, int]): Target (width, height).

        Returns:
            np.ndarray: Preprocessed image tensor.
        """
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - cls.IMAGE_MEAN) / cls.IMAGE_STD
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0).astype(np.float32)
        return image

    @classmethod
    def encode_text(
        cls,
        text_prompt: str,
        tokenizer: Tokenizer,
        max_text_len: int,
    ) -> Tuple[Dict, np.ndarray, np.ndarray, str]:
        """
        Encode text prompt for model input.

        Args:
            text_prompt (str): Text prompt to encode.
            tokenizer (Tokenizer): Tokenizer instance.
            max_text_len (int): Maximum text length.

        Returns:
            Tuple containing:
                - tokenized (Dict): Tokenized text.
                - text_self_attention_masks (np.ndarray): Attention masks.
                - position_ids (np.ndarray): Position IDs.
                - caption (str): Formatted caption.
        """
        caption = cls.get_caption(str(text_prompt))
        tokenized_raw_results = tokenizer.encode(caption)
        tokenized = {
            "input_ids": np.array([tokenized_raw_results.ids], dtype=np.int64),
            "token_type_ids": np.array(
                [tokenized_raw_results.type_ids], dtype=np.int64
            ),
            "attention_mask": np.array([tokenized_raw_results.attention_mask]),
        }
        (
            text_self_attention_masks,
            position_ids,
            _,
        ) = cls.generate_masks_with_special_tokens_and_transfer_map(
            tokenized, cls.SPECIAL_TOKENS
        )
        if text_self_attention_masks.shape[1] > max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, :max_text_len, :max_text_len
            ]
            position_ids = position_ids[:, :max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, :max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][
                :, :max_text_len
            ]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][
                :, :max_text_len
            ]
        return tokenized, text_self_attention_masks, position_ids, caption

    @classmethod
    def decode_predictions(
        cls,
        logits: np.ndarray,
        boxes: np.ndarray,
        caption: str,
        tokenizer: Tokenizer,
        box_threshold: float,
        text_threshold: float,
        apply_sigmoid: bool = False,
        with_logits: bool = True,
    ) -> Tuple[np.ndarray, List[List]]:
        """
        Decode model predictions to boxes and phrases.

        Args:
            logits (np.ndarray): Raw logits from model.
            boxes (np.ndarray): Raw boxes from model.
            caption (str): Input caption.
            tokenizer (Tokenizer): Tokenizer instance.
            box_threshold (float): Box confidence threshold.
            text_threshold (float): Text threshold.
            apply_sigmoid (bool): Whether to apply sigmoid to logits.
            with_logits (bool): Whether to include logit scores.

        Returns:
            Tuple[np.ndarray, List[List]]: Filtered boxes and phrase predictions.
        """
        logits_filt = np.squeeze(logits, 0)
        if apply_sigmoid:
            logits_filt = cls.sigmoid(logits_filt)
        boxes_filt = np.squeeze(boxes, 0)

        filt_mask = logits_filt.max(axis=1) > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        tokenized_raw_results = tokenizer.encode(caption)
        tokenized = {
            "input_ids": np.array(tokenized_raw_results.ids, dtype=np.int64),
            "token_type_ids": np.array(
                tokenized_raw_results.type_ids, dtype=np.int64
            ),
            "attention_mask": np.array(tokenized_raw_results.attention_mask),
        }

        pred_phrases = []
        special_token_ids = set(cls.SPECIAL_TOKENS)
        num_tokens = len(tokenized["input_ids"])
        phrase_ranges = cls.get_phrase_token_ranges(
            tokenized["input_ids"], special_token_ids
        )
        for logit in logits_filt:
            posmap = logit > text_threshold
            pred_phrase = cls.get_phrases_from_posmap(
                posmap, tokenized, tokenizer
            )
            if not pred_phrase or "##" in pred_phrase:
                best_phrase_score = -np.inf
                best_phrase_range = None
                for start_idx, end_idx in phrase_ranges:
                    phrase_score = logit[start_idx:end_idx].max()
                    if phrase_score > best_phrase_score:
                        best_phrase_score = phrase_score
                        best_phrase_range = (start_idx, end_idx)
                if best_phrase_range:
                    fallback_posmap = np.zeros(num_tokens, dtype=bool)
                    fallback_posmap[
                        best_phrase_range[0] : best_phrase_range[1]
                    ] = True
                    pred_phrase = cls.get_phrases_from_posmap(
                        fallback_posmap, tokenized, tokenizer
                    )
            if with_logits:
                pred_phrases.append([pred_phrase, logit.max()])
            else:
                pred_phrases.append([pred_phrase, 1.0])

        return boxes_filt, pred_phrases
