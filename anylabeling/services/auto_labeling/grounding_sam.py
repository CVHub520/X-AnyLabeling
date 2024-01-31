import logging
import os
import traceback

import cv2
import numpy as np
import onnxruntime
from typing import Dict
from copy import deepcopy
from tokenizers import Tokenizer

from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QCoreApplication

from anylabeling.utils import GenericWorker
from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .lru_cache import LRUCache
from .utils.general import Args
from .engines.build_onnx_engine import OnnxBaseModel


class SegmentAnythingONNX:
    """Segmentation model using SAM-HQ"""

    def __init__(self, encoder_model_path, decoder_model_path) -> None:
        self.target_size = 1024
        self.input_size = (684, 1024)

        # Load models
        providers = onnxruntime.get_available_providers()

        # Pop TensorRT Runtime due to crashing issues
        # TODO: Add back when TensorRT backend is stable
        providers = [p for p in providers if p != "TensorrtExecutionProvider"]

        self.encoder_session = onnxruntime.InferenceSession(
            encoder_model_path, providers=providers
        )
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_model_path, providers=providers
        )

    def run_encoder(self, encoder_inputs):
        """Run encoder"""
        features = self.encoder_session.run(None, encoder_inputs)
        image_embeddings, interm_embeddings = features[0], np.stack(
            features[1:]
        )
        return image_embeddings, interm_embeddings

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def apply_coords(self, coords: np.ndarray, original_size, target_length):
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def run_decoder(
        self,
        image_embeddings,
        interm_embeddings,
        original_size,
        transform_matrix,
        input_points,
        input_labels,
    ):
        """Run decoder"""

        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate(
            [input_points, np.array([[0.0, 0.0]])], axis=0
        )[None, :, :]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[
            None, :
        ].astype(np.float32)
        onnx_coord = self.apply_coords(
            onnx_coord, self.input_size, self.target_size
        ).astype(np.float32)

        # Apply the transformation matrix to the coordinates.
        onnx_coord = np.concatenate(
            [
                onnx_coord,
                np.ones((1, onnx_coord.shape[1], 1), dtype=np.float32),
            ],
            axis=2,
        )
        onnx_coord = np.matmul(onnx_coord, transform_matrix.T)
        onnx_coord = onnx_coord[:, :, :2].astype(np.float32)

        # Create an empty mask input and an indicator for no mask.
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": image_embeddings,
            "interm_embeddings": interm_embeddings,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(self.input_size, dtype=np.float32),
        }
        masks, _, _ = self.decoder_session.run(None, decoder_inputs)

        # Transform the masks back to the original image size.
        inv_transform_matrix = np.linalg.inv(transform_matrix)
        transformed_masks = self.transform_masks(
            masks, original_size, inv_transform_matrix
        )

        return transformed_masks

    def transform_masks(self, masks, original_size, transform_matrix):
        """Transform masks
        Transform the masks back to the original image size.
        """
        output_masks = []
        for batch in range(masks.shape[0]):
            batch_masks = []
            for mask_id in range(masks.shape[1]):
                mask = masks[batch, mask_id]
                mask = cv2.warpAffine(
                    mask,
                    transform_matrix[:2],
                    (original_size[1], original_size[0]),
                    flags=cv2.INTER_LINEAR,
                )
                batch_masks.append(mask)
            output_masks.append(batch_masks)
        return np.array(output_masks)

    def encode(self, cv_image):
        """
        Calculate embedding and metadata for a single image.
        """
        original_size = cv_image.shape[:2]

        # Calculate a transformation matrix to convert to self.input_size
        scale_x = self.input_size[1] / cv_image.shape[1]
        scale_y = self.input_size[0] / cv_image.shape[0]
        scale = min(scale_x, scale_y)
        transform_matrix = np.array(
            [
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1],
            ]
        )
        cv_image = cv2.warpAffine(
            cv_image,
            transform_matrix[:2],
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_LINEAR,
        )

        encoder_inputs = {
            self.encoder_input_name: cv_image.astype(np.float32),
        }
        image_embeddings, interm_embeddings = self.run_encoder(encoder_inputs)
        return {
            "image_embeddings": image_embeddings,
            "interm_embeddings": interm_embeddings,
            "original_size": original_size,
            "transform_matrix": transform_matrix,
        }

    def predict_masks(self, embedding, input_points, input_labels):
        """
        Predict masks for a single image.
        """
        masks = self.run_decoder(
            embedding["image_embeddings"],
            embedding["interm_embeddings"],
            embedding["original_size"],
            embedding["transform_matrix"],
            input_points,
            input_labels,
        )

        return masks


class GroundingSAM(Model):
    """Open-Set instance segmentation model using GroundingSAM"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_type",
            "model_path",
            "input_width",
            "input_height",
            "box_threshold",
            "text_threshold",
            "encoder_model_path",
            "decoder_model_path",
        ]
        widgets = [
            "edit_text",
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

        # ----------- HQ-SAM ---------- #
        self.input_size = self.config["input_size"]
        self.max_width = self.config["max_width"]
        self.max_height = self.config["max_height"]

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
                    "Could not download or initialize encoder of SAM_HQ.",
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
                    "Could not download or initialize decoder of SAM_HQ.",
                )
            )

        # Load models
        self.model = SegmentAnythingONNX(
            encoder_model_abs_path, decoder_model_abs_path
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

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def preprocess(self, image, text_prompt, img_mask=None):
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
        if img_mask is None:
            inputs["img_mask"] = np.zeros(
                (1, image.shape[0], image.shape[2], image.shape[3]),
                dtype=np.float32,
            )
        else:
            inputs["img_mask"] = img_mask
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
        logits_filt = np.squeeze(
            logits, 0
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
                shape.line_width = 1
                shape.label = "AUTOLABEL_OBJECT" if label is None else label
                shape.selected = False
                shapes.append(shape)
        elif self.output_mode in ["rectangle", "rotation"]:
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
            shape.add_point(QtCore.QPointF(x_max, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_max))
            shape.add_point(QtCore.QPointF(x_min, y_max))
            shape.shape_type = (
                "rectangle" if self.output_mode == "rectangle" else "rotation"
            )
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.line_width = 1
            shape.label = "AUTOLABEL_OBJECT" if label is None else label
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
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        try:
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
                blob, inputs, caption = self.preprocess(
                    cv_image, text_prompt, None
                )
                outputs = self.net.get_ort_inference(
                    blob, inputs=inputs, extract=False
                )
                boxes_filt, pred_phrases = self.postprocess(outputs, caption)
                img_h, img_w, _ = cv_image.shape
                boxes = self.rescale_boxes(boxes_filt, img_h, img_w)
                shapes = []
                for box, label_info in zip(boxes, pred_phrases):
                    x1, y1, x2, y2 = box
                    label, _ = label_info
                    point_coords = np.array(
                        [[x1, y1], [x2, y2]], dtype=np.float32
                    )
                    point_labels = np.array([2, 3], dtype=np.float32)
                    masks = self.model.predict_masks(
                        image_embedding, point_coords, point_labels
                    )
                    if len(masks.shape) == 4:
                        masks = masks[0][0]
                    else:
                        masks = masks[0]
                    results = self.post_process(masks, label=label)
                    shapes.append(results)
                result = AutoLabelingResult(shapes, replace=False)
            else:
                point_coords, point_labels = self.get_input_points()
                masks = self.model.predict_masks(
                    image_embedding, point_coords, point_labels
                )
                if len(masks.shape) == 4:
                    masks = masks[0][0]
                else:
                    masks = masks[0]
                shapes = self.post_process(masks)
                result = AutoLabelingResult(shapes, replace=False)
            return result
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

    def get_input_points(self):
        """Get input points"""
        points = []
        labels = []
        for mark in self.marks:
            if mark["type"] == "point":
                points.append(mark["data"])
                labels.append(mark["label"])
            elif mark["type"] == "rectangle":
                points.append([mark["data"][0], mark["data"][1]])  # top left
                points.append(
                    [mark["data"][2], mark["data"][3]]
                )  # bottom right
                labels.append(2)
                labels.append(3)
        points, labels = np.array(points).astype(np.float32), np.array(
            labels
        ).astype(np.float32)
        return points, labels

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
        current_dir = os.path.dirname(__file__)
        cfg_name = text_encoder_type.replace("-", "_") + "_tokenizer.json"
        cfg_file = os.path.join(current_dir, "configs", cfg_name)
        tokenizer = Tokenizer.from_file(cfg_file)
        return tokenizer

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
