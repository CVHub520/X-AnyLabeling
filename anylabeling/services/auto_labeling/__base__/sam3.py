from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


class SegmentAnything3ONNX:
    """Segmentation model using Segment Anything 3."""

    def __init__(
        self,
        image_encoder_path,
        decoder_model_path,
        language_encoder_path,
        device,
    ) -> None:
        self.image_encoder = SAM3ImageEncoder(image_encoder_path, device)
        self.language_encoder = SAM3LanguageEncoder(
            language_encoder_path, device
        )
        self.decoder = SAM3ImageDecoder(decoder_model_path, device)

    def encode(self, cv_image: np.ndarray, text_prompt: str) -> dict[str, Any]:
        embedding = self.encode_image(cv_image)
        return self.apply_text_prompt(embedding, text_prompt)

    def encode_image(self, cv_image: np.ndarray) -> dict[str, Any]:
        original_size = cv_image.shape[:2]
        image_encoder_outputs = self.image_encoder(cv_image)
        return {
            "vision_pos_enc_0": image_encoder_outputs[0],
            "vision_pos_enc_1": image_encoder_outputs[1],
            "vision_pos_enc_2": image_encoder_outputs[2],
            "backbone_fpn_0": image_encoder_outputs[3],
            "backbone_fpn_1": image_encoder_outputs[4],
            "backbone_fpn_2": image_encoder_outputs[5],
            "original_size": original_size,
        }

    def apply_text_prompt(
        self, embedding: dict[str, Any], text_prompt: str
    ) -> dict[str, Any]:
        lang_outputs = self.language_encoder(text_prompt)
        embedding = dict(embedding)
        embedding["language_mask"] = lang_outputs[0]
        embedding["language_features"] = lang_outputs[1]
        embedding["language_embeds"] = lang_outputs[2]
        return embedding

    def predict_masks(
        self,
        embedding: dict[str, Any],
        prompt,
        confidence_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        original_size = embedding["original_size"]
        box_coords = [0.0, 0.0, 0.0, 0.0]
        box_labels = [1]
        box_masks = [True]

        for mark in prompt:
            if mark["type"] == "rectangle":
                x1, y1, x2, y2 = mark["data"]
                box_coords = [
                    (x1 + x2) / 2.0 / original_size[1],
                    (y1 + y2) / 2.0 / original_size[0],
                    (x2 - x1) / original_size[1],
                    (y2 - y1) / original_size[0],
                ]
                box_masks = [False]
                break
            if mark["type"] == "point":
                x, y = mark["data"]
                box_coords = [
                    x / original_size[1],
                    y / original_size[0],
                    0.01,
                    0.01,
                ]
                box_masks = [False]
                break

        masks, scores, _ = self.decoder(
            original_size,
            embedding["vision_pos_enc_0"],
            embedding["vision_pos_enc_1"],
            embedding["vision_pos_enc_2"],
            embedding["backbone_fpn_0"],
            embedding["backbone_fpn_1"],
            embedding["backbone_fpn_2"],
            embedding["language_mask"],
            embedding["language_features"],
            embedding["language_embeds"],
            np.array(box_coords, dtype=np.float32).reshape(1, 1, 4),
            np.array([box_labels], dtype=np.int64),
            np.array([box_masks], dtype=np.bool_),
        )

        if len(scores) == 0:
            return masks, scores
        keep = np.where(scores > confidence_threshold)[0]
        if len(keep) == 0:
            masks = np.zeros((0,) + masks.shape[1:], dtype=masks.dtype)
            scores = np.zeros((0,), dtype=scores.dtype)
            return masks, scores
        return masks[keep], scores[keep]


class SAM3ImageEncoder:
    def __init__(self, path: str, device: str) -> None:
        self.session = create_ort_session(path, device)
        encoder_input = self.session.get_inputs()[0]
        self.input_name = encoder_input.name
        self.input_shape = encoder_input.shape
        self.input_type = encoder_input.type
        if len(self.input_shape) == 3:
            self.input_height = int(self.input_shape[1]) or 1008
            self.input_width = int(self.input_shape[2]) or 1008
        else:
            self.input_height = int(self.input_shape[2]) or 1008
            self.input_width = int(self.input_shape[3]) or 1008

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        return self.session.run(
            None, {self.input_name: self.prepare_input(image)}
        )

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        input_img = cv2.resize(
            image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        input_img = input_img.transpose(2, 0, 1)

        if self.input_type == "tensor(float)":
            input_tensor = ((input_img / 255.0) - 0.5) / 0.5
            return input_tensor.astype(np.float32)
        return input_img.astype(np.uint8)


class SAM3LanguageEncoder:
    def __init__(self, path: str, device: str) -> None:
        self.session = create_ort_session(path, device)
        from ..osam.clip import tokenize

        self._tokenize = tokenize

    def __call__(self, text: str) -> list[np.ndarray]:
        tokens = self._tokenize([text], context_length=32)
        if not isinstance(tokens, np.ndarray):
            tokens = np.asarray(tokens, dtype=np.int64)
        return self.session.run(None, {"tokens": tokens})


class SAM3ImageDecoder:
    def __init__(self, path: str, device: str) -> None:
        self.session = create_ort_session(path, device)
        self.input_names = [i.name for i in self.session.get_inputs()]

    def __call__(
        self,
        original_size,
        vision_pos_enc_0,
        vision_pos_enc_1,
        vision_pos_enc_2,
        backbone_fpn_0,
        backbone_fpn_1,
        backbone_fpn_2,
        language_mask,
        language_features,
        language_embeds,
        box_coords,
        box_labels,
        box_masks,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs = {
            "original_height": np.array(original_size[0], dtype=np.int64),
            "original_width": np.array(original_size[1], dtype=np.int64),
            "vision_pos_enc_0": vision_pos_enc_0,
            "vision_pos_enc_1": vision_pos_enc_1,
            "vision_pos_enc_2": vision_pos_enc_2,
            "backbone_fpn_0": backbone_fpn_0,
            "backbone_fpn_1": backbone_fpn_1,
            "backbone_fpn_2": backbone_fpn_2,
            "language_mask": language_mask,
            "language_features": language_features,
            "language_embeds": language_embeds,
            "box_coords": box_coords,
            "box_labels": box_labels,
            "box_masks": box_masks,
        }
        outputs = self.session.run(
            None,
            {k: v for k, v in inputs.items() if k in self.input_names},
        )
        return outputs[2], outputs[1], outputs[0]


def get_ort_providers(device: str) -> list[str]:
    providers = ["CPUExecutionProvider"]
    if device.lower() == "gpu":
        providers = ["CUDAExecutionProvider"]
    return providers


def create_ort_session(path: str, device: str) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    return ort.InferenceSession(
        path, providers=get_ort_providers(device), sess_options=sess_options
    )
