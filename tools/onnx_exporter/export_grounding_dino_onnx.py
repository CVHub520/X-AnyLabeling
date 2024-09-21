import argparse
import os
import os.path as osp

import cv2
import time
import numpy as np
import onnxruntime as ort
from typing import Dict
from tokenizers import Tokenizer


"""
The ONNX Export of the Grounding DINO
Written by Wei Wang (CVHub)
    Usage:
        1. git clone https://github.com/IDEA-Research/GroundingDINO.git
        2. cd GroundingDINO and pip install -r requirements.txt
        3. export PYTHONPATH=/path/to/your/GroundingDINO
        4. Place the current script in this directory.
        5. Download the corresponding tokenizer.json and place it in this dir.
        6. Run the script.

        ```bash
        python export_grounding_dino_onnx.py \
            --config_file groundingdino/config/GroundingDINO_SwinB_cfg.py \
            --ckpt_file /path/to/your/groundingdino_swinb_cogcoor.pth or groundingdino_swint_ogc \
            --device 'cpu' or 'gpu'
        ```
"""


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class OnnxBaseModel:
    def __init__(self, model_path, device_type: str = "gpu"):
        self.sess_opts = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(
                os.environ["OMP_NUM_THREADS"]
            )

        self.providers = ["CPUExecutionProvider"]
        if device_type.lower() == "gpu":
            self.providers = ["CUDAExecutionProvider"]

        self.ort_session = ort.InferenceSession(
            model_path,
            providers=self.providers,
            sess_options=self.sess_opts,
        )

    def get_ort_inference(
        self, blob, inputs=None, extract=True, squeeze=False
    ):
        if inputs is None:
            inputs = self.get_input_name()
            outs = self.ort_session.run(None, {inputs: blob})
        else:
            outs = self.ort_session.run(None, inputs)
        if extract:
            outs = outs[0]
        if squeeze:
            outs = outs.squeeze(axis=0)
        return outs

    def get_input_name(self):
        return self.ort_session.get_inputs()[0].name

    def get_input_shape(self):
        return self.ort_session.get_inputs()[0].shape

    def get_output_name(self):
        return [out.name for out in self.ort_session.get_outputs()]


class Grounding_DINO:
    """Open-Set object detection model using Grounding_DINO"""

    def __init__(self, model_config=None) -> None:
        self.config = model_config
        model_abs_path = self.config["model_path"]
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(print("file not found: ", model_abs_path))

        self.net = OnnxBaseModel(
            model_abs_path, device_type=self.config["device"]
        )
        self.model_configs = self.get_configs(self.config["model_type"])
        self.net.max_text_len = self.model_configs.max_text_len
        self.net.tokenizer = self.get_tokenizer()
        self.box_threshold = self.config["box_threshold"]
        self.text_threshold = self.config["text_threshold"]
        self.target_size = (
            self.config["input_width"],
            self.config["input_height"],
        )

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
        captions = self.get_caption(text_prompt)
        # tokenized = self.net.tokenizer(captions, padding="longest", return_tensors="np")
        # specical_tokens = self.net.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        tokenized_raw_results = self.net.tokenizer.encode(captions)
        tokenized = {
            "input_ids": np.array([tokenized_raw_results.ids], dtype=np.int64),
            "token_type_ids": np.array(
                [tokenized_raw_results.type_ids], dtype=np.int64
            ),
            "attention_mask": np.array([tokenized_raw_results.attention_mask]),
        }
        # [self.net.tokenizer.token_to_id(i) for i in ["[CLS]", "[SEP]", ".", "?"]]
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

        blob, inputs, caption = self.preprocess(image, text_prompt)
        start_time = time.time()
        outputs = self.net.get_ort_inference(
            blob, inputs=inputs, extract=False
        )
        end_time = time.time()
        print("Inference time: {:.3f}s".format(end_time - start_time))

        boxes_filt, pred_phrases = self.postprocess(outputs, caption)

        shapes = []
        img_h, img_w, _ = image.shape
        boxes = self.rescale_boxes(boxes_filt, img_h, img_w)
        for box, label_info in zip(boxes, pred_phrases):
            x1, y1, x2, y2 = box
            label, conf = label_info

            shapes.append([label, conf, x1, y1, x2, y2])

        return shapes

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
                print("Invalid model_type in GroundingDINO model.")
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
    def get_tokenizer():
        config_json_file = "tokenizer.json"
        tokenizer = Tokenizer.from_file(config_json_file)
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


def load_pt_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"

    # modified config
    args.use_checkpoint = False
    args.use_transformer_ckpt = False

    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model


def export_onnx(model, output_file, is_quantize):
    caption = "the running dog ."  # ". ".join(input_text)
    input_ids = model.tokenizer([caption], return_tensors="pt")["input_ids"]
    position_ids = torch.tensor([[0, 0, 1, 2, 3, 0]])
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[True, True, True, True, True, True]])
    text_token_mask = torch.tensor(
        [
            [
                [True, False, False, False, False, False],
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, True, True, True, True, False],
                [False, False, False, False, False, True],
            ]
        ]
    )

    img = torch.randn(1, 3, 800, 1200)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "position_ids": {0: "batch_size", 1: "seq_len"},
        "token_type_ids": {0: "batch_size", 1: "seq_len"},
        "text_token_mask": {0: "batch_size", 1: "seq_len", 2: "seq_len"},
        "img": {0: "batch_size", 2: "height", 3: "width"},
        "logits": {0: "batch_size"},
        "boxes": {0: "batch_size"},
    }
    args = (
        img,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids,
        text_token_mask,
    )
    input_names = [
        "img",
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "text_token_mask",
    ]
    output_names = ["logits", "boxes"]

    # export onnx model
    torch.onnx.export(
        model,
        f=output_file,
        args=args,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=16,
    )

    if is_quantize:
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        import onnx

        onnx_version = tuple(map(int, onnx.__version__.split(".")))
        assert onnx_version >= (
            1,
            14,
            0,
        ), f"The onnx version must be large equal than '1.14.0', but got {onnx_version}"

        model_output = osp.splitext(output_file)[0] + "_quant.onnx"
        print(f"Quantizing model and writing to {output_file}...")
        quantize_dynamic(
            model_input=output_file,
            model_output=model_output,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Export Grounding DINO Model to ONNX", add_help=True
    )
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        required=True,
        help="path to config file",
    )
    parser.add_argument(
        "--ckpt_file",
        "-p",
        type=str,
        required=True,
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, help="output directory"
    )
    parser.add_argument(
        "--is_quantize",
        type=bool,
        default=False,
        help=(
            "If set, will quantize the model and save it with the *_quan.onnx name. "
            "Quantization is performed with quantize_dynamic from "
            "onnxruntime.quantization.quantize."
        ),
    )
    parser.add_argument(
        "--img_path",
        "-i",
        type=str,
        default="asset/demo2.jpg",
        help="Test image",
    )
    parser.add_argument(
        "--text_prompt",
        "-t",
        type=str,
        default="The running dog",
        help="Text prompt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device",
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.3, help="Box prediction score"
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25, help="Text prompt score"
    )
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    ckpt_file = args.ckpt_file  # change the path of the model
    output_dir = args.output_dir
    is_quantize = args.is_quantize
    img_path = args.img_path
    text_prompt = args.text_prompt
    device = args.device
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold

    onnx_file = osp.splitext(osp.basename(ckpt_file))[0] + ".onnx"
    # make dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        onnx_file = osp.join(output_dir, onnx_file)
    print(f"onnx_file = {onnx_file}")
    if not osp.exists(onnx_file):
        import torch
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict

        # load model
        model = load_pt_model(config_file, ckpt_file, cpu_only=True)
        # export model
        export_onnx(model, onnx_file, is_quantize)

    # inference on a image and test the speed
    swinb = "groundingdino_swinb_cogcoor"
    swint = "groundingdino_swint_ogc"
    model_type = swinb if swinb in onnx_file else swint
    configs = {
        "model_path": onnx_file,
        "model_type": model_type,
        "device": device,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "input_width": 1200,
        "input_height": 800,
    }

    model = Grounding_DINO(configs)
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    warm_up = 3
    for i in range(warm_up):
        model.predict_shapes(image, text_prompt=text_prompt)

    loop = 10
    start_time = time.time()
    for i in range(loop):
        model.predict_shapes(image, text_prompt=text_prompt)
    end_time = time.time()
    print("avg time: ", (end_time - start_time) / loop)
