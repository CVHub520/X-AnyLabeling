import argparse
import os
import os.path as osp

import cv2
import time
import shutil
import numpy as np
from tempfile import mkdtemp
from types import MethodType
from collections import OrderedDict

import onnx
import onnxruntime as ort
from onnx.external_data_helper import convert_model_to_external_data

"""
The ONNX Export of the GeCo
Written by TaleBolano 
    Usage:
        1. git clone https://github.com/jerpelhan/GeCo.git
        2. cd GeCo and pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        3. Place the current script in GeCo directory.
        4. Run the script.

        ```bash
        python export_geco_onnx.py \
            --output_dir onnx_outputs \
            --ckpt_file GeCo.pth \
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


class GeCo:
    """few shot object conut model GeCo"""

    def __init__(self, model_config=None) -> None:
        self.config = model_config
        encoder_path = self.config["encoder_path"]
        decoder_path = self.config["decoder_path"]
        if not encoder_path or not os.path.isfile(encoder_path):
            raise FileNotFoundError(print("file not found: ", encoder_path))
        if not decoder_path or not os.path.isfile(decoder_path):
            raise FileNotFoundError(print("file not found: ", decoder_path))

        self.encoder = OnnxBaseModel(
            encoder_path, device_type=self.config["device"]
        )
        self.decoder = OnnxBaseModel(
            decoder_path, device_type=self.config["device"]
        )
        self.box_threshold = self.config["box_threshold"]

        self.target_size = self.config["input_size"]

    def preprocess(self, image):
        # Resize the image
        scale_x = self.target_size / image.shape[1]
        scale_y = self.target_size / image.shape[0]
        scale_factor = min(scale_x, scale_y)

        new_H = int(scale_factor * image.shape[0])
        new_W = int(scale_factor * image.shape[1])

        image = cv2.resize(
            image, (new_W, new_H), interpolation=cv2.INTER_LINEAR
        )

        top = 0
        bottom = self.target_size - new_H
        left = 0
        right = self.target_size - new_W

        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )  # add border
        image = np.transpose(image / 255.0, axes=(2, 0, 1))[None].astype(
            np.float32
        )

        return image, scale_factor

    def postprocess(self, outputs, scale_factor):
        pred_bboxes, box_v = outputs[0][0], outputs[1][0]

        keep = box_v > box_v.max() / self.box_threshold

        pred_bboxes = pred_bboxes[keep]
        box_v = box_v[keep]

        xywh = pred_bboxes.copy()  # xyxy
        xywh[:, :2] = pred_bboxes[:, :2]
        xywh[:, 2:] = pred_bboxes[:, 2:] - pred_bboxes[:, :2]

        keep = cv2.dnn.NMSBoxes(xywh.tolist(), box_v.tolist(), 0, 0.6)

        pred_bboxes = pred_bboxes[keep]
        box_v = box_v[keep]

        pred_bboxes = np.clip(pred_bboxes, 0, 1)

        pred_bboxes = (
            pred_bboxes
            / np.array(
                [scale_factor, scale_factor, scale_factor, scale_factor]
            )
            * self.target_size
        )

        return pred_bboxes

    def predict_shapes(self, image, box_prompt, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        image, scale_factor = self.preprocess(image)
        start_time = time.time()
        image_embeddings, hq_features = self.encoder.get_ort_inference(
            image, inputs={"input_image": image}, extract=False
        )

        prompt_bboxes = np.array([box_prompt])  # n,4
        prompt_bboxes = prompt_bboxes * np.array(
            [scale_factor, scale_factor, scale_factor, scale_factor]
        )
        prompt_bboxes = prompt_bboxes[None].astype(np.float32)

        decoder_inputs = {
            "image_embeddings": image_embeddings,
            "hq_features": hq_features,
            "bboxes": prompt_bboxes,
        }

        outputs = self.decoder.get_ort_inference(
            None, inputs=decoder_inputs, extract=False
        )

        end_time = time.time()
        print("Inference time: {:.3f}s".format(end_time - start_time))

        pred_bboxes = self.postprocess(outputs, scale_factor)

        shapes = []
        for box in pred_bboxes:
            x1, y1, x2, y2 = box
            shapes.append([x1, y1, x2, y2])

        return shapes

    def unload(self):
        del self.encoder
        del self.decoder


def export_onnx(model, output_encoder_file, output_decoder_file, is_quantize):
    # export encoder
    tmp_dir = mkdtemp()
    tmp_model_path = os.path.join(tmp_dir, f"encoder.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model.backbone,
            torch.randn(1, 3, 1024, 1024, dtype=torch.float),
            tmp_model_path,
            input_names=["input_image"],
            output_names=["image_embeddings", "hq_features"],
            export_params=True,
            do_constant_folding=True,
            verbose=True,
        )

    # Combine the weights into a single file
    onnx_model = onnx.load(tmp_model_path)
    convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=True,
        location=osp.basename(output_encoder_file).replace(
            ".onnx", "_data.bin"
        ),
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save(onnx_model, output_encoder_file)
    # Cleanup the temporary directory
    shutil.rmtree(tmp_dir)

    if is_quantize:
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        onnx_version = tuple(map(int, onnx.__version__.split(".")))
        assert onnx_version >= (
            1,
            14,
            0,
        ), f"The onnx version must be large equal than '1.14.0', but got {onnx_version}"

        model_output = osp.splitext(output_encoder_file)[0] + "_quant.onnx"
        print(f"Quantizing model and writing to {output_encoder_file}...")
        quantize_dynamic(
            model_input=output_encoder_file,
            model_output=model_output,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )

    # export decdoer
    def decoder_forward(self, src, src_hq, bboxes):
        if not self.zero_shot:
            prototype_embeddings = self.create_prototypes(src, bboxes)

        else:  # zero shot
            prototype_embeddings = self.exemplars.expand(bs, -1, -1)
        adapted_f = self.adapt_features(
            image_embeddings=src,
            image_pe=self.prompt_encoder.get_dense_pe(),
            prototype_embeddings=prototype_embeddings,
            hq_features=src_hq,
        )

        # Predict class [fg, bg] and l,r,t,b
        bs, c, w, h = adapted_f.shape
        adapted_f = adapted_f.view(bs, self.emb_dim, -1).permute(0, 2, 1)
        centerness = (
            self.class_embed(adapted_f).view(bs, w, h, 1).permute(0, 3, 1, 2)
        )
        outputs_coord = (
            self.bbox_embed(adapted_f)
            .sigmoid()
            .view(bs, w, h, 4)
            .permute(0, 3, 1, 2)
        )
        outputs, ref_points = boxes_with_scores(
            centerness, outputs_coord, batch_thresh=0.001
        )

        return (
            outputs[0]["pred_boxes"],
            outputs[0]["box_v"],
        )  # , ref_points, centerness, outputs_coord, masks, prototype_embeddings

    model.forward = MethodType(decoder_forward, model)

    dummy_inputs = {
        "image_embeddings": torch.randn(1, 256, 64, 64, dtype=torch.float),
        "hq_features": torch.randn(1, 32, 256, 256, dtype=torch.float),
        "bboxes": torch.randn(size=(1, 1, 4), dtype=torch.float),
    }

    dynamic_axes = {
        "bboxes": {1: "num_bboxes"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            output_decoder_file,
            input_names=list(dummy_inputs.keys()),
            output_names=["pred_boxes", "box_v"],
            dynamic_axes=dynamic_axes,
            export_params=True,
            do_constant_folding=True,
            verbose=True,
        )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Export GeCo Model to ONNX", add_help=True
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
        default="asset/demo_grape.jpg",
        help="Test image",
    )
    parser.add_argument(
        "--box_prompt",
        "-b",
        type=float,
        nargs=4,
        default=[243.0, 74.0, 280.0, 103.0],
        help="box prompt [x1,y1,x2,y2]",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device",
    )
    parser.add_argument(
        "--box_threshold", type=float, default=4, help="Box prediction score"
    )
    args = parser.parse_args()

    # cfg
    ckpt_file = args.ckpt_file
    output_dir = args.output_dir
    is_quantize = args.is_quantize
    img_path = args.img_path
    device = args.device
    box_threshold = args.box_threshold

    onnx_encoder_file = (
        osp.splitext(osp.basename(ckpt_file))[0] + "_encoder.onnx"
    )
    onnx_decoder_file = (
        osp.splitext(osp.basename(ckpt_file))[0] + "_decoder.onnx"
    )
    # make dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        onnx_encoder_file = osp.join(output_dir, onnx_encoder_file)
        onnx_decoder_file = osp.join(output_dir, onnx_decoder_file)
    print(f"onnx_encoder_file = {onnx_encoder_file}")
    print(f"onnx_decoder_file = {onnx_decoder_file}")
    if not (osp.exists(onnx_encoder_file) and osp.exists(onnx_decoder_file)):
        import torch
        from models.geco_infer import GeCo as GeCo_pytorch
        from utils.box_ops import boxes_with_scores

        # load model
        model = GeCo_pytorch(
            image_size=1024,
            num_objects=3,
            emb_dim=256,
            num_heads=8,
            kernel_dim=1,
            train_backbone=False,
            reduction=16,
            zero_shot=False,
            model_path=None,
            return_masks=False,
        )

        # 加载并处理检查点
        checkpoint = torch.load(args.ckpt_file, map_location="cpu")
        state_dict = checkpoint["model"]

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_k = k.replace("module.", "", 1)
            new_state_dict[new_k] = v

        model.load_state_dict(
            new_state_dict,
            strict=False,
        )

        model.eval()

        # export model
        export_onnx(model, onnx_encoder_file, onnx_decoder_file, is_quantize)

    # inference on a image and test the speed
    configs = {
        "encoder_path": onnx_encoder_file,
        "decoder_path": onnx_decoder_file,
        "device": device,
        "box_threshold": box_threshold,
        "input_size": 1024,
    }

    if is_quantize:
        configs["encoder_path"] = (
            osp.splitext(onnx_encoder_file)[0] + "_quant.onnx"
        )

    model = GeCo(configs)
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    warm_up = 3
    for i in range(warm_up):
        model.predict_shapes(image, box_prompt=args.box_prompt)

    loop = 10
    start_time = time.time()
    for i in range(loop):
        model.predict_shapes(image, box_prompt=args.box_prompt)
    end_time = time.time()
    print("avg time: ", (end_time - start_time) / loop)
