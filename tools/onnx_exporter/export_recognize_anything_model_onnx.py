import os
import os.path as osp
import argparse

import cv2
import numpy as np
import onnxruntime as ort

"""
The ONNX Export of the Recognize Anything Model (RAM)
Written by Wei Wang (CVHub)

    Before handling, you should add the following functions to the \
        RAM class in ram/models/ran.py

    ```python
    def forward(
        self,
        image,
    ):
        label_embed = torch.nn.functional.relu(self.wordvec_proj(self.label_embed))

        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        # recognized image tags using image-tag recogntiion decoder
        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]
        label_embed = label_embed.unsqueeze(0).repeat(bs, 1, 1)
        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = self.fc(tagging_embed[0]).squeeze(-1)

        targets = torch.where(
            torch.sigmoid(logits) > self.class_threshold.to(image.device),
            torch.tensor(1.0).to(image.device),
            torch.zeros(self.num_class).to(image.device))

        # Create a constant tensor for bs
        bs_tensor = torch.tensor([bs]).to(image.device)

        return (targets, bs_tensor)
    ```

    Usage:
        1. git clone https://github.com/xinyu1205/recognize-anything
        2. cd recognize-anything and pip install -r requirements.txt
        3. export PYTHONPATH=/path/to/your/recognize-anything
        4. Place the current script in this directory.
        5. Download the *.pth file.
        6. Run the script.

        ```bash
        python export_recognize_anything_model_onnx.py  \
            --ckpt_file pretrained/ram_swin_large_14m.pth \
            --is_quantize True \
            --device 'cpu' or 'gpu'
        ```

"""


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


class RAM:
    def __init__(self, model_config=None) -> None:
        """
        Args:
            model_config (str): model's configuration file
            threshold (int): tagging threshold
            delete_tag_index (list): delete some tags that may disturb captioning
        """
        self.config = model_config
        model_abs_path = self.config["model_path"]
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(print("file not found: ", model_abs_path))

        self.net = OnnxBaseModel(
            model_abs_path, device_type=self.config["device"]
        )
        self.input_shape = self.net.get_input_shape()[-2:]
        self.delete_tag_index = []

        # load tag list
        self.tag_list = self.load_tag_list(self.config["tag_list"])
        self.tag_list_chinese = self.load_tag_list(
            self.config["tag_list_chinese"]
        )

    def preprocess(self, image):
        h, w = self.input_shape
        image = cv2.resize(image, (w, h))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0).astype(np.float32)
        return image

    def postprocess(self, outs):
        tags, bs = outs
        tags[:, self.delete_tag_index] = 0
        tag_output = []
        tag_output_chinese = []
        for b in range(bs[0]):
            index = np.argwhere(tags[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            tag_output.append(" | ".join(token))
            token_chinese = self.tag_list_chinese[index].squeeze(axis=1)
            tag_output_chinese.append(" | ".join(token_chinese))

        return tag_output, tag_output_chinese

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        blob = self.preprocess(image)
        outs = self.net.get_ort_inference(blob, extract=False)
        tags = self.postprocess(outs)
        print("Image Tags: ", tags[0])
        print("图像标签: ", tags[1])

    @staticmethod
    def load_tag_list(tag_list_file):
        with open(tag_list_file, "r", encoding="utf-8") as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list

    def unload(self):
        del self.net


def export_onnx(onnx_file, is_quantize):
    if not osp.exists(onnx_file):
        import torch
        from ram.models import ram

        model = ram(pretrained=ckpt_file, image_size=image_size, vit="swin_l")
        model.eval()
        model = model.to(device)
        image = torch.randn(1, 3, image_size, image_size).to(device)
        dynamic_axes = {"targets": {0: "batch_size"}, "bs": {0: "batch_size"}}
        torch.onnx.export(
            model,
            image,
            onnx_file,
            verbose=True,
            opset_version=opset,
            export_params=True,
            input_names=["img"],
            dynamic_axes=dynamic_axes,
            output_names=["targets", "bs"],
        )
        # Optional: Verify the ONNX model using onnx.checker
        import onnx

        onnx.checker.check_model(onnx.load(onnx_file))

    model_output = osp.splitext(onnx_file)[0] + "_quant.onnx"
    if is_quantize and not osp.exists(model_output):
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        import onnx

        onnx_version = tuple(map(int, onnx.__version__.split(".")))
        assert onnx_version >= (
            1,
            14,
            0,
        ), f"The onnx version must be large equal than '1.14.0', but got {onnx_version}"
        print(f"Quantizing model and writing to {model_output}...")
        quantize_dynamic(
            model_input=onnx_file,
            model_output=model_output,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAM inferece for tagging and captioning"
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
    parser.add_argument("--opset", type=int, default=16, help="opset version")
    parser.add_argument(
        "--image-size",
        default=384,
        type=int,
        metavar="N",
        help="input image size (default: 448)",
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
        default="images/demo/demo1.jpg",
        help="Test image",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device",
    )
    parser.add_argument(
        "--tag_list",
        type=str,
        default="ram/data/ram_tag_list.txt",
        help="RAM Tag Label List",
    )
    parser.add_argument(
        "--tag_list_chinese",
        type=str,
        default="ram/data/ram_tag_list_chinese.txt",
        help="RAM Tag Chinese Label List",
    )
    args = parser.parse_args()

    opset = args.opset
    device = args.device
    img_path = args.img_path
    ckpt_file = args.ckpt_file
    image_size = args.image_size
    output_dir = args.output_dir
    is_quantize = args.is_quantize

    tag_list = args.tag_list
    tag_list_chinese = args.tag_list_chinese

    onnx_file = osp.splitext(osp.basename(ckpt_file))[0] + ".onnx"
    # make dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        onnx_file = osp.join(output_dir, onnx_file)

    export_onnx(onnx_file, is_quantize)

    onnx_quant_file = osp.splitext(onnx_file)[0] + "_quant.onnx"
    onnx_file_list = [onnx_file, onnx_quant_file]

    for model_path in onnx_file_list:
        if model_path:
            print(f"Inference using {model_path}")
            print("-" * 100)
            configs = {
                "model_path": model_path,
                "device": "cpu",
                "delete_tag_index": [],
                "tag_list": tag_list,
                "tag_list_chinese": tag_list_chinese,
            }
            model = RAM(configs)
            image = cv2.imread(img_path)
            model.predict_shapes(image)
            print("-" * 100)

    """
    Inference using ram_swin_large_14m.onnx (866M)
    -------------------------------------------------------------------------------
    Image Tags:  ['floor | furniture | living room | plant | room | stool | white']
    图像标签:  ['地板/地面 | 家具  | 客厅  | 植物  | 房间  | 凳子  | 白色']
    -------------------------------------------------------------------------------
    Inference using ram_swin_large_14m_quant.onnx (261M)
    -------------------------------------------------------------------------------
    Image Tags:  ['floor | furniture | living room | room | stool']
    图像标签:  ['地板/地面 | 家具  | 客厅  | 房间  | 凳子 ']
    -------------------------------------------------------------------------------
    """
