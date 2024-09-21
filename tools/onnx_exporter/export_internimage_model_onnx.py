import os
import os.path as osp
from typing import Any

import cv2
import random
import numpy as np
import onnxruntime as ort


"""
The ONNX Export of the InternImage Model
Written by Wei Wang (CVHub)

    Export:
        1. git clone https://github.com/OpenGVLab/InternImage
        2. cd InternImage and follow the official tutorial to install package.
        3. before u convert the *pt to *onnx model, remember set the opset_version >= 16 \
            and modify parameters in the configuration file:
            Set CORE_OP to 'DCNv3_pytorch' and USE_CHECKPOINT to false.
        4. Download the weight file and refer to the following running script:
        ```bash
            python export.py \
                --model_name internimage_l_22kto1k_384 \
                --ckpt_dir /home/cvhub/workspace/resources/weights/internimage/classification \
                --onnx
        ```
"""


class OnnxBaseModel:
    def __init__(self, model_path, device_type: str = "cpu"):
        self.sess_opts = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(
                os.environ["OMP_NUM_THREADS"]
            )

        self.providers = ["CPUExecutionProvider"]
        if device_type.lower() != "cpu":
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


class Model(OnnxBaseModel):
    def __init__(
        self, model_path, device_type: str = "cpu", configs: dict = {}
    ):
        super().__init__(model_path, device_type)

        self.configs = configs
        self.class_names = self.configs["class_names"]
        self.num_classes = len(self.class_names)
        input_width = self.configs.get(
            "input_width", self.get_input_shape()[-1]
        )
        input_height = self.configs.get(
            "input_height", self.get_input_shape()[-2]
        )
        self.input_shape = (input_height, input_width)
        self.color_map = [
            (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            for _ in range(self.num_classes)
        ]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.predict(*args, **kwds)

    def preprocess(self, input_image, mean=None, std=None):
        """
        Pre-processes the input image before feeding it to the network.

        Args:
            input_image (numpy.ndarray): The input image to be processed.
            mean (numpy.ndarray): Mean values for normalization.
                If not provided, default values are used.
            std (numpy.ndarray): Standard deviation values for normalization.
                If not provided, default values are used.

        Returns:
            numpy.ndarray: The processed input image.
        """
        h, w = self.input_shape
        # Resize the input image
        input_data = cv2.resize(input_image, (w, h))
        # Transpose the dimensions of the image
        input_data = input_data.transpose((2, 0, 1))
        if not mean:
            mean = np.array([0.485, 0.456, 0.406])
        if not std:
            std = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(input_data.shape).astype("float32")
        # Normalize the image data
        for channel in range(input_data.shape[0]):
            norm_img_data[channel, :, :] = (
                input_data[channel, :, :] / 255 - mean[channel]
            ) / std[channel]
        blob = norm_img_data.reshape(1, 3, h, w).astype("float32")
        return blob

    @staticmethod
    def softmax(x):
        """
        Applies the softmax function to the input array.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output array after applying softmax.
        """
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def postprocess(self, outs):
        """
        Classification: Post-processes the output of the network.

        Args:
            outs (list): Output predictions from the network.
            topk (int): Number of top predictions to consider. Default is 1.

        Returns:
            str: Predicted label.
        """
        res = self.softmax(np.array(outs)).tolist()
        index = np.argmax(res)
        label = str(self.classes[index])

        return label

    def predict(self, img: np.array):
        blob = self.preprocess(img)
        predictions = self.get_ort_inference(blob)
        label = self.postprocess(predictions)
        return label


if __name__ == "__main__":
    model_path = "/path/to/internimage_l_22kto1k_384.onnx"
    image_path = "/path/to/demo_imagenet.jpg"
    device_type = "cpu"  # 'cpu' or 'gpu'
    configs = {"class_names": [...]}
    model = Model(model_path, device_type, configs)
    image = cv2.imread(image_path)
    label = model(image)
    print(f"{image_path}: {label}")
