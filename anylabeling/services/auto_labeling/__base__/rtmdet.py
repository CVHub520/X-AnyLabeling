from typing import List, Tuple

import cv2
import numpy as np

from ..engines import OnnxBaseModel

"""
The onnxruntime demo of the RTMDet model
Ref:
    https://github.com/open-mmlab/mmyolo/tree/dev/configs/rtmdet
    https://platform.openmmlab.com/deploee/onnx-detail?search=rtmdet&tag=&page=1
    https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/object_detection/rtmdet.py
"""


class RTMDet:
    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple = (640, 640),
        score_thr: float = 0.3,
        mean: tuple = (103.5300, 116.2800, 123.6750),
        std: tuple = (57.3750, 57.1200, 58.3950),
        backend: str = "onnxruntime",
        device: str = "cpu",
    ):
        super().__init__()
        self.net = OnnxBaseModel(onnx_model, device_type=device)
        self.model_input_size = self.net.get_input_shape()[-2:]
        if not isinstance(self.model_input_size[0], int):
            self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.score_thr = score_thr

    def __call__(self, image: np.ndarray):
        image, ratio = self.preprocess(image)
        outputs = self.inference(image)
        results = self.postprocess(outputs, ratio)
        return results

    def inference(self, blob: np.ndarray):
        """Inference model.

        Args:
            blob (np.ndarray): Input image in shape.

        Returns:
            outputs (np.ndarray): Output of RTMPose model.
        """
        # run model
        outputs = self.net.get_ort_inference(blob)

        return outputs

    def preprocess(self, img: np.ndarray):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        if len(img.shape) == 3:
            padded_img = (
                np.ones(
                    (self.model_input_size[0], self.model_input_size[1], 3),
                    dtype=np.uint8,
                )
                * 114
            )
        else:
            padded_img = np.ones(self.model_input_size, dtype=np.uint8) * 114

        ratio = min(
            self.model_input_size[0] / img.shape[0],
            self.model_input_size[1] / img.shape[1],
        )
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[: padded_shape[0], : padded_shape[1]] = resized_img

        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            padded_img = (padded_img - self.mean) / self.std

        # build input to (1, 3, H, W)
        transposed_img = padded_img.transpose(2, 0, 1)
        transposed_img = np.ascontiguousarray(transposed_img, dtype=np.float32)
        blob = transposed_img[None, :, :, :]

        return blob, ratio

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RTMDet model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMDet model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """

        if outputs.shape[-1] == 4:
            # onnx without nms module
            raise NotImplementedError(
                "RTMDet without nms module is not supported yet."
            )
        elif outputs.shape[-1] == 5:
            # onnx contains nms module
            pack_dets = (outputs[0, :, :4], outputs[0, :, 4])
            final_boxes, final_scores = pack_dets
            final_boxes /= ratio
            isscore = final_scores > self.score_thr
            isbbox = [i for i in isscore]
            final_boxes = final_boxes[isbbox]

        return final_boxes


if __name__ == "__main__":
    onnx_model = "/home/cvhub/xanylabeling_data/models/rtmdet_nano_coco_person_rtmo_s-r20240112/rtmdet_nano_320-8xb32_coco-person.onnx"
    image_path = "/home/cvhub/workspace/projects/python/toolkits/X-AnyLabeling/assets/examples/demo_sod.jpeg"
    rtmdet = RTMDet(onnx_model, score_thr=0.3)
    image = cv2.imread(image_path)
    results = rtmdet(image)
    print(f"results: {results}")
