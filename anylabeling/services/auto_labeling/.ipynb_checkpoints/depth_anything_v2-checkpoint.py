import os
import cv2

from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel
from . import _THUMBNAIL_RENDER_MODELS


class DepthAnythingV2(Model):
    """DepthAnything demo"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_name = self.config["type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_name} model.",
                )
            )
        self.model_path = model_abs_path
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.input_shape = self.net.get_input_shape()[-2:]
        self.render_mode = self.config.get("render_mode", "color")
        self.device = "cuda" if __preferred_device__ == "GPU" else "cpu"
        self.save_dir, self.file_ext = _THUMBNAIL_RENDER_MODELS[
            "depth_anything_v2"
        ]

    def preprocess(self, input_image):
        """
        Pre-processes the input image before feeding it to the network.

        Args:
            input_image (numpy.ndarray): The input image to be processed.

        Returns:
            numpy.ndarray: The pre-processed output.
        """
        height, width = self.input_shape
        orig_shape = input_image.shape[:2]
        image = input_image / 255.0
        image = cv2.resize(
            image, (width, height), interpolation=cv2.INTER_CUBIC
        )
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = image.transpose(2, 0, 1)[None].astype("float32")
        return image, orig_shape

    def forward(self, blob):
        return self.net.get_ort_inference(blob, extract=True)

    def postprocess(self, depth, orig_shape):
        """
        Post-processes the network's output.

        Args:
            depth (numpy.ndarray): The output from the network.

        Returns:
            depth_color: xxx
        """
        orig_h, orig_w = orig_shape
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.transpose(1, 2, 0).astype("uint8")
        depth = cv2.resize(
            depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC
        )
        if self.render_mode == "color":
            return cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        return depth

    def predict_shapes(self, image, image_path=None):
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

        blob, orig_shape = self.preprocess(image)
        outputs = self.forward(blob)
        depth = self.postprocess(outputs, orig_shape)

        image_dir_path = os.path.dirname(image_path)
        save_path = os.path.join(image_dir_path, "..", self.save_dir)
        save_path = os.path.realpath(save_path)
        os.makedirs(save_path, exist_ok=True)
        image_file_name = os.path.basename(image_path)
        save_name = os.path.splitext(image_file_name)[0] + self.file_ext
        save_file = os.path.join(save_path, save_name)
        cv2.imwrite(save_file, depth)

        return AutoLabelingResult([], replace=False)

    def unload(self):
        del self.net
