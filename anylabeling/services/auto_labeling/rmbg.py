import os
import cv2
import numpy as np
from PIL import Image

from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel
from . import _THUMBNAIL_RENDER_MODELS


class RMBG(Model):
    """A class for removing backgrounds from images using BRIA RMBG model."""

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
        self.device = "cuda" if __preferred_device__ == "GPU" else "cpu"
        self.model_version = float(self.config.get("version", 1.4))
        assert self.model_version in [
            1.4,
            2.0,
        ], "Only support model versions 1.4 and 2.0"

        if self.device == "cpu" and self.model_version == 2.0:
            logger.warning(
                "⚠️ RMBG model running on CPU will be very slow. Please consider using GPU acceleration for better performance."
            )

        # Set default input shape for different versions
        if self.model_version == 2.0:
            self.input_shape = (1024, 1024)
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        else:
            self.input_shape = self.net.get_input_shape()[-2:]
            self.mean = 0.5
            self.std = 1.0

        self.save_dir, self.file_ext = _THUMBNAIL_RENDER_MODELS["rmbg"]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Preprocessed image.
        """
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=2)
        image = cv2.resize(
            image, self.input_shape, interpolation=cv2.INTER_LINEAR
        )
        image = image.astype(np.float32) / 255.0

        if self.model_version >= 2.0:
            for i in range(3):
                image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
        else:
            image = (image - self.mean) / self.std

        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0)

    def forward(self, blob):
        return self.net.get_ort_inference(blob, extract=True, squeeze=True)

    def postprocess(
        self, result: np.ndarray, original_size: tuple
    ) -> np.ndarray:
        """
        Postprocess the model output.

        Args:
            result (np.ndarray): Model output.
            original_size (tuple): Original image size (height, width).

        Returns:
            np.ndarray: Postprocessed image as a numpy array.
        """
        h, w = original_size
        resize_shape = (w, h)

        result = cv2.resize(
            np.squeeze(result),
            resize_shape,
            interpolation=cv2.INTER_LINEAR,
        )
        max_val, min_val = np.max(result), np.min(result)
        result = (result - min_val) / (max_val - min_val)
        return (result * 255).astype(np.uint8)

    def predict_shapes(self, image, image_path=None):
        """
        Remove the background from an image and save the result.

        Args:
            image (np.ndarray): Input image as a numpy array.
            image_path (str): Path to the input image.
        """
        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        blob = self.preprocess(image)
        output = self.forward(blob)
        result_image = self.postprocess(output, image.shape[:2])

        # Create the final image with transparent background
        pil_mask = Image.fromarray(result_image)
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGBA")
        pil_mask = pil_mask.convert("L")

        # Create a new image with an alpha channel
        output_image = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
        output_image.paste(pil_image, (0, 0), pil_mask)

        # Save the result
        image_dir_path = os.path.dirname(image_path)
        save_path = os.path.join(image_dir_path, "..", self.save_dir)
        save_path = os.path.realpath(save_path)
        os.makedirs(save_path, exist_ok=True)
        image_file_name = os.path.basename(image_path)
        save_name = os.path.splitext(image_file_name)[0] + self.file_ext
        save_file = os.path.join(save_path, save_name)
        output_image.save(save_file)

        return AutoLabelingResult([], replace=False)

    def unload(self):
        del self.net
