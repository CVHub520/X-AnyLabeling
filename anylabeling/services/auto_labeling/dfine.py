import os
import numpy as np
from PIL import Image

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel


class DFINE(Model):
    """Object detection model using DFINE"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "conf_threshold",
            "classes",
        ]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "toggle_preserve_existing_annotations",
        ]
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
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]
        self.input_shape = (640, 640)
        self.conf_thres = self.config["conf_threshold"]
        self.replace = True

    def set_auto_labeling_conf(self, value):
        """set auto labeling confidence threshold"""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def preprocess(self, input_image, interpolation=Image.BILINEAR):
        """
        Preprocesses the input image by resizing while preserving aspect ratio and adding padding.

        Args:
            input_image (PIL.Image): The input image to be processed.
            interpolation: The interpolation method to use when resizing. Defaults to PIL.Image.BILINEAR.

        Returns:
            tuple: A tuple containing:
                - blob (np.ndarray): The preprocessed image as a normalized numpy array in NCHW format
                - orig_size (np.ndarray): Original image size as [[height, width]]
                - ratio (float): The resize ratio used
                - pad_w (int): The horizontal padding value
                - pad_h (int): The vertical padding value
        """
        image_w, image_h = input_image.size
        input_h, input_w = self.input_shape

        ratio = min(input_w / image_w, input_h / image_h)
        new_width = int(image_w * ratio)
        new_height = int(image_h * ratio)

        image = input_image.resize((new_width, new_height), interpolation)

        # Create a new image with the desired size and paste the resized image onto it
        new_image = Image.new("RGB", (input_w, input_h))

        pad_h, pad_w = (input_h - new_height) // 2, (input_w - new_width) // 2
        new_image.paste(image, (pad_w, pad_h))

        orig_size = np.array(
            [[new_image.size[1], new_image.size[0]]], dtype=np.int64
        )
        im_data = np.array(new_image).astype(np.float32) / 255.0
        im_data = im_data.transpose(2, 0, 1)
        blob = np.expand_dims(im_data, axis=0)

        return blob, orig_size, ratio, pad_w, pad_h

    def postprocess(self, outputs, orig_size, ratio, padding):
        """
        Post-processes the network's output.

        Args:
            outputs (list): The outputs from the network.
            orig_size (int, int): Original image size (img_w, img_h).
            ratio (float): The resize ratio.
            padding (tuple): Padding info (pad_w, pad_h).

        Returns:
            list: List of dictionaries containing the output boxes, labels, and scores.
        """
        labels, boxes, scores = outputs

        pad_w, pad_h = padding
        ori_w, ori_h = orig_size

        results = []

        # Only process boxes with scores above threshold
        for i, score in enumerate(scores[0]):
            if score > self.conf_thres:
                label_idx = int(labels[0][i])
                if label_idx < len(self.classes):
                    label = self.classes[label_idx]
                else:
                    label = str(label_idx)

                # Get box coordinates and adjust for padding and resize
                box = boxes[0][i]
                x1 = int((box[0] - pad_w) / ratio)
                y1 = int((box[1] - pad_h) / ratio)
                x2 = int((box[2] - pad_w) / ratio)
                y2 = int((box[3] - pad_h) / ratio)

                # Clip coordinates to image boundaries
                x1 = max(0, min(x1, ori_w))
                y1 = max(0, min(y1, ori_h))
                x2 = max(0, min(x2, ori_w))
                y2 = max(0, min(y2, ori_h))

                result = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": label,
                    "score": float(score),
                }
                results.append(result)

        return results

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """
        if image is None:
            return []

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        # Preprocess image
        blob, orig_size, ratio, pad_w, pad_h = self.preprocess(image)

        # Run inference
        inputs = {"images": blob, "orig_target_sizes": orig_size}
        outputs = self.net.get_ort_inference(
            blob, inputs=inputs, extract=False
        )

        # Process outputs
        results = self.postprocess(outputs, image.size, ratio, (pad_w, pad_h))

        shapes = []
        for result in results:
            xmin = result["x1"]
            ymin = result["y1"]
            xmax = result["x2"]
            ymax = result["y2"]
            shape = Shape(
                label=result["label"],
                score=result["score"],
                shape_type="rectangle",
            )
            shape.add_point(QtCore.QPointF(xmin, ymin))
            shape.add_point(QtCore.QPointF(xmax, ymin))
            shape.add_point(QtCore.QPointF(xmax, ymax))
            shape.add_point(QtCore.QPointF(xmin, ymax))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=self.replace)
        return result

    def unload(self):
        del self.net
