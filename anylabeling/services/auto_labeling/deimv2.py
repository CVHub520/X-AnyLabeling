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


class DEIMv2(Model):
    """Object detection model using DEIMv2"""

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
        self.imgsz = self.config.get("img_size", 640)
        self.conf_thres = self.config.get("conf_threshold", 0.40)
        self.replace = True

    def set_auto_labeling_conf(self, value):
        """set auto labeling confidence threshold"""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def preprocess(self, image_path):
        """Preprocess image for model inference."""
        img_size = self.imgsz
        image = Image.open(image_path).convert("RGB")
        orig_size = (image.height, image.width)

        # Resize with aspect ratio preservation
        ratio = min(img_size / image.width, img_size / image.height)
        new_w, new_h = int(image.width * ratio), int(image.height * ratio)
        resized = image.resize((new_w, new_h), Image.BILINEAR)

        # Pad to square
        padded = Image.new("RGB", (img_size, img_size), (0, 0, 0))
        pad_w, pad_h = (img_size - new_w) // 2, (img_size - new_h) // 2
        padded.paste(resized, (pad_w, pad_h))

        # To tensor
        img_array = np.array(padded, dtype=np.float32).transpose(2, 0, 1)
        img_tensor = np.ascontiguousarray(img_array[None, :, :, :] / 255.0)

        return img_tensor, orig_size, ratio, (pad_w, pad_h)

    def postprocess(self, outputs, orig_size, ratio, padding):
        """Postprocess model outputs."""
        labels, boxes, scores = outputs

        # Filter by confidence
        mask = scores[0] >= self.conf_thres
        labels = labels[0][mask].astype(int)
        boxes = boxes[0][mask]
        scores = scores[0][mask]

        # Rescale boxes to original image size
        if len(boxes) > 0:
            pad_w, pad_h = padding
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio

            # Clip to image boundaries
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_size[1])
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_size[0])
            boxes = boxes.astype(int)

        return labels, boxes, scores

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image_path is None:
            logger.warning("Image path is None. Could not inference model.")
            return []

        # Preprocess
        img_tensor, orig_size, ratio, padding = self.preprocess(image_path)

        # Inference
        orig_target_sizes = np.array(
            [[self.imgsz, self.imgsz]], dtype=np.int64
        )
        inputs = {"images": img_tensor, "orig_target_sizes": orig_target_sizes}
        detections = self.net.get_ort_inference(
            inputs=inputs, extract=False, squeeze=False
        )

        # Postprocess
        labels, boxes, scores = self.postprocess(
            detections, orig_size, ratio, padding
        )

        shapes = []
        for box, score, label in zip(boxes, scores, labels):
            xmin = float(box[0])
            ymin = float(box[1])
            xmax = float(box[2])
            ymax = float(box[3])
            shape = Shape(
                label=self.classes[int(label)],
                score=float(score),
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
