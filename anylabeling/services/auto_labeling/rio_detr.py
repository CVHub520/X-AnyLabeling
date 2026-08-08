import os

import numpy as np
from PIL import Image
from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape

from .engines.build_onnx_engine import OnnxBaseModel
from .model import Model
from .types import AutoLabelingResult
from .utils.general import calculate_rotation_theta
from .utils.points_conversion import xywhr2xyxyxyxy


class RiODETR(Model):
    """Oriented object detection model using RiO-DETR."""

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
            "button_classes_filter",
        ]
        output_modes = {
            "rotation": QCoreApplication.translate("Model", "Rotation"),
        }
        default_output_mode = "rotation"

    def __init__(self, model_config, on_message) -> None:
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
        self.input_shape = self.net.get_input_shape()[-2:]
        self.conf_thres = self.config["conf_threshold"]
        self.filter_classes = None
        self.replace = True

    def set_auto_labeling_conf(self, value):
        """Set auto-labeling confidence threshold."""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle preservation of existing annotations."""
        self.replace = not state

    def set_auto_labeling_filter_classes(self, class_names):
        """Set filter classes by name."""
        if not class_names or len(class_names) == len(self.classes):
            self.filter_classes = None
        else:
            self.filter_classes = class_names

    def preprocess(self, input_image):
        """Resize an image with unchanged aspect ratio and bottom-right padding."""
        image_width, image_height = input_image.size
        input_height, input_width = self.input_shape
        ratio = min(input_width / image_width, input_height / image_height)
        resized_width = max(1, int(round(image_width * ratio)))
        resized_height = max(1, int(round(image_height * ratio)))

        resized_image = input_image.resize(
            (resized_width, resized_height), Image.BILINEAR
        )
        padded_image = Image.new("RGB", (input_width, input_height))
        padded_image.paste(resized_image, (0, 0))

        blob = np.asarray(padded_image, dtype=np.float32)
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None] / 255.0)
        orig_size = np.array([[image_height, image_width]], dtype=np.int64)
        return {"images": blob, "orig_target_sizes": orig_size}

    def postprocess(self, outputs):
        """Filter the deployed RiO-DETR outputs by confidence."""
        labels, boxes, scores = outputs
        keep = scores[0] > self.conf_thres
        return boxes[0][keep], scores[0][keep], labels[0][keep]

    def predict_shapes(self, image, image_path=None):
        """Predict oriented bounding-box shapes from an image."""
        if image is None:
            return []

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        inputs = self.preprocess(image)
        outputs = self.net.get_ort_inference(
            None, inputs=inputs, extract=False
        )
        boxes, scores, labels = self.postprocess(outputs)

        shapes = []
        for box, score, label_index in zip(boxes, scores, labels):
            label = self.classes[int(label_index)]
            if self.filter_classes and label not in self.filter_classes:
                continue

            points = xywhr2xyxyxyxy(box)
            shape = Shape(
                label=label,
                score=float(score),
                shape_type="rotation",
                direction=calculate_rotation_theta(points),
            )
            for x, y in points:
                shape.add_point(QtCore.QPointF(float(x), float(y)))
            shape.closed = True
            shapes.append(shape)

        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        del self.net
