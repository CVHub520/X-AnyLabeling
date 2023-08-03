import logging
import os

import cv2
import numpy as np
import onnxruntime as ort
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult


class YOLOv8(Model):
    """Object detection model using YOLOv8"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "input_width",
            "input_height",
            "score_threshold",
            "nms_threshold",
            "confidence_threshold",
            "classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize YOLOv8 model."
                )
            )

        self.sess_opts = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])
        self.providers = ['CPUExecutionProvider']
        if __preferred_device__ == "GPU":
            self.providers = ['CUDAExecutionProvider']

        self.net = ort.InferenceSession(
                        model_abs_path, 
                        providers=self.providers,
                        sess_options=self.sess_opts,
                    )
        self.classes = self.config["classes"]

    def pre_process(self, input_image, net):
        """
        Pre-process the input image before feeding it to the network.
        """
        # Resized
        input_img = cv2.resize(input_image, (640, 640))

        # Norm
        input_img = input_img / 255.0

        # Transposed
        input_img = input_img.transpose(2, 0, 1)
        
        # Processed
        blob = input_img[np.newaxis, :, :, :].astype(np.float32)

        inputs = net.get_inputs()[0].name
        outputs = net.run(None, {inputs: blob})[0]
        outputs = np.transpose(outputs, (0, 2, 1))

        return outputs

    def post_process(self, input_image, outputs):
        """
        Post-process the network's output, to get the bounding boxes and
        their confidence scores.
        """
        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []

        # Rows.
        rows = outputs.shape[1]

        image_height, image_width = input_image.shape[:2]

        # Resizing factor.
        x_factor = image_width / self.config["input_width"]
        y_factor = image_height / self.config["input_height"]

        # Iterate through 8400 rows.
        for r in range(rows):
            row = outputs[0][r]
            classes_scores = row[4:]

            # Get the index of max class score and confidence.
            _, confidence, _, (_, class_id) = cv2.minMaxLoc(classes_scores)

            # Discard confidence lower than threshold
            if confidence >= self.config["confidence_threshold"]:
                confidences.append(confidence)
                class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                boxes.append(box)

        # Perform non maximum suppression to eliminate redundant
        # overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.config["confidence_threshold"],
            self.config["nms_threshold"],
        )

        output_boxes = []
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            label = self.classes[class_ids[i]]
            score = confidences[i]

            output_box = {
                "x1": left,
                "y1": top,
                "x2": left + width,
                "y2": top + height,
                "label": label,
                "score": score,
            }

            output_boxes.append(output_box)

        return output_boxes

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """

        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        detections = self.pre_process(image, self.net)
        boxes = self.post_process(image, detections)
        shapes = []

        for box in boxes:
            shape = Shape(label=box["label"], shape_type="rectangle", flags={})
            shape.add_point(QtCore.QPointF(box["x1"], box["y1"]))
            shape.add_point(QtCore.QPointF(box["x2"], box["y2"]))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.net