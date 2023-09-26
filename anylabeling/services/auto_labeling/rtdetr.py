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


class RTDETR(Model):
    """Object detection model using RTDETR"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "input_width",
            "input_height",
            "score_threshold",
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
                    "Model", "Could not download or initialize RTDETR model."
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

    @staticmethod
    def bbox_cxcywh_to_xyxy(boxes):

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        return np.stack([x1, y1, x2, y2], axis=1)

    def pre_process(self, input_image):
        """
        Pre-processes the input image before feeding it to the network.
        
        Args:
            input_image (numpy.ndarray): The input image to be processed.
        
        Returns:
            numpy.ndarray: The pre-processed output.
        """
        # Get the image width and height
        image_h, image_w = input_image.shape[:2]

        input_h, input_w = self.config['input_height'], self.config['input_width']

        # Compute the scaling factors
        ratio_h = input_h / image_h
        ratio_w = input_w / image_w

        # Perform the pre-processing steps
        img = cv2.resize(input_image, (0, 0), fx=ratio_w, fy=ratio_h, interpolation=2)
        img = img[:, :, ::-1] / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        blob = np.ascontiguousarray(img, dtype=np.float32)

        outs = self.net.run(None, {'image': blob})[0][0]

        return outs

    def post_process(self, input_image, outputs):
        """
        Post-processes the network's output to obtain the bounding boxes and their confidence scores.
        
        Args:
            input_image (numpy.ndarray): The input image.
            outputs (numpy.ndarray): The output from the network.
        
        Returns:
            list: List of dictionaries containing the output bounding boxes, labels, and scores.
        """
        image_height, image_width = input_image.shape[:2]

        boxes, scores = outputs[:, :4], outputs[:, 4:]

        # Normalize scores if they are not already in the range (0, 1)
        if not (np.all((scores > 0) & (scores < 1))):
            scores = 1 / (1 + np.exp(-scores))

        boxes = self.bbox_cxcywh_to_xyxy(boxes)
        _max = scores.max(-1)
        _mask = _max > self.config['score_threshold']
        boxes, scores = boxes[_mask], scores[_mask]
        indexs, scores = scores.argmax(-1), scores.max(-1)

        # Normalize the bounding box coordinates
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = np.floor(np.minimum(np.maximum(1, x1 * image_width), image_width - 1)).astype(int)
        y1 = np.floor(np.minimum(np.maximum(1, y1 * image_height), image_height - 1)).astype(int)
        x2 = np.ceil(np.minimum(np.maximum(1, x2 * image_width), image_width - 1)).astype(int)
        y2 = np.ceil(np.minimum(np.maximum(1, y2 * image_height), image_height - 1)).astype(int)
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        output_boxes = []
        for box, index, score in zip(boxes, indexs, scores):
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            label = self.classes[index]

            output_box = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
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

        detections = self.pre_process(image)
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
