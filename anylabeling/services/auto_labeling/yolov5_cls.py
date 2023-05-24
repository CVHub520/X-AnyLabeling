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


class YOLOv5_CLS(Model):
    """Object detection with Classify model using YOLOv5_CLS"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "det_model_path",
            "cls_model_path",
            "det_input_width",
            "det_input_height",
            "cls_input_width",
            "cls_input_height",
            "det_score_threshold",
            "cls_score_threshold",
            "nms_threshold",
            "confidence_threshold",
            "det_classes",
            "cls_classes"
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        det_model_abs_path = self.get_model_abs_path(self.config, "det_model_path")
        if not det_model_abs_path or not os.path.isfile(det_model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize YOLOv5 model."
                )
            )
        cls_model_abs_path = self.get_model_abs_path(self.config, "cls_model_path")
        if not cls_model_abs_path or not os.path.isfile(cls_model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize ResNet50 model."
                )
            )

        self.det_net = cv2.dnn.readNet(det_model_abs_path)
        self.sess_opts = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])
        self.providers = ['CPUExecutionProvider']

        if __preferred_device__ == "GPU":
            self.det_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.det_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.providers = ['CUDAExecutionProvider']

        self.cls_net = ort.InferenceSession(
                        cls_model_abs_path, 
                        providers=self.providers,
                        sess_options=self.sess_opts,
                    )

        self.det_classes = self.config["det_classes"]
        self.cls_classes = self.config["cls_classes"]

    def det_pre_process(self, input_image, net):
        """Detection
        Pre-process the input image before feeding it to the network.
        """
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(
            input_image,
            1 / 255,
            (self.config["det_input_width"], self.config["det_input_height"]),
            [0, 0, 0],
            1,
            crop=False,
        )

        # Sets the input to the network.
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers.
        output_layers = net.getUnconnectedOutLayersNames()
        outputs = net.forward(output_layers)

        return outputs

    def det_post_process(self, input_image, outputs):
        """Detection
        Post-process the network's output, to get the bounding boxes and
        their confidence scores.
        """
        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []

        # Rows.
        rows = outputs[0].shape[1]

        image_height, image_width = input_image.shape[:2]

        # Resizing factor.
        x_factor = image_width / self.config["det_input_width"]
        y_factor = image_height / self.config["det_input_height"]

        # Iterate through 25200 detections.
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]

            # Discard bad detections and continue.
            if confidence >= self.config["confidence_threshold"]:
                classes_scores = row[5:]

                # Get the index of max class score.
                class_id = np.argmax(classes_scores)

                #  Continue if the class score is above threshold.
                if classes_scores[class_id] > self.config["det_score_threshold"]:
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
            label = self.det_classes[class_ids[i]]
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

    def cls_pre_process(self, input_image, mean=None, std=None, swap=(2, 0, 1)):
        """
        Classification: Pre-processes the input image before feeding it to the network.
        
        Args:
            input_image (numpy.ndarray): The input image to be processed.
            mean (numpy.ndarray): Mean values for normalization. If not provided, default values are used.
            std (numpy.ndarray): Standard deviation values for normalization. If not provided, default values are used.
            swap (tuple): Order of color channels. Default is (2, 0, 1).
        
        Returns:
            numpy.ndarray: The processed input image.
        """
        img_width = self.config["cls_input_width"]
        img_height = self.config["cls_input_height"]
        
        # Resize the input image
        input_data = cv2.resize(input_image, (img_width, img_height))
        
        # Transpose the dimensions of the image according to the specified order of color channels
        input_data = input_data.transpose(swap)
        
        if not mean:
            mean = np.array([0.485, 0.456, 0.406])
        
        if not std:
            std = np.array([0.229, 0.224, 0.225])
        
        norm_img_data = np.zeros(input_data.shape).astype('float32')
        
        # Normalize the image data
        for channel in range(input_data.shape[0]):
            norm_img_data[channel, :, :] = (input_data[channel, :, :] / 255 - mean[channel]) / std[channel]
        
        blob = norm_img_data.reshape(1, 3, img_height, img_width).astype('float32')
        
        inputs = self.cls_net.get_inputs()[0].name
        
        # Pass the pre-processed image through the network
        outputs = self.cls_net.run(None, {inputs: blob})

        return outputs

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

    def cls_post_process(self, outs, topk=1):
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
        label = self.cls_classes[index]

        return label

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

        detections = self.det_pre_process(image, self.det_net)
        boxes = self.det_post_process(image, detections)

        shapes = []

        for box in boxes:
            x1 = int(box["x1"])
            y1 = int(box["y1"])
            x2 = int(box["x2"])
            y2 = int(box["y2"])
            img = image[y1: y2, x1: x2]

            outs = self.cls_pre_process(img)
            label = self.cls_post_process(outs)

            box["label"] = label
            shape = Shape(label=box["label"], shape_type="rectangle", flags={})
            shape.add_point(QtCore.QPointF(box["x1"], box["y1"]))
            shape.add_point(QtCore.QPointF(box["x2"], box["y2"]))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.cls_net
        del self.det_net
