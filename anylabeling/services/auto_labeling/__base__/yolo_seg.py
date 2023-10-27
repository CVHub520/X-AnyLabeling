import logging
import os
import math

import cv2
import numpy as np
import onnxruntime as ort
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from ..model import Model
from ..types import AutoLabelingResult


class YOLO_Seg(Model):
    """Object detection model using YOLO_Seg"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "num_masks",
            "score_threshold",
            "nms_threshold",
            "classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize YOLOv8_Seg model."
                )
            )

        self.sess_opts = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])
        self.providers = ['CPUExecutionProvider']
        if __preferred_device__ == "GPU":
            self.providers = ['CUDAExecutionProvider']

        self.num_masks = self.config["num_masks"]
        self.iou_threshold = self.config["nms_threshold"]
        self.conf_threshold = self.config["score_threshold"]
        self.classes = self.config["classes"]
        self.input_height = self.config.get("input_height", -1)
        self.input_width = self.config.get("input_width", -1)

        self.net = ort.InferenceSession(
                        model_abs_path, 
                        providers=self.providers,
                        sess_options=self.sess_opts,
                    )
        self.get_input_nodes()
        self.get_output_nodes()

    def preprocess(self, image):
        self.img_height, self.img_width = image.shape[:2]

        # Resized
        input_img = cv2.resize(image, (self.input_width, self.input_height))

        # Norm
        input_img = input_img / 255.0

        # Transposed
        input_img = input_img.transpose(2, 0, 1)
        
        # Processed
        blob = input_img[np.newaxis, :, :, :].astype(np.float32)

        return blob

    def get_input_nodes(self):
        model_inputs = self.net.get_inputs()
        self.input_names = [
            model_inputs[i].name for i in range(len(model_inputs))
        ]
        self.input_shape = model_inputs[0].shape

        if not isinstance(self.input_shape[0], int):
            self.input_shape[0] = 1
        if isinstance(self.input_shape[2], int):
            self.input_height = self.input_shape[2]
        else:
            self.input_shape[2] = self.input_height
        if isinstance(self.input_shape[3], int):
            self.input_width = self.input_shape[3]
        else:
            self.input_shape[3] = self.input_width

    def get_output_nodes(self):
        model_outputs = self.net.get_outputs()
        self.output_names = [
            model_outputs[i].name for i in range(len(model_outputs))
        ]

    def get_infer_results(self, blob):
        outputs = self.net.run(
            self.output_names, {self.input_names[0]: blob}
        )
        return outputs

    def get_mask_results(self, boxes, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = self.numpy_sigmoid(
            mask_predictions @ mask_output.reshape((num_mask, -1))
        )
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(boxes,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros(
            (len(scale_boxes), self.img_height, self.img_width)
        )
        blur_size = (
            int(self.img_width / mask_width), 
            int(self.img_height / mask_height),
        )

        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(boxes[i][0]))
            y1 = int(math.floor(boxes[i][1]))
            x2 = int(math.ceil(boxes[i][2]))
            y2 = int(math.ceil(boxes[i][3]))

            scale_crop_mask = masks[i][
                scale_y1:scale_y2, scale_x1:scale_x2
            ]
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def get_box_results(self, bboxes):
        """
        This method should be implemented by subclasses to process bounding boxes and
        return specific results based on the provided bounding boxes.

        Args:
            bboxes (list): A list of bounding boxes to be processed.

        Returns:
            results: The specific results based on the input bounding boxes.
        """
        pass

    def postprocess(self, outputs):

        bboxes, masks = outputs[0], outputs[1]
        
        boxes, scores, class_ids, mask_pred = self.get_box_results(bboxes)
        mask_pred = self.get_mask_results(boxes, mask_pred, masks)

        return boxes, scores, class_ids, mask_pred 

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

        blob = self.preprocess(image)
        outputs = self.get_infer_results(blob)
        boxes, _, class_ids, mask_pred = self.postprocess(outputs)

        shapes = []

        # Parse the bounding boxes
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):

            # x1, y1, x2, y2 = box.astype(int)
            # shape = Shape(flags={})
            # shape.add_point(QtCore.QPointF(x1, y1))
            # shape.add_point(QtCore.QPointF(x2, y2))
            # shape.shape_type = "rectangle"
            # shape.closed = True
            # shape.fill_color = "#000000"
            # shape.line_color = "#000000"
            # shape.line_width = 1
            # shape.label = self.classes[class_id]
            # shape.selected = False
            # shapes.append(shape)

            mask = mask_pred[i]
            points = self.get_largest_polygon(mask)
            shape = Shape(flags={})
            for point in points:
                point[0] = int(point[0])
                point[1] = int(point[1])
                shape.add_point(QtCore.QPointF(point[0], point[1]))
            shape.shape_type = "polygon"
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.line_width = 1
            shape.label = str(self.classes[class_id])
            shape.selected = False
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)

        return result

    @staticmethod
    def get_largest_polygon(mask, threshold=0.5):
        # Convert the mask image to binary image
        mask = mask.astype(np.uint8)
        _, binary = cv2.threshold(
            mask, threshold, 255, cv2.THRESH_BINARY
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the points of the largest contour
        points = []
        for point in largest_contour:
            x, y = point[0]
            points.append([x, y])
        
        # Add the first point to the end to ensure a closed polygon
        points.append(points[0])
        
        return points
    
    @staticmethod
    def xywh2xyxy(x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):

        # Rescale boxes to original image dimensions
        input_shape = np.array(
            [input_shape[1], input_shape[0], input_shape[1], input_shape[0]]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [image_shape[1], image_shape[0], image_shape[1], image_shape[0]]
        )

        return boxes

    @staticmethod
    def compute_iou(box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    @staticmethod
    def numpy_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def numpy_nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(
                boxes[box_id, :], boxes[sorted_indices[1:], :]
            )

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def unload(self):
        del self.net