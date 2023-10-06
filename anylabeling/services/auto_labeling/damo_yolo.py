import logging
import os
import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult

from .engines.build_onnx_engine import OnnxBaseModel


class DAMO_YOLO(Model):
    """Object detection model using DAMO_YOLO"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
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
                    "Model", "Could not download or initialize DAMO_YOLO model."
                )
            )

        self.net = OnnxBaseModel(self.config["model_path"], __preferred_device__)
        self.classes = self.config["classes"]
        self.input_shape = self.net.get_input_shape()
        self.input_size = self.input_shape[-2:]
        self.nms_thres = self.config["nms_threshold"]
        self.conf_thres = self.config["confidence_threshold"]

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32, return_int=False):
        '''Resize and pad image while meeting stride-multiple constraints.'''
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        elif isinstance(new_shape, list) and len(new_shape) == 1:
            new_shape = (new_shape[0], new_shape[0])

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        if not return_int:
            return im, r, (dw, dh)
        else:
            return im, r, (left, top)

    def preprocess(self, input_image):
        src_h, src_w, _ = input_image.shape
        _, dst_c, dst_h, dst_w = self.input_shape
        transformed_image = np.ones((dst_h, dst_w, dst_c), dtype=np.uint8)
        ratio_hw = min(dst_h / src_h, dst_w / src_w)
        new_h, new_w = int(ratio_hw * src_h), int(ratio_hw * src_w)
        image = cv2.resize(input_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        transformed_image[:new_h, :new_w, :] = image
        transformed_image = transformed_image.transpose((2, 0, 1))
        transformed_image = np.ascontiguousarray(transformed_image).astype('float32')
        if len(transformed_image.shape) == 3:
            transformed_image = transformed_image[None]
        return transformed_image, ratio_hw

    def postprocess(self, predictions, ratio_hw):
        scores = predictions[0].squeeze(axis=0)
        bboxes = predictions[1].squeeze(axis=0)
        bboxes /= ratio_hw

        boxes, confidences, class_ids = [], [], []
        for i in range(len(bboxes)):
            score = np.max(scores[i, :])
            if score < self.conf_thres:
                continue
            class_id = np.argmax(scores[i, :])
            xmin, ymin, xmax, ymax = bboxes[i, :].astype(np.int32)
            width = xmax - xmin
            height = ymax - ymin
            boxes.append([xmin, ymin, width, height])
            confidences.append(score)
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.nms_thres)
        output_infos = []
        for i in indices:
            x, y, w, h = boxes[i]
            output_info = {
                "xmin": x,
                "ymin": y,
                "xmax": x + w,
                "ymax": y + h,
                "label": self.classes[int(class_ids[i])],
            }
            output_infos.append(output_info)
        
        return output_infos

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

        blob, ratio_hw = self.preprocess(image)
        predictions = self.net.get_ort_inference(blob, extract=False)
        results = self.postprocess(predictions, ratio_hw)

        shapes = []
        for result in results:
            rectangle_shape = Shape(label=result["label"], shape_type="rectangle", flags={})
            rectangle_shape.add_point(QtCore.QPointF(result["xmin"], result["ymin"]))
            rectangle_shape.add_point(QtCore.QPointF(result["xmax"], result["ymax"]))
            shapes.append(rectangle_shape)
        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.net

