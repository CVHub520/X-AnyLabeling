import os
import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel
from .utils import xywh2xyxy


class YOLOv5CarPlateDetRec(Model):
    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "det_model_path",
            "rec_model_path",
            "names",
            "classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "point": QCoreApplication.translate("Model", "Point"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)

        det_model_abs_path = self.get_model_abs_path(
            self.config, "det_model_path"
        )
        if not det_model_abs_path or not os.path.isfile(det_model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize YOLOv5CarPlate Detection model.",
                )
            )
        self.det_net = OnnxBaseModel(det_model_abs_path, __preferred_device__)
        rec_model_abs_path = self.get_model_abs_path(
            self.config, "rec_model_path"
        )
        if not rec_model_abs_path or not os.path.isfile(rec_model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize YOLOv5CarPlate Recognition model.",
                )
            )
        self.rec_net = OnnxBaseModel(rec_model_abs_path, __preferred_device__)

        self.std = self.config.get("std", 0.193)
        self.mean = self.config.get("mean", 0.588)
        self.iou_thres = self.config.get("iou_thres", 0.5)
        self.conf_thres = self.config.get("conf_thres", 0.3)
        self.names = self.config["names"]
        self.classes = self.config["classes"]

        _, _, det_h, det_w = self.det_net.get_input_shape()
        self.det_input_shape = (det_h, det_w)
        _, _, rec_h, rec_w = self.rec_net.get_input_shape()
        self.rec_input_shape = (rec_h, rec_w)

    def preprocess(self, image):
        img, r, left, top = self._letterbox(image, self.det_input_shape)
        img = img.transpose(2, 0, 1).astype(np.float32)
        img /= 255.0
        img = img.reshape(1, *img.shape)
        return img, r, left, top

    def postprocess(self, dets, r, left, top):
        choice = dets[:, :, 4] > self.conf_thres
        dets = dets[choice]
        dets[:, 13:15] *= dets[:, 4:5]
        box = dets[:, :4]
        boxes = xywh2xyxy(box)
        score = np.max(dets[:, 13:15], axis=-1, keepdims=True)
        index = np.argmax(dets[:, 13:15], axis=-1).reshape(-1, 1)
        output = np.concatenate((boxes, score, dets[:, 5:13], index), axis=1)
        reserve_ = self._nms(output, self.iou_thres)
        output = output[reserve_]
        output = self.restore_box(output, r, left, top)
        return output

    def rec_pre_processing(self, img, size=(168, 48)):
        # Preprocessing before recognition
        img = cv2.resize(img, size)
        img = img.astype(np.float32)
        img = (img / 255 - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = img.reshape(1, *img.shape)
        return img

    def get_plate_result(self, img):
        blob = self.rec_pre_processing(img)
        y_onnx_plate, y_onnx_color = self.rec_net.get_ort_inference(
            blob, extract=False
        )
        index = np.argmax(y_onnx_plate, axis=-1)
        index_color = np.argmax(y_onnx_color)
        plate_color = self.classes[index_color]
        plate_no = self.decodePlate(index[0])
        return plate_no, plate_color

    def rec_plate(self, outputs, img0):
        # Recognize license plates
        dict_list = []

        for output in outputs:
            result_dict = {}
            rect = output[:4].tolist()
            landmarks = output[5:13].reshape(4, 2)
            roi_img = self.four_point_transform(img0, landmarks)
            label = int(output[-1])
            score = output[4]

            if label == 1:  # Represents a double-layer license plate
                roi_img = self.get_split_merge(roi_img)

            plate_no, plate_color = self.get_plate_result(roi_img)

            result_dict["rect"] = rect
            result_dict["score"] = score
            result_dict["landmarks"] = landmarks.tolist()
            result_dict["plate_no"] = plate_no
            result_dict["roi_height"] = roi_img.shape[0]
            result_dict["plate_color"] = plate_color

            dict_list.append(result_dict)

        return dict_list

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

        blob, r, left, top = self.preprocess(image)
        predictions = self.det_net.get_ort_inference(blob)
        outputs = self.postprocess(predictions, r, left, top)
        results = self.rec_plate(outputs, image)
        shapes = []
        for i, result in enumerate(results):
            label = result["plate_color"]
            landmarks = result["landmarks"]
            x1, y1, x2, y2 = list(map(int, result["rect"]))
            shape = Shape(label=label, shape_type="rectangle", group_id=int(i))
            shape.add_point(QtCore.QPointF(x1, y1))
            shape.add_point(QtCore.QPointF(x2, y1))
            shape.add_point(QtCore.QPointF(x2, y2))
            shape.add_point(QtCore.QPointF(x1, y2))
            shape.description = result["plate_no"]
            shapes.append(shape)
            labels = ["tl", "tr", "bl", "br"]
            for j, label in enumerate(labels):
                point = landmarks[j]
                shape = Shape(label=label, shape_type="point", group_id=int(i))
                shape.add_point(QtCore.QPointF(point[0], point[1]))
                shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)

        return result

    def four_point_transform(self, image, pts):
        # Perspective transformation to obtain a corrected image for easier recognition
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

    def decodePlate(self, preds):
        # Post-processing after recognition
        pre = 0
        new_preds = []

        # Filter out repeated and zero predictions
        for i in range(len(preds)):
            if preds[i] != 0 and preds[i] != pre:
                new_preds.append(preds[i])
            pre = preds[i]

        plate = ""

        # Decode the plate using the corresponding characters from plateName
        for i in new_preds:
            plate += self.names[int(i)]

        return plate

    @staticmethod
    def get_split_merge(img):
        # Recognition after segmentation for double-layer license plates
        h, w, c = img.shape

        # Extract upper and lower portions of the image
        img_upper = img[0 : int(5 / 12 * h), :]
        img_lower = img[int(1 / 3 * h) :, :]

        # Resize the upper portion to match the lower portion's size
        img_upper = cv2.resize(
            img_upper, (img_lower.shape[1], img_lower.shape[0])
        )

        # Concatenate the resized upper portion and the lower portion
        new_img = np.hstack((img_upper, img_lower))

        return new_img

    @staticmethod
    def order_points(pts):
        # Order the points (top-left, top-right, bottom-right, bottom-left)
        rect = np.zeros((4, 2), dtype="float32")

        # Calculate the sum of coordinates for each point
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left point has the smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right point has the largest sum

        # Calculate the difference between coordinates for each point
        diff = np.diff(pts, axis=1)
        rect[1] = pts[
            np.argmin(diff)
        ]  # Top-right point has the smallest difference
        rect[3] = pts[
            np.argmax(diff)
        ]  # Bottom-left point has the largest difference

        return rect

    @staticmethod
    def restore_box(boxes, r, left, top):
        boxes[:, [0, 2, 5, 7, 9, 11]] -= left
        boxes[:, [1, 3, 6, 8, 10, 12]] -= top

        boxes[:, [0, 2, 5, 7, 9, 11]] /= r
        boxes[:, [1, 3, 6, 8, 10, 12]] /= r

        return boxes

    @staticmethod
    def _nms(boxes, iou_thres):
        index = np.argsort(boxes[:, 4])[::-1]
        keep = []

        while index.size > 0:
            i = index[0]
            keep.append(i)

            x1 = np.maximum(boxes[i, 0], boxes[index[1:], 0])
            y1 = np.maximum(boxes[i, 1], boxes[index[1:], 1])
            x2 = np.minimum(boxes[i, 2], boxes[index[1:], 2])
            y2 = np.minimum(boxes[i, 3], boxes[index[1:], 3])

            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)

            inter_area = w * h
            union_area = (boxes[i, 2] - boxes[i, 0]) * (
                boxes[i, 3] - boxes[i, 1]
            ) + (boxes[index[1:], 2] - boxes[index[1:], 0]) * (
                boxes[index[1:], 3] - boxes[index[1:], 1]
            )
            iou = inter_area / (union_area - inter_area)
            idx = np.where(iou <= iou_thres)[0]
            index = index[idx + 1]

        return keep

    @staticmethod
    def _letterbox(img, size=(640, 640)):
        h, w, _ = img.shape
        r = min(size[0] / h, size[1] / w)
        new_h, new_w = int(h * r), int(w * r)
        top = int((size[0] - new_h) / 2)
        left = int((size[1] - new_w) / 2)
        bottom = size[0] - new_h - top
        right = size[1] - new_w - left
        img_resize = cv2.resize(img, (new_w, new_h))
        img = cv2.copyMakeBorder(
            img_resize,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return img, r, left, top

    def unload(self):
        del self.det_net
        del self.rec_net
