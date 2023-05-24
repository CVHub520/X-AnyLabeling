import logging
import os

import cv2
import math
import numpy as np
import onnxruntime as ort
from numpy import array
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult


class YOLOv6(Model):
    """Object detection model using YOLOv6 v4.0"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "input_width",
            "input_height",
            "stride",
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
                    "Model", "Could not download or initialize YOLOv6 model."
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
        self.img_size = self.check_img_size(
            [self.config["input_width"], self.config["input_height"]], 
            s=self.config["stride"]
        )

    def pre_process(self, input_image, net):
        """
        Pre-process the input RGB image before feeding it to the network.
        """
        image = self.letterbox(input_image, self.img_size, stride=self.config['stride'])[0]
        image = image.transpose((2, 0, 1)) # HWC to CHW
        image = np.ascontiguousarray(image).astype('float32')
        image /= 255  # 0 - 255 to 0.0 - 1.0
        if len(image.shape) == 3:
            image = image[None]
        inputs = net.get_inputs()[0].name
        outputs = net.run(None, {inputs: image})[0]

        return image, outputs

    def post_process(self, img_src, img_processed, outputs):
        """
        Post-process the network's output, to get the bounding boxes, key-points and
        their confidence scores.
        """
        det = self.non_max_suppression(outputs)[0]
        output_infos = []
        if len(det):
            det[:, :4] = self.rescale(img_processed.shape[2:], det[:, :4], img_src.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = xyxy
                output_info = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": self.classes[int(cls)],
                    "score": conf,
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

        processed_img, detections = self.pre_process(image, self.net)
        infos = self.post_process(image, processed_img, detections)

        shapes = []
        for info in infos:
            rectangle_shape = Shape(label=info["label"], shape_type="rectangle", flags={})
            rectangle_shape.add_point(QtCore.QPointF(info["x1"], info["y1"]))
            rectangle_shape.add_point(QtCore.QPointF(info["x2"], info["y2"]))
            shapes.append(rectangle_shape)

        result = AutoLabelingResult(shapes, replace=True)

        return result

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

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

    @staticmethod
    def xywh2xyxy(x):
        '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = ((ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2)
        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio
        boxes[:, 0] = np.clip(boxes[:, 0], 0, target_shape[1])  # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, target_shape[0])  # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, target_shape[1])  # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, target_shape[0])  # y2
        return boxes

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=1000):
        """Runs Non-Maximum Suppression (NMS) on inference results.
        This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
        Args:
            prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
            conf_thres: (float) confidence threshold.
            iou_thres: (float) iou threshold.
            classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
            agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
            multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
            max_det:(int), max number of output bboxes.

        Returns:
            list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
        """
        conf_thres = self.config["confidence_threshold"]
        iou_thres = self.config["nms_threshold"]

        num_classes = prediction.shape[2] - 5  # number of classes
        pred_candidates = np.logical_and(prediction[..., 4] > conf_thres, np.max(prediction[..., 5:], axis=-1) > conf_thres)  # candidates
        # Check the parameters.
        assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
        assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

        # Function settings.
        max_wh = 4096  # maximum box width and height
        max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
        multi_label &= num_classes > 1  # multiple labels per box

        output = [np.zeros((0, 6))] * prediction.shape[0]
        for img_idx, x in enumerate(prediction):  # image index, image inference
            x = x[pred_candidates[img_idx]]  # confidence

            # If no box remains, skip the next process.
            if not x.shape[0]:
                continue

            # confidence multiply the objectness
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
            if multi_label:
                box_idx, class_idx = np.nonzero(x[:, 5:] > conf_thres)
                box = box[box_idx]
                conf = x[box_idx, class_idx + 5][:, None]
                class_idx = class_idx[:, None].astype(float)
                x = np.concatenate((box, conf, class_idx), axis=1)
            else:
                conf = np.max(x[:, 5:], axis=1, keepdims=True)
                class_idx = np.argmax(x[:, 5:], axis=1)
                x = np.concatenate((box, conf, class_idx[:, None].astype(float)), axis=1)[conf.flatten() > conf_thres]

            # Filter by class, only keep boxes whose category is in classes.
            if classes is not None:
                x = x[(x[:, 5:6] == np.array(classes)).any(1)]

            # Check shape
            num_box = x.shape[0]  # number of boxes
            if not num_box:  # no boxes kept.
                continue
            elif num_box > max_nms:  # excess max boxes' number.
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
            keep_box_idx = self.numpy_nms(boxes, scores, iou_thres)  # NMS
            if keep_box_idx.shape[0] > max_det:  # limit detections
                keep_box_idx = keep_box_idx[:max_det]

            output[img_idx] = x[keep_box_idx]

        return output

    @staticmethod
    def box_area(boxes :array):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def box_iou(self, box1 :array, box2: array):
        area1 = self.box_area(box1)  # N
        area2 = self.box_area(box2)  # M
        # broadcasting
        lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
        rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
        wh = rb - lt
        wh = np.maximum(0, wh) # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]
        iou = inter / (area1[:, np.newaxis] + area2 - inter)
        return iou  # NxM

    def numpy_nms(self, boxes :array, scores :array, iou_threshold :float):
        idxs = scores.argsort()
        keep = []
        while idxs.size > 0:
            max_score_index = idxs[-1]
            max_score_box = boxes[max_score_index][None, :]
            keep.append(max_score_index)
            if idxs.size == 1:
                break
            idxs = idxs[:-1]
            other_boxes = boxes[idxs]
            ious = self.box_iou(max_score_box, other_boxes)
            idxs = idxs[ious[0] <= iou_threshold]
        keep = np.array(keep)  
        return keep

    def unload(self):
        del self.net
