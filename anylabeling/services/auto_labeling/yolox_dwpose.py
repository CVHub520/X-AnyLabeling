import os
import cv2
import numpy as np
import onnxruntime as ort
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .pose.dwpose_onnx import inference_pose


class YOLOX_DWPose(Model):
    """Effective Whole-body Pose Estimation with Two-stages Distillation"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "det_model_path",
            "pose_model_path",
            "det_input_width",
            "det_input_height",
            "pose_input_width",
            "pose_input_height",
            "p6",
            "score_threshold",
            "nms_threshold",
            "confidence_threshold",
            "kpt_threshold",
            "det_cat_ids",
            "det_classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            "point": QCoreApplication.translate("Model", "Point"),
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
                    "Model", "Could not download or initialize YOLOX-L model."
                )
            )
        pose_model_abs_path = self.get_model_abs_path(
            self.config, "pose_model_path"
        )
        if not pose_model_abs_path or not os.path.isfile(pose_model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", "Could not download or initialize DWPose model."
                )
            )

        sess_opts = ort.SessionOptions()
        if "OMP_NUM_THREADS" in os.environ:
            sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

        if __preferred_device__ == "GPU":
            ox_providers = ["CUDAExecutionProvider"]
            backend = cv2.dnn.DNN_BACKEND_CUDA
            cv_providers = cv2.dnn.DNN_TARGET_CUDA
        else:
            ox_providers = ["CPUExecutionProvider"]
            backend = cv2.dnn.DNN_BACKEND_OPENCV
            cv_providers = cv2.dnn.DNN_TARGET_CPU

        self.det_net = ort.InferenceSession(
            det_model_abs_path,
            providers=ox_providers,
            sess_options=sess_opts,
        )
        self.pose_net = cv2.dnn.readNetFromONNX(pose_model_abs_path)
        self.pose_net.setPreferableBackend(backend)
        self.pose_net.setPreferableTarget(cv_providers)

        self.p6 = self.config["p6"]
        self.draw_det_box = self.config["draw_det_box"]
        self.kpt_thr = self.config["kpt_threshold"]
        self.det_cat_ids = self.config["det_cat_ids"]
        self.det_classes = self.config["det_classes"]
        self.det_input_size = (
            self.config["det_input_height"],
            self.config["det_input_width"],
        )
        self.pose_input_size = (
            self.config["pose_input_width"],
            self.config["pose_input_height"],
        )

    def det_pre_process(self, img, net, swap=(2, 0, 1)):
        """
        Pre-process the input RGB image before feeding it to the network.
        """
        if len(img.shape) == 3:
            padded_img = (
                np.ones(
                    (self.det_input_size[0], self.det_input_size[1], 3),
                    dtype=np.uint8,
                )
                * 114
            )
        else:
            padded_img = np.ones(self.det_input_size, dtype=np.uint8) * 114

        r = min(
            self.det_input_size[0] / img.shape[0],
            self.det_input_size[1] / img.shape[1],
        )
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = (
            resized_img
        )

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

        ort_inputs = {net.get_inputs()[0].name: padded_img[None, :, :, :]}
        outputs = net.run(None, ort_inputs)

        return r, outputs

    def det_post_process(self, outputs):
        """
        Post-process the network's output, to get the bounding boxes, key-points and
        their confidence scores.
        """
        grids = []
        expanded_strides = []
        p6 = self.p6
        img_size = self.det_input_size
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

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

        # Object Detection
        ratio, outputs = self.det_pre_process(image, self.det_net)
        predictions = self.det_post_process(outputs[0])[0]
        results = self.det_rescale(predictions, ratio)

        # Pose Estimation
        if results is not None:
            final_boxes, final_scores, final_cls_inds = (
                results[:, :4],
                results[:, 4],
                results[:, 5],
            )
            isscore = final_scores > self.config["confidence_threshold"]
            iscat = final_cls_inds == self.det_cat_ids
            isbbox = [i and j for (i, j) in zip(isscore, iscat)]
            final_boxes = final_boxes[isbbox]

            keypoints, scores = inference_pose(
                self.pose_net,
                final_boxes,
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                self.pose_input_size,
            )
            keypoints, scores = self.pose_rescale(keypoints, scores)

        # Output
        shapes = []
        for box, _, cls_inds, kpt_points, kpt_scores in zip(
            final_boxes, final_scores, final_cls_inds, keypoints, scores
        ):
            if self.draw_det_box:
                x1, y1, x2, y2 = box
                rectangle_shape = Shape(
                    label=str(self.det_classes[int(cls_inds)]),
                    shape_type="rectangle",
                )
                rectangle_shape.add_point(QtCore.QPointF(x1, y1))
                rectangle_shape.add_point(QtCore.QPointF(x2, y1))
                rectangle_shape.add_point(QtCore.QPointF(x2, y2))
                rectangle_shape.add_point(QtCore.QPointF(x1, y2))
                shapes.append(rectangle_shape)

            num_kpts = len(kpt_scores)
            for i in range(num_kpts):
                kpt_point, kpt_score = kpt_points[i], kpt_scores[i]
                if kpt_score <= self.kpt_thr:
                    continue
                point_shape = Shape(label=str(i), shape_type="point", flags={})
                point_shape.add_point(
                    QtCore.QPointF(kpt_point[0], kpt_point[1])
                )
                shapes.append(point_shape)

        result = AutoLabelingResult(shapes, replace=True)

        return result

    def pose_rescale(self, keypoints, scores):
        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1
        )
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3
        ).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]
        return keypoints, scores

    def det_rescale(self, predictions, ratio):
        """Rescale the output to the original image shape"""

        nms_thr = self.config["nms_threshold"]
        score_thr = self.config["confidence_threshold"]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio
        dets = self.multiclass_nms_class_agnostic(
            boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr
        )

        return dets

    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [
                    valid_boxes[keep],
                    valid_scores[keep, None],
                    valid_cls_inds[keep, None],
                ],
                1,
            )
        return dets

    @staticmethod
    def nms(boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def unload(self):
        del self.det_net
        del self.pose_net
