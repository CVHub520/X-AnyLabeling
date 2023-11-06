import logging
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .types import AutoLabelingResult
from .__base__.yolo import YOLO
from .utils import (
    numpy_nms,
    xywh2xyxy,
    xyxy2tlwh,
    rescale_tlwh,
)

class YOLOv8_Pose(YOLO):

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
            "point": QCoreApplication.translate("Model", "Point"),
        }
        default_output_mode = "rectangle"

    def postprocess(
            self, 
            prediction, 
            max_det=1000,
        ):

        """
        Args:
            prediction: (1, 56, *), where 56 = 4 + 1 + 3 * 17
                4 -> box_xywh
                1 -> box_score
                3*17 -> (x, y, kpt_score) * 17 keypoints
        """
        prediction = prediction.transpose((0, 2, 1))[0]
        x = prediction[prediction[:, 4] > self.conf_thres]
        if len(x) == 0:
            return []
        x[:, :4] = xywh2xyxy(x[:, :4])
        keep_idx = numpy_nms(x[:, :4], x[:, 4], self.nms_thres)  # NMS
        if keep_idx.shape[0] > max_det:  # limit detections
            keep_idx = keep_idx[:max_det]
        keep_label = []
        for i in keep_idx:
            keep_label.append(x[i].tolist())
        xyxy = np.array(keep_label)
        tlwh = xyxy2tlwh(xyxy)
        return tlwh

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
        predictions = self.net.get_ort_inference(blob)
        results = self.postprocess(predictions)

        if len(results) == 0: 
            return AutoLabelingResult([], replace=True)
        results = rescale_tlwh(
            self.input_shape, results, image.shape, kpts=True
        )

        shapes = []
        for group_id, r in enumerate(reversed(results)):
            xyxy, _, kpts = r[:4], r[4], r[5:]

            if not self.hide_box:
                rectangle_shape = Shape(
                    label=str(self.classes[0]), 
                    shape_type="rectangle",
                )
                rectangle_shape.add_point(QtCore.QPointF(xyxy[0], xyxy[1]))
                rectangle_shape.add_point(QtCore.QPointF(xyxy[2], xyxy[3]))
                shapes.append(rectangle_shape)

            interval = 3
            for i in range(0, len(kpts), interval):
                x, y, kpt_score = kpts[i: i + 3]
                if kpt_score > self.conf_thres:
                    label = self.keypoints[int(i//interval)]
                    point_shape = Shape(label=label, shape_type="point", group_id=group_id)
                    point_shape.add_point(QtCore.QPointF(x, y))
                    shapes.append(point_shape) 
        result = AutoLabelingResult(shapes, replace=True)
        return result
