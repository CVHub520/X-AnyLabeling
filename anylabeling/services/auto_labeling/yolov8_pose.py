import logging
import numpy as np

from PyQt5 import QtCore

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .types import AutoLabelingResult
from .__base__.yolo import YOLO
from .utils import (
    numpy_nms,
    xywh2xyxy,
    xyxy2ltwh,
    rescale_tlwh,
    point_in_bbox,
)


class YOLOv8_Pose(YOLO):
    def postprocess(self, prediction):
        """
        Args:
            prediction: (1, 56, *), where 56 = 4 + 1 + 3 * 17
                4 -> box_xywh
                1 -> box_score
                3*17 -> (x, y, kpt_score) * 17 keypoints
        """
        prediction = prediction.transpose((0, 2, 1)).squeeze()
        x = prediction[prediction[:, 4] > self.conf_thres]
        if len(x) == 0:
            return []
        x[:, :4] = xywh2xyxy(x[:, :4])
        keep_idx = numpy_nms(x[:, :4], x[:, 4], self.iou_thres)  # NMS
        keep_label = []
        for i in keep_idx:
            keep_label.append(x[i].tolist())
        xyxy = np.array(keep_label)
        tlwh = xyxy2ltwh(xyxy)
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
            self.input_shape, results,
            image.shape, kpts=True,
            has_visible=self.has_visible,
            multi_label=self.multi_label
        )

        shapes = []
        for group_id, r in enumerate(reversed(results)):
            if self.multi_label:
                xyxy, score, class_ids, kpts = r[:4], r[4], r[5], r[6:]
            else:
                class_ids = 0
                xyxy, score, kpts = r[:4], r[4], r[5:]
            xmin, ymin, xmax, ymax = xyxy
            class_name = str(self.classes[int(class_ids)])
            rectangle_shape = Shape(
                label=class_name,
                score=score,
                shape_type="rectangle",
                group_id=group_id,
            )
            rectangle_shape.add_point(QtCore.QPointF(xmin, ymin))
            rectangle_shape.add_point(QtCore.QPointF(xmax, ymin))
            rectangle_shape.add_point(QtCore.QPointF(xmax, ymax))
            rectangle_shape.add_point(QtCore.QPointF(xmin, ymax))
            shapes.append(rectangle_shape)

            interval = 3 if self.has_visible else 2
            for i in range(0, len(kpts), interval):
                if self.has_visible:
                    x, y, kpt_score = kpts[i : i + 3]
                else:
                    x, y =  kpts[i : i + 2]
                    kpt_score = 1.0
                if x == 0 and y == 0:
                    continue
                inside_flag = point_in_bbox((x, y), xyxy)
                if (kpt_score > self.kpt_threshold) and inside_flag:
                    label = self.keypoints[class_name][int(i // interval)]
                    point_shape = Shape(
                        label=label, shape_type="point", group_id=group_id
                    )
                    point_shape.add_point(QtCore.QPointF(x, y))
                    shapes.append(point_shape)

        result = AutoLabelingResult(shapes, replace=self.replace)

        return result
