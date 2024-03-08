import logging

from PyQt5 import QtCore

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .types import AutoLabelingResult
from .__base__.yolo import YOLO
from .utils import denormalize_bbox


class YOLOW(YOLO):
    """https://github.com/AILab-CVC/YOLO-World"""

    def postprocess(self, outputs, image_shape):
        num_objs, bboxes, scores, class_ids = [out[0] for out in outputs]
        bboxes = [
            denormalize_bbox(bbox, self.input_shape, image_shape)
            for bbox in bboxes
        ]
        return num_objs, bboxes, scores, class_ids

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

        blob = self.preprocess(image, upsample_mode="resize")
        outputs = self.net.get_ort_inference(blob, extract=False)
        _, bboxes, scores, class_ids = self.postprocess(
            outputs, image.shape[:2]
        )

        shapes = []
        for bbox, score, cls_id in zip(bboxes, scores, class_ids):
            if score < self.conf_thres or (int(cls_id) == -1):
                continue
            xmin, ymin, xmax, ymax = bbox
            rectangle_shape = Shape(
                label=str(self.classes[int(cls_id)]),
                shape_type="rectangle",
            )
            rectangle_shape.add_point(QtCore.QPointF(xmin, ymin))
            rectangle_shape.add_point(QtCore.QPointF(xmax, ymin))
            rectangle_shape.add_point(QtCore.QPointF(xmax, ymax))
            rectangle_shape.add_point(QtCore.QPointF(xmin, ymax))
            shapes.append(rectangle_shape)

        result = AutoLabelingResult(shapes, replace=True)

        return result
