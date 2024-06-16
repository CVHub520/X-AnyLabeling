import logging
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .types import AutoLabelingResult
from .__base__.yolo import YOLO
from .utils import softmax


class YOLOv5_CLS(YOLO):
    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
        ]
        widgets = [
            "button_run",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def postprocess(self, outs, topk=1):
        """
        Classification:
            Post-processes the output of the network.

        Args:
            outs (list): Output predictions from the network.
            topk (int): Number of top predictions to consider.

        Returns:
            str: Predicted label.
        """
        res = softmax(np.array(outs)).tolist()
        index = np.argmax(res)
        label = str(self.classes[index])

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

        blob = self.preprocess(image, upsample_mode="centercrop")
        predictions = self.net.get_ort_inference(blob)
        label = self.postprocess(predictions)

        shapes = []
        shape = Shape(
            label=label,
            shape_type="rectangle",
        )
        h, w = image.shape[:2]
        shape.add_point(QtCore.QPointF(0, 0))
        shape.add_point(QtCore.QPointF(w, 0))
        shape.add_point(QtCore.QPointF(w, h))
        shape.add_point(QtCore.QPointF(0, h))
        shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)

        return result
