import numpy as np

from PyQt5.QtCore import QCoreApplication

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .__base__.yolo import YOLO
from .types import AutoLabelingResult
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
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        blob = self.preprocess(image, upsample_mode="centercrop")
        predictions = self.net.get_ort_inference(blob)
        label = self.postprocess(predictions)
        result = AutoLabelingResult(
            shapes=[], replace=False, description=label
        )
        return result
