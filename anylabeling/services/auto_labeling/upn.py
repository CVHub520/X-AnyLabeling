import warnings

warnings.filterwarnings("ignore")

import os
from PIL import Image

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult

try:
    from .__base__.upn import UPNWrapper

    UPN_AVAILABLE = True
except ImportError:
    UPN_AVAILABLE = False


class UPN(Model):
    """Universal Proposal Network (UPN) is a robust object proposal model
    for comprehensive object detection across diverse domains"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "iou_threshold",
            "conf_threshold",
        ]
        widgets = [
            "button_run",
            "upn_select_combobox",
            "input_conf",
            "edit_conf",
            "input_iou",
            "edit_iou",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        if not UPN_AVAILABLE:
            message = "UPN model will not be available. Please install related packages and try again."
            raise ImportError(message)

        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_name = self.config["type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_name} model.",
                )
            )

        # TODO: Add CPU support for UPN
        self.net = UPNWrapper(model_abs_path, "cuda")
        self.prompt_type = "coarse_grained_prompt"
        self.conf_thres = self.config.get("iou_threshold", 0.3)
        self.nms_thres = self.config.get("conf_threshold", 0.8)
        self._check_prompt_type()

    def _check_prompt_type(self):
        """Check if the prompt type is valid"""
        valid_prompt_types = ["fine_grained_prompt", "coarse_grained_prompt"]
        if self.prompt_type not in valid_prompt_types:
            logger.warning(
                f"""
                            ⚠️ Invalid prompt type: {self.prompt_type}. 
                            Please use one of the following: {valid_prompt_types}.
                            """
            )

    def set_upn_mode(self, mode):
        """Set UPN mode"""
        self.prompt_type = mode

    def set_auto_labeling_conf(self, value):
        """set auto labeling confidence threshold"""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_iou(self, value):
        """set auto labeling iou threshold"""
        if value > 0:
            self.nms_thres = value

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

        proposals = self.net.inference(
            Image.fromarray(image), self.prompt_type
        )
        results = self.net.filter(
            proposals, min_score=self.conf_thres, nms_value=self.nms_thres
        )
        if not results["original_xyxy_boxes"] or not results["scores"]:
            return AutoLabelingResult([], replace=True)

        bboxes = results["original_xyxy_boxes"][0]
        scores = results["scores"][0]
        if not bboxes:
            return AutoLabelingResult([], replace=True)

        shapes = []
        for bbox, score in zip(bboxes, scores):
            xmin, ymin, xmax, ymax = map(int, bbox)
            shape = Shape(
                label="OBJECT",
                score=float(score),
                shape_type="rectangle",
            )
            shape.add_point(QtCore.QPointF(xmin, ymin))
            shape.add_point(QtCore.QPointF(xmax, ymin))
            shape.add_point(QtCore.QPointF(xmax, ymax))
            shape.add_point(QtCore.QPointF(xmin, ymax))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):
        del self.net
