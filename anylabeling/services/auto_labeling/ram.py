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


class RAM(Model):
    """Image tagging model using Recognize Anything Model (RAM)"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
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
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.input_shape = self.net.get_input_shape()[-2:]
        self.tag_mode = self.config.get("tag_mode", "")  # ['en', 'cn']

        # load tag list
        self.tag_list, self.tag_list_chinese = self.load_tag_list()
        delete_tags = self.config.get("delete_tags", [])
        filter_tags = self.config.get("filter_tags", [])
        if delete_tags:
            self.delete_tag_index = [
                self.tag_list.tolist().index(label) for label in delete_tags
            ]
        elif filter_tags:
            self.delete_tag_index = [
                index
                for index, item in enumerate(self.tag_list)
                if item not in filter_tags
            ]
        else:
            self.delete_tag_index = []

    def preprocess(self, input_image):
        """
        Pre-processes the input image before feeding it to the network.
        """
        h, w = self.input_shape
        image = cv2.resize(input_image, (w, h))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0).astype(np.float32)
        return image

    def postprocess(self, outs):
        """
        Post-processes the network's output.
        """
        tags, bs = outs
        tags[:, self.delete_tag_index] = 0
        tag_output = []
        tag_output_chinese = []
        for b in range(bs[0]):
            index = np.argwhere(tags[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            tag_output.append(" | ".join(token))
            token_chinese = self.tag_list_chinese[index].squeeze(axis=1)
            tag_output_chinese.append(" | ".join(token_chinese))

        return tag_output, tag_output_chinese

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
        outs = self.net.get_ort_inference(blob, extract=False)
        tags = self.postprocess(outs)
        image_text = self.get_results(tags)

        shapes = []
        shape = Shape(
            label="tag",
            text=image_text,
            shape_type="rectangle",
        )
        h, w = image.shape[:2]
        shape.add_point(QtCore.QPointF(0, 0))
        shape.add_point(QtCore.QPointF(w, h))
        shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result

    @staticmethod
    def load_tag_list():
        current_dir = os.path.dirname(__file__)
        tag_list_file = os.path.join(
            current_dir, "configs", "ram_tag_list.txt"
        )
        tag_list_chinese_file = os.path.join(
            current_dir, "configs", "ram_tag_list_chinese.txt"
        )

        with open(tag_list_file, "r", encoding="utf-8") as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        with open(tag_list_chinese_file, "r", encoding="utf-8") as f:
            tag_list_chinese = f.read().splitlines()
        tag_list_chinese = np.array(tag_list_chinese)

        return tag_list, tag_list_chinese

    def get_results(self, tags):
        en_tags, zh_tag = tags
        image_text = en_tags[0] + "\n" + zh_tag[0]
        if self.tag_mode == "en":
            return en_tags[0]
        elif self.tag_mode == "zh":
            return zh_tag[0]
        return image_text

    def unload(self):
        del self.net
