import logging
import os

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .types import AutoLabelingResult
from .utils import (
    rescale_box,
)
from .__base__.yolo import YOLO
from .engines.build_onnx_engine import OnnxBaseModel


class YOLOv5_RAM(YOLO):

    class Meta:
        required_config_names = [
            "tag_model_path",
        ]
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_name = self.config['type']
        det_model_abs_path = self.get_model_abs_path(self.config, "model_path")
        tag_model_abs_path = self.get_model_abs_path(self.config, "tag_model_path")
        if (not det_model_abs_path or not os.path.isfile(det_model_abs_path)) or \
           (not tag_model_abs_path or not os.path.isfile(tag_model_abs_path)):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model", 
                    f"Could not download or initialize {model_name} model."
                )
            )
        
        # YOLOv5
        self.net = OnnxBaseModel(det_model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]
        self.input_shape = self.net.get_input_shape()[-2:]
        self.nms_thres = self.config["nms_threshold"]
        self.conf_thres = self.config["confidence_threshold"]
        self.stride = self.config.get("stride", 32)
        self.anchors = self.config.get("anchors", None)
        self.agnostic = self.config.get("agnostic", False)
        self.filter_classes = self.config.get("filter_classes", None)

        if self.anchors:
            self.nl = len(self.anchors)
            self.na = len(self.anchors[0]) // 2
            self.grid = [np.zeros(1)] * self.nl
            self.stride = np.array(
                [self.stride//4, self.stride//2, self.stride]
            ) if not isinstance(self.stride, list) else \
            np.array(self.stride)
            self.anchor_grid = np.asarray(
                self.anchors, dtype=np.float32
            ).reshape(self.nl, -1, 2)
        if self.filter_classes:
            self.filter_classes = [
                i for i, item in enumerate(self.classes) 
                if item in self.filter_classes
            ]
        
        # RAM
        self.ram_net = OnnxBaseModel(tag_model_abs_path, __preferred_device__)
        self.ram_input_shape = self.ram_net.get_input_shape()[-2:]
        self.tag_mode = self.config.get("tag_mode", '')  # ['en', 'cn']
        self.tag_list, self.tag_list_chinese = self.load_tag_list()
        
        delete_tags = self.config.get("delete_tags", [])
        filter_tags = self.config.get("filter_tags", [])
        if delete_tags:
            self.delete_tag_index = [
                self.tag_list.tolist().index(label) for label in delete_tags
            ]
        elif filter_tags:
            self.delete_tag_index = [
                index for index, item in enumerate(self.tag_list) 
                if item not in filter_tags
            ]
        else:
            self.delete_tag_index = []

    def ram_preprocess(self, input_image):
        """
        Pre-processes the input image before feeding it to the network.
        """
        h, w = self.ram_input_shape
        image = cv2.resize(input_image, (w, h))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0).astype(np.float32)
        return image

    def ram_postprocess(self, outs):
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
            tag_output.append(' | '.join(token))
            token_chinese = self.tag_list_chinese[index].squeeze(axis=1)
            tag_output_chinese.append(' | '.join(token_chinese))

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
        predictions = self.net.get_ort_inference(blob)
        results = self.postprocess(predictions)[0]

        if len(results) == 0: 
            return AutoLabelingResult([], replace=True)
        results[:, :4] = rescale_box(
            self.input_shape, results[:, :4], image.shape
        ).round()

        results = self.get_attributes(image, results)

        shapes = []
        for res in results:
            x1, y1, x2, y2 = res["xyxy"]
            shape = Shape(
                label=res["label"],
                text=res["text"],
                shape_type="rectangle",
            )
            shape.add_point(QtCore.QPointF(x1, y1))
            shape.add_point(QtCore.QPointF(x2, y2))
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

        with open(tag_list_file, 'r', encoding="utf-8") as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        with open(tag_list_chinese_file, 'r', encoding="utf-8") as f:
            tag_list_chinese = f.read().splitlines()
        tag_list_chinese = np.array(tag_list_chinese)

        return tag_list, tag_list_chinese

    def get_results(self, tags):
        en_tags, zh_tag = tags
        image_text = en_tags[0] + '\n' + zh_tag[0]
        if self.tag_mode == 'en':
            return en_tags[0]
        elif self.tag_mode == 'zh':
            return zh_tag[0]
        return image_text

    def get_attributes(self, image, results):
        outputs = []
        for *xyxy, _, cls_id in reversed(results):
            x1, y1, x2, y2 = list(map(int, xyxy))
            img = image[y1: y2, x1: x2]
            blob = self.ram_preprocess(img)
            outs = self.ram_net.get_ort_inference(blob, extract=False)
            tags = self.ram_postprocess(outs)
            text = self.get_results(tags)
            outputs.append({
                "xyxy": xyxy,
                "text": text,
                "label": self.classes[int(cls_id)]
            })
        return outputs

    def unload(self):
        del self.net
        del self.ram_net