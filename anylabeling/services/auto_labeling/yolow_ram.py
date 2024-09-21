import os

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .types import AutoLabelingResult
from .__base__.yolo import YOLO
from .__base__.ram import RecognizeAnything
from .engines.build_onnx_engine import OnnxBaseModel


class YOLOW_RAM(YOLO, RecognizeAnything):
    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        YOLO.__init__(self, model_config, on_message)
        RecognizeAnything.__init__(self, model_config, on_message)

        """Tagging"""
        tag_model_abs_path = self.get_model_abs_path(
            self.config, "tag_model_path"
        )
        if not tag_model_abs_path or not os.path.isfile(tag_model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {self.config['type']} model.",
                )
            )
        self.ram_net = OnnxBaseModel(tag_model_abs_path, __preferred_device__)
        self.ram_input_shape = self.ram_net.get_input_shape()[-2:]
        self.tag_mode = self.config.get("tag_mode", "")  # ['en', 'cn']
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

        blob = YOLO.preprocess(self, image, upsample_mode="letterbox")
        outs = YOLO.inference(self, blob=blob)
        boxes, class_ids, _, _, _ = YOLO.postprocess(self, outs)

        shapes = []
        for box, cls_id in zip(boxes, class_ids):
            label = self.classes[int(cls_id)]
            xmin, ymin, xmax, ymax = list(map(int, box))
            img = image[ymin:ymax, xmin:xmax]
            blob = RecognizeAnything.preprocess(
                self, img, self.ram_input_shape
            )
            outs = self.ram_net.get_ort_inference(blob, extract=False)
            tags = RecognizeAnything.postprocess(self, outs)
            description = RecognizeAnything.get_results(self, tags)
            shape = Shape(
                label=label,
                description=description,
                shape_type="rectangle",
            )
            shape.add_point(QtCore.QPointF(xmin, ymin))
            shape.add_point(QtCore.QPointF(xmax, ymin))
            shape.add_point(QtCore.QPointF(xmax, ymax))
            shape.add_point(QtCore.QPointF(xmin, ymax))
            shapes.append(shape)

        result = AutoLabelingResult(shapes, replace=True)
        return result
