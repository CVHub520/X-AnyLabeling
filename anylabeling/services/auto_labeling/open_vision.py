import warnings

warnings.filterwarnings("ignore")

import os
import cv2
import argparse
import traceback
import numpy as np
from PIL import Image
from typing import List

from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QCoreApplication

from anylabeling.utils import GenericWorker
from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import (
    get_bounding_boxes,
    qt_img_to_rgb_cv_img,
)
from anylabeling.services.auto_labeling.utils import calculate_rotation_theta

from .model import Model
from .types import AutoLabelingResult
from .lru_cache import LRUCache
from .__base__.sam2 import SegmentAnything2ONNX

try:
    import torch
    from .visualgd.datasets import transforms as T
    from .visualgd.registry import MODULE_BUILD_FUNCS
    from .visualgd.util.misc import nested_tensor_from_tensor_list
    from .visualgd.config.cfg_handler import ConfigurationHandler

    OPEN_VISION_AVAILABLE = True
except ImportError:
    OPEN_VISION_AVAILABLE = False


class OpenVision(Model):
    """Open Vision Model"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "encoder_model_path",
            "decoder_model_path",
        ]
        widgets = [
            "edit_text",
            "button_send",
            "output_label",
            "output_select_combobox",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
            "button_auto_decode",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rotation": QCoreApplication.translate("Model", "Rotation"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        if not OPEN_VISION_AVAILABLE:
            message = "OpenVision model will not be available. Please install related packages and try again."
            raise ImportError(message)

        # Run the parent class's init method
        super().__init__(model_config, on_message)

        # ----------- Open Vision ---------- #
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize model.",
                )
            )

        if torch.cuda.is_available() and __preferred_device__ == "GPU":
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.keywords = self.config.get("keywords", "")
        self.box_threshold = self.config.get("box_threshold", 0.3)
        self.text_encoder_type = self.config.get(
            "text_encoder_type", "bert-base-uncased"
        )

        model_config = dict(
            keywords=self.keywords,
            conf_threshold=self.box_threshold,
            pretrain_model_path=model_abs_path,
            device=(
                "cuda"
                if torch.cuda.is_available() and __preferred_device__ == "GPU"
                else "cpu"
            ),
        )
        self.net = self.build_model(model_config)
        self.net.to(self.device)
        self.transform = self.build_transforms()

        # ----------- Segment-Anything-2 ---------- #
        encoder_model_abs_path = self.get_model_abs_path(
            self.config, "encoder_model_path"
        )
        if not encoder_model_abs_path or not os.path.isfile(
            encoder_model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize encoder of SAM2.",
                )
            )
        decoder_model_abs_path = self.get_model_abs_path(
            self.config, "decoder_model_path"
        )
        if not decoder_model_abs_path or not os.path.isfile(
            decoder_model_abs_path
        ):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize decoder of SAM2.",
                )
            )

        # Load models
        self.model = SegmentAnything2ONNX(
            encoder_model_abs_path,
            decoder_model_abs_path,
            __preferred_device__,
        )

        # Mark for auto labeling
        # points, rectangles
        self.marks = []

        # Cache for image embedding
        self.cache_size = 1
        self.preloaded_size = 1
        self.image_embedding_cache = LRUCache(self.cache_size)

        # Pre-inference worker
        self.pre_inference_thread = None
        self.pre_inference_worker = None
        self.stop_inference = False

    def build_transforms(self):
        normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        data_transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                normalize,
            ]
        )
        return data_transform

    def build_model(self, model_config: dict):
        args = argparse.Namespace()
        for key, value in model_config.items():
            setattr(args, key, value)

        cfg = ConfigurationHandler.get_config()
        if self.text_encoder_type:
            cfg.merge_from_dict({"text_encoder_type": self.text_encoder_type})
        cfg_dict = cfg._cfg_dict.to_dict()
        args_vars = vars(args)
        for k, v in cfg_dict.items():
            if k not in args_vars:
                setattr(args, k, v)
            else:
                raise ValueError(f"Key {k} can only be used by args")
        assert args.modelname in MODULE_BUILD_FUNCS._module_dict
        build_func = MODULE_BUILD_FUNCS.get(args.modelname)
        model, _, _ = build_func(args)

        checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")[
            "model"
        ]
        model.load_state_dict(checkpoint, strict=False)

        model.eval()

        return model

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def post_process(self, masks, label=None):
        """
        Post process masks
        """
        # Find contours
        masks[masks > 0.0] = 255
        masks[masks <= 0.0] = 0
        masks = masks.astype(np.uint8)
        contours, _ = cv2.findContours(
            masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Refine contours
        approx_contours = []
        for contour in contours:
            # Approximate contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_contours.append(approx)

        # Remove too big contours ( >90% of image size)
        if len(approx_contours) > 1:
            image_size = masks.shape[0] * masks.shape[1]
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area < image_size * 0.9
            ]

        # Remove small contours (area < 20% of average area)
        if len(approx_contours) > 1:
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            avg_area = np.mean(areas)

            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area > avg_area * 0.2
            ]
            approx_contours = filtered_approx_contours

        # Contours to shapes
        shapes = []
        if self.output_mode == "polygon":
            for approx in approx_contours:
                # Scale points
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue
                points.append(points[0])
                # Create shape
                shape = Shape(flags={})
                for point in points:
                    point[0] = int(point[0])
                    point[1] = int(point[1])
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shape.shape_type = "polygon"
                shape.closed = True
                shape.label = "AUTOLABEL_OBJECT" if label is None else label
                shape.selected = False
                shapes.append(shape)
        elif self.output_mode in ["rectangle", "rotation"]:
            shape = Shape(flags={})
            rectangle_box, rotation_box = get_bounding_boxes(
                approx_contours[0]
            )
            xmin, ymin, xmax, ymax = rectangle_box
            if self.output_mode == "rectangle":
                shape.add_point(QtCore.QPointF(int(xmin), int(ymin)))
                shape.add_point(QtCore.QPointF(int(xmax), int(ymin)))
                shape.add_point(QtCore.QPointF(int(xmax), int(ymax)))
                shape.add_point(QtCore.QPointF(int(xmin), int(ymax)))
            else:
                for point in rotation_box:
                    shape.add_point(
                        QtCore.QPointF(int(point[0]), int(point[1]))
                    )
                shape.direction = calculate_rotation_theta(rotation_box)
            shape.shape_type = self.output_mode
            shape.closed = True
            shape.fill_color = "#000000"
            shape.line_color = "#000000"
            shape.label = "AUTOLABEL_OBJECT" if label is None else label
            shape.selected = False
            shapes.append(shape)

        return shapes if label is None else shapes[0]

    def get_visual_prompt(self) -> List[List[float]]:
        visual_prompt = []
        for mark in self.marks:
            if mark["type"] == "rectangle":
                visual_prompt.append(mark["data"])
        return visual_prompt

    def get_boxes(self, cv_image: np.ndarray, text_prompt: str = ""):
        image = Image.fromarray(cv_image)
        visual_prompt = self.get_visual_prompt()

        input_image, _ = self.transform(image, {"exemplars": torch.tensor([])})
        input_image = input_image.unsqueeze(0).to(self.device)

        exemplar_images, exemplars = self.transform(
            image, {"exemplars": torch.tensor(visual_prompt)}
        )
        exemplar_images = exemplar_images.unsqueeze(0).to(self.device)
        exemplars = [exemplars["exemplars"].to(self.device)]

        with torch.no_grad():
            model_output = self.net(
                nested_tensor_from_tensor_list(input_image),
                nested_tensor_from_tensor_list(exemplar_images),
                exemplars,
                [
                    torch.tensor([0]).to(self.device)
                    for _ in range(len(input_image))
                ],
                captions=[text_prompt + " ."] * len(input_image),
            )

        ind_to_filter = self.get_ind_to_filter(
            text_prompt, model_output["token"][0].word_ids, self.keywords
        )
        logits = model_output["pred_logits"].sigmoid()[0][:, ind_to_filter]
        boxes = model_output["pred_boxes"][0]
        if len(self.keywords.strip()) > 0:
            box_mask = (logits > self.box_threshold).sum(dim=-1) == len(
                ind_to_filter
            )
        else:
            box_mask = logits.max(dim=-1).values > self.box_threshold
        logits = logits[box_mask, :].cpu().numpy()
        boxes = boxes[box_mask, :].cpu().numpy()
        boxes = self.rescale_boxes(image, boxes)
        return boxes

    def predict_shapes(self, image, image_path=None, text_prompt=None):
        """
        Predict shapes from image
        """
        if image is None:
            return []

        try:
            cv_image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            return []

        try:
            cached_data = self.image_embedding_cache.get(image_path)
            if cached_data is not None:
                image_embedding = cached_data
            else:
                if self.stop_inference:
                    return AutoLabelingResult([], replace=False)
                image_embedding = self.model.encode(cv_image)
                self.image_embedding_cache.put(
                    image_path,
                    image_embedding,
                )
                if self.stop_inference:
                    return AutoLabelingResult([], replace=False)

            if text_prompt or self.is_rectangle_mode():
                text_prompt = text_prompt if text_prompt else ""
                boxes = self.get_boxes(cv_image, text_prompt)
                logger.debug(f"generated boxes by open vision:\n{boxes}")

                shapes = []
                for box in boxes:
                    label = "AUTOLABEL_OBJECT"
                    marks = [
                        {
                            "data": box,
                            "label": 1,
                            "type": "rectangle",
                        }
                    ]
                    masks = self.model.predict_masks(image_embedding, marks)
                    if len(masks.shape) == 4:
                        masks = masks[0][0]
                    else:
                        masks = masks[0]
                    shape = self.post_process(masks, label=label)
                    shapes.append(shape)
                result = AutoLabelingResult(shapes, replace=False)
            else:
                masks = self.model.predict_masks(image_embedding, self.marks)
                if len(masks.shape) == 4:
                    masks = masks[0][0]
                else:
                    masks = masks[0]
                shapes = self.post_process(masks)
                result = AutoLabelingResult(shapes, replace=False)
            return result
        except Exception as e:  # noqa
            logger.warning("Could not inference model")
            logger.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

    def is_rectangle_mode(self):
        for mark in self.marks:
            if mark["type"] == "rectangle":
                return True
        return False

    @staticmethod
    def rescale_boxes(image: Image.Image, boxes: np.ndarray):
        converted_boxes = []
        width, height = image.size
        for box in boxes:
            x_center, y_center, w, h = box
            x1 = (x_center - w / 2) * width
            y1 = (y_center - h / 2) * height
            x2 = (x_center + w / 2) * width
            y2 = (y_center + h / 2) * height
            converted_boxes.append([x1, y1, x2, y2])
        return np.array(converted_boxes, dtype=int)

    @staticmethod
    def get_ind_to_filter(text, word_ids, keywords):
        if len(keywords) <= 0:
            return list(range(len(word_ids)))
        input_words = text.split()
        keywords = keywords.split(",")
        keywords = [keyword.strip() for keyword in keywords]

        word_inds = []
        for keyword in keywords:
            if keyword in input_words:
                if len(word_inds) <= 0:
                    ind = input_words.index(keyword)
                    word_inds.append(ind)
                else:
                    ind = input_words.index(keyword, word_inds[-1])
                    word_inds.append(ind)
            else:
                raise Exception("Only specify keywords in the input text!")

        inds_to_filter = []
        for ind in range(len(word_ids)):
            word_id = word_ids[ind]
            if word_id in word_inds:
                inds_to_filter.append(ind)

        return inds_to_filter

    def unload(self):
        del self.net
        self.stop_inference = True
        if self.pre_inference_thread:
            self.pre_inference_thread.quit

    def preload_worker(self, files):
        """
        Preload next files, run inference and cache results
        """
        files = files[: self.preloaded_size]
        for filename in files:
            if self.image_embedding_cache.find(filename):
                continue
            image = self.load_image_from_filename(filename)
            if image is None:
                continue
            if self.stop_inference:
                return
            cv_image = qt_img_to_rgb_cv_img(image)
            image_embedding = self.model.encode(cv_image)
            self.image_embedding_cache.put(
                filename,
                image_embedding,
            )

    def on_next_files_changed(self, next_files):
        """
        Handle next files changed. This function can preload next files
        and run inference to save time for user.
        """
        if (
            self.pre_inference_thread is None
            or not self.pre_inference_thread.isRunning()
        ):
            self.pre_inference_thread = QThread()
            self.pre_inference_worker = GenericWorker(
                self.preload_worker, next_files
            )
            self.pre_inference_worker.finished.connect(
                self.pre_inference_thread.quit
            )
            self.pre_inference_worker.moveToThread(self.pre_inference_thread)
            self.pre_inference_thread.started.connect(
                self.pre_inference_worker.run
            )
            self.pre_inference_thread.start()
