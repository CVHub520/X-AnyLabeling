import logging
import os
import traceback

import cv2
import onnx
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QCoreApplication

from anylabeling.utils import GenericWorker
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

from .lru_cache import LRUCache
from .model import Model
from .types import AutoLabelingResult
from .sam_onnx import SegmentAnythingONNX


class SegmentAnything(Model):
    """Segmentation model using SegmentAnything"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "encoder_model_path",
            "decoder_model_path",
        ]
        widgets = [
            "output_label",
            "output_select_combobox",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "polygon"

    # __init__ 和其他方法保持不变...
    def __init__(self, config_path, on_message) -> None:
        super().__init__(config_path, on_message)
        self.input_size = self.config["input_size"]
        self.max_width = self.config["max_width"]
        self.max_height = self.config["max_height"]
        encoder_model_abs_path = self.get_model_abs_path(
            self.config, "encoder_model_path"
        )
        if not encoder_model_abs_path or not os.path.isfile(encoder_model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize encoder of Segment Anything.",
                )
            )
        decoder_model_abs_path = self.get_model_abs_path(
            self.config, "decoder_model_path"
        )
        if not decoder_model_abs_path or not os.path.isfile(decoder_model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    "Could not download or initialize decoder of Segment Anything.",
                )
            )
        if self.detect_model_variant(decoder_model_abs_path) == "sam2":
            self.model = SegmentAnything2ONNX(
                encoder_model_abs_path, decoder_model_abs_path
            )
        else:
            self.model = SegmentAnythingONNX(
                encoder_model_abs_path, decoder_model_abs_path
            )
        self.marks = []
        self.cache_size = 10
        self.preloaded_size = self.cache_size - 3
        self.image_embedding_cache = LRUCache(self.cache_size)
        self.pre_inference_thread = None
        self.pre_inference_worker = None
        self.stop_inference = False

    def detect_model_variant(self, decoder_model_abs_path):
        model = onnx.load(decoder_model_abs_path)
        input_names = [input.name for input in model.graph.input]
        if "high_res_feats_0" in input_names:
            return "sam2"
        return "sam"

    def set_auto_labeling_marks(self, marks):
        self.marks = marks

    def post_process(self, mask):
        # 这个方法现在只处理单个掩码，保持不变
        mask[mask > 0.0] = 255
        mask[mask <= 0.0] = 0
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        approx_contours = []
        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_contours.append(approx)
        
        # ... (此处省略了后处理的过滤逻辑，保持原样)
        if len(approx_contours) > 1:
            image_size = mask.shape[0] * mask.shape[1]
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area < image_size * 0.9
            ]
            approx_contours = filtered_approx_contours

        if len(approx_contours) > 1:
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            avg_area = np.mean(areas)

            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area > avg_area * 0.2
            ]
            approx_contours = filtered_approx_contours

        shapes = []
        if self.output_mode == "polygon":
            for approx in approx_contours:
                points = approx.reshape(-1, 2).tolist()
                if len(points) < 3: continue
                shape = Shape(label="AUTOLABEL_OBJECT", shape_type="polygon")
                for point in points:
                    shape.add_point(QtCore.QPointF(int(point[0]), int(point[1])))
                shapes.append(shape)
        elif self.output_mode == "rectangle":
            if not approx_contours: return []
            all_points = np.concatenate(approx_contours)
            x_min, y_min = all_points.min(axis=0)[0]
            x_max, y_max = all_points.max(axis=0)[0]
            shape = Shape(label="AUTOLABEL_OBJECT", shape_type="rectangle")
            shape.add_point(QtCore.QPointF(x_min, y_min))
            shape.add_point(QtCore.QPointF(x_max, y_max))
            shapes.append(shape)
        return shapes
    
    # <<< 核心修改在这里 >>>
    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict shapes from image.
        - If a single foreground point is provided, trigger full-image auto-segmentation.
        - Otherwise, perform interactive segmentation based on the provided marks.
        """
        if image is None or not self.marks:
            return AutoLabelingResult([], replace=False)

        try:
            # 步骤 1: 获取图像嵌入 (所有模式通用)
            cached_data = self.image_embedding_cache.get(filename)
            if cached_data is not None:
                image_embedding = cached_data
            else:
                cv_image = qt_img_to_rgb_cv_img(image, filename)
                if self.stop_inference: return AutoLabelingResult([], replace=False)
                image_embedding = self.model.encode(cv_image)
                self.image_embedding_cache.put(filename, image_embedding)

            if self.stop_inference: return AutoLabelingResult([], replace=False)

            # 步骤 2: 判断是触发全自动模式还是交互模式
            # 条件：只有一个标记，且这个标记是前景点 (type='point', label=1)
            is_auto_mode_trigger = (
                len(self.marks) == 1 and 
                self.marks[0]["type"] == "point" and 
                self.marks[0]["label"] == 1
            )

            if is_auto_mode_trigger:
                # --- 全自动分割逻辑 ---
                logging.info("Single point click detected. Triggering full auto-segmentation.")
                
                # 调用模型，prompt传None，触发我们的全自动逻辑
                all_masks = self.model.predict_masks(image_embedding, prompt=None)
                
                if all_masks.shape[0] == 1:
                    all_masks = all_masks[0]
                
                all_shapes = []
                num_masks = all_masks.shape[0]
                logging.info(f"Generated {num_masks} masks. Post-processing...")
                
                for i in range(num_masks):
                    shapes_from_mask = self.post_process(all_masks[i])
                    all_shapes.extend(shapes_from_mask)
                
                logging.info(f"Finished. Found {len(all_shapes)} shapes.")
                # replace=True: 用新生成的所有形状替换当前画布内容
                return AutoLabelingResult(all_shapes, replace=True)

            else:
                # --- 原始的交互式分割逻辑 ---
                logging.info("Multiple points/rect detected. Performing interactive segmentation.")
                
                masks = self.model.predict_masks(image_embedding, self.marks)
                
                if len(masks.shape) == 4:
                    mask = masks[0][0]
                else:
                    mask = masks[0]
                
                shapes = self.post_process(mask)
                # replace=False: 将新生成的形状添加到画布，不替换已有内容
                return AutoLabelingResult(shapes, replace=False)

        except Exception as e:
            logging.warning(f"Could not inference model: {e}")
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

    # unload, preload_worker, on_next_files_changed 等方法保持不变...
    def unload(self):
        self.stop_inference = True
        if self.pre_inference_thread:
            self.pre_inference_thread.quit()

    def preload_worker(self, files):
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
        if (
            self.pre_inference_thread is None
            or not self.pre_inference_thread.isRunning()
        ):
            self.pre_inference_thread = QThread()
            self.pre_inference_worker = GenericWorker(self.preload_worker, next_files)
            self.pre_inference_worker.finished.connect(self.pre_inference_thread.quit)
            self.pre_inference_worker.moveToThread(self.pre_inference_thread)
            self.pre_inference_thread.started.connect(self.pre_inference_worker.run)
            self.pre_inference_thread.start()