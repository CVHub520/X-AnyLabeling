# 文件路径: anylabeling/services/auto_labeling/bacteria_autoseg.py

import logging
import cv2
import numpy as np
from PyQt5 import QtCore

# 导入 X-AnyLabeling 的核心类
from .model import Model  # 从同级目录的 model.py 导入基类
from .types import AutoLabelingResult
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img

# 从同级目录的 bacteria_onnx.py 导入你的后端推理类
from .bacteria_onnx import BacteriaONNX

class BacteriaAutoseg(Model):
    """
    X-AnyLabeling 的细菌自动分割插件。
    这个类名 'BacteriaAutoseg' 必须与 yaml 文件中的 'type' (bacteria_autoseg) 对应。
    """
    
    def __init__(self, config_path, on_message) -> None:
        """
        插件初始化
        """
        super().__init__(config_path, on_message)
        
        # 从配置文件中安全地获取模型路径和参数
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        input_size = self.config.get("input_size", 512)
        
        # 实例化后端推理引擎
        self.model = None
        try:
            self.model = BacteriaONNX(model_abs_path, input_size)
            logging.info("Bacteria Auto-Segmenter plugin loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load BacteriaONNX backend: {e}")
            # 通过回调向用户显示错误消息
            self.on_message(
                f"Error loading Bacteria model from '{model_abs_path}'. Check if the file exists and is valid.",
                "error"
            )

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        当用户点击“自动标注”时，框架会调用此函数。
        """
        if self.model is None:
            self.on_message("Bacteria model is not available. Please check logs.", "warning")
            return AutoLabelingResult([], replace=True)

        logging.info("Starting bacteria auto-segmentation...")
        try:
            # 1. 将 Qt Image 转换为 OpenCV Image
            cv_image = qt_img_to_rgb_cv_img(image, filename)
            
            # 2. 调用后端执行推理
            all_masks = self.model.predict_masks(cv_image)
            
            if all_masks.size == 0:
                logging.info("No bacteria found by the model.")
                self.on_message("No objects were detected.", "info")
                return AutoLabelingResult([], replace=True)

            # 3. 将后端返回的 mask 转换为 X-AnyLabeling 的 Shape 对象
            all_shapes = []
            for mask in all_masks:
                shapes_from_mask = self.post_process(mask)
                all_shapes.extend(shapes_from_mask)
            
            logging.info(f"Finished. Found {len(all_shapes)} shapes.")
            self.on_message(f"Successfully segmented {len(all_shapes)} objects.", "info")
            
            # replace=True: 用新生成的所有形状替换当前画布内容
            return AutoLabelingResult(all_shapes, replace=True)

        except Exception as e:
            logging.error(f"Could not run inference: {e}", exc_info=True)
            self.on_message(f"An error occurred during segmentation: {e}", "error")
            return AutoLabelingResult([], replace=True)

    def post_process(self, mask: np.ndarray) -> list[Shape]:
        """
        将单个二值化 mask (np.uint8) 转换为一个或多个多边形 Shape 对象。
        """
        # 确保 mask 是 uint8 类型，并且值为 0 或 255
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        # 寻找轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for contour in contours:
            # 忽略太小的轮廓
            if cv2.contourArea(contour) < 10:
                continue

            # (可选) 轮廓平滑
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) < 3: 
                continue
                
            # 创建 Shape 对象，标签可以从 UI 读取或硬编码
            shape = Shape(label="bacterium", shape_type="polygon")
            points = approx.reshape(-1, 2)
            for point in points:
                shape.add_point(QtCore.QPointF(float(point[0]), float(point[1])))
            shapes.append(shape)
            
        return shapes