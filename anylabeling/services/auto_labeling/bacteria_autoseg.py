# 文件路径: anylabeling/services/auto_labeling/bacteria_autoseg.py
# 【最终修正版 V5 - 增加多边形近似平滑】

import logging
import traceback
import numpy as np
import cv2
from PyQt5 import QtCore

from .model import Model
from .bacteria_onnx import BacteriaONNX
from .types import AutoLabelingResult
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img


class BacteriaAutoseg(Model):
    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "encoder_model_path",
            "decoder_model_path",
        ]
        widgets = []
        output_modes = {"polygon": "Polygon"}
        default_output_mode = "polygon"

    def __init__(self, config_path, on_message) -> None:
        super().__init__(config_path, on_message)
        encoder_model_abs_path = self.get_model_abs_path(
            self.config, "encoder_model_path"
        )
        decoder_model_abs_path = self.get_model_abs_path(
            self.config, "decoder_model_path"
        )

        if not encoder_model_abs_path:
            raise FileNotFoundError("Encoder model path not found in config.")
        if not decoder_model_abs_path:
            raise FileNotFoundError("Decoder model path not found in config.")

        self.model = BacteriaONNX(model_path=encoder_model_abs_path)
        logging.info("✅ BacteriaAutoseg plugin loaded successfully.")

    def set_auto_labeling_reset_tracker(self):
        pass

    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        logging.info("Triggering full auto-segmentation for bacteria.")
        try:
            cv_image = qt_img_to_rgb_cv_img(image, filename)
            all_masks = self.model.predict_masks(cv_image)

            if all_masks.size == 0:
                logging.info("No masks were found by the model.")
                return AutoLabelingResult([], replace=True)

            all_shapes = []
            num_masks = all_masks.shape[0]
            logging.info(f"Generated {num_masks} masks. Post-processing...")

            for i in range(num_masks):
                shapes_from_mask = self.post_process(all_masks[i])
                all_shapes.extend(shapes_from_mask)
            
            logging.info(f"Finished. Found {len(all_shapes)} shapes.")
            return AutoLabelingResult(all_shapes, replace=True)

        except Exception as e:
            logging.error(f"Could not inference BacteriaAutoseg model: {e}")
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

    def post_process(self, mask: np.ndarray) -> list[Shape]:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        kernel = np.ones((7, 7), np.uint8)
        smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        shapes = []
        if self.output_mode == "polygon":
            for contour in contours:
                if cv2.contourArea(contour) < 10:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                epsilon = 0.004 * perimeter 
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                points = approx_contour.reshape(-1, 2).tolist()
                if len(points) < 3: continue
                
                shape = Shape(label="bacteria", shape_type="polygon")
                for point in points:
                    shape.add_point(QtCore.QPointF(int(point[0]), int(point[1])))
                shapes.append(shape)
        
        return shapes

    def split_mask(self, mask: np.ndarray, line_points: list) -> list[np.ndarray] | None:
        return self.model.split_mask_with_line(mask, line_points)
        
    def unload(self):
        self.model = None
        logging.info("BacteriaAutoseg plugin unloaded.")