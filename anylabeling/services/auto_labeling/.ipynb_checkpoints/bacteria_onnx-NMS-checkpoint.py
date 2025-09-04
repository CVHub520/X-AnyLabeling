# 文件路径: anylabeling/services/auto_labeling/bacteria_onnx.py
# 【最终修正版 V9 - 优化Mask NMS的速度】

import cv2
import numpy as np
import onnxruntime
import logging
import os
import warnings

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

class BacteriaONNX:
    def __init__(self, model_path: str, input_size: int = 1024):
        # ... (超参数保持不变) ...
        self.input_size = input_size
        self.points_per_side = 64
        self.pred_iou_thresh = 0.85
        self.min_mask_region_area = 50
        self.box_nms_thresh = 0.4 
        self.max_area_ratio = 0.04
        self.min_circularity = 0.15
        self.mask_threshold = 0.5
        try:
            # ... (模型加载代码保持不变) ...
            encoder_path = model_path
            decoder_path = os.path.join(os.path.dirname(encoder_path), "bacteria_decoder.onnx")
            if not os.path.exists(encoder_path): raise FileNotFoundError(f"Encoder model not found at: {encoder_path}")
            if not os.path.exists(decoder_path): raise FileNotFoundError(f"Decoder model not found at: {decoder_path}")
            so = onnxruntime.SessionOptions()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.enc_session = onnxruntime.InferenceSession(encoder_path, sess_options=so, providers=providers)
            self.dec_session = onnxruntime.InferenceSession(decoder_path, sess_options=so, providers=providers)
            self.enc_input_name = self.enc_session.get_inputs()[0].name
            logging.info(f"ONNX models loaded successfully, using providers: {self.enc_session.get_providers()}")
        except Exception as e:
            logging.error(f"Failed to load ONNX models: {e}", exc_info=True)
            raise e

    def predict_masks(self, cv_image: np.ndarray) -> np.ndarray:
        # ... (这个函数保持不变) ...
        try:
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            H, W, _ = image_rgb.shape
            padded_img, _, (new_h, new_w), scale = self._resize_longest_side_and_pad(image_rgb, self.input_size)
            enc_input_3d = (padded_img.astype(np.float32) / 255.0)
            enc_input = np.expand_dims(enc_input_3d, axis=0).transpose(0, 3, 1, 2)
            image_embeddings = self.enc_session.run(None, {self.enc_input_name: enc_input})[0]
            gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            roi_mask = self._find_petri_dish(gray_image)
            grid_points = self._generate_grid_points(W, H, self.points_per_side)
            points_inside_roi = grid_points[roi_mask[grid_points[:, 1], grid_points[:, 0]] > 0]
            if len(points_inside_roi) == 0: return np.array([])
            all_masks_data = []
            for point in points_inside_roi:
                results_for_point = self._process_single_point(point, image_embeddings, scale, H, W, new_h, new_w)
                if results_for_point: all_masks_data.extend(results_for_point)
            if not all_masks_data: return np.array([])
            base_filtered = self._mask_nms_postprocess(all_masks_data, self.min_mask_region_area, self.box_nms_thresh)
            if not base_filtered: return np.array([])
            advanced_filtered = self._advanced_postprocess(base_filtered, (H, W), self.max_area_ratio, self.min_circularity)
            final_masks_list = []
            roi_mask_bool = roi_mask.astype(bool)
            for data in advanced_filtered:
                mask = np.logical_and(data['mask'], roi_mask_bool)
                if np.count_nonzero(mask) > self.min_mask_region_area:
                    final_masks_list.append(mask.astype(np.uint8))
            if not final_masks_list: return np.array([])
            return np.stack(final_masks_list, axis=0)
        except Exception as e:
            logging.error(f"Error during predict_masks: {e}", exc_info=True)
            return np.array([])

    def _process_single_point(self, point_data, image_embeddings, scale, H, W, new_h, new_w):
        # ... (这个函数保持不变) ...
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*overflow.*')
            warnings.filterwarnings('ignore', message='.*invalid value.*')
            px, py = point_data
            tx, ty = px * scale, py * scale
            point_coords = np.array([[[tx, ty], [0.0, 0.0]]], dtype=np.float32)
            point_labels = np.array([[1.0, -1.0]], dtype=np.float32)
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([0.0], dtype=np.float32)
            orig_im_size = np.array([H, W], dtype=np.float32)
            feeds = {"image_embeddings": image_embeddings, "point_coords": point_coords, "point_labels": point_labels, "mask_input": mask_input, "has_mask_input": has_mask_input, "orig_im_size": orig_im_size}
            masks, ious, low_res_logits = self.dec_session.run(None, feeds)
            results_for_point = []
            for i in range(masks.shape[1]):
                iou = float(ious[0, i])
                if iou < self.pred_iou_thresh: continue
                logits_256 = low_res_logits[0, i]
                prob_256 = _sigmoid(np.nan_to_num(np.clip(logits_256, -100, 100)))
                prob_1024 = cv2.resize(prob_256, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
                prob_padded = prob_1024[:new_h, :new_w]
                prob_blurred = cv2.GaussianBlur(prob_padded, (5, 5), 0)
                upscale_factor = 2
                prob_upscaled = cv2.resize(prob_blurred, (W * upscale_factor, H * upscale_factor), interpolation=cv2.INTER_CUBIC)
                mask_upscaled = (prob_upscaled >= self.mask_threshold)
                final_mask = cv2.resize(mask_upscaled.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                if np.count_nonzero(final_mask) > 0:
                    results_for_point.append({"mask": final_mask, "iou": iou})
            return results_for_point

    # -------> 【新增的辅助函数】 <-------
    def _do_boxes_overlap(self, boxA, boxB):
        if boxA[2] < boxB[0] or boxA[0] > boxB[2]: return False
        if boxA[3] < boxB[1] or boxA[1] > boxB[3]: return False
        return True

    # -------> 【优化后的核心函数】 <-------
    def _mask_nms_postprocess(self, masks_data: list, min_mask_region_area: int, mask_nms_thresh: float) -> list:
        pre_filtered = []
        for data in masks_data:
            cleaned_mask = self._filter_small_regions(data["mask"], min_mask_region_area)
            if np.count_nonzero(cleaned_mask) == 0: continue
            rows, cols = np.any(cleaned_mask, axis=1), np.any(cleaned_mask, axis=0)
            if not np.any(rows) or not np.any(cols): continue
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            pre_filtered.append({"mask": cleaned_mask, "area": np.count_nonzero(cleaned_mask), "bbox": [x_min, y_min, x_max, y_max], "iou": data["iou"]})
        if not pre_filtered: return []
        scores = np.array([d["iou"] for d in pre_filtered])
        sorted_indices = np.argsort(scores)[::-1]
        keep_indices = []
        while len(sorted_indices) > 0:
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            if len(sorted_indices) == 1: break
            remaining_indices = sorted_indices[1:]
            current_mask = pre_filtered[current_idx]["mask"]
            ious = []
            for other_idx in remaining_indices:
                if self._do_boxes_overlap(pre_filtered[current_idx]["bbox"], pre_filtered[other_idx]["bbox"]):
                    other_mask = pre_filtered[other_idx]["mask"]
                    intersection = np.logical_and(current_mask, other_mask).sum()
                    union = np.logical_or(current_mask, other_mask).sum()
                    iou = intersection / union if union > 0 else 0
                else:
                    iou = 0
                ious.append(iou)
            ious = np.array(ious)
            pass_mask = ious <= mask_nms_thresh
            sorted_indices = remaining_indices[pass_mask]
        final_masks = [pre_filtered[i] for i in keep_indices]
        return final_masks

    # ... (其余所有辅助函数 _find_petri_dish, _generate_grid_points, _advanced_postprocess 等都保持不变) ...
    def _find_petri_dish(self, gray_image: np.ndarray) -> np.ndarray:
        h, w = gray_image.shape
        blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h, param1=100, param2=80, minRadius=int(w * 0.2), maxRadius=int(w * 0.5))
        roi_mask = np.zeros_like(gray_image, dtype=np.uint8)
        if circles is not None:
            cx, cy, r = np.uint16(np.around(circles))[0, 0]
            cv2.circle(roi_mask, (cx, cy), r, 255, -1)
        else:
            roi_mask.fill(255)
        return roi_mask

    def _generate_grid_points(self, width: int, height: int, points_per_side: int) -> np.ndarray:
        grid_x = np.linspace(0, width - 1, points_per_side, dtype=np.float32)
        grid_y = np.linspace(0, height - 1, points_per_side, dtype=np.float32)
        xv, yv = np.meshgrid(grid_x, grid_y, indexing="xy")
        return np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.int32)
    
    def _filter_small_regions(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1: return mask
        keep_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
        return np.isin(labels, keep_labels) if len(keep_labels) > 0 else np.zeros_like(mask, dtype=bool)

    def _advanced_postprocess(self, masks_data: list, original_size: tuple, max_area_ratio: float, min_circularity: float) -> list:
        reference_area = float(original_size[0] * original_size[1])
        advanced_filtered = []
        for data in masks_data:
            if (data["area"] / reference_area) > max_area_ratio: continue
            contours, _ = cv2.findContours(data["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * data["area"] / (perimeter**2)
            if circularity < min_circularity: continue
            advanced_filtered.append(data)
        return advanced_filtered

    def _resize_longest_side_and_pad(self, img: np.ndarray, target_length: int):
        H, W = img.shape[:2]
        scale = float(target_length) / max(H, W)
        new_w, new_h = int(round(W * scale)), int(round(H * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((target_length, target_length, 3), dtype=resized.dtype)
        padded[:new_h, :new_w] = resized
        return padded, (H, W), (new_h, new_w), scale