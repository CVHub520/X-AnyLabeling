# 文件路径: anylabeling/services/auto_labeling/bacteria_onnx.py

import cv2
import numpy as np
import onnxruntime
import logging
import os
import warnings

# --- 辅助函数 (从你的脚本中提取，保持在类外部或设为静态方法更佳) ---
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid aactivitation function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

class BacteriaONNX:
    """
    负责加载细菌分割 ONNX 模型并执行所有计算密集型任务。
    这个类封装了基于双 ONNX 模型 (encoder/decoder) 的推理流程。
    """
    def __init__(self, model_path: str, input_size: int = 1024):
        """
        构造函数：加载 Encoder 和 Decoder ONNX 模型。
        :param model_path: .onnx 编码器 (encoder) 文件的绝对路径。
        :param input_size: 模型期望的输入图像尺寸。
        """
        self.input_size = input_size
        
        # --- 从你的推理脚本中固定的超参数 ---
        self.points_per_side = 64
        self.pred_iou_thresh = 0.8
        self.min_mask_region_area = 50
        self.box_nms_thresh = 0.5
        self.max_area_ratio = 0.03
        self.min_circularity = 0.2
        self.mask_threshold = 0.65 # 通常为0.5

        try:
            encoder_path = model_path
            decoder_path = os.path.join(os.path.dirname(encoder_path), "bacteria_decoder.onnx")

            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder model not found at: {encoder_path}")
            if not os.path.exists(decoder_path):
                raise FileNotFoundError(f"Decoder model not found at: {decoder_path}")

            so = onnxruntime.SessionOptions()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.enc_session = onnxruntime.InferenceSession(encoder_path, sess_options=so, providers=providers)
            self.dec_session = onnxruntime.InferenceSession(decoder_path, sess_options=so, providers=providers)
            
            self.enc_input_name = self.enc_session.get_inputs()[0].name
            
            logging.info(f"ONNX encoder loaded from: {encoder_path}")
            logging.info(f"ONNX decoder loaded from: {decoder_path}")
            logging.info(f"Using providers: {self.enc_session.get_providers()}")

        except Exception as e:
            logging.error(f"Failed to load ONNX models: {e}", exc_info=True)
            raise e

    def predict_masks(self, cv_image: np.ndarray) -> np.ndarray:
        """
        【功能对接点 1: 全自动分割】
        执行完整的 "预处理 -> 编码 -> ROI过滤 -> 解码 -> 后处理" 流程。
        """
        try:
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            H, W, _ = image_rgb.shape

            # 1. 编码器部分 (只执行一次)
            padded_img, _, (new_h, new_w), scale = self._resize_longest_side_and_pad(image_rgb, self.input_size)
            enc_input_3d = (padded_img.astype(np.float32) / 255.0)
            enc_input = np.expand_dims(enc_input_3d, axis=0).transpose(0, 3, 1, 2)
            image_embeddings = self.enc_session.run(None, {self.enc_input_name: enc_input})[0]

            # 2. ROI 预过滤
            gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            roi_mask = self._find_petri_dish(gray_image)
            grid_points = self._generate_grid_points(W, H, self.points_per_side)
            points_inside_roi = grid_points[roi_mask[grid_points[:, 1], grid_points[:, 0]] > 0]
            
            if len(points_inside_roi) == 0:
                logging.warning("No grid points found inside the ROI. Returning empty array.")
                return np.array([])

            # 3. 解码器推理 (对每个点) - 注意：此处为快速落地，使用串行循环
            all_masks_data = []
            for point in points_inside_roi:
                results_for_point = self._process_single_point(
                    point, image_embeddings, scale, H, W, new_h, new_w
                )
                if results_for_point:
                    all_masks_data.extend(results_for_point)

            if not all_masks_data:
                logging.info("Decoder did not produce any masks above threshold.")
                return np.array([])

            # 4. 两阶段后处理
            base_filtered = self._base_postprocess(all_masks_data, self.min_mask_region_area, self.box_nms_thresh)
            if not base_filtered: return np.array([])
            
            advanced_filtered = self._advanced_postprocess(base_filtered, (H, W), self.max_area_ratio, self.min_circularity)

            # 5. 格式转换，满足前端“数据契约”
            final_masks_list = []
            roi_mask_bool = roi_mask.astype(bool)
            for data in advanced_filtered:
                mask = np.logical_and(data['mask'], roi_mask_bool)
                if np.count_nonzero(mask) > self.min_mask_region_area:
                    final_masks_list.append(mask.astype(np.uint8)) # 值为 0 或 1
            
            if not final_masks_list:
                return np.array([])

            final_masks_stack = np.stack(final_masks_list, axis=0)
            
            assert final_masks_stack.ndim == 3, "Return array must have 3 dimensions (N, H, W)"
            assert final_masks_stack.shape[1] == H, "Mask height must match original image height"
            assert final_masks_stack.shape[2] == W, "Mask width must match original image width"

            logging.info(f"Successfully generated {final_masks_stack.shape[0]} masks.")
            return final_masks_stack

        except Exception as e:
            logging.error(f"Error during predict_masks: {e}", exc_info=True)
            return np.array([])

    def split_mask_with_line(self, target_mask: np.ndarray, line_points: list) -> list[np.ndarray] | None:
        """
        【功能对接点 2: 画线分割】 - 使用前端提供的稳定非 AI 版本
        """
        try:
            if not line_points or len(line_points) < 2:
                return None

            line_mask = np.zeros_like(target_mask, dtype=np.uint8)
            pts = np.array(line_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(line_mask, [pts], isClosed=False, color=255, thickness=5)

            cut_mask = cv2.bitwise_and(target_mask, cv2.bitwise_not(line_mask))
            num_labels, labels_im = cv2.connectedComponents(cut_mask)

            if num_labels < 3: return None
            
            new_masks = []
            for label_id in range(1, num_labels):
                new_mask = (labels_im == label_id).astype(np.uint8)
                if np.sum(new_mask) > 10:
                    new_masks.append(new_mask)
            
            return new_masks if new_masks else None

        except Exception as e:
            logging.error(f"Error during split_mask_with_line: {e}", exc_info=True)
            return None

    # --- 内部辅助方法 (全部从你的脚本移植而来) ---

    def _process_single_point(self, point_data, image_embeddings, scale, H, W, new_h, new_w):
        """处理单个点的解码器推理"""
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

            feeds = {
                "image_embeddings": image_embeddings, "point_coords": point_coords,
                "point_labels": point_labels, "mask_input": mask_input,
                "has_mask_input": has_mask_input, "orig_im_size": orig_im_size,
            }
            masks, ious, low_res_logits = self.dec_session.run(None, feeds)

            results_for_point = []
            for i in range(masks.shape[1]):
                iou = float(ious[0, i])
                if iou < self.pred_iou_thresh: continue
                
                logits_256 = low_res_logits[0, i]
                prob_256 = _sigmoid(np.nan_to_num(np.clip(logits_256, -100, 100)))
                prob_1024 = cv2.resize(prob_256, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
                prob_padded = prob_1024[:new_h, :new_w]
                final_mask_prob = cv2.resize(prob_padded, (W, H), interpolation=cv2.INTER_LINEAR)
                final_mask = (final_mask_prob >= self.mask_threshold)

                if np.count_nonzero(final_mask) > 0:
                    results_for_point.append({"mask": final_mask, "iou": iou})
            
            return results_for_point

    def _find_petri_dish(self, gray_image: np.ndarray) -> np.ndarray:
        """检测培养皿作为ROI"""
        h, w = gray_image.shape
        blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h,
            param1=100, param2=80, minRadius=int(w * 0.2), maxRadius=int(w * 0.5)
        )
        roi_mask = np.zeros_like(gray_image, dtype=np.uint8)
        if circles is not None:
            cx, cy, r = np.uint16(np.around(circles))[0, 0]
            cv2.circle(roi_mask, (cx, cy), r, 255, -1)
        else:
            roi_mask.fill(255)
        return roi_mask

    def _generate_grid_points(self, width: int, height: int, points_per_side: int) -> np.ndarray:
        """生成网格采样点"""
        grid_x = np.linspace(0, width - 1, points_per_side, dtype=np.float32)
        grid_y = np.linspace(0, height - 1, points_per_side, dtype=np.float32)
        xv, yv = np.meshgrid(grid_x, grid_y, indexing="xy")
        return np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1).astype(np.int32)
    
    def _filter_small_regions(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """过滤小面积连通域"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1: return mask
        keep_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
        return np.isin(labels, keep_labels) if len(keep_labels) > 0 else np.zeros_like(mask, dtype=bool)

    def _calculate_box_iou(self, boxA, boxB) -> float:
        """计算两个边界框的IoU"""
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        union_area = float(boxA_area + boxB_area - inter_area)
        return inter_area / union_area if union_area > 0 else 0.0

    def _base_postprocess(self, masks_data: list, min_mask_region_area: int, box_nms_thresh: float) -> list:
        """第一阶段后处理：过滤小区域并应用 Box NMS"""
        filtered = []
        for data in masks_data:
            cleaned_mask = self._filter_small_regions(data["mask"], min_mask_region_area)
            if np.count_nonzero(cleaned_mask) == 0: continue
            rows, cols = np.any(cleaned_mask, axis=1), np.any(cleaned_mask, axis=0)
            if not np.any(rows) or not np.any(cols): continue
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            filtered.append({
                "mask": cleaned_mask, "area": np.count_nonzero(cleaned_mask),
                "bbox": [x_min, y_min, x_max, y_max], "iou": data["iou"],
            })
        if not filtered: return []
        filtered.sort(key=lambda x: x["iou"], reverse=True)
        final_masks = []
        while filtered:
            best = filtered.pop(0)
            final_masks.append(best)
            filtered = [o for o in filtered if self._calculate_box_iou(best["bbox"], o["bbox"]) < box_nms_thresh]
        return final_masks

    def _advanced_postprocess(self, masks_data: list, original_size: tuple, max_area_ratio: float, min_circularity: float) -> list:
        """第二阶段后处理：应用面积比例和圆形度过滤"""
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
        """保持纵横比缩放图像并填充到目标尺寸"""
        H, W = img.shape[:2]
        scale = float(target_length) / max(H, W)
        new_w, new_h = int(round(W * scale)), int(round(H * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((target_length, target_length, 3), dtype=resized.dtype)
        padded[:new_h, :new_w] = resized
        return padded, (H, W), (new_h, new_w), scale