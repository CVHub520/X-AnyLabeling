from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import onnxruntime


class SegmentAnythingONNX:
    """
    Segmentation model using SegmentAnything, adapted for automatic mask generation.
    This version uses a robust area filtering baseline (full image area) to prevent
    issues from inconsistent Hough Circle Transform results.
    """

    def __init__(
        self,
        encoder_model_path: str,
        decoder_model_path: str,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.65,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 20,
        box_nms_thresh: float = 0.7,
        filter_by_area: bool = True,
        max_area_ratio: float = 0.05,
        filter_by_circularity: bool = True,
        min_circularity: float = 0.25,
        inter_op_threads: int = 1,
        intra_op_threads: int = 0,
        prefer_gpu: bool = True,
        use_iobinding: bool = False,
    ) -> None:
        self.target_size = 1024
        self.points_per_side = int(points_per_side)
        self.pred_iou_thresh = float(pred_iou_thresh)
        self.stability_score_thresh = float(stability_score_thresh)
        self.min_mask_region_area = int(min_mask_region_area)
        self.box_nms_thresh = float(box_nms_thresh)
        self.use_iobinding = bool(use_iobinding)
        self.filter_by_area = filter_by_area
        self.max_area_ratio = max_area_ratio
        self.filter_by_circularity = filter_by_circularity
        self.min_circularity = min_circularity

        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        if inter_op_threads is not None: so.inter_op_num_threads = int(inter_op_threads)
        if intra_op_threads is not None: so.intra_op_num_threads = int(intra_op_threads)

        print("Using simple provider selection from Code 1...")
        providers = onnxruntime.get_available_providers()
        providers = [p for p in providers if p != "TensorrtExecutionProvider"]
        print(f"Selected providers: {providers}")

        self.encoder_session = onnxruntime.InferenceSession(encoder_model_path, sess_options=so, providers=providers)
        self.decoder_session = onnxruntime.InferenceSession(decoder_model_path, sess_options=so, providers=providers)

        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self._dec_in_names = {i.name: i.name for i in self.decoder_session.get_inputs()}
        self._empty_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        self._empty_has_mask = np.zeros(1, dtype=np.float32)
        self._enc_io = self.encoder_session.io_binding() if self.use_iobinding else None
        self._dec_io = self.decoder_session.io_binding() if self.use_iobinding else None

    def run_encoder(self, image: np.ndarray) -> np.ndarray:
        output = self.encoder_session.run(None, {self.encoder_input_name: image})
        return output[0]

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        scale = float(long_side_length) / float(max(oldh, oldw))
        newh = int(oldh * scale + 0.5)
        neww = int(oldw * scale + 0.5)
        return newh, neww

    def run_decoder(
        self,
        image_embedding: np.ndarray,
        original_size: Tuple[int, int],
        prompt: List[Dict[str, Any]],
    ) -> tuple:
        input_points, input_labels = self._get_input_points(prompt)
        onnx_coord = np.concatenate([input_points, np.array([[0.0, 0.0]], dtype=np.float32)], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_labels, np.array([-1], dtype=np.float32)], axis=0)[None, :].astype(np.float32)
        
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(old_h, old_w, self.target_size)
        coords = onnx_coord.astype(np.float32, copy=True)
        coords[..., 0] *= new_w / float(old_w)
        coords[..., 1] *= new_h / float(old_h)

        decoder_inputs = {
            self._dec_in_names.get("image_embeddings", "image_embeddings"): image_embedding,
            self._dec_in_names.get("point_coords", "point_coords"): coords,
            self._dec_in_names.get("point_labels", "point_labels"): onnx_label,
            self._dec_in_names.get("mask_input", "mask_input"): self._empty_mask_input,
            self._dec_in_names.get("has_mask_input", "has_mask_input"): self._empty_has_mask,
            self._dec_in_names.get("orig_im_size", "orig_im_size"): np.array(original_size, dtype=np.float32),
        }
        masks, iou_predictions, low_res_masks = self.decoder_session.run(None, decoder_inputs)
        
        input_size = (self.target_size, self.target_size)
        upscaled = [cv2.resize(lr, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR) for lr in low_res_masks[0]]
        upscaled_masks = np.stack(upscaled, axis=0)
        pre_h, pre_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_size)
        cropped_masks = upscaled_masks[:, :pre_h, :pre_w]
        orig_h, orig_w = original_size
        final_masks = np.empty((cropped_masks.shape[0], orig_h, orig_w), dtype=np.float32)
        for i in range(cropped_masks.shape[0]):
            final_masks[i] = cv2.resize(cropped_masks[i], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        final_masks = (final_masks > 0.0)[np.newaxis, :, :, :]
        return final_masks, iou_predictions

    def _find_petri_dish(self, gray_image: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int]]]:
        h, w = gray_image.shape
        
        # <<< 修正点 >>>
        blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=h,
            param1=100, param2=80, minRadius=int(w * 0.2), maxRadius=int(w * 0.5)
        )
        
        roi_mask = np.zeros_like(gray_image, dtype=np.uint8)
        circle_params = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            cx, cy, r = circles[0, 0]
            # <<< 修正点 >>>
            cv2.circle(roi_mask, (cx, cy), r, 255, -1)
            circle_params = (cx, cy, r)
            print(f"Petri dish detected at ({cx}, {cy}) with radius {r}.")
        else:
            print("Warning: No petri dish detected. Using full image as ROI.")
            roi_mask.fill(255)
        return roi_mask, circle_params

    def encode(self, cv_image: np.ndarray) -> Dict[str, Any]:
        original_size = cv_image.shape[:2]
        new_h, new_w = self.get_preprocess_shape(original_size[0], original_size[1], self.target_size)
        
        # <<< 修正点 >>>
        interpolation = cv2.INTER_AREA if (new_h < original_size[0] or new_w < original_size[1]) else cv2.INTER_LINEAR
        resized_image = cv2.resize(cv_image, (new_w, new_h), interpolation=interpolation)
        
        padded_image = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        padded_image[:new_h, :new_w, :] = resized_image
        input_tensor = padded_image.astype(np.float32, copy=False)
        image_embedding = self.run_encoder(input_tensor)
        
        # <<< 修正点 >>>
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY) if len(cv_image.shape) == 3 else cv_image

        return {
            "image_embedding": image_embedding,
            "original_size": original_size,
            "gray_image": gray_image,
        }

    def predict_masks(self, embedding: Dict[str, Any], prompt: List[Dict[str, Any]] = None) -> np.ndarray:
        image_embedding = embedding["image_embedding"]
        original_size = embedding["original_size"]
        gray_image = embedding["gray_image"]

        roi_mask, circle_params = self._find_petri_dish(gray_image)
        grid_points = self._generate_grid_points(original_size[1], original_size[0])
        points_inside_roi = grid_points[roi_mask[grid_points[:, 1], grid_points[:, 0]] == 255]
        
        num_total_points, num_roi_points = len(grid_points), len(points_inside_roi)
        reduction = (1 - num_roi_points / num_total_points) if num_total_points > 0 else 0
        print(f"Total grid points: {num_total_points}. Points inside ROI: {num_roi_points}. Reduction: {reduction:.2%}")

        if num_roi_points == 0:
            return np.empty((1, 0, *original_size), dtype=bool)

        all_masks_data: List[Dict[str, Any]] = []
        for point in points_inside_roi:
            point_prompt = [{"type": "point", "data": point, "label": 1}]
            masks, ious = self.run_decoder(image_embedding, original_size, point_prompt)
            
            ms, iscores = masks[0], ious[0]
            for i in range(ms.shape[0]):
                iou = float(iscores[i])
                if iou < self.pred_iou_thresh:
                    continue
                all_masks_data.append({"mask": ms[i], "iou": iou})
        
        if not all_masks_data:
            return np.empty((1, 0, *original_size), dtype=bool)

        final_masks_info = self._postprocess_masks(all_masks_data, circle_params, original_size)
        
        final_cleaned_masks = []
        roi_mask_bool = roi_mask.astype(bool, copy=False)
        for data in final_masks_info:
            mask = data["mask"]
            cleaned_mask = np.logical_and(mask, roi_mask_bool)
            if np.count_nonzero(cleaned_mask) > self.min_mask_region_area:
                final_cleaned_masks.append(cleaned_mask)

        if not final_cleaned_masks:
            return np.empty((1, 0, *original_size), dtype=bool)

        return np.stack(final_cleaned_masks, axis=0)[np.newaxis, :, :, :]

    @staticmethod
    def _get_input_points(prompt: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        pts, lbs = [], []
        for mark in prompt:
            if mark["type"] == "point":
                pts.append(mark["data"])
                lbs.append(mark["label"])
            elif mark["type"] == "rectangle":
                x1, y1, x2, y2 = mark["data"]
                pts.extend([[x1, y1], [x2, y2]])
                lbs.extend([2, 3])
        if not pts:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32), np.asarray(lbs, dtype=np.float32)

    def _generate_grid_points(self, width: int, height: int) -> np.ndarray:
        grid_x = np.linspace(0, width - 1, self.points_per_side, dtype=np.float32)
        grid_y = np.linspace(0, height - 1, self.points_per_side, dtype=np.float32)
        xv, yv = np.meshgrid(grid_x, grid_y, indexing="xy")
        points = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)
        return points.astype(np.int32, copy=False)

    def _postprocess_masks(
        self,
        masks_data: List[Dict[str, Any]],
        circle_params: Optional[Tuple[int, int, int]],
        original_size: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        
        base_filtered_info = self._base_postprocess(masks_data)

        if not base_filtered_info:
            return []

        reference_area = float(original_size[0] * original_size[1])
        print(f"Using full image area as reference for filtering: {reference_area}")

        advanced_filtered_info = []
        for data in base_filtered_info:
            mask = data["mask"]
            
            mask_uint8 = mask.astype(np.uint8, copy=False)
            
            # <<< 修正点 >>>
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            
            if self.filter_by_area:
                area_ratio = area / reference_area
                if area_ratio > self.max_area_ratio:
                    continue

            if self.filter_by_circularity:
                # <<< 修正点 >>>
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * area / (perimeter**2)
                if circularity < self.min_circularity:
                    continue
            
            advanced_filtered_info.append(data)
            
        return advanced_filtered_info

    def _base_postprocess(self, masks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []
        for data in masks_data:
            mask = data["mask"]
            area_px = int(np.count_nonzero(mask))
            if area_px < self.min_mask_region_area:
                continue
            mask = self._filter_small_regions(mask, self.min_mask_region_area)
            area_px = int(np.count_nonzero(mask))
            if area_px == 0:
                continue
            rows = np.flatnonzero(np.any(mask, axis=1))
            cols = np.flatnonzero(np.any(mask, axis=0))
            y_min, y_max = int(rows[0]), int(rows[-1])
            x_min, x_max = int(cols[0]), int(cols[-1])
            filtered.append({
                "mask": mask, "area": area_px,
                "bbox": [x_min, y_min, x_max, y_max], "iou": float(data["iou"]),
            })
        if not filtered:
            return []
        filtered.sort(key=lambda x: x["iou"], reverse=True)
        final_masks: List[Dict[str, Any]] = []
        while filtered:
            best = filtered.pop(0)
            final_masks.append(best)
            remain = []
            for other in filtered:
                if self._calculate_box_iou(best["bbox"], other["bbox"]) < self.box_nms_thresh:
                    remain.append(other)
            filtered = remain
        return final_masks

    @staticmethod
    def _filter_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
        m = mask.astype(np.uint8, copy=False)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num_labels <= 1:
            return mask.astype(bool, copy=False)
        keep = stats[1:, cv2.CC_STAT_AREA] >= min_area
        if not np.any(keep):
            return np.zeros_like(mask, dtype=bool)
        new_mask = np.zeros_like(mask, dtype=bool)
        for i, k in enumerate(keep, start=1):
            if k:
                new_mask |= (labels == i)
        return new_mask

    @staticmethod
    def _calculate_box_iou(boxA: List[int], boxB: List[int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter_w = max(0, xB - xA + 1)
        inter_h = max(0, yB - yA + 1)
        inter = inter_w * inter_h
        areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        denom = float(areaA + areaB - inter) if (areaA + areaB - inter) > 0 else 1.0
        return float(inter) / denom