"""Prompt-free AutomaticMaskGenerator core for SAM-family ONNX models.

Framework-agnostic: it takes a `decode_batch` callable and returns
low-resolution boolean masks. No Qt, no onnxruntime, no SAM-specific imports.
"""
import numpy as np


def build_point_grid(points_per_side, height, width):
    """Return an (N, 2) float32 array of (x, y) pixel coords on a regular grid."""
    frac = (np.arange(points_per_side) + 0.5) / points_per_side
    xs = frac * width
    ys = frac * height
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    return grid.astype(np.float32)


def _stability_score(logits, mask_threshold, offset):
    """IoU between masks thresholded at (t + offset) and (t - offset). (K,) array."""
    high = logits > (mask_threshold + offset)
    low = logits > (mask_threshold - offset)
    inter = np.logical_and(high, low).sum(axis=(-2, -1)).astype(np.float64)
    union = np.logical_or(high, low).sum(axis=(-2, -1)).astype(np.float64)
    return np.where(union > 0, inter / np.maximum(union, 1.0), 0.0)


def _mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _nms(masks, scores, iou_thresh):
    """Greedy mask-IoU NMS. `masks` list of bool arrays, `scores` (N,). Returns kept indices."""
    order = list(np.argsort(scores)[::-1])
    suppressed = [False] * len(masks)
    keep = []
    for idx_pos, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order[idx_pos + 1:]:
            if suppressed[j]:
                continue
            if _mask_iou(masks[i], masks[j]) > iou_thresh:
                suppressed[j] = True
    return keep


def generate(
    decode_batch,
    image_hw,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    stability_score_offset=1.0,
    box_nms_thresh=0.7,
    mask_threshold=0.0,
    points_per_chunk=256,
    should_stop=None,
):
    """Generate deduped low-resolution boolean masks for a whole image.

    decode_batch(points_xy[K,2]) -> (logits[K,h,w] float, ious[K] float).
    The grid is processed in chunks of `points_per_chunk` points so peak memory
    stays bounded regardless of `points_per_side`; each chunk is reduced to kept
    boolean masks before the next chunk is decoded. If `should_stop` is provided
    and returns True, generation aborts and returns an empty list.
    """
    height, width = image_hw
    grid = build_point_grid(points_per_side, height, width)

    kept_masks = []
    kept_scores = []
    for start in range(0, len(grid), points_per_chunk):
        if should_stop is not None and should_stop():
            return []
        chunk = grid[start:start + points_per_chunk]
        logits, ious = decode_batch(chunk)
        logits = np.asarray(logits, dtype=np.float32)
        ious = np.asarray(ious, dtype=np.float32)

        stability = _stability_score(
            logits, mask_threshold, stability_score_offset
        )
        keep = (ious >= pred_iou_thresh) & (
            stability >= stability_score_thresh
        )
        for i in np.nonzero(keep)[0]:
            kept_masks.append(logits[i] > mask_threshold)
            kept_scores.append(float(ious[i]))

    if not kept_masks:
        return []

    kept_idx = _nms(kept_masks, np.asarray(kept_scores), box_nms_thresh)
    return [kept_masks[i] for i in kept_idx]
