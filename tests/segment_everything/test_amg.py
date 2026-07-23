import numpy as np
from anylabeling.services.auto_labeling.__base__ import amg


def _stub_decode_factory(h=32, w=32):
    """Two objects at grid-normalized regions A and B in a 32x32 low-res mask.
    A: rows/cols 4..12 ; B: rows/cols 20..28. A point 'hits' an object if it
    falls inside that object's pixel box in a 320x320 image."""
    def decode_batch(points_xy):
        K = len(points_xy)
        logits = np.full((K, h, w), -10.0, dtype=np.float32)
        ious = np.full((K,), 0.5, dtype=np.float32)
        for k, (x, y) in enumerate(points_xy):
            if 40 <= x <= 120 and 40 <= y <= 120:
                logits[k, 4:12, 4:12] = 10.0
                ious[k] = 0.95
            elif 200 <= x <= 280 and 200 <= y <= 280:
                logits[k, 20:28, 20:28] = 10.0
                ious[k] = 0.95
        return logits, ious
    return decode_batch


def test_build_point_grid_shape_and_range():
    grid = amg.build_point_grid(4, 320, 320)
    assert grid.shape == (16, 2)
    assert grid[:, 0].min() > 0 and grid[:, 0].max() < 320


def test_generate_returns_two_deduped_masks():
    decode = _stub_decode_factory()
    masks = amg.generate(decode, (320, 320), points_per_side=32)
    assert len(masks) == 2
    for m in masks:
        assert m.dtype == np.bool_
        assert m.sum() > 0


def test_generate_filters_low_iou_only_background():
    def decode(points_xy):
        K = len(points_xy)
        return np.full((K, 32, 32), -10.0, dtype=np.float32), np.full((K,), 0.5, np.float32)
    masks = amg.generate(decode, (320, 320), points_per_side=16)
    assert masks == []
