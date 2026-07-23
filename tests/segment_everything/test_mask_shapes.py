import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from anylabeling.services.auto_labeling.__base__ import mask_shapes


def _square_mask(size=64, lo=16, hi=48):
    m = np.zeros((size, size), dtype=np.uint8)
    m[lo:hi, lo:hi] = 255
    return m


def test_polygon_mode_produces_closed_polygon():
    shapes = mask_shapes.masks_to_shapes(_square_mask(), "polygon", min_area=10)
    assert len(shapes) == 1
    assert shapes[0].shape_type == "polygon"
    assert shapes[0].closed is True
    assert len(shapes[0].points) >= 4


def test_contour_mode_produces_open_linestrip():
    shapes = mask_shapes.masks_to_shapes(_square_mask(), "contour", min_area=10)
    assert len(shapes) == 1
    assert shapes[0].shape_type == "linestrip"
    assert shapes[0].closed is False


def test_min_area_filters_small_regions():
    m = np.zeros((64, 64), dtype=np.uint8)
    m[10:13, 10:13] = 255  # 9-px region
    shapes = mask_shapes.masks_to_shapes(m, "polygon", min_area=100)
    assert shapes == []
