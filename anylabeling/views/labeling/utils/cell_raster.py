"""Rasterize half-integer orthogonal contours by pixel-center inclusion."""

import numpy as np


def is_cell_polygon(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        return False
    delta = np.roll(points, -1, axis=0) - points
    return bool(
        np.isfinite(points).all()
        and np.allclose(points - 0.5, np.round(points - 0.5), atol=1e-7)
        and np.all((delta[:, 0] == 0) | (delta[:, 1] == 0))
    )


def fill_cell_polygon(mask, points, color):
    points = np.asarray(points, dtype=float)
    ends = np.roll(points, -1, axis=0)
    vertical = points[:, 0] == ends[:, 0]
    xs = points[vertical, 0]
    low = np.minimum(points[vertical, 1], ends[vertical, 1])
    high = np.maximum(points[vertical, 1], ends[vertical, 1])
    start = max(0, int(np.ceil(points[:, 1].min())))
    stop = min(mask.shape[0], int(np.floor(points[:, 1].max())) + 1)
    for y in range(start, stop):
        crossings = np.sort(xs[(low <= y) & (y < high)])
        for left, right in zip(crossings[::2], crossings[1::2]):
            x0 = max(0, int(np.ceil(left)))
            x1 = min(mask.shape[1], int(np.floor(right)) + 1)
            mask[y, x0:x1] = color
    return mask
