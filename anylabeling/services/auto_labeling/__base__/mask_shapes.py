"""Convert a binary mask into AnyLabeling Shape objects (polygon or contour)."""
import cv2
import numpy as np
from PyQt6 import QtCore

from anylabeling.views.labeling.shape import Shape


def masks_to_shapes(
    mask,
    output_mode,
    epsilon=0.001,
    min_area=0,
    label="AUTOLABEL_OBJECT",
):
    """Return a list of Shape objects for every external contour of `mask`.

    output_mode == "contour" -> open linestrip; anything else -> closed polygon.
    Contours with area < min_area are dropped.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    shapes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        approx = cv2.approxPolyDP(
            contour, epsilon * cv2.arcLength(contour, True), True
        )
        points = approx.reshape(-1, 2)
        min_points = 2 if output_mode == "contour" else 3
        if len(points) < min_points:
            continue

        shape = Shape(flags={})
        for x, y in points:
            shape.add_point(QtCore.QPointF(int(x), int(y)))
        if output_mode == "contour":
            shape.shape_type = "linestrip"
            shape.closed = False
        else:
            shape.shape_type = "polygon"
            shape.closed = True
        shape.fill_color = "#000000"
        shape.line_color = "#000000"
        shape.label = label
        shape.selected = False
        shapes.append(shape)
    return shapes
