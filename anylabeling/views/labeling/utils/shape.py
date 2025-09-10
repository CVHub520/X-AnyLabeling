import json
import math
import uuid

import numpy as np
import PIL.Image
import PIL.ImageDraw

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QProgressDialog

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import get_bounding_boxes
from anylabeling.views.labeling.widgets import PolygonSidesDialog, Popup
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import *
from anylabeling.services.auto_labeling.utils import calculate_rotation_theta


def get_conversion_params(self, mode: str):
    """Get parameters required for specific conversion modes.

    Args:
        mode (str): The conversion mode

    Returns:
        dict: Parameters dictionary, or None if user cancelled
    """
    if mode == "circle_to_polygon":
        dialog = PolygonSidesDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            return {"num_sides": dialog.get_value()}
        else:
            return None

    return {}


def shape_conversion(self, mode):
    label_file_list = self.get_label_file_list()
    if len(label_file_list) == 0:
        return

    params = get_conversion_params(self, mode)
    if params is None:
        return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be changed"))
    response.setInformativeText(
        self.tr("Are you sure you want to perform this conversion?")
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Converting..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(get_progress_dialog_style())
    progress_dialog.show()

    try:
        for i, label_file in enumerate(label_file_list):
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for j in range(len(data["shapes"])):

                if mode == "hbb_to_obb" and (
                    data["shapes"][j]["shape_type"] == "rectangle"
                ):
                    data["shapes"][j]["shape_type"] = "rotation"
                    data["shapes"][j]["direction"] = 0

                elif mode == "obb_to_hbb" and (
                    data["shapes"][j]["shape_type"] == "rotation"
                ):
                    del data["shapes"][j]["direction"]
                    data["shapes"][j]["shape_type"] = "rectangle"
                    points = np.array(data["shapes"][j]["points"])
                    if len(points) != 4:
                        continue
                    xmin = int(np.min(points[:, 0]))
                    ymin = int(np.min(points[:, 1]))
                    xmax = int(np.max(points[:, 0]))
                    ymax = int(np.max(points[:, 1]))
                    data["shapes"][j]["points"] = [
                        [xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax],
                    ]

                elif mode == "polygon_to_hbb" and (
                    data["shapes"][j]["shape_type"] == "polygon"
                ):
                    data["shapes"][j]["shape_type"] = "rectangle"
                    points = np.array(data["shapes"][j]["points"])
                    if len(points) < 3:
                        continue
                    xmin = int(np.min(points[:, 0]))
                    ymin = int(np.min(points[:, 1]))
                    xmax = int(np.max(points[:, 0]))
                    ymax = int(np.max(points[:, 1]))
                    data["shapes"][j]["points"] = [
                        [xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax],
                    ]

                elif mode == "polygon_to_obb" and (
                    data["shapes"][j]["shape_type"] == "polygon"
                ):
                    points = np.array(data["shapes"][j]["points"])
                    contours = points.reshape((-1, 1, 2)).astype(np.float32)
                    _, rotation_box = get_bounding_boxes(contours)
                    data["shapes"][j]["shape_type"] = "rotation"
                    data["shapes"][j]["points"] = rotation_box.tolist()
                    data["shapes"][j]["direction"] = calculate_rotation_theta(
                        rotation_box
                    )

                elif mode == "circle_to_polygon" and (
                    data["shapes"][j]["shape_type"] == "circle"
                ):
                    points = np.array(data["shapes"][j]["points"])
                    if len(points) != 2:
                        continue

                    center_x, center_y = points[0]
                    edge_x, edge_y = points[1]
                    radius = math.sqrt(
                        (edge_x - center_x) ** 2 + (edge_y - center_y) ** 2
                    )

                    num_sides = params.get("num_sides", 32)
                    polygon_points = []
                    for i in range(num_sides):
                        angle = 2 * math.pi * i / num_sides
                        x = center_x + radius * math.cos(angle)
                        y = center_y + radius * math.sin(angle)
                        polygon_points.append([x, y])

                    data["shapes"][j]["shape_type"] = "polygon"
                    data["shapes"][j]["points"] = polygon_points

            with open(label_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        popup = Popup(
            self.tr("Conversion completed successfully!"),
            self,
            msec=1000,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

        self.load_file(self.filename)

    except Exception as e:
        logger.error(f"Error occurred while converting shapes: {e}")
        popup = Popup(
            self.tr("Error occurred while converting shapes!"),
            self,
            msec=1000,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def polygons_to_mask(img_shape, polygons, shape_type=None):
    logger.warning(
        "The 'polygons_to_mask' function is deprecated, "
        "use 'shape_to_mask' instead."
    )
    return shape_to_mask(img_shape, points=polygons, shape_type=shape_type)


def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "rotation":
        assert len(xy) == 4, "Shape of shape_type=rotation must have 4 points"
        draw.polygon(xy=xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def masks_to_bboxes(masks):
    if masks.ndim != 3:
        raise ValueError(f"masks.ndim must be 3, but it is {masks.ndim}")
    if masks.dtype != bool:
        raise ValueError(
            f"masks.dtype must be bool type, but it is {masks.dtype}"
        )
    bboxes = []
    for mask in masks:
        where = np.argwhere(mask)
        (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        bboxes.append((y1, x1, y2, x2))
    bboxes = np.asarray(bboxes, dtype=np.float32)
    return bboxes


def rectangle_from_diagonal(diagonal_vertices):
    """
    Generate rectangle vertices from diagonal vertices.

    Parameters:
    - diagonal_vertices (list of lists):
        List containing two points representing the diagonal vertices.

    Returns:
    - list of lists:
        List containing four points representing the rectangle's four corners.
        [tl -> tr -> br -> bl]
    """
    x1, y1 = diagonal_vertices[0]
    x2, y2 = diagonal_vertices[1]

    # Creating the four-point representation
    rectangle_vertices = [
        [x1, y1],  # Top-left
        [x2, y1],  # Top-right
        [x2, y2],  # Bottom-right
        [x1, y2],  # Bottom-left
    ]

    return rectangle_vertices
