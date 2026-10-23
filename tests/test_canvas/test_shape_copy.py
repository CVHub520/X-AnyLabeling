import copy

import numpy as np
import pytest
from PyQt6 import QtCore, QtGui

from anylabeling.views.labeling.shape import Shape


def test_copy_preserves_aliases_and_independent_editable_state():
    shape = Shape(label="object", flags={"reviewed": True})
    point = QtCore.QPointF(10, 20)
    shape.points = [point, QtCore.QPointF(30, 40), point]
    shape.line_color = QtGui.QColor(10, 20, 30, 40)
    shape.fill_color = shape.line_color
    shape.mask = np.ones((8, 8), dtype=np.uint8)
    shape.attributes = {"nested": ["original"]}
    shape.other_data = {
        "point": point,
        "points": shape.points,
        "color": shape.line_color,
        "owner": shape,
    }
    shape.close()

    cloned = shape.copy()

    assert cloned is not shape
    assert cloned.is_closed()
    assert cloned.points == shape.points
    assert cloned.points[0] is cloned.points[2]
    assert cloned.other_data["point"] is cloned.points[0]
    assert cloned.other_data["points"] is cloned.points
    assert cloned.other_data["owner"] is cloned
    assert cloned.fill_color is cloned.line_color
    assert cloned.other_data["color"] is cloned.line_color
    cloned.points[0].setX(100)
    cloned.points.append(QtCore.QPointF())
    cloned.line_color.setRed(200)
    cloned.mask[0, 0] = 0
    cloned.attributes["nested"].append("changed")
    cloned.flags["reviewed"] = False
    assert shape.points[0].x() == 10
    assert len(shape.points) == 3
    assert shape.line_color.red() == 10
    assert shape.mask[0, 0] == 1
    assert shape.attributes == {"nested": ["original"]}
    assert shape.flags == {"reviewed": True}


@pytest.mark.parametrize("shape_type", Shape.get_supported_shape())
def test_copy_matches_deepcopy_for_all_shape_types(shape_type):
    shape = Shape(label="object", shape_type=shape_type)
    shape.points = [QtCore.QPointF(i * 10, i % 3 * 10) for i in range(8)]
    shape.center = QtCore.QPointF(20, 30)
    shape.close()
    shape.locked = True
    shape.selected = True
    shape.line_color = QtGui.QColor(100, 200, 50)
    expected = copy.deepcopy(shape)
    actual = shape.copy()
    assert actual.to_dict() == expected.to_dict()
    assert actual.__dict__ == expected.__dict__


@pytest.mark.parametrize(
    "shape_type", ["polygon", "rectangle", "line", "linestrip", "point"]
)
@pytest.mark.parametrize("count", [0, 1, 2, 3, 64, 256])
def test_vectorized_path_preserves_geometry(shape_type, count):
    shape = Shape(shape_type=shape_type)
    shape.points = [QtCore.QPointF(i % 7, i // 7) for i in range(count)]
    expected = QtGui.QPainterPath()
    if shape.points:
        expected.moveTo(shape.points[0])
        for point in shape.points[1:]:
            expected.lineTo(point)
    actual = shape.make_path()
    assert actual == expected
    assert shape.bounding_rect() == expected.boundingRect()


@pytest.mark.parametrize(
    "coordinates",
    [
        [(10, 20)] * 5,
        [(20, 20), (80, 20), (80, 20), (80, 80), (20, 80)],
        [(10, 20), (30, 40), (10, 20)],
    ],
)
def test_repeated_vertices_preserve_rendering_and_hit_testing(coordinates):
    shape = Shape()
    shape.points = [QtCore.QPointF(*point) for point in coordinates]
    expected = QtGui.QPainterPath(shape.points[0])
    for point in shape.points[1:]:
        expected.lineTo(point)
    actual = shape.make_path()
    assert actual.boundingRect() == expected.boundingRect()
    for x in range(0, 100, 5):
        for y in range(0, 100, 5):
            point = QtCore.QPointF(x, y)
            assert actual.contains(point) == expected.contains(point)
    images = []
    for path in (expected, actual):
        image = QtGui.QImage(100, 100, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor("green"), 2))
        painter.setBrush(QtGui.QColor(100, 200, 100, 128))
        painter.drawPath(path)
        painter.end()
        images.append(image)
    assert images[0] == images[1]
