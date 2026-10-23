import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets.canvas import Canvas

SHAPE_TYPES = [
    "polygon",
    "rectangle",
    "rotation",
    "line",
    "linestrip",
    "quadrilateral",
]


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def canvas(app):
    widget = Canvas(parent=None)
    widget.pixmap = QtGui.QPixmap(100, 80)
    widget.pixmap.fill(QtGui.QColor("white"))
    widget.resize(1600, 1400)
    widget.scale = 10.0
    widget.pan_offset = QtCore.QPointF(35.5, 27.25)
    widget.set_editing(False)
    widget.snapping = False
    yield widget
    widget.restore_cursor()
    widget.close()
    widget.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def send_mouse(
    canvas, point, move=False, modifiers=QtCore.Qt.KeyboardModifier.NoModifier
):
    pos = (QtCore.QPointF(*point) + canvas.offset_to_center()) * canvas.scale
    kind = (
        QtCore.QEvent.Type.MouseMove
        if move
        else QtCore.QEvent.Type.MouseButtonPress
    )
    button = (
        QtCore.Qt.MouseButton.NoButton
        if move
        else QtCore.Qt.MouseButton.LeftButton
    )
    event = QtGui.QMouseEvent(kind, pos, pos, button, button, modifiers)
    if move:
        canvas.mouseMoveEvent(event)
    else:
        canvas.mousePressEvent(event)


def assert_points(points, expected):
    assert len(points) == len(expected)
    for point, (x, y) in zip(points, expected):
        assert point.x() == pytest.approx(x)
        assert point.y() == pytest.approx(y)


def finish_shape(canvas, shape_type, points):
    shape = Shape(shape_type=shape_type)
    shape.points = [QtCore.QPointF(*point) for point in points]
    canvas.current = shape
    canvas.finalise()
    return shape


@pytest.mark.parametrize("shape_type", SHAPE_TYPES)
@pytest.mark.parametrize(
    "point",
    [
        (-20, 45.25),
        (120, 55.75),
        (65.25, -20),
        (75.75, 100),
        (-20, -20),
        (120, 100),
    ],
)
def test_first_click_can_be_outside_image(canvas, shape_type, point):
    canvas.create_mode = shape_type
    send_mouse(canvas, point)
    assert_points(canvas.current.points, [point])
    assert_points(canvas.line.points, [point, point])


@pytest.mark.parametrize("shape_type", SHAPE_TYPES)
@pytest.mark.parametrize("scale", [0.5, 10.0])
def test_preview_follows_cursor_outside_after_pan_and_zoom(
    canvas, shape_type, scale
):
    canvas.create_mode = shape_type
    canvas.scale = scale
    send_mouse(canvas, (25, 20))
    send_mouse(canvas, (130, 65.25), move=True)
    assert_points(canvas.line.points, [(25, 20), (130, 65.25)])
    assert_points(canvas.current.points, [(25, 20)])


def test_polygon_keeps_raw_vertices_until_closed_then_uses_intersections(
    canvas,
):
    canvas.create_mode = "polygon"
    for point in [(20, 20), (120, 60), (20, 70)]:
        send_mouse(canvas, point)
    assert_points(canvas.current.points, [(20, 20), (120, 60), (20, 70)])
    send_mouse(canvas, (20, 20))
    assert canvas.current is None
    assert_points(
        canvas.shapes[0].points, [(20, 20), (100, 52), (100, 62), (20, 70)]
    )
    assert canvas.shapes[0].is_closed()


def test_polygon_surrounding_image_clips_to_full_image(canvas):
    shape = finish_shape(
        canvas, "polygon", [(-20, -20), (120, -20), (120, 100), (-20, 100)]
    )
    assert len(shape.points) == 4
    assert {(point.x(), point.y()) for point in shape.points} == {
        (0, 0),
        (100, 0),
        (100, 80),
        (0, 80),
    }


@pytest.mark.parametrize("shape_type", ["rectangle", "rotation"])
def test_box_preview_is_unbounded_but_final_box_matches_image(
    canvas, shape_type
):
    canvas.create_mode = shape_type
    send_mouse(canvas, (-20, -20))
    send_mouse(canvas, (120, 100), move=True)
    assert_points(canvas.line.points, [(-20, -20), (120, 100)])
    send_mouse(canvas, (120, 100))
    assert canvas.current is None
    assert_points(
        canvas.shapes[0].points, [(0, 0), (100, 0), (100, 80), (0, 80)]
    )
    assert canvas.shapes[0].shape_type == shape_type


def test_rotated_box_is_not_distorted_by_clipping(canvas):
    points = [(-20, 20), (40, -40), (80, 0), (20, 60)]
    shape = Shape(shape_type="rotation", direction=math.pi / 4)
    shape.points = [QtCore.QPointF(*point) for point in points]
    canvas.current = shape
    canvas.finalise()
    assert_points(canvas.shapes[0].points, points)
    assert shape.direction == math.pi / 4


@pytest.mark.parametrize("shape_type", ["line", "linestrip"])
def test_open_line_clips_intersections_without_closing_path(
    canvas, shape_type
):
    canvas.create_mode = shape_type
    send_mouse(canvas, (-20, 10))
    send_mouse(canvas, (120, 66))
    if shape_type == "linestrip":
        assert_points(canvas.current.points, [(-20, 10), (120, 66)])
        canvas.finalise()
    assert_points(canvas.shapes[0].points, [(0, 18), (100, 58)])


def test_linestrip_crops_segments_in_original_order(canvas):
    shape = finish_shape(
        canvas, "linestrip", [(-20, 20), (50, 20), (120, 60), (50, 70)]
    )
    assert_points(
        shape.points,
        [(0, 20), (50, 20), (100, 20 + 200 / 7), (100, 60 + 20 / 7), (50, 70)],
    )


@pytest.mark.parametrize("shape_type", ["line", "linestrip"])
def test_shift_line_constraint_uses_raw_preview_then_clips_on_completion(
    canvas, shape_type
):
    canvas.create_mode = shape_type
    send_mouse(canvas, (20, 20))
    shift = QtCore.Qt.KeyboardModifier.ShiftModifier
    send_mouse(canvas, (130, 65), move=True, modifiers=shift)
    assert_points(canvas.line.points, [(20, 20), (130, 20)])
    send_mouse(canvas, (130, 65), modifiers=shift)
    if shape_type == "linestrip":
        canvas.finalise()
    assert_points(canvas.shapes[0].points, [(20, 20), (100, 20)])


@pytest.mark.parametrize("shape_type", SHAPE_TYPES)
def test_shape_entirely_outside_is_discarded_without_history_or_signal(
    canvas, shape_type
):
    created = []
    canvas.new_shape.connect(lambda: created.append(True))
    points = [(120, 100), (140, 100), (140, 120), (120, 120)]
    if shape_type in ["line", "linestrip"]:
        points = [points[0], points[2]]
    finish_shape(canvas, shape_type, points)
    assert canvas.current is None
    assert not canvas.shapes
    assert not canvas.shapes_backups
    assert not created


def test_quadrilateral_clamps_vertices_and_preserves_type(canvas):
    canvas.create_mode = "quadrilateral"
    for point in [(20, 20), (80, 20), (120, 60), (20, 60)]:
        send_mouse(canvas, point)
    shape = canvas.shapes[0]
    assert shape.shape_type == "quadrilateral"
    assert_points(shape.points, [(20, 20), (80, 20), (100, 60), (20, 60)])
    shape = finish_shape(
        canvas, "quadrilateral", [(-20, 10), (80, 10), (80, 70), (-20, 70)]
    )
    assert shape.shape_type == "quadrilateral"
    assert_points(shape.points, [(0, 10), (80, 10), (80, 70), (0, 70)])


def test_quadrilateral_preserves_four_vertices_when_two_clamp_to_same_corner(
    canvas,
):
    shape = finish_shape(
        canvas, "quadrilateral", [(20, 20), (120, -20), (140, -10), (20, 60)]
    )
    assert shape.shape_type == "quadrilateral"
    assert_points(shape.points, [(20, 20), (100, 0), (100, 0), (20, 60)])
    assert shape.is_closed()


@pytest.mark.parametrize("offset", [(-120, 0), (120, 0), (0, -100), (0, 100)])
def test_quadrilateral_entirely_beyond_one_edge_is_discarded(canvas, offset):
    points = [
        (x + offset[0], y + offset[1])
        for x, y in [(20, 20), (80, 20), (80, 60), (20, 60)]
    ]
    finish_shape(canvas, "quadrilateral", points)
    assert canvas.current is None
    assert not canvas.shapes
    assert not canvas.shapes_backups


def test_clip_preserves_fractional_coordinates_and_avoids_duplicate_border_vertices(
    canvas,
):
    shape = finish_shape(
        canvas, "polygon", [(0, 10.25), (100, 10.25), (100, 79.75), (0, 79.75)]
    )
    assert_points(
        shape.points, [(0, 10.25), (100, 10.25), (100, 79.75), (0, 79.75)]
    )
    assert len(canvas.shapes_backups) == 1
    assert_points(
        canvas.shapes_backups[0][0].points,
        [(0, 10.25), (100, 10.25), (100, 79.75), (0, 79.75)],
    )


def test_freehand_polygon_collects_outside_points_before_finishing(canvas):
    canvas.create_mode = "polygon"
    send_mouse(canvas, (20, 20))
    canvas._brush_drawing = True
    send_mouse(canvas, (120, 60), move=True)
    send_mouse(canvas, (20, 70), move=True)
    assert_points(canvas.current.points, [(20, 20), (120, 60), (20, 70)])
    canvas.finalise()
    assert_points(
        canvas.shapes[0].points, [(20, 20), (100, 52), (100, 62), (20, 70)]
    )
