"""Selection replacement, numeric sessions and absolute rotation snapping."""

import math
from types import SimpleNamespace
from unittest import mock
from collections.abc import Iterator

import pytest
from PyQt6 import QtCore, QtGui, QtWidgets, QtTest

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.ellipse import (
    ellipse_frame,
    ellipse_frame_points,
)
from anylabeling.views.labeling.widgets.canvas import Canvas
from anylabeling.views.labeling.widgets.ellipse_parameters import (
    EllipseParameterBar,
)

_APPLICATION = None


@pytest.fixture
def app() -> Iterator[QtWidgets.QApplication]:
    """Keep the Qt application alive throughout each interaction check."""
    global _APPLICATION
    _APPLICATION = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        []
    )
    yield _APPLICATION


def make_canvas(angle: float = 0.37) -> tuple[Canvas, Shape]:
    """Build one selected native ellipse in a real canvas.

    Args:
        angle: Initial clockwise axis angle.

    Returns:
        Canvas and its selected ellipse.
    """
    canvas = Canvas(parent=None)
    pixmap = QtGui.QPixmap(800, 600)
    pixmap.fill(QtCore.Qt.GlobalColor.white)
    canvas.load_pixmap(pixmap)
    shape = Shape(label="object", shape_type="ellipse")
    shape.points = [
        QtCore.QPointF(*p)
        for p in ellipse_frame_points(300, 250, 200, 80, angle)
    ]
    shape.direction = angle
    shape.close()
    canvas.shapes = [shape]
    canvas.selected_shapes = [shape]
    shape.selected = True
    canvas.store_shapes()
    return canvas, shape


def test_copy_undo_and_live_selection(app: QtWidgets.QApplication) -> None:
    """Editing must resolve replacement objects rather than old pointers."""
    canvas, original = make_canvas()
    canvas._ellipse_edit_target = original
    copied = original.copy()
    canvas.shapes = [copied]
    canvas.selected_shapes = [copied]
    assert canvas._editable_ellipse() is copied
    assert canvas.apply_ellipse_parameters(220, 90, 30)
    canvas.restore_shape()
    restored = canvas.shapes[0]
    canvas.selected_shapes = [restored]
    assert canvas._editable_ellipse() is restored
    assert canvas.apply_ellipse_parameters(230, 95, 35)
    assert original.points != restored.points
    canvas.close()


def test_shift_snaps_final_axis_and_cancel_restores(
    app: QtWidgets.QApplication,
) -> None:
    """Snap the final axis, including an off-center initial handle click."""
    canvas, shape = make_canvas()
    before = list(shape.points)
    center = QtCore.QPointF(300, 250)
    start_angle = 0.19
    start = center + QtCore.QPointF(
        160 * math.cos(start_angle), 160 * math.sin(start_angle)
    )
    canvas._start_rotation_handle_drag(
        shape, start, False, QtCore.Qt.KeyboardModifier.NoModifier
    )
    target = center + QtCore.QPointF(
        160 * math.cos(0.47), 160 * math.sin(0.47)
    )
    canvas._update_rotation_handle_drag(
        target, QtCore.Qt.KeyboardModifier.ShiftModifier
    )
    degrees = math.degrees(
        ellipse_frame([(p.x(), p.y()) for p in shape.points])[4]
    )
    assert degrees / 15 == pytest.approx(round(degrees / 15))
    canvas._cancel_ellipse_edit()
    assert shape.points == before
    assert len(canvas.shapes_backups) == 1
    canvas.close()


def test_tool_switch_cancels_resize_and_preview(
    app: QtWidgets.QApplication,
) -> None:
    """An unfinished ellipse edit must not leak into another tool."""
    canvas, shape = make_canvas(0)
    before = list(shape.points)
    canvas._start_ellipse_resize(shape, 5, shape.points[1])
    canvas._update_ellipse_resize(
        QtCore.QPointF(450, 250), QtCore.Qt.KeyboardModifier.NoModifier
    )
    canvas.create_mode = "rectangle"
    assert shape.points == before
    assert canvas._ellipse_resize is None
    assert len(canvas.shapes_backups) == 1
    canvas.close()


def test_numeric_commit_and_escape(app: QtWidgets.QApplication) -> None:
    """Pending input is cancelable and editingFinished commits only once."""
    bar = EllipseParameterBar()
    commits = []
    bar.geometry_committed.connect(lambda *values: commits.append(values))
    bar.show_geometry(200, 80, 30)
    bar.width_spin.setValue(220)
    assert commits == []
    QtTest.QTest.keyClick(bar.width_spin, QtCore.Qt.Key.Key_Escape)
    assert bar.width_spin.value() == 200
    bar.width_spin.setValue(230)
    bar.width_spin.editingFinished.emit()
    bar.width_spin.editingFinished.emit()
    assert commits == [(230, 80, 30)]
    bar.close()


def test_lock_hide_and_other_canvas_config(
    app: QtWidgets.QApplication,
) -> None:
    """Lock/visibility guards and sample counts remain instance-local."""
    canvas, shape = make_canvas()
    shape.locked = True
    assert not canvas.apply_ellipse_parameters(220, 90, 35)
    shape.locked = False
    canvas.set_shape_visible(shape, False)
    assert canvas._editable_ellipse() is None
    other = Canvas(parent=None, ellipse={"polygon_points": 12})
    assert canvas.ellipse_polygon_points == 128
    assert other.ellipse_polygon_points == 12
    canvas.close()
    other.close()


def test_widget_duplicate_restores_selection_after_list_refresh(
    app: QtWidgets.QApplication,
) -> None:
    """Reproduce the real widget clearing selection during duplicate refresh."""
    from anylabeling.views.labeling.label_widget import LabelingWidget

    canvas, original = make_canvas()
    canvas.selection_changed.connect(
        lambda shapes: setattr(canvas, "selected_shapes", shapes)
    )
    label_list = mock.Mock()
    label_list.clearSelection.side_effect = lambda: setattr(
        canvas, "selected_shapes", []
    )
    widget = SimpleNamespace(
        canvas=canvas,
        label_list=label_list,
        add_label=mock.Mock(),
        _refresh_shape_filters=mock.Mock(),
        set_dirty=mock.Mock(),
    )
    LabelingWidget.duplicate_selected_shape(widget)
    assert len(canvas.shapes) == 2
    assert canvas._editable_ellipse() is canvas.shapes[-1]
    assert canvas._editable_ellipse() is not original
    assert canvas.apply_ellipse_parameters(240, 85, 40)
    canvas.close()


def test_widget_paste_selects_the_live_copy(
    app: QtWidgets.QApplication,
) -> None:
    """Paste must select the actual copied shape after canvas loading."""
    from anylabeling.views.labeling.label_widget import LabelingWidget

    canvas, original = make_canvas()
    canvas.selection_changed.connect(
        lambda shapes: setattr(canvas, "selected_shapes", shapes)
    )
    widget = SimpleNamespace(
        canvas=canvas,
        _config={"system_clipboard": False},
        _copied_shapes=[original.copy()],
        _copied_group_id=None,
        set_dirty=mock.Mock(),
    )
    widget.load_shapes = canvas.load_shapes
    LabelingWidget.paste_selected_shape(widget)
    assert len(canvas.shapes) == 2
    assert canvas._editable_ellipse() is canvas.shapes[-1]
    assert canvas._editable_ellipse() is not original
    canvas.close()


def test_numeric_edit_retains_undisplayed_precision(app: QtWidgets.QApplication) -> None:
    """Changing one displayed field must not round the other native values."""
    bar = EllipseParameterBar()
    commits = []
    bar.geometry_committed.connect(lambda *values: commits.append(values))
    bar.show_geometry(200.123456, 80.654321, 25.123456)
    bar.width_spin.setValue(220)
    bar.width_spin.editingFinished.emit()
    assert commits == [(220, 80.654321, 25.123456)]
    bar.show_geometry(2_000_000, 80, 25)
    assert bar.width_spin.value() == 2_000_000
    bar.close()
