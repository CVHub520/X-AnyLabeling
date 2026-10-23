from collections.abc import Iterator
import json
import math
import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets.canvas import Canvas


def mouse_event(
    kind: QtCore.QEvent.Type,
    point: tuple[float, float],
    button: QtCore.Qt.MouseButton,
    buttons: QtCore.Qt.MouseButton,
    modifiers: QtCore.Qt.KeyboardModifier | None = None,
) -> QtGui.QMouseEvent:
    """Mouse event."""
    position = QtCore.QPointF(*point)
    return QtGui.QMouseEvent(
        kind,
        position,
        position,
        button,
        buttons,
        modifiers or QtCore.Qt.KeyboardModifier.NoModifier,
    )


@pytest.fixture
def qapp() -> Iterator[QtWidgets.QApplication]:
    """Qapp."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    app.processEvents()


def test_ellipse_create_drag_save_and_rotate(
    qapp: QtWidgets.QApplication,
) -> None:
    """Ellipse create drag save and rotate."""
    canvas = Canvas(parent=None, ellipse={"polygon_points": 64})
    canvas.pixmap = QtGui.QPixmap(200, 200)
    canvas.pixmap.fill(QtGui.QColor("black"))
    canvas.resize(200, 200)
    canvas.set_editing(False)
    canvas.create_mode = "ellipse"
    canvas.mousePressEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseButtonPress,
            (20, 10),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
        )
    )
    canvas.mouseMoveEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseMove,
            (80, 40),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.LeftButton,
        )
    )
    canvas.mouseReleaseEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseButtonRelease,
            (80, 40),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.NoButton,
        )
    )
    assert len(canvas.shapes) == 1
    ellipse = canvas.shapes[0]
    assert ellipse.shape_type == "ellipse"
    assert not ellipse.contains_point(QtCore.QPointF(21, 11))
    assert ellipse.contains_point(QtCore.QPointF(50, 25))
    canvas.close()
    stored = json.loads(json.dumps(ellipse.to_dict()))
    loaded = Shape().load_from_dict(stored)
    assert loaded.shape_type == "ellipse"
    assert loaded.points == ellipse.points
    assert loaded.direction == ellipse.direction
    canvas.bounded_rotate_shapes(0, ellipse, -math.pi / 2)
    assert ellipse.direction == math.pi / 2
    assert ellipse.contains_point(QtCore.QPointF(50, 25))


def test_ellipse_create_two_clicks(qapp: QtWidgets.QApplication) -> None:
    """Ellipse create two clicks."""
    canvas = Canvas(parent=None)
    canvas.pixmap = QtGui.QPixmap(200, 200)
    canvas.resize(200, 200)
    canvas.set_editing(False)
    canvas.create_mode = "ellipse"
    for point in ((80, 40), (20, 10)):
        canvas.mousePressEvent(
            mouse_event(
                QtCore.QEvent.Type.MouseButtonPress,
                point,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.LeftButton,
            )
        )
        canvas.mouseReleaseEvent(
            mouse_event(
                QtCore.QEvent.Type.MouseButtonRelease,
                point,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.NoButton,
            )
        )
    assert len(canvas.shapes) == 1
    assert [(p.x(), p.y()) for p in canvas.shapes[0].points] == [
        (20, 10),
        (80, 10),
        (80, 40),
        (20, 40),
    ]
    canvas.close()


@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("handle", range(8))
def test_selected_ellipse_survives_blank_space_and_resizes(
    qapp: QtWidgets.QApplication, scale: float, handle: int
) -> None:
    """Selected ellipse survives blank space and resizes."""
    canvas = Canvas(parent=None, auto_highlight_shape=True)
    canvas.h_shape_is_hovered = True
    canvas.pixmap = QtGui.QPixmap(800, 600)
    canvas.resize(int(800 * scale), int(600 * scale))
    canvas.scale = scale
    ellipse = Shape(label="table", shape_type="ellipse")
    ellipse.points = [
        QtCore.QPointF(100, 100),
        QtCore.QPointF(500, 100),
        QtCore.QPointF(500, 300),
        QtCore.QPointF(100, 300),
    ]
    ellipse.close()
    canvas.shapes = [ellipse]
    canvas.store_shapes()

    def select(shapes: object) -> None:
        """Select."""
        ellipse.selected = ellipse in shapes
        canvas.selected_shapes = shapes

    canvas.selection_changed.connect(select)

    def screen(point: tuple[float, float]) -> None:
        """Screen."""
        return (point[0] * scale, point[1] * scale)

    def move(point: tuple[float, float], pressed: object = False) -> None:
        """Move."""
        canvas.mouseMoveEvent(
            mouse_event(
                QtCore.QEvent.Type.MouseMove,
                screen(point),
                QtCore.Qt.MouseButton.NoButton,
                (
                    QtCore.Qt.MouseButton.LeftButton
                    if pressed
                    else QtCore.Qt.MouseButton.NoButton
                ),
            )
        )

    move((300, 200))
    canvas.mousePressEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseButtonPress,
            screen((300, 200)),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
        )
    )
    canvas.mouseReleaseEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseButtonRelease,
            screen((300, 200)),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.NoButton,
        )
    )
    assert canvas._ellipse_edit_target is ellipse
    move((150, 120))
    assert canvas.selected_shapes == [ellipse]
    assert ellipse.selected
    points = [(p.x(), p.y()) for p in ellipse.points]
    from anylabeling.views.labeling.utils.ellipse import (
        ellipse_handle_positions,
    )

    origin = ellipse_handle_positions(points)[handle]
    move(origin)
    assert canvas.h_ellipse_handle == handle
    assert canvas.selected_shapes == [ellipse]
    canvas.mousePressEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseButtonPress,
            screen(origin),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
        )
    )
    target = (
        origin[0] + (20 if handle in (1, 2, 5) else -20),
        origin[1] + (20 if handle in (2, 3, 6) else -20),
    )
    move(target, pressed=True)
    canvas.mouseReleaseEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseButtonRelease,
            screen(target),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.NoButton,
        )
    )
    assert ellipse.points != [QtCore.QPointF(*p) for p in points]
    assert len(canvas.shapes_backups) == 2
    assert canvas.selected_shapes == [ellipse]
    canvas.close()


def test_ellipse_numeric_edit_is_one_undo_step(
    qapp: QtWidgets.QApplication,
) -> None:
    """Ellipse numeric edit is one undo step."""
    canvas = Canvas(parent=None)
    ellipse = Shape(label="table", shape_type="ellipse")
    ellipse.points = [
        QtCore.QPointF(100, 100),
        QtCore.QPointF(500, 100),
        QtCore.QPointF(500, 300),
        QtCore.QPointF(100, 300),
    ]
    ellipse.close()
    canvas.shapes = [ellipse]
    canvas.store_shapes()
    canvas.selected_shapes = [ellipse]
    ellipse.selected = True
    canvas._ellipse_edit_target = ellipse
    assert canvas.apply_ellipse_parameters(440, 240, 89.9)
    assert ellipse.direction == pytest.approx(math.radians(89.9))
    assert len(canvas.shapes_backups) == 2
    canvas.restore_shape()
    assert canvas.shapes[0].points[0] == QtCore.QPointF(100, 100)
    canvas.close()


def test_ellipse_rotation_handle_remains_active_outside_area(
    qapp: QtWidgets.QApplication,
) -> None:
    """Ellipse rotation handle remains active outside area."""
    canvas = Canvas(parent=None, auto_highlight_shape=True)
    canvas.h_shape_is_hovered = True
    canvas.pixmap = QtGui.QPixmap(800, 600)
    canvas.resize(800, 600)
    ellipse = Shape(label="table", shape_type="ellipse")
    ellipse.points = [
        QtCore.QPointF(100, 100),
        QtCore.QPointF(500, 100),
        QtCore.QPointF(500, 300),
        QtCore.QPointF(100, 300),
    ]
    ellipse.close()
    canvas.shapes = [ellipse]
    canvas.store_shapes()

    def select(shapes: object) -> None:
        """Select."""
        ellipse.selected = ellipse in shapes
        canvas.selected_shapes = shapes

    canvas.selection_changed.connect(select)
    canvas.select_shapes([ellipse])
    edge_mid, handle, center = canvas._rotation_handle_geometry(ellipse)
    assert edge_mid == QtCore.QPointF(300, 100)
    canvas.mouseMoveEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseMove,
            (150, 120),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
        )
    )
    assert canvas.selected_shapes == [ellipse]
    target = (center.x() + 50, center.y())
    canvas.mousePressEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseButtonPress,
            (handle.x(), handle.y()),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
        )
    )
    canvas.mouseMoveEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseMove,
            target,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.LeftButton,
        )
    )
    canvas.mouseReleaseEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseButtonRelease,
            target,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.NoButton,
        )
    )
    assert ellipse.direction == pytest.approx(math.pi / 2)
    assert canvas.selected_shapes == [ellipse]
    assert len(canvas.shapes_backups) == 2
    canvas.close()


def test_selected_ellipse_handle_wins_over_other_shape_and_escape_reverts(
    qapp: QtWidgets.QApplication,
) -> None:
    """Selected ellipse handle wins over other shape and escape reverts."""
    canvas = Canvas(parent=None, auto_highlight_shape=True)
    canvas.h_shape_is_hovered = True
    canvas.pixmap = QtGui.QPixmap(800, 600)
    canvas.resize(800, 600)
    ellipse = Shape(label="table", shape_type="ellipse")
    ellipse.points = [
        QtCore.QPointF(100, 100),
        QtCore.QPointF(500, 100),
        QtCore.QPointF(500, 300),
        QtCore.QPointF(100, 300),
    ]
    ellipse.close()
    other = Shape(label="other", shape_type="polygon")
    other.points = [
        QtCore.QPointF(80, 80),
        QtCore.QPointF(130, 80),
        QtCore.QPointF(130, 130),
        QtCore.QPointF(80, 130),
    ]
    other.close()
    canvas.shapes = [ellipse, other]
    canvas.store_shapes()

    def select(shapes: object) -> None:
        """Select."""
        canvas.selected_shapes = shapes
        ellipse.selected = ellipse in shapes
        other.selected = other in shapes

    canvas.selection_changed.connect(select)
    canvas.select_shapes([ellipse])
    canvas.mouseMoveEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseMove,
            (100, 100),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
        )
    )
    assert canvas.h_ellipse_handle == 0
    assert canvas.selected_shapes == [ellipse]
    canvas.mousePressEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseButtonPress,
            (100, 100),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
        )
    )
    canvas.mouseMoveEvent(
        mouse_event(
            QtCore.QEvent.Type.MouseMove,
            (50, 50),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.LeftButton,
        )
    )
    assert ellipse.points[0] == QtCore.QPointF(50, 50)
    canvas.keyPressEvent(
        QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Escape,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
    )
    assert ellipse.points[0] == QtCore.QPointF(100, 100)
    assert len(canvas.shapes_backups) == 1
    assert canvas.selected_shapes == [ellipse]
    ellipse.locked = True
    assert canvas._ellipse_handle_at(QtCore.QPointF(100, 100)) is None
    canvas.close()
