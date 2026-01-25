from PyQt6 import QtCore, QtGui

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets.canvas import Canvas


def _new_canvas(qtbot):
    canvas = Canvas(parent=None)
    qtbot.addWidget(canvas)
    pixmap = QtGui.QPixmap(100, 100)
    pixmap.fill(QtGui.QColor("white"))
    canvas.load_pixmap(pixmap, clear_shapes=True)
    canvas.resize(100, 100)
    canvas.show()
    return canvas


def test_canvas_draw_rectangle_creates_shape(qtbot):
    canvas = _new_canvas(qtbot)
    canvas.mode = Canvas.CREATE
    canvas.create_mode = "rectangle"

    qtbot.mouseClick(
        canvas,
        QtCore.Qt.MouseButton.LeftButton,
        pos=QtCore.QPoint(10, 10),
    )
    qtbot.mouseMove(canvas, QtCore.QPoint(80, 80))
    with qtbot.waitSignal(canvas.new_shape, timeout=2000):
        qtbot.mouseClick(
            canvas,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(80, 80),
        )

    assert len(canvas.shapes) == 1
    shape = canvas.shapes[0]
    assert shape.shape_type == "rectangle"
    assert len(shape.points) == 4


def test_canvas_store_moving_shape_emits_signal(qtbot):
    canvas = _new_canvas(qtbot)
    shape = Shape(shape_type="rectangle")
    shape.points = [
        QtCore.QPointF(10, 10),
        QtCore.QPointF(20, 10),
        QtCore.QPointF(20, 20),
        QtCore.QPointF(10, 20),
    ]
    shape.close()
    canvas.shapes = [shape]
    canvas.store_shapes()

    shape.points[0] = QtCore.QPointF(11, 11)
    canvas.selected_shapes = [shape]
    canvas.moving_shape = True

    with qtbot.waitSignal(canvas.shape_moved, timeout=1000):
        canvas.store_moving_shape()


def test_canvas_restore_shape_undo(qtbot):
    canvas = _new_canvas(qtbot)
    shape = Shape(shape_type="rectangle")
    shape.points = [
        QtCore.QPointF(10, 10),
        QtCore.QPointF(20, 10),
        QtCore.QPointF(20, 20),
        QtCore.QPointF(10, 20),
    ]
    shape.close()
    canvas.shapes = [shape]
    canvas.store_shapes()

    shape.points[0] = QtCore.QPointF(12, 12)
    canvas.store_shapes()

    assert canvas.is_shape_restorable is True
    canvas.restore_shape()

    assert len(canvas.shapes) == 1
    assert canvas.shapes[0].points[0] == QtCore.QPointF(10, 10)
    assert canvas.selected_shapes == []

