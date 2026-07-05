import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.widgets.canvas import Canvas

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for canvas brush drawing tests"
)
class TestCanvasBrushDrawing(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.canvas = Canvas(parent=None)
        self.canvas.resize(100, 100)
        self.canvas.pixmap = QtGui.QPixmap(100, 100)
        self.canvas.scale = 1.0
        self.canvas.set_editing(False)
        self.canvas.create_mode = "polygon"
        self.canvas._brush_drawing = True
        self.canvas.current = Shape(shape_type="polygon")
        self.canvas.current.points = [
            QtCore.QPointF(10.0, 10.0),
            QtCore.QPointF(10.0, 50.0),
            QtCore.QPointF(50.0, 50.0),
        ]
        self.canvas.line.points = [
            QtCore.QPointF(50.0, 50.0),
            QtCore.QPointF(50.0, 50.0),
        ]

    def tearDown(self):
        self.canvas.close()
        self.app.processEvents()

    def mouse_move(self, pos, buttons):
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(*pos),
            QtCore.QPointF(*pos),
            QtCore.Qt.MouseButton.NoButton,
            buttons,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        self.canvas.mouseMoveEvent(event)

    def test_mouse_move_without_left_button_samples_freehand_points(self):
        self.mouse_move((80.0, 50.0), QtCore.Qt.MouseButton.NoButton)

        self.assertEqual(len(self.canvas.current.points), 4)
        self.assertEqual(
            self.canvas.current.points[-1], QtCore.QPointF(80.0, 50.0)
        )

    def test_crossing_start_is_detected_as_closed(self):
        self.canvas.current.points[-1] = QtCore.QPointF(50.0, 10.0)

        can_close = self.canvas._brush_drawing_can_close(
            QtCore.QPointF(0.0, 10.0)
        )

        self.assertTrue(can_close)

    def test_mouse_move_near_start_finalizes_polygon(self):
        self.mouse_move((11.0, 11.0), QtCore.Qt.MouseButton.NoButton)

        self.assertIsNone(self.canvas.current)
        self.assertEqual(len(self.canvas.shapes), 1)
        self.assertTrue(self.canvas.shapes[0].is_closed())
