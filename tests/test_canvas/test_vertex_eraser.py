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
    PYQT_AVAILABLE, "PyQt6 is required for canvas vertex eraser tests"
)
class TestCanvasVertexEraser(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.canvas = Canvas(parent=None)
        self.canvas.scale = 1.0

    def tearDown(self):
        self.canvas.close()
        self.app.processEvents()

    def make_shape(self, shape_type, points):
        shape = Shape(label="object", shape_type=shape_type)
        shape.points = [QtCore.QPointF(x, y) for x, y in points]
        shape.close()
        self.canvas.shapes = [shape]
        self.canvas.selected_shapes = [shape]
        self.canvas.store_shapes()
        return shape

    def test_vertex_eraser_removes_polygon_vertex(self):
        shape = self.make_shape(
            "polygon",
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        )

        removed = self.canvas.erase_selected_vertex_at(
            QtCore.QPointF(10.0, 0.0)
        )

        self.assertTrue(removed)
        self.assertEqual(len(shape.points), 3)
        self.assertNotIn(QtCore.QPointF(10.0, 0.0), shape.points)
        self.assertEqual(self.canvas.selected_shapes, [shape])

    def test_vertex_eraser_deletes_polygon_when_it_becomes_invalid(self):
        shape = self.make_shape(
            "polygon",
            [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)],
        )
        deleted_shapes = []
        self.canvas.shapes_deleted.connect(deleted_shapes.extend)

        removed = self.canvas.erase_selected_vertex_at(
            QtCore.QPointF(10.0, 0.0)
        )

        self.assertTrue(removed)
        self.assertNotIn(shape, self.canvas.shapes)
        self.assertEqual(self.canvas.selected_shapes, [])
        self.assertEqual(deleted_shapes, [shape])

    def test_vertex_eraser_ignores_multiple_selection(self):
        shape = self.make_shape(
            "linestrip",
            [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)],
        )
        other = Shape(label="other", shape_type="linestrip")
        other.points = [QtCore.QPointF(0.0, 10.0), QtCore.QPointF(10.0, 10.0)]
        self.canvas.shapes.append(other)
        self.canvas.selected_shapes = [shape, other]

        removed = self.canvas.erase_selected_vertex_at(
            QtCore.QPointF(10.0, 0.0)
        )

        self.assertFalse(removed)
        self.assertEqual(len(shape.points), 3)

    def test_vertex_eraser_deletes_linestrip_when_it_becomes_invalid(self):
        shape = self.make_shape("linestrip", [(0.0, 0.0), (10.0, 0.0)])
        deleted_shapes = []
        self.canvas.shapes_deleted.connect(deleted_shapes.extend)

        removed = self.canvas.erase_selected_vertex_at(
            QtCore.QPointF(10.0, 0.0)
        )

        self.assertTrue(removed)
        self.assertNotIn(shape, self.canvas.shapes)
        self.assertEqual(deleted_shapes, [shape])

    def test_vertex_eraser_shows_tooltip_when_alt_is_pressed(self):
        self.make_shape(
            "polygon",
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        )
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Alt,
            QtCore.Qt.KeyboardModifier.AltModifier,
        )

        self.canvas.keyPressEvent(event)

        self.assertIn("erase points", self.canvas.toolTip())
        self.assertIn("object", self.canvas.toolTip())
