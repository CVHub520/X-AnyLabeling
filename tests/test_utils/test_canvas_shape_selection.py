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
    PYQT_AVAILABLE, "PyQt6 is required for canvas shape selection tests"
)
class TestCanvasShapeSelection(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.canvas = Canvas(parent=None)
        self.canvas.pixmap = QtGui.QPixmap(200, 200)
        self.canvas.pixmap.fill(QtGui.QColor("black"))
        self.canvas.resize(200, 200)
        self.canvas.selection_changed.connect(self._set_selection)

    def tearDown(self):
        self.canvas.close()
        self.app.processEvents()

    def _set_selection(self, shapes):
        for shape in self.canvas.selected_shapes:
            shape.selected = False
        self.canvas.selected_shapes = shapes
        for shape in shapes:
            shape.selected = True

    @staticmethod
    def make_rectangle(label, left, top, right, bottom):
        shape = Shape(label=label, shape_type="rectangle")
        shape.points = [
            QtCore.QPointF(left, top),
            QtCore.QPointF(right, top),
            QtCore.QPointF(right, bottom),
            QtCore.QPointF(left, bottom),
        ]
        shape.close()
        return shape

    def test_click_selects_nested_shape_below_larger_shape(self):
        inner = self.make_rectangle("inner", 40, 40, 80, 80)
        outer = self.make_rectangle("outer", 10, 10, 150, 150)
        self.canvas.shapes = [inner, outer]

        self.canvas.select_shape_point(
            QtCore.QPointF(60, 60), multiple_selection_mode=False
        )

        self.assertEqual(self.canvas.selected_shapes, [inner])

    def test_hover_finds_nested_shape_vertex_below_larger_shape(self):
        inner = self.make_rectangle("inner", 40, 40, 80, 80)
        outer = self.make_rectangle("outer", 10, 10, 150, 150)
        self.canvas.shapes = [inner, outer]

        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(40, 40),
            QtCore.QPointF(40, 40),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        self.canvas.mouseMoveEvent(event)

        self.assertIs(self.canvas.h_shape, inner)
        self.assertEqual(self.canvas.h_vertex, 0)

    def test_clicking_unselected_vertex_selects_its_shape(self):
        inner = self.make_rectangle("inner", 40, 40, 80, 80)
        outer = self.make_rectangle("outer", 10, 10, 150, 150)
        self.canvas.shapes = [inner, outer]
        self.canvas.h_shape = inner
        self.canvas.h_vertex = 0

        self.canvas.select_shape_point(
            QtCore.QPointF(40, 40), multiple_selection_mode=False
        )

        self.assertEqual(self.canvas.selected_shapes, [inner])

    def test_vertex_proximity_outweighs_smaller_overlapping_area(self):
        outer = self.make_rectangle("outer", 10, 10, 150, 150)
        overlapping = self.make_rectangle("overlapping", -20, -20, 30, 30)
        self.canvas.shapes = [outer, overlapping]

        candidates = self.canvas._shape_hit_candidates(QtCore.QPointF(12, 12))

        self.assertEqual(candidates[:2], [outer, overlapping])

    def test_equal_area_overlap_keeps_top_shape_priority(self):
        lower = self.make_rectangle("lower", 20, 20, 100, 100)
        upper = self.make_rectangle("upper", 20, 20, 100, 100)
        self.canvas.shapes = [lower, upper]

        candidates = self.canvas._shape_hit_candidates(QtCore.QPointF(60, 60))

        self.assertEqual(candidates, [upper, lower])

    def test_hidden_nested_shape_is_not_a_candidate(self):
        inner = self.make_rectangle("inner", 40, 40, 80, 80)
        outer = self.make_rectangle("outer", 10, 10, 150, 150)
        self.canvas.shapes = [inner, outer]
        self.canvas.visible[inner] = False

        candidates = self.canvas._shape_hit_candidates(QtCore.QPointF(60, 60))

        self.assertEqual(candidates, [outer])
