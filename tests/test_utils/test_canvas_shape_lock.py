import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.utils.shape import _apply_shape_conversion
    from anylabeling.views.labeling.widgets.canvas import Canvas

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for canvas shape lock tests"
)
class TestCanvasShapeLock(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.canvas = Canvas(parent=None)
        self.canvas.pixmap = QtGui.QPixmap(100, 100)
        self.canvas.pixmap.fill(QtGui.QColor("black"))

    def tearDown(self):
        self.canvas.close()
        self.app.processEvents()

    @staticmethod
    def make_shape(locked=False):
        shape = Shape(label="object", shape_type="polygon")
        shape.points = [
            QtCore.QPointF(10.0, 10.0),
            QtCore.QPointF(30.0, 10.0),
            QtCore.QPointF(30.0, 30.0),
            QtCore.QPointF(10.0, 30.0),
        ]
        shape.locked = locked
        shape.close()
        return shape

    def test_lock_state_round_trip(self):
        shape = self.make_shape(locked=True)

        data = shape.to_dict()
        loaded = Shape().load_from_dict(data)

        self.assertTrue(data["locked"])
        self.assertTrue(loaded.locked)
        self.assertNotIn("locked", self.make_shape().to_dict())

    def test_locked_shape_cannot_move_or_rotate(self):
        shape = self.make_shape(locked=True)
        original_points = [QtCore.QPointF(point) for point in shape.points]
        self.canvas.shapes = [shape]
        self.canvas.selected_shapes = [shape]
        self.canvas.prev_point = QtCore.QPointF(10.0, 10.0)

        moved = self.canvas.bounded_move_shapes(
            [shape], QtCore.QPointF(20.0, 20.0)
        )
        rotated = self.canvas.bounded_rotate_shapes(0, shape, 0.5)

        self.assertFalse(moved)
        self.assertFalse(rotated)
        self.assertEqual(shape.points, original_points)

    def test_locked_shape_rejects_point_and_brush_edits(self):
        shape = self.make_shape(locked=True)
        self.canvas.shapes = [shape]
        self.canvas.selected_shapes = [shape]
        self.canvas.prev_h_shape = shape
        self.canvas.prev_h_edge = 1
        self.canvas.prev_move_point = QtCore.QPointF(20.0, 10.0)
        original_points = [QtCore.QPointF(point) for point in shape.points]

        self.canvas.add_point_to_edge()
        self.canvas.prev_h_vertex = 1
        self.canvas.remove_selected_point()
        self.canvas.set_brush_mode(True)

        self.assertEqual(shape.points, original_points)
        self.assertFalse(self.canvas.is_brush_mode)
        self.assertFalse(self.canvas.can_erase_selected_vertices())

    def test_delete_selected_keeps_locked_shapes(self):
        locked = self.make_shape(locked=True)
        unlocked = self.make_shape()
        unlocked.label = "unlocked"
        self.canvas.shapes = [locked, unlocked]
        self.canvas.selected_shapes = [locked, unlocked]

        deleted = self.canvas.delete_selected()

        self.assertEqual(deleted, [unlocked])
        self.assertEqual(self.canvas.shapes, [locked])
        self.assertEqual(self.canvas.selected_shapes, [locked])

    def test_unlock_restores_coordinate_editing(self):
        shape = self.make_shape(locked=False)
        self.canvas.prev_point = QtCore.QPointF(10.0, 10.0)

        moved = self.canvas.bounded_move_shapes(
            [shape], QtCore.QPointF(20.0, 20.0)
        )

        self.assertTrue(moved)
        self.assertEqual(shape.points[0], QtCore.QPointF(20.0, 20.0))

    def test_bulk_conversion_skips_locked_shapes(self):
        points = [[10, 10], [30, 10], [30, 30], [10, 30]]
        data = {
            "shapes": [
                {
                    "shape_type": "rectangle",
                    "points": points,
                    "locked": True,
                },
                {"shape_type": "rectangle", "points": points},
            ]
        }

        _apply_shape_conversion(data, "rectangle_to_rotation", {})

        self.assertEqual(data["shapes"][0]["shape_type"], "rectangle")
        self.assertEqual(data["shapes"][1]["shape_type"], "rotation")
