import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import numpy as np
    from PyQt6 import QtCore, QtGui, QtWidgets

    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.widgets.canvas import Canvas

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for canvas brush edit tests"
)
class TestCanvasBrushEdit(unittest.TestCase):

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

    def make_shape(self, shape_type, points):
        shape = Shape(label="object", shape_type=shape_type)
        shape.points = [QtCore.QPointF(x, y) for x, y in points]
        shape.close()
        shape.selected = True
        self.canvas.shapes = [shape]
        self.canvas.selected_shapes = [shape]
        self.canvas.store_shapes()
        return shape

    def test_no_op_brush_edit_preserves_original_shape_type(self):
        shape = self.make_shape(
            "polygon",
            [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        )
        original_points = [QtCore.QPointF(point) for point in shape.points]

        self.canvas.set_brush_mode(True)
        self.canvas.set_brush_mode(False)

        self.assertEqual(shape.shape_type, "polygon")
        self.assertEqual(shape.points, original_points)
        self.assertIsNone(shape.mask)

    def test_brush_edit_rejects_non_polygon_shape(self):
        self.make_shape(
            "rectangle",
            [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        )

        self.canvas.set_brush_mode(True)

        self.assertFalse(self.canvas.is_brush_mode)
        self.assertIsNone(self.canvas._brush_target_shape)

    def test_mask_conversion_keeps_only_largest_component(self):
        shape = self.make_shape(
            "polygon",
            [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        )
        shape.mask = np.zeros((100, 100), dtype=np.uint8)
        shape.mask[10:51, 10:51] = 255
        shape.mask[80:85, 80:85] = 255

        converted = self.canvas._update_shape_points_from_mask(shape)

        self.assertTrue(converted)
        self.assertEqual(shape.mask[82, 82], 0)
        self.assertEqual(shape.mask[30, 30], 255)
        self.assertEqual(shape.shape_type, "polygon")

    def test_zero_simplification_tolerance_preserves_contour_points(self):
        contour = np.array(
            [
                [[0, 0]],
                [[5, 0]],
                [[10, 0]],
                [[10, 10]],
                [[0, 10]],
            ],
            dtype=np.int32,
        )

        preserved = self.canvas._simplify_contour(contour, 0.0)
        simplified = self.canvas._simplify_contour(contour, 2.0)

        self.assertEqual(len(preserved), len(contour))
        self.assertLess(len(simplified), len(preserved))

    def test_empty_mask_deletes_shape_when_brush_mode_exits(self):
        shape = self.make_shape(
            "polygon",
            [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        )
        deleted_shapes = []
        self.canvas.shapes_deleted.connect(deleted_shapes.extend)
        self.canvas.set_brush_mode(True)
        shape.mask.fill(0)
        self.canvas._brush_modified = True

        self.canvas.set_brush_mode(False)

        self.assertNotIn(shape, self.canvas.shapes)
        self.assertEqual(self.canvas.selected_shapes, [])
        self.assertEqual(deleted_shapes, [shape])

    def test_undo_to_baseline_restores_original_geometry_on_exit(self):
        shape = self.make_shape(
            "polygon",
            [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        )
        original_points = [QtCore.QPointF(point) for point in shape.points]
        self.canvas.set_brush_mode(True)
        self.canvas._apply_brush_to_mask(
            shape.mask, 52.0, 30.0, radius=5, add=True
        )
        self.canvas._brush_modified = True
        self.canvas._update_shape_points_from_mask(shape)
        self.canvas._push_brush_undo_state()

        self.canvas.brush_undo()
        self.canvas.set_brush_mode(False)

        self.assertEqual(shape.shape_type, "polygon")
        self.assertEqual(shape.points, original_points)

    def test_redo_reapplies_brush_stroke(self):
        shape = self.make_shape(
            "polygon",
            [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        )
        self.canvas.set_brush_mode(True)
        baseline = shape.mask.copy()
        self.canvas._apply_brush_to_mask(
            shape.mask, 52.0, 30.0, radius=5, add=True
        )
        self.canvas._brush_modified = True
        self.canvas._update_shape_points_from_mask(shape)
        self.canvas._push_brush_undo_state()
        edited = shape.mask.copy()

        self.canvas.brush_undo()
        self.assertTrue(np.array_equal(shape.mask, baseline))
        self.canvas.brush_redo()

        self.assertTrue(np.array_equal(shape.mask, edited))
        self.assertTrue(self.canvas._brush_modified)

    def test_cancel_restores_geometry_without_storing_history(self):
        shape = self.make_shape(
            "polygon",
            [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        )
        original_points = [QtCore.QPointF(point) for point in shape.points]
        backup_count = len(self.canvas.shapes_backups)
        moved = []
        deleted = []
        self.canvas.shape_moved.connect(lambda: moved.append(True))
        self.canvas.shapes_deleted.connect(deleted.extend)
        self.canvas.set_brush_mode(True)
        self.canvas._apply_brush_to_mask(
            shape.mask, 52.0, 30.0, radius=5, add=True
        )
        self.canvas._brush_modified = True
        self.canvas._update_shape_points_from_mask(shape)

        self.canvas.cancel_brush_mode()

        self.assertFalse(self.canvas.is_brush_mode)
        self.assertEqual(shape.points, original_points)
        self.assertIsNone(shape.mask)
        self.assertEqual(len(self.canvas.shapes_backups), backup_count)
        self.assertEqual(moved, [])
        self.assertEqual(deleted, [])

    def test_escape_cancels_brush_edit(self):
        shape = self.make_shape(
            "polygon",
            [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        )
        original_points = [QtCore.QPointF(point) for point in shape.points]
        self.canvas.set_brush_mode(True)
        self.canvas._apply_brush_to_mask(
            shape.mask, 52.0, 30.0, radius=5, add=True
        )
        self.canvas._brush_modified = True
        self.canvas._update_shape_points_from_mask(shape)
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Escape,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        self.canvas.keyPressEvent(event)

        self.assertTrue(event.isAccepted())
        self.assertFalse(self.canvas.is_brush_mode)
        self.assertEqual(shape.points, original_points)

    def test_loading_pixmap_cancels_brush_edit(self):
        shape = self.make_shape(
            "polygon",
            [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        )
        original_points = [QtCore.QPointF(point) for point in shape.points]
        self.canvas.set_brush_mode(True)
        self.canvas._apply_brush_to_mask(
            shape.mask, 52.0, 30.0, radius=5, add=True
        )
        self.canvas._brush_modified = True
        self.canvas._update_shape_points_from_mask(shape)

        self.canvas.load_pixmap(QtGui.QPixmap(120, 120), clear_shapes=False)

        self.assertFalse(self.canvas.is_brush_mode)
        self.assertEqual(shape.points, original_points)
        self.assertIsNone(shape.mask)

    def test_right_click_exits_on_release(self):
        self.make_shape(
            "polygon",
            [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        )
        self.canvas.set_brush_mode(True)
        press = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(20.0, 20.0),
            QtCore.QPointF(20.0, 20.0),
            QtCore.Qt.MouseButton.RightButton,
            QtCore.Qt.MouseButton.RightButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        release = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QPointF(20.0, 20.0),
            QtCore.QPointF(20.0, 20.0),
            QtCore.Qt.MouseButton.RightButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        self.assertTrue(
            self.canvas._brush_mouse_press(press, QtCore.QPointF(20.0, 20.0))
        )
        self.assertTrue(self.canvas.is_brush_mode)
        self.assertTrue(self.canvas._brush_mouse_release(release))
        self.assertFalse(self.canvas.is_brush_mode)

    def test_undo_snapshots_respect_memory_limit(self):
        shape = self.make_shape(
            "polygon",
            [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)],
        )
        self.canvas.set_brush_mode(True)
        self.canvas._brush_max_undo_bytes = shape.mask.nbytes * 2

        for index in range(5):
            shape.mask[70, 70 + index] = 255
            self.canvas._push_brush_undo_state()

        self.assertEqual(len(self.canvas._brush_undo_stack), 2)
