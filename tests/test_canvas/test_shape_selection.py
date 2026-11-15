import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

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

    def _move_mouse(self, x, y):
        point = QtCore.QPointF(x, y)
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            point,
            point,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        self.canvas.mouseMoveEvent(event)

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

    def test_double_click_polygon_vertex(self):
        for label_edit, locked, point_count in (
            (True, False, 4),
            (False, False, 4),
            (True, True, 4),
            (True, False, 3),
        ):
            with self.subTest(
                label_edit=label_edit, locked=locked, point_count=point_count
            ):
                shape = self.make_rectangle("polygon", 20, 20, 80, 80)
                shape.shape_type = "polygon"
                shape.points = shape.points[:point_count]
                shape.locked = locked
                original_points = list(shape.points)
                self.canvas.load_shapes([shape])
                self._set_selection([shape])
                self.canvas.double_click_edit_label = label_edit
                moved = QtTest.QSignalSpy(self.canvas.shape_moved)
                edited = QtTest.QSignalSpy(self.canvas.edit_label_requested)
                self._move_mouse(20, 20)
                QtTest.QTest.mouseClick(
                    self.canvas,
                    QtCore.Qt.MouseButton.LeftButton,
                    pos=QtCore.QPoint(20, 20),
                )
                QtTest.QTest.mouseDClick(
                    self.canvas,
                    QtCore.Qt.MouseButton.LeftButton,
                    pos=QtCore.QPoint(20, 20),
                )
                QtTest.QTest.mouseRelease(
                    self.canvas,
                    QtCore.Qt.MouseButton.LeftButton,
                    pos=QtCore.QPoint(20, 20),
                )
                if not locked and point_count > 3:
                    self.assertEqual(shape.points, original_points[1:])
                    self.assertEqual(len(moved), 1)
                    self.assertEqual(len(edited), 0)
                    self.assertIsNone(self.canvas.h_vertex)
                    self.canvas.restore_shape()
                    self.assertEqual(
                        self.canvas.shapes[0].points, original_points
                    )
                else:
                    self.assertEqual(shape.points, original_points)
                    self.assertEqual(len(moved), 0)
                    if not locked:
                        self.assertEqual(len(edited), 0)

    def test_paint_handles_stale_highlight_after_vertex_reduction(
        self,
    ) -> None:
        """Keep repainting safe when geometry outlives its hover index."""
        shape = self.make_rectangle("polygon", 20, 20, 80, 80)
        shape.shape_type = "polygon"
        shape.selected = True
        shape.highlight_vertex(3, Shape.MOVE_VERTEX)
        shape.points = shape.points[:3]
        points = list(shape.points)
        image = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_ARGB32)
        painter = QtGui.QPainter(image)
        try:
            shape.paint(painter)
        finally:
            painter.end()

        self.assertIsNone(shape._highlight_index)
        self.assertEqual(shape.points, points)

    def test_invalid_point_geometry_does_not_crash_paint(self) -> None:
        """Reject a malformed point without altering stored coordinates."""
        shape = Shape(label="point", shape_type="point")
        shape.points = [QtCore.QPointF(20, 20), QtCore.QPointF(30, 30)]
        points = list(shape.points)
        shape.selected = True
        shape.highlight_vertex(3, Shape.MOVE_VERTEX)
        image = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_ARGB32)
        painter = QtGui.QPainter(image)
        try:
            with mock.patch(
                "anylabeling.views.labeling.shape.logger.error"
            ) as error:
                shape.paint(painter)
        finally:
            painter.end()

        error.assert_called_once()
        self.assertEqual(shape.points, points)

    def test_double_click_polygon_edge_keeps_vertices_and_edits_label(self):
        shape = self.make_rectangle("polygon", 20, 20, 80, 80)
        shape.shape_type = "polygon"
        original_points = list(shape.points)
        self.canvas.load_shapes([shape])
        self._set_selection([shape])
        edited = QtTest.QSignalSpy(self.canvas.edit_label_requested)
        self._move_mouse(50, 20)
        QtTest.QTest.mouseClick(
            self.canvas,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(50, 20),
        )
        QtTest.QTest.mouseDClick(
            self.canvas,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(50, 20),
        )
        self.assertEqual(shape.points, original_points)
        self.assertEqual(len(edited), 1)

    def test_blank_click_does_not_extend_polygon_with_duplicate_points(self):
        for points in (
            [(20, 20), (80, 20), (80, 20), (80, 80), (20, 80)],
            [(20, 20), (80, 20), (80, 80), (20, 80), (20, 20)],
        ):
            with self.subTest(points=points):
                shape = Shape(label="polygon", shape_type="polygon")
                shape.points = [QtCore.QPointF(x, y) for x, y in points]
                shape.close()
                original_points = list(shape.points)
                self.canvas.load_shapes([shape])

                self._move_mouse(150, 150)
                self.assertIsNone(self.canvas.h_edge)
                QtTest.QTest.mouseClick(
                    self.canvas,
                    QtCore.Qt.MouseButton.LeftButton,
                    pos=QtCore.QPoint(150, 150),
                )
                self.assertEqual(shape.points, original_points)
                self.assertEqual(self.canvas.selected_shapes, [])

                self._move_mouse(50, 20)
                self.assertIs(self.canvas.h_shape, shape)
                self.assertIsNone(self.canvas.h_edge)
                QtTest.QTest.mouseClick(
                    self.canvas,
                    QtCore.Qt.MouseButton.LeftButton,
                    pos=QtCore.QPoint(50, 20),
                )
                self.assertEqual(shape.points, original_points)
                self.assertEqual(self.canvas.selected_shapes, [shape])
                self._move_mouse(50, 20)
                self.assertIsNotNone(self.canvas.h_edge)

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
        self.assertIsNone(self.canvas.h_vertex)
        self.assertEqual(self.canvas.selected_shapes, [])
        self._set_selection([inner])
        self._move_mouse(40, 40)
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

    def test_hover_preserves_explicit_selection(self):
        shape = self.make_rectangle("shape", 40, 40, 80, 80)
        self.canvas.shapes = [shape]
        self.canvas.h_shape_is_hovered = True

        self._move_mouse(60, 60)
        self.assertEqual(self.canvas.selected_shapes, [])
        QtTest.QTest.mouseClick(
            self.canvas,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(60, 60),
        )

        self.assertEqual(self.canvas.selected_shapes, [shape])
        before = shape.points[0].x()
        with mock.patch.object(self.canvas, "update") as update:
            QtTest.QTest.keyClick(self.canvas, QtCore.Qt.Key.Key_Right)

        self.assertEqual(shape.points[0].x(), before + 1.0)
        update.assert_called_once_with()
        self.assertFalse(self.canvas.moving_shape)

        self._move_mouse(180, 180)

        self.assertEqual(self.canvas.selected_shapes, [shape])

    def test_finishing_shape_move_requests_repaint(self):
        shape = self.make_rectangle("shape", 40, 40, 80, 80)
        self.canvas.shapes = [shape]
        self.canvas.selected_shapes = [shape]
        self.canvas.moving_shape = True

        with mock.patch.object(self.canvas, "update") as update:
            self.canvas.store_moving_shape()

        update.assert_called_once_with()

    def test_hover_paints_once_after_event_processing(self) -> None:
        """Process one hover event and verify Qt paints once."""
        self.canvas.shapes = [self.make_rectangle("shape", 40, 40, 80, 80)]
        self.canvas.show()
        self.app.processEvents()
        with mock.patch.object(
            self.canvas, "paintEvent", wraps=self.canvas.paintEvent
        ) as paint:
            self._move_mouse(60, 60)
            paint.assert_not_called()
            self.app.processEvents()
            paint.assert_called_once()

    def test_polygon_hit_keeps_nearby_and_coincident_vertices(self) -> None:
        """Keep vertices selectable near both regular and null bounds."""
        point = QtCore.QPointF(50, 50)
        polygon = Shape(label="polygon", shape_type="polygon")
        self.canvas.shapes = [polygon]
        for points in (
            [point, QtCore.QPointF(80, 50), QtCore.QPointF(60, 80)],
            [point, point, point],
        ):
            polygon.points = points
            with self.subTest(points=points):
                candidates = self.canvas._shape_hit_candidates(
                    QtCore.QPointF(49, 50)
                )
                self.assertEqual(candidates, [polygon])

    def test_drawing_highlight_survives_deferred_paint_and_clears(self):
        self.canvas.set_editing(False)
        self.canvas.show()
        for mode in ("polygon", "quadrilateral", "rotation"):
            with self.subTest(mode=mode):
                self.canvas.create_mode = mode
                shape = Shape(shape_type=mode)
                shape.points = [QtCore.QPointF(40, 40)]
                if mode != "rotation":
                    shape.points.extend(
                        [QtCore.QPointF(100, 40), QtCore.QPointF(100, 100)]
                    )
                self.canvas.current = shape
                self.canvas.line.points = [shape[-1], QtCore.QPointF(100, 100)]
                self.app.processEvents()
                painted = []
                original_paint = shape.paint

                def record_paint(*args, **kwargs):
                    painted.append(shape._highlight_index)
                    return original_paint(*args, **kwargs)

                with mock.patch.object(
                    shape, "paint", side_effect=record_paint
                ):
                    self._move_mouse(41, 41)
                    self.assertEqual(painted, [])
                    self.app.processEvents()
                    self.assertEqual(painted, [0])
                    painted.clear()
                    self._move_mouse(150, 150)
                    self.app.processEvents()
                    self.assertEqual(painted, [None])

    def test_closing_polygon_clears_highlight_before_storing_history(self):
        self.canvas.set_editing(False)
        self.canvas.create_mode = "polygon"
        shape = Shape(shape_type="polygon")
        shape.points = [
            QtCore.QPointF(40, 40),
            QtCore.QPointF(100, 40),
            QtCore.QPointF(100, 100),
        ]
        self.canvas.current = shape
        self.canvas.line.points = [shape[-1], shape[0]]
        self._move_mouse(41, 41)
        QtTest.QTest.mouseClick(
            self.canvas,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(40, 40),
        )

        self.assertIsNone(self.canvas.current)
        self.assertEqual(self.canvas.shapes, [shape])
        self.assertTrue(shape.is_closed())
        self.assertIsNone(shape._highlight_index)
        self.assertIsNone(self.canvas.shapes_backups[-1][0]._highlight_index)

    def test_hover_burst_paints_final_highlight_once(self):
        shape = self.make_rectangle("polygon", 40, 40, 100, 100)
        shape.shape_type = "polygon"
        self.canvas.shapes = [shape]
        self._set_selection([shape])
        self.canvas.show()
        self.app.processEvents()
        with mock.patch.object(
            self.canvas, "paintEvent", wraps=self.canvas.paintEvent
        ) as paint:
            self._move_mouse(40, 40)
            self._move_mouse(100, 100)
            self._move_mouse(150, 150)
            paint.assert_not_called()
            self.app.processEvents()
            paint.assert_called_once()
        self.assertIsNone(self.canvas.h_shape)
        self.assertIsNone(shape._highlight_index)
        self.assertEqual(self.canvas.selected_shapes, [shape])

    def test_polygon_bounds_keep_tolerance_and_follow_geometry_changes(self):
        for scale in (0.1, 0.5, 1.0, 2.0, 10.0):
            self.canvas.scale = scale
            epsilon = self.canvas.epsilon / scale
            for points in (
                [(40, 40), (100, 40), (100, 100)],
                [(40, 40), (40, 70), (40, 100)],
                [(40, 40), (70, 40), (100, 40)],
                [(40, 40)] * 3,
            ):
                with self.subTest(scale=scale, points=points):
                    shape = Shape(shape_type="polygon")
                    shape.points = [QtCore.QPointF(*p) for p in points]
                    self.canvas.shapes = [shape]
                    point = QtCore.QPointF(40 - epsilon, 40)
                    self.assertEqual(
                        self.canvas._shape_hit_candidates(point), [shape]
                    )
                    self.assertEqual(
                        self.canvas._shape_hit_candidates(
                            point - QtCore.QPointF(0.01, 0)
                        ),
                        [],
                    )
                    shape.move_by(QtCore.QPointF(1000, 1000))
                    self.assertEqual(
                        self.canvas._shape_hit_candidates(point), []
                    )
                    self.assertEqual(
                        self.canvas._shape_hit_candidates(
                            point + QtCore.QPointF(1000, 1000)
                        ),
                        [shape],
                    )

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

    def _box_event(self, event_type, point):
        event = QtGui.QMouseEvent(
            event_type,
            QtCore.QPointF(*point),
            QtCore.QPointF(*point),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        QtWidgets.QApplication.sendEvent(self.canvas, event)

    def test_box_selection_contains_visible_unlocked_shapes_and_adds(self):
        existing = self.make_rectangle("existing", 160, 160, 180, 180)
        inside = self.make_rectangle("inside", 40, 40, 70, 70)
        partial = self.make_rectangle("partial", 80, 80, 140, 140)
        hidden = self.make_rectangle("hidden", 30, 30, 60, 60)
        locked = self.make_rectangle("locked", 35, 35, 65, 65)
        point = Shape(label="point", shape_type="point")
        point.points = [QtCore.QPointF(50, 50)]
        locked.locked = True
        shapes = [existing, inside, partial, hidden, locked, point]
        self.canvas.load_shapes(shapes)
        self.canvas.visible[hidden] = False
        self._set_selection([existing])
        original = [list(shape.points) for shape in shapes]
        self._box_event(QtCore.QEvent.Type.MouseButtonPress, (100, 100))
        self._box_event(QtCore.QEvent.Type.MouseMove, (20, 20))
        self.assertEqual(self.canvas.selected_shapes, [existing])
        self.canvas.show()
        self.app.processEvents()
        self.canvas.grab()
        self._box_event(QtCore.QEvent.Type.MouseButtonRelease, (20, 20))
        self.assertEqual(
            self.canvas.selected_shapes, [existing, inside, point]
        )
        self.assertEqual([shape.points for shape in shapes], original)
        self.assertEqual(len(self.canvas.shapes_backups), 1)

    def test_box_selection_from_selected_vertex_does_not_move_it(self):
        shape = self.make_rectangle("shape", 40, 40, 80, 80)
        self.canvas.load_shapes([shape])
        self._set_selection([shape])
        self._move_mouse(40, 40)
        original = list(shape.points)
        self._box_event(QtCore.QEvent.Type.MouseButtonPress, (40, 40))
        self._box_event(QtCore.QEvent.Type.MouseMove, (100, 100))
        self._box_event(QtCore.QEvent.Type.MouseButtonRelease, (100, 100))
        self.assertEqual(shape.points, original)
        self.assertEqual(self.canvas.selected_shapes, [shape])
        self.assertFalse(self.canvas.is_move_editing)

    def test_ctrl_click_toggles_selection_below_drag_threshold(self):
        shape = self.make_rectangle("shape", 40, 40, 80, 80)
        self.canvas.load_shapes([shape])
        for expected in ([shape], []):
            self._move_mouse(60, 60)
            self._box_event(QtCore.QEvent.Type.MouseButtonPress, (60, 60))
            self._box_event(QtCore.QEvent.Type.MouseButtonRelease, (61, 61))
            self.assertEqual(self.canvas.selected_shapes, expected)

    def test_escape_cancels_box_without_changing_selection(self):
        shape = self.make_rectangle("shape", 40, 40, 80, 80)
        self.canvas.load_shapes([shape])
        self._set_selection([shape])
        self._move_mouse(60, 60)
        self._box_event(QtCore.QEvent.Type.MouseButtonPress, (60, 60))
        self._box_event(QtCore.QEvent.Type.MouseMove, (100, 100))
        QtTest.QTest.keyClick(self.canvas, QtCore.Qt.Key.Key_Escape)
        self._box_event(QtCore.QEvent.Type.MouseButtonRelease, (100, 100))
        self.assertEqual(self.canvas.selected_shapes, [shape])
        self.assertIsNone(self.canvas._selection_rect)

    def _key_event(self, event_type, key, repeat=False, modifiers=None):
        if modifiers is None:
            modifiers = QtCore.Qt.KeyboardModifier.NoModifier
        event = QtGui.QKeyEvent(event_type, key, modifiers, "", repeat)
        QtWidgets.QApplication.sendEvent(self.canvas, event)

    def test_repeated_keyboard_edits_create_one_undo_step(self):
        for key in (
            QtCore.Qt.Key.Key_Z,
            QtCore.Qt.Key.Key_X,
            QtCore.Qt.Key.Key_C,
            QtCore.Qt.Key.Key_V,
            QtCore.Qt.Key.Key_Right,
        ):
            with self.subTest(key=key):
                shape = self.make_rectangle("shape", 60, 60, 100, 100)
                shape.shape_type = "rotation"
                self.canvas.shapes_backups = []
                self.canvas.load_shapes([shape])
                self._set_selection([shape])
                original = list(shape.points)
                self._key_event(QtCore.QEvent.Type.KeyPress, key)
                for _ in range(4):
                    self._key_event(QtCore.QEvent.Type.KeyRelease, key, True)
                    self._key_event(QtCore.QEvent.Type.KeyPress, key, True)
                self.assertEqual(len(self.canvas.shapes_backups), 1)
                self.assertNotEqual(shape.points, original)
                self._key_event(QtCore.QEvent.Type.KeyRelease, key)
                self.assertEqual(len(self.canvas.shapes_backups), 2)
                self.canvas.restore_shape()
                self.assertEqual(self.canvas.shapes[0].points, original)

    def test_keyboard_rotation_mixed_selection_and_focus_loss(self):
        rectangle = self.make_rectangle("rectangle", 10, 10, 30, 30)
        rotation = self.make_rectangle("rotation", 60, 60, 100, 100)
        rotation.shape_type = "rotation"
        self.canvas.load_shapes([rectangle, rotation])
        self._set_selection([rectangle, rotation])
        original = list(rotation.points)
        self._key_event(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Z)
        self.canvas.focusOutEvent(
            QtGui.QFocusEvent(QtCore.QEvent.Type.FocusOut)
        )
        self.assertEqual(len(self.canvas.shapes_backups), 2)
        self.assertFalse(self.canvas.rotating_shape)
        self.canvas.restore_shape()
        self.assertEqual(self.canvas.shapes[1].points, original)

    def test_ctrl_z_does_not_rotate(self):
        shape = self.make_rectangle("shape", 60, 60, 100, 100)
        shape.shape_type = "rotation"
        self.canvas.load_shapes([shape])
        self._set_selection([shape])
        original = list(shape.points)
        self._key_event(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Z,
            modifiers=QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        self.assertEqual(shape.points, original)
        self.assertEqual(len(self.canvas.shapes_backups), 1)
