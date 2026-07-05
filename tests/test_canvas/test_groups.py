import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.widgets.canvas import Canvas

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for canvas group tests"
)
class TestCanvasGroups(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.canvas = Canvas(parent=None)
        self.canvas.show_groups = True
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
    def make_shape(left, top, right, bottom, group_id=3, locked=False):
        shape = Shape(
            label="object", shape_type="rectangle", group_id=group_id
        )
        shape.points = [
            QtCore.QPointF(left, top),
            QtCore.QPointF(right, top),
            QtCore.QPointF(right, bottom),
            QtCore.QPointF(left, bottom),
        ]
        shape.locked = locked
        shape.close()
        return shape

    def test_group_label_uses_compact_group_and_shape_count(self):
        self.assertEqual(self.canvas._group_label(3, 5), "G3 · S5")

    def test_shape_opacity_applies_to_kie_linking(self):
        first = self.make_shape(20, 20, 40, 40, group_id=3)
        second = self.make_shape(120, 20, 140, 40, group_id=4)
        first.kie_linking = [[3, 4]]
        self.canvas.shapes = [first, second]
        self.canvas.show_masks = False
        self.canvas.show_texts = False
        self.canvas.show_labels = False
        self.canvas.show_scores = False
        self.canvas.show_attributes = False
        self.canvas.cross_line_show = False
        self.canvas.show()
        self.app.processEvents()

        opaque_image = self.canvas.grab().toImage()
        self.assertNotEqual(
            opaque_image.pixelColor(80, 30),
            QtGui.QColor(QtCore.Qt.GlobalColor.black),
        )

        self.canvas.shape_opacity = 0.0
        self.canvas.update()
        self.app.processEvents()
        image = self.canvas.grab().toImage()

        self.assertEqual(
            image.pixelColor(80, 30), QtGui.QColor(QtCore.Qt.GlobalColor.black)
        )

    def test_group_label_matches_shape_label_font_and_touches_frame(self):
        shape = self.make_shape(20, 30, 40, 50)
        group_rect = self.canvas._group_rect([shape])
        label_rect = self.canvas._group_label_rect(3, 1, group_rect)
        shape_label_font = QtGui.QFont(
            "Arial",
            int(
                max(
                    6.0,
                    int(round(8.0 / self.canvas.scale)),
                )
            ),
        )

        self.assertEqual(
            self.canvas._group_label_font().pointSize(),
            shape_label_font.pointSize(),
        )
        self.assertEqual(label_rect.bottom(), group_rect.top())

    def test_group_border_selects_all_members(self):
        first = self.make_shape(20, 20, 40, 40)
        second = self.make_shape(80, 80, 100, 100)
        other = self.make_shape(120, 120, 140, 140, group_id=4)
        self.canvas.shapes = [first, second, other]

        QtTest.QTest.mouseClick(
            self.canvas,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(20, 60),
        )

        self.assertEqual(self.canvas.selected_shapes, [first, second])
        self.assertEqual(self.canvas._active_group_shapes(), [first, second])

    def test_hidden_group_members_are_counted_but_not_hit_tested(self):
        visible = self.make_shape(20, 20, 40, 40)
        hidden = self.make_shape(80, 80, 100, 100)
        hidden.visible = False
        self.canvas.shapes = [visible, hidden]

        grouped_shapes = self.canvas._grouped_shapes()

        self.assertEqual(grouped_shapes[3], [visible, hidden])
        self.assertEqual(
            self.canvas._group_at_point(QtCore.QPointF(30, 30)), 3
        )
        self.assertIsNone(self.canvas._group_at_point(QtCore.QPointF(90, 90)))

    def test_shape_click_keeps_individual_selection(self):
        first = self.make_shape(20, 20, 40, 40)
        second = self.make_shape(80, 80, 100, 100)
        self.canvas.shapes = [first, second]

        self.canvas.select_shape_point(
            QtCore.QPointF(30, 30), multiple_selection_mode=False
        )

        self.assertEqual(self.canvas.selected_shapes, [first])
        self.assertIsNone(self.canvas._selected_group_id)

    def test_shape_on_group_border_takes_selection_priority(self):
        first = self.make_shape(20, 40, 40, 80)
        second = self.make_shape(80, 80, 100, 100)
        self.canvas.shapes = [first, second]

        QtTest.QTest.mouseClick(
            self.canvas,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(20, 60),
        )

        self.assertEqual(self.canvas.selected_shapes, [first])
        self.assertIsNone(self.canvas._selected_group_id)

    def test_group_keyboard_move_updates_every_member(self):
        first = self.make_shape(20, 20, 40, 40)
        second = self.make_shape(80, 80, 100, 100)
        self.canvas.shapes = [first, second]
        self.canvas._select_group(3, QtCore.QPointF(20, 60))

        QtTest.QTest.keyClick(self.canvas, QtCore.Qt.Key.Key_Right)
        QtTest.QTest.keyClick(
            self.canvas,
            QtCore.Qt.Key.Key_Right,
            QtCore.Qt.KeyboardModifier.ShiftModifier,
        )

        self.assertEqual(first.points[0], QtCore.QPointF(31, 20))
        self.assertEqual(second.points[0], QtCore.QPointF(91, 80))

    def test_escape_clears_group_selection(self):
        first = self.make_shape(20, 20, 40, 40)
        second = self.make_shape(80, 80, 100, 100)
        self.canvas.shapes = [first, second]
        self.canvas._select_group(3, QtCore.QPointF(20, 60))

        QtTest.QTest.keyClick(self.canvas, QtCore.Qt.Key.Key_Escape)

        self.assertEqual(self.canvas.selected_shapes, [])
        self.assertIsNone(self.canvas._selected_group_id)

    def test_locked_group_does_not_move_partially(self):
        first = self.make_shape(20, 20, 40, 40)
        second = self.make_shape(80, 80, 100, 100, locked=True)
        self.canvas.shapes = [first, second]
        self.canvas._select_group(3, QtCore.QPointF(20, 60))
        original = [
            [QtCore.QPointF(point) for point in shape.points]
            for shape in self.canvas.shapes
        ]

        self.canvas.move_by_keyboard(QtCore.QPointF(5, 0))

        self.assertEqual(first.points, original[0])
        self.assertEqual(second.points, original[1])

    def test_pasted_group_gets_next_group_id_and_keeps_coordinates(self):
        first = self.make_shape(20, 20, 40, 40)
        second = self.make_shape(80, 80, 100, 100)
        existing = self.make_shape(120, 120, 140, 140, group_id=7)
        self.canvas.shapes = [first, second, existing]

        pasted = self.canvas.prepare_pasted_shapes([first, second], 3)

        self.assertTrue(all(shape.group_id == 8 for shape in pasted))
        self.assertEqual(pasted[0].points[0], QtCore.QPointF(20, 20))
        self.assertEqual(pasted[1].points[0], QtCore.QPointF(80, 80))
        self.assertIsNot(pasted[0], first)

    def test_group_overlay_paints(self):
        first = self.make_shape(20, 20, 40, 40)
        second = self.make_shape(80, 80, 100, 100)
        self.canvas.shapes = [first, second]
        image = QtGui.QImage(
            200, 200, QtGui.QImage.Format.Format_ARGB32_Premultiplied
        )
        image.fill(QtGui.QColor("black"))
        painter = QtGui.QPainter(image)

        self.canvas._paint_groups(painter)
        painter.end()

        self.assertNotEqual(image.pixelColor(20, 60), QtGui.QColor("black"))

    def test_group_overlay_restores_painter_state(self):
        first = self.make_shape(20, 20, 40, 40)
        second = self.make_shape(80, 80, 100, 100)
        self.canvas.shapes = [first, second]
        image = QtGui.QImage(
            200, 200, QtGui.QImage.Format.Format_ARGB32_Premultiplied
        )
        image.fill(QtGui.QColor("black"))
        painter = QtGui.QPainter(image)
        pen = QtGui.QPen(QtGui.QColor("red"), 3)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        self.canvas._paint_groups(painter)

        actual_pen = painter.pen()
        actual_brush_style = painter.brush().style()
        painter.end()
        self.assertEqual(actual_pen, pen)
        self.assertEqual(actual_brush_style, QtCore.Qt.BrushStyle.NoBrush)
