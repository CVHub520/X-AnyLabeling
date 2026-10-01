import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets.canvas import Canvas


class TestDrawingModeSwitch(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.canvas = Canvas(parent=None)
        self.canvas.resize(200, 200)
        pixmap = QtGui.QPixmap(200, 200)
        pixmap.fill(QtGui.QColor("white"))
        self.canvas.load_pixmap(pixmap)
        self.canvas.set_editing(False)
        self.drawing_states = []
        self.canvas.drawing_polygon.connect(self.drawing_states.append)

    def tearDown(self):
        self.canvas.restore_cursor()
        self.canvas.close()
        self.app.processEvents()

    def click(self, x, y):
        pos = QtCore.QPointF(x, y)
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            pos,
            pos,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        self.canvas.mousePressEvent(event)

    def test_linestrip_rectangle_circle_sequence_starts_fresh_shapes(self):
        self.canvas.create_mode = "linestrip"
        self.click(20, 20)
        self.click(40, 40)
        self.click(60, 20)
        self.assertEqual(len(self.canvas.current.points), 3)

        self.canvas.set_editing(False)
        self.canvas.create_mode = "rectangle"
        self.assertIsNone(self.canvas.current)
        self.assertEqual(self.canvas.line.points, [])
        self.assertFalse(self.drawing_states[-1])
        self.click(80, 80)
        self.assertEqual(self.canvas.current.shape_type, "rectangle")

        self.canvas.set_editing(False)
        self.canvas.create_mode = "circle"
        self.assertIsNone(self.canvas.current)
        self.click(100, 100)
        self.click(120, 100)

        self.assertIsNone(self.canvas.current)
        self.assertEqual(len(self.canvas.shapes), 1)
        circle = self.canvas.shapes[0]
        self.assertEqual(circle.shape_type, "circle")
        self.assertEqual(
            circle.points,
            [QtCore.QPointF(100, 100), QtCore.QPointF(120, 100)],
        )

    def test_all_shape_type_switches_cancel_draft_and_preserve_history(self):
        self.canvas.create_mode = "rectangle"
        self.click(10, 10)
        self.click(30, 30)
        saved_shapes = list(self.canvas.shapes)
        saved_backups = list(self.canvas.shapes_backups)

        for source in Shape.get_supported_shape():
            for target in Shape.get_supported_shape():
                if source == target:
                    continue
                with self.subTest(source=source, target=target):
                    self.canvas.create_mode = source
                    draft = Shape(shape_type=source)
                    draft.add_point(QtCore.QPointF(50, 50))
                    self.canvas.current = draft
                    self.canvas.line.points = [QtCore.QPointF(50, 50)]
                    self.canvas._brush_drawing = True
                    self.canvas.hide_backround = True
                    self.canvas.set_hiding(True)

                    self.canvas.create_mode = target

                    self.assertIsNone(self.canvas.current)
                    self.assertEqual(self.canvas.line.points, [])
                    self.assertFalse(self.canvas._brush_drawing)
                    self.assertFalse(self.canvas._hide_backround)
                    self.assertFalse(self.drawing_states[-1])
                    self.assertEqual(self.canvas.shapes, saved_shapes)
                    self.assertEqual(self.canvas.shapes_backups, saved_backups)

    def test_same_mode_preserves_draft(self):
        self.canvas.create_mode = "linestrip"
        self.click(20, 20)
        self.click(40, 40)
        draft = self.canvas.current
        states = list(self.drawing_states)

        self.canvas.set_editing(False)
        self.canvas.create_mode = "linestrip"

        self.assertIs(self.canvas.current, draft)
        self.assertEqual(len(draft.points), 2)
        self.assertEqual(self.drawing_states, states)

    def test_editing_cancels_draft_even_when_shape_type_stays_same(self):
        self.canvas.create_mode = "rectangle"
        self.click(20, 20)

        self.canvas.set_editing(True)

        self.assertTrue(self.canvas.editing())
        self.assertIsNone(self.canvas.current)
        self.assertEqual(self.canvas.line.points, [])
        self.assertFalse(self.drawing_states[-1])

    def test_escape_clears_preview_and_restores_background(self):
        self.canvas.create_mode = "linestrip"
        self.click(20, 20)
        self.canvas.hide_backround = True
        self.canvas.set_hiding(True)
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Escape,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        self.canvas.keyPressEvent(event)

        self.assertIsNone(self.canvas.current)
        self.assertEqual(self.canvas.line.points, [])
        self.assertFalse(self.canvas._hide_backround)
        self.assertFalse(self.drawing_states[-1])
        self.assertEqual(self.canvas.shapes, [])

    def test_invalid_mode_preserves_draft(self):
        self.click(20, 20)
        draft = self.canvas.current
        mode = self.canvas.create_mode

        with self.assertRaises(ValueError):
            self.canvas.create_mode = "unsupported"

        self.assertEqual(self.canvas.create_mode, mode)
        self.assertIs(self.canvas.current, draft)
