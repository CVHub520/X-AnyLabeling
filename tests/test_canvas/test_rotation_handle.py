import math
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
    PYQT_AVAILABLE, "PyQt6 is required for canvas rotation handle tests"
)
class TestCanvasRotationHandle(unittest.TestCase):

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
    def make_rotation():
        shape = Shape(label="object", shape_type="rotation")
        shape.points = [
            QtCore.QPointF(40.0, 40.0),
            QtCore.QPointF(80.0, 40.0),
            QtCore.QPointF(80.0, 60.0),
            QtCore.QPointF(40.0, 60.0),
        ]
        shape.close()
        return shape

    @staticmethod
    def mouse_event(event_type, pos, button, buttons):
        return QtGui.QMouseEvent(
            event_type,
            QtCore.QPointF(pos),
            QtCore.QPointF(pos),
            button,
            buttons,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

    def assert_points_close(self, shape, expected):
        for point, (expected_x, expected_y) in zip(shape.points, expected):
            self.assertAlmostEqual(point.x(), expected_x, places=6)
            self.assertAlmostEqual(point.y(), expected_y, places=6)

    def test_selected_rotation_shape_exposes_handle_above_top_edge(self):
        shape = self.make_rotation()
        self.canvas.shapes = [shape]
        self._set_selection([shape])

        edge_mid, handle, center = self.canvas._rotation_handle_geometry(shape)

        self.assertEqual(edge_mid, QtCore.QPointF(60.0, 40.0))
        self.assertEqual(handle, QtCore.QPointF(60.0, 8.0))
        self.assertEqual(center, QtCore.QPointF(60.0, 50.0))
        self.assertIs(self.canvas._rotation_handle_shape_at(handle), shape)
        self.assertIs(
            self.canvas._rotation_handle_shape_at(
                QtCore.QPointF(
                    (edge_mid.x() + handle.x()) / 2.0,
                    (edge_mid.y() + handle.y()) / 2.0,
                )
            ),
            shape,
        )

    def test_rotation_handle_drag_rotates_shape_and_stores_history(self):
        shape = self.make_rotation()
        self.canvas.shapes = [shape]
        self._set_selection([shape])
        self.canvas.store_shapes()
        rotated = []
        self.canvas.shape_rotated.connect(lambda: rotated.append(True))

        _, handle, center = self.canvas._rotation_handle_geometry(shape)
        target = QtCore.QPointF(center.x() + 32.0, center.y())

        self.canvas.mousePressEvent(
            self.mouse_event(
                QtCore.QEvent.Type.MouseButtonPress,
                handle,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.LeftButton,
            )
        )
        self.canvas.mouseMoveEvent(
            self.mouse_event(
                QtCore.QEvent.Type.MouseMove,
                target,
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.MouseButton.LeftButton,
            )
        )
        self.canvas.mouseReleaseEvent(
            self.mouse_event(
                QtCore.QEvent.Type.MouseButtonRelease,
                target,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.NoButton,
            )
        )

        self.assertAlmostEqual(shape.direction, math.pi / 2.0, places=6)
        self.assert_points_close(
            shape,
            [
                (70.0, 30.0),
                (70.0, 70.0),
                (50.0, 70.0),
                (50.0, 30.0),
            ],
        )
        self.assertEqual(len(self.canvas.shapes_backups), 2)
        self.assertEqual(len(rotated), 1)
