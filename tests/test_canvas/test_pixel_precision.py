import io
import json
import os
import tempfile
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PIL import Image

    from anylabeling.views.labeling.label_file import LabelFile
    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.widgets.canvas import Canvas
    from anylabeling.views.labeling.widgets.zoom_widget import ZoomWidget

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for pixel precision tests"
)
class TestCanvasPixelPrecision(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.canvas = Canvas(parent=None)
        pixmap = QtGui.QPixmap(200, 100)
        pixmap.fill(QtGui.QColor("black"))
        self.canvas.load_pixmap(pixmap)
        self.canvas.resize(200, 100)

    def tearDown(self):
        self.canvas.close()
        self.app.processEvents()

    def _press(self, x, y):
        point = QtCore.QPointF(x, y)
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            point,
            point,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        self.canvas.mousePressEvent(event)

    def test_smoothing_switches_off_at_five_times_zoom(self):
        self.canvas.scale = 4.0
        self.assertTrue(self.canvas.uses_smooth_pixmap_transform())
        self.canvas.scale = 5.0
        self.assertFalse(self.canvas.uses_smooth_pixmap_transform())
        self.canvas.scale = 32.0
        self.assertFalse(self.canvas.uses_smooth_pixmap_transform())

    def test_zoom_control_reaches_6400_percent(self):
        zoom = ZoomWidget()
        zoom.setValue(6400)

        self.assertEqual(zoom.maximum(), 6400)
        self.assertEqual(zoom.value(), 6400)

        zoom.close()

    def test_snap_can_be_toggled_without_losing_subpixel_precision(self):
        cursor = QtCore.QPointF(100.38, 50.71)

        snapped = self.canvas.snap_to_pixel_grid(cursor)
        self.assertEqual(snapped, QtCore.QPointF(100.5, 50.5))

        self.canvas.pixel_snap_step = 1.0
        whole_pixel = self.canvas.snap_to_pixel_grid(cursor)
        self.assertEqual(whole_pixel, QtCore.QPointF(100.0, 51.0))

        self.canvas.pixel_snap_enabled = False
        unsnapped = self.canvas.snap_to_pixel_grid(cursor)
        self.assertEqual(unsnapped, cursor)

    def test_polygon_click_uses_snap_and_off_mode_keeps_qpointf(self):
        self.canvas.set_editing(False)
        self.canvas.create_mode = "polygon"

        self._press(100.38, 50.71)
        self.assertEqual(self.canvas.current[0], QtCore.QPointF(100.5, 50.5))

        self.canvas.current = None
        self.canvas.pixel_snap_enabled = False
        self._press(100.38, 50.71)
        self.assertEqual(self.canvas.current[0], QtCore.QPointF(100.38, 50.71))

    def test_polygon_vertex_drag_snaps_but_rectangle_does_not(self):
        polygon = Shape(label="hole", shape_type="polygon")
        polygon.points = [QtCore.QPointF(10.0, 10.0)]
        self.canvas.h_shape = polygon
        self.canvas.h_vertex = 0

        self.canvas.bounded_move_vertex(QtCore.QPointF(100.38, 50.71))
        self.assertEqual(polygon[0], QtCore.QPointF(100.5, 50.5))

        rectangle = Shape(label="box", shape_type="rectangle")
        rectangle.points = [
            QtCore.QPointF(10.0, 10.0),
            QtCore.QPointF(20.0, 10.0),
            QtCore.QPointF(20.0, 20.0),
            QtCore.QPointF(10.0, 20.0),
        ]
        self.canvas.h_shape = rectangle
        self.canvas.h_vertex = 0
        self.canvas.bounded_move_vertex(QtCore.QPointF(100.38, 50.71))
        self.assertEqual(rectangle[0], QtCore.QPointF(100.38, 50.71))

    def test_outer_image_cell_boundary_is_a_valid_polygon_coordinate(self):
        polygon = Shape(label="hole", shape_type="polygon")
        polygon.points = [QtCore.QPointF(10.0, 10.0)]
        self.canvas.h_shape = polygon
        self.canvas.h_vertex = 0

        self.canvas.bounded_move_vertex(QtCore.QPointF(199.8, 99.8))

        self.assertEqual(polygon[0], QtCore.QPointF(200.0, 100.0))

    def test_grid_bounds_are_limited_to_visible_viewport(self):
        pixmap = QtGui.QPixmap(4000, 3000)
        self.canvas.load_pixmap(pixmap)
        self.canvas.scale = 32.0
        self.canvas.resize(self.canvas.sizeHint())
        viewport = QtCore.QRect(32000, 16000, 640, 480)

        bounds = self.canvas._visible_pixel_grid_bounds(viewport)

        self.assertEqual(bounds, (1000, 1020, 500, 515))
        left, right, top, bottom = bounds
        self.assertLessEqual((right - left + 1) + (bottom - top + 1), 40)

    def test_shape_json_round_trip_preserves_fractional_coordinates(self):
        shape = Shape(label="hole", shape_type="polygon")
        shape.points = [
            QtCore.QPointF(532.375, 271.812),
            QtCore.QPointF(533.164, 271.491),
            QtCore.QPointF(533.0, 272.0),
        ]
        shape.close()

        data = json.loads(json.dumps(shape.to_dict()))
        restored = Shape().load_from_dict(data)

        self.assertEqual(restored.points, shape.points)

    def test_label_file_save_and_reload_preserves_fractional_coordinates(self):
        shape = Shape(label="hole", shape_type="polygon")
        shape.points = [
            QtCore.QPointF(532.375, 271.812),
            QtCore.QPointF(533.164, 271.491),
            QtCore.QPointF(533.0, 272.0),
        ]
        shape.close()
        image_buffer = io.BytesIO()
        Image.new("RGB", (2, 2), "black").save(image_buffer, format="PNG")

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as directory:
            filename = os.path.join(directory, "pixel-precision.json")
            LabelFile().save(
                filename=filename,
                shapes=[shape.to_dict()],
                image_path="source.png",
                image_height=2,
                image_width=2,
                image_data=image_buffer.getvalue(),
            )
            restored = LabelFile(filename).shapes[0]

        self.assertEqual(restored.points, shape.points)

    def test_brush_edit_emits_precision_loss_warning(self):
        shape = Shape(label="hole", shape_type="polygon")
        shape.points = [
            QtCore.QPointF(10.25, 10.25),
            QtCore.QPointF(50.25, 10.25),
            QtCore.QPointF(50.25, 50.25),
        ]
        shape.close()
        shape.selected = True
        self.canvas.shapes = [shape]
        self.canvas.selected_shapes = [shape]
        warnings = []
        self.canvas.precision_warning_requested.connect(warnings.append)

        self.canvas.set_brush_mode(True)

        self.assertTrue(self.canvas.is_brush_mode)
        self.assertEqual(len(warnings), 1)

    def test_double_click_edge_refine_preempts_label_editor(self):
        shape = Shape(label="hole", shape_type="polygon")
        shape.points = [
            QtCore.QPointF(10.0, 10.0),
            QtCore.QPointF(60.0, 10.0),
            QtCore.QPointF(60.0, 60.0),
            QtCore.QPointF(10.0, 60.0),
        ]
        shape.close()
        self.canvas.shapes = [shape]
        self.canvas.edge_refine_on_double_click = True
        refine_requests = []
        label_requests = []
        self.canvas.shape_edge_refine_requested.connect(refine_requests.append)
        self.canvas.edit_label_requested.connect(
            lambda: label_requests.append(True)
        )
        point = QtCore.QPointF(30.0, 30.0)
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonDblClick,
            point,
            point,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        self.canvas.mouseDoubleClickEvent(event)

        self.assertEqual(refine_requests, [shape])
        self.assertEqual(label_requests, [])

    def test_live_brush_overlay_does_not_extract_contours(self):
        shape = Shape(label="hole", shape_type="polygon")
        shape.points = [
            QtCore.QPointF(10.0, 10.0),
            QtCore.QPointF(60.0, 10.0),
            QtCore.QPointF(60.0, 60.0),
            QtCore.QPointF(10.0, 60.0),
        ]
        shape.close()
        shape.selected = True
        self.canvas.shapes = [shape]
        self.canvas.selected_shapes = [shape]
        self.canvas.set_brush_mode(True)
        self.canvas._brush_stroke_dirty = True

        def fail_if_called(_shape):
            raise AssertionError("live brush paint must not run findContours")

        self.canvas._get_brush_render_data = fail_if_called
        target = QtGui.QImage(
            self.canvas.size(), QtGui.QImage.Format.Format_ARGB32
        )
        target.fill(QtCore.Qt.GlobalColor.transparent)

        self.canvas.render(target)

    def test_polygon_mouse_moves_queue_paints_instead_of_sync_repaint(self):
        self.canvas.set_editing(False)
        self.canvas.create_mode = "polygon"
        self._press(20.0, 20.0)
        original_update = self.canvas.update
        try:
            self.canvas.update = Mock()
            self.canvas.repaint = Mock()
            for x in range(21, 81):
                point = QtCore.QPointF(float(x), 25.0)
                event = QtGui.QMouseEvent(
                    QtCore.QEvent.Type.MouseMove,
                    point,
                    point,
                    QtCore.Qt.MouseButton.NoButton,
                    QtCore.Qt.MouseButton.NoButton,
                    QtCore.Qt.KeyboardModifier.NoModifier,
                )
                self.canvas.mouseMoveEvent(event)

            self.assertGreater(self.canvas.update.call_count, 0)
            self.canvas.repaint.assert_not_called()
        finally:
            self.canvas.update = original_update

    def test_cell_boundary_edit_removes_duplicates_and_clears_connector(self):
        shape = Shape(label="edge", shape_type="polygon")
        shape.points = [
            QtCore.QPointF(10.5, 10.5),
            QtCore.QPointF(30.5, 10.5),
            QtCore.QPointF(30.5, 10.5),
            QtCore.QPointF(30.5, 30.5),
            QtCore.QPointF(10.5, 30.5),
        ]
        shape.other_data["pixel_edge_geometry"] = "cell_boundary"
        shape.close()
        self.canvas.shapes = [shape]
        self.canvas.selected_shapes = [shape]
        self.canvas.store_shapes()
        self.canvas.moving_shape = True
        self.canvas.is_move_editing = True
        self.canvas.current = Shape(shape_type="linestrip")
        self.canvas.line.points = [
            QtCore.QPointF(30.5, 10.5),
            QtCore.QPointF(190.0, 90.0),
        ]

        self.canvas.store_moving_shape()

        self.assertEqual(len(shape.points), 4)
        self.assertIsNone(self.canvas.current)
        self.assertEqual(self.canvas.line.points, [])
        self.assertFalse(self.canvas.is_move_editing)

    def test_escape_cancels_pending_cell_boundary_connector(self):
        shape = Shape(label="edge", shape_type="polygon")
        shape.points = [
            QtCore.QPointF(10.5, 10.5),
            QtCore.QPointF(30.5, 10.5),
            QtCore.QPointF(30.5, 30.5),
            QtCore.QPointF(10.5, 30.5),
        ]
        shape.other_data["pixel_edge_geometry"] = "cell_boundary"
        shape.close()
        self.canvas.shapes = [shape]
        self.canvas.selected_shapes = [shape]
        self.canvas.is_move_editing = True
        self.canvas.line.points = [shape.points[0], QtCore.QPointF(190, 90)]
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Escape,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        self.canvas.keyPressEvent(event)

        self.assertFalse(self.canvas.is_move_editing)
        self.assertEqual(self.canvas.line.points, [])


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for pixel precision tests"
)
class TestPixelatedRendering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def _render_two_pixels(self, scale):
        source = QtGui.QImage(2, 1, QtGui.QImage.Format.Format_RGB32)
        source.setPixelColor(0, 0, QtGui.QColor(0, 0, 0))
        source.setPixelColor(1, 0, QtGui.QColor(255, 255, 255))
        canvas = Canvas(parent=None)
        canvas.load_pixmap(QtGui.QPixmap.fromImage(source))
        canvas.show_pixel_grid = False
        canvas.scale = scale
        canvas.resize(canvas.sizeHint())
        target = QtGui.QImage(canvas.size(), QtGui.QImage.Format.Format_ARGB32)
        target.fill(QtCore.Qt.GlobalColor.transparent)
        canvas.render(target)
        colors = {
            target.pixelColor(x, target.height() // 2).red()
            for x in range(target.width())
        }
        canvas.close()
        return colors

    def test_400_percent_is_smoothed_but_500_percent_is_pixelated(self):
        smooth_colors = self._render_two_pixels(4.0)
        pixelated_colors = self._render_two_pixels(5.0)

        self.assertTrue(any(value not in {0, 255} for value in smooth_colors))
        self.assertEqual(pixelated_colors, {0, 255})

    def test_pixel_grid_is_drawn_on_exact_pixel_boundaries(self):
        source = QtGui.QImage(2, 2, QtGui.QImage.Format.Format_RGB32)
        source.fill(QtGui.QColor("black"))
        canvas = Canvas(parent=None)
        canvas.load_pixmap(QtGui.QPixmap.fromImage(source))
        canvas.show_pixel_grid = True
        canvas.scale = 10.0
        canvas.resize(canvas.sizeHint())
        target = QtGui.QImage(canvas.size(), QtGui.QImage.Format.Format_ARGB32)
        target.fill(QtCore.Qt.GlobalColor.transparent)

        canvas.render(target)

        self.assertEqual(target.pixelColor(5, 5).red(), 0)
        self.assertGreater(target.pixelColor(10, 5).red(), 0)
        self.assertGreater(target.pixelColor(5, 10).red(), 0)
        canvas.close()
