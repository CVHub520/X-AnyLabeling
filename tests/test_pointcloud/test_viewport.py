import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.pointcloud.selection import (
    project_points,
    select_polygon,
)
from anylabeling.views.labeling.pointcloud.viewport import PointCloudViewport


class TestPointCloudViewport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )

    def setUp(self):
        self.widget = PointCloudViewport()
        self.widget.resize(400, 300)
        self.points = np.array(
            [[0, 0, 0, 0.2], [1, 0, 0, 0.8], [0, 0, 1, 0.5]], dtype=np.float32
        )
        self.widget.set_cloud(self.points)
        self.app.processEvents()

    def tearDown(self):
        self.widget.close()
        self.widget.deleteLater()
        self.app.processEvents()

    def event(self, kind, x, y, button=QtCore.Qt.MouseButton.LeftButton):
        return QtGui.QMouseEvent(
            kind,
            QtCore.QPointF(x, y),
            QtCore.QPointF(x, y),
            button,
            button,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

    def prepare_selection(self, tool):
        self.widget._error = None
        self.widget.set_tool(tool)
        self.widget.set_depth_mode("through")
        self.widget.set_view("front")
        screen, _, _ = project_points(
            self.points, self.widget._matrix(), *self.widget._physical_size()
        )
        return screen / self.widget.devicePixelRatioF()

    def test_polygon_closes_near_start_and_on_double_click(self):
        for double_click in (False, True):
            self.prepare_selection("polygon")
            completed = []
            self.widget.selection_completed.connect(completed.append)
            for x, y in ((20, 20), (380, 20), (380, 280), (20, 280)):
                self.widget._mouse_press(
                    self.event(QtCore.QEvent.Type.MouseButtonPress, x, y)
                )
            self.assertTrue(self.widget._near_polygon_start((24, 24)))
            if double_click:
                self.widget._mouse_double_click(
                    self.event(QtCore.QEvent.Type.MouseButtonDblClick, 20, 280)
                )
            else:
                self.widget._mouse_press(
                    self.event(QtCore.QEvent.Type.MouseButtonPress, 24, 24)
                )
            self.assertEqual(len(completed), 1)
            self.assertFalse(self.widget.selection_active)
            self.widget.selection_completed.disconnect(completed.append)

    def test_polygon_survives_zoom_and_control_pan(self):
        self.prepare_selection("polygon")
        widget = self.widget
        completed = []
        widget.selection_completed.connect(completed.append)
        for x, y in ((80, 60), (320, 60), (320, 240), (80, 240)):
            widget._mouse_press(
                self.event(QtCore.QEvent.Type.MouseButtonPress, x, y)
            )
        vertices = np.array(widget._polygon)
        expected = select_polygon(
            widget._snapshot, vertices * widget.devicePixelRatioF()
        )
        old_scale = widget._scale
        event = QtGui.QWheelEvent(
            QtCore.QPointF(200, 150),
            QtCore.QPointF(200, 150),
            QtCore.QPoint(),
            QtCore.QPoint(0, 120),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        widget._wheel(event)
        transformed = (vertices - [200, 150]) * old_scale / widget._scale + [
            200,
            150,
        ]
        np.testing.assert_allclose(widget._polygon, transformed)
        self.assertTrue(widget.selection_active)
        press = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(200, 150),
            QtCore.QPointF(200, 150),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        widget._mouse_press(press)
        self.assertEqual(len(widget._polygon), 4)
        widget._mouse_move(self.event(QtCore.QEvent.Type.MouseMove, 220, 160))
        widget._mouse_release(
            self.event(QtCore.QEvent.Type.MouseButtonRelease, 220, 160)
        )
        np.testing.assert_allclose(widget._polygon, transformed + [20, 10])
        self.assertFalse(completed)
        widget.finish_polygon()
        self.assertEqual(len(completed), 1)
        np.testing.assert_array_equal(completed[0], expected)

    def test_display_changes_preserve_points_and_fit_ignores_filter(self):
        original = self.points.copy()
        self.widget.set_visible_mask(np.array([False, False, False]))
        self.widget.set_view("top")
        self.widget.set_point_size(6)
        self.widget.set_colors(np.array([[255, 0, 0]] * 3, dtype=np.uint8))
        self.widget.fit_all()
        _, _, in_view = project_points(
            self.points, self.widget._matrix(), *self.widget._physical_size()
        )
        self.assertTrue(in_view.all())
        self.assertFalse(self.widget._visible.any())
        np.testing.assert_array_equal(self.points, original)

    def test_camera_drag_buttons_pan_or_rotate_without_changing_points(self):
        completed = []
        self.widget.selection_completed.connect(completed.append)
        original = self.points.copy()
        cases = [("browse", QtCore.Qt.MouseButton.LeftButton, True)]
        cases.extend(
            (tool, button, pan)
            for tool in ("browse", "brush", "polygon")
            for button, pan in (
                (QtCore.Qt.MouseButton.MiddleButton, True),
                (QtCore.Qt.MouseButton.RightButton, False),
            )
        )
        for tool, button, pan in cases:
            with self.subTest(tool=tool, button=button):
                self.widget.reset_view()
                self.widget.set_tool(tool)
                center = self.widget._center.copy()
                angles = (self.widget._yaw, self.widget._pitch)
                screen, _, _ = project_points(
                    self.points,
                    self.widget._matrix(),
                    *self.widget._physical_size(),
                )
                self.widget._mouse_press(
                    self.event(
                        QtCore.QEvent.Type.MouseButtonPress,
                        120,
                        140,
                        button,
                    )
                )
                self.widget._mouse_move(
                    self.event(QtCore.QEvent.Type.MouseMove, 156, 116, button)
                )
                self.widget._mouse_release(
                    self.event(
                        QtCore.QEvent.Type.MouseButtonRelease,
                        156,
                        116,
                        button,
                    )
                )
                moved, _, _ = project_points(
                    self.points,
                    self.widget._matrix(),
                    *self.widget._physical_size(),
                )
                if pan:
                    self.assertFalse(
                        np.array_equal(self.widget._center, center)
                    )
                    self.assertEqual(
                        (self.widget._yaw, self.widget._pitch), angles
                    )
                    np.testing.assert_allclose(
                        (moved - screen) / self.widget.devicePixelRatioF(),
                        np.tile([36, -24], (len(self.points), 1)),
                        atol=1e-4,
                    )
                else:
                    np.testing.assert_array_equal(self.widget._center, center)
                    self.assertNotEqual(self.widget._yaw, angles[0])
                    self.assertNotEqual(self.widget._pitch, angles[1])
                    self.assertFalse(np.array_equal(moved, screen))
                self.assertFalse(self.widget.selection_active)
                np.testing.assert_array_equal(self.points, original)
                self.assertEqual(completed, [])

    def test_fixed_view_signal_survives_pan_zoom_and_clears_on_rotation(self):
        views = []
        self.widget.camera_view_changed.connect(views.append)
        for view in ("top", "front", "side"):
            self.widget.set_view(view)
        self.assertEqual(views, ["top", "front", "side"])
        views.clear()
        for button in (
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.MiddleButton,
        ):
            self.widget._mouse_press(
                self.event(
                    QtCore.QEvent.Type.MouseButtonPress, 120, 140, button
                )
            )
            self.widget._mouse_move(
                self.event(QtCore.QEvent.Type.MouseMove, 156, 116, button)
            )
            self.widget._mouse_release(
                self.event(
                    QtCore.QEvent.Type.MouseButtonRelease, 156, 116, button
                )
            )
        self.widget._wheel(
            QtGui.QWheelEvent(
                QtCore.QPointF(120, 140),
                QtCore.QPointF(120, 140),
                QtCore.QPoint(),
                QtCore.QPoint(0, 120),
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
                QtCore.Qt.ScrollPhase.NoScrollPhase,
                False,
            )
        )
        self.widget.fit_all()
        self.assertEqual(views, [])
        self.widget._mouse_press(
            self.event(
                QtCore.QEvent.Type.MouseButtonPress,
                120,
                140,
                QtCore.Qt.MouseButton.RightButton,
            )
        )
        self.widget._mouse_move(
            self.event(
                QtCore.QEvent.Type.MouseMove,
                120,
                140,
                QtCore.Qt.MouseButton.RightButton,
            )
        )
        self.assertEqual(views, [])
        self.widget._mouse_move(
            self.event(
                QtCore.QEvent.Type.MouseMove,
                156,
                116,
                QtCore.Qt.MouseButton.RightButton,
            )
        )
        self.assertEqual(views, [""])
        self.widget.reset_view()
        self.assertEqual(views, ["", ""])

    def test_brush_commits_entire_path_once_and_cancellation_commits_nothing(
        self,
    ):
        screen = self.prepare_selection("brush")
        camera = self.widget._matrix().copy()
        completed = []
        self.widget.selection_completed.connect(completed.append)
        self.widget._mouse_press(
            self.event(QtCore.QEvent.Type.MouseButtonPress, *screen[0])
        )
        self.widget._mouse_move(
            self.event(QtCore.QEvent.Type.MouseMove, *screen[1])
        )
        self.assertEqual(completed, [])
        self.widget._mouse_release(
            self.event(QtCore.QEvent.Type.MouseButtonRelease, *screen[1])
        )
        self.assertEqual(len(completed), 1)
        np.testing.assert_array_equal(completed[0], [0, 1])
        np.testing.assert_array_equal(self.widget._matrix(), camera)
        self.widget._mouse_press(
            self.event(QtCore.QEvent.Type.MouseButtonPress, *screen[0])
        )
        self.widget.cancel_selection()
        self.widget._mouse_release(
            self.event(QtCore.QEvent.Type.MouseButtonRelease, *screen[0])
        )
        self.assertEqual(len(completed), 1)
        self.assertFalse(self.widget.selection_active)

    def test_filter_change_cancels_unfinished_selection(self):
        screen = self.prepare_selection("brush")
        completed = []
        cancelled = []
        self.widget.selection_completed.connect(completed.append)
        self.widget.selection_cancelled.connect(lambda: cancelled.append(True))
        self.widget._mouse_press(
            self.event(QtCore.QEvent.Type.MouseButtonPress, *screen[0])
        )
        self.widget.set_visible_mask(np.ones(3, dtype=bool))
        self.assertFalse(self.widget.selection_active)
        self.assertEqual(completed, [])
        self.assertEqual(cancelled, [True])

    def test_selection_cache_tracks_geometry_without_rebuilding_for_colors(
        self,
    ):
        if self.widget._gl is not None:
            self.widget.show()
            self.app.processEvents()
        self.prepare_selection("brush")
        self.assertTrue(self.widget._begin_selection())
        snapshot = self.widget._snapshot
        self.widget.cancel_selection()
        self.widget.set_colors(np.array([[255, 0, 0]], dtype=np.uint8), [0])
        self.widget.set_visible_mask(np.ones(3, dtype=bool))
        self.assertTrue(self.widget._begin_selection())
        self.assertIs(self.widget._snapshot, snapshot)
        self.widget.set_visible_mask([False], [0])
        self.assertTrue(self.widget._begin_selection())
        self.assertIsNot(self.widget._snapshot, snapshot)
        np.testing.assert_array_equal(self.widget._snapshot.indices, [1, 2])
        for change in (
            lambda: self.widget.set_view("top"),
            lambda: self.widget.set_point_size(5),
            lambda: self.widget.set_depth_mode("surface"),
            lambda: self.widget.resize(500, 350),
            lambda: self.widget.set_cloud(self.points.copy()),
        ):
            snapshot = self.widget._snapshot
            self.widget.cancel_selection()
            change()
            self.assertTrue(self.widget._begin_selection())
            self.assertIsNot(self.widget._snapshot, snapshot)

    def test_indexed_display_updates_preserve_other_points(self):
        colors = self.widget._colors.copy()
        self.widget.set_colors(np.array([[255, 0, 0]], dtype=np.uint8), [1])
        np.testing.assert_array_equal(
            self.widget._colors[[0, 2]], colors[[0, 2]]
        )
        np.testing.assert_array_equal(self.widget._colors[1], [1, 0, 0, 1])
        self.widget.set_visible_mask([False], [1])
        np.testing.assert_array_equal(
            self.widget._visible, [True, False, True]
        )
        self.assertEqual(self.widget._visible_count, 2)
        for indices in ([-1], [3], [[0]]):
            with self.assertRaises(ValueError):
                self.widget.set_colors([[0, 0, 0]], indices)
            with self.assertRaises(ValueError):
                self.widget.set_visible_mask([True], indices)

    def test_polygon_requires_valid_contour_and_explicit_completion(self):
        screen = self.prepare_selection("polygon")
        camera = self.widget._matrix().copy()
        completed = []
        self.widget.selection_completed.connect(completed.append)
        center = screen[0]
        vertices = center + np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]])
        for vertex in vertices[:2]:
            self.widget._mouse_press(
                self.event(QtCore.QEvent.Type.MouseButtonPress, *vertex)
            )
        self.widget.finish_polygon()
        self.assertTrue(self.widget.selection_active)
        self.assertEqual(completed, [])
        for vertex in vertices[2:]:
            self.widget._mouse_press(
                self.event(QtCore.QEvent.Type.MouseButtonPress, *vertex)
            )
        self.widget.finish_polygon()
        self.assertFalse(self.widget.selection_active)
        np.testing.assert_array_equal(completed[0], [0])
        np.testing.assert_array_equal(self.widget._matrix(), camera)

    def test_unsupported_qt_platform_does_not_construct_opengl_widget(self):
        if QtGui.QGuiApplication.platformName() not in (
            "offscreen",
            "minimal",
        ):
            self.skipTest("Requires an unsupported Qt platform")
        self.assertIsNone(self.widget._gl)
        self.assertIn("OpenGL", self.widget._error)
        self.assertEqual(len(self.widget._points), 3)

    def test_native_renderer_uses_matching_square_point_footprint(self):
        if self.widget._gl is None:
            self.skipTest("Requires a native OpenGL display")
        self.widget.resize(400, 300)
        self.widget.set_cloud(np.array([[-0.5, 0.5, 0, 1]], dtype=np.float32))
        self.widget._matrix = lambda: np.eye(4, dtype=np.float32)
        self.widget.set_colors(np.array([[255, 0, 0]], dtype=np.uint8))
        self.widget.show()
        self.app.processEvents()
        for size in (1, 2, 3, 4, 10):
            self.widget.set_point_size(size)
            image = self.widget._gl.grabFramebuffer()
            self.assertIsNone(self.widget._error)
            ratio = self.widget.devicePixelRatioF()
            pixel_size = round(size * ratio)
            left = int(100 * ratio) - pixel_size // 2
            top = int(75 * ratio) - pixel_size // 2
            for y in range(top - 1, top + pixel_size + 1):
                for x in range(left - 1, left + pixel_size + 1):
                    is_red = image.pixelColor(x, y).red() == 255
                    expected = (
                        left <= x < left + pixel_size
                        and top <= y < top + pixel_size
                    )
                    self.assertEqual(is_red, expected, (size, x, y))


if __name__ == "__main__":
    unittest.main()


class TestNativeSurfaceSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )
        if QtGui.QGuiApplication.platformName() in ("offscreen", "minimal"):
            raise unittest.SkipTest("Requires a native OpenGL display")

    def test_scene_allocation_failure_clears_stale_errors_before_fallback(
        self,
    ):
        widget = PointCloudViewport()
        widget.resize(400, 300)
        widget.set_cloud(np.array([[0, 0, 0]], dtype=np.float32))
        widget.set_colors(np.array([[255, 0, 0]], dtype=np.uint8))
        widget.show()
        self.app.processEvents()
        try:
            gl = widget._gl
            if not gl.scene_cache_supported:
                self.skipTest("Requires framebuffer blit support")
            expected = gl.grabFramebuffer()
            failed_framebuffer = Mock()
            failed_framebuffer.size.return_value = QtCore.QSize(
                *widget._physical_size()
            )
            failed_framebuffer.isValid.return_value = False
            widget._update(data=True)
            with (
                patch.object(gl, "scene_framebuffer", failed_framebuffer),
                patch.object(
                    gl.functions,
                    "glGetError",
                    side_effect=[0x0505, 0x0502, 0, 0, 0, 0],
                ) as errors,
            ):
                self.assertEqual(gl.grabFramebuffer(), expected)
                self.assertEqual(errors.call_count, 6)
            self.assertIsNone(widget._error)
            self.assertFalse(gl.scene_cache_supported)
            self.assertFalse(gl.data_dirty)
            self.assertFalse(gl.color_dirty)
            self.assertFalse(gl.visibility_dirty)
        finally:
            widget.close()
            widget.deleteLater()
            self.app.processEvents()

    def test_preview_reuses_scene_color_and_depth_without_drawing_cloud(self):
        widget = PointCloudViewport()
        widget.resize(400, 300)
        widget.set_cloud(
            np.array(
                [[0, 0, -0.5], [0, 0, 0.5], [-0.5, 0.5, 0], [0.5, 0.5, 0]],
                dtype=np.float32,
            )
        )
        matrix = np.eye(4, dtype=np.float32)
        widget._matrix = lambda: matrix
        widget.set_colors(np.tile([0, 255, 0], (4, 1)))
        widget.show()
        self.app.processEvents()
        try:
            gl = widget._gl
            if not gl.scene_cache_supported:
                self.skipTest("Requires framebuffer blit support")
            initial = gl.grabFramebuffer()
            framebuffer, key = gl.scene_framebuffer, gl.scene_key
            owners = gl.capture_surface()
            widget._preview = np.array([1, 2], dtype=np.int64)
            widget._update(preview=True)
            with patch.object(
                gl.functions, "glDrawArrays", wraps=gl.functions.glDrawArrays
            ) as draw:
                cached = gl.grabFramebuffer()
                self.assertEqual(
                    [call.args for call in draw.call_args_list], [(0, 0, 2)]
                )
            self.assertIs(gl.scene_framebuffer, framebuffer)
            self.assertEqual(gl.scene_key, key)
            ratio = widget.devicePixelRatioF()
            self.assertEqual(
                cached.pixelColor(
                    round(200 * ratio), round(150 * ratio)
                ).getRgb(),
                (0, 255, 0, 255),
            )
            self.assertEqual(
                cached.pixelColor(round(100 * ratio), round(75 * ratio)).red(),
                255,
            )
            gl.scene_cache_supported = False
            self.assertEqual(gl.grabFramebuffer(), cached)
            gl.scene_cache_supported = True
            np.testing.assert_array_equal(gl.capture_surface(), owners)
            widget.cancel_selection()
            with patch.object(
                gl.functions, "glDrawArrays", wraps=gl.functions.glDrawArrays
            ) as draw:
                self.assertEqual(gl.grabFramebuffer(), initial)
                draw.assert_not_called()
            matrix[0, 3] = 0.25
            with patch.object(
                gl.functions, "glDrawArrays", wraps=gl.functions.glDrawArrays
            ) as draw:
                moved = gl.grabFramebuffer()
                self.assertEqual(
                    [call.args for call in draw.call_args_list], [(0, 0, 4)]
                )
            gl.scene_cache_supported = False
            self.assertEqual(gl.grabFramebuffer(), moved)
            gl.scene_cache_supported = True
            widget.resize(500, 350)
            self.app.processEvents()
            resized = gl.grabFramebuffer()
            self.assertIsNot(gl.scene_framebuffer, framebuffer)
            gl.scene_cache_supported = False
            self.assertEqual(gl.grabFramebuffer(), resized)
        finally:
            widget.close()
            widget.deleteLater()
            self.app.processEvents()

    def test_color_and_filter_updates_only_upload_changed_attributes(self):
        widget = PointCloudViewport()
        widget.resize(400, 300)
        points = np.full((131072, 3), 2, dtype=np.float32)
        points[32] = [-0.5, 0.5, 0]
        points[65536] = [0.5, 0.5, 0]
        widget.set_cloud(points)
        widget._matrix = lambda: np.eye(4, dtype=np.float32)
        widget.set_colors(np.tile([0, 255, 0], (len(points), 1)))
        widget.show()
        self.app.processEvents()
        try:
            gl = widget._gl
            self.assertIsNone(widget._error)
            owners = gl.capture_surface()
            framebuffer = gl.picking_framebuffer
            ratio = widget.devicePixelRatioF()
            left, right, y = (
                round(100 * ratio),
                round(300 * ratio),
                round(75 * ratio),
            )
            self.assertEqual(owners[y, left], 33)
            self.assertEqual(owners[y, right], 65537)
            with patch.object(
                gl, "_upload_buffer", wraps=gl._upload_buffer
            ) as upload:
                widget.set_colors(
                    np.array([[255, 0, 0]], dtype=np.uint8), [32]
                )
                widget.set_colors(
                    np.array([[0, 0, 255]], dtype=np.uint8), [65536]
                )
                image = gl.grabFramebuffer()
                self.assertEqual(
                    image.pixelColor(left, y).getRgb(), (255, 0, 0, 255)
                )
                self.assertEqual(
                    image.pixelColor(right, y).getRgb(), (0, 0, 255, 255)
                )
                upload.assert_not_called()
                np.testing.assert_array_equal(gl.capture_surface(), owners)
                self.assertIs(gl.picking_framebuffer, framebuffer)
                upload.assert_not_called()
                widget.set_visible_mask([False], [32])
                gl.grabFramebuffer()
                self.assertEqual(len(upload.call_args_list), 1)
                self.assertIs(upload.call_args.args[0], gl.visibility_buffer)
                owners = gl.capture_surface()
                self.assertEqual(owners[y, left], 0)
                self.assertEqual(owners[y, right], 65537)
        finally:
            widget.close()
            widget.deleteLater()
            self.app.processEvents()

    def test_surface_follows_rendered_depth_and_freezes_filter_snapshot(self):
        from anylabeling.views.labeling.pointcloud.selection import (
            select_brush,
            select_polygon,
        )

        widget = PointCloudViewport()
        widget.resize(400, 300)
        points = np.array(
            [
                [0, 0, -0.5, 1],
                [0, 0, 0.5, 1],
                [0, 0, -0.5, 1],
                [0.5, 0.5, 0, 1],
            ],
            dtype=np.float32,
        )
        widget.set_cloud(points)
        widget._matrix = lambda: np.eye(4, dtype=np.float32)
        widget.show()
        self.app.processEvents()
        try:
            self.assertIsNone(widget._error)
            ratio = widget.devicePixelRatioF()
            center = np.array([200, 150]) * ratio
            polygon = (
                center + np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]]) * ratio
            )
            for size in (1, 2, 4, 10):
                with self.subTest(size=size):
                    widget.set_point_size(size)
                    widget.set_depth_mode("surface")
                    widget.set_visible_mask(np.ones(4, dtype=bool))
                    self.assertTrue(widget._begin_selection())
                    snapshot = widget._snapshot
                    np.testing.assert_array_equal(
                        select_brush(snapshot, center, center, 20 * ratio),
                        [0, 2],
                    )
                    np.testing.assert_array_equal(
                        select_polygon(snapshot, polygon), [0, 2]
                    )
                    widget.set_visible_mask(
                        np.array([False, True, False, True])
                    )
                    self.assertFalse(widget.selection_active)
                    self.assertTrue(widget._begin_selection())
                    np.testing.assert_array_equal(
                        select_brush(
                            widget._snapshot, center, center, 20 * ratio
                        ),
                        [1],
                    )
                    np.testing.assert_array_equal(
                        select_brush(snapshot, center, center, 20 * ratio),
                        [0, 2],
                    )
                    widget.set_depth_mode("through")
                    self.assertTrue(widget._begin_selection())
                    np.testing.assert_array_equal(
                        select_brush(
                            widget._snapshot, center, center, 20 * ratio
                        ),
                        [1],
                    )
                    widget.set_visible_mask(np.ones(4, dtype=bool))
                    self.assertTrue(widget._begin_selection())
                    np.testing.assert_array_equal(
                        select_brush(
                            widget._snapshot, center, center, 20 * ratio
                        ),
                        [0, 1, 2],
                    )
        finally:
            widget.close()
            widget.deleteLater()
            self.app.processEvents()
