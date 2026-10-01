from dataclasses import replace
import math

import numpy as np
import pytest
from anylabeling.views.labeling.pointcloud.cuboid import Cuboid
from anylabeling.views.labeling.pointcloud.cuboid_viewport import (
    CuboidViewport,
)
from anylabeling.views.labeling.pointcloud.selection import (
    SelectionSnapshot,
    project_points,
    select_brush,
    select_polygon,
)

from .test_dialog import app


@pytest.fixture
def view(app):
    widget = CuboidViewport()
    widget.resize(400, 300)
    widget.set_cloud(np.array([[0, 0, 0]], dtype=np.float32))
    widget._yaw = widget._pitch = 0
    widget._center = np.zeros(3)
    widget._set_scale(5 * math.tan(math.radians(25)))
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_perspective_projection_depth_and_overlay_match(view):
    points = np.array([[1, -2, 0], [1, 2, 0], [0, -6, 0]], dtype=np.float32)
    screen, depths, valid = project_points(
        points, view._matrix(), view.width(), view.height()
    )
    np.testing.assert_allclose(view.project(points[:2]), screen[:2], atol=1e-4)
    assert screen[0, 0] - 200 > screen[1, 0] - 200 > 0
    assert depths[0] < depths[1]
    assert valid.tolist() == [True, True, False]
    assert np.isnan(view.project(points[2:])).all()
    pixel = (230, 170)
    np.testing.assert_allclose(view.project([view.unproject(pixel)])[0], pixel)


def test_dolly_enlarges_near_points_and_preserves_target(view):
    view._points = np.array([[0, -2, 0], [0, 2, 0]], dtype=np.float32)
    view.set_point_size(10)
    sizes = view._point_sizes(view._matrix())
    assert sizes[0] > sizes[1]
    center = view._center.copy()
    view._set_scale(view._scale / 2)
    closer = view._point_sizes(view._matrix())
    assert np.all(closer > sizes)
    np.testing.assert_array_equal(view._center, center)


def test_cuboid_crossing_near_plane_remains_selectable(view):
    box = Cuboid(1, 10, (0, -5, 0), (2, 2, 2), (0, 0, 0))
    view.set_cuboids([box], None)
    assert view._hit_box((view.width() / 2, view.height() / 2)) == box.id


@pytest.mark.parametrize("name", ["top", "front", "side"])
def test_side_views_keep_fixed_screen_point_sizes(app, name):
    widget = CuboidViewport(view=name)
    widget.resize(400, 300)
    try:
        widget.set_point_size(2)
        before = widget.project([[1, 1, 1]])
        size = widget._pixel_size()
        widget._set_scale(widget._scale / 2)
        after = widget.project([[1, 1, 1]])
        np.testing.assert_allclose(
            after - [200, 150], 2 * (before - [200, 150])
        )
        assert widget._pixel_size() == size
        assert widget._point_scale() == 0
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("mode", ["through", "surface"])
def test_variable_point_footprints(mode):
    screen = np.array([[20, 20], [40, 20]], dtype=np.float32)
    snapshot = SelectionSnapshot.create(
        screen,
        np.array([0.2, 0.3], dtype=np.float32),
        np.ones(2, dtype=bool),
        64,
        64,
        np.array([7, 1]),
        mode,
    )
    np.testing.assert_array_equal(
        select_brush(snapshot, [17.5, 20.5], [17.5, 20.5], 0.1), [0]
    )
    assert not len(select_brush(snapshot, [37.5, 20.5], [37.5, 20.5], 0.1))


def test_gpu_perspective_footprints_and_occlusion_match_selection(view, app):
    if view._gl is None:
        pytest.skip("An OpenGL display is required")
    view._points = np.array(
        [
            [-0.5, -2, 0],
            [0.5, 2, 0],
            [0, -2, 0],
            [0, 2, 0],
            [0, -2, 0],
        ],
        dtype=np.float32,
    )
    points = view._points.copy()
    view.set_cloud(points)
    view._yaw = view._pitch = 0
    view._center = np.zeros(3)
    view._set_scale(5 * math.tan(math.radians(25)))
    view.set_point_size(10)
    view.show()
    app.processEvents()
    assert view._error is None
    for distance in (5, 3):
        view._set_scale(distance * math.tan(math.radians(25)))
        owners = view._gl.capture_surface()
        screen, _, valid = project_points(
            points, view._matrix(), *view._physical_size()
        )
        sizes = view._point_sizes(view._matrix())
        for index in (0, 1):
            x, y = np.floor(screen[index]).astype(int) - sizes[index] // 2
            assert valid[index]
            footprint = owners[y : y + sizes[index], x : x + sizes[index]]
            assert footprint.shape == (sizes[index], sizes[index])
            assert np.all(footprint == index + 1)
            assert np.count_nonzero(owners == index + 1) == sizes[index] ** 2
        center = np.array(view._physical_size()) / 2 + 0.5
        for mode, expected in (("surface", [2, 4]), ("through", [2, 3, 4])):
            view.set_depth_mode(mode)
            assert view._begin_selection()
            np.testing.assert_array_equal(
                select_brush(view._snapshot, center, center, 0.1), expected
            )
            polygon = center + np.array(
                [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]
            )
            np.testing.assert_array_equal(
                select_polygon(view._snapshot, polygon), expected
            )


@pytest.mark.parametrize("name", [None, "top", "front", "side"])
def test_selected_cuboid_fill_and_clear_use_real_gl(app, name):
    widget = CuboidViewport(view=name)
    if widget._gl is None:
        widget.deleteLater()
        pytest.skip("An OpenGL display is required")
    widget.resize(400, 300)
    widget.detection_enabled = True
    widget.set_cloud(np.array([[0, 0, 0]], dtype=np.float32))
    box = Cuboid(1, 10, (0, 0, 0), (4, 2, 2))
    widget.show()
    app.processEvents()
    try:
        widget.align_cuboid(box, fit=True)
        widget.set_cuboids([box], None)
        assert widget._error is None
        original = widget._gl.capture_surface().copy()
        ratio = widget.devicePixelRatioF()
        x, y = round(205 * ratio), round(155 * ratio)
        assert widget._gl.grabFramebuffer().pixelColor(x, y).red() == 0
        widget.set_cuboids([box], box.id)
        image = widget._gl.grabFramebuffer()
        color = image.pixelColor(x, y)
        assert 25 <= color.red() <= 35
        assert 40 <= color.green() <= 60
        assert 65 <= color.blue() <= 85
        np.testing.assert_array_equal(widget._gl.capture_surface(), original)
        widget.set_cuboids([box], None)
        assert widget._gl.grabFramebuffer().pixelColor(x, y).red() == 0
        assert widget._show_axes == (name is None)
        assert bool(len(widget._cuboid_mesh[0])) == (name is None)
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_overlapping_cuboids_select_nearest_surface(view):
    near = Cuboid(1, 10, (0, -2, 0), (4, 1, 4))
    far = Cuboid(2, 10, (0, 2, 0), (1, 1, 1))
    view.set_cuboids([far, near], None)
    assert view._hit_box((200, 150)) == near.id


def test_rotating_offset_cuboid_stays_registered_with_rendered_points(
    view, app
):
    if view._gl is None:
        pytest.skip("An OpenGL display is required")
    box = Cuboid(1, 10, (8, -5, 2), (4, 2, 1))
    view._show_axes = False
    view.set_cloud(np.array([box.center], dtype=np.float32))
    view.set_colors(np.array([[1, 1, 1, 1]], dtype=np.float32))
    view.set_point_size(10)
    view._center = np.array(box.center) + (0.4, 0.2, 0)
    view._set_scale(4)
    view.show()
    app.processEvents()
    for yaw, pitch in ((-45, 35), (20, -15), (70, 60)):
        view._yaw, view._pitch = yaw, pitch
        for angles in ((0.2, -0.3, 0.4), (1.2, 0.5, -0.8), (-0.5, 1.2, 1.5)):
            view.set_cuboids([replace(box, rotation=angles)], box.id)
            owners = view._gl.capture_surface()
            image = view._gl.grabFramebuffer()
            center = view.project([box.center])[0] * view.devicePixelRatioF()
            x, y = np.floor(center).astype(int)
            assert owners[y, x] == 1
            color = image.pixelColor(x, y)
            assert color.red() > 100 and color.green() > 100
            offset = max(4, int(view._point_sizes(view._matrix())[0]) // 2 + 2)
            fill = image.pixelColor(x + offset, y)
            assert 20 <= fill.red() <= 40
            assert 45 <= fill.green() <= 65
            assert 60 <= fill.blue() <= 90
            assert view._error is None


def test_creation_ray_chooses_nearest_visible_depth_and_ignores_template(view):
    view.set_cloud(
        np.array([[0, -2, 0], [0, 2, 0], [0, -6, 0]], dtype=np.float32)
    )
    view._yaw = view._pitch = 0
    view._center = np.zeros(3)
    view._set_scale(5 * math.tan(math.radians(25)))
    view.creating = True
    view._update_creation_preview((200, 150))
    np.testing.assert_allclose(view._creation_preview.center, (0, -2, 0))
    cached = view._creation_ray_cache
    for _ in range(3):
        view._update_creation_preview((200, 150))
        np.testing.assert_allclose(view._creation_preview.center, (0, -2, 0))
        assert view._creation_ray_cache is cached
    view.set_visible_mask([False, True, True])
    assert view._creation_preview is None
    view._update_creation_preview((200, 150))
    np.testing.assert_allclose(view._creation_preview.center, (0, 2, 0))
    view.set_visible_mask([False, False, True])
    view._update_creation_preview((200, 150))
    assert view._creation_preview is None


def test_creation_preview_is_amber_and_uses_same_gl_camera_as_cloud(view, app):
    if view._gl is None:
        pytest.skip("An OpenGL display is required")
    view.show()
    app.processEvents()
    view.set_colors(np.array([[1, 1, 1, 1]], dtype=np.float32))
    view.creating = True
    view._update_creation_preview((200, 150))
    assert view._creation_preview is not None
    assert view.cuboids == ()
    ratio = view.devicePixelRatioF()
    image = view._gl.grabFramebuffer()
    color = image.pixelColor(round(204 * ratio), round(154 * ratio))
    assert 90 <= color.red() <= 115
    assert 75 <= color.green() <= 95
    assert 25 <= color.blue() <= 40
    composited = view.grab().toImage()
    point = composited.pixelColor(round(200 * ratio), round(150 * ratio))
    assert point.red() >= point.green() >= point.blue()
    assert point.red() >= 245
    view.cancel_selection()
    image = view._gl.grabFramebuffer()
    assert image.pixelColor(round(204 * ratio), round(154 * ratio)).red() == 0
