import math
from unittest.mock import patch

import numpy as np
import pytest
from PyQt6 import QtCore, QtGui, QtTest

from .test_cuboids import detection
from .test_dialog import app, window


@pytest.fixture
def camera(detection, app):
    workspace = detection.detection
    workspace.create((2, 1, 0), (2, 2, 2))
    detection.resize(1200, 800)
    detection.show()
    detection.activateWindow()
    QtTest.QTest.qWait(30)
    detection.camera_controls_button.click()
    view = workspace.views[0]
    view.cancel_selection()
    view._center = np.zeros(3)
    view._yaw = view._pitch = 0
    view._set_scale(10 * math.tan(math.radians(25)))
    return view


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize(
    "key,center,distance,angles",
    [
        ("U", (0, 0, -2), 10, (0, 0)),
        ("I", (0, 0, 0), 5, (0, 0)),
        ("O", (0, 0, 2), 10, (0, 0)),
        ("J", (-2, 0, 0), 10, (0, 0)),
        ("K", (0, 0, 0), 15, (0, 0)),
        ("L", (2, 0, 0), 10, (0, 0)),
        ("Up", (0, 0, 0), 10, (0, 10)),
        ("Left", (0, 0, 0), 10, (-20, 0)),
        ("Down", (0, 0, 0), 10, (0, -10)),
        ("Right", (0, 0, 0), 10, (20, 0)),
    ],
)
def test_camera_buttons_and_shortcuts_match_cvat_without_editing_boxes(
    camera, detection, app, index, key, center, distance, angles
):
    workspace = detection.detection
    boxes = detection.document.cuboids
    labels = detection.document.labels.copy()
    history = len(detection.document._undo)
    selected_id = workspace.selected_id
    side_cameras = [view._matrix().copy() for view in workspace.views[1:]]
    start = camera._center.copy(), camera._scale, camera._yaw, camera._pitch
    for button in (True, False):
        camera.cancel_selection()
        camera._center = start[0].copy()
        camera._scale, camera._yaw, camera._pitch = start[1:]
        matrix = camera._matrix().copy()
        if button:
            QtTest.QTest.mouseClick(
                camera.camera_buttons[key], QtCore.Qt.MouseButton.LeftButton
            )
        else:
            view = workspace.views[index]
            view.setFocus()
            app.processEvents()
            QtTest.QTest.keyClick(
                view._gl,
                getattr(QtCore.Qt.Key, f"Key_{key}"),
                (
                    QtCore.Qt.KeyboardModifier.AltModifier
                    if len(key) == 1
                    else QtCore.Qt.KeyboardModifier.ShiftModifier
                ),
            )
        assert (
            camera._focus_animation.state()
            == QtCore.QAbstractAnimation.State.Running
        )
        np.testing.assert_array_equal(camera._matrix(), matrix)
        camera._focus_animation.setCurrentTime(150)
        assert not np.array_equal(camera._matrix(), matrix)
        camera._focus_animation.setCurrentTime(
            camera._focus_animation.duration()
        )
        np.testing.assert_allclose(camera._center, center, atol=1e-8)
        assert camera._scale == pytest.approx(
            distance * math.tan(math.radians(25))
        )
        assert (camera._yaw, camera._pitch) == angles
        assert detection.document.cuboids == boxes
        np.testing.assert_array_equal(detection.document.labels, labels)
        assert len(detection.document._undo) == history
        assert workspace.selected_id == selected_id
        for view, matrix in zip(workspace.views[1:], side_cameras):
            np.testing.assert_array_equal(view._matrix(), matrix)


def test_repeated_camera_commands_accumulate_without_reversing(camera):
    for _ in range(12):
        camera.camera_buttons["Right"].click()
    camera._focus_animation.setCurrentTime(200)
    assert camera._yaw > 0
    camera._focus_animation.setCurrentTime(camera._focus_animation.duration())
    assert camera._yaw == 240
    for key, bound in (("I", 0), ("K", 1)):
        for _ in range(30):
            camera.camera_buttons[key].click()
        camera._focus_animation.setCurrentTime(
            camera._focus_animation.duration()
        )
        assert camera._scale == pytest.approx(camera._scale_limits[bound])
    for key, sign in (("Up", 1), ("Down", -1)):
        for _ in range(30):
            camera.camera_buttons[key].click()
        camera._focus_animation.setCurrentTime(
            camera._focus_animation.duration()
        )
        assert 89 < camera._pitch * sign < 90


def test_camera_shortcuts_do_not_override_text_editing(camera, detection, app):
    detection.detection.edit_properties()
    field = detection.detection.fields["center"][0].lineEdit()
    field.setFocus()
    field.deselect()
    field.setCursorPosition(len(field.text()))
    app.processEvents()
    matrix = camera._matrix().copy()
    QtTest.QTest.keyClick(
        field,
        QtCore.Qt.Key.Key_Left,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
    )
    assert field.hasSelectedText()
    np.testing.assert_array_equal(camera._matrix(), matrix)
    assert (
        camera._focus_animation.state()
        == QtCore.QAbstractAnimation.State.Stopped
    )


def test_camera_controls_remain_at_canvas_corners(camera, detection, app):
    workspace = detection.detection
    for width, height in ((960, 700), (1400, 900)):
        detection.resize(width, height)
        for expanded in (False, True):
            workspace.toggle_views(show=expanded)
            app.processEvents()
            left, right = camera._camera_panels
            assert left.x() == 12
            assert right.x() + right.width() == camera.width() - 12
            for panel in (left, right):
                assert panel.y() + panel.height() == camera.height() - 12
            assert all(
                button.isVisible() for button in camera.camera_buttons.values()
            )
    assert all(not view.camera_buttons for view in workspace.views[1:])


def test_camera_controls_default_hidden_and_toggle_preserves_navigation(
    detection, app
):
    detection.show()
    detection.activateWindow()
    app.processEvents()
    camera = detection.viewport
    button = detection.camera_controls_button
    assert not button.isChecked()
    assert not button.icon().isNull()
    assert all(panel.isHidden() for panel in camera._camera_panels)
    assert button.toolTip() == detection.tr("Show camera controls")
    detection._toggle_panel(2)
    app.processEvents()
    buttons = [
        button,
        detection.detection.views_button,
        detection.show_annotation_button,
    ]
    centers = [item.mapToGlobal(item.rect().center()) for item in buttons]
    assert centers[0].x() < centers[1].x() < centers[2].x()
    assert len({center.y() for center in centers}) == 1
    start = camera._matrix().copy()
    button.click()
    assert all(panel.isVisible() for panel in camera._camera_panels)
    assert button.toolTip() == detection.tr("Hide camera controls")
    np.testing.assert_array_equal(camera._matrix(), start)
    button.click()
    assert all(panel.isHidden() for panel in camera._camera_panels)
    for show in (False, True):
        detection.detection.toggle_views(show=show)
        app.processEvents()
        assert all(panel.isHidden() for panel in camera._camera_panels)
    camera.setFocus()
    app.processEvents()
    QtTest.QTest.keyClick(
        camera._gl, QtCore.Qt.Key.Key_J, QtCore.Qt.KeyboardModifier.AltModifier
    )
    assert (
        camera._focus_animation.state()
        == QtCore.QAbstractAnimation.State.Running
    )
    camera._focus_animation.setCurrentTime(camera._focus_animation.duration())
    assert not np.array_equal(camera._matrix(), start)
    assert all(panel.isHidden() for panel in camera._camera_panels)


@pytest.mark.parametrize(
    "points,origin",
    [
        ([[-2, -3, -4], [6, 7, 8]], [0, 0, 0]),
        ([[10, 20, 30], [20, 40, 60]], [15, 30, 45]),
        ([[0, 0, 0], [5, 10, 15]], [0, 0, 0]),
    ],
)
def test_world_axes_anchor_matches_cvat_and_stays_fixed(
    camera, points, origin
):
    camera.set_cloud(np.array(points, dtype=np.float32))
    axes = camera._axes_vertices.copy()
    np.testing.assert_allclose(axes[::2, :3], np.tile(origin, (3, 1)))
    np.testing.assert_allclose(axes[1::2, :3] - axes[::2, :3], np.eye(3) * 5)
    screen = camera.project(axes[:, :3])
    camera.move_camera("J")
    camera._focus_animation.setCurrentTime(camera._focus_animation.duration())
    assert not np.allclose(camera.project(axes[:, :3]), screen)
    np.testing.assert_array_equal(camera._axes_vertices, axes)
    camera.focus_indices([0])
    np.testing.assert_array_equal(camera._axes_vertices, axes)
    camera.set_cloud(np.empty((0, 4)))
    assert not len(camera._axes_vertices)


def test_world_axes_depth_and_point_picking_use_real_gl(camera, app):
    camera.set_cloud(
        np.array([[-1, -1, -1], [3, 1, 1], [2, -2, 0]], dtype=np.float32)
    )
    camera._center = np.zeros(3)
    camera._yaw = camera._pitch = 0
    camera._set_scale(10 * math.tan(math.radians(25)))
    camera.set_cuboids([], None)
    camera.set_point_size(10)
    camera.set_colors(np.ones((3, 4), dtype=np.float32))
    app.processEvents()
    x, y = np.floor(
        camera.project([[2, -2, 0]])[0] * camera.devicePixelRatioF()
    ).astype(int)
    owners = camera._gl.capture_surface()
    image = camera._gl.grabFramebuffer()
    assert owners[y, x] == 3
    color = image.pixelColor(x, y)
    assert min(color.red(), color.green(), color.blue()) > 240
    camera.set_visible_mask(np.array([True, True, False]))
    image = camera._gl.grabFramebuffer()
    colors = [
        image.pixelColor(x + dx, y + dy)
        for dx in range(-2, 3)
        for dy in range(-2, 3)
    ]
    assert any(
        color.red() > 60 and color.green() < 10 and color.blue() < 10
        for color in colors
    )
    owners = camera._gl.capture_surface()
    assert not owners[y - 2 : y + 3, x - 2 : x + 3].any()


def test_shortcuts_panel_lists_all_camera_bindings(camera, detection):
    with patch(
        "anylabeling.views.labeling.widgets.pointcloud_dialog.ShortcutsDialog"
    ) as dialog:
        detection._show_shortcuts()
    groups = dict(dialog.call_args.args[0])
    rows = groups[detection.tr("3D camera")]
    assert len(rows) == 10
    assert {keys[0] for _, keys in rows} == {
        action.shortcut().toString(
            QtGui.QKeySequence.SequenceFormat.NativeText
        )
        for action in camera.camera_actions.values()
    }
    assert not any(
        "Shift+Arrow keys" in keys
        for _, rows in dialog.call_args.args[0]
        for _, keys in rows
    )
