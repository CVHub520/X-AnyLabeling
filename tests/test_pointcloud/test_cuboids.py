from dataclasses import replace
import json
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

from anylabeling.views.labeling.pointcloud.cuboid import (
    Cuboid,
    MIN_SIZE,
    rotation_angles,
    rotation_matrix,
)
from anylabeling.views.labeling.pointcloud.io import (
    load_cuboids,
    load_frame,
    save_cuboids,
)
from anylabeling.views.labeling.pointcloud.model import (
    AnnotationDocument,
    ClassDefinition,
    Frame,
)

from .test_dialog import app, window, open_cloud, select_class, wait_load


@pytest.fixture
def box():
    return Cuboid(1, 10, (4, 5, 6), (4, 2, 1.5), (0.2, -0.3, 0.6))


@pytest.mark.parametrize(
    "angles", [(0, 0, 0), (0.2, -0.3, 0.6), (0.4, np.pi / 2, 0.8)]
)
def test_rotation_roundtrip_and_orthogonality(angles):
    matrix = rotation_matrix(angles)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        rotation_matrix(rotation_angles(matrix)), matrix, atol=1e-7
    )


def test_rotated_resize_keeps_opposite_face_fixed(box):
    resized = box.resized((0, 2), (1, -1), (2, 0, -0.5))
    old_corner = np.array(box.center) + box.matrix @ (-2, 0, 0.75)
    new_corner = np.array(resized.center) + resized.matrix @ (-3, 0, 1)
    np.testing.assert_allclose(old_corner, new_corner)
    assert resized.size == (6, 2, 2)
    assert box.resized((0,), (1,), (-20, 0, 0)).size[0] == MIN_SIZE


def test_containment_and_fit_use_object_coordinates(box):
    points = (
        np.array([[-1, -0.5, -0.25], [1, 0.5, 0.25]]) @ box.matrix.T
        + box.center
    )
    assert box.contains(points).all()
    assert not box.contains([np.array(box.center) + box.matrix[:, 0] * 3])[0]
    fitted = box.fitted(points)
    np.testing.assert_allclose(fitted.size, (2, 1, 0.5))
    np.testing.assert_allclose(fitted.center, box.center)
    assert fitted.rotation == box.rotation


@pytest.mark.parametrize(
    "change",
    [
        {"size": (0, 1, 1)},
        {"center": (np.nan, 0, 0)},
        {"rotation": (0, 0, np.inf)},
        {"id": True},
        {"class_id": 0},
        {"locked": "false"},
    ],
)
def test_invalid_geometry_rejected(box, change):
    with pytest.raises(ValueError):
        replace(box, **change)


def test_shared_undo_history_and_independent_save_baselines(box, tmp_path):
    document = AnnotationDocument(
        Frame(
            tmp_path / "scan.bin",
            np.zeros((4, 4)),
            np.zeros(4, dtype=np.uint32),
        )
    )
    document.set_cuboid(box)
    document.assign_semantic([0], 10)
    assert document.dirty
    document.mark_saved()
    assert document.dirty and not document.labels_dirty
    document.mark_cuboids_saved(tmp_path / "scan.cuboids.json")
    assert not document.dirty
    document.undo()
    assert document.labels[0] == 0 and document.cuboids == (box,)
    document.undo()
    assert document.cuboids == () and document.cuboids_dirty
    document.redo()
    document.redo()
    assert not document.dirty
    document.set_cuboid(replace(box, locked=True))
    with pytest.raises(ValueError, match="Unlock"):
        document.delete_cuboid(box.id)
    with pytest.raises(ValueError, match="Unlock"):
        document.set_cuboid(replace(box, center=(0, 0, 0)))
    document.set_cuboid(box)
    assert document.delete_cuboid(box.id)


def test_cuboid_json_roundtrip_and_bad_file_retention(box, tmp_path):
    source = tmp_path / "scan.bin"
    np.zeros((4, 4), dtype="<f4").tofile(source)
    target = source.with_suffix(".cuboids.json")
    save_cuboids(target, [box], source)
    assert load_frame(source).cuboids == (box,)
    content = target.read_bytes()
    with pytest.raises(ValueError):
        save_cuboids(target, [box, box], source)
    assert target.read_bytes() == content
    data = json.loads(content)
    data["point_cloud"] = "other.bin"
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="this point cloud"):
        load_frame(source)
    assert not source.with_suffix(".label").exists()


@pytest.fixture
def detection(window, app, tmp_path):
    window.sidebar_tabs.setCurrentWidget(window.detection.panel)
    source = tmp_path / "scan.bin"
    points = np.array(
        [
            [x, y, z, 0.5]
            for x in (-2, 0, 2)
            for y in (-1, 0, 1)
            for z in (-0.5, 0.5)
        ],
        dtype="<f4",
    )
    points.tofile(source)
    open_cloud(window, app, source)
    window.class_definitions["segmentation"] = [
        ClassDefinition(0, "Unlabeled", "#808080"),
        ClassDefinition(10, "Car", "#6496F5"),
    ]
    window.class_definitions["detection"] = window.class_definitions[
        "segmentation"
    ][1:]
    window._saved_classes = {
        task: list(classes)
        for task, classes in window.class_definitions.items()
    }
    window._refresh()
    assert window.detection.enabled
    assert not window.detection.orthographic.isHidden()
    for view in window.detection.views:
        view._error = None
        view.resize(400, 300)
    window._confirm = Mock(return_value=True)
    return window


def mouse(
    view,
    method,
    point,
    button=QtCore.Qt.MouseButton.LeftButton,
    modifiers=QtCore.Qt.KeyboardModifier.NoModifier,
):
    event_type = {
        "_mouse_press": QtCore.QEvent.Type.MouseButtonPress,
        "_mouse_move": QtCore.QEvent.Type.MouseMove,
        "_mouse_release": QtCore.QEvent.Type.MouseButtonRelease,
        "_mouse_double_click": QtCore.QEvent.Type.MouseButtonDblClick,
    }[method]
    event = QtGui.QMouseEvent(
        event_type,
        QtCore.QPointF(*point),
        QtCore.QPointF(*point),
        button,
        button,
        modifiers,
    )
    getattr(view, method)(event)


def test_three_views_create_preview_cancel_commit_and_undo(detection):
    window = detection
    workspace = window.detection
    top = workspace.views[1]
    workspace.start_creation()
    start, end = top.project([[-2.5, -1.5, 0], [2.5, 1.5, 0]])
    mouse(top, "_mouse_press", start)
    mouse(top, "_mouse_move", end)
    assert window.document.cuboids == ()
    mouse(top, "_mouse_release", end)
    box = workspace.selected
    assert box is not None
    np.testing.assert_allclose(box.center, (0, 0, 0), atol=1e-6)
    np.testing.assert_allclose(box.size, (5, 3, 1), atol=1e-6)
    center = top.project([box.center])[0]
    mouse(top, "_mouse_press", center)
    mouse(top, "_mouse_move", center + (15, -5))
    assert window.document.cuboids == (box,)
    assert all(
        view.selected_cuboid.center != box.center for view in workspace.views
    )
    assert not window._autosave()
    workspace.cancel()
    assert all(view.selected_cuboid == box for view in workspace.views)
    center = top.project([box.center])[0]
    mouse(top, "_mouse_press", center)
    mouse(top, "_mouse_release", center + (15, -5))
    assert workspace.selected.center != box.center
    window.undo()
    assert window.document.cuboids == (box,)
    window.undo()
    assert window.document.cuboids == ()
    window.redo()
    assert window.document.cuboids == (box,)


@pytest.mark.parametrize(
    "button",
    [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton],
)
@pytest.mark.parametrize("view_index,axis", [(1, 2), (2, 1), (3, 0)])
def test_each_orthographic_view_rotates_about_its_normal(
    detection, view_index, axis, button
):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    view = workspace.views[view_index]
    before = workspace.selected
    _, handle = view._handles(before)
    center = view.project([before.center])[0]
    vector = handle - center
    destination = center + np.array([-vector[1], vector[0]])
    mouse(view, "_mouse_press", handle, button)
    mouse(view, "_mouse_release", destination, button)
    after = workspace.selected
    assert after.rotation != before.rotation
    np.testing.assert_allclose(
        after.matrix[:, axis], before.matrix[:, axis], atol=1e-6
    )
    np.testing.assert_allclose(after.size, before.size)


@pytest.mark.parametrize("on_box", [False, True])
@pytest.mark.parametrize(
    "button",
    [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton],
)
def test_overview_drag_changes_camera_without_editing_or_deselecting_box(
    detection, on_box, button
):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    box = workspace.selected
    assert detection._autosave()
    view = workspace.views[0]
    center = view._center.copy()
    angles = view._yaw, view._pitch
    history = len(detection.document._undo)
    side_matrices = [side._matrix().copy() for side in workspace.views[1:]]
    start = view.project([box.center])[0] if on_box else np.array((5, 5))
    mouse(view, "_mouse_press", start, button)
    mouse(view, "_mouse_move", start + (30, 20), button)
    mouse(view, "_mouse_release", start + (30, 20), button)
    if button == QtCore.Qt.MouseButton.LeftButton:
        assert (view._yaw, view._pitch) != angles
        np.testing.assert_array_equal(view._center, center)
    else:
        assert (view._yaw, view._pitch) == angles
        assert not np.array_equal(view._center, center)
    assert workspace.selected == box
    assert detection.document.cuboids == (box,)
    assert not detection.document.dirty
    assert len(detection.document._undo) == history
    for side, matrix in zip(workspace.views[1:], side_matrices):
        np.testing.assert_array_equal(side._matrix(), matrix)


def test_overview_click_selects_on_release_but_drag_does_not_select(detection):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    box = workspace.selected
    workspace.select(None)
    view = workspace.views[0]
    center = view.project([box.center])[0]
    mouse(view, "_mouse_press", center)
    assert workspace.selected is None
    mouse(view, "_mouse_move", center + (1, 1))
    matrix = view._matrix().copy()
    mouse(view, "_mouse_release", center + (1, 1))
    assert workspace.selected == box
    np.testing.assert_array_equal(view._matrix(), matrix)
    mouse(view, "_mouse_press", (5, 5))
    mouse(view, "_mouse_release", (5, 5))
    assert workspace.selected is None
    mouse(view, "_mouse_press", center)
    mouse(view, "_mouse_move", center + (40, 20))
    mouse(view, "_mouse_release", center)
    assert workspace.selected is None
    assert detection.document.cuboids == (box,)
    assert view._navigation_button is None


@pytest.mark.parametrize("view_index", [1, 2, 3])
@pytest.mark.parametrize(
    "button",
    [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton],
)
def test_side_view_body_drag_uses_initiating_button_and_one_undo(
    detection, view_index, button
):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    box = workspace.selected
    view = workspace.views[view_index]
    start = view.project([box.center])[0]
    end = start + (15, -5)
    expected = (
        np.array(box.center) + view.unproject(end) - view.unproject(start)
    )
    mouse(view, "_mouse_press", start, button)
    mouse(view, "_mouse_move", end, button)
    other_button = (
        QtCore.Qt.MouseButton.RightButton
        if button == QtCore.Qt.MouseButton.LeftButton
        else QtCore.Qt.MouseButton.LeftButton
    )
    mouse(view, "_mouse_release", end, other_button)
    assert detection.document.cuboids == (box,)
    assert view.selection_active
    np.testing.assert_allclose(view.selected_cuboid.center, expected)
    mouse(view, "_mouse_release", end, button)
    assert not view.selection_active
    np.testing.assert_allclose(workspace.selected.center, expected)
    assert workspace.selected.size == box.size
    assert workspace.selected.rotation == box.rotation
    detection.undo()
    assert detection.document.cuboids == (box,)


@pytest.mark.parametrize(
    "view_index,axes", [(1, (0, 1)), (2, (0, 2)), (3, (1, 2))]
)
@pytest.mark.parametrize(
    "button",
    [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton],
)
def test_side_view_handle_resizes_instead_of_translating(
    detection, view_index, axes, button
):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    box = workspace.selected
    view = workspace.views[view_index]
    handles, _ = view._handles(box)
    signs, start = handles[2]
    mouse(view, "_mouse_press", start, button)
    mouse(view, "_mouse_release", start + (20, 10), button)
    after = workspace.selected
    assert after.size != box.size
    opposite = np.zeros(3)
    opposite[list(axes)] = -np.array(signs) / 2
    np.testing.assert_allclose(
        np.array(box.center) + box.matrix @ (opposite * box.size),
        np.array(after.center) + after.matrix @ (opposite * after.size),
    )
    assert after.rotation == box.rotation


def test_right_drag_can_be_cancelled_and_cannot_edit_locked_box(detection):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    box = workspace.selected
    view = workspace.views[1]
    center = view.project([box.center])[0]
    mouse(view, "_mouse_press", center, QtCore.Qt.MouseButton.RightButton)
    mouse(
        view,
        "_mouse_move",
        center + (20, 10),
        QtCore.Qt.MouseButton.RightButton,
    )
    QtTest.QTest.keyClick(view, QtCore.Qt.Key.Key_Escape)
    mouse(
        view,
        "_mouse_release",
        center + (20, 10),
        QtCore.Qt.MouseButton.RightButton,
    )
    assert detection.document.cuboids == (box,)
    assert all(member.selected_cuboid == box for member in workspace.views)
    workspace.locked.setChecked(True)
    box = workspace.selected
    mouse(view, "_mouse_press", center, QtCore.Qt.MouseButton.RightButton)
    mouse(
        view,
        "_mouse_release",
        center + (20, 10),
        QtCore.Qt.MouseButton.RightButton,
    )
    assert detection.document.cuboids == (box,)


@pytest.mark.parametrize("view_index", [0, 1])
@pytest.mark.parametrize(
    "button,modifiers",
    [
        (
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        ),
        (
            QtCore.Qt.MouseButton.MiddleButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        ),
    ],
)
def test_explicit_pan_does_not_change_box(
    detection, view_index, button, modifiers
):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    box = workspace.selected
    view = workspace.views[view_index]
    center = view._center.copy()
    angles = view._yaw, view._pitch
    start = view.project([box.center])[0]
    mouse(view, "_mouse_press", start, button, modifiers)
    mouse(view, "_mouse_release", start + (30, 20), button, modifiers)
    assert not np.array_equal(view._center, center)
    assert (view._yaw, view._pitch) == angles
    assert detection.document.cuboids == (box,)


@pytest.mark.parametrize(
    "button",
    [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton],
)
def test_shared_camera_buttons(detection, button):
    view = detection.viewport
    center = view._center.copy()
    angles = view._yaw, view._pitch
    mouse(view, "_mouse_press", (200, 150), button)
    mouse(view, "_mouse_move", (230, 170), button)
    mouse(view, "_mouse_release", (230, 170), button)
    if button == QtCore.Qt.MouseButton.LeftButton:
        np.testing.assert_array_equal(view._center, center)
        assert (view._yaw, view._pitch) != angles
    else:
        assert not np.array_equal(view._center, center)
        assert (view._yaw, view._pitch) == angles


def test_cuboid_autosave_does_not_write_point_labels(detection):
    window = detection
    source = window.document.frame.path
    original = source.read_bytes()
    window.detection.create((0, 0, 0), (4, 2, 1))
    assert window._autosave()
    assert not window.document.dirty
    assert not source.with_suffix(".label").exists()
    assert source.read_bytes() == original
    assert (
        load_cuboids(source.with_suffix(".cuboids.json"), source)
        == window.document.cuboids
    )
    window.detection.delete()
    assert window._autosave()
    assert load_frame(source).cuboids == ()
    window.undo()
    assert window.document.cuboids_dirty


def test_copy_across_frames_and_output_directory(detection, app, tmp_path):
    window = detection
    first = window.document.frame.path
    second = tmp_path / "next.bin"
    second.write_bytes(first.read_bytes())
    window.detection.create((0, 0, 0), (4, 2, 1))
    box = window.detection.selected
    window.detection.copy()
    output = tmp_path / "output"
    with patch.object(
        QtWidgets.QFileDialog, "getExistingDirectory", return_value=str(output)
    ):
        window._change_output_directory()
    assert load_cuboids(output / "scan.cuboids.json", first) == (box,)
    assert not first.with_suffix(".cuboids.json").exists()
    window.open_paths([first, second])
    wait_load(window, app)
    window.navigate(1)
    wait_load(window, app)
    assert not window.document.cuboids
    window.detection.paste()
    assert window.document.cuboids[0].center == box.center
    assert window._autosave()
    assert (output / "next.cuboids.json").exists()
    window.navigate(-1)
    wait_load(window, app)
    assert window.document.cuboids == (box,)


def test_save_failure_retains_cuboid_dirty_state(detection):
    window = detection
    window.detection.create((0, 0, 0), (4, 2, 1))
    with patch(
        "anylabeling.views.labeling.widgets.pointcloud_dialog.save_cuboids",
        side_effect=OSError("disk full"),
    ):
        assert not window._autosave()
    assert window.document.cuboids_dirty
    assert "disk full" in window._errors[-1]


def test_escape_exits_creation_and_segmentation_remains_available(detection):
    window = detection
    window.detection.start_creation()
    assert window.viewport.creating
    QtTest.QTest.keyClick(window.detection.views[1], QtCore.Qt.Key.Key_Escape)
    assert not any(view.creating for view in window.detection.views)
    window.sidebar_tabs.setCurrentIndex(1)
    window._select_tool("brush")
    assert window.viewport._tool == "brush"
    assert not window.viewport.detection_enabled
    assert window.detection.orthographic.isHidden()
    assert window.tool_actions["polygon"].isEnabled()
    assert window.through_action.isEnabled()


def test_point_labels_and_cuboids_share_the_workspace(detection):
    window = detection
    window.detection.create((0, 0, 0), (4, 2, 1))
    box = window.detection.selected
    window.sidebar_tabs.setCurrentIndex(1)
    select_class(window, 10)
    window._select_tool("brush")
    window._apply_selection(np.array([0]))
    assert window.document.semantic_view[0] == 10
    assert window.document.cuboids == (box,)
    assert window.viewport._tool == "brush"
    assert not window.viewport.detection_enabled
    assert not window.viewport.cuboids


def test_three_view_default_height_and_restore(detection, app):
    workspace = detection.detection
    detection.resize(1400, 1000)
    detection.show()
    app.processEvents()
    sizes = workspace.sizes()
    assert sizes[1] == max(
        view.minimumHeight() for view in workspace.views[1:]
    )
    detection.resize(1400, 800)
    app.processEvents()
    assert workspace.sizes()[1] == sizes[1]
    workspace.setSizes([500, 300])
    sizes = workspace.sizes()
    workspace.toggle_views(show=False)
    workspace.toggle_views(show=True)
    assert workspace.sizes() == sizes


@pytest.mark.parametrize("index", range(4))
def test_loaded_cloud_zoom_is_limited_before_first_wheel(detection, index):
    view = detection.detection.views[index]
    view.set_cloud(
        np.array(
            [[-1000, -1000, -1000, 0], [1000, 1000, 1000, 0]], dtype=np.float32
        )
    )
    limit = 100 * np.tan(np.radians(25)) if index == 0 else 50
    assert view._scale == pytest.approx(limit)
    event = Mock()
    event.angleDelta.return_value = QtCore.QPoint(0, -120)
    event.modifiers.return_value = QtCore.Qt.KeyboardModifier.NoModifier
    view._wheel(event)
    assert view._scale == pytest.approx(limit)
    view._scale = 1
    view.reset_view()
    assert view._scale == pytest.approx(limit)
    view._scale = 1
    view.focus_indices([0, 1])
    assert view._scale == pytest.approx(limit)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("size", [0.01, 1000])
def test_cuboid_focus_obeys_zoom_limits(detection, box, index, size):
    view = detection.detection.views[index]
    view.align_cuboid(replace(box, size=(size, size, size)), fit=True)
    limit = view._scale_limits[0 if size < 1 else 1]
    assert view._scale == pytest.approx(limit)


@pytest.mark.parametrize("index", range(4))
def test_view_wheel_zoom_limits(detection, index):
    view = detection.detection.views[index]
    limit = 100 * np.tan(np.radians(25)) if index == 0 else 50
    event = Mock()
    event.angleDelta.return_value = QtCore.QPoint(0, -120)
    event.modifiers.return_value = QtCore.Qt.KeyboardModifier.NoModifier
    view._scale = 10
    center = view._center.copy()
    for _ in range(100):
        view._wheel(event)
    assert view._scale == pytest.approx(limit)
    view._wheel(event)
    assert view._scale == pytest.approx(limit)
    np.testing.assert_array_equal(view._center, center)
    event.angleDelta.return_value = QtCore.QPoint(0, 120)
    view._wheel(event)
    assert view._scale < limit
    for _ in range(100):
        view._wheel(event)
    minimum = 0.3 * np.tan(np.radians(25)) if index == 0 else 0.025
    assert view._scale == pytest.approx(minimum)


def test_three_view_toggle_stays_at_toolbar_right(detection, app):
    workspace = detection.detection
    button = workspace.views_button
    detection.resize(900, 650)
    detection.show()
    app.processEvents()
    assert button.isVisible()
    assert not workspace.orthographic.isHidden()
    assert (
        button.mapToGlobal(QtCore.QPoint()).y()
        < workspace.mapToGlobal(QtCore.QPoint()).y()
    )
    toolbar = detection.view_tool_scroll
    assert (
        button.mapToGlobal(QtCore.QPoint()).x()
        >= toolbar.mapToGlobal(QtCore.QPoint(toolbar.width(), 0)).x()
    )
    button.click()
    assert workspace.orthographic.isHidden()
    assert detection.viewport.detection_enabled
    assert not detection.tool_actions["brush"].isEnabled()
    assert button.isVisible()
    button.click()
    assert not workspace.orthographic.isHidden()
    detection.viewport.pointer_moved.emit(
        detection.viewport.mapToGlobal(QtCore.QPoint(10, 10))
    )
    assert button.isVisible()


def test_drawing_a_cuboid_expands_views_and_uses_browse(detection):
    window = detection
    window._select_tool("brush")
    window.detection.toggle_views(show=False)
    window.detection.start_creation()
    assert not window.detection.orthographic.isHidden()
    assert window.viewport._tool == "browse"
    assert all(view.creating for view in window.detection.views)


def test_destination_cuboids_override_source_even_without_destination_labels(
    box, tmp_path, app
):
    from anylabeling.views.labeling.widgets.pointcloud_dialog import (
        FrameLoader,
    )

    source = tmp_path / "scan.bin"
    np.zeros((4, 4), dtype="<f4").tofile(source)
    source.with_suffix(".label").write_bytes(bytes(16))
    source.with_suffix(".cuboids.json").write_text("invalid json")
    output = tmp_path / "output" / "scan.label"
    save_cuboids(output.with_suffix(".cuboids.json"), [box], source)
    loader = FrameLoader(
        source, source.with_suffix(".label"), None, None, output
    )
    loaded, failed = [], []
    loader.loaded.connect(loaded.append)
    loader.failed.connect(failed.append)
    loader.run()
    assert not failed
    frame = loaded[0][0]
    assert frame.cuboids == (box,)
    assert frame.cuboid_exists and not frame.label_exists
    assert frame.cuboid_path == output.with_suffix(".cuboids.json")
    save_cuboids(frame.cuboid_path, [], source)
    assert load_frame(source, output).cuboids == ()


def test_source_cuboids_are_retained_when_changing_to_empty_output(
    box, tmp_path
):
    source = tmp_path / "scan.bin"
    np.zeros((4, 4), dtype="<f4").tofile(source)
    save_cuboids(source.with_suffix(".cuboids.json"), [box], source)
    output = tmp_path / "output" / "scan.label"
    frame = load_frame(source, output)
    assert frame.cuboids == (box,)
    assert not frame.cuboid_exists
    assert frame.cuboid_path == output.with_suffix(".cuboids.json")


def test_invalid_cuboid_load_retains_open_document(detection, tmp_path, app):
    window = detection
    window.detection.create((0, 0, 0), (4, 2, 1))
    original = window.document
    bad = tmp_path / "bad.bin"
    bad.write_bytes(original.frame.path.read_bytes())
    bad.with_suffix(".cuboids.json").write_text("invalid json")
    window.open_paths([bad])
    wait_load(window, app)
    assert window.document is original
    assert window._errors
    assert len(window.document.cuboids) == 1


def test_visibility_lock_and_precise_numeric_edit(detection):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    workspace.fields["rotation"][2].setValue(45)
    workspace._edit_field("rotation", 2)
    assert workspace.selected.rotation[2] == pytest.approx(np.pi / 4)
    workspace.locked.setChecked(True)
    before = workspace.selected
    assert before.locked
    workspace.delete()
    assert workspace.selected == before
    assert not workspace.fields["size"][0].isEnabled()
    assert not workspace.class_combo.isEnabled()
    workspace.objects.item(0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert not any(view.cuboids for view in workspace.views)
    assert detection.document.cuboids == (before,)
    workspace.objects.item(0).setCheckState(QtCore.Qt.CheckState.Checked)
    workspace.locked.setChecked(False)
    assert not workspace.selected.locked
    assert workspace.fields["size"][0].isEnabled()


def test_resizing_orthographic_view_keeps_all_box_corners_visible(detection):
    workspace = detection.detection
    workspace.create((0, 0, 0), (12, 2, 1))
    for view in workspace.views[1:]:
        view.resize(240, 500)
        view.resizeEvent(
            QtGui.QResizeEvent(QtCore.QSize(240, 500), QtCore.QSize(400, 300))
        )
        points = view.project(workspace.selected.corners())
        assert (points.min(axis=0) > (0, 0)).all()
        assert (points.max(axis=0) < (view.width(), view.height())).all()


def test_newly_appeared_cuboid_output_is_not_overwritten_without_confirmation(
    detection,
):
    window = detection
    window.detection.create((0, 0, 0), (4, 2, 1))
    target = window.document.frame.cuboid_path
    target.write_text("another result")
    window._confirm.return_value = False
    assert not window._autosave()
    assert target.read_text() == "another result"
    assert window.document.cuboids_dirty


def test_segmentation_filters_do_not_hide_detection_points(detection):
    window = detection
    window.document.assign_semantic([0, 1], 10)
    window._refresh()
    for index in range(window.class_list.count()):
        item = window.class_list.item(index)
        if item.data(QtCore.Qt.ItemDataRole.UserRole) == 10:
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
    for view in window.detection.views:
        assert view._visible.all()
    window.sidebar_tabs.setCurrentIndex(1)
    assert not window.viewport._visible[:2].any()
    assert window.viewport._visible[2:].all()


def test_create_box_from_instance_preserves_labels(detection):
    window = detection
    window.document.assign_semantic([0, 1, 2, 3], 10)
    key = window.document.create_instance([0, 1, 2, 3], 10)
    labels = window.document.labels.copy()
    window._refresh(selected_instance=key)
    window.detection.from_instance()
    assert window.detection.selected.class_id == 10
    assert window.detection.selected.contains(
        window.document.frame.points[:4]
    ).all()
    np.testing.assert_array_equal(window.document.labels, labels)


def test_click_selection_fills_only_active_box_and_blank_clears(
    detection, app
):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    first = workspace.selected
    workspace.create((8, 0, 0), (2, 2, 2))
    workspace.select(None)
    main = workspace.views[0]
    assert all(not view.cuboids for view in workspace.views[1:])
    point = main.project([first.center])[0]
    mouse(main, "_mouse_press", point)
    mouse(main, "_mouse_release", point)
    assert workspace.selected_id == first.id
    assert len(main.cuboids) == 2
    assert main._cuboid_mesh[1] == 36
    for view in workspace.views[1:]:
        assert view.cuboids == (first,)
        assert view._cuboid_mesh[1] == 36
        assert (
            view._focus_animation.state()
            == QtCore.QAbstractAnimation.State.Running
        )
    QtTest.QTest.qWait(100)
    before = workspace.views[1]._scale
    mouse(main, "_mouse_press", (0, 0))
    mouse(main, "_mouse_release", (0, 0))
    assert workspace.selected_id is None
    assert main._cuboid_mesh[1] == 0
    assert all(not view.cuboids for view in workspace.views[1:])
    assert all(not len(view._cuboid_mesh[0]) for view in workspace.views[1:])
    QtTest.QTest.qWait(300)
    assert workspace.views[1]._scale == before


def test_side_view_focus_animates_and_manual_navigation_interrupts(detection):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    box = workspace.selected
    view = workspace.views[1]
    view._center = np.array([20.0, 20.0, 20.0])
    view._scale = 20
    start = view._center.copy()
    view.align_cuboid(box, fit=True, animate=True)
    np.testing.assert_array_equal(view._center, start)
    view._focus_animation.setCurrentTime(125)
    assert (
        0
        < np.linalg.norm(view._center - box.center)
        < np.linalg.norm(start - box.center)
    )
    assert view._focus_target[1] < view._scale < 20
    mouse(view, "_mouse_press", (10, 10), QtCore.Qt.MouseButton.MiddleButton)
    assert (
        view._focus_animation.state()
        == QtCore.QAbstractAnimation.State.Stopped
    )
    mouse(view, "_mouse_release", (10, 10), QtCore.Qt.MouseButton.MiddleButton)
    view.align_cuboid(box, fit=True, animate=True)
    view._focus_animation.setCurrentTime(250)
    np.testing.assert_array_equal(view._center, box.center)
    assert view._scale == view._focus_target[1]


def test_switching_targets_restarts_focus_from_current_camera(detection):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    first = workspace.selected
    workspace.create((8, 0, 0), (2, 2, 2))
    second = workspace.selected
    workspace.select(first.id)
    for view in workspace.views[1:]:
        view._focus_animation.setCurrentTime(100)
    cameras = [
        (view._center.copy(), view._scale) for view in workspace.views[1:]
    ]
    workspace.select(second.id)
    for view, (center, scale) in zip(workspace.views[1:], cameras):
        np.testing.assert_array_equal(view._center, center)
        assert view._scale == scale
        assert view.cuboids == (second,)
        view._focus_animation.setCurrentTime(250)
        np.testing.assert_array_equal(view._center, second.center)


@pytest.mark.parametrize("index", [1, 2, 3])
@pytest.mark.parametrize("animating", [False, True])
@pytest.mark.parametrize(
    "button",
    [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton],
)
def test_side_view_blank_click_preserves_selection_and_focus(
    detection, index, animating, button
):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    box = workspace.selected
    workspace.select(None)
    main = workspace.views[0]
    center = main.project([box.center])[0]
    mouse(main, "_mouse_press", center)
    mouse(main, "_mouse_release", center)
    for side in workspace.views[1:]:
        side._focus_animation.setCurrentTime(125 if animating else 250)
    cameras = [
        (side._center.copy(), side._scale, side._focus_animation.state())
        for side in workspace.views[1:]
    ]
    history = len(detection.document._undo)
    view = workspace.views[index]
    selected = Mock()
    view.cuboid_selected.connect(selected)
    assert view._hit_box((0, 0)) is None
    mouse(view, "_mouse_press", (0, 0), button)
    mouse(view, "_mouse_release", (0, 0), button)
    selected.assert_not_called()
    assert workspace.selected_id == box.id
    assert len(detection.document._undo) == history
    assert all(
        side.selected_cuboid == box and side._cuboid_mesh[1] == 36
        for side in workspace.views
    )
    for side, (center, scale, state) in zip(workspace.views[1:], cameras):
        np.testing.assert_array_equal(side._center, center)
        assert side._scale == scale
        assert side._focus_animation.state() == state


def test_main_blank_double_click_smoothly_restores_initial_camera(detection):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    view = workspace.views[0]
    view.reset_view()
    initial = (view._center.copy(), view._scale, view._yaw, view._pitch)
    view._center = np.array([20.0, 20.0, 20.0])
    view._scale = 8
    view._yaw, view._pitch = 120, -10
    start = view._center.copy()
    history = len(detection.document._undo)
    assert view._hit_box((0, 0)) is None
    mouse(view, "_mouse_double_click", (0, 0))
    mouse(view, "_mouse_release", (0, 0))
    assert workspace.selected_id is None
    assert (
        view._focus_animation.state()
        == QtCore.QAbstractAnimation.State.Running
    )
    np.testing.assert_array_equal(view._center, start)
    view._focus_animation.setCurrentTime(125)
    assert (
        0
        < np.linalg.norm(view._center - initial[0])
        < np.linalg.norm(start - initial[0])
    )
    view._focus_animation.setCurrentTime(view._focus_animation.duration())
    np.testing.assert_allclose(view._center, initial[0])
    assert (view._scale, view._yaw, view._pitch) == initial[1:]
    assert len(detection.document._undo) == history


def test_main_object_double_click_smoothly_focuses_box(detection):
    workspace = detection.detection
    workspace.create((1, 0, 0), (1, 1, 1))
    box = workspace.selected
    workspace.select(None)
    view = workspace.views[0]
    view.reset_view()
    angles = view._yaw, view._pitch
    center = view.project([box.center])[0]
    mouse(view, "_mouse_press", center)
    mouse(view, "_mouse_release", center)
    mouse(view, "_mouse_double_click", center)
    mouse(view, "_mouse_release", center)
    assert workspace.selected_id == box.id
    assert (
        view._focus_animation.state()
        == QtCore.QAbstractAnimation.State.Running
    )
    view._focus_animation.setCurrentTime(view._focus_animation.duration())
    np.testing.assert_allclose(view._center, box.center)
    assert (view._yaw, view._pitch) == angles
    assert view._cuboid_mesh[1] == 36


@pytest.mark.parametrize("frame_ms", [8, 16, 33])
@pytest.mark.parametrize("size", [0.5, 4])
def test_main_focus_enlarges_distant_box_without_a_visual_jump(
    detection, frame_ms, size
):
    workspace = detection.detection
    workspace.create((2, 1, 0), (size, size, size))
    box = workspace.selected
    view = workspace.views[0]
    view._set_scale(view._scale_limits[1])
    history = len(detection.document._undo)
    mouse(view, "_mouse_double_click", view.project([box.center])[0])
    animation = view._focus_animation
    heights = []
    for elapsed in range(0, animation.duration(), frame_ms):
        animation.setCurrentTime(elapsed)
        heights.append(np.ptp(view.project(box.corners())[:, 1]))
    animation.setCurrentTime(animation.duration())
    heights.append(np.ptp(view.project(box.corners())[:, 1]))
    changes = np.diff(heights)
    assert np.all(changes >= 0)
    assert changes.max() / heights[-1] < 0.03 * frame_ms / 16
    assert changes[-1] < 0.01
    np.testing.assert_allclose(view._center, box.center)
    assert animation.state() == QtCore.QAbstractAnimation.State.Stopped
    assert detection.document.cuboids == (box,)
    assert len(detection.document._undo) == history


def test_main_focus_retargets_from_current_camera_and_drag_interrupts(
    detection,
):
    workspace = detection.detection
    workspace.create((0, 0, 0), (2, 2, 2))
    first = workspace.selected
    workspace.create((6, 0, 0), (2, 2, 2))
    second = workspace.selected
    view = workspace.views[0]
    view._set_scale(view._scale_limits[1])
    mouse(view, "_mouse_double_click", view.project([first.center])[0])
    view._focus_animation.setCurrentTime(400)
    matrix = view._matrix().copy()
    mouse(view, "_mouse_double_click", view.project([second.center])[0])
    np.testing.assert_array_equal(view._matrix(), matrix)
    assert workspace.selected_id == second.id
    view._focus_animation.setCurrentTime(400)
    assert not np.array_equal(view._matrix(), matrix)
    center, scale = view._center.copy(), view._scale
    mouse(view, "_mouse_press", (10, 10), QtCore.Qt.MouseButton.MiddleButton)
    assert (
        view._focus_animation.state()
        == QtCore.QAbstractAnimation.State.Stopped
    )
    np.testing.assert_array_equal(view._center, center)
    assert view._scale == scale
    mouse(view, "_mouse_move", (40, 30), QtCore.Qt.MouseButton.MiddleButton)
    mouse(view, "_mouse_release", (40, 30), QtCore.Qt.MouseButton.MiddleButton)
    assert not np.array_equal(view._center, center)
    assert detection.document.cuboids == (first, second)


@pytest.mark.parametrize("index", [1, 2, 3])
def test_side_blank_double_click_does_not_reset_camera(detection, index):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    box = workspace.selected
    view = workspace.views[index]
    view.align_cuboid(box, fit=True)
    center, scale = view._center.copy(), view._scale
    mouse(view, "_mouse_double_click", (0, 0))
    mouse(view, "_mouse_release", (0, 0))
    assert workspace.selected_id == box.id
    np.testing.assert_array_equal(view._center, center)
    assert view._scale == scale


@pytest.mark.parametrize("index", [1, 2, 3])
@pytest.mark.parametrize(
    "button",
    [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton],
)
@pytest.mark.parametrize("finish", ["commit", "cancel"])
def test_rotation_preview_keeps_side_boxes_fixed_and_rotates_cloud(
    detection, index, button, finish
):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    workspace.commit(replace(workspace.selected, rotation=(0.2, -0.3, 0.4)))
    before = workspace.selected
    sides = workspace.views[1:]
    for side in sides:
        side.align_cuboid(before, fit=True)
    corners = [side.project(before.corners()) for side in sides]
    handles = [side._handles(before)[1] for side in sides]
    centers = [side._center.copy() for side in sides]
    clouds = [side.project(side._points[:, :3]) for side in sides]
    scales = [side._scale for side in sides]
    main = workspace.views[0]
    main_matrix = main._matrix().copy()
    main_box = main.project(before.corners())
    points = detection.document.frame.points.copy()
    history = len(detection.document._undo)
    view = workspace.views[index]
    handle = view._handles(before)[1]
    center = view.project([before.center])[0]
    vector = handle - center
    mouse(view, "_mouse_press", handle, button)
    for angle in (0.3, 0.7):
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        destination = center + rotation @ vector
        mouse(view, "_mouse_move", destination, button)
        preview = view.selected_cuboid
        assert preview.rotation != before.rotation
        assert view._gesture is not None
        assert detection.document.cuboids == (before,)
        for side, expected, expected_handle, scale in zip(
            sides, corners, handles, scales
        ):
            np.testing.assert_allclose(
                side.project(preview.corners()), expected, atol=1e-4
            )
            np.testing.assert_allclose(
                side._handles(preview)[1], expected_handle, atol=1e-4
            )
            assert side._scale == scale
            forward = side._basis()[2]
            depth = (side._points[:, :3] - preview.center) @ forward
            half = np.abs(preview.matrix.T @ forward) @ (
                np.array(preview.size) / 2
            )
            np.testing.assert_array_equal(
                side._visible,
                (
                    main._visible
                    if side.orthographic_view == "top"
                    else main._visible & (np.abs(depth) <= half + 0.1)
                ),
            )
        assert not np.allclose(
            view.project(view._points[:, :3]), clouds[index - 1]
        )
        assert not np.allclose(main.project(preview.corners()), main_box)
        np.testing.assert_array_equal(main._matrix(), main_matrix)
        np.testing.assert_array_equal(detection.document.frame.points, points)
    if finish == "cancel":
        workspace.cancel()
        assert len(detection.document._undo) == history
    else:
        mouse(view, "_mouse_release", destination, button)
        np.testing.assert_allclose(
            workspace.selected.rotation, preview.rotation, atol=1e-6
        )
        assert len(detection.document._undo) == history + 1
        for side, expected in zip(sides, corners):
            np.testing.assert_allclose(
                side.project(workspace.selected.corners()), expected, atol=1e-4
            )
        detection.undo()
    assert workspace.selected == before
    for side, expected, cloud in zip(sides, centers, clouds):
        np.testing.assert_allclose(side._center, expected, atol=1e-6)
        np.testing.assert_allclose(side._orientation, before.matrix, atol=1e-6)
        np.testing.assert_allclose(
            side.project(side._points[:, :3]), cloud, atol=1e-4
        )


@pytest.mark.parametrize("index", [1, 2, 3])
@pytest.mark.parametrize(
    "button",
    [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton],
)
@pytest.mark.parametrize("finish", ["commit", "cancel"])
def test_translation_keeps_other_side_boxes_fixed(
    detection, index, button, finish
):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    workspace.commit(replace(workspace.selected, rotation=(0.2, -0.3, 0.4)))
    before = workspace.selected
    sides = workspace.views[1:]
    for side in sides:
        side.align_cuboid(before, fit=True)
    corners = [side.project(before.corners()) for side in sides]
    handles = [side._handles(before)[1] for side in sides]
    centers = [side._center.copy() for side in sides]
    clouds = [side.project(side._points[:, :3]) for side in sides]
    scales = [side._scale for side in sides]
    main = workspace.views[0]
    main_matrix = main._matrix().copy()
    original_points = detection.document.frame.points.copy()
    history = len(detection.document._undo)
    view = workspace.views[index]
    start = view.project([before.center])[0]
    mouse(view, "_mouse_press", start, button)
    for offset in (np.array([12, 9]), np.array([30, 24])):
        destination = start + offset
        mouse(view, "_mouse_move", destination, button)
        preview = view.selected_cuboid
        assert preview.center != before.center
        assert view._gesture is not None
        assert detection.document.cuboids == (before,)
        for side, expected, handle, center, cloud, scale in zip(
            sides, corners, handles, centers, clouds, scales
        ):
            if side is view:
                np.testing.assert_allclose(
                    side.project(preview.corners()),
                    expected + offset,
                    atol=1e-4,
                )
                np.testing.assert_array_equal(side._center, center)
                np.testing.assert_array_equal(
                    side.project(side._points[:, :3]), cloud
                )
            else:
                np.testing.assert_allclose(
                    side.project(preview.corners()), expected, atol=1e-4
                )
                np.testing.assert_allclose(
                    side._handles(preview)[1], handle, atol=1e-4
                )
                assert not np.allclose(
                    side.project(side._points[:, :3]), cloud
                )
            assert side._scale == scale
            forward = side._basis()[2]
            depth = (side._points[:, :3] - preview.center) @ forward
            half = np.abs(preview.matrix.T @ forward) @ (
                np.array(preview.size) / 2
            )
            np.testing.assert_array_equal(
                side._visible,
                (
                    main._visible
                    if side.orthographic_view == "top"
                    else main._visible & (np.abs(depth) <= half + 0.1)
                ),
            )
        np.testing.assert_array_equal(main._matrix(), main_matrix)
        np.testing.assert_array_equal(
            detection.document.frame.points, original_points
        )
    if finish == "cancel":
        workspace.cancel()
        assert workspace.selected == before
        assert len(detection.document._undo) == history
        for side, center in zip(sides, centers):
            np.testing.assert_allclose(side._center, center, atol=1e-6)
    else:
        mouse(view, "_mouse_release", destination, button)
        np.testing.assert_allclose(
            workspace.selected.center, preview.center, atol=1e-6
        )
        assert len(detection.document._undo) == history + 1
        for side, expected in zip(sides, corners):
            np.testing.assert_allclose(
                side.project(workspace.selected.corners()),
                expected,
                atol=1e-4,
            )
        detection.undo()
        assert workspace.selected == before
        detection.redo()
        np.testing.assert_allclose(
            workspace.selected.center, preview.center, atol=1e-6
        )


@pytest.mark.parametrize("index", [1, 2, 3])
def test_side_view_expand_button_hover_and_layout_restore(
    detection, app, index
):
    window = detection
    workspace = window.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    selected = workspace.selected
    history = len(window.document._undo)
    window.resize(1200, 900)
    window.show()
    window.activateWindow()
    assert QtTest.QTest.qWaitForWindowActive(window)
    workspace.setSizes([500, 220])
    workspace.orthographic.setSizes([220, 300, 250])
    app.processEvents()
    sizes = workspace.sizes(), workspace.orthographic.sizes()
    view = workspace.views[index]
    button = view.expand_button
    toolbar = window.view_tool_scroll.viewport()
    QtTest.QTest.mouseMove(toolbar, QtCore.QPoint(2, 2))
    QtTest.QTest.qWait(50)
    assert button.isHidden()
    assert workspace.views[0].expand_button is None
    QtTest.QTest.mouseMove(view._gl, QtCore.QPoint(20, 20))
    QtTest.QTest.qWait(50)
    assert button.isVisible()
    assert button.x() + button.width() == view.width() - 12
    assert button.y() == 6
    QtTest.QTest.mouseMove(button, button.rect().center())
    QtTest.QTest.qWait(50)
    assert button.isVisible()
    assert not button.icon().isNull()
    QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
    app.processEvents()
    assert workspace._expanded_view is view
    assert view._expanded
    assert view.width() == workspace.width()
    assert view.height() == workspace.height()
    assert all(
        member.isHidden() for member in workspace.views if member is not view
    )
    QtTest.QTest.mouseMove(toolbar, QtCore.QPoint(2, 2))
    QtTest.QTest.qWait(50)
    assert button.isVisible()
    assert button.toolTip() == view.tr("Restore view")
    assert workspace.selected == selected
    assert len(window.document._undo) == history
    QtTest.QTest.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
    app.processEvents()
    assert workspace._expanded_view is None
    assert all(member.isVisible() for member in workspace.views)
    assert (workspace.sizes(), workspace.orthographic.sizes()) == sizes
    assert workspace.selected == selected
    assert len(window.document._undo) == history
    QtTest.QTest.mouseMove(toolbar, QtCore.QPoint(2, 2))
    QtTest.QTest.qWait(50)
    assert button.isHidden()


def test_expanded_view_can_edit_and_toolbar_collapse_restores_layout(
    detection, app
):
    window = detection
    workspace = window.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    box = workspace.selected
    window.resize(1200, 900)
    window.show()
    app.processEvents()
    sizes = workspace.sizes(), workspace.orthographic.sizes()
    view = workspace.views[1]
    workspace.toggle_expanded_view(view)
    app.processEvents()
    point = view.project([box.center])[0]
    mouse(view, "_mouse_press", point)
    mouse(view, "_mouse_release", point + (20, 10))
    assert workspace.selected.center != box.center
    assert workspace._expanded_view is view
    window.undo()
    assert workspace.selected == box
    workspace.views_button.click()
    app.processEvents()
    assert workspace._expanded_view is None
    assert workspace.orthographic.isHidden()
    assert window.viewport.isVisible()
    workspace.views_button.click()
    app.processEvents()
    assert all(member.isVisible() for member in workspace.views)
    assert (workspace.sizes(), workspace.orthographic.sizes()) == sizes


@pytest.mark.parametrize("index", [1, 2, 3])
@pytest.mark.parametrize("gesture", ["move", "resize"])
def test_edit_then_zoom_keeps_box_at_view_center(detection, index, gesture):
    workspace = detection.detection
    workspace.create((8, -5, 2), (4, 2, 1))
    workspace.commit(replace(workspace.selected, rotation=(0.2, -0.3, 0.4)))
    for view in workspace.views:
        view.align_cuboid(workspace.selected, fit=True)
    view = workspace.views[index]
    start = (
        view.project([workspace.selected.center])[0]
        if gesture == "move"
        else view._handles(workspace.selected)[0][4][1]
    )
    mouse(view, "_mouse_press", start)
    mouse(view, "_mouse_release", start + (35, 20))
    box = workspace.selected
    for side in workspace.views[1:]:
        center = np.array((side.width(), side.height())) / 2
        np.testing.assert_allclose(
            side.project([box.center])[0], center, atol=1e-4
        )
        for delta in (120, -120):
            before = side.project(box.corners())
            scale = side._scale
            wheel = Mock()
            wheel.angleDelta.return_value = QtCore.QPoint(0, delta)
            wheel.position.return_value = QtCore.QPointF(*center)
            side._wheel(wheel)
            np.testing.assert_allclose(
                side.project(box.corners()) - center,
                (before - center) * scale / side._scale,
                atol=1e-4,
            )
    detection.undo()
    for side in workspace.views[1:]:
        np.testing.assert_allclose(side._center, workspace.selected.center)
    detection.redo()
    assert workspace.selected == box
    for side in workspace.views[1:]:
        np.testing.assert_allclose(side._center, box.center)


def test_top_keeps_cloud_depth_and_side_crops_do_not_hide_main(detection):
    workspace = detection.detection
    workspace.create((0, 0, 80), (4, 2, 1))
    workspace.commit(replace(workspace.selected, rotation=(0.2, -0.3, 0.4)))
    top, side, front = workspace.views[1:]
    main = workspace.views[0]
    assert main._visible.all()
    np.testing.assert_array_equal(top._visible, main._visible)
    assert top._visible_count == len(top._points)
    assert not front._visible_count
    for view in workspace.views[1:]:
        image = QtGui.QImage(
            view.size(), QtGui.QImage.Format.Format_ARGB32_Premultiplied
        )
        painter = QtGui.QPainter(image)
        with patch.object(painter, "drawText") as draw_text:
            view._paint_overlay(painter)
        painter.end()
        assert all(
            "Restore All" not in str(call.args)
            for call in draw_text.call_args_list
        )
    main.set_visible_mask(np.zeros(len(main._points), dtype=bool))
    workspace.sync_display()
    painter = Mock()
    main._paint_overlay(painter)
    assert any(
        "Restore All" in str(call.args)
        for call in painter.drawText.call_args_list
    )
    assert not top._visible_count


def test_rotation_fields_keep_world_center_and_main_cloud_registered(
    detection, app
):
    window = detection
    workspace = window.detection
    workspace.create((8, -5, 2), (4, 2, 1))
    window.resize(1200, 900)
    window.show()
    app.processEvents()
    main = workspace.views[0]
    for view in workspace.views:
        view.align_cuboid(workspace.selected, fit=True)
    main.set_cloud(np.array([[8, -5, 2]], dtype=np.float32))
    main.align_cuboid(workspace.selected, fit=True)
    main.set_colors(np.array([[1, 1, 1, 1]], dtype=np.float32))
    main.set_point_size(10)
    workspace.sync_display()
    camera = main._matrix().copy()
    center = workspace.selected.center
    owners = main._gl.capture_surface().copy()
    for axis, value in ((0, 35), (1, -25), (2, 65), (0, -75), (1, 80)):
        workspace.fields["rotation"][axis].setValue(value)
        workspace.fields["rotation"][axis].editingFinished.emit()
        app.processEvents()
        assert workspace.selected.center == center
        np.testing.assert_array_equal(main._matrix(), camera)
        np.testing.assert_allclose(
            main.selected_cuboid.corners().mean(axis=0), center
        )
        np.testing.assert_allclose(
            main._cuboid_mesh[0][:36, :3].mean(axis=0), center, atol=1e-5
        )
        np.testing.assert_array_equal(main._gl.capture_surface(), owners)
        assert main._error is None


def test_repeated_moves_and_rotations_keep_cuboid_registration(detection, app):
    workspace = detection.detection
    workspace.create((8, -5, 2), (4, 2, 1))
    workspace.commit(replace(workspace.selected, rotation=(0.2, -0.3, 0.4)))
    detection.resize(1200, 900)
    detection.show()
    app.processEvents()
    main = workspace.views[0]
    for view in workspace.views:
        view.align_cuboid(workspace.selected, fit=True)
    camera = main._matrix().copy()
    points = main._points.copy()
    for index in (1, 2, 3, 1, 3, 2):
        view = workspace.views[index]
        before = workspace.selected
        start = view.project([before.center])[0]
        end = start + (15, -10)
        expected = (
            np.asarray(before.center)
            + view.unproject(end)
            - view.unproject(start)
        )
        mouse(view, "_mouse_press", start)
        mouse(view, "_mouse_release", end)
        np.testing.assert_allclose(
            workspace.selected.center, expected, atol=1e-5
        )
        before = workspace.selected
        center = view.project([before.center])[0]
        handle = view._handles(before)[1]
        vector = handle - center
        mouse(view, "_mouse_press", handle)
        assert view._gesture[0] == "rotate"
        for angle in (0.1, 0.3, 0.7):
            rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )
            end = center + rotation @ vector
            mouse(view, "_mouse_move", end)
            app.processEvents()
            assert main.selected_cuboid.center == before.center
            np.testing.assert_array_equal(main._matrix(), camera)
            np.testing.assert_array_equal(main._points, points)
            np.testing.assert_allclose(
                main._cuboid_mesh[0][:36, :3].mean(axis=0),
                before.center,
                atol=1e-5,
            )
            assert main._error is None
        mouse(view, "_mouse_release", end)
        assert workspace.selected.center == before.center
        assert workspace.selected.size == before.size
        for side in workspace.views[1:]:
            np.testing.assert_allclose(
                side.project([before.center])[0],
                np.array((side.width(), side.height())) / 2,
                atol=1e-4,
            )


@pytest.mark.parametrize("index", [1, 2, 3])
@pytest.mark.parametrize(
    "end", ["leave", "outside_move", "outside_release", "lost_release"]
)
@pytest.mark.parametrize("gesture", ["move", "resize", "rotate"])
def test_cuboid_drag_stops_at_subview_boundary(
    detection, app, index, end, gesture
):
    workspace = detection.detection
    workspace.create((4.6, 16.6, 0.3), (4.5, 2.8, 3.1))
    for side in workspace.views[1:]:
        side.align_cuboid(workspace.selected, fit=True)
    view = workspace.views[index]
    view._set_scale(12)
    handles, rotation = view._handles(workspace.selected)
    start = {
        "move": view.project([workspace.selected.center])[0],
        "resize": handles[5][1],
        "rotate": rotation,
    }[gesture]
    history = len(detection.document._undo)
    mouse(view, "_mouse_press", start)
    assert view._gesture[0] == gesture
    mouse(view, "_mouse_move", start + (5, -5))
    preview = view.selected_cuboid
    outside = start + (0, -600)
    if end == "leave":
        app.sendEvent(view._gl, QtCore.QEvent(QtCore.QEvent.Type.Leave))
    elif end == "outside_move":
        mouse(view, "_mouse_move", outside)
    elif end == "outside_release":
        mouse(view, "_mouse_release", outside)
    else:
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(*(start + (0, -100))),
            QtCore.QPointF(*(start + (0, -100))),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        app.sendEvent(view._gl, event)
    assert view._gesture is None
    assert workspace.selected == preview
    assert len(detection.document._undo) == history + 1
    mouse(view, "_mouse_move", outside)
    mouse(view, "_mouse_release", outside)
    assert workspace.selected == preview
    assert len(detection.document._undo) == history + 1
    for side in workspace.views[1:]:
        np.testing.assert_allclose(side._center, preview.center)


@pytest.mark.parametrize("index", [1, 2, 3])
def test_small_cuboid_body_drag_does_not_resize(detection, index):
    workspace = detection.detection
    workspace.create((4.6, 16.6, 0.3), (3.6, 0.73, 1.87))
    for side in workspace.views[1:]:
        side.align_cuboid(workspace.selected, fit=True)
    view = workspace.views[index]
    view._set_scale(20)
    box = workspace.selected
    start = view.project([box.center])[0]
    mouse(view, "_mouse_press", start)
    assert view._gesture[0] == "move"
    mouse(view, "_mouse_release", start + (10, -10))
    assert workspace.selected.size == box.size
    for signs, point in view._handles(workspace.selected)[0]:
        mouse(view, "_mouse_press", point)
        assert view._gesture[0] == "resize"
        assert view._gesture[3] == signs
        workspace.cancel()


def test_main_creation_previews_until_double_click_and_undoes_once(detection):
    workspace = detection.detection
    main = workspace.views[0]
    labels = detection.document.labels.copy()
    history = len(detection.document._undo)
    workspace.start_creation()
    point = main.project([[0, 0, 0]])[0]
    mouse(main, "_mouse_move", point)
    preview = main._creation_preview
    assert preview is not None
    assert preview.size == (1, 1, 1)
    np.testing.assert_allclose(
        main.project([preview.center])[0], point, atol=1e-4
    )
    assert detection.document.cuboids == ()
    assert not detection.document.dirty
    assert main.cuboids == ()
    assert main._cuboid_mesh[1] == 36
    mouse(main, "_mouse_press", point)
    mouse(main, "_mouse_release", point)
    assert detection.document.cuboids == ()
    assert main.creating
    mouse(main, "_mouse_double_click", point)
    mouse(main, "_mouse_release", point)
    created = workspace.selected
    assert created.center == preview.center
    assert created.size == (1, 1, 1)
    assert created.class_id == workspace.class_combo.currentData()
    assert main._creation_preview is None
    assert not any(view.creating for view in workspace.views)
    assert len(detection.document._undo) == history + 1
    np.testing.assert_array_equal(detection.document.labels, labels)
    detection.undo()
    assert detection.document.cuboids == ()
    detection.redo()
    assert detection.document.cuboids == (created,)


@pytest.mark.parametrize("finish", ["escape", "toggle", "leave"])
def test_main_creation_preview_clears_without_saving(detection, app, finish):
    workspace = detection.detection
    main = workspace.views[0]
    workspace.start_creation()
    point = main.project([[0, 0, 0]])[0]
    mouse(main, "_mouse_move", point)
    assert main._creation_preview is not None
    if finish == "escape":
        QtTest.QTest.keyClick(main._gl, QtCore.Qt.Key.Key_Escape)
    elif finish == "toggle":
        workspace.start_creation()
    else:
        app.sendEvent(main._gl, QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert main._creation_preview is None
    assert not len(main._cuboid_mesh[0])
    assert detection.document.cuboids == ()
    assert not detection.document.dirty
    assert main.creating == (finish == "leave")
    if finish == "leave":
        mouse(main, "_mouse_move", point)
        assert main._creation_preview is not None


def test_n_starts_preview_at_current_mouse_position(detection, app):
    workspace = detection.detection
    main = workspace.views[0]
    detection.resize(1200, 900)
    detection.show()
    detection.activateWindow()
    app.processEvents()
    point = main.project([[0, 0, 0]])[0]
    QtTest.QTest.mouseMove(
        main._gl, QtCore.QPoint(*np.rint(point).astype(int))
    )
    main._gl.setFocus()
    QtTest.QTest.qWait(50)
    with patch.object(
        QtGui.QCursor,
        "pos",
        return_value=main.mapToGlobal(
            QtCore.QPoint(*np.rint(point).astype(int))
        ),
    ):
        QtTest.QTest.keyClick(main._gl, QtCore.Qt.Key.Key_N)
    assert main.creating
    assert main._creation_preview is not None
    assert main.hasFocus() or main._gl.hasFocus()
    QtTest.QTest.keyClick(main, QtCore.Qt.Key.Key_N)
    assert not main.creating
    assert main._creation_preview is None


def test_creation_restores_main_view_and_refreshes_preview_after_zoom(
    detection, app
):
    workspace = detection.detection
    detection.show()
    app.processEvents()
    workspace.toggle_expanded_view(workspace.views[1])
    workspace.start_creation()
    main = workspace.views[0]
    assert workspace._expanded_view is None
    assert main.isVisible()
    point = main.project([[0, 0, 0]])[0]
    mouse(main, "_mouse_move", point)
    preview = main._creation_preview
    assert preview is not None
    before = main._scale
    wheel = QtGui.QWheelEvent(
        QtCore.QPointF(*point),
        QtCore.QPointF(*point),
        QtCore.QPoint(),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    main._wheel(wheel)
    assert main._scale < before
    assert main.creating
    assert main._creation_preview is not None
    np.testing.assert_allclose(
        main.project([main._creation_preview.center])[0], point, atol=1e-4
    )
    mouse(main, "_mouse_press", point)
    mouse(main, "_mouse_move", point + (20, 0))
    mouse(main, "_mouse_release", point + (20, 0))
    assert detection.document.cuboids == ()
    assert not detection.document.dirty
