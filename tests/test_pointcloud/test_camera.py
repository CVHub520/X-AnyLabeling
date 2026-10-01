import json
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.pointcloud.camera import (
    CameraConfigurationDialog,
    CameraPanel,
    image_files,
    load_calibration,
    project_cuboid,
    project_points,
)
from anylabeling.views.labeling.pointcloud.cuboid import Cuboid, EDGES
from .test_dialog import app, window, cloud, open_cloud


def calibration_data():
    return {
        "schema_version": 1,
        "image_size": [40, 40],
        "camera_model": "pinhole",
        "camera_matrix": [[10, 0, 20], [0, 10, 20], [0, 0, 1]],
        "T_pointcloud_to_camera": np.eye(4).tolist(),
        "distortion_model": "none",
        "distortion_coefficients": [],
    }


def save_calibration(tmp_path, data=None):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration_data() if data is None else data))
    return load_calibration(path)


def save_image(path, width=40, height=40):
    image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("gray"))
    assert image.save(str(path))


def test_directory_matching_and_duplicate_names(app, tmp_path):
    save_image(tmp_path / "frame.png")
    assert image_files(tmp_path)["frame"].name == "frame.png"
    save_image(tmp_path / "frame.jpg")
    with pytest.raises(ValueError, match="same basename"):
        image_files(tmp_path)


def test_camera_directory_projection_and_missing_frame(window, app, tmp_path):
    images = tmp_path / "photographs"
    images.mkdir()
    save_image(images / "0000.jpg")
    open_cloud(window, app, cloud(tmp_path / "0000.bin"))

    def accept_dialog(dialog):
        dialog.directory_input.setText(str(images))
        dialog.accept()
        return dialog.result()

    with patch.object(CameraConfigurationDialog, "exec", accept_dialog):
        window._configure_camera()
    panel = window.camera_panel
    assert panel.parent() is window.viewport
    assert not panel._image.isNull()
    assert panel.filename.text() == "photographs"
    assert not panel.overlay_action.isEnabled()

    def accept_calibration(dialog):
        path = tmp_path / "calibration.json"
        path.write_text(json.dumps(calibration_data()))
        dialog.calibration_input.setText(str(path))
        dialog.accept()
        return dialog.result()

    with patch.object(CameraConfigurationDialog, "exec", accept_calibration):
        window._configure_camera()
    assert len(panel._indices) > 0
    assert panel.overlay_action.isEnabled()
    window.classes_visibility_action.trigger()
    overlay = panel.view.overlay_item.pixmap().toImage()
    assert all(
        overlay.pixelColor(x, y).alpha() == 0
        for x in range(40)
        for y in range(40)
    )
    window.viewport.resize(700, 500)
    panel.position_panel()
    assert panel.geometry().right() < window.viewport.width()
    assert panel.geometry().top() == 8
    open_cloud(window, app, cloud(tmp_path / "0001.bin"))
    assert panel._image.isNull()
    assert panel.view.image_item.pixmap().isNull()
    assert not panel.overlay_action.isEnabled()
    assert not window.document.dirty


def test_cancel_image_directory_leaves_panel_hidden(window):
    with patch.object(
        CameraConfigurationDialog,
        "exec",
        return_value=QtWidgets.QDialog.DialogCode.Rejected,
    ):
        window._configure_camera()
    assert window.camera_panel.isHidden()


def test_invalid_calibration_does_not_apply_configuration(window, tmp_path):
    save_image(tmp_path / "frame.png")
    dialog = CameraConfigurationDialog(window.camera_panel, window)
    dialog.directory_input.setText(str(tmp_path))
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("{}")
    dialog.calibration_input.setText(str(invalid))
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    assert not dialog.error_label.isHidden()
    assert window.camera_panel.directory is None
    dialog.calibration_input.clear()
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    assert dialog.configuration[0].calibration is None


def test_bundled_example_projects_into_image(app):
    from pathlib import Path
    from anylabeling.views.labeling.pointcloud.io import load_frame

    root = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "pointcloud"
        / "segmentation"
        / "2011_10_03_drive_0042_sync"
    )
    frame = load_frame(root / "velodyne" / "0000000000.bin")
    image = QtGui.QImage(str(root / "image_02" / "0000000000.png"))
    calibration = load_calibration(root / "calibration.json")
    indices, pixels = project_points(
        frame.points, calibration, image.width(), image.height()
    )
    assert len(indices) > 10000
    assert pixels[:, 0].max() < image.width()
    assert pixels[:, 1].max() < image.height()
    reference = np.array(
        [
            [
                607.4843712729912,
                -718.5373919042582,
                -10.187583601729493,
                -140.65278362874417,
            ],
            [
                180.02745718312917,
                5.899223315134038,
                -720.1486522469515,
                -93.07940403331872,
            ],
            [
                0.9999738645903279,
                0.0004859485810390038,
                -0.0072069336924223334,
                -0.28841710386859104,
            ],
        ]
    )
    np.testing.assert_allclose(
        calibration["camera_matrix"]
        @ calibration["T_pointcloud_to_camera"][:3],
        reference,
        rtol=0,
        atol=1e-10,
    )


def test_projection_rejects_points_behind_camera_and_outside_image(tmp_path):
    points = np.array(
        [
            [0, 0, 2],
            [0, 0, -1],
            [100, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
            [np.nan, 0, 1],
            [0, np.inf, 1],
            [2, 0, 1],
            [-2.01, 0, 1],
        ]
    )
    indices, pixels = project_points(
        points, save_calibration(tmp_path), 40, 40
    )
    np.testing.assert_array_equal(indices, [0, 3])
    np.testing.assert_array_equal(pixels, [[20, 20], [30, 30]])


def test_projection_uses_pointcloud_to_camera_direction(tmp_path):
    data = calibration_data()
    data["T_pointcloud_to_camera"] = [
        [0, -1, 0, 1],
        [1, 0, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
    ]
    points = np.array([[1, 0, 1, 99], [0, 1, 2, 88], [0, 0, -4, 77]])
    indices, pixels = project_points(
        points, save_calibration(tmp_path, data), 40, 40
    )
    np.testing.assert_array_equal(indices, [1, 0])
    np.testing.assert_array_equal(pixels, [[20, 24], [22, 27]])


@pytest.mark.parametrize("model", ["none", "opencv5"])
def test_projection_matches_opencv(tmp_path, model):
    data = calibration_data()
    data["image_size"] = [640, 480]
    data["camera_matrix"] = [[180, 0, 320], [0, 200, 240], [0, 0, 1]]
    data["distortion_model"] = model
    coefficients = [0.2, -0.03, 0.015, -0.02, 0.004]
    data["distortion_coefficients"] = (
        coefficients if model == "opencv5" else []
    )
    rotation_vector = np.array([0.15, -0.2, 0.1])
    translation = np.array([0.7, -0.3, 1.2])
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    transform[:3, 3] = translation
    data["T_pointcloud_to_camera"] = transform.tolist()
    calibration = save_calibration(tmp_path, data)
    points = np.random.default_rng(42).uniform(-4, 4, (200, 3))
    expected, _ = cv2.projectPoints(
        points,
        rotation_vector,
        translation,
        calibration["camera_matrix"],
        np.array(coefficients) if model == "opencv5" else None,
    )
    expected = expected.reshape(-1, 2)
    depth = points @ transform[2, :3] + transform[2, 3]
    valid = (
        (depth > 0)
        & (expected >= 0).all(axis=1)
        & (expected < [640, 480]).all(axis=1)
    )
    expected_indices = np.flatnonzero(valid)
    expected_indices = expected_indices[
        np.argsort(-depth[expected_indices], kind="stable")
    ]
    indices, pixels = project_points(points, calibration, 640, 480)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(
        pixels, expected[expected_indices].astype(np.int32)
    )


@pytest.mark.parametrize("model", ["none", "opencv5"])
def test_projection_handles_empty_and_unprojectable_points(tmp_path, model):
    data = calibration_data()
    data["distortion_model"] = model
    data["distortion_coefficients"] = [0.1] * 5 if model == "opencv5" else []
    calibration = save_calibration(tmp_path, data)
    for points in (
        np.empty((0, 3)),
        np.array([[0, 0, -1], [1e300, 1e300, 1e-300]]),
    ):
        indices, pixels = project_points(points, calibration, 40, 40)
        assert indices.shape == (0,)
        assert pixels.shape == (0, 2)
        assert pixels.dtype == np.int32


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("camera_model", "fisheye"),
        ("image_size", [40, 0]),
        ("image_size", [40.0, 40]),
        ("image_size", [True, 40]),
        ("image_size", [40]),
        ("camera_matrix", [[1, 0, 0]]),
        ("camera_matrix", [[-10, 0, 20], [0, 10, 20], [0, 0, 1]]),
        ("camera_matrix", [[10, 1, 20], [0, 10, 20], [0, 0, 1]]),
        ("camera_matrix", [[10, 0, 20], [0, 10, 20], [0, 0, 2]]),
        ("camera_matrix", [[10, 0, float("nan")], [0, 10, 20], [0, 0, 1]]),
        (
            "camera_matrix",
            [["10", "0", "20"], ["0", "10", "20"], ["0", "0", "1"]],
        ),
        ("T_pointcloud_to_camera", np.eye(3).tolist()),
        ("T_pointcloud_to_camera", np.diag([2, 1, 1, 1]).tolist()),
        ("T_pointcloud_to_camera", np.diag([-1, 1, 1, 1]).tolist()),
        ("T_pointcloud_to_camera", np.diag([1, 1, 1, 2]).tolist()),
        (
            "T_pointcloud_to_camera",
            [
                [1, 0, 0, float("inf")],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        ),
        ("distortion_model", "fisheye"),
        ("distortion_coefficients", [0, 0, 0, 0, 0]),
    ],
)
def test_invalid_calibration_fields(tmp_path, field, value):
    data = calibration_data()
    data[field] = value
    with pytest.raises(ValueError, match=field):
        save_calibration(tmp_path, data)


@pytest.mark.parametrize(
    "coefficients", [[], [0] * 4, [0] * 8, [0, 0, 0, 0, float("inf")]]
)
def test_invalid_opencv5_coefficients(tmp_path, coefficients):
    data = calibration_data()
    data.update(
        distortion_model="opencv5", distortion_coefficients=coefficients
    )
    with pytest.raises(ValueError, match="distortion_coefficients"):
        save_calibration(tmp_path, data)


@pytest.mark.parametrize(
    "data", [[], None, {"projection_matrix": np.eye(3, 4).tolist()}]
)
def test_rejects_invalid_root_and_old_format(tmp_path, data):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_calibration(path)


def test_rejects_missing_and_unknown_fields(tmp_path):
    data = calibration_data()
    del data["distortion_model"]
    with pytest.raises(ValueError, match="Missing.*distortion_model"):
        save_calibration(tmp_path, data)
    data = calibration_data()
    data["extrinsic"] = np.eye(4).tolist()
    with pytest.raises(ValueError, match="Unknown.*extrinsic"):
        save_calibration(tmp_path, data)


def test_mismatched_image_size_rejects_projection(tmp_path):
    with pytest.raises(ValueError, match="image_size"):
        project_points(
            np.array([[0, 0, 1]]), save_calibration(tmp_path), 80, 40
        )


def test_mismatched_image_keeps_reference_and_recovers(window, app, tmp_path):
    save_image(tmp_path / "0000.png")
    save_image(tmp_path / "0001.png", width=80)
    first = cloud(tmp_path / "0000.bin")
    second = cloud(tmp_path / "0001.bin")
    open_cloud(window, app, first)
    panel = window.camera_panel
    panel.configure(
        tmp_path,
        image_files(tmp_path),
        save_calibration(tmp_path),
        tmp_path / "calibration.json",
    )
    assert not panel.view.overlay_item.pixmap().isNull()
    open_cloud(window, app, second)
    assert not panel._image.isNull()
    assert not panel.view.isHidden()
    assert panel.view.overlay_item.pixmap().isNull()
    assert not panel.overlay_action.isEnabled()
    assert not panel.message.isHidden()
    assert "80 x 40" in panel.message.text()
    assert "40 x 40" in panel.message.text()
    assert len(panel._indices) == 0
    panel.refresh()
    assert not panel.overlay_action.isEnabled()
    open_cloud(window, app, first)
    assert panel.overlay_action.isEnabled()
    assert panel.message.isHidden()
    assert not panel.view.overlay_item.pixmap().isNull()
    panel.configure(tmp_path, image_files(tmp_path), None, None)
    assert not panel._image.isNull()
    assert not panel.overlay_action.isEnabled()
    assert panel.view.overlay_item.pixmap().isNull()
    assert len(panel._indices) == 0
    assert not window.document.dirty


@pytest.fixture
def projected_panel(window, app, tmp_path):
    save_image(tmp_path / "0000.png")
    source = cloud(tmp_path / "0000.bin")
    np.array(
        [
            [-1, 0, 2, 0],
            [0.2, 0, 2, 0],
            [0, 0, 8, 0],
            [2, 1, 8, 0],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    ).tofile(source)
    open_cloud(window, app, source)
    panel = window.camera_panel
    panel.configure(
        tmp_path,
        image_files(tmp_path),
        save_calibration(tmp_path),
        tmp_path / "calibration.json",
    )
    return panel


def test_projection_depth_is_independent_and_filter_range_is_stable(
    projected_panel, window
):
    panel = projected_panel
    assert panel.overlay_color_buttons["depth"].isChecked()
    assert panel.overlay_size.value() == 1
    assert panel.view.overlay_item.opacity() == pytest.approx(0.65)
    original = panel.view.overlay_item.pixmap().toImage()
    near, far = original.pixelColor(15, 20), original.pixelColor(20, 20)
    assert near.red() > near.blue()
    assert far.blue() > far.red()
    assert original.pixelColor(14, 20).alpha() == 0
    legend = panel.depth_legend.text()
    window.viewport._colors[:, :3] = [0, 1, 0]
    with patch(
        "anylabeling.views.labeling.pointcloud.camera.project_points",
        side_effect=AssertionError("Unexpected reprojection"),
    ):
        panel.refresh()
        assert panel.view.overlay_item.pixmap().toImage() == original
        window._visible[0] = False
        panel.refresh()
    filtered = panel.view.overlay_item.pixmap().toImage()
    assert filtered.pixelColor(15, 20).alpha() == 0
    assert filtered.pixelColor(20, 20) == far
    assert panel.depth_legend.text() == legend
    assert not window.document.dirty


def test_projection_controls_preserve_geometry_and_remember_settings(
    projected_panel, window
):
    panel = projected_panel
    points, pixels = window.document.frame.points.copy(), panel._pixels.copy()
    labels = window.document.labels.copy()
    main_size, main_color = (
        window.point_size.value(),
        window.color_mode.currentData(),
    )
    window.viewport._colors[:, :3] = [0, 1, 0]
    panel.overlay_color_buttons["cloud"].click()
    panel.overlay_size.setValue(3)
    rendered = panel.view.overlay_item.pixmap().toImage()
    assert rendered.pixelColor(14, 19).getRgb() == (0, 255, 0, 255)
    assert rendered.pixelColor(13, 19).alpha() == 0
    with patch.object(
        panel,
        "refresh",
        side_effect=AssertionError("Opacity should not rerasterize"),
    ):
        panel.overlay_opacity.setValue(25)
    assert panel.view.overlay_item.opacity() == 0.25
    assert panel.view.overlay_item.pixmap().toImage() == rendered
    np.testing.assert_array_equal(window.document.frame.points, points)
    np.testing.assert_array_equal(panel._pixels, pixels)
    np.testing.assert_array_equal(window.document.labels, labels)
    assert (window.point_size.value(), window.color_mode.currentData()) == (
        main_size,
        main_color,
    )
    assert not window.document.dirty
    restored = CameraPanel(window)
    assert restored.overlay_color_buttons["cloud"].isChecked()
    assert restored.overlay_size.value() == 3
    assert restored.overlay_opacity.value() == 25
    assert restored.view.overlay_item.opacity() == 0.25
    restored.deleteLater()


def test_enlarged_projected_points_keep_nearest_color(projected_panel, window):
    panel = projected_panel
    window.viewport._colors[:, :3] = [0, 0, 1]
    window.viewport._colors[1, :3] = [1, 0, 0]
    panel.overlay_color_buttons["cloud"].click()
    panel.overlay_size.setValue(3)
    image = panel.view.overlay_item.pixmap().toImage()
    assert image.pixelColor(20, 20).getRgb() == (255, 0, 0, 255)
    assert image.pixelColor(19, 20).getRgb() == (0, 0, 255, 255)
    window._visible[1] = False
    panel.refresh()
    assert panel.view.overlay_item.pixmap().toImage().pixelColor(
        20, 20
    ).getRgb() == (0, 0, 255, 255)
    window._visible[:] = False
    panel.refresh()
    image = panel.view.overlay_item.pixmap().toImage()
    assert all(
        image.pixelColor(x, y).alpha() == 0
        for x in range(40)
        for y in range(40)
    )


def test_projection_depth_resets_on_empty_frame(
    projected_panel, window, app, tmp_path
):
    panel = projected_panel
    save_image(tmp_path / "0001.png")
    source = cloud(tmp_path / "0001.bin")
    np.array([[0, 0, -1, 0]], dtype=np.float32).tofile(source)
    panel.files = image_files(tmp_path)
    open_cloud(window, app, source)
    assert panel._depth_colors.shape == (0, 3)
    panel.overlay_size.setValue(7)
    assert (
        panel.view.overlay_item.pixmap().toImage().pixelColor(20, 20).alpha()
        == 0
    )
    assert panel.depth_legend.text() == panel.tr("Near → Far")


@pytest.mark.parametrize("model", ["none", "opencv5"])
def test_projected_cuboid_edges_match_opencv(tmp_path, model):
    data = calibration_data()
    data["image_size"] = [640, 480]
    data["camera_matrix"] = [[180, 0, 320], [0, 200, 240], [0, 0, 1]]
    data["distortion_model"] = model
    data["distortion_coefficients"] = (
        [0.2, -0.03, 0.015, -0.02, 0.004] if model == "opencv5" else []
    )
    rotation = np.array([0.15, -0.2, 0.1])
    translation = np.array([0.7, -0.3, 1.2])
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rotation)[0]
    transform[:3, 3] = translation
    data["T_pointcloud_to_camera"] = transform.tolist()
    calibration = save_calibration(tmp_path, data)
    box = Cuboid(1, 10, (0, 0, 6), (2, 1.5, 2), (0.2, -0.3, 0.6))
    edges = box.corners()[np.asarray(EDGES)]
    samples = 33 if model == "opencv5" else 2
    expected = []
    for start, end in edges:
        world = np.linspace(start, end, samples)
        pixels, _ = cv2.projectPoints(
            world,
            rotation,
            translation,
            calibration["camera_matrix"],
            calibration["distortion_coefficients"],
        )
        pixels = pixels.reshape(-1, 2)
        expected.extend(np.stack((pixels[:-1], pixels[1:]), axis=1))
    actual = project_cuboid(box, calibration, 640, 480)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("model", ["none", "opencv5"])
def test_projected_cuboids_clip_near_plane_and_image(tmp_path, model):
    data = calibration_data()
    data["distortion_model"] = model
    data["distortion_coefficients"] = (
        [0.1, 0, 0, 0, 0] if model == "opencv5" else []
    )
    calibration = save_calibration(tmp_path, data)
    box = Cuboid(1, 10, (0, 0, 0.5), (2, 2, 2))
    crossing = project_cuboid(box, calibration, 40, 40)
    assert len(crossing) >= 8
    assert np.isfinite(crossing).all()
    assert crossing.min() >= -1e-7
    assert crossing.max() <= 39 + 1e-7
    behind = replace(box, center=(0, 0, -2))
    assert project_cuboid(behind, calibration, 40, 40).shape == (0, 2, 2)
    outside = replace(box, center=(100, 0, 5))
    assert project_cuboid(outside, calibration, 40, 40).shape == (0, 2, 2)
    with pytest.raises(ValueError, match="image_size"):
        project_cuboid(box, calibration, 80, 40)


def test_projected_edge_can_cross_image_with_both_corners_outside(tmp_path):
    box = Cuboid(1, 10, (0, 0, 5), (30, 2, 2))
    lines = project_cuboid(box, save_calibration(tmp_path), 40, 40)
    assert len(lines) == 4
    np.testing.assert_allclose(
        np.sort(lines[:, :, 0], axis=1), [[0, 39]] * 4, atol=1e-10
    )


def test_camera_boxes_follow_preview_commit_selection_and_undo(
    projected_panel, window
):
    panel = projected_panel
    window.sidebar_tabs.setCurrentWidget(window.detection.panel)
    detection = window.detection
    box = Cuboid(1, 10, (0, 0, 6), (2, 2, 2))
    detection.commit(box)
    item = panel.view.cuboid_item
    assert panel.cuboid_action.isEnabled() and item.isVisible()
    assert [entry[0] for entry in item.paths] == [box.id]
    original = QtGui.QPainterPath(item.paths[0][1])
    selected_width = item.paths[0][2].widthF()
    assert item.paths[0][2].isCosmetic()
    assert (
        item.paths[0][2].color().name()
        == window.viewport.class_colors[10].lower()
    )
    detection.select(None)
    assert item.paths[0][2].widthF() < selected_width
    detection.select(box.id)
    moved = replace(
        box, center=(1, 0.5, 7), rotation=(0.2, -0.3, 0.4), size=(3, 2, 1)
    )
    with patch(
        "anylabeling.views.labeling.pointcloud.camera.project_points",
        side_effect=AssertionError("Box preview must not reproject points"),
    ):
        detection.preview(moved, detection.views[1])
    preview = QtGui.QPainterPath(item.paths[0][1])
    assert preview != original
    assert window.document.cuboids == (box,)
    detection.preview(None)
    assert item.paths[0][1] == original
    detection.commit(moved)
    assert item.paths[0][1] == preview
    window.undo_action.trigger()
    assert item.paths[0][1] == original
    window.redo_action.trigger()
    assert item.paths[0][1] == preview
    detection.delete()
    assert item.paths == []


def test_camera_box_visibility_is_independent_of_point_overlay(
    projected_panel, window
):
    panel = projected_panel
    window.sidebar_tabs.setCurrentWidget(window.detection.panel)
    detection = window.detection
    box = Cuboid(1, 10, (0, 0, 6), (2, 2, 2), occluded=True)
    detection.commit(box)
    item = panel.view.cuboid_item
    assert item.paths[0][2].style() == QtCore.Qt.PenStyle.DashLine
    assert item.acceptedMouseButtons() == QtCore.Qt.MouseButton.NoButton
    panel.overlay_action.trigger()
    panel.overlay_opacity.setValue(0)
    assert not panel.view.overlay_item.isVisible()
    assert item.isVisible() and item.opacity() == 1
    panel.cuboid_action.trigger()
    assert not item.isVisible()
    panel.cuboid_action.trigger()
    assert item.isVisible()
    detection._set_visible([box.id], False)
    assert item.paths == []
    detection._set_visible([box.id], True)
    assert len(item.paths) == 1
    window.sidebar_tabs.setCurrentIndex(1)
    assert not item.isVisible() and not panel.cuboid_action.isEnabled()
    window.sidebar_tabs.setCurrentWidget(detection.panel)
    assert item.isVisible() and len(item.paths) == 1
    assert panel.view.sceneRect() == QtCore.QRectF(panel._image.rect())


def test_camera_boxes_clear_on_frame_or_calibration_change(
    projected_panel, window, app, tmp_path
):
    panel = projected_panel
    window.sidebar_tabs.setCurrentWidget(window.detection.panel)
    window.detection.commit(Cuboid(1, 10, (0, 0, 6), (2, 2, 2)))
    item = panel.view.cuboid_item
    panel.configure(tmp_path, image_files(tmp_path), None, None)
    assert not item.isVisible() and item.paths == []
    assert not panel.cuboid_action.isEnabled()
    panel.configure(
        tmp_path,
        image_files(tmp_path),
        save_calibration(tmp_path),
        tmp_path / "calibration.json",
    )
    assert len(item.paths) == 1 and item.isVisible()
    save_image(tmp_path / "0001.png", width=80)
    panel.files = image_files(tmp_path)
    open_cloud(window, app, cloud(tmp_path / "0001.bin"))
    assert item.paths == [] and not item.isVisible()
    assert not panel.cuboid_action.isEnabled()


def test_camera_boxes_show_and_clear_creation_preview(projected_panel, window):
    panel = projected_panel
    window.sidebar_tabs.setCurrentWidget(window.detection.panel)
    assert window.document.cuboids == ()
    window.viewport._set_creation_preview(Cuboid(1, 10, (0, 0, 6), (2, 2, 2)))
    item = panel.view.cuboid_item
    assert len(item.paths) == 1
    assert item.paths[0][2].color().name() == "#ffd54f"
    window.viewport._set_creation_preview(None)
    assert item.paths == []
    assert not window.document.dirty
