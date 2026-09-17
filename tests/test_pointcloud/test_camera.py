import json
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PyQt6 import QtGui, QtWidgets

from anylabeling.views.labeling.pointcloud.camera import (
    CameraConfigurationDialog,
    image_files,
    load_calibration,
    project_points,
)
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
        window.camera_action.trigger()
    panel = window.camera_panel
    assert panel.parent() is window.viewport
    assert not panel._image.isNull()
    assert panel.filename.text() == "0000.jpg"
    assert not panel.overlay_action.isEnabled()

    def accept_calibration(dialog):
        path = tmp_path / "calibration.json"
        path.write_text(json.dumps(calibration_data()))
        dialog.calibration_input.setText(str(path))
        dialog.accept()
        return dialog.result()

    with patch.object(CameraConfigurationDialog, "exec", accept_calibration):
        window.camera_action.trigger()
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
    assert panel.view.isHidden()
    assert not panel.overlay_action.isEnabled()
    assert not window.document.dirty


def test_cancel_image_directory_leaves_panel_hidden(window):
    with patch.object(
        CameraConfigurationDialog,
        "exec",
        return_value=QtWidgets.QDialog.DialogCode.Rejected,
    ):
        window.camera_action.trigger()
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
    assert dialog.configuration[2] is None


def test_bundled_example_projects_into_image(app):
    from pathlib import Path
    from anylabeling.views.labeling.pointcloud.io import load_frame

    root = Path(__file__).resolve().parents[2] / "assets" / "pointcloud"
    frame = load_frame(root / "0000000000.bin")
    image = QtGui.QImage(str(root / "0000000000.png"))
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
