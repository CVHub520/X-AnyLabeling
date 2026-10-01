import threading
import time
from unittest.mock import patch

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets

from anylabeling.views.labeling.pointcloud import import_dialog
from anylabeling.views.labeling.pointcloud.cuboid import Cuboid
from anylabeling.views.labeling.pointcloud.export import (
    EXPORT_FORMATS,
    ExportCancelled,
    ExportFrame,
    export_dataset,
)
from anylabeling.views.labeling.pointcloud.import_dataset import ImportTarget
from anylabeling.views.labeling.pointcloud.io import load_cuboids, load_classes
from anylabeling.views.labeling.pointcloud.model import ClassDefinition

from . import test_dialog
from .test_dialog import cloud, open_cloud
from .test_export_dialog import wait_export

app = test_dialog.app
window = test_dialog.window


@pytest.mark.parametrize("format_name", EXPORT_FORMATS)
def test_upload_imports_and_preserves_unsaved_segmentation(
    window, app, tmp_path, format_name
):
    assert not window.import_action.isEnabled()
    path = cloud(tmp_path / "scan.bin")
    open_cloud(window, app, path)
    window._autosave_timer.stop()
    window.document.assign_semantic([0], 200)
    window.class_definitions["segmentation"].append(
        ClassDefinition(200, "Car", "#556677")
    )
    segmentation = list(window.class_definitions["segmentation"])
    labels = window.document.labels.copy()
    window.document.set_cuboid(
        Cuboid(1, 10, (10, 20, 30), (2, 3, 4), locked=True)
    )
    replacement = Cuboid(
        4, 999, (1, 2, 3), (4, 5, 6), (0.2, -0.3, 0.4), True, True
    )
    archive = tmp_path / "annotations.zip"
    export_dataset(
        archive,
        format_name,
        [ExportFrame(path, path.with_suffix(".cuboids.json"), (replacement,))],
        [ClassDefinition(999, "Car", "#AABBCC")],
    )
    original_exec = import_dialog.PointCloudImportDialog.exec

    def run(dialog):
        assert not dialog.save_images.isVisible()
        assert dialog.scope_label.text() == "Replace matched frames"
        dialog.path_input.setText(str(archive))
        dialog.format_buttons[format_name].setChecked(True)
        QtCore.QTimer.singleShot(0, dialog.accept)
        return original_exec(dialog)

    toolbar = window.findChild(QtWidgets.QToolBar, "pointcloudFileTools")
    actions = toolbar.actions()
    start = actions.index(window.save_as_action)
    assert actions[start : start + 3] == [
        window.save_as_action,
        window.import_action,
        window.export_action,
    ]
    assert all(
        not action.icon().isNull() for action in actions[start : start + 3]
    )
    with (
        patch.object(import_dialog.PointCloudImportDialog, "exec", run),
        patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.StandardButton.Yes,
        ) as confirm,
    ):
        window.import_action.trigger()
    window._autosave_timer.stop()
    assert not window._errors
    confirm.assert_called_once()
    assert "1 existing objects" in confirm.call_args.args[2]
    boxes = window.document.cuboids
    assert len(boxes) == 1
    assert boxes[0].center == replacement.center
    assert boxes[0].size == replacement.size
    assert boxes[0].rotation == replacement.rotation
    assert boxes[0].class_id == 71
    assert boxes[0].locked == (format_name != "kitti_raw")
    assert window.sidebar_tabs.currentWidget() is window.detection.panel
    np.testing.assert_array_equal(window.document.labels, labels)
    assert window.document.labels_dirty
    assert not window.document.cuboids_dirty
    assert load_cuboids(path.with_suffix(".cuboids.json"), path) == boxes
    assert any(
        item.name == "Car" and item.id == 71
        for item in load_classes(window.config_path)["detection"]
    )
    assert window.class_definitions["segmentation"] == segmentation
    assert load_classes(window.config_path)["segmentation"] == segmentation
    assert not window.config_dirty
    window.document.undo()
    assert window.document.cuboids[0].center == (10, 20, 30)


def test_import_preview_decline_does_not_change_files(window, app, tmp_path):
    path = cloud(tmp_path / "scan.bin")
    open_cloud(window, app, path)
    frame = ExportFrame(path, path.with_suffix(".cuboids.json"), ())
    archive = tmp_path / "annotations.zip"
    export_dataset(
        archive, "datumaro", [frame], window.class_definitions["detection"]
    )
    target = ImportTarget(
        path, path.with_suffix(".label"), frame.cuboid_path, ()
    )
    dialog = import_dialog.PointCloudImportDialog(
        [target], window.class_definitions["detection"], window
    )
    dialog.path_input.setText(str(archive))
    dialog.show()
    with patch.object(
        QtWidgets.QMessageBox,
        "question",
        return_value=QtWidgets.QMessageBox.StandardButton.No,
    ):
        dialog.accept()
        wait_export(dialog, app)
    assert dialog.import_plan is None
    assert dialog.isVisible()
    assert dialog.ok_button.isEnabled()
    assert not frame.cuboid_path.exists()
    dialog.reject()
    dialog.deleteLater()


@pytest.mark.parametrize("name", ["Vehicle", "New class"])
def test_fixed_task_import_keeps_class_definitions(
    window, app, tmp_path, name
):
    path = cloud(tmp_path / "scan.bin")
    open_cloud(window, app, path)
    window.task_type = "detection"
    cuboid_path = path.with_suffix(".cuboids.json")
    archive = tmp_path / "annotations.zip"
    export_dataset(
        archive,
        "datumaro",
        [
            ExportFrame(
                path, cuboid_path, (Cuboid(1, 999, (0, 0, 0), (1, 1, 1)),)
            )
        ],
        [ClassDefinition(999, name, "#AABBCC")],
    )
    classes = list(window.class_definitions["detection"])
    before = window.config_path.read_bytes()
    dialog = import_dialog.PointCloudImportDialog(
        [ImportTarget(path, path.with_suffix(".label"), cuboid_path, ())],
        classes,
        window,
    )
    dialog.path_input.setText(str(archive))
    dialog.show()
    with (
        patch.object(QtWidgets.QMessageBox, "warning") as warning,
        patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.StandardButton.Yes,
        ) as confirm,
    ):
        dialog.accept()
        wait_export(dialog, app)
    if name == "New class":
        warning.assert_called_once()
        confirm.assert_not_called()
        assert dialog.import_plan is None
        assert not cuboid_path.exists()
        assert window.config_path.read_bytes() == before
    else:
        warning.assert_not_called()
        confirm.assert_called_once()
        assert dialog.import_plan.classes == tuple(classes)
        assert load_cuboids(cuboid_path, path)[0].class_id == 10
    assert window.class_definitions["detection"] == classes
    dialog.reject()
    dialog.deleteLater()


def test_cancel_waits_for_import_validation(window, app, tmp_path):
    path = cloud(tmp_path / "scan.bin")
    open_cloud(window, app, path)
    archive = tmp_path / "annotations.zip"
    archive.touch()
    target = ImportTarget(
        path, path.with_suffix(".label"), path.with_suffix(".cuboids.json"), ()
    )
    dialog = import_dialog.PointCloudImportDialog(
        [target], window.class_definitions["detection"], window
    )
    dialog.path_input.setText(str(archive))
    started = threading.Event()

    def prepare(**options):
        started.set()
        deadline = time.monotonic() + 5
        while not options["cancelled"]() and time.monotonic() < deadline:
            time.sleep(0.001)
        raise ExportCancelled()

    with patch.object(import_dialog, "prepare_import", prepare):
        dialog.show()
        dialog.accept()
        assert started.wait(5)
        dialog.close()
        assert dialog.worker is not None
        assert dialog.isVisible()
        wait_export(dialog, app)
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    assert dialog.import_plan is None
    dialog.deleteLater()
