import json
import threading
import time
from unittest.mock import patch
from zipfile import ZipFile

from PyQt6 import QtCore, QtWidgets

from anylabeling.views.labeling.pointcloud import export_dialog
from anylabeling.views.labeling.pointcloud.cuboid import Cuboid
from anylabeling.views.labeling.pointcloud.export import (
    ExportCancelled,
    ExportFrame,
)
from anylabeling.views.labeling.pointcloud.io import save_cuboids
from anylabeling.views.labeling.widgets import pointcloud_dialog

from . import test_dialog
from .test_dialog import cloud, open_cloud, wait_load

app = test_dialog.app
window = test_dialog.window


def wait_export(dialog, app):
    deadline = time.monotonic() + 10
    while dialog.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        QtCore.QThread.msleep(1)
    assert dialog.worker is None
    app.processEvents()


def test_dialog_layout_browse_radios_and_export(window, app, tmp_path):
    assert not window.export_action.isEnabled()
    source = cloud(tmp_path / "000000.bin")
    open_cloud(window, app, source)
    assert window.export_action.isEnabled()
    frame = ExportFrame(source, source.with_suffix(".cuboids.json"), ())
    window.settings.setValue("export_directory", str(source.parent))
    dialog = export_dialog.PointCloudExportDialog(
        [frame], window.class_definitions["detection"], window
    )
    assert dialog.path_input.text() == str(
        source.parent.parent / "dataset_export.zip"
    )
    dialog.show()
    app.processEvents()
    assert not dialog.save_images.isChecked()
    assert dialog.format_buttons["datumaro"].isChecked()
    third = dialog.format_buttons["sly_pointcloud"].geometry()
    assert dialog.browse_button.geometry().left() == third.left()
    assert dialog.browse_button.geometry().right() == third.right()
    assert dialog.save_images.x() == dialog.format_buttons["datumaro"].x()
    assert dialog.ok_button.geometry().right() == third.right()
    assert not dialog.browse_button.icon().isNull()
    for key, button in dialog.format_buttons.items():
        button.click()
        assert button.isChecked()
        assert (
            sum(item.isChecked() for item in dialog.format_buttons.values())
            == 1
        )
    path = tmp_path / "export.zip"
    with patch.object(
        QtWidgets.QFileDialog, "getSaveFileName", return_value=(str(path), "")
    ):
        dialog.browse_button.click()
    assert dialog.path_input.text() == str(path)
    dialog.save_images.setChecked(True)
    dialog.ok_button.click()
    assert not dialog.path_input.isEnabled()
    wait_export(dialog, app)
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    assert dialog.export_result == (1, 0)
    with ZipFile(path) as archive:
        assert "ds0/pointcloud/000000.pcd" in archive.namelist()
    dialog.deleteLater()


def test_toolbar_exports_all_frames_and_current_unsaved_boxes(
    window, app, tmp_path
):
    first, second = cloud(tmp_path / "1.bin"), cloud(tmp_path / "2.bin")
    box = Cuboid(1, 10, (2, 3, 4), (5, 6, 7))
    save_cuboids(second.with_suffix(".cuboids.json"), (box,), second)
    window.open_paths([first, second])
    wait_load(window, app)
    window._autosave_timer.stop()
    window.document.set_cuboid(box)
    window.detection.hidden_ids.add(1)
    path = tmp_path / "export.zip"
    original = export_dialog.PointCloudExportDialog.exec

    def run(dialog):
        dialog.path_input.setText(str(path))
        QtCore.QTimer.singleShot(0, dialog.accept)
        return original(dialog)

    with patch.object(pointcloud_dialog.PointCloudExportDialog, "exec", run):
        window.export_action.trigger()
    window._autosave_timer.stop()
    with ZipFile(path) as archive:
        items = json.loads(archive.read("annotations/default.json"))["items"]
    assert [item["id"] for item in items] == ["1", "2"]
    assert [len(item["annotations"]) for item in items] == [1, 1]
    assert not first.with_suffix(".cuboids.json").exists()
    assert window.document.dirty


def test_cancel_waits_for_worker_and_does_not_accept(window, app, tmp_path):
    source = cloud(tmp_path / "frame.bin")
    frame = ExportFrame(source, source.with_suffix(".cuboids.json"), ())
    dialog = export_dialog.PointCloudExportDialog(
        [frame], window.class_definitions["detection"], window
    )
    started = threading.Event()

    def export(**options):
        started.set()
        deadline = time.monotonic() + 5
        while not options["cancelled"]() and time.monotonic() < deadline:
            time.sleep(0.001)
        raise ExportCancelled()

    with patch.object(export_dialog, "export_dataset", export):
        dialog.show()
        dialog.accept()
        assert started.wait(5)
        dialog.close()
        assert dialog.worker is not None
        assert dialog.isVisible()
        assert not dialog.cancel_button.isEnabled()
        wait_export(dialog, app)
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    assert not dialog.output_path.exists()
    dialog.deleteLater()


def test_invalid_path_and_worker_error_allow_retry(window, app, tmp_path):
    source = cloud(tmp_path / "frame.bin")
    frame = ExportFrame(source, source.with_suffix(".cuboids.json"), ())
    dialog = export_dialog.PointCloudExportDialog(
        [frame], window.class_definitions["detection"], window
    )
    with patch.object(QtWidgets.QMessageBox, "warning") as warning:
        dialog.path_input.setText(str(tmp_path / "bad.label"))
        dialog.accept()
        assert dialog.worker is None
        warning.assert_called_once()
        dialog.path_input.setText(str(tmp_path / "export.zip"))
        with patch.object(
            export_dialog, "export_dataset", side_effect=OSError("Disk full")
        ):
            dialog.accept()
            wait_export(dialog, app)
        assert warning.call_args.args[-1] == "Disk full"
        assert dialog.ok_button.isEnabled()
        assert dialog.path_input.isEnabled()
        dialog.accept()
        wait_export(dialog, app)
    assert dialog.export_result == (1, 0)
    dialog.deleteLater()
