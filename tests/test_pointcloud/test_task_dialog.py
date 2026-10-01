from unittest.mock import patch

import numpy as np
import pytest
from PyQt6 import QtCore, QtTest, QtWidgets

from anylabeling.views.labeling.pointcloud.io import load_classes, save_classes
from anylabeling.views.labeling.pointcloud.model import ClassDefinition
from anylabeling.views.labeling.pointcloud.task_dialog import CreateTaskDialog
from .test_dialog import app, window, cloud, open_cloud, wait_load
from .test_multicamera import source

VEHICLE = ClassDefinition(1, "Vehicle", "#6496F5")
ROAD = ClassDefinition(2, "Road", "#FF00FF")


def configure(dialog, directory, task="detection", output=None):
    dialog.task_buttons[task].setChecked(True)
    dialog.add_class(VEHICLE if task == "detection" else ROAD)
    dialog.next_button.click()
    assert dialog.pages.currentIndex() == 1
    dialog.directory_input.setText(str(directory))
    if output:
        dialog.output_input.setText(str(output))
    dialog.next_button.click()
    assert dialog.pages.currentIndex() == 2, dialog.error_label.text()


def test_toolbar_has_one_task_entry(window):
    toolbar = window.findChild(QtWidgets.QToolBar, "pointcloudFileTools")
    names = [action.text() for action in toolbar.actions()]
    assert names[0] == "Create task"
    assert not {"Open file", "Open dir", "Output dir", "Camera image"} & set(
        names
    )


def test_class_edits_validation_and_task_drafts(window):
    dialog = CreateTaskDialog(window)
    dialog.next_button.click()
    assert dialog.pages.currentIndex() == 0
    assert not dialog.error_label.isHidden()
    dialog.add_class(VEHICLE)
    dialog.add_class()
    dialog.next_button.click()
    assert dialog.pages.currentIndex() == 0
    dialog.rows[1].name_input.setText("Person")
    dialog.rows[1].id_input.setValue(1)
    dialog.next_button.click()
    assert dialog.pages.currentIndex() == 0
    dialog._remove_class(dialog.rows[1])
    dialog.task_buttons["segmentation"].setChecked(True)
    assert not dialog.rows
    dialog.add_class(ROAD)
    dialog.task_buttons["detection"].setChecked(True)
    assert dialog._classes() == [VEHICLE]
    dialog.task_buttons["segmentation"].setChecked(True)
    assert dialog._classes() == [ROAD]
    dialog.next_button.click()
    assert [item.id for item in dialog.classes] == [0, 2]
    assert window.class_definitions["detection"] == []
    dialog.reject()
    dialog.deleteLater()


def test_upload_requires_confirmation_and_matching_task(window, tmp_path):
    path = tmp_path / "classes.json"
    save_classes(path, {"detection": [VEHICLE]})
    dialog = CreateTaskDialog(window)
    dialog.add_class(ROAD)
    with patch.object(
        QtWidgets.QFileDialog, "getOpenFileName", return_value=(str(path), "")
    ):
        with patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.StandardButton.Cancel,
        ):
            dialog.upload_button.click()
            assert dialog._classes() == [ROAD]
        with patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.StandardButton.Ok,
        ):
            dialog.upload_button.click()
            assert dialog._classes() == [VEHICLE]
        dialog.task_buttons["segmentation"].setChecked(True)
        dialog.upload_button.click()
        assert not dialog.error_label.isHidden()
        assert not dialog.rows
    dialog.deleteLater()


def test_back_preserves_paths_classes_and_camera_settings(window, tmp_path):
    cloud(tmp_path / "0000.bin")
    camera = source(tmp_path, 0)
    dialog = CreateTaskDialog(window)
    configure(dialog, tmp_path)
    dialog.camera_page.directory_input.setText(str(camera.directory))
    dialog.camera_page.calibration_input.setText(str(camera.calibration_path))
    dialog.back_button.click()
    assert dialog.directory_input.text() == str(tmp_path)
    dialog.back_button.click()
    assert dialog._classes() == [VEHICLE]
    dialog.next_button.click()
    dialog.next_button.click()
    dialog.next_button.click()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    assert dialog.configuration.output_directory == tmp_path
    assert dialog.configuration.cameras[0].directory == camera.directory
    assert window.document is None
    dialog.deleteLater()


@pytest.mark.parametrize(
    "key", [QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Escape]
)
def test_camera_page_keyboard_finishes_or_cancels_wizard(
    window, app, tmp_path, key
):
    cloud(tmp_path / "0000.bin")
    dialog = CreateTaskDialog(window)
    configure(dialog, tmp_path)
    dialog.show()
    app.processEvents()
    field = dialog.camera_page.directory_input
    field.setFocus()
    QtTest.QTest.keyClick(field, key)
    assert not dialog.isVisible()
    assert (dialog.configuration is not None) == (
        key == QtCore.Qt.Key.Key_Return
    )
    dialog.deleteLater()


def test_camera_browse_buttons_align_with_wizard_footer(window, app, tmp_path):
    cloud(tmp_path / "0000.bin")
    dialog = CreateTaskDialog(window)
    configure(dialog, tmp_path)
    dialog.show()
    app.processEvents()

    def right(widget):
        return widget.mapTo(dialog, QtCore.QPoint()).x() + widget.width()

    for count in range(4):
        if count:
            dialog.camera_page.add_camera()
            QtTest.QTest.qWait(1)
        for entry in dialog.camera_page.entries:
            for row in (1, 2):
                assert right(
                    entry.layout().itemAtPosition(row, 1).widget()
                ) == right(dialog.next_button)
    assert dialog.camera_page.sources_scroll.verticalScrollBar().maximum() > 0
    dialog.reject()
    dialog.deleteLater()


def test_invalid_directories_and_calibration_keep_current_step(
    window, tmp_path
):
    dialog = CreateTaskDialog(window)
    dialog.add_class(VEHICLE)
    dialog.next_button.click()
    dialog.next_button.click()
    assert dialog.pages.currentIndex() == 1
    dialog.directory_input.setText(str(tmp_path))
    dialog.next_button.click()
    assert dialog.pages.currentIndex() == 1
    path = cloud(tmp_path / "0000.bin")
    dialog.output_input.setText(str(path))
    dialog.next_button.click()
    assert dialog.pages.currentIndex() == 1
    dialog.output_input.clear()
    dialog.next_button.click()
    camera = source(tmp_path, 0)
    dialog.camera_page.directory_input.setText(str(camera.directory))
    dialog.camera_page.calibration_input.setText(
        str(tmp_path / "missing.json")
    )
    dialog.next_button.click()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    assert not dialog.error_label.isHidden()
    assert dialog.configuration is None
    dialog.deleteLater()


@pytest.mark.parametrize("task", ["detection", "segmentation"])
@pytest.mark.parametrize("custom_output", [False, True])
def test_create_loads_task_without_saving_classes(
    window, app, tmp_path, task, custom_output
):
    directory = tmp_path / "sequence" / "velodyne"
    directory.mkdir(parents=True)
    files = [cloud(directory / f"{index:04d}.bin") for index in range(2)]
    existing_classes = directory / "pointcloud_classes.json"
    existing_bytes = existing_classes.read_bytes()
    output = tmp_path / "results" if custom_output else None
    window.color_mode.setCurrentIndex(window.color_mode.findData("instance"))

    def finish(dialog):
        configure(dialog, directory.parent, task, output)
        dialog.next_button.click()
        return dialog.result()

    with patch.object(CreateTaskDialog, "exec", finish):
        window.create_task_action.trigger()
    wait_load(window, app)
    assert not window._errors
    assert window.files == files
    assert window.detection.enabled == (task == "detection")
    mode = "rgb" if task == "detection" else "semantic"
    assert window.color_mode.currentData() == mode
    assert window.render_actions[mode].isChecked()
    selected = 0 if task == "detection" else 1
    assert window.task_type == task
    assert window.sidebar_tabs.isTabEnabled(selected)
    assert not window.sidebar_tabs.isTabEnabled(1 - selected)
    window.sidebar_tabs.setCurrentIndex(1 - selected)
    assert window.sidebar_tabs.currentIndex() == selected
    assert window.detection.enabled == (task == "detection")
    assert window.camera_panel.sources == []
    destination = output or directory
    assert window.label_directory == destination
    assert window.config_path is None
    expected = VEHICLE if task == "detection" else ROAD
    assert expected in window.class_definitions[task]
    assert window._autosave()
    assert existing_classes.read_bytes() == existing_bytes
    if custom_output:
        assert not (destination / "pointcloud_classes.json").exists()
    assert not window.config_dirty
    window.color_mode.setCurrentIndex(window.color_mode.findData("rgb"))
    window.navigate(1)
    wait_load(window, app)
    assert window.color_mode.currentData() == "rgb"
    assert window.document.frame.label_path == destination / "0001.label"
    assert expected in window.class_definitions[task]
    if task == "segmentation":
        assert window.class_definitions["segmentation"][0].id == 0


@pytest.mark.parametrize("task", ["detection", "segmentation"])
def test_save_labels_exports_current_classes_and_keeps_setup(
    window, tmp_path, task
):
    dialog = CreateTaskDialog(window)
    dialog.task_buttons[task].setChecked(True)
    definition = VEHICLE if task == "detection" else ROAD
    dialog.add_class(definition)
    target = tmp_path / "custom" / "labels.json"
    with patch.object(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        return_value=(str(target), ""),
    ):
        dialog.save_classes_button.click()
    expected = [definition]
    if task == "segmentation":
        expected = [dialog.unlabeled] + expected
    assert load_classes(target) == {task: expected}
    assert dialog.pages.currentIndex() == 0
    assert not dialog.save_classes_button.isHidden()
    assert window.config_path is None
    dialog.next_button.click()
    assert dialog.save_classes_button.isHidden()
    dialog.back_button.click()
    assert not dialog.save_classes_button.isHidden()
    dialog.deleteLater()


def test_save_labels_validation_cancel_and_write_failure(window, tmp_path):
    dialog = CreateTaskDialog(window)
    with patch.object(QtWidgets.QFileDialog, "getSaveFileName") as choose:
        dialog.save_classes_button.click()
        choose.assert_not_called()
    assert not dialog.error_label.isHidden()
    dialog.add_class(VEHICLE)
    with patch.object(
        QtWidgets.QFileDialog, "getSaveFileName", return_value=("", "")
    ):
        dialog.save_classes_button.click()
    target = tmp_path / "labels.json"
    with (
        patch.object(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            return_value=(str(target), ""),
        ),
        patch(
            "anylabeling.views.labeling.pointcloud.task_dialog.save_classes",
            side_effect=OSError("Write failed"),
        ),
    ):
        dialog.save_classes_button.click()
    assert dialog.error_label.text() == "Write failed"
    assert not target.exists()
    assert dialog._classes() == [VEHICLE]
    assert dialog.pages.currentIndex() == 0
    dialog.deleteLater()


def test_fixed_wizard_size_alignment_and_camera_removal(window, app, tmp_path):
    cloud(tmp_path / "0000.bin")
    dialog = CreateTaskDialog(window)
    dialog.show()
    app.processEvents()
    size = dialog.size()
    assert dialog.minimumSize() == size == dialog.maximumSize()
    assert not any(
        button.isVisible() and button.text() == "Cancel"
        for button in dialog.findChildren(QtWidgets.QPushButton)
    )
    dialog.accept()
    assert not dialog.error_label.isHidden()
    for number in range(1, 32):
        dialog.add_class(ClassDefinition(number, f"Class {number}", "#6496F5"))
    QtTest.QTest.qWait(10)
    assert dialog.classes_scroll.verticalScrollBar().maximum() > 0
    det = dialog.task_buttons["detection"]
    seg = dialog.task_buttons["segmentation"]
    group_center = (
        det.mapTo(dialog, QtCore.QPoint()).x()
        + seg.mapTo(dialog, QtCore.QPoint()).x()
        + seg.width()
    ) / 2
    button_center = (
        dialog.add_class_button.mapTo(dialog, QtCore.QPoint()).x()
        + dialog.add_class_button.width() / 2
    )
    assert abs(group_center - button_center) <= 1
    assert dialog.size() == size
    dialog.accept()
    assert dialog.progress.step == 1
    dialog.directory_input.setText(str(tmp_path))
    dialog.accept()
    assert dialog.progress.step == 2
    cameras = [source(tmp_path, index) for index in range(4)]
    page = dialog.camera_page
    for camera in cameras:
        page.add_camera(camera)
    page._refresh_previews()
    QtTest.QTest.qWait(10)
    assert dialog.size() == size
    assert page.sources_scroll.verticalScrollBar().maximum() > 0
    page._step_preview(1)
    assert page.preview_caption.text() == "image_01"
    page.entries[2].remove_button.click()
    assert page.preview_caption.text() != "image_01"
    assert dialog.size() == size
    for entry in page.entries[:]:
        entry.remove_button.click()
    assert not page.entries
    assert page.preview.camera_count == 0
    assert page.preview.image_item.pixmap().isNull()
    page.add_camera(cameras[0])
    assert page.directory_input.text() == str(cameras[0].directory)
    page.entries[0].remove_button.click()
    dialog.back_button.click()
    assert dialog.progress.step == 1
    dialog.next_button.click()
    dialog._error("A long validation message " * 200)
    QtTest.QTest.qWait(10)
    assert dialog.size() == size
    dialog.accept()
    assert dialog.configuration.cameras == ()
    dialog.deleteLater()


@pytest.mark.parametrize("task", ["detection", "segmentation"])
def test_created_task_classes_cannot_be_edited_or_deleted(
    window, app, tmp_path, task
):
    cloud(tmp_path / "0000.bin")

    def finish(dialog):
        configure(dialog, tmp_path, task)
        dialog.accept()
        return dialog.result()

    with patch.object(CreateTaskDialog, "exec", finish):
        window.create_task()
    wait_load(window, app)
    classes = {
        name: list(values) for name, values in window.class_definitions.items()
    }
    listing = (
        window.detection.labels if task == "detection" else window.class_list
    )
    assert not listing.allow_remove
    with patch.object(QtWidgets.QFileDialog, "getOpenFileName") as upload:
        window._import_classes()
        upload.assert_not_called()
    window._edit_class(False)
    window._edit_class(True)
    if task == "detection":
        window.detection._delete_label(listing.item(0))
    else:
        window._remove_class(listing.item(1))
    assert not window._save_config(choose_path=True)
    assert window.class_definitions == classes


@pytest.mark.parametrize("failure", ["cancel", "invalid_frame", "cancel_load"])
def test_cancel_or_failed_creation_preserves_previous_task(
    window, app, tmp_path, failure
):
    previous_path = cloud(tmp_path / "old.bin")
    open_cloud(window, app, previous_path)
    camera = source(tmp_path, 0, ("old",))
    window.camera_panel.configure_sources([camera])
    previous = window.document
    classes = {
        task: list(values) for task, values in window.class_definitions.items()
    }
    config_path = window.config_path
    directory = tmp_path / "new"
    directory.mkdir()
    if failure == "invalid_frame":
        (directory / "broken.bin").write_bytes(b"invalid")
    else:
        cloud(directory / "0000.bin")

    def finish(dialog):
        configure(dialog, directory)
        if failure == "cancel":
            dialog.reject()
        else:
            dialog.accept()
        return dialog.result()

    with patch.object(CreateTaskDialog, "exec", finish):
        window.create_task_action.trigger()
    if failure == "cancel_load":
        window._worker.requestInterruption()
    wait_load(window, app)
    assert window.document is previous
    assert window.class_definitions == classes
    assert window.config_path == config_path
    assert window.camera_panel.directory == camera.directory
    assert window.files == [previous_path]
    assert window.sidebar_tabs.currentIndex() == 1


@pytest.mark.parametrize("task", ["detection", "segmentation"])
@pytest.mark.parametrize("previous_count", [0, 2, 16])
@pytest.mark.parametrize("overlay_mode", ["depth", "cloud"])
def test_create_calibrated_task_initializes_point_display(
    window, app, tmp_path, task, previous_count, overlay_mode
):
    if previous_count:
        open_cloud(
            window, app, cloud(tmp_path / "previous.bin", previous_count)
        )
    directory = tmp_path / "new"
    directory.mkdir()
    cloud(directory / "0000.bin")
    camera = source(directory, 0)
    panel = window.camera_panel
    panel.overlay_color_buttons[overlay_mode].setChecked(True)

    def finish(dialog):
        configure(dialog, directory, task)
        dialog.camera_page.add_camera(camera)
        dialog.accept()
        return dialog.result()

    with patch.object(CreateTaskDialog, "exec", finish):
        window.create_task()
    wait_load(window, app)
    assert not window._errors
    assert len(window._visible) == 8
    assert len(window.viewport._colors) == 8
    assert len(panel._indices) > 0
    np.testing.assert_array_equal(window._visible, np.ones(8, dtype=bool))
    overlay = panel.view.overlay_item.pixmap().toImage()
    assert not overlay.isNull()
    assert any(
        overlay.pixelColor(x, y).alpha() > 0
        for x in range(overlay.width())
        for y in range(overlay.height())
    )


def test_create_applies_multiple_cameras_and_replaces_previous_sources(
    window, app, tmp_path
):
    cloud(tmp_path / "0000.bin")
    cameras = [source(tmp_path, index) for index in range(2)]

    def finish(dialog):
        configure(dialog, tmp_path)
        for camera in cameras:
            dialog.camera_page.add_camera(camera)
        dialog.accept()
        return dialog.result()

    with patch.object(CreateTaskDialog, "exec", finish):
        window.create_task_action.trigger()
    wait_load(window, app)
    assert [item.name for item in window.camera_panel.sources] == [
        "image_00",
        "image_01",
    ]
    assert not window.camera_panel._image.isNull()
    assert window.camera_panel_action.isChecked()

    def without_camera(dialog):
        configure(dialog, tmp_path, "segmentation")
        dialog.accept()
        return dialog.result()

    with patch.object(CreateTaskDialog, "exec", without_camera):
        window.create_task_action.trigger()
    wait_load(window, app)
    assert window.camera_panel.sources == []
    assert window.camera_panel.isHidden()
