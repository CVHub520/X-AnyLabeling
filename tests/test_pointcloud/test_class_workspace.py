from unittest.mock import Mock, patch

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets

from anylabeling.views.labeling.pointcloud.cuboid import Cuboid
from anylabeling.views.labeling.pointcloud.io import load_classes, save_classes
from anylabeling.views.labeling.pointcloud.model import ClassDefinition
from anylabeling.views.labeling.widgets import pointcloud_dialog as module

from .test_dialog import app, cloud, open_cloud, window
from .test_detection_panel import item_with_id


@pytest.fixture
def independent(window, app, tmp_path):
    open_cloud(window, app, cloud(tmp_path / "scan.bin"))
    window.class_definitions = {
        "detection": [ClassDefinition(10, "Car", "#112233")],
        "segmentation": [
            ClassDefinition(0, "Unlabeled", "#808080"),
            ClassDefinition(10, "Road", "#AABBCC"),
        ],
    }
    window.document.assign_semantic([0, 1], 10)
    window.document.set_cuboid(Cuboid(1, 10, (0, 0, 0), (2, 2, 2)))
    window._refresh()
    assert window.save_work()
    window._confirm = Mock(return_value=True)
    return window


def test_task_panels_and_colors_use_independent_definitions(independent):
    ui = independent
    ui.sidebar_tabs.setCurrentIndex(0)
    assert "Car" in item_with_id(ui.detection.labels, 10).text()
    assert ui.viewport.class_names[10] == "Car"
    assert ui.viewport.class_colors[10] == "#112233"
    ui.sidebar_tabs.setCurrentIndex(1)
    assert "Road" in item_with_id(ui.class_list, 10).text()
    np.testing.assert_allclose(
        ui._class_palette()[10], np.array([170, 187, 204]) / 255
    )
    ui.class_definitions["detection"].append(
        ClassDefinition(20, "Person", "#123456")
    )
    ui._refresh()
    assert 20 not in [
        ui.class_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
        for i in range(ui.class_list.count())
    ]


@pytest.mark.parametrize("task", ["detection", "segmentation"])
def test_delete_class_preserves_other_task(independent, task):
    ui = independent
    ui.sidebar_tabs.setCurrentIndex(0 if task == "detection" else 1)
    other = "segmentation" if task == "detection" else "detection"
    definitions = list(ui.class_definitions[other])
    labels, boxes = ui.document.labels.copy(), ui.document.cuboids
    if task == "detection":
        ui.document.locked_classes.add(10)
        ui.detection._delete_label(item_with_id(ui.detection.labels, 10))
        np.testing.assert_array_equal(ui.document.labels, labels)
        assert not ui.document.cuboids
    else:
        ui.document.set_cuboids_locked({1}, True)
        boxes = ui.document.cuboids
        ui._remove_class(item_with_id(ui.class_list, 10))
        assert ui.document.cuboids == boxes
        assert not ui.document.labels.any()
    assert ui.class_definitions[other] == definitions
    assert all(item.id != 10 for item in ui.class_definitions[task])


@pytest.mark.parametrize("editing", [False, True])
def test_edit_detection_class_does_not_change_segmentation(
    independent, editing
):
    ui = independent
    ui.sidebar_tabs.setCurrentIndex(0)
    ui.class_definitions["segmentation"].append(
        ClassDefinition(1, "Unknown", "#010203")
    )
    segmentation = list(ui.class_definitions["segmentation"])
    ui.detection.labels.setCurrentItem(item_with_id(ui.detection.labels, 10))

    def edit(dialog):
        assert dialog.id_input.minimum() == 1
        if not editing:
            assert dialog.id_input.value() == 1
            assert dialog.name_input.text() == "Unknown"
        dialog.name_input.setText("Truck")
        dialog.color_input.setText("#123456")
        return QtWidgets.QDialog.DialogCode.Accepted

    with patch.object(module.ClassDefinitionDialog, "exec", edit):
        ui._edit_class(editing)
    assert ui.class_definitions["segmentation"] == segmentation
    assert (
        ClassDefinition(10 if editing else 1, "Truck", "#123456")
        in ui.class_definitions["detection"]
    )


@pytest.mark.parametrize("task", ["detection", "segmentation"])
def test_import_and_export_classes_only_affect_active_task(
    independent, tmp_path, task
):
    ui = independent
    ui.sidebar_tabs.setCurrentIndex(0 if task == "detection" else 1)
    other = "segmentation" if task == "detection" else "detection"
    ui.class_definitions[other].append(
        ClassDefinition(99, "Unsaved", "#456789")
    )
    preserved = list(ui.class_definitions[other])
    definitions = {
        "detection": [ClassDefinition(2, "Bus", "#223344")],
        "segmentation": [
            ClassDefinition(0, "Unlabeled", "#808080"),
            ClassDefinition(2, "Sky", "#556677"),
        ],
    }
    source = tmp_path / "import.json"
    save_classes(source, definitions)
    original = source.read_bytes()
    config_path = ui.config_path
    with patch.object(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        return_value=(str(source), ""),
    ):
        ui._import_classes()
    ui._confirm.assert_not_called()
    assert ui.class_definitions[task] == definitions[task]
    assert ui.class_definitions[other] == preserved
    assert ui.config_path == config_path
    assert ui.config_dirty
    assert ui._autosave()
    assert load_classes(config_path) == ui.class_definitions
    assert not ui.config_dirty
    assert source.read_bytes() == original
    target = tmp_path / "export.json"
    for existing in (False, True):
        if existing:
            save_classes(target, {other: definitions[other]})
        with patch.object(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            return_value=(str(target), ""),
        ):
            assert ui._save_config(choose_path=True)
        expected = {task: ui.class_definitions[task]}
        if existing:
            expected[other] = definitions[other]
        assert load_classes(target) == expected
        assert ui.config_path == config_path


def test_import_rejects_file_for_other_task(independent, tmp_path):
    ui = independent
    ui.sidebar_tabs.setCurrentIndex(0)
    before = dict(ui.class_definitions)
    source = tmp_path / "seg.json"
    save_classes(source, {"segmentation": before["segmentation"]})
    with patch.object(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        return_value=(str(source), ""),
    ):
        ui._import_classes()
    assert ui.class_definitions == before
    assert not ui.config_dirty
    assert ui._errors == [
        "This file does not contain classes for the current task."
    ]


def test_classes_save_and_reload_in_workspace(app, tmp_path):
    settings = QtCore.QSettings(
        str(tmp_path / "settings.ini"), QtCore.QSettings.Format.IniFormat
    )
    target = (
        tmp_path
        / "xanylabeling_data"
        / "pointcloud"
        / "pointcloud_classes.json"
    )
    classes = [
        ClassDefinition(0, "Unlabeled", "#808080"),
        ClassDefinition(1, "Vehicle", "#60A5FA"),
    ]
    source = tmp_path / "dataset" / "frame.bin"
    source.parent.mkdir()
    np.zeros((2, 4), dtype="<f4").tofile(source)
    legacy = source.parent / "pointcloud_classes.json"
    save_classes(
        legacy,
        {
            "segmentation": [
                classes[0],
                ClassDefinition(2, "Legacy", "#FF0000"),
            ]
        },
    )
    legacy_bytes = legacy.read_bytes()
    with (
        patch.object(module, "get_work_directory", return_value=str(tmp_path)),
        patch.object(module.QtCore, "QSettings", return_value=settings),
    ):
        window = module.PointCloudDialog()
        try:
            window.class_definitions = {
                "detection": [ClassDefinition(1, "Car", "#112233")],
                "segmentation": classes,
            }
            assert window._save_config()
            assert window.config_path == target
            assert load_classes(target) == window.class_definitions
        finally:
            window.close_after_approval()
        reopened = module.PointCloudDialog()
        try:
            open_cloud(reopened, app, source)
            assert reopened.class_definitions == window.class_definitions
            assert reopened.config_path == target
            assert legacy.read_bytes() == legacy_bytes
            custom = tmp_path / "custom.json"
            with patch.object(
                QtWidgets.QFileDialog,
                "getSaveFileName",
                return_value=(str(custom), ""),
            ):
                assert reopened._save_config(choose_path=True)
            reopened.class_definitions["segmentation"] = classes + [
                ClassDefinition(3, "Road", "#FFA500")
            ]
            assert reopened._save_config()
            assert load_classes(custom) == {
                "detection": window.class_definitions["detection"]
            }
            assert load_classes(target) == reopened.class_definitions
            assert reopened.config_path == target
        finally:
            reopened.close_after_approval()
