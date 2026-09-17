from unittest.mock import patch

import numpy as np
from PyQt6 import QtCore, QtWidgets

from anylabeling.views.labeling.pointcloud.io import load_classes, save_classes
from anylabeling.views.labeling.pointcloud.model import ClassDefinition
from anylabeling.views.labeling.widgets import pointcloud_dialog as module

from .test_dialog import app, open_cloud


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
    save_classes(legacy, [classes[0], ClassDefinition(2, "Legacy", "#FF0000")])
    legacy_bytes = legacy.read_bytes()
    with (
        patch.object(module, "get_work_directory", return_value=str(tmp_path)),
        patch.object(module.QtCore, "QSettings", return_value=settings),
    ):
        window = module.PointCloudDialog()
        try:
            window.classes = classes
            assert window._save_config()
            assert window.config_path == target
            assert load_classes(target) == classes
        finally:
            window.close_after_approval()
        reopened = module.PointCloudDialog()
        try:
            open_cloud(reopened, app, source)
            assert reopened.classes == classes
            assert reopened.config_path == target
            assert legacy.read_bytes() == legacy_bytes
            custom = tmp_path / "custom.json"
            with patch.object(
                QtWidgets.QFileDialog,
                "getSaveFileName",
                return_value=(str(custom), ""),
            ):
                assert reopened._save_config(choose_path=True)
            reopened.classes = classes + [
                ClassDefinition(3, "Road", "#FFA500")
            ]
            assert reopened._save_config()
            assert load_classes(custom) == reopened.classes
            assert load_classes(target) == classes
        finally:
            reopened.close_after_approval()
