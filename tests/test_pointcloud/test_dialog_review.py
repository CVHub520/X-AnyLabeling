import os
import time
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets

from anylabeling.views.labeling.pointcloud.io import save_classes
from anylabeling.views.labeling.pointcloud.model import ClassDefinition
from anylabeling.views.labeling.widgets import pointcloud_dialog as module


@pytest.fixture
def review_window(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings_class = QtCore.QSettings
    with patch.object(
        module.QtCore,
        "QSettings",
        lambda *args: settings_class(
            str(tmp_path / "review.ini"), settings_class.Format.IniFormat
        ),
    ):
        window = module.PointCloudDialog()
    window._error = Mock()
    app.processEvents()
    yield window, app
    if window._worker is not None:
        window._worker.requestInterruption()
        _wait_load(window, app)
    window.close_after_approval()
    app.processEvents()


def _wait_load(window, app):
    deadline = time.monotonic() + 10
    while window._worker is not None and time.monotonic() < deadline:
        app.processEvents()
        QtCore.QThread.msleep(1)
    assert window._worker is None
    app.processEvents()


def _open(window, app, source):
    np.arange(32, dtype=np.float32).reshape(-1, 4).tofile(source)
    window.open_paths([source])
    _wait_load(window, app)
    assert window.document is not None


def test_first_save_protects_a_newly_appeared_label_target(
    review_window, tmp_path
):
    window, app = review_window
    _open(window, app, tmp_path / "1.bin")
    window.document.assign_semantic([0], 10)
    target = window.document.frame.label_path
    target.parent.mkdir(parents=True, exist_ok=True)
    other_result = np.full(8, 30, dtype="<u4").tobytes()
    target.write_bytes(other_result)
    window._confirm = Mock(return_value=False)
    with patch.object(
        QtWidgets.QMessageBox,
        "question",
        return_value=QtWidgets.QMessageBox.StandardButton.Cancel,
    ):
        assert not window.save_work()
    assert target.read_bytes() == other_result
    assert window.document.dirty


def test_merging_special_zero_semantic_instances_reports_error(
    review_window, tmp_path
):
    window, app = review_window
    labels = np.array([1 << 16, 2 << 16] + [0] * 6, dtype="<u4")
    labels.tofile(tmp_path / "1.label")
    _open(window, app, tmp_path / "1.bin")
    window.instance_list.setCurrentRow(0)
    for row in range(window.instance_list.count()):
        window.instance_list.item(row).setSelected(True)
    window._confirm = lambda text: True
    window._error.reset_mock()

    window._merge_instances()

    window._error.assert_called_once()
    np.testing.assert_array_equal(window.document.labels, labels)
    assert not window.document.dirty


def test_explicit_classes_loaded_in_empty_workspace_survive_first_cloud(
    review_window, tmp_path
):
    window, app = review_window
    config_path = tmp_path / "shared-classes.json"
    classes = list(module.DEFAULT_CLASSES)
    classes.append(ClassDefinition(10, "Custom vehicle", "#123456"))
    save_classes(config_path, classes)
    with patch.object(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        return_value=(str(config_path), ""),
    ):
        window._import_classes()
    assert window.classes == classes

    _open(window, app, tmp_path / "1.bin")

    assert window.classes == classes
    assert window.config_path == config_path
