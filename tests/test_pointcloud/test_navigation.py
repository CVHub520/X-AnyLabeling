from unittest.mock import patch

import numpy as np
from PyQt6 import QtCore, QtWidgets

from anylabeling.views.labeling.widgets import pointcloud_dialog as module

from .test_dialog import app, wait_load, window


def test_navigation_reuses_frame_rows_and_delays_progress(
    window, app, tmp_path
):
    files = []
    for index in range(3):
        path = tmp_path / f"{index}.bin"
        np.arange(32, dtype="<f4").tofile(path)
        files.append(path)
    window.open_paths(files)
    assert window._progress.minimumDuration() == 500
    assert not window._progress.isVisible()
    wait_load(window, app)
    first_row = window.file_list.item(0)
    with patch.object(
        window, "_rebuild_files", wraps=window._rebuild_files
    ) as rebuild:
        with patch.object(
            window.viewport,
            "validate_rendering",
            side_effect=lambda: _assert_loading(window),
        ):
            window.navigate(1)
            wait_load(window, app)
        window.navigate(1)
        wait_load(window, app)
        assert rebuild.call_count == 0
    assert window.file_list.item(0) is first_row
    assert window.frame_index == 2
    assert window.frame_progress.text() == "3/3"
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    assert not window.findChildren(module.FrameLoader)
    assert not window.findChildren(QtWidgets.QProgressDialog)


def _assert_loading(window):
    assert window._worker is not None
    assert not window.centralWidget().isEnabled()


def test_rapid_navigation_keeps_pending_target(window, app, tmp_path):
    files = []
    for index in range(5):
        path = tmp_path / f"{index}.bin"
        np.arange(32, dtype="<f4").tofile(path)
        files.append(path)
    window.open_paths(files)
    wait_load(window, app)
    window.navigate(1)
    worker = window._worker
    window.navigate(1)
    window.navigate(1)
    window.navigate(-1)
    assert window._queued_frame == 2
    assert not worker.isInterruptionRequested()
    wait_load(window, app)
    assert window.frame_index == 2
    assert window._queued_frame is None


def test_navigation_shortcuts_survive_loading_and_restore_focus(
    window, app, tmp_path
):
    from PyQt6 import QtTest

    files = []
    for index in range(3):
        path = tmp_path / f"{index}.bin"
        np.arange(32, dtype="<f4").tofile(path)
        files.append(path)
    window.open_paths(files)
    wait_load(window, app)
    window.show()
    window.activateWindow()
    window.file_list.setFocus()
    app.processEvents()
    QtTest.QTest.keyClick(window.file_list, QtCore.Qt.Key.Key_D)
    assert window._worker is not None
    QtTest.QTest.keyClick(window, QtCore.Qt.Key.Key_D)
    assert window._queued_frame == 2
    wait_load(window, app)
    assert window.frame_index == 2
    assert window.file_list.hasFocus()
