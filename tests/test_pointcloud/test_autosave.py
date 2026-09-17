from pathlib import Path
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PyQt6 import QtCore, QtTest, QtWidgets

from anylabeling.views.labeling.pointcloud.io import load_classes, load_frame
from anylabeling.views.labeling.pointcloud.model import ClassDefinition
from anylabeling.views.labeling.widgets import pointcloud_dialog as module

from . import test_dialog as dialog_tests
from .test_dialog import open_cloud, wait_load

app = dialog_tests.app


@pytest.fixture
def autosave_window(app, tmp_path):
    settings_class = QtCore.QSettings
    with patch.object(
        module.QtCore,
        "QSettings",
        lambda *args: settings_class(
            str(tmp_path / "autosave.ini"), settings_class.Format.IniFormat
        ),
    ):
        window = module.PointCloudDialog()
    window._errors = []
    window._error = window._errors.append
    window._confirm = Mock(return_value=True)
    app.processEvents()
    yield window
    window._autosave_timer.stop()
    if window._worker is not None:
        window._worker.requestInterruption()
        wait_load(window, app)
    window.close_after_approval()
    app.processEvents()


def _cloud(path, labels=None):
    np.arange(16, dtype=np.float32).reshape(-1, 4).tofile(path)
    if labels is not None:
        np.asarray(labels, dtype="<u4").tofile(path.with_suffix(".label"))
    return path.resolve()


def _wait_for(predicate, app):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        QtCore.QThread.msleep(5)
    assert predicate(), "Automatic save did not finish"


def _edit(window, indices=(0,), semantic=10):
    window.document.assign_semantic(indices, semantic, overwrite=True)
    window._refresh()
    window._schedule_autosave()


def test_annotation_tool_displays_committed_semantic_color(
    autosave_window, app, tmp_path
):
    window = autosave_window
    source = dialog_tests.cloud(tmp_path / "brush.bin")
    open_cloud(window, app, source)
    dialog_tests.select_class(window, 10)
    window._select_tool("brush")
    assert window.color_mode.currentData() == "semantic"
    window._apply_selection(np.array([0, 1]))
    colors = window.viewport._colors.copy()
    np.testing.assert_allclose(colors[0, :3], np.array([100, 150, 245]) / 255)
    _wait_for(lambda: not window.document.dirty, app)
    np.testing.assert_array_equal(window.viewport._colors, colors)
    np.testing.assert_array_equal(load_frame(source).labels[:2], [10, 10])


def test_output_directory_is_reused_when_reopening_dataset(
    autosave_window, app, tmp_path
):
    window = autosave_window
    source = _cloud(tmp_path / "reopen.bin", [10, 30, 40, 0])
    open_cloud(window, app, source)
    destination = tmp_path / "output"
    _change_output(window, destination)
    _edit(window, [3], 70)
    assert window._autosave()
    expected = window.document.labels.copy()
    window.label_directory = None
    window.label_overrides.clear()
    window.open_paths([source])
    wait_load(window, app)
    assert window.label_directory == destination
    assert window.document.frame.label_path == destination / "reopen.label"
    np.testing.assert_array_equal(window.document.labels, expected)


def _change_output(window, directory):
    directory.mkdir(exist_ok=True)
    with patch.object(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        return_value=str(directory),
    ):
        window._change_output_directory()


@pytest.fixture
def sequence(autosave_window, app, tmp_path):
    window = autosave_window
    first = _cloud(tmp_path / "1.bin", [10, 30, 40, 70])
    second = _cloud(tmp_path / "2.bin", [(7 << 16) | 10, 0xFFFFEA60, 30, 0])
    window.open_paths([first, second])
    wait_load(window, app)
    assert window.document is not None, window._errors
    return window, first, second


def test_autosave_debounces_edits_and_preserves_complete_frame(
    autosave_window, app, tmp_path
):
    window = autosave_window
    source = _cloud(tmp_path / "scan.bin")
    original = source.read_bytes()
    open_cloud(window, app, source)
    target = source.with_suffix(".label")
    with patch.object(module, "save_labels", wraps=module.save_labels) as save:
        _edit(window, [0])
        QtTest.QTest.qWait(200)
        _edit(window, [1], 30)
        QtTest.QTest.qWait(200)
        assert not target.exists()
        assert window.document.dirty
        _wait_for(lambda: not window.document.dirty, app)
        save.assert_called_once()
        QtTest.QTest.qWait(400)
        save.assert_called_once()
    np.testing.assert_array_equal(load_frame(source).labels, [10, 30, 0, 0])
    assert source.read_bytes() == original
    assert window.document.can_undo
    assert not window._errors


def test_active_selection_defers_autosave_without_cancelling_it(
    autosave_window, app, tmp_path
):
    window = autosave_window
    source = _cloud(tmp_path / "scan.bin")
    open_cloud(window, app, source)
    _edit(window)
    window.viewport._error = None
    window.viewport.set_tool("brush")
    window.viewport.set_depth_mode("through")
    assert window.viewport._begin_selection()

    QtTest.QTest.qWait(450)

    assert window.viewport.selection_active
    assert window.document.dirty
    assert not source.with_suffix(".label").exists()
    window.viewport.cancel_selection()
    _wait_for(lambda: not window.document.dirty, app)
    np.testing.assert_array_equal(load_frame(source).labels, [10, 0, 0, 0])


def test_undo_and_redo_automatically_save_restored_labels(
    autosave_window, app, tmp_path
):
    window = autosave_window
    source = _cloud(tmp_path / "scan.bin")
    open_cloud(window, app, source)
    _edit(window)
    _wait_for(lambda: not window.document.dirty, app)

    window.undo()
    assert window.document.dirty
    _wait_for(lambda: not window.document.dirty, app)
    np.testing.assert_array_equal(load_frame(source).labels, [0, 0, 0, 0])
    assert window.document.can_redo
    window.redo()
    _wait_for(lambda: not window.document.dirty, app)
    np.testing.assert_array_equal(load_frame(source).labels, [10, 0, 0, 0])


def test_leaving_flushes_pending_autosave_without_save_prompt(
    autosave_window, app, tmp_path
):
    window = autosave_window
    source = _cloud(tmp_path / "scan.bin")
    open_cloud(window, app, source)
    _edit(window)
    with patch.object(QtWidgets.QMessageBox, "warning") as warning:
        assert window._leave_decision() == "continue"
        warning.assert_not_called()
    assert not window.document.dirty
    assert not window._autosave_timer.isActive()
    np.testing.assert_array_equal(load_frame(source).labels, [10, 0, 0, 0])


@pytest.mark.parametrize(
    "choice, expected",
    [
        (QtWidgets.QMessageBox.StandardButton.Cancel, "cancel"),
        (QtWidgets.QMessageBox.StandardButton.Discard, "discard"),
        (QtWidgets.QMessageBox.StandardButton.Save, "cancel"),
    ],
)
def test_failed_autosave_retains_labels_and_original_leave_protection(
    sequence, choice, expected
):
    window, source, _ = sequence
    target = source.with_suffix(".label")
    previous = target.read_bytes()
    _edit(window, [0], 70)
    edited = window.document.labels.copy()
    with (
        patch.object(module, "save_labels", side_effect=OSError("disk full")),
        patch.object(
            QtWidgets.QMessageBox, "warning", return_value=choice
        ) as warning,
    ):
        assert window._leave_decision() == expected
        warning.assert_called_once()
    assert target.read_bytes() == previous
    np.testing.assert_array_equal(window.document.labels, edited)
    assert window.document.dirty
    assert not window._autosave_timer.isActive()
    assert window._errors


def test_class_only_autosave_uses_settings_directory_without_file_dialog(
    autosave_window, app
):
    window = autosave_window
    window.classes.append(ClassDefinition(10, "Vehicle", "#6496F5"))
    target = (
        Path(window.settings.fileName()).parent / "pointcloud_classes.json"
    )
    with patch.object(QtWidgets.QFileDialog, "getSaveFileName") as choose:
        window._schedule_autosave()
        _wait_for(lambda: not window.config_dirty, app)
        choose.assert_not_called()
    assert window.document is None
    assert window.config_path == target
    assert load_classes(target) == window.classes


def test_partial_autosave_keeps_failed_class_configuration_dirty(
    autosave_window, app, tmp_path
):
    window = autosave_window
    source = _cloud(tmp_path / "scan.bin")
    open_cloud(window, app, source)
    window.classes.append(ClassDefinition(10, "Vehicle", "#6496F5"))
    _edit(window)
    with patch.object(
        module, "save_classes", side_effect=OSError("read-only config")
    ) as save:
        assert not window._autosave()
        QtTest.QTest.qWait(450)
        save.assert_called_once()
    assert not window.document.dirty
    assert window.config_dirty
    assert not window._autosave_timer.isActive()
    np.testing.assert_array_equal(load_frame(source).labels, [10, 0, 0, 0])
    assert any("read-only config" in error for error in window._errors)


def test_output_directory_preserves_source_labels_and_missing_output_reload(
    sequence, app, tmp_path
):
    window, first, second = sequence
    first_labels = first.with_suffix(".label").read_bytes()
    second_labels = second.with_suffix(".label").read_bytes()
    _edit(window, [0], 70)
    current = window.document.labels.copy()
    output = tmp_path / "output"

    _change_output(window, output)

    assert window.label_directory == output
    assert window.document.frame.label_path == output / "1.label"
    assert window.document.frame.label_exists
    assert not window.document.dirty
    np.testing.assert_array_equal(
        load_frame(first, output / "1.label").labels, current
    )
    assert first.with_suffix(".label").read_bytes() == first_labels
    window.navigate(1)
    wait_load(window, app)
    assert window.document.frame.path == second
    assert window.document.frame.label_path == output / "2.label"
    assert not window.document.frame.label_exists
    assert not (output / "2.label").exists()
    assert window.document.labels.tobytes() == second_labels
    window.reload_frame()
    wait_load(window, app)
    assert window.document.labels.tobytes() == second_labels
    assert not window.document.frame.label_exists
    _edit(window, [2], 70)
    _wait_for(lambda: not window.document.dirty, app)
    assert (output / "2.label").exists()
    assert second.with_suffix(".label").read_bytes() == second_labels
    window.navigate(-1)
    wait_load(window, app)
    np.testing.assert_array_equal(window.document.labels, current)
    assert not window._errors


def test_existing_output_labels_take_priority_over_source(
    sequence, app, tmp_path
):
    window, _, second = sequence
    output = tmp_path / "output"
    _change_output(window, output)
    expected = np.array([70, 70, 70, 70], dtype="<u4")
    expected.tofile(output / "2.label")

    window.navigate(1)
    wait_load(window, app)

    assert window.document.frame.path == second
    assert window.document.frame.label_exists
    np.testing.assert_array_equal(window.document.labels, expected)


@pytest.mark.parametrize("failure", ["write", "overwrite_cancel"])
def test_failed_output_directory_change_retains_original_session(
    sequence, tmp_path, failure
):
    window, first, _ = sequence
    _edit(window, [0], 70)
    old_target = window.document.frame.label_path
    old_labels = old_target.read_bytes()
    old_overrides = dict(window.label_overrides)
    output = tmp_path / "output"
    output.mkdir()
    target = output / "1.label"
    target.write_bytes(np.array([30] * 4, dtype="<u4").tobytes())
    old_output = target.read_bytes()
    window._confirm.return_value = failure != "overwrite_cancel"
    with patch.object(
        module, "save_labels", side_effect=OSError("output is read-only")
    ):
        _change_output(window, output)
    assert window.label_directory is None
    assert window.label_overrides == old_overrides
    assert window.document.frame.path == first
    assert window.document.frame.label_path == old_target
    assert window.document.dirty
    assert old_target.read_bytes() == old_labels
    assert target.read_bytes() == old_output


@pytest.mark.parametrize("bad_output", ["truncated", "broken_symlink"])
def test_existing_invalid_output_never_falls_back_to_source(
    sequence, app, tmp_path, bad_output
):
    window, first, _ = sequence
    output = tmp_path / "output"
    _change_output(window, output)
    target = output / "2.label"
    if bad_output == "truncated":
        target.write_bytes(b"invalid")
    else:
        target.symlink_to(output / "missing.label")
    original = window.document

    window.navigate(1)
    wait_load(window, app)

    assert window.document is original
    assert window.document.frame.path == first
    assert window._errors


def test_source_discovery_failure_during_output_fallback_preserves_frame(
    sequence, tmp_path
):
    window, first, second = sequence
    _change_output(window, tmp_path / "output")
    original = module.label_candidates

    def candidates(path, directory=None):
        if path == second and directory is None:
            raise PermissionError("source label directory is unreadable")
        return original(path, directory)

    with patch.object(module, "label_candidates", side_effect=candidates):
        window.navigate(1)
    assert window.document.frame.path == first
    assert window._worker is None
    assert any("unreadable" in error for error in window._errors)


def test_output_created_during_fallback_load_requires_overwrite_confirmation(
    sequence, app, tmp_path
):
    window, _, second = sequence
    output = tmp_path / "output"
    _change_output(window, output)
    target = output / "2.label"
    external = np.array([70] * 4, dtype="<u4").tobytes()
    original = module.load_frame

    def load_then_create(path, label_path):
        frame = original(path, label_path)
        if path == second:
            target.write_bytes(external)
        return frame

    with patch.object(module, "load_frame", side_effect=load_then_create):
        window.navigate(1)
        wait_load(window, app)
    assert window.document.frame.path == second
    assert not window.document.frame.label_exists
    window._confirm.return_value = False
    window._confirm.reset_mock()
    _edit(window, [2], 40)

    assert not window._autosave()

    window._confirm.assert_called_once()
    assert window.document.dirty
    assert target.read_bytes() == external


@pytest.mark.parametrize("cancel_at", ["file_dialog", "renamed_confirmation"])
def test_cancelled_save_as_resumes_pending_autosave(
    sequence, app, tmp_path, cancel_at
):
    window, first, _ = sequence
    _edit(window, [0], 70)
    path = (
        "" if cancel_at == "file_dialog" else str(tmp_path / "renamed.label")
    )
    window._confirm.return_value = False
    with patch.object(
        QtWidgets.QFileDialog, "getSaveFileName", return_value=(path, "")
    ):
        assert not window.save_as()
    assert window._autosave_timer.isActive()
    _wait_for(lambda: not window.document.dirty, app)
    assert load_frame(first).labels[0] == 70
    assert not (tmp_path / "renamed.label").exists()


def test_save_as_keeps_global_output_directory_and_only_binds_current_frame(
    sequence, app, tmp_path
):
    window, first, second = sequence
    output = tmp_path / "output"
    _change_output(window, output)
    alternate = tmp_path / "alternate" / "1.label"
    _edit(window, [0], 70)
    with patch.object(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        return_value=(str(alternate), ""),
    ):
        assert window.save_as()
    assert window.label_directory == output
    assert window.document.frame.label_path == alternate
    assert window.save_as_action.shortcut().isEmpty()
    window.navigate(1)
    wait_load(window, app)
    assert window.document.frame.path == second
    assert window.document.frame.label_path == output / "2.label"
    window.navigate(-1)
    wait_load(window, app)
    assert window.document.frame.path == first
    assert window.document.frame.label_path == alternate
    assert window.document.labels[0] == 70
