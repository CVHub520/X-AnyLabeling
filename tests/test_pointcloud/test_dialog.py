import os
import time
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets

from anylabeling.views.labeling.pointcloud.io import load_frame, save_classes
from anylabeling.views.labeling.pointcloud.model import ClassDefinition
from anylabeling.views.labeling.widgets import pointcloud_dialog as module


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app, tmp_path):
    settings_class = QtCore.QSettings
    with patch.object(
        module.QtCore,
        "QSettings",
        lambda *args: settings_class(
            str(tmp_path / "settings.ini"), settings_class.Format.IniFormat
        ),
    ):
        widget = module.PointCloudDialog()
    widget._errors = []
    widget._error = widget._errors.append
    app.processEvents()
    yield widget
    if widget._worker:
        widget._worker.requestInterruption()
        wait_load(widget, app)
    widget.close_after_approval()
    app.processEvents()


def cloud(path, count=8):
    points = np.arange(count * 4, dtype=np.float32).reshape(-1, 4)
    points.tofile(path)
    config_path = path.parent / "pointcloud_classes.json"
    if not config_path.exists():
        save_classes(
            config_path,
            [
                ClassDefinition(0, "Unlabeled", "#808080"),
                ClassDefinition(10, "Vehicle", "#6496F5"),
                ClassDefinition(30, "Person", "#FF1E1E"),
                ClassDefinition(40, "Road", "#FF00FF"),
                ClassDefinition(70, "Vegetation", "#00AF00"),
            ],
        )
    return path.resolve()


def wait_load(window, app):
    deadline = time.monotonic() + 10
    while window._worker is not None and time.monotonic() < deadline:
        app.processEvents()
        QtCore.QThread.msleep(1)
    assert window._worker is None, "Loading did not finish"
    app.processEvents()


def open_cloud(window, app, path):
    window.open_paths([path])
    wait_load(window, app)
    assert window.document is not None, window._errors


@pytest.mark.parametrize("adjacent_exists", [False, True])
def test_open_single_ply_uses_adjacent_labels(
    window, app, tmp_path, adjacent_exists
):
    from .test_io import ply_file

    source = ply_file(tmp_path / "scan.ply", "ascii").resolve()
    adjacent = source.with_suffix(".label")
    legacy = tmp_path / "labels" / "scan.label"
    legacy.parent.mkdir()
    np.array([30, 40], dtype="<u4").tofile(legacy)
    if adjacent_exists:
        np.array([10, 70], dtype="<u4").tofile(adjacent)
    window.settings.setValue(
        window._output_directory_key(window._dataset_path(source)),
        str(legacy.parent),
    )
    window.label_overrides[source] = legacy
    with patch.object(
        module.QtWidgets.QFileDialog,
        "getOpenFileName",
        return_value=(str(source), ""),
    ):
        window.open_file()
    wait_load(window, app)

    assert window.document is not None, window._errors
    assert window.document.frame.label_path == adjacent
    assert window.label_directory is None
    np.testing.assert_array_equal(
        window.document.labels, [10, 70] if adjacent_exists else [0, 0]
    )


def select_class(window, value):
    for index in range(window.class_list.count()):
        if (
            window.class_list.item(index).data(QtCore.Qt.ItemDataRole.UserRole)
            == value
        ):
            window.class_list.setCurrentRow(index)
            return
    raise AssertionError(value)


def set_operation(window, value):
    window.operation_actions[value].trigger()


def test_open_edit_instance_save_reload_preserves_geometry(
    window, app, tmp_path
):
    source = cloud(tmp_path / "1.bin")
    original = source.read_bytes()
    open_cloud(window, app, source)
    assert not window.document.dirty
    select_class(window, 10)
    window._apply_selection(np.array([0, 1, 2, 3]))
    set_operation(window, "create")
    window._apply_selection(np.array([0, 1]))
    key = window._current_instance()
    assert key == (10, 1)
    assert window.document.instance_counts() == {key: 2}
    assert window.save_work()
    expected = window.document.labels.copy()
    window.document.clear([0])
    window.undo()
    assert not window.document.dirty
    window.reload_frame()
    wait_load(window, app)
    np.testing.assert_array_equal(window.document.labels, expected)
    assert source.read_bytes() == original
    assert not window.document.can_undo
    assert window._current_instance() is None


@pytest.mark.parametrize(
    "operation",
    ["navigate", "list", "open", "reload", "label_file", "label_directory"],
)
def test_discard_then_failed_replacement_retains_both_dirty_states(
    window, app, tmp_path, operation
):
    first = cloud(tmp_path / "1.bin")
    broken = tmp_path / "2.bin"
    broken.write_bytes(b"broken")
    window.open_paths([first, broken])
    wait_load(window, app)
    doc = window.document
    doc.assign_semantic([0], 10)
    window.classes[1] = ClassDefinition(10, "Custom", "#010203")
    classes = list(window.classes)
    window._leave_decision = lambda: "discard"
    if operation == "navigate":
        window.navigate(1)
    elif operation == "list":
        window.file_list.setCurrentRow(1)
    elif operation == "open":
        window.open_paths([broken])
    elif operation == "reload":
        first.write_bytes(b"broken")
        window.reload_frame()
    elif operation == "label_file":
        labels = tmp_path / "invalid.label"
        labels.write_bytes(b"broken")
        window._request_frame(window.files, 0, None, {first: labels})
    else:
        label_dir = tmp_path / "invalid_labels"
        label_dir.mkdir()
        (label_dir / "1.label").write_bytes(b"broken")
        window._request_frame(window.files, 0, label_dir, {})
    wait_load(window, app)
    assert window._errors
    assert window.document is doc
    assert doc.dirty and doc.can_undo
    assert window.classes == classes and window.config_dirty
    assert window.frame_index == 0
    assert window.file_list.currentRow() == 0


def test_successful_discard_restores_config_baseline_and_clears_history(
    window, app, tmp_path
):
    files = [cloud(tmp_path / f"{index}.bin") for index in (1, 2)]
    window.open_paths(files)
    wait_load(window, app)
    window.document.assign_semantic([0], 10)
    original_classes = list(window.classes)
    window.classes[1] = ClassDefinition(10, "Unsaved", "#010203")
    window._leave_decision = lambda: "discard"
    window.navigate(1)
    wait_load(window, app)
    assert window.frame_index == 1
    assert window.classes == original_classes
    assert not window.config_dirty
    assert not window.document.dirty
    assert not window.document.can_undo


def test_cancelled_async_load_retains_current_labels_config_and_history(
    window, app, tmp_path
):
    first = cloud(tmp_path / "1.bin")
    second = cloud(tmp_path / "2.bin")
    open_cloud(window, app, first)
    doc = window.document
    doc.assign_semantic([0], 10)
    window._leave_decision = lambda: "discard"
    window.open_paths([second])
    window._worker.requestInterruption()
    wait_load(window, app)
    assert window.document is doc
    assert doc.dirty and doc.can_undo
    assert window.frame_index == 0


@pytest.mark.parametrize(
    "decision",
    [
        QtWidgets.QMessageBox.StandardButton.Cancel,
        QtWidgets.QMessageBox.StandardButton.Save,
    ],
)
def test_leave_cancel_or_save_failure_cannot_replace_current_frame(
    window, app, tmp_path, decision
):
    first, second = [cloud(tmp_path / f"{index}.bin") for index in (1, 2)]
    open_cloud(window, app, first)
    doc = window.document
    doc.assign_semantic([0], 10)
    with (
        patch.object(QtWidgets.QMessageBox, "warning", return_value=decision),
        patch.object(
            module,
            "save_labels",
            side_effect=OSError("injected write failure"),
        ),
    ):
        window.open_paths([second])
        assert not window.can_close()
    assert window._worker is None
    assert window.document is doc and doc.dirty


def test_discard_close_approval_does_not_destroy_or_mutate_pending_work(
    window, app, tmp_path
):
    open_cloud(window, app, cloud(tmp_path / "1.bin"))
    doc = window.document
    doc.assign_semantic([0], 10)
    with (
        patch.object(
            QtWidgets.QMessageBox,
            "warning",
            return_value=QtWidgets.QMessageBox.StandardButton.Discard,
        ),
        patch.object(
            module, "save_labels", side_effect=OSError("injected save failure")
        ),
    ):
        assert window.can_close()
    assert window.document is doc
    assert doc.dirty and doc.can_undo


def test_partial_save_keeps_only_config_dirty(window, app, tmp_path):
    open_cloud(window, app, cloud(tmp_path / "1.bin"))
    window.document.assign_semantic([0], 10)
    window.classes[1] = ClassDefinition(10, "Custom", "#010203")
    with patch.object(
        module, "save_classes", side_effect=OSError("injected config failure")
    ):
        assert not window.save_work()
    assert not window.document.dirty
    assert window.config_dirty
    np.testing.assert_array_equal(
        load_frame(
            window.document.frame.path, window.document.frame.label_path
        ).labels,
        window.document.labels,
    )
    assert window.save_work()
    assert not window.config_dirty


def test_save_as_affects_only_current_frame_and_survives_navigation(
    window, app, tmp_path
):
    files = [cloud(tmp_path / f"{index}.bin") for index in (1, 2)]
    window.open_paths(files)
    wait_load(window, app)
    window.document.assign_semantic([0], 10)
    target = tmp_path / "results" / "1.label"
    with patch.object(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        return_value=(str(target), ""),
    ):
        assert window.save_as()
    window.navigate(1)
    wait_load(window, app)
    assert window.document.frame.label_path != target
    window.navigate(-1)
    wait_load(window, app)
    assert window.document.frame.label_path == target
    assert window.document.semantic[0] == 10


def test_failed_save_as_preserves_previous_association_and_file(
    window, app, tmp_path
):
    open_cloud(window, app, cloud(tmp_path / "1.bin"))
    window.document.assign_semantic([0], 10)
    assert window.save_work()
    previous = window.document.frame.label_path
    content = previous.read_bytes()
    window.document.assign_semantic([1], 10)
    target = tmp_path / "results" / "1.label"
    with (
        patch.object(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            return_value=(str(target), ""),
        ),
        patch.object(
            module, "save_labels", side_effect=OSError("injected failure")
        ),
    ):
        assert not window.save_as()
    assert window.document.frame.label_path == previous
    assert previous.read_bytes() == content
    assert window.document.dirty


def test_semantic_overwrite_is_direct_and_same_class_preserves_instance(
    window, app, tmp_path
):
    open_cloud(window, app, cloud(tmp_path / "1.bin"))
    window.document.assign_semantic([0, 1], 10)
    window.document.create_instance([0, 1], 10)
    before = window.document.labels.copy()
    window._refresh()
    select_class(window, 30)
    select_class(window, 10)
    window._apply_selection(np.array([0, 1]))
    np.testing.assert_array_equal(window.document.labels, before)
    select_class(window, 30)
    window._confirm = lambda text: pytest.fail(
        "Painting must not ask for confirmation"
    )
    window._apply_selection(np.array([0, 1, 2]))
    np.testing.assert_array_equal(window.document.instance[:3], [0, 0, 0])
    window.undo()
    np.testing.assert_array_equal(window.document.labels, before)


def test_filtered_points_are_excluded_from_selection_but_whole_instance_delete_is_explicit(
    window, app, tmp_path
):
    open_cloud(window, app, cloud(tmp_path / "1.bin"))
    window.document.assign_semantic([0, 1], 10)
    key = window.document.create_instance([0, 1], 10)
    window._refresh(key)
    for row in range(window.class_list.count()):
        item = window.class_list.item(row)
        if item.data(QtCore.Qt.ItemDataRole.UserRole) == 10:
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
    select_class(window, 0)
    set_operation(window, "assign")
    window._apply_selection(np.arange(8))
    assert window.document.instance_counts() == {key: 2}
    confirmations = []
    window._confirm = lambda text: confirmations.append(text) or True
    window._delete_instance()
    assert "2 hidden" in confirmations[0]
    assert window.document.instance_counts() == {}
    np.testing.assert_array_equal(window.document.semantic[:2], [10, 10])


def test_dataset_config_is_loaded_and_unknown_ids_survive_definition_changes(
    window, app, tmp_path
):
    path = cloud(tmp_path / "1.bin")
    labels = np.array([0, 60000 | (7 << 16)] + [0] * 6, dtype="<u4")
    labels.tofile(tmp_path / "1.label")
    classes = list(module.DEFAULT_CLASSES) + [
        ClassDefinition(60000, "Custom", "#ABCDEF")
    ]
    save_classes(tmp_path / "pointcloud_classes.json", classes)
    open_cloud(window, app, path)
    assert window._class_name(60000) == "Custom"
    select_class(window, 60000)
    window._confirm = lambda text: True
    window._remove_class()
    assert window.document.labels[1] == 0
    assert window.document.dirty and window.config_dirty
    assert all(
        window.class_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
        != 60000
        for i in range(window.class_list.count())
    )
    window.undo()
    np.testing.assert_array_equal(window.document.labels, labels)
    assert window._class_name(60000) == "Unknown"


def test_failed_render_preparation_retains_previous_document(
    window, app, tmp_path
):
    first = cloud(tmp_path / "1.bin")
    second = cloud(tmp_path / "2.bin")
    open_cloud(window, app, first)
    doc = window.document
    doc.assign_semantic([0], 10)
    window._leave_decision = lambda: "discard"
    with patch.object(
        window.viewport,
        "validate_rendering",
        side_effect=RuntimeError("injected render failure"),
    ):
        window.open_paths([second])
        wait_load(window, app)
    assert window.document is doc
    assert doc.dirty and doc.can_undo
    np.testing.assert_array_equal(window.viewport._points, doc.frame.points)
    assert window._errors == ["injected render failure"]


def test_leaving_with_only_config_changes_does_not_rewrite_labels(
    window, app, tmp_path
):
    open_cloud(window, app, cloud(tmp_path / "1.bin"))
    window.classes[1] = ClassDefinition(10, "Custom", "#123456")
    with (
        patch.object(
            QtWidgets.QMessageBox,
            "warning",
            return_value=QtWidgets.QMessageBox.StandardButton.Save,
        ),
        patch.object(
            module, "save_labels", side_effect=OSError("labels are read-only")
        ) as writer,
    ):
        assert window.can_close()
    writer.assert_not_called()
    assert not window.config_dirty
    assert not window.document.frame.label_path.exists()


def test_unknown_name_defaults_and_missing_definition_deletion(
    window, app, tmp_path
):
    open_cloud(window, app, cloud(tmp_path / "unknown.bin"))
    window.document.assign_semantic([0, 1], 60000, overwrite=True)
    window._refresh()
    select_class(window, 60000)
    names = []

    def inspect(dialog):
        names.append(dialog.name_input.text())
        return QtWidgets.QDialog.DialogCode.Rejected

    with patch.object(module.ClassDefinitionDialog, "exec", inspect):
        window._edit_class(True)
        window.classes.append(ClassDefinition(60001, "Unknown", "#60A5FA"))
        window._edit_class(False)
        window.classes.append(ClassDefinition(60002, "Unknown(1)", "#60A5FA"))
        window._edit_class(False)
    assert names == ["Unknown", "Unknown(1)", "Unknown(2)"]
    window._confirm = lambda text: True
    window._remove_class()
    assert not np.any(window.document.semantic_view == 60000)
    assert all(
        window.class_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
        != 60000
        for i in range(window.class_list.count())
    )
    window.undo()
    assert np.count_nonzero(window.document.semantic_view == 60000) == 2


def test_close_hides_workspace_and_reopen_preserves_session(
    window, app, tmp_path
):
    open_cloud(window, app, cloud(tmp_path / "session.bin"))
    window.show()
    app.processEvents()
    select_class(window, 10)
    window._apply_selection(np.array([0, 1]))
    window._select_tool("polygon")
    window.viewport._scale = 42
    window.viewport._center[:] = [1, 2, 3]
    document = window.document
    labels = document.labels.copy()
    destroyed = []
    window.destroyed.connect(lambda: destroyed.append(True))
    with patch.object(
        window,
        "can_close",
        side_effect=AssertionError("Hiding must not end the session"),
    ):
        window.close()
    app.processEvents()
    assert not window.isVisible()
    assert not destroyed
    window.show()
    app.processEvents()
    assert window.isVisible()
    assert window.document is document
    assert window._current_tool() == "polygon"
    assert window.viewport._scale == 42
    np.testing.assert_array_equal(window.viewport._center, [1, 2, 3])
    np.testing.assert_array_equal(document.labels, labels)
    window.undo()
    assert not np.any(document.semantic_view)


def test_approved_application_close_deletes_hidden_workspace(app, tmp_path):
    from PyQt6 import sip

    widget = module.PointCloudDialog()
    widget.hide()
    widget.close_after_approval()
    QtCore.QCoreApplication.sendPostedEvents(
        None, QtCore.QEvent.Type.DeferredDelete
    )
    assert sip.isdeleted(widget)
