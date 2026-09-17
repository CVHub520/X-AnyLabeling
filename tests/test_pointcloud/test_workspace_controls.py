from unittest.mock import patch

import numpy as np
import pytest
from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

from anylabeling.views.labeling.pointcloud.io import load_frame
from anylabeling.views.labeling.pointcloud.selection import project_points

from . import test_dialog as dialog_tests
from .test_dialog import open_cloud, select_class, set_operation, wait_load
from .test_display_updates import select_instance

app = dialog_tests.app
window = dialog_tests.window


@pytest.fixture
def controls_window(window, app, tmp_path):
    labels = np.array(
        [
            10,
            30,
            (7 << 16) | 30,
            (7 << 16) | 30,
            (8 << 16) | 30,
            (1 << 16) | 10,
            0,
            40,
        ],
        dtype="<u4",
    )
    source = dialog_tests.cloud(tmp_path / "controls.bin")
    labels.tofile(source.with_suffix(".label"))
    open_cloud(window, app, source)
    return window


@pytest.mark.parametrize("operation", ["add", "remove", "split"])
def test_instance_actions_show_and_edit_the_actual_target_class(
    controls_window, operation
):
    window = controls_window
    select_class(window, 10)
    select_instance(window, (30, 7))
    set_operation(window, operation)
    assert window._current_class() == 10
    assert window._current_instance() == (30, 7)
    before = window.document.labels.copy()

    indices = [0, 1] if operation == "add" else [0, 2]
    window._apply_selection(np.array(indices))

    expected = before.copy()
    if operation == "add":
        expected[1] = (7 << 16) | 30
    elif operation == "remove":
        expected[2] = 30
    else:
        new_key = window._current_instance()
        assert new_key[0] == 30 and new_key[1] not in (7, 8)
        expected[2] = (new_key[1] << 16) | 30
    np.testing.assert_array_equal(window.document.labels, expected)
    assert not window._errors
    window.undo()
    np.testing.assert_array_equal(window.document.labels, before)


def test_clear_removes_both_labels_and_restores_them_with_undo(
    controls_window,
):
    window = controls_window
    select_class(window, 10)
    select_instance(window, (30, 7))
    select_class(window, 0)
    set_operation(window, "assign")
    before = window.document.labels.copy()

    window._apply_selection(np.array([2]))

    expected = before.copy()
    expected[2] = 0
    np.testing.assert_array_equal(window.document.labels, expected)
    window.undo()
    np.testing.assert_array_equal(window.document.labels, before)
    set_operation(window, "assign")
    assert window._current_class() == 0


def test_class_and_instance_controls_follow_valid_selections(controls_window):
    window = controls_window
    select_class(window, 0)
    assert not window.class_list.currentItem().data(
        window.class_list.REMOVABLE_ROLE
    )
    select_class(window, 10)
    assert window.class_list.currentItem().data(
        window.class_list.REMOVABLE_ROLE
    )
    window.class_list.setCurrentRow(-1)
    assert not window.locate_action.isEnabled()
    assert not window.merge_action.isEnabled()

    select_instance(window, (30, 7))
    assert window.locate_action.isEnabled()
    assert not window.merge_action.isEnabled()
    for row in range(window.instance_list.count()):
        item = window.instance_list.item(row)
        if item.data(QtCore.Qt.ItemDataRole.UserRole) == (30, 8):
            item.setSelected(True)
    assert window.merge_action.isEnabled()
    window._confirm = lambda text: True
    window.merge_action.trigger()
    assert window._current_instance() == (30, 7)
    assert window.document.instance_counts() == {(10, 1): 1, (30, 7): 3}
    assert not window.merge_action.isEnabled()
    assert window.locate_action.isEnabled()
    window.instance_list.remove_requested.emit(
        window.instance_list.currentItem()
    )
    assert window._current_instance() is None
    assert not window.locate_action.isEnabled()
    assert not window.merge_action.isEnabled()
    window.undo()
    assert window.document.instance_counts() == {(10, 1): 1, (30, 7): 3}
    select_instance(window, (30, 7))
    assert window.locate_action.isEnabled()


def _click_vertices(viewport, vertices):
    for x, y in vertices:
        viewport._mouse_press(
            QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QPointF(x, y),
                QtCore.QPointF(x, y),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )
        )


def test_polygon_actions_follow_start_invalid_finish_cancel_and_commit(
    controls_window,
):
    window = controls_window
    viewport = window.viewport
    viewport._error = None
    select_class(window, 10)
    window._select_tool("polygon")
    window.through_action.trigger()
    screen, _, _ = project_points(
        window.document.frame.points,
        viewport._matrix(),
        *viewport._physical_size(),
    )
    center = screen[6] / viewport.devicePixelRatioF()
    vertices = center + np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]])
    before = window.document.labels.copy()
    assert not window.finish_action.isEnabled()
    assert not window.cancel_action.isEnabled()

    _click_vertices(viewport, vertices[:2])
    assert window.finish_action.isEnabled()
    assert window.cancel_action.isEnabled()
    window.finish_action.trigger()
    assert viewport.selection_active
    np.testing.assert_array_equal(window.document.labels, before)
    assert not window.document.can_undo
    window.cancel_action.trigger()
    assert not viewport.selection_active
    assert not window.finish_action.isEnabled()
    assert not window.cancel_action.isEnabled()
    assert not window.document.can_undo

    _click_vertices(viewport, vertices)
    window.finish_action.trigger()
    assert not viewport.selection_active
    assert not window.finish_action.isEnabled()
    assert not window.cancel_action.isEnabled()
    expected = before.copy()
    expected[6] = 10
    np.testing.assert_array_equal(window.document.labels, expected)
    assert window.document.can_undo
    window.undo()
    np.testing.assert_array_equal(window.document.labels, before)


def test_file_toolbar_exports_labels_and_preserves_edit_shortcut_scope(
    controls_window, app, tmp_path
):
    window = controls_window
    toolbar = window.findChild(QtWidgets.QToolBar, "pointcloudFileTools")
    assert window.save_as_action in toolbar.actions()
    assert window.output_directory_action in toolbar.actions()
    assert window.save_as_action.shortcut().isEmpty()
    for action in (
        window.undo_action,
        window.redo_action,
        window.finish_action,
        window.cancel_action,
    ):
        assert action in window.viewport.actions()
        assert (
            action.shortcutContext()
            == QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        )

    select_class(window, 10)
    window._apply_selection(np.array([6]))
    expected = window.document.labels.copy()
    destination = tmp_path / "results" / "controls.label"
    with patch.object(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        return_value=(str(destination), ""),
    ):
        window.save_as_action.trigger()
    assert destination.exists()
    assert window.document.frame.label_path == destination
    assert not window.document.dirty
    window.document.clear([0])
    window._refresh()
    assert window._autosave()
    expected[0] = 0
    frame = load_frame(window.document.frame.path, destination)
    np.testing.assert_array_equal(frame.labels, expected)

    window.reload_frame()
    wait_load(window, app)
    np.testing.assert_array_equal(window.document.labels, expected)
    assert window.document.frame.label_path == destination
    assert not window.document.can_undo
    assert not window._errors

    output_directory = tmp_path / "automatic"
    output_directory.mkdir()
    with patch.object(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        return_value=str(output_directory),
    ):
        window.output_directory_action.trigger()
    target = output_directory / "controls.label"
    assert window.label_directory == output_directory
    assert window.document.frame.label_path == target
    np.testing.assert_array_equal(np.fromfile(target, dtype="<u4"), expected)
    previous_export = destination.read_bytes()
    window.document.clear([1])
    assert window._autosave()
    expected[1] = 0
    np.testing.assert_array_equal(np.fromfile(target, dtype="<u4"), expected)
    assert destination.read_bytes() == previous_export


def test_loading_blocks_file_actions_and_restores_edit_controls(
    controls_window,
):
    window = controls_window
    select_class(window, 10)
    window._apply_selection(np.array([6]))
    select_instance(window, (30, 7))
    assert window.undo_action.isEnabled()
    assert window.locate_action.isEnabled()
    before = window.document.labels.copy()

    window._set_loading(True)

    assert not window.centralWidget().isEnabled()
    for action in (
        window.output_directory_action,
        window.save_as_action,
        window.undo_action,
        window.redo_action,
    ):
        assert not action.isEnabled()
    with (
        patch.object(window, "_save_labels") as save,
        patch.object(window, "_request_frame") as load,
        patch.object(QtWidgets.QFileDialog, "getSaveFileName") as choose_path,
        patch.object(
            QtWidgets.QFileDialog, "getExistingDirectory"
        ) as choose_directory,
    ):
        window.save_as_action.trigger()
        window.output_directory_action.trigger()
        window.undo_action.trigger()
        save.assert_not_called()
        load.assert_not_called()
        choose_path.assert_not_called()
        choose_directory.assert_not_called()
    np.testing.assert_array_equal(window.document.labels, before)
    assert window.document.dirty

    window._set_loading(False)
    window._refresh()

    assert window.centralWidget().isEnabled()
    assert window.save_as_action.isEnabled()
    assert window.output_directory_action.isEnabled()
    assert window.undo_action.isEnabled()
    assert window.locate_action.isEnabled()
    assert not window.redo_action.isEnabled()
    window.undo_action.trigger()
    assert not window.document.dirty
    assert window.redo_action.isEnabled()


def test_toolbar_groups_and_brush_wheel(controls_window):
    window = controls_window
    assert not hasattr(window, "overwrite")
    assert not hasattr(window, "brush_radius")
    window._select_tool("brush")
    set_operation(window, "create")
    assert (
        sum(action.isChecked() for action in window.operation_actions.values())
        == 1
    )
    assert window._current_tool() == "brush"
    viewport = window.viewport
    scale, radius = viewport._scale, viewport._brush_radius
    event = QtGui.QWheelEvent(
        QtCore.QPointF(100, 100),
        QtCore.QPointF(100, 100),
        QtCore.QPoint(),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.ControlModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    viewport._wheel(event)
    assert viewport._brush_radius > radius
    assert viewport._scale == scale


def test_instance_second_click_clears_target_and_restores_point_colors(
    controls_window, app
):
    window = controls_window
    window.color_mode.setCurrentIndex(window.color_mode.findData("semantic"))
    window.show()
    app.processEvents()
    listing = window.instance_list
    item = next(
        listing.item(row)
        for row in range(listing.count())
        if listing.item(row).data(QtCore.Qt.ItemDataRole.UserRole) == (30, 7)
    )
    window.sidebar_tabs.widget(0).ensureWidgetVisible(listing)
    listing.scrollToItem(item)
    app.processEvents()
    position = listing.visualItemRect(item).center()
    original_colors = window.viewport._colors.copy()
    original_labels = window.document.labels.copy()

    QtTest.QTest.mouseClick(
        listing.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=position
    )

    assert window._current_instance() == (30, 7)
    assert window.locate_action.isEnabled()
    assert not np.array_equal(window.viewport._colors, original_colors)
    item.setCheckState(QtCore.Qt.CheckState.Checked)
    assert window._visible_count == 2

    QtTest.QTest.mouseClick(
        listing.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=position
    )

    assert window._current_instance() is None
    assert not listing.selectedItems()
    assert item.checkState() == QtCore.Qt.CheckState.Checked
    assert window._visible_count == 2
    item.setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert not window.locate_action.isEnabled()
    assert not window.merge_action.isEnabled()
    assert window._visible_count == len(original_labels)
    np.testing.assert_array_equal(window.viewport._colors, original_colors)
    np.testing.assert_array_equal(window.document.labels, original_labels)


@pytest.mark.parametrize("focus_widget", ["viewport", "file_list"])
def test_frame_navigation_shortcuts(
    controls_window, app, tmp_path, focus_widget
):
    window = controls_window
    first = window.document.frame.path
    second = dialog_tests.cloud(tmp_path / "second.bin")
    window.open_paths([first, second])
    wait_load(window, app)
    window.show()
    window.activateWindow()
    widget = getattr(window, focus_widget)
    widget.setFocus()
    app.processEvents()

    for key, index in [
        (QtCore.Qt.Key.Key_A, 0),
        (QtCore.Qt.Key.Key_D, 1),
        (QtCore.Qt.Key.Key_D, 1),
        (QtCore.Qt.Key.Key_A, 0),
    ]:
        window.activateWindow()
        widget.setFocus()
        app.processEvents()
        QtTest.QTest.keyClick(widget, key)
        wait_load(window, app)
        assert window.frame_index == index
    assert not window._errors


def test_instance_eyes_filter_without_selecting_or_editing(controls_window, app):
    window = controls_window
    window.show()
    app.processEvents()
    listing = window.instance_list
    original = window.document.labels.copy()
    items = [listing.item(0), listing.item(1)]
    assert all(
        listing.item(row).checkState() == QtCore.Qt.CheckState.Unchecked
        for row in range(listing.count())
    )
    codes = []
    for item in items:
        key = item.data(QtCore.Qt.ItemDataRole.UserRole)
        codes.append((key[1] << 16) | key[0])
        QtTest.QTest.mouseClick(
            listing.viewport(), QtCore.Qt.MouseButton.LeftButton,
            pos=listing.visibility_rect(listing.indexFromItem(item)).center(),
        )
        assert item.checkState() == QtCore.Qt.CheckState.Checked
        assert not listing.selectedItems()
        np.testing.assert_array_equal(window._visible, np.isin(original, codes))
    for item in items:
        QtTest.QTest.mouseClick(
            listing.viewport(), QtCore.Qt.MouseButton.LeftButton,
            pos=listing.visibility_rect(listing.indexFromItem(item)).center(),
        )
    assert window._visible.all()
    np.testing.assert_array_equal(window.document.labels, original)


def test_class_eye_toggles_visibility_without_changing_target(
    controls_window, app
):
    window = controls_window
    window.show()
    app.processEvents()
    select_class(window, 10)
    item = window.class_list.currentItem()
    assert item.text() == "Vehicle (10) · 2"
    assert item.background().color().name() == "#6496f5"
    original = window.document.labels.copy()
    index = window.class_list.indexFromItem(item)
    for state in (
        QtCore.Qt.CheckState.Unchecked, QtCore.Qt.CheckState.Checked
    ):
        QtTest.QTest.mouseClick(
            window.class_list.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=window.class_list.visibility_rect(index).center(),
        )
        assert item.checkState() == state
        assert window._current_class() == 10
        assert window._visible_count == (
            6 if state == QtCore.Qt.CheckState.Unchecked else 8
        )
    np.testing.assert_array_equal(window.document.labels, original)


def test_frame_list_selection_saves_only_the_edited_frame(
    controls_window, app, tmp_path
):
    window = controls_window
    first = window.document.frame.path
    second = dialog_tests.cloud(tmp_path / "second.bin")
    third = dialog_tests.cloud(tmp_path / "third.bin")
    third_labels = np.full(8, (9 << 16) | 40, dtype="<u4")
    third_labels.tofile(third.with_suffix(".label"))
    original_sources = {
        path: path.read_bytes() for path in (first, second, third)
    }
    window.open_paths([first, second, third])
    wait_load(window, app)
    select_class(window, 10)
    window._apply_selection(np.array([6]))
    expected_first = window.document.labels.copy()

    window.file_list.setCurrentRow(2)
    wait_load(window, app)

    assert window.frame_index == 2
    assert window.file_list.currentRow() == 2
    np.testing.assert_array_equal(window.document.labels, third_labels)
    np.testing.assert_array_equal(
        np.fromfile(first.with_suffix(".label"), dtype="<u4"), expected_first
    )
    assert not second.with_suffix(".label").exists()

    window.file_list.setCurrentRow(0)
    wait_load(window, app)

    assert window.frame_index == 0
    np.testing.assert_array_equal(window.document.labels, expected_first)
    np.testing.assert_array_equal(
        np.fromfile(third.with_suffix(".label"), dtype="<u4"), third_labels
    )
    for path, original in original_sources.items():
        assert path.read_bytes() == original
    assert not window._errors


def test_file_checkboxes_and_review_dots_track_saved_labels(
    controls_window, app, tmp_path
):
    window = controls_window
    first = window.document.frame.path
    second = dialog_tests.cloud(tmp_path / "unlabeled.bin")
    window.open_paths([first, second])
    wait_load(window, app)
    labeled_item = window.file_list.item(0)
    unlabeled_item = window.file_list.item(1)
    original = window.document.labels.copy()
    assert labeled_item.text() == first.name
    assert unlabeled_item.text() == second.name
    assert labeled_item.checkState() == QtCore.Qt.CheckState.Checked
    assert unlabeled_item.checkState() == QtCore.Qt.CheckState.Unchecked
    for item in (labeled_item, unlabeled_item):
        assert not item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable
        assert not item.data(QtCore.Qt.ItemDataRole.UserRole)
        image = item.icon().pixmap(24, 24).toImage()
        assert image.pixelColor(12, 12).alpha() == 0

    window.show()
    app.processEvents()
    position = window.file_list.visualItemRect(labeled_item).center()
    position.setX(14)
    QtTest.QTest.mouseClick(
        window.file_list.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=position,
    )
    QtTest.QTest.keyClick(window.file_list, QtCore.Qt.Key.Key_Space)
    assert labeled_item.checkState() == QtCore.Qt.CheckState.Checked
    np.testing.assert_array_equal(window.document.labels, original)
    window._toggle_reviewed(1)
    assert not unlabeled_item.data(QtCore.Qt.ItemDataRole.UserRole)
    assert not second.with_suffix(".label").exists()

    window._toggle_reviewed(0)

    assert labeled_item.data(QtCore.Qt.ItemDataRole.UserRole)
    image = labeled_item.icon().pixmap(24, 24).toImage()
    assert image.pixelColor(12, 12).alpha() > 0
    assert "Reviewed" in labeled_item.toolTip()
    np.testing.assert_array_equal(window.document.labels, original)
    np.testing.assert_array_equal(
        np.fromfile(first.with_suffix(".label"), dtype="<u4"), original
    )

    window.document.clear([0])
    window._refresh()

    assert not labeled_item.data(QtCore.Qt.ItemDataRole.UserRole)
    assert labeled_item.checkState() == QtCore.Qt.CheckState.Checked
    assert window._autosave()
    assert not labeled_item.data(QtCore.Qt.ItemDataRole.UserRole)
    window._toggle_reviewed(0)
    assert labeled_item.data(QtCore.Qt.ItemDataRole.UserRole)
    window._toggle_reviewed(0)
    assert not labeled_item.data(QtCore.Qt.ItemDataRole.UserRole)

    window.file_list.setCurrentRow(1)
    wait_load(window, app)

    unlabeled_item = window.file_list.item(1)
    assert unlabeled_item.checkState() == QtCore.Qt.CheckState.Unchecked
    select_class(window, 30)
    window._apply_selection(np.array([1]))
    assert window._autosave()
    assert unlabeled_item.checkState() == QtCore.Qt.CheckState.Checked
    assert not unlabeled_item.data(QtCore.Qt.ItemDataRole.UserRole)
    expected_second = np.zeros(8, dtype="<u4")
    expected_second[1] = 30
    np.testing.assert_array_equal(
        np.fromfile(second.with_suffix(".label"), dtype="<u4"),
        expected_second,
    )


def test_cancelled_frame_list_selection_restores_current_frame(
    controls_window, app, tmp_path
):
    window = controls_window
    first = window.document.frame.path
    second = dialog_tests.cloud(tmp_path / "second.bin")
    window.open_paths([first, second])
    wait_load(window, app)
    document = window.document
    original = document.labels.copy()
    window._leave_decision = lambda: "cancel"

    window.file_list.setCurrentRow(1)

    assert window._worker is None
    assert window.document is document
    assert window.frame_index == 0
    assert window.file_list.currentRow() == 0
    np.testing.assert_array_equal(document.labels, original)
