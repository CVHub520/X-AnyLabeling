from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from PyQt6 import QtCore, QtTest, QtWidgets

from anylabeling.views.labeling.pointcloud.controls import PointCloudListWidget
from anylabeling.views.labeling.pointcloud.cuboid import Cuboid
from anylabeling.views.labeling.pointcloud.model import ClassDefinition

from . import test_cuboids as cuboid_tests

app = cuboid_tests.app
detection = cuboid_tests.detection
window = cuboid_tests.window


@pytest.fixture
def panel(detection, app):
    detection._schedule_autosave = Mock()
    detection.class_definitions["detection"].append(
        ClassDefinition(30, "Person", "#FF1E1E")
    )
    for box in (
        Cuboid(1, 10, (0, 0, 0), (4, 2, 1)),
        Cuboid(2, 10, (5, 0, 0), (4, 2, 1)),
        Cuboid(3, 30, (0, 5, 0), (1, 1, 2)),
    ):
        detection.document.set_cuboid(box)
    detection._refresh()
    detection.resize(1280, 800)
    detection.show()
    app.processEvents()
    return detection.detection


def item_with_id(listing, value):
    return next(
        listing.item(row)
        for row in range(listing.count())
        if listing.item(row).data(QtCore.Qt.ItemDataRole.UserRole) == value
    )


def click_control(listing, value, control):
    item = item_with_id(listing, value)
    listing.scrollToItem(item)
    index = listing.indexFromItem(item)
    position = getattr(listing, f"{control}_rect")(index).center()
    QtTest.QTest.mouseMove(listing.viewport(), position)
    QtTest.QTest.mouseClick(
        listing.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=position
    )


def test_task_panel_layout_and_counts(panel):
    tabs = panel.window.sidebar_tabs
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "Det",
        "Seg",
    ]
    assert tabs.currentIndex() == 0
    assert tabs.tabToolTip(0) == "Detection"
    assert tabs.tabToolTip(1) == "Segmentation"
    assert panel.labels_heading.text() == "Labels (2)"
    assert panel.objects_heading.text() == "Objects (3)"
    assert panel.properties.isHidden()
    assert not panel.panel.findChildren(QtWidgets.QPushButton)
    buttons = panel.panel.findChildren(QtWidgets.QToolButton)
    assert [button.defaultAction().text() for button in buttons] == [
        "Lock all classes",
        "Toggle all classes",
        "Lock all objects",
        "Toggle all objects",
    ]
    assert panel.draw_action.isVisible()
    assert panel.window.color_mode.currentData() == "rgb"
    assert not panel.window.tool_actions["brush"].isEnabled()
    buttons = panel.window.view_tool_scroll.findChildren(QtWidgets.QToolButton)
    assert all(
        button.isVisible() == button.defaultAction().isVisible()
        for button in buttons
        if button.defaultAction() is not None
    )


def test_switching_tasks_restores_segmentation_filters_and_tools(panel):
    window = panel.window
    panel.select(1)
    original_boxes = window.document.cuboids
    window.document.assign_semantic(np.array([0, 1]), 10)
    window.sidebar_tabs.setCurrentIndex(1)
    window._refresh()
    assert not panel.enabled
    assert not window.viewport.cuboids
    assert panel.orthographic.isHidden()
    assert not panel.views_button.isVisible()
    item_with_id(window.class_list, 10).setCheckState(
        QtCore.Qt.CheckState.Unchecked
    )
    window._select_tool("brush")
    window.color_mode.setCurrentIndex(window.color_mode.findData("instance"))
    visible = window._visible.copy()
    colors = window.viewport._colors.copy()
    labels = window.document.labels.copy()
    history = len(window.document._undo)
    window.sidebar_tabs.setCurrentWidget(panel.panel)
    assert panel.enabled
    assert window._visible.all()
    np.testing.assert_allclose(window.viewport._colors[:, :3], 1)
    assert len(window.viewport.cuboids) == 3
    assert window.viewport.selected_cuboid.id == 1
    assert window.viewport._tool == "browse"
    assert not any(
        action.isEnabled() for action in window.operation_actions.values()
    )
    window._apply_selection(np.array([2]))
    np.testing.assert_array_equal(window.document.labels, labels)
    window.sidebar_tabs.setCurrentIndex(1)
    np.testing.assert_array_equal(window._visible, visible)
    np.testing.assert_array_equal(window.viewport._colors, colors)
    assert window.viewport._tool == "brush"
    assert window.color_mode.currentData() == "instance"
    assert window.document.cuboids == original_boxes
    assert len(window.document._undo) == history


def test_switching_tasks_cancels_creation_and_disables_detection_shortcuts(
    panel, app
):
    panel.window.activateWindow()
    assert QtTest.QTest.qWaitForWindowActive(panel.window)
    panel.start_creation()
    assert panel.views[0].creating
    panel.window.sidebar_tabs.setCurrentIndex(1)
    assert not any(view.creating for view in panel.views)
    assert not any(action.isEnabled() for action in panel.shortcuts)
    panel.views[0].setFocus()
    app.processEvents()
    QtTest.QTest.keyClick(panel.views[0]._gl, QtCore.Qt.Key.Key_N)
    assert not panel.views[0].creating
    panel.window.sidebar_tabs.setCurrentWidget(panel.panel)
    assert all(action.isEnabled() for action in panel.shortcuts)
    QtTest.QTest.keyClick(panel.views[0]._gl, QtCore.Qt.Key.Key_N)
    assert panel.views[0].creating


def test_task_switch_preserves_three_view_visibility_and_exits_fullscreen(
    panel,
):
    panel.toggle_views(show=False)
    panel.window.sidebar_tabs.setCurrentIndex(1)
    panel.window.sidebar_tabs.setCurrentWidget(panel.panel)
    assert panel.orthographic.isHidden()
    panel.toggle_views(show=True)
    panel.toggle_expanded_view(panel.views[1])
    panel.window.sidebar_tabs.setCurrentIndex(1)
    assert not panel.views[0].isHidden()
    assert panel.orthographic.isHidden()
    panel.window.sidebar_tabs.setCurrentWidget(panel.panel)
    assert not panel.orthographic.isHidden()
    assert all(not view.isHidden() for view in panel.views)


def test_hover_lock_changes_only_target_and_preserves_selection(panel):
    panel.select(3)
    history = len(panel.document._undo)
    click_control(panel.objects, 1, "lock")
    assert [box.locked for box in panel.document.cuboids] == [
        True,
        False,
        False,
    ]
    assert panel.selected_id == 3
    assert len(panel.document._undo) == history + 1
    assert (
        item_with_id(panel.objects, 1).data(
            PointCloudListWidget.REMOVABLE_ROLE
        )
        is False
    )
    click_control(panel.objects, 1, "remove")
    assert len(panel.document.cuboids) == 3
    panel.select(1)
    before = panel.selected
    panel.delete()
    panel.fit()
    assert panel.selected == before
    with pytest.raises(ValueError, match="Unlock"):
        panel.document.set_cuboid(replace(before, center=(9, 9, 9)))
    click_control(panel.objects, 1, "lock")
    assert not panel.selected.locked


def test_label_lock_and_global_lock_are_single_undo_steps(panel):
    history = len(panel.document._undo)
    click_control(panel.labels, 10, "lock")
    assert [box.locked for box in panel.document.cuboids] == [
        True,
        True,
        False,
    ]
    assert len(panel.document._undo) == history + 1
    assert item_with_id(panel.labels, 10).data(
        PointCloudListWidget.LOCKED_ROLE
    )
    panel.window.undo()
    assert not any(box.locked for box in panel.document.cuboids)
    panel.lock_all_action.trigger()
    assert all(box.locked for box in panel.document.cuboids)
    assert panel.lock_all_action.text() == "Unlock all objects"
    assert len(panel.document._undo) == history + 1
    panel.lock_all_action.trigger()
    assert not any(box.locked for box in panel.document.cuboids)
    panel.window.undo()
    assert all(box.locked for box in panel.document.cuboids)


def test_visibility_controls_only_filter_boxes_and_aggregate_labels(panel):
    panel.select(1)
    colors = panel.views[0]._colors.copy()
    visible = panel.views[0]._visible.copy()
    history = len(panel.document._undo)
    click_control(panel.labels, 10, "visibility")
    assert [box.id for box in panel.views[0].cuboids] == [3]
    assert not any(view.cuboids for view in panel.views[1:])
    assert (
        item_with_id(panel.objects, 1).checkState()
        == QtCore.Qt.CheckState.Unchecked
    )
    click_control(panel.objects, 1, "visibility")
    assert [box.id for box in panel.views[0].cuboids] == [1, 3]
    assert (
        item_with_id(panel.labels, 10).checkState()
        == QtCore.Qt.CheckState.Checked
    )
    panel.visibility_action.trigger()
    assert not panel.views[0].cuboids
    panel.visibility_action.trigger()
    assert len(panel.views[0].cuboids) == 3
    np.testing.assert_array_equal(panel.views[0]._visible, visible)
    np.testing.assert_array_equal(panel.views[0]._colors, colors)
    assert len(panel.document._undo) == history


def test_delete_unselected_object_keeps_selection_and_updates_count(panel):
    panel.select(3)
    click_control(panel.objects, 1, "remove")
    assert [box.id for box in panel.document.cuboids] == [2, 3]
    assert panel.selected_id == 3
    assert panel.objects_heading.text() == "Objects (2)"
    panel.window.undo()
    assert len(panel.document.cuboids) == 3
    click_control(panel.objects, 3, "remove")
    assert panel.selected_id is None
    assert not any(view.cuboids for view in panel.views[1:])


def test_class_headers_lock_and_hide_without_changing_definitions(panel):
    panel.document.assign_semantic(np.array([0, 1]), 10)
    labels = panel.document.labels.copy()
    definitions = list(panel.window.class_definitions["detection"])
    assert not panel.labels.allow_remove
    panel.classes_lock_action.trigger()
    assert all(box.locked for box in panel.document.cuboids)
    assert panel.classes_lock_action.text() == "Unlock all classes"
    panel.classes_lock_action.trigger()
    assert not any(box.locked for box in panel.document.cuboids)
    panel.classes_visibility_action.trigger()
    assert panel.hidden_ids == {1, 2, 3}
    panel.classes_visibility_action.trigger()
    assert not panel.hidden_ids
    assert panel.window.class_definitions["detection"] == definitions
    np.testing.assert_array_equal(panel.document.labels, labels)
    assert len(panel.document.cuboids) == 3
    assert panel.labels_heading.text() == "Labels (2)"


def test_row_controls_cancel_when_released_elsewhere(panel):
    listing = panel.objects
    item = item_with_id(listing, 1)
    index = listing.indexFromItem(item)
    position = listing.lock_rect(index).center()
    QtTest.QTest.mousePress(
        listing.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=position
    )
    QtTest.QTest.mouseRelease(
        listing.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=QtCore.QPoint(12, 100),
    )
    assert not any(box.locked for box in panel.document.cuboids)


def test_selecting_label_sets_creation_class_without_relabeling_objects(panel):
    panel.select(1)
    original = panel.document.cuboids
    panel.labels.setCurrentItem(item_with_id(panel.labels, 30))
    assert panel.selected_id is None
    assert panel.current_class() == 30
    assert panel.document.cuboids == original
    panel.create((1, 2, 3), (1, 1, 1))
    assert panel.selected.class_id == 30
    assert panel.objects_heading.text() == "Objects (4)"
