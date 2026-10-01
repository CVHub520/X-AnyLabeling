from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

from anylabeling.views.labeling.pointcloud.controls import PointCloudListWidget
from anylabeling.views.labeling.pointcloud.icons import get_icon
from anylabeling.views.labeling.pointcloud.model import ClassDefinition
from anylabeling.views.labeling.utils.theme import get_theme

from . import test_cuboids as cuboid_tests
from .test_detection_panel import click_control, item_with_id

app = cuboid_tests.app
window = cuboid_tests.window
detection = cuboid_tests.detection


@pytest.fixture
def segmentation(detection, app):
    detection.sidebar_tabs.setCurrentIndex(1)
    detection._schedule_autosave = Mock()
    detection.class_definitions["segmentation"].append(
        ClassDefinition(30, "Person", "#F6B26B")
    )
    doc = detection.document
    doc.assign_semantic(np.arange(8), 10)
    doc.create_instance([0, 1], 10)
    doc.create_instance([2, 3], 10)
    doc.assign_semantic([8, 9], 30)
    detection._refresh()
    detection.resize(1280, 800)
    detection.show()
    app.processEvents()
    return detection


def test_segmentation_class_lock_protects_points_targets_and_deletion(
    segmentation,
):
    ui = segmentation
    doc = ui.document
    original = doc.labels.copy()
    click_control(ui.class_list, 10, "lock")
    assert doc.locked_classes == {10}
    assert all(
        item_with_id(ui.instance_list, key).data(
            PointCloudListWidget.LOCKED_ROLE
        )
        for key in ((10, 1), (10, 2))
    )
    ui.class_list.setCurrentItem(item_with_id(ui.class_list, 30))
    ui._apply_selection(np.array([0, 1, 10]))
    np.testing.assert_array_equal(doc.labels[:10], original[:10])
    assert doc.semantic_view[10] == 30
    ui.class_list.setCurrentItem(item_with_id(ui.class_list, 10))
    assert not ui.operation_actions["assign"].isEnabled()
    before = doc.labels.copy()
    ui._apply_selection(np.array([9, 11]))
    np.testing.assert_array_equal(doc.labels, before)
    ui._confirm.reset_mock()
    click_control(ui.class_list, 10, "remove")
    ui._remove_class(item_with_id(ui.class_list, 10))
    ui._delete_instance(item_with_id(ui.instance_list, (10, 1)))
    ui._confirm.assert_not_called()
    np.testing.assert_array_equal(doc.labels, before)
    click_control(ui.class_list, 10, "lock")
    ui._apply_selection(np.array([11]))
    assert doc.semantic_view[11] == 10


def test_segmentation_instance_lock_blocks_merge_split_delete_and_relabel(
    segmentation,
):
    ui = segmentation
    doc = ui.document
    click_control(ui.instance_list, (10, 1), "lock")
    original = doc.labels.copy()
    for operation in (
        lambda: doc.assign_semantic([0, 1], 30, overwrite=True),
        lambda: doc.clear([0, 1]),
        lambda: doc.add_to_instance([4], (10, 1)),
        lambda: doc.remove_from_instance([0], (10, 1)),
        lambda: doc.split_instance([0], (10, 1)),
        lambda: doc.delete_instance((10, 1)),
        lambda: doc.merge_instances((10, 2), [(10, 1)]),
        lambda: doc.merge_instances((10, 1), [(10, 2)]),
    ):
        with pytest.raises(ValueError, match="Unlock"):
            operation()
        np.testing.assert_array_equal(doc.labels, original)
    for key in ((10, 1), (10, 2)):
        item_with_id(ui.instance_list, key).setSelected(True)
    ui.instance_list.setCurrentItem(
        item_with_id(ui.instance_list, (10, 1)),
        QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
    )
    assert not ui.merge_action.isEnabled()
    ui._merge_instances()
    np.testing.assert_array_equal(doc.labels, original)
    ui.sidebar_tabs.setCurrentWidget(ui.detection.panel)
    ui.sidebar_tabs.setCurrentIndex(1)
    assert doc.locked_instances == {(10, 1)}
    click_control(ui.instance_list, (10, 1), "lock")
    doc.merge_instances((10, 1), [(10, 2)])
    assert doc.instance_view[2] == 1


def test_segmentation_lock_and_eye_do_not_change_row_selection(segmentation):
    ui = segmentation
    ui.class_list.setCurrentItem(item_with_id(ui.class_list, 30))
    click_control(ui.class_list, 10, "lock")
    assert ui._current_class() == 30
    click_control(ui.class_list, 10, "visibility")
    assert not ui._visible[:8].any()
    assert ui._current_class() == 30
    assert ui.document.locked_classes == {10}
    click_control(ui.class_list, 10, "visibility")
    assert ui._visible.all()
    ui._schedule_autosave.assert_not_called()


def test_segmentation_header_locks_all_classes_without_changing_labels(
    segmentation,
):
    ui = segmentation
    original = ui.document.labels.copy()
    definitions = list(ui.class_definitions["segmentation"])
    ids = {
        ui.class_list.item(index).data(QtCore.Qt.ItemDataRole.UserRole)
        for index in range(ui.class_list.count())
    }
    ui.classes_lock_action.trigger()
    assert ui.document.locked_classes == ids
    assert ui.classes_lock_action.text() == "Unlock all classes"
    assert all(
        ui.class_list.item(index).data(PointCloudListWidget.LOCKED_ROLE)
        for index in range(ui.class_list.count())
    )
    ui.classes_lock_action.trigger()
    assert not ui.document.locked_classes
    assert ui.class_definitions["segmentation"] == definitions
    np.testing.assert_array_equal(ui.document.labels, original)
    ui._schedule_autosave.assert_not_called()


@pytest.mark.parametrize("task", [0, 1])
@pytest.mark.parametrize("scrolling", [False, True])
def test_header_icons_share_row_columns_even_with_scrollbars(
    segmentation, app, task, scrolling
):
    ui = segmentation
    if scrolling:
        ui.class_definitions[
            "segmentation" if task == 0 else "detection"
        ].extend(
            ClassDefinition(i, f"Class {i}", "#6496F5") for i in range(40, 120)
        )
    ui._refresh()
    ui.sidebar_tabs.setCurrentIndex(1 - task)
    listings = (
        (ui.class_list, ui.instance_list)
        if task == 0
        else (ui.detection.labels, ui.detection.objects)
    )
    for listing in listings:
        if not listing.count():
            item = QtWidgets.QListWidgetItem("Example")
            listing.addItem(item)
    for width in (1050, 1450):
        ui.resize(width, 800)
        QtTest.QTest.qWait(20)
        for listing in listings:
            index = listing.indexFromItem(listing.item(0))
            row = [
                listing.viewport()
                .mapToGlobal(getattr(listing, name + "_rect")(index).center())
                .x()
                for name in (
                    ("lock", "visibility", "remove")
                    if listing.allow_remove
                    else ("lock", "visibility")
                )
            ]
            buttons = sorted(
                listing.parentWidget().findChildren(QtWidgets.QToolButton),
                key=lambda button: button.mapToGlobal(QtCore.QPoint()).x(),
            )
            header = [
                button.mapToGlobal(button.rect().center()).x()
                for button in buttons
            ]
            assert header[-1] == pytest.approx(row[-1], abs=1), (
                listing.accessibleName(),
                listing._header.getContentsMargins(),
                listing._header.geometry(),
                listing.viewport().geometry(),
                listing.geometry(),
                header,
                row,
            )
            assert header[-2] == pytest.approx(row[-2], abs=1)
            assert all(
                button.size() == QtCore.QSize(24, 24) for button in buttons
            )
            assert all(
                button.iconSize() == QtCore.QSize(16, 16) for button in buttons
            )


def test_file_checkbox_aligns_with_panel_toggle(segmentation):
    ui = segmentation
    listing = ui.file_list
    item = listing.item(0)
    option = QtWidgets.QStyleOptionViewItem()
    option.initFrom(listing)
    delegate = QtWidgets.QStyledItemDelegate(listing)
    delegate.initStyleOption(option, listing.indexFromItem(item))
    option.rect = listing.visualItemRect(item)
    rect = listing.style().subElementRect(
        QtWidgets.QStyle.SubElement.SE_ItemViewItemCheckIndicator,
        option,
        listing,
    )
    checkbox = (
        listing.viewport().mapToGlobal(rect.adjusted(2, 0, 0, 0).center()).x()
    )
    button = listing.parentWidget().findChild(QtWidgets.QToolButton)
    toggle = button.mapToGlobal(button.rect().center()).x()
    assert checkbox == pytest.approx(toggle, abs=1)


@pytest.mark.parametrize("index", [1, 2, 3])
def test_rotation_handle_is_hollow_unlinked_and_uses_grab_cursor(
    detection, app, index
):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    detection.show()
    app.processEvents()
    view = workspace.views[index]
    view._focus_animation.setCurrentTime(view._focus_animation.duration())
    handles, rotation = view._handles(workspace.selected)
    point = QtCore.QPointF(*rotation)
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        point,
        point,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    view._mouse_move(event)
    assert view._hover_handle == ("rotate", None)
    assert view._gl.cursor().shape() == QtCore.Qt.CursorShape.OpenHandCursor
    cuboid_tests.mouse(view, "_mouse_press", rotation)
    assert view._gesture[0] == "rotate"
    assert view._gl.cursor().shape() == QtCore.Qt.CursorShape.ClosedHandCursor
    view.cancel_selection()
    assert view._hover_handle is None
    assert view._gl.cursor().shape() == QtCore.Qt.CursorShape.ArrowCursor
    image = QtGui.QImage(
        view.size(), QtGui.QImage.Format.Format_ARGB32_Premultiplied
    )
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    view._paint_overlay(painter)
    painter.end()
    assert image.pixelColor(point.toPoint()).alpha() == 0
    midpoint = (rotation + handles[5][1]) / 2
    assert image.pixelColor(QtCore.QPointF(*midpoint).toPoint()).alpha() == 0
    region = image.copy(
        QtCore.QRect(
            point.toPoint() - QtCore.QPoint(10, 10), QtCore.QSize(21, 21)
        )
    )
    assert any(
        region.pixelColor(x, y).alpha() for x in range(21) for y in range(21)
    )


@pytest.mark.parametrize("index", [1, 2, 3])
def test_resize_handle_hover_matches_hit_and_clears_after_drag(
    detection, app, index
):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    detection.show()
    app.processEvents()
    view = workspace.views[index]
    view._focus_animation.setCurrentTime(view._focus_animation.duration())
    boxes = detection.document.cuboids
    history = len(detection.document._undo)
    handles, _ = view._handles(workspace.selected)

    def render():
        image = QtGui.QImage(
            view.size(), QtGui.QImage.Format.Format_ARGB32_Premultiplied
        )
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        view._paint_overlay(painter)
        painter.end()
        return image

    normal = render()
    for signs, point in handles:
        cuboid_tests.mouse(
            view, "_mouse_move", point, QtCore.Qt.MouseButton.NoButton
        )
        assert view._hover_handle == ("resize", signs)
        assert view._gl.cursor().shape() != QtCore.Qt.CursorShape.ArrowCursor
        pixel = QtCore.QPointF(*point).toPoint()
        assert normal.pixelColor(pixel) != QtGui.QColor("#ffffff")
        assert render().pixelColor(pixel) == QtGui.QColor("#ffffff")
        cuboid_tests.mouse(view, "_mouse_press", point)
        assert view._gesture[0] == "resize"
        assert view._gesture[3] == signs
        view.cancel_selection()
        assert view._hover_handle is None
    assert detection.document.cuboids == boxes
    assert len(detection.document._undo) == history
    signs, point = handles[0]
    cuboid_tests.mouse(view, "_mouse_press", point)
    cuboid_tests.mouse(view, "_mouse_move", point + (12, 8))
    assert view._hover_handle == ("resize", signs)
    assert detection.document.cuboids == boxes
    cuboid_tests.mouse(view, "_mouse_release", point + (12, 8))
    assert view._hover_handle is None
    assert len(detection.document._undo) == history + 1
    point = view._handles(workspace.selected)[0][0][1]
    cuboid_tests.mouse(
        view, "_mouse_move", point, QtCore.Qt.MouseButton.NoButton
    )
    assert view._hover_handle is not None
    view.eventFilter(view._gl, QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert view._hover_handle is None
    workspace.commit(replace(workspace.selected, locked=True))
    cuboid_tests.mouse(
        view, "_mouse_move", point, QtCore.Qt.MouseButton.NoButton
    )
    assert view._hover_handle is None
    assert view._gl.cursor().shape() == QtCore.Qt.CursorShape.ArrowCursor


def test_subviews_only_draw_view_titles(detection, app):
    workspace = detection.detection
    workspace.create((0, 0, 0), (4, 2, 1))
    detection.show()
    app.processEvents()

    class RecordingPainter(QtGui.QPainter):
        def drawText(self, *args):
            self.texts.append(args[-1])
            return super().drawText(*args)

    for view in workspace.views:
        image = QtGui.QImage(
            view.size(), QtGui.QImage.Format.Format_ARGB32_Premultiplied
        )
        painter = RecordingPainter(image)
        painter.texts = []
        view._paint_overlay(painter)
        painter.end()
        labels = [text for text in painter.texts if text.startswith("#")]
        assert bool(labels) == (view.orthographic_view is None)


def test_draw_cuboid_icon_is_distinct_from_view_icons(app):
    icon = (
        get_icon("draw-cuboid", "#000000", "#0071E3").pixmap(32, 32).toImage()
    )
    assert not icon.isNull()
    for view in ("top", "front", "side"):
        assert (
            icon
            != get_icon(view, "#000000", "#0071E3").pixmap(32, 32).toImage()
        )


def test_workspace_prepares_gl_before_native_show(detection, app):
    events = []

    class Probe(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.Type.Show:
                events.append(detection.viewport._gl.program is not None)
            return False

    probe = Probe(detection)
    detection.installEventFilter(probe)
    detection.show()
    app.processEvents()
    assert events and all(events)
    assert detection.palette().color(QtGui.QPalette.ColorRole.Window) == (
        QtGui.QColor(get_theme()["background"])
    )
    assert detection.viewport._error is None


def test_file_and_folder_choosers_use_qt_without_shell_icon_lookup(window):
    for method, callback, result in (
        ("getOpenFileName", window.open_file, ("", "")),
        ("getExistingDirectory", window.open_directory, ""),
    ):
        with patch.object(
            QtWidgets.QFileDialog, method, return_value=result
        ) as dialog:
            callback()
        options = dialog.call_args.kwargs["options"]
        assert options & QtWidgets.QFileDialog.Option.DontUseNativeDialog
        assert (
            options & QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons
        )
