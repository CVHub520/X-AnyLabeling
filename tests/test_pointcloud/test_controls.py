import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

from anylabeling.views.labeling.pointcloud.controls import (
    ClassDefinitionDialog,
    PointCloudListWidget,
)
from anylabeling.views.labeling.pointcloud.model import ClassDefinition
from anylabeling.views.labeling.pointcloud.icons import get_icon
from anylabeling.views.labeling.pointcloud.style import get_pointcloud_style


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def instance_list(app):
    widget = PointCloudListWidget(
        toggle_selection=True, remove_tooltip="Delete instance"
    )
    widget.setObjectName("pointcloudList")
    widget.setStyleSheet(get_pointcloud_style())
    widget.resize(300, 160)
    for index in range(3):
        item = QtWidgets.QListWidgetItem(f"Instance {index}", widget)
        item.setSizeHint(QtCore.QSize(200, 32))
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def _row_point(widget, row):
    return widget.visualItemRect(widget.item(row)).center()


def test_repeat_click_clears_selection_and_current(instance_list):
    widget = instance_list
    changes = []
    widget.itemSelectionChanged.connect(
        lambda: changes.append(len(widget.selectedItems()))
    )
    point = _row_point(widget, 0)
    QtTest.QTest.mouseClick(
        widget.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point
    )
    assert widget.currentItem() is widget.item(0)
    assert widget.selectedItems() == [widget.item(0)]
    QtTest.QTest.mouseClick(
        widget.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point
    )
    assert not widget.selectedItems()
    assert widget.currentItem() is None
    assert changes[-2:] == [1, 0]


def test_modifier_multiselection_and_last_ctrl_deselection(instance_list):
    widget = instance_list
    for row, modifier in (
        (0, QtCore.Qt.KeyboardModifier.NoModifier),
        (1, QtCore.Qt.KeyboardModifier.ControlModifier),
    ):
        QtTest.QTest.mouseClick(
            widget.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            modifier,
            _row_point(widget, row),
        )
    assert len(widget.selectedItems()) == 2
    QtTest.QTest.mouseClick(
        widget.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
        _row_point(widget, 2),
    )
    assert widget.item(1).isSelected() and widget.item(2).isSelected()
    widget.clearSelection()
    widget.setCurrentItem(None)
    for modifier in (
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    ):
        QtTest.QTest.mouseClick(
            widget.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            modifier,
            _row_point(widget, 0),
        )
    assert not widget.selectedItems()
    assert widget.currentItem() is None


def test_hover_delete_does_not_select_target_and_release_outside_cancels(
    instance_list,
):
    widget = instance_list
    widget.setCurrentRow(0)
    requests = []
    widget.remove_requested.connect(requests.append)
    target = widget.item(1)
    point = widget.remove_rect(widget.indexFromItem(target)).center()
    QtTest.QTest.mouseMove(widget.viewport(), point)
    QtTest.QTest.mouseClick(
        widget.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point
    )
    assert requests == [target]
    assert widget.currentItem() is widget.item(0)
    assert widget.selectedItems() == [widget.item(0)]
    QtTest.QTest.mousePress(
        widget.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point
    )
    QtTest.QTest.mouseRelease(
        widget.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=_row_point(widget, 1),
    )
    assert requests == [target]
    widget.item(2).setData(widget.REMOVABLE_ROLE, False)
    point = widget.remove_rect(widget.indexFromItem(widget.item(2))).center()
    QtTest.QTest.mouseClick(
        widget.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point
    )
    assert requests == [target]


def test_only_remove_button_has_tooltip(instance_list):
    widget = instance_list
    item = widget.item(0)
    item.setToolTip("Redundant row tooltip")
    for point, expected in (
        (
            widget.remove_rect(widget.indexFromItem(item)).center(),
            "Delete instance",
        ),
        (_row_point(widget, 0), ""),
    ):
        event = QtGui.QHelpEvent(
            QtCore.QEvent.Type.ToolTip,
            point,
            widget.viewport().mapToGlobal(point),
        )
        QtWidgets.QApplication.sendEvent(widget.viewport(), event)
        if expected:
            assert QtWidgets.QToolTip.text() == expected
        else:
            QtTest.QTest.qWait(400)
            assert not QtWidgets.QToolTip.isVisible()


def test_class_double_click_preserves_edit_signal_and_selection(app):
    widget = PointCloudListWidget(remove_tooltip="Remove definition")
    widget.resize(300, 100)
    widget.addItem("Road")
    widget.show()
    app.processEvents()
    edits = []
    widget.itemDoubleClicked.connect(edits.append)
    point = _row_point(widget, 0)
    QtTest.QTest.mouseClick(
        widget.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point
    )
    QtTest.QTest.mouseDClick(
        widget.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point
    )
    assert edits == [widget.item(0)]
    assert widget.currentItem() is widget.item(0)
    widget.close()


def test_class_dialog_validates_inline_then_accepts_unique_definition(app):
    dialog = ClassDefinitionDialog(used_ids={0, 10}, suggested_id=10)
    dialog.show()
    app.processEvents()
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    assert "name" in dialog.error_label.text()
    dialog.name_input.setText("  Road  ")
    dialog.color_input.setText("#12345g")
    dialog.accept()
    assert "#RRGGBB" in dialog.error_label.text()
    dialog.color_input.setText("#abc123")
    dialog.accept()
    assert "already" in dialog.error_label.text()
    dialog.id_input.setValue(40)
    QtTest.QTest.mouseClick(
        dialog.save_button, QtCore.Qt.MouseButton.LeftButton
    )
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    assert dialog.definition() == ClassDefinition(40, "Road", "#ABC123")


def test_edit_dialog_locks_id_and_color_picker_updates_preview(app):
    original = ClassDefinition(0, "Unlabeled", "#808080")
    dialog = ClassDefinitionDialog(original, used_ids={0, 10})
    dialog.show()
    app.processEvents()
    assert not dialog.id_input.isEnabled()
    previous = dialog.color_button.icon().cacheKey()
    with patch.object(
        QtWidgets.QColorDialog,
        "getColor",
        return_value=QtGui.QColor("#123456"),
    ):
        QtTest.QTest.mouseClick(
            dialog.color_button, QtCore.Qt.MouseButton.LeftButton
        )
    assert dialog.color_input.text() == "#123456"
    assert dialog.color_button.icon().cacheKey() != previous
    dialog.name_input.setText("Unknown")
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    assert dialog.definition() == ClassDefinition(0, "Unknown", "#123456")


@pytest.mark.parametrize("editing", [False, True])
def test_class_dialog_fields_buttons_and_embedded_color_stay_aligned(
    app, editing
):
    dialog = ClassDefinitionDialog(
        ClassDefinition(10, "Vehicle", "#6496F5") if editing else None
    )
    dialog.show()
    app.processEvents()
    inputs = (dialog.id_input, dialog.name_input, dialog.color_input)
    labels = [
        label
        for label in dialog.findChildren(QtWidgets.QLabel)
        if label.buddy() in inputs
    ]
    assert len(labels) == 3
    assert dialog.findChild(QtWidgets.QLabel, "classDefinitionTitle") is None
    for width in (400, 560):
        dialog.resize(width, dialog.height())
        app.processEvents()
        fields = [widget.geometry() for widget in inputs]
        assert len({rect.left() for rect in fields}) == 1
        assert len({rect.right() for rect in fields}) == 1
        for label in labels:
            assert (
                abs(
                    label.geometry().center().y()
                    - label.buddy().geometry().center().y()
                )
                <= 1
            )
        assert dialog.save_button.geometry().left() == fields[0].left()
        assert dialog.cancel_button.geometry().right() == fields[0].right()
        assert dialog.color_button.parentWidget() is dialog.color_input
        assert dialog.color_button.size() == QtCore.QSize(22, 22)
        assert dialog.color_input.rect().contains(
            dialog.color_button.geometry()
        )
        assert (
            abs(
                dialog.color_button.geometry().center().y()
                - dialog.color_input.rect().center().y()
            )
            <= 1
        )
        assert (
            dialog.color_input.textMargins().right()
            >= dialog.color_button.width()
        )
    dialog.close()


@pytest.mark.parametrize("save", [False, True])
def test_class_dialog_cancel_rejects_and_enter_saves(app, save):
    dialog = ClassDefinitionDialog(suggested_id=40)
    dialog.name_input.setText("Road")
    dialog.show()
    app.processEvents()
    if save:
        QtTest.QTest.keyClick(dialog.name_input, QtCore.Qt.Key.Key_Return)
        assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
        assert dialog.definition().name == "Road"
    else:
        QtTest.QTest.mouseClick(
            dialog.cancel_button, QtCore.Qt.MouseButton.LeftButton
        )
        assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    assert not dialog.isVisible()


def test_upload_icon_uses_one_theme_color(app):
    image = get_icon("upload", "#636363", "#0071E3").pixmap(18, 18).toImage()
    painted = False
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixelColor(x, y)
            if pixel.alpha():
                painted = True
                assert pixel.red() == pixel.green() == pixel.blue()
    assert painted
