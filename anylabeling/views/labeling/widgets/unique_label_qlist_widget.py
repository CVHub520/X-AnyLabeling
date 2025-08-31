# -*- encoding: utf-8 -*-

import html

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt, pyqtSignal

from .escapable_qlist_widget import EscapableQListWidget


class UniqueLabelQListWidget(EscapableQListWidget):
    # A signal that is emitted when the visibility of a label changes.
    label_visibility_changed = pyqtSignal(str, bool)  # label, visible
    # 新增：选中项变化信号
    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        # Set the selection mode to allow multiple selections.
        self.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        # Connect the itemChanged signal to the on_item_changed slot.
        self.itemChanged.connect(self.on_item_changed)
        # 连接选中变化信号
        self.itemSelectionChanged.connect(self.on_selection_changed)

    def mousePressEvent(self, event):
        # 只有在未按下Ctrl/Shift时，点击空白才清除选择，保证Ctrl/Shift多选原生行为
        if not self.indexAt(event.pos()).isValid():
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if not (modifiers & Qt.ControlModifier or modifiers & Qt.ShiftModifier):
                self.clearSelection()
        
        # 调用父类方法处理点击事件
        super().mousePressEvent(event)
        
        # 确保选中状态正确同步
        self.on_selection_changed()

    def find_items_by_label(self, label):
        """Find all items with the given label."""
        items = []
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.UserRole) == label:
                items.append(item)
        return items

    def create_item_from_label(self, label):
        """Create a new QListWidgetItem for the given label."""
        item = QtWidgets.QListWidgetItem()
        item.setData(Qt.UserRole, label)
        # Use a checkbox to control the visibility of the label.
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)  # Initially visible
        return item

    def set_item_label(self, item, label, color=None, opacity=128):
        """设置标签项的文本和颜色（仅影响显示，不影响选中/勾选状态）"""
        item.setText(label)
        if color:
            background_color = QtGui.QColor(*color, opacity)
            item.setBackground(background_color)

    def update_item_color(self, label, color, opacity=255):
        """更新所有同名标签项的颜色（不影响选中/勾选状态）"""
        items = self.find_items_by_label(label)
        for item in items:
            background_color = QtGui.QColor(*color, opacity)
            item.setBackground(background_color)

    def remove_items_by_label(self, label):
        """Remove all items with the given label."""
        items = self.find_items_by_label(label)
        for item in items:
            row = self.row(item)
            self.takeItem(row)

    def on_item_changed(self, item):
        """checkbox勾选变化时，仅发可见性信号，不影响选中高亮"""
        label = item.data(Qt.UserRole)
        is_visible = item.checkState() == Qt.Checked
        self.label_visibility_changed.emit(label, is_visible)

    def get_visible_labels(self):
        """Get a list of all visible labels."""
        visible_labels = []
        for row in range(self.count()):
            item = self.item(row)
            if item.checkState() == Qt.Checked:
                visible_labels.append(item.data(Qt.UserRole))
        return visible_labels

    def get_hidden_labels(self):
        """Get a list of all hidden labels."""
        hidden_labels = []
        for row in range(self.count()):
            item = self.item(row)
            if item.checkState() == Qt.Unchecked:
                hidden_labels.append(item.data(Qt.UserRole))
        return hidden_labels

    def select_all_labels(self):
        """Select all items in the list."""
        self.selectAll()

    def deselect_all_labels(self):
        """Deselect all items in the list."""
        self.clearSelection()

    def invert_selection(self):
        """Invert the selection of all items in the list."""
        for row in range(self.count()):
            item = self.item(row)
            item.setSelected(not item.isSelected())

    def show_all_labels(self):
        """Set all labels to be visible."""
        for row in range(self.count()):
            item = self.item(row)
            item.setCheckState(Qt.Checked)

    def hide_all_labels(self):
        """Set all labels to be hidden."""
        for row in range(self.count()):
            item = self.item(row)
            item.setCheckState(Qt.Unchecked)

    def invert_visibility(self):
        """Invert the visibility of all labels."""
        for row in range(self.count()):
            item = self.item(row)
            item.setCheckState(
                Qt.Checked
                if item.checkState() == Qt.Unchecked
                else Qt.Unchecked
            )

    def on_selection_changed(self):
        # 选中项变化时发射信号
        self.selection_changed.emit()
