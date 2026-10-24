# -*- encoding: utf-8 -*-

import html

from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtCore import Qt

from .escapable_qlist_widget import EscapableQListWidget
from .label_list_widget import HTMLDelegate, LOCKED_ROLE


class UniqueLabelQListWidget(EscapableQListWidget):
    items_lock_requested = QtCore.pyqtSignal(list)
    label_visibility_changed = QtCore.pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lock_states = {}
        self.setItemDelegate(
            HTMLDelegate(self, highlight_selected_check=False)
        )
        self.setMouseTracking(True)
        self.model().dataChanged.connect(self._on_data_changed)

    def _on_data_changed(self, top_left, bottom_right, roles):
        if Qt.ItemDataRole.CheckStateRole not in roles:
            return
        for row in range(top_left.row(), bottom_right.row() + 1):
            item = self.item(row)
            self.label_visibility_changed.emit(
                item.data(Qt.ItemDataRole.UserRole),
                item.checkState() == Qt.CheckState.Checked,
            )

    # QT Overload
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                if not item.isSelected():
                    self.clearSelection()
                    item.setSelected(True)
                self.items_lock_requested.emit(self.selectedItems())
            event.accept()
            return
        super().mousePressEvent(event)
        if not self.indexAt(event.position().toPoint()).isValid():
            self.clearSelection()

    def find_items_by_label(self, label):
        items = []
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == label:
                items.append(item)
        return items

    def create_item_from_label(self, label):
        item = QtWidgets.QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, label)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        return item

    def set_item_label(self, item, label, color=None, opacity=255):
        item.setText(html.escape(label))
        if color is not None:
            item.setBackground(QtGui.QColor(*color, opacity))
        self.set_item_locked(item, self.lock_states.get(label, False))

    def set_item_locked(self, item, locked):
        item.setData(LOCKED_ROLE, bool(locked))

    def update_item_color(self, label, color, opacity=255):
        for item in self.find_items_by_label(label):
            item.setBackground(QtGui.QColor(*color, opacity))

    def remove_items_by_label(self, label):
        items = self.find_items_by_label(label)
        for item in items:
            row = self.row(item)
            self.takeItem(row)
