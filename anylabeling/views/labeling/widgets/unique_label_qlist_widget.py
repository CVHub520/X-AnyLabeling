# -*- encoding: utf-8 -*-

import html

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from .escapable_qlist_widget import EscapableQListWidget


class UniqueLabelQListWidget(EscapableQListWidget):
    # QT Overload
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.clearSelection()

    def find_items_by_label(self, label):
        items = []
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.UserRole) == label:
                items.append(item)
        return items

    def create_item_from_label(self, label):
        item = QtWidgets.QListWidgetItem()
        item.setData(Qt.UserRole, label)
        return item

    def set_item_label(self, item, label, color=None):
        qlabel = QtWidgets.QLabel()
        if color is None:
            qlabel.setText(f"{label}")
        else:
            qlabel.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    html.escape(label), *color
                )
            )
        qlabel.setAlignment(Qt.AlignBottom)
        item.setSizeHint(qlabel.sizeHint())
        self.setItemWidget(item, qlabel)
