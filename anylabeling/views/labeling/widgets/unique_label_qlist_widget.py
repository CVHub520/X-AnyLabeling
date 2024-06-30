# -*- encoding: utf-8 -*-

import html

from PyQt5 import QtWidgets, QtGui
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

    def set_item_label(self, item, label, color=None, opacity=255):
        qlabel = QtWidgets.QLabel()
        if color is None:
            qlabel.setText(f"{label}")
        else:
            qlabel.setText("{}".format(html.escape(label)))
            background_color = QtGui.QColor(*color, opacity)
            style_sheet = (
                f"background-color: rgba("
                f"{background_color.red()}, "
                f"{background_color.green()}, "
                f"{background_color.blue()}, "
                f"{background_color.alpha()}"
                ");"
            )
            qlabel.setStyleSheet(style_sheet)
        qlabel.setAlignment(Qt.AlignBottom)
        item.setSizeHint(qlabel.sizeHint())
        self.setItemWidget(item, qlabel)

    def update_item_color(self, label, color, opacity=255):
        items = self.find_items_by_label(label)
        for item in items:
            qlabel = self.itemWidget(item)
            if qlabel:
                background_color = QtGui.QColor(*color, opacity)
                style_sheet = (
                    f"background-color: rgba("
                    f"{background_color.red()}, "
                    f"{background_color.green()}, "
                    f"{background_color.blue()}, "
                    f"{background_color.alpha()}"
                    ");"
                )
                qlabel.setStyleSheet(style_sheet)
                break

    def remove_items_by_label(self, label):
        items = self.find_items_by_label(label)
        for item in items:
            row = self.row(item)
            self.takeItem(row)
