from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt


class EscapableQListWidget(QtWidgets.QListWidget):
    # QT Overload
    def keyPressEvent(self, event):
        super(EscapableQListWidget, self).keyPressEvent(event)
        if event.key() == Qt.Key.Key_Escape:
            self.clearSelection()
