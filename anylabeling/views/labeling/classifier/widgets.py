from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.classifier.style import get_overlay_text_style


class ClassificationOverlay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(get_overlay_text_style())
        self.hide()

    def update_text(self, text: str):
        if text:
            self.setText(text)
            self.adjustSize()
            self.show()
        else:
            self.hide()

    def position_overlay(self, parent_widget):
        if self.isVisible():
            parent_rect = parent_widget.rect()
            self.move(parent_rect.width() - self.width() - 10, 10)


class ClassificationCheckBoxGroup(QWidget):
    def __init__(self, labels, is_multiclass=True, parent=None):
        super().__init__(parent)
        self.labels = labels
        self.is_multiclass = is_multiclass
        self.checkboxes = {}
        self.button_group = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        for i, label in enumerate(self.labels):
            checkbox = QCheckBox(f"{label}({i})")
            checkbox.setObjectName(label)
            self.checkboxes[label] = checkbox

            if self.is_multiclass:
                checkbox.toggled.connect(self._handle_multiclass_toggle)

            layout.addWidget(checkbox)

    def _handle_multiclass_toggle(self, checked):
        if not self.is_multiclass:
            return

        sender = self.sender()
        if checked:
            for _, checkbox in self.checkboxes.items():
                if checkbox != sender and checkbox.isChecked():
                    checkbox.blockSignals(True)
                    checkbox.setChecked(False)
                    checkbox.blockSignals(False)

    def get_selected_flags(self):
        flags = {}
        for label, checkbox in self.checkboxes.items():
            flags[label] = checkbox.isChecked()
        return flags

    def set_flags(self, flags):
        for label, checkbox in self.checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(flags.get(label, False))
            checkbox.blockSignals(False)

    def clear_selection(self):
        for checkbox in self.checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)


class PageInputLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.classifier_dialog = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            text = self.text().strip()
            if not text:
                if self.classifier_dialog:
                    self.classifier_dialog.restore_current_page_number()
                return
            if self.classifier_dialog:
                self.classifier_dialog.jump_to_page(int(text))
            return
        super().keyPressEvent(event)
