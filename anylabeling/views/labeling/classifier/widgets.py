from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.classifier.style import get_overlay_text_style
from anylabeling.views.labeling.utils.theme import get_theme


class ClassificationImagePreview(QLabel):
    def image_rect(self):
        image_area = QRectF(self.rect()).adjusted(10.5, 10.5, -10.5, -10.5)
        pixmap = self.pixmap()
        if pixmap.isNull():
            return image_area
        image_size = pixmap.deviceIndependentSize()
        image_size.scale(image_area.size(), Qt.AspectRatioMode.KeepAspectRatio)
        image_rect = QRectF(image_area.topLeft(), image_size)
        image_rect.moveCenter(image_area.center())
        return image_rect

    def paintEvent(self, event):
        theme = get_theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        frame = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        background = QLinearGradient(frame.topLeft(), frame.bottomRight())
        top_color = QColor(theme["background_secondary"])
        top_color.setAlpha(220)
        bottom_color = QColor(theme["surface"])
        bottom_color.setAlpha(180)
        background.setColorAt(0, top_color)
        background.setColorAt(1, bottom_color)
        painter.setBrush(background)
        painter.setPen(QPen(QColor(theme["border"]), 1))
        painter.drawRoundedRect(frame, 24, 24)

        pixmap = self.pixmap()
        if pixmap.isNull():
            return

        image_rect = self.image_rect()
        clip = QPainterPath()
        clip.addRoundedRect(image_rect, 18, 18)
        painter.setClipPath(clip)
        painter.drawPixmap(image_rect, pixmap, QRectF(pixmap.rect()))


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
            image_rect = parent_widget.image_rect()
            self.move(
                round(image_rect.right() - self.width()),
                round(image_rect.top()),
            )
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        image_rect = (
            self.parentWidget().image_rect().translated(-self.x(), -self.y())
        )
        image_clip = QPainterPath()
        image_clip.addRoundedRect(image_rect, 18, 18)
        background = QPainterPath()
        background.addRect(QRectF(self.rect()))
        painter.fillPath(background.intersected(image_clip), QColor(0, 0, 0))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            self.rect().adjusted(4, 4, -4, -4),
            Qt.AlignmentFlag.AlignCenter,
            self.text(),
        )


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
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            text = self.text().strip()
            if not text:
                if self.classifier_dialog:
                    self.classifier_dialog.restore_current_page_number()
                return
            if self.classifier_dialog:
                self.classifier_dialog.jump_to_page(int(text))
            return
        super().keyPressEvent(event)
