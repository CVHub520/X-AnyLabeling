from __future__ import annotations

from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.utils.qt import new_icon
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_lineedit_style,
    get_normal_button_style,
    get_ok_btn_style,
)

from .style import (
    get_danger_button_style,
    get_primary_button_style,
    get_secondary_button_style,
)


class PPOCRServiceUnavailableDialog(QDialog):
    def __init__(self, server_url: str, details: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Service Unavailable"))
        self.resize(560, 360)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                self.tr(
                    "PaddleOCR service is unavailable. Existing parsed files "
                    "can still be viewed and edited locally."
                )
            )
        )
        layout.addWidget(QLabel(self.tr("Server URL:")))
        layout.addWidget(QLabel(server_url))
        layout.addWidget(QLabel(self.tr("Details:")))

        self.details_edit = QPlainTextEdit()
        self.details_edit.setPlainText(details)
        self.details_edit.setReadOnly(True)
        layout.addWidget(self.details_edit, 1)

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        self.copy_button = QPushButton(self.tr("Copy Details"))
        self.copy_button.setStyleSheet(get_secondary_button_style())
        self.copy_button.clicked.connect(self.details_edit.selectAll)
        self.copy_button.clicked.connect(self.details_edit.copy)
        self.close_button = QPushButton(self.tr("Got it"))
        self.close_button.setStyleSheet(get_primary_button_style())
        self.close_button.clicked.connect(self.accept)
        buttons_layout.addWidget(self.copy_button)
        buttons_layout.addWidget(self.close_button)
        layout.addLayout(buttons_layout)


class PPOCRConfirmDeleteDialog(QDialog):
    def __init__(self, filename: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Delete File"))
        self.resize(460, 180)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                self.tr(
                    "Are you sure you want to delete this file? This action "
                    "cannot be undone and will remove all associated data."
                )
            )
        )
        layout.addWidget(QLabel(filename))

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(get_secondary_button_style())
        self.cancel_button.clicked.connect(self.reject)
        self.delete_button = QPushButton(self.tr("Delete"))
        self.delete_button.setStyleSheet(get_danger_button_style())
        self.delete_button.clicked.connect(self.accept)
        buttons_layout.addWidget(self.cancel_button)
        buttons_layout.addWidget(self.delete_button)
        layout.addLayout(buttons_layout)


class PPOCRApiSettingsDialog(QDialog):
    def __init__(
        self,
        parent=None,
        current_api_url: str = "",
        current_api_key: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("PPOCR API Settings"))
        self.setMinimumWidth(600)
        self._current_api_url = str(current_api_url or "").strip()
        self._current_api_key = str(current_api_key or "").strip()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.api_url_input = QLineEdit("")
        self.api_url_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_url_input.setPlaceholderText("API_URL")
        self.api_url_input.setStyleSheet(get_lineedit_style())
        self.api_url_input.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        url_row = QHBoxLayout()
        url_row.setContentsMargins(0, 0, 0, 0)
        url_row.setSpacing(8)
        url_row.addWidget(self.api_url_input)

        self.toggle_api_url_visibility_btn = QPushButton()
        self.toggle_api_url_visibility_btn.setCheckable(True)
        self.toggle_api_url_visibility_btn.setFixedSize(
            self.api_url_input.sizeHint().height(),
            self.api_url_input.sizeHint().height(),
        )
        self.toggle_api_url_visibility_btn.setStyleSheet(
            get_normal_button_style()
        )
        self.toggle_api_url_visibility_btn.clicked.connect(
            self.toggle_api_url_visibility
        )
        url_row.addWidget(self.toggle_api_url_visibility_btn)
        layout.addLayout(url_row)

        self.api_key_input = QLineEdit("")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("API_KEY")
        self.api_key_input.setStyleSheet(get_lineedit_style())
        self.api_key_input.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        key_row = QHBoxLayout()
        key_row.setContentsMargins(0, 0, 0, 0)
        key_row.setSpacing(8)
        key_row.addWidget(self.api_key_input)

        self.toggle_api_key_visibility_btn = QPushButton()
        self.toggle_api_key_visibility_btn.setCheckable(True)
        self.toggle_api_key_visibility_btn.setFixedSize(
            self.api_key_input.sizeHint().height(),
            self.api_key_input.sizeHint().height(),
        )
        self.toggle_api_key_visibility_btn.setStyleSheet(
            get_normal_button_style()
        )
        self.toggle_api_key_visibility_btn.clicked.connect(
            self.toggle_api_key_visibility
        )
        key_row.addWidget(self.toggle_api_key_visibility_btn)
        layout.addLayout(key_row)

        self.description_label = QLabel(
            self.tr(
                'Get API_URL and API_KEY from the <a href="https://aistudio.baidu.com/paddleocr/task">PaddleOCR website</a>.<br/>See the API invocation examples.'
            )
        )
        self.description_label.setWordWrap(True)
        desc_font = QFont(self.description_label.font())
        desc_font.setPointSize(max(9, desc_font.pointSize() - 1))
        self.description_label.setFont(desc_font)
        self.description_label.setOpenExternalLinks(True)
        self.description_label.setTextFormat(Qt.TextFormat.RichText)
        self.description_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self.description_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        cancel_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Cancel
        )
        if ok_button:
            ok_button.setStyleSheet(get_ok_btn_style())
            ok_button.setIcon(QIcon())
        if cancel_button:
            cancel_button.setStyleSheet(get_cancel_btn_style())
            cancel_button.setIcon(QIcon())
        self.button_box.accepted.connect(self.accept_with_validation)
        self.button_box.rejected.connect(self.reject)
        footer_row = QHBoxLayout()
        footer_row.setContentsMargins(0, 0, 0, 0)
        footer_row.setSpacing(10)
        footer_row.addWidget(
            self.description_label,
            1,
            Qt.AlignmentFlag.AlignVCenter,
        )
        footer_row.addWidget(
            self.button_box,
            0,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
        )
        layout.addLayout(footer_row)

        self.api_url_input.setText(self._current_api_url)
        self.api_key_input.setText(self._current_api_key)
        self._update_visibility_buttons(False, False)

    def _update_visibility_buttons(
        self, show_url: bool, show_key: bool
    ) -> None:
        if show_url:
            self.toggle_api_url_visibility_btn.setIcon(
                QIcon(new_icon("eye", "svg"))
            )
            self.toggle_api_url_visibility_btn.setToolTip(self.tr("Hide"))
        else:
            self.toggle_api_url_visibility_btn.setIcon(
                QIcon(new_icon("eye-off", "svg"))
            )
            self.toggle_api_url_visibility_btn.setToolTip(self.tr("Show"))

        if show_key:
            self.toggle_api_key_visibility_btn.setIcon(
                QIcon(new_icon("eye", "svg"))
            )
            self.toggle_api_key_visibility_btn.setToolTip(self.tr("Hide"))
        else:
            self.toggle_api_key_visibility_btn.setIcon(
                QIcon(new_icon("eye-off", "svg"))
            )
            self.toggle_api_key_visibility_btn.setToolTip(self.tr("Show"))

    def toggle_api_url_visibility(self, checked: bool) -> None:
        self.api_url_input.setEchoMode(
            QLineEdit.EchoMode.Normal
            if checked
            else QLineEdit.EchoMode.Password
        )
        self._update_visibility_buttons(
            checked,
            self.toggle_api_key_visibility_btn.isChecked(),
        )

    def toggle_api_key_visibility(self, checked: bool) -> None:
        self.api_key_input.setEchoMode(
            QLineEdit.EchoMode.Normal
            if checked
            else QLineEdit.EchoMode.Password
        )
        self._update_visibility_buttons(
            self.toggle_api_url_visibility_btn.isChecked(),
            checked,
        )

    def get_api_url(self) -> str:
        value = self.api_url_input.text().strip()
        if value:
            return value
        return self._current_api_url

    def get_api_key(self) -> str:
        value = self.api_key_input.text().strip()
        if value:
            return value
        return self._current_api_key

    def accept_with_validation(self) -> None:
        if not self.get_api_url() or not self.get_api_key():
            QMessageBox.warning(
                self,
                self.tr("Invalid Settings"),
                self.tr("API_URL and API_KEY are required."),
            )
            return
        self.accept()


class PPOCRFilterRadioButton(QRadioButton):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._indicator_size = 16
        self._indicator_inner_size = 6
        self._label_spacing = 8
        self._text_color = QColor(45, 49, 64)
        self._unchecked_border = QColor(214, 220, 235)
        self._unchecked_fill = QColor(248, 250, 255)
        self._checked_fill = QColor(76, 93, 255)
        self._checked_inner = QColor(255, 255, 255)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:
        text_width = self.fontMetrics().horizontalAdvance(self.text())
        text_height = self.fontMetrics().height()
        width = self._indicator_size + self._label_spacing + text_width + 2
        height = max(self._indicator_size, text_height) + 4
        return QSize(width, height)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        indicator_y = (self.height() - self._indicator_size) / 2
        indicator_rect = QRectF(
            0,
            indicator_y,
            self._indicator_size,
            self._indicator_size,
        )

        if self.isChecked():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._checked_fill)
            painter.drawEllipse(indicator_rect)

            inner_offset = (
                self._indicator_size - self._indicator_inner_size
            ) / 2
            inner_rect = QRectF(
                indicator_rect.left() + inner_offset,
                indicator_rect.top() + inner_offset,
                self._indicator_inner_size,
                self._indicator_inner_size,
            )
            painter.setBrush(self._checked_inner)
            painter.drawEllipse(inner_rect)
        else:
            painter.setPen(self._unchecked_border)
            painter.setBrush(self._unchecked_fill)
            painter.drawEllipse(indicator_rect)

        text_rect = QRectF(
            self._indicator_size + self._label_spacing,
            0,
            self.width() - self._indicator_size - self._label_spacing,
            self.height(),
        )
        painter.setPen(self._text_color)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self.text(),
        )
        painter.end()


class PPOCRFilterDialog(QWidget):
    filtersConfirmed = pyqtSignal(str, str, str)

    def __init__(
        self,
        sort_mode: str,
        file_type: str,
        status: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet(
            "QLabel { font-size: 13px; font-weight: 600; color: rgb(38, 42, 56); }"
        )
        self.resize(440, 0)
        self._arrow_center_x = 14
        self._corner_radius = 14
        self._arrow_height = 10
        self._arrow_half_width = 9
        self._panel_margin = 8
        self._halo_width = 10
        self._outer_spacing = 20
        self._column_spacing = 16
        self._row_spacing = 24
        self._section_spacing = 32
        self._button_spacing = 8
        self._label_width = (
            max(
                self.fontMetrics().horizontalAdvance(text)
                for text in (
                    self.tr("Sort"),
                    self.tr("File Type"),
                )
            )
            + 6
        )

        self._sort_buttons = QButtonGroup(self)
        self._file_type_buttons = QButtonGroup(self)
        self._status_buttons = QButtonGroup(self)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(
            self._panel_margin,
            0,
            self._panel_margin,
            self._panel_margin,
        )
        outer_layout.setSpacing(0)

        self.content_widget = QWidget(self)
        self.content_widget.setStyleSheet(
            "QWidget { background: transparent; }"
        )
        layout = QVBoxLayout(self.content_widget)
        layout.setContentsMargins(
            self._outer_spacing,
            self._outer_spacing,
            self._outer_spacing,
            self._outer_spacing,
        )
        layout.setSpacing(0)

        layout.addLayout(
            self._build_radio_row(
                self.tr("Sort"),
                [
                    (self.tr("Newest First"), "newest"),
                    (self.tr("Oldest First"), "oldest"),
                ],
                sort_mode,
                self._sort_buttons,
            )
        )
        layout.addSpacing(self._row_spacing)
        layout.addLayout(
            self._build_radio_row(
                self.tr("File Type"),
                [
                    (self.tr("All"), "all"),
                    (self.tr("Document"), "pdf"),
                    (self.tr("Image"), "image"),
                ],
                file_type,
                self._file_type_buttons,
            )
        )
        layout.addSpacing(self._row_spacing)
        layout.addLayout(
            self._build_radio_row(
                self.tr("Parsing Status"),
                [
                    (self.tr("All"), "all"),
                    (self.tr("Parsing"), "pending"),
                    (self.tr("Failed"), "error"),
                    (self.tr("Completed"), "parsed"),
                ],
                status,
                self._status_buttons,
            )
        )
        layout.addSpacing(self._section_spacing)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(self._button_spacing)
        buttons_layout.addStretch()
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(
            get_secondary_button_style()
            + "QPushButton { color: rgb(86, 87, 114); font-size: 13px; }"
        )
        self.cancel_button.clicked.connect(self.close)
        self.confirm_button = QPushButton(self.tr("Confirm"))
        self.confirm_button.setStyleSheet(
            get_primary_button_style() + "QPushButton { font-size: 13px; }"
        )
        self.confirm_button.clicked.connect(self._confirm)
        buttons_layout.addWidget(self.cancel_button)
        buttons_layout.addWidget(self.confirm_button)
        layout.addLayout(buttons_layout)
        outer_layout.addSpacing(self._arrow_height)
        outer_layout.addWidget(self.content_widget)

    def _build_radio_row(
        self,
        label_text: str,
        items: list[tuple[str, str]],
        selected_value: str,
        button_group: QButtonGroup,
    ):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(self._column_spacing)

        label = QLabel(label_text)
        label.setWordWrap(True)
        label.setFixedWidth(self._label_width)
        label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        row.addWidget(label, 0, Qt.AlignmentFlag.AlignTop)

        options_layout = QHBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(self._column_spacing)
        for index, (text, value) in enumerate(items):
            button = PPOCRFilterRadioButton(text)
            button.setProperty("value", value)
            if value == selected_value:
                button.setChecked(True)
            button_group.addButton(button, index)
            options_layout.addWidget(button)
        options_layout.addStretch()
        row.addLayout(options_layout, 1)
        return row

    def selected_filters(self) -> tuple[str, str, str]:
        return (
            self._checked_value(self._sort_buttons),
            self._checked_value(self._file_type_buttons),
            self._checked_value(self._status_buttons),
        )

    def set_anchor_offset(self, center_x: int) -> None:
        self._arrow_center_x = max(
            self._arrow_half_width + 6,
            center_x,
        )
        self.update()

    def panel_margin(self) -> int:
        return self._panel_margin

    @staticmethod
    def _checked_value(button_group: QButtonGroup) -> str:
        button = button_group.checkedButton()
        if button is None:
            return ""
        return str(button.property("value") or "")

    def _confirm(self) -> None:
        self.filtersConfirmed.emit(*self.selected_filters())
        self.close()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        rect = QRectF(
            self._panel_margin,
            self._arrow_height,
            self.width() - self._panel_margin * 2,
            self.height() - self._arrow_height - self._panel_margin,
        )
        panel_path = QPainterPath()
        panel_path.addRoundedRect(
            rect,
            self._corner_radius,
            self._corner_radius,
        )
        arrow = QPainterPath()
        center_x = min(
            max(
                self._arrow_center_x,
                int(rect.left()) + self._arrow_half_width + 4,
            ),
            int(rect.right()) - self._arrow_half_width - 4,
        )
        arrow.moveTo(center_x, 2)
        arrow.lineTo(center_x - self._arrow_half_width, self._arrow_height + 2)
        arrow.lineTo(center_x + self._arrow_half_width, self._arrow_height + 2)
        arrow.closeSubpath()
        bubble_path = panel_path.united(arrow)

        halo_stroker = QPainterPathStroker()
        halo_stroker.setWidth(self._halo_width)
        halo_stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        halo_stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
        halo_path = halo_stroker.createStroke(bubble_path)

        painter.setBrush(QColor(36, 44, 80, 18))
        painter.drawPath(halo_path)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawPath(bubble_path)
        painter.end()
