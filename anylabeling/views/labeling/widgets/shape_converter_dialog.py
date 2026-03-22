import json

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.shape import (
    CONVERSION_MODE_MAP,
    CONVERSION_TARGETS,
    _apply_shape_conversion,
    get_conversion_params,
)
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_dialog_style,
    get_msg_box_style,
    get_ok_btn_style,
    get_settings_combo_style,
    get_theme,
)
from anylabeling.views.labeling.widgets.popup import Popup


class ShapeConverterDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_widget = parent
        self.is_processing = False
        self.cancel_requested = False
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle(self.tr("Shape Converter"))
        self.resize(820, 520)
        self.setStyleSheet(get_dialog_style())
        theme = get_theme()

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        type_layout = QtWidgets.QHBoxLayout()
        self.source_combo = QtWidgets.QComboBox(self)
        self.source_combo.addItems(list(CONVERSION_TARGETS.keys()))
        self.target_combo = QtWidgets.QComboBox(self)
        self._combo_style = get_settings_combo_style()
        self.source_combo.setStyleSheet(self._combo_style)
        self.target_combo.setStyleSheet(self._combo_style)
        self._prepare_combo_popup(self.source_combo)
        self._prepare_combo_popup(self.target_combo)
        self.arrow_label = QtWidgets.QLabel(self)
        self.arrow_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.arrow_label.setMinimumWidth(56)
        arrow_pixmap = QtGui.QPixmap(new_icon_path("arrow-right", "svg"))
        if not arrow_pixmap.isNull():
            self.arrow_label.setPixmap(
                arrow_pixmap.scaled(
                    24,
                    24,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self.arrow_label.setText("->")
        type_layout.addWidget(self.source_combo, 1)
        type_layout.addWidget(self.arrow_label, 0)
        type_layout.addWidget(self.target_combo, 1)
        main_layout.addLayout(type_layout)

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {theme["surface"]};
                border: 1px solid {theme["border_light"]};
                border-radius: 0px;
                text-align: center;
                color: {theme["text"]};
                font-size: 12px;
                font-weight: 500;
                min-height: 22px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0066FF,
                    stop:0.5 #00A6FF,
                    stop:1 #0066FF);
                border-radius: 0px;
            }}
            """)
        main_layout.addWidget(self.progress_bar)

        self.log_terminal = QtWidgets.QListWidget(self)
        self.log_terminal.setIconSize(QtCore.QSize(16, 16))
        self.log_terminal.setSpacing(2)
        self.log_terminal.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.log_terminal.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.log_terminal.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.log_terminal.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.log_terminal.setStyleSheet(f"""
            QListWidget {{
                background-color: {theme["background_secondary"]};
                color: {theme["text"]};
                border: 1px solid {theme["border"]};
                border-radius: 0px;
                padding: 6px;
                outline: none;
            }}
            QListWidget::item {{
                border: none;
                padding: 6px 8px;
                margin: 2px 0px;
            }}
            QListWidget::item:selected {{
                background-color: {theme["selection"]};
                color: {theme["selection_text"]};
            }}
            """)
        main_layout.addWidget(self.log_terminal, 1)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(get_cancel_btn_style())
        self.confirm_button = QtWidgets.QPushButton(self.tr("Confirm"))
        self.confirm_button.setStyleSheet(get_ok_btn_style())
        self.confirm_button.setDefault(True)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.confirm_button)
        main_layout.addLayout(button_layout)

        self.source_combo.currentTextChanged.connect(
            self._update_target_options
        )
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.confirm_button.clicked.connect(self._on_confirm_clicked)
        self._update_target_options(self.source_combo.currentText())
        self.check_icon = QtGui.QIcon(new_icon_path("check", "svg"))
        self.error_icon = QtGui.QIcon(new_icon_path("error", "svg"))
        self._set_progress(0, self._get_initial_total())

    def _get_initial_total(self):
        label_total = len(self.parent_widget.get_label_file_list())
        if label_total > 0:
            return label_total
        image_list = getattr(self.parent_widget, "image_list", None)
        if image_list:
            return len(image_list)
        if getattr(self.parent_widget, "filename", None):
            return 1
        return 0

    def _update_target_options(self, source_type):
        self.target_combo.clear()
        self.target_combo.addItems(CONVERSION_TARGETS.get(source_type, []))
        self.target_combo.setStyleSheet(self._combo_style)
        self.target_combo.setMaxVisibleItems(
            min(12, max(1, self.target_combo.count()))
        )

    def _prepare_combo_popup(self, combo):
        popup_view = QtWidgets.QListView(combo)
        combo.setView(popup_view)
        combo.setMaxVisibleItems(min(12, max(1, combo.count())))
        popup_view.setUniformItemSizes(True)
        popup_view.setMouseTracking(True)
        popup_view.viewport().setMouseTracking(True)
        popup_view.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        popup_view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        popup_view.doItemsLayout()

    def _set_progress(self, current, total):
        safe_total = max(total, 1)
        self.progress_bar.setRange(0, safe_total)
        self.progress_bar.setValue(min(current, safe_total))
        percent = int(current * 100 / safe_total)
        self.progress_bar.setFormat(f"{current}/{total} ({percent}%)")

    def _append_success_log(self, file_path):
        item = QtWidgets.QListWidgetItem(self.check_icon, f" {file_path}")
        self.log_terminal.addItem(item)
        self.log_terminal.scrollToBottom()

    def _append_error_log(self, file_path, error):
        item = QtWidgets.QListWidgetItem(
            self.error_icon, f" {file_path} ({error})"
        )
        self.log_terminal.addItem(item)
        self.log_terminal.scrollToBottom()

    def _set_controls_enabled(self, enabled):
        self.source_combo.setEnabled(enabled)
        self.target_combo.setEnabled(enabled)
        self.confirm_button.setEnabled(enabled)
        self.cancel_button.setEnabled(True)

    def _confirm_conversion(self):
        response = QtWidgets.QMessageBox(self)
        response.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        response.setWindowTitle(self.tr("Warning"))
        response.setText(self.tr("Current annotation will be changed"))
        response.setInformativeText(
            self.tr("Are you sure you want to perform this conversion?")
        )
        response.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Cancel
            | QtWidgets.QMessageBox.StandardButton.Ok
        )
        response.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Ok)
        response.setStyleSheet(get_msg_box_style())
        return response.exec() == QtWidgets.QMessageBox.StandardButton.Ok

    def _on_cancel_clicked(self):
        if self.is_processing:
            self.cancel_requested = True
            self.cancel_button.setEnabled(False)
            return
        self.reject()

    def _on_confirm_clicked(self):
        if self.is_processing:
            return

        mode = CONVERSION_MODE_MAP.get(
            (self.source_combo.currentText(), self.target_combo.currentText())
        )
        if mode is None:
            return

        label_file_list = self.parent_widget.get_label_file_list()
        if len(label_file_list) == 0:
            return

        if not self._confirm_conversion():
            return

        params = get_conversion_params(self, mode)
        if params is None:
            return

        self.log_terminal.clear()
        total_files = len(label_file_list)
        self._set_progress(0, total_files)
        self.cancel_requested = False
        self.is_processing = True
        self._set_controls_enabled(False)

        processed_count = 0
        error_count = 0

        for index, label_file in enumerate(label_file_list, start=1):
            QtWidgets.QApplication.processEvents()
            if self.cancel_requested:
                break
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                _apply_shape_conversion(data, mode, params)
                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                self._append_success_log(label_file)
            except Exception as e:
                error_count += 1
                logger.error(
                    f"Error occurred while converting {label_file}: {e}"
                )
                self._append_error_log(label_file, e)
            processed_count = index
            self._set_progress(processed_count, total_files)
            QtWidgets.QApplication.processEvents()

        was_canceled = self.cancel_requested
        self.is_processing = False
        self.cancel_requested = False
        self._set_controls_enabled(True)

        if processed_count > 0 and self.parent_widget.filename:
            self.parent_widget.load_file(self.parent_widget.filename)

        if was_canceled:
            popup = Popup(
                self.tr("Conversion canceled."),
                self.parent_widget,
                msec=1200,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self.parent_widget, position="center")
        elif error_count > 0:
            popup = Popup(
                self.tr("Conversion completed with errors."),
                self.parent_widget,
                msec=1200,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self.parent_widget, position="center")
        else:
            popup = Popup(
                self.tr("Conversion completed successfully!"),
                self.parent_widget,
                msec=1000,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(
                self.parent_widget, popup_height=65, position="center"
            )
