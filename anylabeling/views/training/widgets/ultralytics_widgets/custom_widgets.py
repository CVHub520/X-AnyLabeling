from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor

from anylabeling.services.auto_training.ultralytics.config import *
from anylabeling.services.auto_training.ultralytics.style import *
from anylabeling.views.labeling.utils.qt import new_icon_path


class CustomCheckBox(QCheckBox):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            f"""
            QCheckBox {{
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 1px solid #d2d2d7;
                background-color: white;
            }}
            QCheckBox::indicator:checked {{
                background-color: white;
                border: 1px solid #d2d2d7;
                image: url({new_icon_path("checkmark", "svg")});
            }}
        """
        )


class CustomComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            """
            QComboBox {
                padding: 5px 8px;
                background: white;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
                min-height: 24px;
                selection-background-color: #0071e3;
                color: #1d1d1f;
            }
            QComboBox:hover {
                border-color: #0071e3;
            }
            QComboBox:focus {
                border-color: #0071e3;
                outline: none;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border: none;
                background: #f0f0f0;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            QComboBox::drop-down:hover {
                background: #e0e0e0;
            }
            QComboBox::down-arrow {
                image: url("""
            + new_icon_path("caret-down", "svg")
            + """);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background: white;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
                padding: 4px;
                selection-background-color: #0071e3;
                selection-color: white;
                color: #1d1d1f;
            }
            QComboBox QAbstractItemView::item {
                padding: 6px 8px;
                border-radius: 4px;
                min-height: 20px;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #f0f8ff;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #0071e3;
                color: white;
            }
        """
        )

    def wheelEvent(self, event):
        event.ignore()


class CustomSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            """
            QSpinBox {
                padding: 5px 8px;
                background: white;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
                min-height: 24px;
                selection-background-color: #0071e3;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
                border: none;
                background: #f0f0f0;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: #e0e0e0;
            }
            QSpinBox::up-arrow {
                image: url("""
            + new_icon_path("caret-up", "svg")
            + """);
                width: 12px;
                height: 12px;
            }
            QSpinBox::down-arrow {
                image: url("""
            + new_icon_path("caret-down", "svg")
            + """);
                width: 12px;
                height: 12px;
            }
        """
        )

    def wheelEvent(self, event):
        event.ignore()


class CustomDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            """
            QDoubleSpinBox {
                padding: 5px 8px;
                background: white;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
                min-height: 24px;
                selection-background-color: #0071e3;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 20px;
                border: none;
                background: #f0f0f0;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background: #e0e0e0;
            }
            QDoubleSpinBox::up-arrow {
                image: url("""
            + new_icon_path("caret-up", "svg")
            + """);
                width: 12px;
                height: 12px;
            }
            QDoubleSpinBox::down-arrow {
                image: url("""
            + new_icon_path("caret-down", "svg")
            + """);
                width: 12px;
                height: 12px;
            }
        """
        )

    def wheelEvent(self, event):
        event.ignore()


class CustomSlider(QSlider):
    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setStyleSheet(
            """
            QSlider {
                height: 28px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #d2d2d7;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0071e3;
                border: none;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #0071e3;
                border-radius: 2px;
            }
        """
        )

    def wheelEvent(self, event):
        event.ignore()


class CustomLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            """
            QLineEdit {
                border: 1px solid #E5E5E5;
                border-radius: 8;
                background-color: #F9F9F9;
                font-size: 13px;
                height: 36px;
                padding-left: 4px;
            }
            QLineEdit:hover {
                background-color: #DBDBDB;
                border-radius: 8px;
            }
            QLineEdit:focus {
                border: 3px solid "#60A5FA";
                background-color: "#F9F9F9";
            }
        """
        )


class CustomQPushButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(32)
        self.setMinimumWidth(80)
        self.selected = False
        self.setFocusPolicy(Qt.NoFocus)
        self.update_style()

    def set_selected(self, selected):
        self.selected = selected
        self.update_style()

    def update_style(self):
        if self.selected:
            self.setStyleSheet(
                """
                QPushButton {
                    background-color: #007ACC;
                    color: white;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                    outline: none;
                }
                QPushButton:hover {
                    background-color: #005A9E;
                }
                QPushButton:focus {
                    outline: none;
                }
            """
            )
        else:
            self.setStyleSheet(
                """
                QPushButton {
                    background-color: #F0F0F0;
                    color: #333333;
                    border: 1px solid #CCCCCC;
                    border-radius: 4px;
                    padding: 4px 8px;
                    outline: none;
                }
                QPushButton:hover {
                    background-color: #E0E0E0;
                }
                QPushButton:pressed {
                    background-color: #D0D0D0;
                }
                QPushButton:focus {
                    outline: none;
                }
            """
            )


class PrimaryButton(QPushButton):
    def __init__(self, text="OK", parent=None):
        super().__init__(text, parent)
        self.setFixedSize(100, 32)
        self.setStyleSheet(
            """
            QPushButton {
                background-color: #0071e3;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0077ED;
            }
            QPushButton:pressed {
                background-color: #0068D0;
            }
        """
        )


class SecondaryButton(QPushButton):
    def __init__(self, text="Cancel", parent=None):
        super().__init__(text, parent)
        self.setFixedSize(100, 32)
        self.setStyleSheet(
            """
            QPushButton {
                background-color: #f5f5f7;
                color: #1d1d1f;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #e5e5e5;
            }
            QPushButton:pressed {
                background-color: #d5d5d5;
            }
        """
        )


class TrainingConfirmDialog(QDialog):
    def __init__(self, cmd_parts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start Training")
        self.setFixedSize(752, 320)
        self.setStyleSheet(
            """
            QDialog {
                background-color: white;
                border-radius: 8px;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title_label = QLabel("Ready to Start Training")
        title_label.setStyleSheet(
            """
            font-size: 16px;
            font-weight: bold;
            color: #1d1d1f;
        """
        )
        layout.addWidget(title_label)

        desc_label = QLabel("The following command will be executed:")
        desc_label.setStyleSheet("color: #6e6e73; font-size: 13px;")
        layout.addWidget(desc_label)

        cmd_text = self._format_command(cmd_parts)
        command_display = QTextEdit()
        command_display.setPlainText(cmd_text)
        command_display.setReadOnly(True)
        command_display.setFixedHeight(160)
        command_display.setStyleSheet(
            """
            QTextEdit {
                background-color: #2d3748;
                color: #e2e8f0;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #4a5568;
                border-radius: 6px;
                padding: 12px;
                line-height: 1.4;
            }
        """
        )
        layout.addWidget(command_display)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = SecondaryButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.start_btn = PrimaryButton("Start")
        self.start_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.start_btn)

        layout.addLayout(button_layout)

    def _format_command(self, cmd_parts):
        if len(cmd_parts) <= 3:
            return " ".join(cmd_parts)

        base_cmd = " ".join(cmd_parts[:3])
        params = cmd_parts[3:]

        formatted_cmd = base_cmd + " \\\n"
        for i, param in enumerate(params):
            if i == len(params) - 1:
                formatted_cmd += f"    {param}"
            else:
                formatted_cmd += f"    {param} \\\n"

        return formatted_cmd


class CustomTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_data = []
        self.setup_table()

    def setup_table(self):
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.verticalHeader().setVisible(False)
        self.setStyleSheet(get_custom_table_style())

        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)

    def load_data(self, table_data):
        if not table_data:
            self.clear()
            return

        self.original_data = table_data
        self.populate_table()

    def populate_table(self):
        if not self.original_data:
            return

        headers = self.original_data[0]
        data_rows = self.original_data[1:]

        self.setRowCount(len(data_rows))
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)

        for row, row_data in enumerate(data_rows):
            for col, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)

                if col == 0:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

                self.setItem(row, col, item)
