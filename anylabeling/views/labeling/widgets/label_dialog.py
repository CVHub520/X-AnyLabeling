import os
import re
import json

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont, QColor, QIntValidator
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtWidgets import (
    QColorDialog,
    QTableWidgetItem,
    QTableWidget,
    QCheckBox,
    QApplication,
)

from anylabeling.views.labeling import utils
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.widgets.popup import Popup
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import (
    get_ok_btn_style,
    get_spinbox_style,
)

# TODO(unknown):
# - Calculate optimal position so as not to go out of screen area.


def natural_sort_key(s):
    return [
        int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)
    ]


class ColoredComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None):
        super(ColoredComboBox, self).__init__(parent)
        self.mode_colors = {
            "polygon": QtGui.QColor("#D81B60"),  # Magenta
            "rectangle": QtGui.QColor("#1E88E5"),  # Bright Blue
            "rotation": QtGui.QColor("#8E24AA"),  # Purple
            "circle": QtGui.QColor("#00C853"),  # Bright Green
            "line": QtGui.QColor("#FF6D00"),  # Bright Orange
            "point": QtGui.QColor("#00ACC1"),  # Teal
            "linestrip": QtGui.QColor("#6D4C41"),  # Brown
        }

    def addModeItem(self, text, userData=None):
        self.addItem(text, userData)
        if text in self.mode_colors:
            index = self.count() - 1
            self.setItemData(
                index, self.mode_colors[text], QtCore.Qt.ForegroundRole
            )

    def paintEvent(self, event):
        painter = QtWidgets.QStylePainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.Text))

        # Draw the combobox frame, button, etc.
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)
        painter.drawComplexControl(QtWidgets.QStyle.CC_ComboBox, opt)

        # Draw the current text with proper color
        current_text = self.currentText()
        if current_text in self.mode_colors:
            painter.setPen(self.mode_colors[current_text])

        # Draw the text
        opt.currentText = current_text
        rect = self.style().subElementRect(
            QtWidgets.QStyle.SE_ComboBoxFocusRect, opt, self
        )
        rect.adjust(2, 0, -2, 0)  # adjust the text rectangle
        painter.drawText(
            rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, current_text
        )


class DigitShortcutDialog(QtWidgets.QDialog):
    """Dialog for managing digit shortcuts"""

    def __init__(self, parent=None):
        """
        Initialize the digit shortcut dialog.

        Args:
            parent: The parent widget.
        """
        super(DigitShortcutDialog, self).__init__(parent)

        self.parent = parent
        self.digit_shortcuts = {}
        if (
            hasattr(self.parent, "drawing_digit_shortcuts")
            and self.parent.drawing_digit_shortcuts is not None
        ):
            self.digit_shortcuts = self.parent.drawing_digit_shortcuts.copy()

        self.available_modes = [
            "polygon",
            "rectangle",
            "rotation",
            "circle",
            "line",
            "point",
            "linestrip",
        ]

        self.setWindowTitle(self.tr("Digit Shortcut Manager"))
        self.setModal(True)
        self.setMinimumSize(500, 435)
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint
        )

        self.setStyleSheet(
            f"""
                QDialog {{
                    background-color: #f5f5f7;
                    border-radius: 10px;
                }}
                QLabel {{
                    color: #1d1d1f;
                    font-size: 13px;
                }}
                QComboBox {{
                    padding: 2px 6px;
                    background: white;
                    border: 1px solid #d2d2d7;
                    border-radius: 4px;
                    min-height: 20px;
                    selection-background-color: #0071e3;
                }}
                QComboBox::drop-down {{
                    subcontrol-origin: padding;
                    subcontrol-position: center right;
                    width: 20px;
                    border: none;
                }}
                QComboBox::down-arrow {{
                    image: url({new_icon_path("caret-down", "svg")});
                    width: 12px;
                    height: 12px;
                }}
                QLineEdit {{
                    padding: 2px 6px;
                    background: white;
                    border: 1px solid #d2d2d7;
                    border-radius: 4px;
                    min-height: 20px;
                    selection-background-color: #0071e3;
                }}
                QLineEdit:disabled {{
                    background: #f0f0f0;
                    color: #999999;
                }}
                QHeaderView::section {{
                    background-color: #f0f0f0;
                    padding: 5px;
                    border: 1px solid #d2d2d7;
                    font-weight: bold;
                }}
        """
        )

        # Create layout with proper spacing
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Header label
        header_label = QtWidgets.QLabel(
            self.tr("Configure digit keys (0-9) for quick shape creation:")
        )
        header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header_label)

        # Create table for digit shortcuts
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setRowCount(10)  # 0-9 digits
        self.table.setHorizontalHeaderLabels(
            [self.tr("Digit"), self.tr("Drawing Mode"), self.tr("Label")]
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.Stretch
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Populate table
        for digit in range(10):
            # Digit column
            digit_item = QtWidgets.QTableWidgetItem(str(digit))
            digit_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(digit, 0, digit_item)

            # Mode combobox column
            mode_combo = ColoredComboBox()
            mode_combo.addItem(self.tr("None"), None)
            for mode in self.available_modes:
                mode_combo.addModeItem(mode, mode)

            # Set current mode if exists
            if (
                int(digit) in self.digit_shortcuts
                and "mode" in self.digit_shortcuts[int(digit)]
            ):
                mode = self.digit_shortcuts[int(digit)]["mode"]
                index = mode_combo.findData(mode)
                if index >= 0:
                    mode_combo.setCurrentIndex(index)

            # Connect mode change to enable/disable label field
            mode_combo.currentIndexChanged.connect(
                lambda idx, d=digit: self.on_mode_changed(d, idx)
            )
            self.table.setCellWidget(digit, 1, mode_combo)

            # Label text field
            label_edit = QtWidgets.QLineEdit()
            if (
                int(digit) in self.digit_shortcuts
                and "label" in self.digit_shortcuts[int(digit)]
            ):
                label_edit.setText(self.digit_shortcuts[int(digit)]["label"])

            # Initially disable if no mode is selected
            if (
                int(digit) not in self.digit_shortcuts
                or "mode" not in self.digit_shortcuts[int(digit)]
                or self.digit_shortcuts[int(digit)]["mode"] is None
            ):
                label_edit.setEnabled(False)

            self.table.setCellWidget(digit, 2, label_edit)

        layout.addWidget(self.table)

        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(8)

        self.reset_button = QtWidgets.QPushButton(self.tr("Reset"))
        self.reset_button.setFixedSize(100, 32)
        self.reset_button.clicked.connect(self.reset_settings)
        self.reset_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f5f5f7;
                color: #1d1d1f;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #e5e5e5;
            }
            QPushButton:pressed {
                background-color: #d5d5d5;
            }
            """
        )

        ok_button = QtWidgets.QPushButton(self.tr("OK"))
        ok_button.setFixedSize(100, 32)
        ok_button.clicked.connect(self.save_settings)
        ok_button.setStyleSheet(
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

        cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        cancel_button.setFixedSize(100, 32)
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f5f5f7;
                color: #1d1d1f;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #e5e5e5;
            }
            QPushButton:pressed {
                background-color: #d5d5d5;
            }
            """
        )

        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Center the dialog
        self.move_to_center()

        # Set compact row heights
        self.table.verticalHeader().setDefaultSectionSize(28)

    def on_mode_changed(self, digit, index):
        """Enable/disable label field based on mode selection"""
        combo = self.table.cellWidget(digit, 1)
        label_edit = self.table.cellWidget(digit, 2)

        # Enable label field only if a valid mode is selected
        mode = combo.itemData(index)
        label_edit.setEnabled(mode is not None)

        if mode is None:
            label_edit.clear()
            label_edit.setStyleSheet("")
        else:
            # Reset style but add a placeholder to indicate requirement
            label_edit.setStyleSheet("")
            label_edit.setPlaceholderText(self.tr("Required"))

    def reset_settings(self):
        """Reset all settings to None"""
        # Show confirmation dialog
        confirm = QtWidgets.QMessageBox.warning(
            self,
            self.tr("Confirm Reset"),
            self.tr(
                "Are you sure you want to reset all shortcuts? This cannot be undone."
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )

        # Only proceed if user confirmed
        if confirm == QtWidgets.QMessageBox.Yes:
            for digit in range(10):
                mode_combo = self.table.cellWidget(digit, 1)
                mode_combo.setCurrentIndex(0)  # "None" option

                label_edit = self.table.cellWidget(digit, 2)
                label_edit.clear()
                label_edit.setEnabled(False)

    def save_settings(self):
        """Save settings to parent and close dialog"""
        result = {}
        has_error = False

        # Reset all error styling
        for digit in range(10):
            label_edit = self.table.cellWidget(digit, 2)
            label_edit.setStyleSheet("")

        for digit in range(10):
            mode_combo = self.table.cellWidget(digit, 1)
            label_edit = self.table.cellWidget(digit, 2)

            mode = mode_combo.currentData()
            label = label_edit.text().strip()

            # Validate: if mode is set, label must not be empty
            if mode is not None and not label:
                has_error = True
                # Highlight the empty label field
                label_edit.setStyleSheet(
                    "border: 2px solid #FF3B30; background-color: #FFEEEE;"
                )

            if mode is not None and label:
                result[int(digit)] = {"mode": mode, "label": label}

        if has_error:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Validation Error"),
                self.tr(
                    "Please provide a label for each enabled drawing mode."
                ),
                QtWidgets.QMessageBox.Ok,
            )
            return

        # Update parent's digit shortcuts
        if hasattr(self.parent, "drawing_digit_shortcuts"):
            self.parent.drawing_digit_shortcuts = result

        self.accept()

        popup = Popup(
            self.tr("Digit shortcuts saved successfully"),
            self.parent,
            msec=1000,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self.parent)

    def move_to_center(self):
        """Move dialog to center of the screen"""
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2,
        )


class GroupIDModifyDialog(QtWidgets.QDialog):
    """A dialog for modifying group IDs across multiple files."""

    def __init__(self, parent=None):
        """Initialize the dialog.

        Args:
            parent: The parent widget.
        """
        super(GroupIDModifyDialog, self).__init__(parent)

        self.parent = parent
        self.image_file_list = self.get_image_file_list()
        self.shape_list = self.get_shape_file_list()
        self.gid_info = self.get_gid_info()
        self.start_index = 1
        self.end_index = len(self.image_file_list)

        self.init_ui()

    def get_image_file_list(self):
        image_file_list = []
        count = self.parent.file_list_widget.count()
        for c in range(count):
            image_file = self.parent.file_list_widget.item(c).text()
            image_file_list.append(image_file)
        return image_file_list

    def get_shape_file_list(self):
        shape_file_list = []
        for image_file in self.image_file_list:
            label_dir, filename = os.path.split(image_file)
            if self.parent.output_dir:
                label_dir = self.parent.output_dir
            label_file = os.path.join(
                label_dir, os.path.splitext(filename)[0] + ".json"
            )
            if os.path.exists(label_file):
                shape_file_list.append(label_file)
        return shape_file_list

    def get_gid_info(self):
        """Get the group IDs from the shape files.

        Returns:
            list: A list of group IDs.
        """

        gid_info = set()

        for shape_file in self.shape_list:
            with open(shape_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            shapes = data.get("shapes", [])
            for shape in shapes:
                group_id = shape.get("group_id", None)
                if group_id is not None:
                    gid_info.add(group_id)

        return sorted(list(gid_info))

    def init_ui(self):
        """Initialize the UI."""

        self.setWindowTitle(self.tr("Group ID Change Manager"))
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )

        self.resize(960, 480)
        self.move_to_center()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        title_list = ["Ori Group-ID", "New Group-ID", "Delete Group by ID"]
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(len(title_list))
        self.table_widget.setHorizontalHeaderLabels(title_list)

        # Set table to be adaptive
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Fixed
        )
        for idx in range(len(title_list[:])):
            self.table_widget.setColumnWidth(idx, 260)

        # Table style
        self.table_widget.setStyleSheet(
            """
            QTableWidget {
                border: none;
                border-radius: 8px;
                background-color: #FAFAFA;
                outline: none;
                selection-background-color: transparent;
                show-decoration-selected: 0;
                gridline-color: #EBEBEB;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #EBEBEB;
            }
            QTableWidget::item:selected {
                background-color: transparent;
                color: #000000;
                border: none;
            }
            QTableWidget::item:focus {
                border: none;
                outline: none;
                background-color: transparent;
            }
            QTableWidget::focus {
                outline: none;
            }
            QHeaderView::section {
                background-color: #F5F5F7;
                padding: 12px 8px;
                border: none;
                font-weight: 600;
                color: #1d1d1f;
            }
            QTableWidget QLineEdit {
                padding: 2px 8px;
                margin: 2px 4px;
                border: 1px solid #D8D8D8;
                border-radius: 6px;
                background: white;
                selection-background-color: #0071e3;
                min-width: 200px;
            }
            QTableWidget QLineEdit:focus {
                border: 3px solid #60A5FA;
                outline: none;
            }
            QTableView QTableCornerButton::section {
                background-color: #FAFAFA;
                border: none;
            }
            QHeaderView {
                background-color: #FAFAFA;
            }
            QHeaderView::section:vertical {
                background-color: #FAFAFA;
                color: #666666;
                font-weight: 500;
                padding: 8px;
                border: none;
                border-right: 1px solid #EBEBEB;
            }
            """
        )

        self.table_widget.verticalHeader().setStyleSheet(
            "color: #666666; font-size: 13px;"
        )
        self.table_widget.verticalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.table_widget.verticalHeader().setFixedWidth(50)

        layout.addWidget(self.table_widget)

        range_layout = QtWidgets.QHBoxLayout()
        range_layout.addStretch(1)

        from_label = QtWidgets.QLabel("From:")
        self.from_input = QtWidgets.QSpinBox()
        self.from_input.setMinimum(1)
        self.from_input.setMaximum(len(self.image_file_list))
        self.from_input.setSingleStep(1)
        self.from_input.setValue(self.start_index)
        self.from_input.setStyleSheet(get_spinbox_style())
        range_layout.addWidget(from_label)
        range_layout.addWidget(self.from_input)

        to_label = QtWidgets.QLabel("To:")
        self.to_input = QtWidgets.QSpinBox()
        self.to_input.setMinimum(1)
        self.to_input.setMaximum(len(self.image_file_list))
        self.to_input.setSingleStep(1)
        self.to_input.setValue(len(self.image_file_list))
        self.to_input.setStyleSheet(get_spinbox_style())
        range_layout.addWidget(to_label)
        range_layout.addWidget(self.to_input)

        self.range_button = QtWidgets.QPushButton("Go")
        self.range_button.setStyleSheet(get_ok_btn_style())
        range_layout.addWidget(self.range_button)
        self.range_button.clicked.connect(self.update_range)

        range_layout.addStretch(1)

        layout.addLayout(range_layout)

        self.populate_table()

    def move_to_center(self):
        """Move the dialog to the center of the screen."""
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)

        self.move(qr.topLeft())

    def populate_table(self):
        """Populate the table with the group IDs."""
        for i, group_id in enumerate(self.gid_info):
            self.table_widget.insertRow(i)

            # Ori Group-ID
            old_gid_item = QTableWidgetItem(str(group_id))
            old_gid_item.setTextAlignment(Qt.AlignCenter)
            old_gid_item.setFlags(
                old_gid_item.flags() ^ QtCore.Qt.ItemIsEditable
            )

            # New Group-ID
            line_edit = QtWidgets.QLineEdit(self.table_widget)
            line_edit.setValidator(QIntValidator(0, 9999, self))
            line_edit.setPlaceholderText("Enter new ID")
            line_edit.setAlignment(Qt.AlignCenter)
            line_edit.setFixedHeight(28)

            # Create a widget to hold the line edit and center it vertically
            container = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(container)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setAlignment(Qt.AlignCenter)
            layout.addWidget(line_edit)

            # Delete Group by ID
            delete_gid_checkbox = QCheckBox()
            delete_gid_checkbox.stateChanged.connect(
                lambda state, row=i: self.on_delete_checkbox_changed(
                    row, state
                )
            )

            delete_container = QtWidgets.QWidget()
            delete_layout = QtWidgets.QHBoxLayout(delete_container)
            delete_layout.setContentsMargins(4, 4, 4, 4)
            delete_layout.setAlignment(Qt.AlignCenter)
            delete_layout.addWidget(delete_gid_checkbox)

            self.table_widget.setItem(i, 0, old_gid_item)
            self.table_widget.setCellWidget(i, 1, container)
            self.table_widget.setCellWidget(i, 2, delete_container)

            # Set row height
            self.table_widget.setRowHeight(i, 50)

    def on_delete_checkbox_changed(self, row, state):
        """Deactivate linedit when checkbox is checked"""
        container = self.table_widget.cellWidget(row, 1)
        value_item = container.layout().itemAt(0).widget()

        delete_container = self.table_widget.cellWidget(row, 2)
        delete_checkbox = delete_container.layout().itemAt(0).widget()

        if state == QtCore.Qt.Checked:
            value_item.clear()
            value_item.setReadOnly(True)
            value_item.setStyleSheet("background-color: lightgray;")
            delete_checkbox.setCheckable(True)
        else:
            value_item.setReadOnly(False)
            value_item.setStyleSheet("background-color: white;")
            delete_checkbox.setCheckable(False)

        if value_item.text():
            delete_checkbox.setCheckable(False)
        else:
            delete_checkbox.setCheckable(True)

    def update_range(self):
        from_value = (
            int(self.from_input.text())
            if self.from_input.text()
            else self.start_index
        )
        to_value = (
            int(self.to_input.text())
            if self.to_input.text()
            else self.end_index
        )
        if (
            (from_value > to_value)
            or (from_value < 1)
            or (to_value > len(self.image_file_list))
        ):
            self.from_input.setValue(1)
            self.to_input.setValue(len(self.image_file_list))
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Invalid Range"),
                self.tr("Please enter a valid range."),
            )
        else:
            self.start_index = from_value
            self.end_index = to_value
            self.confirm_changes(self.start_index, self.end_index)

    def confirm_changes(self, start_index: int = -1, end_index: int = -1):
        """Confirm the changes."""
        total_num = self.table_widget.rowCount()
        if total_num == 0:
            return

        # Temporary dictionary to handle changes
        new_gid_info = []
        updated_gid_info = {}
        deleted_gid_info = []

        # Iterate over each row to get the old and new group IDs
        for i in range(total_num):
            old_gid_item = self.table_widget.item(i, 0)
            container = self.table_widget.cellWidget(i, 1)
            line_edit = container.layout().itemAt(0).widget()
            new_gid = line_edit.text()
            old_gid = old_gid_item.text()

            del_container = self.table_widget.cellWidget(i, 2)
            del_checkbox = del_container.layout().itemAt(0).widget()

            if del_checkbox.isChecked():
                deleted_gid_info.append(int(old_gid))
                continue

            # Only add to updated_gid_info
            # if the new group ID is not empty and different
            if new_gid and old_gid != new_gid:
                new_gid_info.append(new_gid)
                updated_gid_info[int(old_gid)] = {"new_gid": int(new_gid)}
            else:
                new_gid_info.append(old_gid)

        # Try to modify group IDs
        if self.modify_group_id(
            updated_gid_info, deleted_gid_info, start_index, end_index
        ):

            # Update original gid info
            self.gid_info = new_gid_info

            popup = Popup(
                self.tr("Group IDs modified successfully!"),
                self.parent,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self.parent)
            self.accept()
        else:
            popup = Popup(
                self.tr("An error occurred while updating the Group IDs."),
                self.parent,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self.parent)

    def modify_group_id(
        self,
        updated_gid_info,
        deleted_gid_info: list,
        start_index: int = -1,
        end_index: int = -1,
    ):
        """Modify the group IDs."""
        try:
            if start_index == -1:
                start_index = self.start_index
            if end_index == -1:
                end_index = self.end_index
            for i, image_file in enumerate(self.image_file_list):
                if i < start_index - 1 or i > end_index - 1:
                    continue
                label_dir, filename = os.path.split(image_file)
                if self.parent.output_dir:
                    label_dir = self.parent.output_dir
                label_file = os.path.join(
                    label_dir, os.path.splitext(filename)[0] + ".json"
                )
                if not os.path.exists(label_file):
                    continue
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                src_shapes, dst_shapes = data["shapes"], []
                for shape in reversed(src_shapes):
                    group_id = shape.get("group_id")
                    if group_id is not None:
                        group_id = int(group_id)
                        if group_id in updated_gid_info:
                            shape["group_id"] = updated_gid_info[group_id][
                                "new_gid"
                            ]
                        if group_id in deleted_gid_info:
                            src_shapes.remove(shape)
                            continue
                    dst_shapes.append(shape)
                data["shapes"] = dst_shapes

                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Error occurred while updating Group IDs: {e}")
            return False


class LabelColorButton(QtWidgets.QWidget):
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = color
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        self.color_label = QtWidgets.QLabel()
        self.color_label.setFixedSize(15, 15)
        self.color_label.setStyleSheet(
            f"background-color: {self.color.name()}; border: 1px solid transparent; border-radius: 10px;"
        )

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.color_label)

    def set_color(self, color):
        self.color = color
        self.color_label.setStyleSheet(
            f"background-color: {self.color.name()}; border: 1px solid transparent; border-radius: 10px;"
        )

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.parent.change_color(self)


class LabelModifyDialog(QtWidgets.QDialog):
    """A dialog for modifying labels across multiple files.

    This dialog allows users to:
    - Change label names
    - Delete labels
    - Modify label colors
    - Select file range for applying changes
    """

    def __init__(self, parent=None, opacity=128):
        """Initialize the dialog.

        Args:
            parent: Parent widget. Defaults to None.
            opacity: Opacity value for colors. Defaults to 128.
        """
        super(LabelModifyDialog, self).__init__(parent)
        self.parent = parent
        self.opacity = opacity
        self.image_file_list = self.get_image_file_list()
        self.start_index = 1
        self.end_index = len(self.image_file_list)
        self.init_label_info()
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(self.tr("Label Change Manager"))
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        self.resize(700, 400)
        self.move_to_center()

        title_list = ["Category", "Delete", "New Value", "Visible", "Color"]
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(len(title_list))
        self.table_widget.setHorizontalHeaderLabels(title_list)
        self.table_widget.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
        )

        # Set header font and alignment
        for i in range(len(title_list)):
            self.table_widget.horizontalHeaderItem(i).setFont(
                QFont("Arial", 8, QFont.Bold)
            )
            self.table_widget.horizontalHeaderItem(i).setTextAlignment(
                QtCore.Qt.AlignCenter
            )
            if i == 0:
                self.table_widget.horizontalHeaderItem(i).setToolTip(
                    self.tr("Double-click to copy label text")
                )

        self.table_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_widget.customContextMenuRequested.connect(
            self.show_context_menu
        )
        self.table_widget.itemDoubleClicked.connect(
            self.on_item_double_clicked
        )

        # Add input fields for range selection
        range_layout = QtWidgets.QHBoxLayout()
        # Add stretch to center the widgets
        range_layout.addStretch(1)

        from_label = QtWidgets.QLabel("From:")
        self.from_input = QtWidgets.QSpinBox()
        self.from_input.setMinimum(1)
        self.from_input.setMaximum(len(self.image_file_list))
        self.from_input.setSingleStep(1)
        self.from_input.setValue(self.start_index)
        self.from_input.setStyleSheet(get_spinbox_style())
        range_layout.addWidget(from_label)
        range_layout.addWidget(self.from_input)

        to_label = QtWidgets.QLabel("To:")
        self.to_input = QtWidgets.QSpinBox()
        self.to_input.setMinimum(1)
        self.to_input.setMaximum(len(self.image_file_list))
        self.to_input.setSingleStep(1)
        self.to_input.setValue(len(self.image_file_list))
        self.to_input.setStyleSheet(get_spinbox_style())
        range_layout.addWidget(to_label)
        range_layout.addWidget(self.to_input)

        self.range_button = QtWidgets.QPushButton("Go")
        self.range_button.setStyleSheet(get_ok_btn_style())
        range_layout.addWidget(self.range_button)
        self.range_button.clicked.connect(self.update_range)

        # Add stretch to center the widgets
        range_layout.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table_widget)
        layout.addLayout(range_layout)

        self.populate_table()

    def get_image_file_list(self):
        image_file_list = []
        count = self.parent.file_list_widget.count()
        for c in range(count):
            image_file = self.parent.file_list_widget.item(c).text()
            image_file_list.append(image_file)
        return image_file_list

    def move_to_center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def populate_table(self):
        sorted_labels = sorted(
            self.parent.label_info.items(),
            key=lambda x: natural_sort_key(x[0]),
        )
        for i, (label, info) in enumerate(sorted_labels):
            self.table_widget.insertRow(i)

            class_item = QTableWidgetItem(label)
            class_item.setFlags(class_item.flags() ^ QtCore.Qt.ItemIsEditable)
            class_item.setToolTip(self.tr("Double-click to copy label text"))

            delete_checkbox = QCheckBox()
            delete_checkbox.setChecked(info["delete"])
            delete_checkbox.setIcon(QtGui.QIcon(":/images/images/delete.png"))
            delete_checkbox.stateChanged.connect(
                lambda state, row=i: self.on_delete_checkbox_changed(
                    row, state
                )
            )

            value_item = QTableWidgetItem(
                info["value"] if info["value"] else ""
            )
            value_item.setFlags(
                value_item.flags() & ~QtCore.Qt.ItemIsEditable
                if info["delete"]
                else value_item.flags() | QtCore.Qt.ItemIsEditable
            )
            value_item.setBackground(
                QtGui.QColor("lightgray")
                if info["delete"]
                else QtGui.QColor("white")
            )

            visible_checkbox = QCheckBox()
            visible_checkbox.setChecked(info.get("visible", True))
            visible_checkbox.stateChanged.connect(
                lambda state, row=i: self.on_visible_checkbox_changed(
                    row, state
                )
            )

            visible_container = QtWidgets.QWidget()
            visible_layout = QtWidgets.QHBoxLayout(visible_container)
            visible_layout.setContentsMargins(0, 0, 0, 0)
            visible_layout.setAlignment(Qt.AlignCenter)
            visible_layout.addWidget(visible_checkbox)

            color = QColor(*info["color"])
            color.setAlpha(info["opacity"])
            color_button = LabelColorButton(color, self)
            color_button.setParent(self.table_widget)
            self.table_widget.setItem(i, 0, class_item)
            self.table_widget.setCellWidget(i, 1, delete_checkbox)
            self.table_widget.setItem(i, 2, value_item)
            self.table_widget.setCellWidget(i, 3, visible_container)
            self.table_widget.setCellWidget(i, 4, color_button)

    def change_color(self, button):
        row = self.table_widget.indexAt(button.pos()).row()
        current_color = self.parent.label_info[
            self.table_widget.item(row, 0).text()
        ]["color"]
        color = QColorDialog.getColor(QColor(*current_color), self)
        if color.isValid():
            self.parent.label_info[self.table_widget.item(row, 0).text()][
                "color"
            ] = [color.red(), color.green(), color.blue()]
            self.parent.label_info[self.table_widget.item(row, 0).text()][
                "opacity"
            ] = color.alpha()
            button.set_color(color)

    def on_delete_checkbox_changed(self, row, state):
        value_item = self.table_widget.item(row, 2)
        delete_checkbox = self.table_widget.cellWidget(row, 1)

        if state == QtCore.Qt.Checked:
            value_item.setFlags(value_item.flags() & ~QtCore.Qt.ItemIsEditable)
            value_item.setBackground(QtGui.QColor("lightgray"))
            delete_checkbox.setCheckable(True)
        else:
            value_item.setFlags(value_item.flags() | QtCore.Qt.ItemIsEditable)
            value_item.setBackground(QtGui.QColor("white"))
            delete_checkbox.setCheckable(False)

        if value_item.text():
            delete_checkbox.setCheckable(False)
        else:
            delete_checkbox.setCheckable(True)

    def on_visible_checkbox_changed(self, row, state):
        pass

    def on_item_double_clicked(self, item: QTableWidgetItem) -> None:
        """Copy label text to clipboard on double-click.

        Args:
            item: The table widget item that was double-clicked.
        """
        column = item.column()
        if column == 0:
            label_text = item.text()
            clipboard = QApplication.clipboard()
            clipboard.setText(label_text)
            popup = Popup(
                self.tr("Label copied to clipboard"),
                self.parent,
                msec=1000,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self.parent)

    def show_context_menu(self, pos):
        column = self.table_widget.columnAt(pos.x())
        if column != 3:
            return

        menu = QtWidgets.QMenu(self)
        select_all_action = menu.addAction(self.tr("Select All"))
        deselect_all_action = menu.addAction(self.tr("Deselect All"))

        action = menu.exec_(self.table_widget.mapToGlobal(pos))
        if action == select_all_action:
            self.select_all_visible()
        elif action == deselect_all_action:
            self.deselect_all_visible()

    def select_all_visible(self):
        for i in range(self.table_widget.rowCount()):
            visible_container = self.table_widget.cellWidget(i, 3)
            if visible_container:
                visible_checkbox = (
                    visible_container.layout().itemAt(0).widget()
                )
                if visible_checkbox:
                    visible_checkbox.setChecked(True)

    def deselect_all_visible(self):
        for i in range(self.table_widget.rowCount()):
            visible_container = self.table_widget.cellWidget(i, 3)
            if visible_container:
                visible_checkbox = (
                    visible_container.layout().itemAt(0).widget()
                )
                if visible_checkbox:
                    visible_checkbox.setChecked(False)

    def confirm_changes(self, start_index: int = -1, end_index: int = -1):
        total_num = self.table_widget.rowCount()
        if total_num == 0:
            self.reject()
            return

        # Temporary dictionary to handle changes
        updated_label_info = {}

        for i in range(total_num):
            label = self.table_widget.item(i, 0).text()
            delete_checkbox = self.table_widget.cellWidget(i, 1)
            value_item = self.table_widget.item(i, 2)

            is_delete = delete_checkbox.isChecked()
            new_value = value_item.text()

            visible_container = self.table_widget.cellWidget(i, 3)
            visible_checkbox = visible_container.layout().itemAt(0).widget()
            is_visible = visible_checkbox.isChecked()

            # Update the label info in the temporary dictionary
            self.parent.label_info[label]["delete"] = is_delete
            self.parent.label_info[label]["value"] = new_value
            self.parent.label_info[label]["visible"] = is_visible

            # Update the color
            color = self.parent.label_info[label]["color"]
            self.parent.unique_label_list.update_item_color(
                label, color, self.opacity
            )

            # Handle delete and change of labels
            if is_delete:
                self.parent.unique_label_list.remove_items_by_label(label)
                self.parent.label_dialog.remove_label_history(label)
                continue
            elif new_value:
                self.parent.unique_label_list.remove_items_by_label(label)
                self.parent.label_dialog.remove_label_history(label)
                self.parent.label_dialog.add_label_history(new_value)
                updated_label_info[new_value] = self.parent.label_info[label]
            else:
                updated_label_info[label] = self.parent.label_info[label]

        if self.modify_label(start_index, end_index):
            self.parent.label_info = updated_label_info
            if hasattr(self.parent, "apply_label_visibility"):
                self.parent.apply_label_visibility()
            popup = Popup(
                self.tr("Labels modified successfully!"),
                self.parent,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self.parent)
            self.accept()
        else:
            popup = Popup(
                self.tr("An error occurred while updating the labels."),
                self.parent,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self.parent)

    def modify_label(self, start_index: int = -1, end_index: int = -1):
        try:
            if start_index == -1:
                start_index = self.start_index
            if end_index == -1:
                end_index = self.end_index
            for i, image_file in enumerate(self.image_file_list):
                if i < start_index - 1 or i > end_index - 1:
                    continue
                label_dir, filename = os.path.split(image_file)
                if self.parent.output_dir:
                    label_dir = self.parent.output_dir
                label_file = os.path.join(
                    label_dir, os.path.splitext(filename)[0] + ".json"
                )
                if not os.path.exists(label_file):
                    continue
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                src_shapes, dst_shapes = data["shapes"], []
                for shape in src_shapes:
                    label = shape["label"]
                    if self.parent.label_info[label]["delete"]:
                        continue
                    if self.parent.label_info[label]["value"]:
                        shape["label"] = self.parent.label_info[label]["value"]
                    dst_shapes.append(shape)
                data["shapes"] = dst_shapes
                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error occurred while updating labels: {e}")
            return False

    def init_label_info(self):
        classes = set()

        for image_file in self.image_file_list:
            label_dir, filename = os.path.split(image_file)
            if self.parent.output_dir:
                label_dir = self.parent.output_dir
            label_file = os.path.join(
                label_dir, os.path.splitext(filename)[0] + ".json"
            )
            if not os.path.exists(label_file):
                continue
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            for shape in shapes:
                label = shape["label"]
                classes.add(label)

        for i in range(self.parent.unique_label_list.count()):
            item = self.parent.unique_label_list.item(i)
            if item:
                label_text = item.text()
                if label_text:
                    classes.add(label_text)

        for c in sorted(classes):
            # Update unique label list
            if not self.parent.unique_label_list.find_items_by_label(c):
                unique_label_item = (
                    self.parent.unique_label_list.create_item_from_label(c)
                )
                self.parent.unique_label_list.addItem(unique_label_item)
                rgb = self.parent._get_rgb_by_label(c)
                self.parent.unique_label_list.set_item_label(
                    unique_label_item, c, rgb, self.opacity
                )
            else:
                rgb = self.parent._get_rgb_by_label(c)
            # Update label info
            # Preserve existing color if label already exists in label_info
            if c in self.parent.label_info:
                color = self.parent.label_info[c].get("color", list(rgb))
                opacity = self.parent.label_info[c].get(
                    "opacity", self.opacity
                )
                visible = self.parent.label_info[c].get("visible", True)
            else:
                color = list(rgb)
                opacity = self.opacity
                visible = True
            self.parent.label_info[c] = dict(
                delete=False,
                value=None,
                color=color,
                opacity=opacity,
                visible=visible,
            )

    def update_range(self):
        from_value = (
            int(self.from_input.text())
            if self.from_input.text()
            else self.start_index
        )
        to_value = (
            int(self.to_input.text())
            if self.to_input.text()
            else self.end_index
        )
        if (
            (from_value > to_value)
            or (from_value < 1)
            or (to_value > len(self.image_file_list))
        ):
            self.from_input.setValue(1)
            self.to_input.setValue(len(self.image_file_list))
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Invalid Range"),
                self.tr("Please enter a valid range."),
            )
        else:
            self.start_index = from_value
            self.end_index = to_value
            self.confirm_changes(self.start_index, self.end_index)


class LabelQLineEdit(QtWidgets.QLineEdit):
    def __init__(self) -> None:
        super().__init__()
        self.list_widget = None

    def set_list_widget(self, list_widget):
        self.list_widget = list_widget

    # QT Overload
    def keyPressEvent(self, e):
        if e.key() in [QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            self.list_widget.keyPressEvent(e)
        else:
            super(LabelQLineEdit, self).keyPressEvent(e)


class LabelDialog(QtWidgets.QDialog):
    def __init__(
        self,
        text=None,
        parent=None,
        labels=None,
        sort_labels=True,
        show_text_field=True,
        completion="startswith",
        fit_to_content=None,
        flags=None,
        difficult=False,
    ):
        if text is None:
            text = QCoreApplication.translate(
                "LabelDialog", "Enter object label"
            )

        if fit_to_content is None:
            fit_to_content = {"row": False, "column": True}
        self._fit_to_content = fit_to_content

        super(LabelDialog, self).__init__(parent)
        self.edit = LabelQLineEdit()
        self.edit.setPlaceholderText(text)
        self.edit.setValidator(utils.label_validator())
        self.edit.editingFinished.connect(self.postprocess)
        if flags:
            self.edit.textChanged.connect(self.update_flags)
        self.edit_group_id = QtWidgets.QLineEdit()
        self.edit_group_id.setPlaceholderText(self.tr("Group ID"))
        self.edit_group_id.setValidator(
            QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression(r"\d*"), None
            )
        )
        self.edit_group_id.setAlignment(QtCore.Qt.AlignCenter)

        # Add difficult checkbox
        self.edit_difficult = QtWidgets.QCheckBox(self.tr("useDifficult"))
        self.edit_difficult.setChecked(difficult)

        # Add linking input
        self.linking_input = QtWidgets.QLineEdit()
        self.linking_input.setPlaceholderText(
            self.tr("Enter linking, e.g., [0,1]")
        )
        linking_font = (
            self.linking_input.font()
        )  # Adjust placeholder font size
        linking_font.setPointSize(8)
        self.linking_input.setFont(linking_font)
        self.linking_list = QtWidgets.QListWidget()
        self.linking_list.setHidden(True)  # Initially hide the list
        row_height = self.linking_list.fontMetrics().height()
        self.linking_list.setFixedHeight(
            row_height * 4 + 2 * self.linking_list.frameWidth()
        )
        self.add_linking_button = QtWidgets.QPushButton(self.tr("Add"))
        self.add_linking_button.clicked.connect(self.add_linking_pair)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        if show_text_field:
            layout_edit = QtWidgets.QHBoxLayout()
            layout_edit.addWidget(self.edit, 4)
            layout_edit.addWidget(self.edit_group_id, 2)
            layout.addLayout(layout_edit)

        # Add linking layout
        layout_linking = QtWidgets.QHBoxLayout()
        layout_linking.addWidget(self.linking_input, 4)
        layout_linking.addWidget(self.add_linking_button, 2)
        layout.addLayout(layout_linking)
        layout.addWidget(self.linking_list)

        # buttons
        self.button_box = bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        bb.button(bb.Ok).setIcon(utils.new_icon("done"))
        bb.button(bb.Cancel).setIcon(utils.new_icon("undo"))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)

        # text edit
        self.edit_description = QtWidgets.QTextEdit()
        self.edit_description.setPlaceholderText(self.tr("Label description"))
        self.edit_description.setFixedHeight(50)
        layout.addWidget(self.edit_description)

        # difficult & confirm button
        layout_button = QtWidgets.QHBoxLayout()
        layout_button.addWidget(self.edit_difficult)
        layout_button.addWidget(self.button_box)
        layout.addLayout(layout_button)

        # label_list
        self.label_list = QtWidgets.QListWidget()
        if self._fit_to_content["row"]:
            self.label_list.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )
        if self._fit_to_content["column"]:
            self.label_list.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )
        self._sort_labels = sort_labels
        if labels:
            self.label_list.addItems(labels)
        if self._sort_labels:
            self.sort_labels()
        else:
            self.label_list.setDragDropMode(
                QtWidgets.QAbstractItemView.InternalMove
            )
        self.label_list.currentItemChanged.connect(self.label_selected)
        self.label_list.itemDoubleClicked.connect(self.label_double_clicked)
        self.edit.set_list_widget(self.label_list)
        layout.addWidget(self.label_list)
        # label_flags
        if flags is None:
            flags = {}
        self._flags = flags
        self.flags_layout = QtWidgets.QVBoxLayout()
        self.reset_flags()
        layout.addItem(self.flags_layout)
        self.edit.textChanged.connect(self.update_flags)
        self.setLayout(layout)
        # completion
        completer = QtWidgets.QCompleter()
        if completion == "startswith":
            completer.setCompletionMode(QtWidgets.QCompleter.InlineCompletion)
            # Default settings.
            # completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        elif completion == "contains":
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            completer.setFilterMode(QtCore.Qt.MatchContains)
        else:
            raise ValueError(f"Unsupported completion: {completion}")
        completer.setModel(self.label_list.model())
        self.edit.setCompleter(completer)
        # Save last label
        self._last_label = ""
        self._last_gid = None

    def add_linking_pair(self):
        linking_text = self.linking_input.text()
        try:
            linking_pairs = eval(linking_text)
            if (
                isinstance(linking_pairs, list)
                and len(linking_pairs) == 2
                and all(isinstance(item, int) for item in linking_pairs)
            ):
                if linking_pairs in self.get_kie_linking():
                    QtWidgets.QMessageBox.warning(
                        self,
                        self.tr("Duplicate Entry"),
                        self.tr("This linking pair already exists."),
                    )
                self.linking_list.addItem(str(linking_pairs))
                self.linking_input.clear()
                self.linking_list.setHidden(
                    False
                )  # Show the list when an item is added
            else:
                raise ValueError
        except Exception as e:
            logger.error(f"An Error occurred while adding linking pair: {e}")
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Invalid Input"),
                self.tr(
                    "Please enter a valid list of linking pairs like [1,2]."
                ),
            )

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete:
            if hasattr(self, "linking_list") and self.linking_list is not None:
                selected_items = self.linking_list.selectedItems()
                if selected_items:
                    for item in selected_items:
                        self.linking_list.takeItem(self.linking_list.row(item))
        else:
            super(LabelDialog, self).keyPressEvent(event)

    def remove_linking_item(self, item_widget):
        list_item = self.linking_list.itemWidget(item_widget)
        self.linking_list.takeItem(self.linking_list.row(list_item))
        item_widget.deleteLater()

    def reset_linking(self, kie_linking=[]):
        self.linking_list.clear()
        for linking_pair in kie_linking:
            self.linking_list.addItem(str(linking_pair))
        self.linking_list.setHidden(False if kie_linking else True)

    def get_last_label(self):
        return self._last_label

    def get_last_gid(self):
        return self._last_gid

    def sort_labels(self):
        items = []
        for index in range(self.label_list.count()):
            items.append(self.label_list.item(index).text())

        items.sort(key=natural_sort_key)
        self.label_list.clear()
        self.label_list.addItems(items)

    def add_label_history(self, label, update_last_label=True):
        if update_last_label:
            self._last_label = label
        if self.label_list.findItems(label, QtCore.Qt.MatchExactly):
            return
        self.label_list.addItem(label)
        if self._sort_labels:
            self.sort_labels()
        items = self.label_list.findItems(label, QtCore.Qt.MatchExactly)
        if items:
            self.label_list.setCurrentItem(items[0])

    def remove_label_history(self, label):
        items = self.label_list.findItems(label, QtCore.Qt.MatchExactly)
        if not items:
            logger.warning(f"Skipping empty items.")
            return

        for item in items:
            self.label_list.takeItem(self.label_list.row(item))

        if self._last_label == label:
            self._last_label = ""

        if self.edit.text() == label:
            self.edit.clear()

    def label_selected(self, item):
        if item is not None:
            self.edit.setText(item.text())
        else:
            # Clear the edit field if no item is selected
            self.edit.clear()

    def validate(self):
        text = self.edit.text()
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = text.trimmed()
        if text:
            self.accept()

    def label_double_clicked(self, _):
        self.validate()

    def postprocess(self):
        text = self.edit.text()
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = text.trimmed()
        self.edit.setText(text)

    def upload_flags(self, flags):
        self._flags = flags

    def update_flags(self, label_new):
        # keep state of shared flags
        flags_old = self.get_flags()

        flags_new = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label_new):
                for key in keys:
                    flags_new[key] = flags_old.get(key, False)
        self.set_flags(flags_new)

    def delete_flags(self):
        for i in reversed(range(self.flags_layout.count())):
            item = self.flags_layout.itemAt(i).widget()
            self.flags_layout.removeWidget(item)
            item.setParent(None)

    def reset_flags(self, label=""):
        flags = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label):
                for key in keys:
                    flags[key] = False
        self.set_flags(flags)

    def set_flags(self, flags):
        self.delete_flags()
        for key in flags:
            item = QtWidgets.QCheckBox(key, self)
            item.setChecked(flags[key])
            self.flags_layout.addWidget(item)
            item.show()

    def get_flags(self):
        flags = {}
        for i in range(self.flags_layout.count()):
            item = self.flags_layout.itemAt(i).widget()
            flags[item.text()] = item.isChecked()
        return flags

    def get_group_id(self):
        group_id = self.edit_group_id.text()
        if group_id:
            return int(group_id)
        return None

    def get_description(self):
        return self.edit_description.toPlainText()

    def get_difficult_state(self):
        return self.edit_difficult.isChecked()

    def get_kie_linking(self):
        kie_linking = []
        for index in range(self.linking_list.count()):
            item = self.linking_list.item(index)
            kie_linking.append(eval(item.text()))
        return kie_linking

    def pop_up(
        self,
        text=None,
        move=True,
        move_mode="auto",
        flags=None,
        group_id=None,
        description=None,
        difficult=False,
        kie_linking=[],
    ):
        if self._fit_to_content["row"]:
            self.label_list.setMinimumHeight(
                self.label_list.sizeHintForRow(0) * self.label_list.count() + 2
            )
        if self._fit_to_content["column"]:
            self.label_list.setMinimumWidth(
                self.label_list.sizeHintForColumn(0) + 2
            )
        # if text is None, the previous label in self.edit is kept
        if text is None:
            text = self.edit.text()
        # description is always initialized by empty text c.f., self.edit.text
        if description is None:
            description = ""
        self.edit_description.setPlainText(description)
        # Set initial values for kie_linking
        self.reset_linking(kie_linking)
        if flags:
            self.set_flags(flags)
        else:
            self.reset_flags(text)
        if difficult:
            self.edit_difficult.setChecked(True)
        else:
            self.edit_difficult.setChecked(False)
        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        if group_id is None:
            self.edit_group_id.clear()
        else:
            self.edit_group_id.setText(str(group_id))
        items = self.label_list.findItems(text, QtCore.Qt.MatchFixedString)

        if items:
            if len(items) != 1:
                logger.warning(f"Label list has duplicate '{text}'")
            self.label_list.setCurrentItem(items[0])
            row = self.label_list.row(items[0])
            self.edit.completer().setCurrentRow(row)
        self.edit.setFocus(QtCore.Qt.PopupFocusReason)

        if move:
            if move_mode == "auto":
                cursor_pos = QtGui.QCursor.pos()
                screen = QtWidgets.QApplication.desktop().screenGeometry(
                    cursor_pos
                )
                dialog_frame_size = self.frameGeometry()
                # Calculate the ideal top-left corner position for the dialog based on the mouse click
                ideal_pos = cursor_pos
                # Adjust to prevent the dialog from exceeding the right screen boundary
                if (
                    ideal_pos.x() + dialog_frame_size.width()
                ) > screen.right():
                    ideal_pos.setX(screen.right() - dialog_frame_size.width())
                # Adjust to prevent the dialog's bottom from going off-screen
                if (
                    ideal_pos.y() + dialog_frame_size.height()
                ) > screen.bottom():
                    ideal_pos.setY(
                        screen.bottom() - dialog_frame_size.height()
                    )
                self.move(ideal_pos)
            elif move_mode == "center":
                # Calculate the center position to move the dialog to
                screen = QtWidgets.QApplication.desktop().screenNumber(
                    QtWidgets.QApplication.desktop().cursor().pos()
                )
                centerPoint = (
                    QtWidgets.QApplication.desktop()
                    .screenGeometry(screen)
                    .center()
                )
                qr = self.frameGeometry()
                qr.moveCenter(centerPoint)
                self.move(qr.topLeft())

        if self.exec_():
            return (
                self.edit.text(),
                self.get_flags(),
                self.get_group_id(),
                self.get_description(),
                self.get_difficult_state(),
                self.get_kie_linking(),
            )
        return None, None, None, None, False, []
