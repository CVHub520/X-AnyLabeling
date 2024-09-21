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
)

from .. import utils
from ..logger import logger


# TODO(unknown):
# - Calculate optimal position so as not to go out of screen area.


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


class GroupIDModifyDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(GroupIDModifyDialog, self).__init__(parent)
        self.parent = parent
        self.gid_info = []
        self.shape_list = parent.get_label_file_list()
        self.init_gid_info()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Group ID Change Manager"))
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        self.resize(600, 400)
        self.move_to_center()

        title_list = ["Ori Group-ID", "New Group-ID"]
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(len(title_list))
        self.table_widget.setHorizontalHeaderLabels(title_list)

        # Set header font and alignment
        for i in range(len(title_list)):
            self.table_widget.horizontalHeaderItem(i).setFont(
                QFont("Arial", 8, QFont.Bold)
            )
            self.table_widget.horizontalHeaderItem(i).setTextAlignment(
                QtCore.Qt.AlignCenter
            )

        self.buttons_layout = QtWidgets.QHBoxLayout()

        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"), self)
        self.cancel_button.clicked.connect(self.reject)

        self.confirm_button = QtWidgets.QPushButton(self.tr("Confirm"), self)
        self.confirm_button.clicked.connect(self.confirm_changes)

        self.buttons_layout.addWidget(self.cancel_button)
        self.buttons_layout.addWidget(self.confirm_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table_widget)
        layout.addLayout(self.buttons_layout)

        self.populate_table()

    def move_to_center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def populate_table(self):
        for i, group_id in enumerate(self.gid_info):
            self.table_widget.insertRow(i)

            old_gid_item = QTableWidgetItem(str(group_id))
            old_gid_item.setFlags(
                old_gid_item.flags() ^ QtCore.Qt.ItemIsEditable
            )

            new_gid_item = QTableWidgetItem("")
            new_gid_item.setFlags(
                new_gid_item.flags() | QtCore.Qt.ItemIsEditable
            )

            # Set QIntValidator to ensure only non-negative integers can be entered
            validator = QIntValidator(0, 9999, self)
            line_edit = QtWidgets.QLineEdit(self.table_widget)
            line_edit.setValidator(validator)
            self.table_widget.setCellWidget(i, 1, line_edit)

            self.table_widget.setItem(i, 0, old_gid_item)

    def confirm_changes(self):
        total_num = self.table_widget.rowCount()
        if total_num == 0:
            self.reject()
            return

        # Temporary dictionary to handle changes
        new_gid_info = []
        updated_gid_info = {}

        # Iterate over each row to get the old and new group IDs
        for i in range(total_num):
            old_gid_item = self.table_widget.item(i, 0)
            line_edit = self.table_widget.cellWidget(i, 1)
            new_gid = line_edit.text()
            old_gid = old_gid_item.text()

            # Only add to updated_gid_info
            # if the new group ID is not empty and different
            if new_gid and old_gid != new_gid:
                new_gid_info.append(new_gid)
                updated_gid_info[int(old_gid)] = {"new_gid": int(new_gid)}
            else:
                new_gid_info.append(old_gid)
        # Update original gid info
        self.gid_info = new_gid_info

        # Try to modify group IDs
        if self.modify_group_id(updated_gid_info):
            QtWidgets.QMessageBox.information(
                self,
                "Success",
                "Group IDs modified successfully!",
            )
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Warning",
                "An error occurred while updating the Group IDs.",
            )

    def modify_group_id(self, updated_gid_info):
        try:
            for shape_file in self.shape_list:
                with open(shape_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                src_shapes, dst_shapes = data["shapes"], []
                for shape in src_shapes:
                    group_id = shape.get("group_id")
                    if group_id is not None:
                        group_id = int(group_id)
                        if group_id in updated_gid_info:
                            shape["group_id"] = updated_gid_info[group_id]["new_gid"]
                    dst_shapes.append(shape)
                data["shapes"] = dst_shapes
                with open(shape_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error occurred while updating Group IDs: {e}")
            return False

    def init_gid_info(self):
        for shape_file in self.shape_list:
            with open(shape_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            for shape in shapes:
                group_id = shape.get("group_id", None)
                if group_id is not None and group_id not in self.gid_info:
                    self.gid_info.append(group_id)
        self.gid_info.sort()


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
    def __init__(self, parent=None, opacity=128):
        super(LabelModifyDialog, self).__init__(parent)
        self.parent = parent
        self.opacity = opacity
        self.label_file_list = parent.get_label_file_list()
        self.init_label_info()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Label Change Manager"))
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        self.resize(600, 400)
        self.move_to_center()

        title_list = ["Category", "Delete", "New Value", "Color"]
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(len(title_list))
        self.table_widget.setHorizontalHeaderLabels(title_list)

        # Set header font and alignment
        for i in range(len(title_list)):
            self.table_widget.horizontalHeaderItem(i).setFont(
                QFont("Arial", 8, QFont.Bold)
            )
            self.table_widget.horizontalHeaderItem(i).setTextAlignment(
                QtCore.Qt.AlignCenter
            )

        self.buttons_layout = QtWidgets.QHBoxLayout()

        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"), self)
        self.cancel_button.clicked.connect(self.reject)

        self.confirm_button = QtWidgets.QPushButton(self.tr("Confirm"), self)
        self.confirm_button.clicked.connect(self.confirm_changes)

        self.buttons_layout.addWidget(self.cancel_button)
        self.buttons_layout.addWidget(self.confirm_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table_widget)
        layout.addLayout(self.buttons_layout)

        self.populate_table()

    def move_to_center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def populate_table(self):
        for i, (label, info) in enumerate(self.parent.label_info.items()):
            self.table_widget.insertRow(i)

            class_item = QTableWidgetItem(label)
            class_item.setFlags(class_item.flags() ^ QtCore.Qt.ItemIsEditable)

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

            color = QColor(*info["color"])
            color.setAlpha(info["opacity"])
            color_button = LabelColorButton(color, self)
            color_button.setParent(self.table_widget)
            self.table_widget.setItem(i, 0, class_item)
            self.table_widget.setCellWidget(i, 1, delete_checkbox)
            self.table_widget.setItem(i, 2, value_item)
            self.table_widget.setCellWidget(i, 3, color_button)

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
        hidden_checkbox = self.table_widget.cellWidget(row, 3)

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

    def confirm_changes(self):
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

            # Update the label info in the temporary dictionary
            self.parent.label_info[label]["delete"] = is_delete
            self.parent.label_info[label]["value"] = new_value

            # Update the color
            color = self.parent.label_info[label]["color"]
            self.parent.unique_label_list.update_item_color(
                label, color, self.opacity
            )

            # Handle delete and change of labels
            if is_delete:
                self.parent.unique_label_list.remove_items_by_label(label)
                continue  # Skip adding this to updated_label_info to effectively delete it
            elif new_value:
                self.parent.unique_label_list.remove_items_by_label(label)
                updated_label_info[new_value] = self.parent.label_info[label]
            else:
                updated_label_info[label] = self.parent.label_info[label]

        # Try to modify labels
        if self.modify_label():
            # If modification is successful, update self.parent.label_info
            self.parent.label_info = updated_label_info
            QtWidgets.QMessageBox.information(
                self,
                "Success",
                "Labels modified successfully!",
            )
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "An error occurred while updating the labels."
            )

    def modify_label(self):
        try:
            for label_file in self.label_file_list:
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

        for label_file in self.label_file_list:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            for shape in shapes:
                label = shape["label"]
                classes.add(label)

        for c in sorted(classes):
            # Update unique label list
            if not self.parent.unique_label_list.find_items_by_label(c):
                unique_label_item = (
                    self.parent.unique_label_list.create_item_from_label(c)
                )
                self.parent.unique_label_list.addItem(unique_label_item)
                rgb = self.parent._get_rgb_by_label(c, skip_label_info=True)
                self.parent.unique_label_list.set_item_label(
                    unique_label_item, c, rgb, self.opacity
                )
            # Update label info
            color = [0, 0, 0]
            opacity = 255
            items = self.parent.unique_label_list.find_items_by_label(c)
            for item in items:
                qlabel = self.parent.unique_label_list.itemWidget(item)
                if qlabel:
                    style_sheet = qlabel.styleSheet()
                    start_index = style_sheet.find("rgba(") + 5
                    end_index = style_sheet.find(")", start_index)
                    rgba_color = style_sheet[start_index:end_index].split(",")
                    rgba_color = [int(x.strip()) for x in rgba_color]
                    color = rgba_color[:-1]
                    opacity = rgba_color[-1]
                    break
            self.parent.label_info[c] = dict(
                delete=False,
                value=None,
                color=color,
                opacity=opacity,
            )


class TextInputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Text Input Dialog"))

        layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel(self.tr("Enter the text prompt below:"))
        self.text_input = QtWidgets.QLineEdit()

        self.ok_button = QtWidgets.QPushButton(self.tr("OK"))
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))

        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        layout.addWidget(self.label)
        layout.addWidget(self.text_input)
        layout.addWidget(self.ok_button)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

    def get_input_text(self):
        result = self.exec_()
        if result == QtWidgets.QDialog.Accepted:
            return self.text_input.text()
        else:
            return ""


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
        except:
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
