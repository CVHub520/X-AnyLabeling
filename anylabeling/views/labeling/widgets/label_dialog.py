import re
import json

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import QCoreApplication

from .. import utils
from ..logger import logger


# TODO(unknown):
# - Calculate optimal position so as not to go out of screen area.


class LabelModifyDialog(QtWidgets.QDialog):
    def __init__(self, label_file_list, parent=None, hidden_cls=[]):
        super().__init__(parent)
        self.label_file_list = label_file_list
        self.hidden_cls = hidden_cls
        self.label_info = self.get_classes()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Label Change Manager")
        self.setGeometry(100, 100, 600, 400)

        self.table_widget = QtWidgets.QTableWidget(self)
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(
            ["Category", "Delete", "New Value", "Hidden"]
        )

        # Set header font and alignment
        for i in range(4):
            self.table_widget.horizontalHeaderItem(i).setFont(
                QFont("Arial", 8, QFont.Bold)
            )
            self.table_widget.horizontalHeaderItem(i).setTextAlignment(
                QtCore.Qt.AlignCenter
            )

        self.buttons_layout = QtWidgets.QHBoxLayout()

        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)

        self.confirm_button = QtWidgets.QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.confirm_changes)

        self.buttons_layout.addWidget(self.cancel_button)
        self.buttons_layout.addWidget(self.confirm_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table_widget)
        layout.addLayout(self.buttons_layout)

        self.populate_table()

    def populate_table(self):
        for i, (class_name, info) in enumerate(self.label_info.items()):
            self.table_widget.insertRow(i)

            class_item = QtWidgets.QTableWidgetItem(class_name)
            class_item.setFlags(class_item.flags() ^ QtCore.Qt.ItemIsEditable)

            delete_checkbox = QtWidgets.QCheckBox()
            delete_checkbox.setChecked(info["delete"])
            delete_checkbox.setIcon(QIcon(":/images/images/delete.png"))
            delete_checkbox.stateChanged.connect(
                lambda state, row=i: self.on_delete_checkbox_changed(
                    row, state
                )
            )

            hidden_checkbox = QtWidgets.QCheckBox()
            hidden_checkbox.setChecked(info["hidden"])
            hidden_checkbox.setIcon(QIcon(":/images/images/hidden.png"))
            hidden_checkbox.stateChanged.connect(
                lambda state, row=i: self.on_hidden_checkbox_changed(
                    row, state
                )
            )

            delete_checkbox.setCheckable(not info["hidden"])

            value_item = QtWidgets.QTableWidgetItem(info["value"])
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

            self.table_widget.setItem(i, 0, class_item)
            self.table_widget.setCellWidget(i, 1, delete_checkbox)
            self.table_widget.setItem(i, 2, value_item)
            self.table_widget.setCellWidget(i, 3, hidden_checkbox)

    def on_delete_checkbox_changed(self, row, state):
        value_item = self.table_widget.item(row, 2)
        delete_checkbox = self.table_widget.cellWidget(row, 1)
        hidden_checkbox = self.table_widget.cellWidget(row, 3)

        if state == QtCore.Qt.Checked:
            value_item.setFlags(value_item.flags() & ~QtCore.Qt.ItemIsEditable)
            value_item.setBackground(QtGui.QColor("lightgray"))
            delete_checkbox.setCheckable(True)
            hidden_checkbox.setCheckable(False)
        else:
            value_item.setFlags(value_item.flags() | QtCore.Qt.ItemIsEditable)
            value_item.setBackground(QtGui.QColor("white"))
            delete_checkbox.setCheckable(False)
            hidden_checkbox.setCheckable(True)

        if value_item.text():
            delete_checkbox.setCheckable(False)
        else:
            delete_checkbox.setCheckable(True)

    def on_hidden_checkbox_changed(self, row, state):
        delete_checkbox = self.table_widget.cellWidget(row, 1)

        if state == QtCore.Qt.Checked:
            delete_checkbox.setCheckable(False)
        else:
            delete_checkbox.setCheckable(True)

    def confirm_changes(self):
        self.hidden_cls.clear()
        for i in range(self.table_widget.rowCount()):
            class_name = self.table_widget.item(i, 0).text()
            delete_checkbox = self.table_widget.cellWidget(i, 1)
            hidden_checkbox = self.table_widget.cellWidget(i, 3)
            value_item = self.table_widget.item(i, 2)

            self.label_info[class_name]["delete"] = delete_checkbox.isChecked()
            self.label_info[class_name]["value"] = value_item.text()
            if not delete_checkbox.isChecked() and hidden_checkbox.isChecked():
                self.hidden_cls.append(
                    class_name
                    if value_item.text() == ""
                    else value_item.text()
                )
        self.accept()
        if self._modify_label():
            QtWidgets.QMessageBox.information(
                self,
                "Success",
                "Labels modified successfully!",
            )
        else:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "An error occurred while updating the labels."
            )

    def _modify_label(self):
        try:
            for label_file in self.label_file_list:
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                src_shapes, dst_shapes = data["shapes"], []
                for shape in src_shapes:
                    label = shape["label"]
                    if self.label_info[label]["delete"]:
                        continue
                    if self.label_info[label]["value"]:
                        shape["label"] = self.label_info[label]["value"]
                    dst_shapes.append(shape)
                data["shapes"] = dst_shapes
                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error occurred while updating labels: {e}")
            return False

    def get_classes(self):
        classes = set()
        for label_file in self.label_file_list:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            for shape in shapes:
                label = shape["label"]
                classes.add(label)
        label_info = {}
        for c in classes:
            label_info[c] = dict(
                delete=False, value=None, hidden=c in self.hidden_cls
            )
        return label_info


class TextInputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Text Input Dialog")

        layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel("Enter the text prompt below:")
        self.text_input = QtWidgets.QLineEdit()

        self.ok_button = QtWidgets.QPushButton("OK")
        self.cancel_button = QtWidgets.QPushButton("Cancel")

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
        self.edit_difficult = QtWidgets.QCheckBox("useDifficult")
        self.edit_difficult.setChecked(difficult)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        if show_text_field:
            layout_edit = QtWidgets.QHBoxLayout()
            layout_edit.addWidget(self.edit, 6)
            layout_edit.addWidget(self.edit_group_id, 2)
            layout.addLayout(layout_edit)
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
            self.label_list.sortItems()
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
        # text edit
        self.edit_description = QtWidgets.QTextEdit()
        self.edit_description.setPlaceholderText("Label description")
        self.edit_description.setFixedHeight(50)
        layout.addWidget(self.edit_description)
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

    def get_last_label(self):
        return self._last_label

    def add_label_history(self, label):
        self._last_label = label
        if self.label_list.findItems(label, QtCore.Qt.MatchExactly):
            return
        self.label_list.addItem(label)
        if self._sort_labels:
            self.label_list.sortItems()

    def label_selected(self, item):
        self.edit.setText(item.text())

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

    def pop_up(
        self,
        text=None,
        move=True,
        flags=None,
        group_id=None,
        description=None,
        difficult=False,
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
                logger.warning("Label list has duplicate '%s'", text)
            self.label_list.setCurrentItem(items[0])
            row = self.label_list.row(items[0])
            self.edit.completer().setCurrentRow(row)
        self.edit.setFocus(QtCore.Qt.PopupFocusReason)
        if move:
            self.move(QtGui.QCursor.pos())
        if self.exec_():
            return (
                self.edit.text(),
                self.get_flags(),
                self.get_group_id(),
                self.get_description(),
                self.get_difficult_state(),
            )

        return None, None, None, None, False
