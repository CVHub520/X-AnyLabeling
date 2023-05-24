import re

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QCoreApplication

from .. import utils
from ..logger import logger

# TODO(unknown):
# - Calculate optimal position so as not to go out of screen area.


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
        layout.addWidget(bb)
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

    def pop_up(self, text=None, move=True, flags=None, group_id=None):
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
        if flags:
            self.set_flags(flags)
        else:
            self.reset_flags(text)
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
            return self.edit.text(), self.get_flags(), self.get_group_id()

        return None, None, None
