from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt

from .. import utils
from ..utils.style import get_dialog_style, get_scrollbar_style
from ..utils.theme import get_theme
from .combo_box import ComboBox
from .outline_button import OutlineButton
from .text_input import TextInput


def get_relation_dialog_style():
    t = get_theme()
    return get_dialog_style() + f"""
        QLineEdit, QComboBox, QListWidget {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 8px;
            margin: 0px;
            selection-background-color: {t["selection"]};
            selection-color: {t["selection_text"]};
        }}
        QLineEdit {{ padding: 2px 4px; height: 18px; }}
        QComboBox {{ padding: 2px 4px; min-height: 0px; }}
        QListWidget {{ padding: 0px; outline: none; }}
        QListWidget::item {{ padding: 4px; }}
        QLineEdit:hover, QComboBox:hover {{ border-color: {t["border_light"]}; }}
        QLineEdit:focus, QComboBox:focus {{ border: 1px solid {t["highlight"]}; }}
        QPushButton {{
            background-color: {t["surface"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 8px;
            margin: 0px;
            min-width: 72px;
            min-height: 22px;
            max-height: 22px;
            height: 22px;
            padding: 0 8px;
        }}
        QPushButton:hover {{ background-color: {t["surface_hover"]}; }}
        QPushButton:pressed {{ background-color: {t["surface_pressed"]}; }}
        QPushButton:disabled {{ color: {t["text_secondary"]}; }}
        {get_scrollbar_style()}
    """


class RelationDialog(QtWidgets.QDialog):
    def __init__(self, shapes, selected_shapes=(), relation=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            QtCore.QCoreApplication.translate(
                "LabelingWidget", "Edit Relation"
            )
            if relation
            else QtCore.QCoreApplication.translate(
                "LabelingWidget", "Create Relation"
            )
        )
        self.setMinimumWidth(360)
        self.setStyleSheet(get_relation_dialog_style())
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        subject = ComboBox(self)
        obj = ComboBox(self)
        for shape_index, shape in enumerate(shapes, 1):
            name = shape.label or QtCore.QCoreApplication.translate(
                "LabelingWidget", "Unlabeled"
            )
            if shape.group_id is not None:
                name += f" #{shape.group_id}"
            else:
                name += f" [{shape_index}]"
            for combo in (subject, obj):
                combo.addItem(name, shape)
        if relation:
            for combo, key in ((subject, "subject_id"), (obj, "object_id")):
                for i, shape in enumerate(shapes):
                    if shape.other_data.get("shape_id") == relation.get(key):
                        combo.setCurrentIndex(i)
                        break
        elif len(selected_shapes) >= 2:
            for combo, shape in zip((subject, obj), selected_shapes):
                if shape in shapes:
                    combo.setCurrentIndex(shapes.index(shape))
        else:
            obj.setCurrentIndex(1)
        label = TextInput(relation.get("label", "") if relation else "", self)
        form.addRow(
            QtCore.QCoreApplication.translate("LabelingWidget", "Subject"),
            subject,
        )
        form.addRow(
            QtCore.QCoreApplication.translate("LabelingWidget", "Relation"),
            label,
        )
        form.addRow(
            QtCore.QCoreApplication.translate("LabelingWidget", "Object"), obj
        )
        for field in (subject, label, obj):
            form.labelForField(field).setFixedHeight(field.height())
        layout.addLayout(form)
        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(6)
        if relation is not None:
            delete = OutlineButton(
                QtCore.QCoreApplication.translate("LabelingWidget", "Delete"),
                self,
            )
            delete.setObjectName("relationDeleteButton")
            delete.setIcon(utils.new_icon("trash"))
            delete.clicked.connect(lambda: self.done(2))
            button_row.addWidget(delete)
        button_row.addStretch()
        cancel = OutlineButton(
            QtCore.QCoreApplication.translate("LabelingWidget", "Cancel"), self
        )
        cancel.setObjectName("relationCancelButton")
        cancel.setIcon(utils.new_icon("undo"))
        cancel.clicked.connect(self.reject)
        button_row.addWidget(cancel)
        confirm = OutlineButton(
            QtCore.QCoreApplication.translate("LabelDialog", "OK"), self
        )
        confirm.setObjectName("relationConfirmButton")
        confirm.setIcon(utils.new_icon("check"))
        confirm.setDefault(True)
        confirm.clicked.connect(self.accept)
        button_row.addWidget(confirm)
        layout.addLayout(button_row)
        self.subject = subject
        self.object = obj
        self.label = label

    def values(self):
        return (
            self.subject.currentData(),
            self.object.currentData(),
            self.label.text().strip(),
        )


class RelationManagerDialog(QtWidgets.QDialog):
    edit_requested = QtCore.pyqtSignal(int)
    preview_requested = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            QtCore.QCoreApplication.translate(
                "LabelingWidget", "Manage Relations"
            )
        )
        self.resize(480, 320)
        self.setStyleSheet(get_relation_dialog_style())
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.relation_list = QtWidgets.QListWidget(self)
        layout.addWidget(self.relation_list)
        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(6)
        self.edit_button = OutlineButton(
            QtCore.QCoreApplication.translate("LabelingWidget", "Edit"), self
        )
        self.edit_button.setObjectName("relationEditButton")
        self.edit_button.setIcon(utils.new_icon("edit"))
        self.edit_button.setEnabled(False)
        button_row.addWidget(self.edit_button)
        button_row.addStretch()
        close = OutlineButton(
            QtCore.QCoreApplication.translate("LabelingWidget", "Close"), self
        )
        close.setObjectName("relationCloseButton")
        close.setIcon(utils.new_icon("undo"))
        close.clicked.connect(self.reject)
        button_row.addWidget(close)
        layout.addLayout(button_row)

        self.relation_list.itemSelectionChanged.connect(self._preview_selected)
        self.relation_list.itemDoubleClicked.connect(self._edit_selected)
        self.edit_button.clicked.connect(self._edit_selected)

    def selected_index(self):
        selected = self.relation_list.selectedItems()
        return selected[0].data(Qt.ItemDataRole.UserRole) if selected else None

    def _preview_selected(self):
        index = self.selected_index()
        self.edit_button.setEnabled(index is not None)
        self.preview_requested.emit(index)

    def _edit_selected(self):
        index = self.selected_index()
        if index is not None:
            self.edit_requested.emit(index)

    def refresh(self, shapes, relations):
        self.relation_list.clear()
        shapes = {
            shape.other_data.get("shape_id"): (index, shape)
            for index, shape in enumerate(shapes, 1)
            if shape.other_data.get("shape_id")
        }
        for index, relation in enumerate(relations):
            subject = shapes.get(relation.get("subject_id"))
            obj = shapes.get(relation.get("object_id"))
            if subject is None or obj is None:
                continue

            def shape_name(item):
                shape_index, shape = item
                suffix = (
                    f" #{shape.group_id}"
                    if shape.group_id is not None
                    else f" [{shape_index}]"
                )
                return f"{shape.label}{suffix}"

            text = f"{shape_name(subject)} → {relation.get('label', '')} → {shape_name(obj)}"
            if relation.get("score") is not None:
                text += f" ({relation['score']:.2f})"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.relation_list.addItem(item)


def show_relation_message(icon, text, parent=None):
    message = QtWidgets.QMessageBox(
        icon,
        QtCore.QCoreApplication.translate("LabelingWidget", "Relation"),
        text,
        QtWidgets.QMessageBox.StandardButton.NoButton,
        parent,
    )
    message.addButton(
        QtCore.QCoreApplication.translate("LabelingWidget", "OK"),
        QtWidgets.QMessageBox.ButtonRole.AcceptRole,
    )
    message.exec()
