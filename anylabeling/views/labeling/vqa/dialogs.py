import json
import os

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QWheelEvent, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QGraphicsDropShadowEffect,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.vqa.config import (
    AI_PROMPT_PLACEHOLDER,
    DEFAULT_COMPONENT_WINDOW_SIZE,
    DEFAULT_TEMPLATES,
    PROMPTS_CONFIG_PATH,
    SUPPORTED_WIDGETS,
)
from anylabeling.views.labeling.vqa.style import (
    get_content_input_style,
    get_component_dialog_combobox_style,
    get_dialog_button_style,
    get_filename_label_style,
    get_name_input_style,
    get_message_label_style,
    get_page_input_style,
    get_prompt_input_style,
    get_status_label_style,
    get_table_style,
    get_title_label_style,
    get_ui_style,
)


class QComboBox(QComboBox):
    def wheelEvent(self, e: QWheelEvent) -> None:
        pass


class PromptTemplateDialog(QDialog):
    """Prompt template management dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle(self.tr("Template Gallery"))
        self.setMinimumSize(600, 450)
        self.selected_template = None
        self.setup_ui()
        self.load_templates()

    def setup_ui(self):
        self.setStyleSheet(get_ui_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(
            [self.tr("Select"), self.tr("Template Name"), self.tr("Action")]
        )

        self.table.setStyleSheet(get_table_style())
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.itemDoubleClicked.connect(self.on_item_double_clicked)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.resizeSection(0, 100)
        header.resizeSection(2, 120)

        layout.addWidget(self.table)

        button_layout = QHBoxLayout()

        add_button = QPushButton(self.tr("Add"))
        add_button.setStyleSheet(get_dialog_button_style("success", "medium"))
        add_button.clicked.connect(self.add_template)
        button_layout.addWidget(add_button)
        button_layout.addStretch()

        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        cancel_btn.clicked.connect(self.reject)

        ok_btn = QPushButton(self.tr("OK"))
        ok_btn.setStyleSheet(get_dialog_button_style("primary", "medium"))
        ok_btn.clicked.connect(self.accept)

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)

    def get_default_templates(self):
        """Get default system templates"""
        return [
            {"name": name, "content": content, "is_system": True}
            for name, content in DEFAULT_TEMPLATES.items()
        ]

    def load_templates(self):
        """Load templates from config file"""
        user_templates = []
        system_templates = self.get_default_templates()

        if os.path.exists(PROMPTS_CONFIG_PATH):
            try:
                with open(PROMPTS_CONFIG_PATH, "r", encoding="utf-8") as f:
                    user_templates = json.load(f)
                    for template in user_templates:
                        template["is_system"] = False
            except Exception as e:
                print(f"Error loading templates: {e}")

        templates = user_templates + system_templates
        self.populate_table(templates)

    def populate_table(self, templates):
        """Populate table with templates"""
        self.table.setRowCount(len(templates))

        for row in range(len(templates)):
            self.table.setRowHeight(row, 36)

        for row, template in enumerate(templates):
            checkbox = QCheckBox()
            checkbox.toggled.connect(
                lambda checked, r=row: self.on_checkbox_toggled(r, checked)
            )
            checkbox_widget = QWidget()
            checkbox_layout = QVBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.setSpacing(0)

            self.table.setCellWidget(row, 0, checkbox_widget)

            name_item = QTableWidgetItem(template["name"])
            if template["is_system"]:
                name_item.setForeground(QColor("#6366F1"))
            else:
                name_item.setForeground(QColor("#374151"))
            name_item.setToolTip(template["content"])
            name_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.table.setItem(row, 1, name_item)

            delete_btn = QPushButton(self.tr("Delete"))
            delete_btn.setFixedSize(70, 24)

            if template["is_system"]:
                delete_btn.setEnabled(False)
                delete_btn.setStyleSheet(
                    get_dialog_button_style(
                        "secondary", "small", disabled=True
                    )
                )
            else:
                delete_btn.setStyleSheet(
                    get_dialog_button_style("danger", "small")
                )
                delete_btn.clicked.connect(
                    lambda _, r=row: self.delete_template(r)
                )

            delete_widget = QWidget()
            delete_layout = QVBoxLayout(delete_widget)
            delete_layout.addWidget(delete_btn)
            delete_layout.setAlignment(Qt.AlignCenter)
            delete_layout.setContentsMargins(0, 0, 0, 0)
            delete_layout.setSpacing(0)

            self.table.setCellWidget(row, 2, delete_widget)

    def on_checkbox_toggled(self, row, checked):
        """Handle checkbox toggle with mutual exclusion"""
        if checked:
            for i in range(self.table.rowCount()):
                if i != row:
                    checkbox_widget = self.table.cellWidget(i, 0)
                    if checkbox_widget:
                        checkbox = checkbox_widget.findChild(QCheckBox)
                        if checkbox:
                            checkbox.blockSignals(True)
                            checkbox.setChecked(False)
                            checkbox.blockSignals(False)

            name_item = self.table.item(row, 1)
            if name_item:
                self.selected_template = {
                    "name": name_item.text(),
                    "content": name_item.toolTip(),
                }
        else:
            self.selected_template = None

    def add_template(self):
        """Add new template"""
        dialog = AddTemplateDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            template_data = dialog.get_template_data()
            self.save_user_template(
                template_data["name"], template_data["content"]
            )
            self.load_templates()

    def delete_template(self, row):
        """Delete template"""
        name_item = self.table.item(row, 1)
        if not name_item:
            return

        reply = QMessageBox.question(
            self,
            self.tr("Delete Template"),
            self.tr("Are you sure you want to delete this template?"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.remove_user_template(name_item.text())
            self.load_templates()

    def save_user_template(self, name, content):
        """Save user template to config file"""
        user_templates = []

        if os.path.exists(PROMPTS_CONFIG_PATH):
            try:
                with open(PROMPTS_CONFIG_PATH, "r", encoding="utf-8") as f:
                    user_templates = json.load(f)
            except Exception:
                user_templates = []

        user_templates.append({"name": name, "content": content})

        os.makedirs(os.path.dirname(PROMPTS_CONFIG_PATH), exist_ok=True)
        with open(PROMPTS_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(user_templates, f, ensure_ascii=False, indent=2)

    def remove_user_template(self, name):
        """Remove user template from config file"""
        if not os.path.exists(PROMPTS_CONFIG_PATH):
            return

        try:
            with open(PROMPTS_CONFIG_PATH, "r", encoding="utf-8") as f:
                user_templates = json.load(f)

            user_templates = [t for t in user_templates if t["name"] != name]

            with open(PROMPTS_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(user_templates, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error removing template: {e}")

    def get_selected_template(self):
        """Get selected template"""
        return self.selected_template

    def on_item_double_clicked(self, item):
        """Handle double click on table item"""
        if item.column() != 1:
            return

        row = item.row()
        name_item = self.table.item(row, 1)
        if not name_item:
            return

        if name_item.foreground().color() == QColor(
            "#6366F1"
        ):  # System template
            return

        template_data = {
            "name": name_item.text(),
            "content": name_item.toolTip(),
        }

        dialog = AddTemplateDialog(self, edit_data=template_data)
        if dialog.exec_() == QDialog.Accepted:
            new_data = dialog.get_template_data()
            self.update_user_template(
                template_data["name"], new_data["name"], new_data["content"]
            )
            self.load_templates()

    def update_user_template(self, old_name, new_name, new_content):
        """Update existing user template"""
        if not os.path.exists(PROMPTS_CONFIG_PATH):
            return

        try:
            with open(PROMPTS_CONFIG_PATH, "r", encoding="utf-8") as f:
                user_templates = json.load(f)

            for template in user_templates:
                if template["name"] == old_name:
                    template["name"] = new_name
                    template["content"] = new_content
                    break

            with open(PROMPTS_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(user_templates, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error updating template: {e}")


class AddTemplateDialog(QDialog):
    """Custom dialog for adding new templates"""

    def __init__(self, parent=None, edit_data=None):
        super().__init__(parent)
        self.parent = parent
        self.edit_data = edit_data
        self.is_edit_mode = edit_data is not None

        if self.is_edit_mode:
            self.setWindowTitle(self.tr("Edit Template"))
        else:
            self.setWindowTitle(self.tr("Add Template"))

        self.setModal(True)
        self.setFixedSize(520, 350)
        self.setup_ui()

        if self.is_edit_mode:
            self.prefill_data()

    def setup_ui(self):
        self.setStyleSheet(get_ui_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        name_label = QLabel(self.tr("Template Name:"))
        name_label.setStyleSheet(get_filename_label_style())
        layout.addWidget(name_label)

        self.name_input = QLineEdit()
        self.name_input.setStyleSheet(get_name_input_style())
        layout.addWidget(self.name_input)

        content_label = QLabel(self.tr("Template Content:"))
        content_label.setStyleSheet(get_filename_label_style())
        layout.addWidget(content_label)

        self.content_input = QTextEdit()
        self.content_input.setStyleSheet(get_content_input_style())
        self.content_input.setMinimumHeight(120)
        layout.addWidget(self.content_input)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        cancel_btn.clicked.connect(self.reject)

        ok_btn = QPushButton(self.tr("OK"))
        ok_btn.setStyleSheet(get_dialog_button_style("primary", "medium"))
        ok_btn.clicked.connect(self.validate_and_accept)

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)

    def prefill_data(self):
        """Prefill data for edit mode"""
        if self.edit_data:
            self.name_input.setText(self.edit_data["name"])
            self.content_input.setPlainText(self.edit_data["content"])

    def validate_and_accept(self):
        """Validate input and accept dialog"""
        name = self.name_input.text().strip()
        content = self.content_input.toPlainText().strip()

        if not name:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Template name cannot be empty!"),
            )
            return

        if not content:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Template content cannot be empty!"),
            )
            return

        self.accept()

    def get_template_data(self):
        """Get template data"""
        return {
            "name": self.name_input.text().strip(),
            "content": self.content_input.toPlainText().strip(),
        }


class AILoadingDialog(QDialog):
    """AI loading dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle(self.tr("AI Processing"))
        self.setFixedSize(420, 180)
        self.setModal(True)
        self.dot_count = 1
        self.setup_ui()
        self.setup_animation()

    def setup_ui(self):
        self.setStyleSheet(get_ui_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        content_layout = QVBoxLayout()
        content_layout.setSpacing(12)

        title_label = QLabel(self.tr("AI Processing"))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(get_title_label_style())
        content_layout.addWidget(title_label)

        self.message_label = QLabel(
            self.tr("Generating content, please wait.")
        )
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet(get_message_label_style())
        content_layout.addWidget(self.message_label)

        layout.addLayout(content_layout)
        layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addStretch()

        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        self.cancel_button.setCursor(Qt.PointingHandCursor)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setWindowFlags(
            Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 25))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

    def setup_animation(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_dots)
        self.timer.start(300)

    def update_dots(self):
        base_text = self.tr("Generating content, please wait")
        dots = "." * self.dot_count
        spaces = " " * (3 - self.dot_count)
        self.message_label.setText(base_text + dots + spaces)
        self.dot_count = self.dot_count % 3 + 1

    def center_on_parent(self):
        if self.parent:
            center_point = self.parent.mapToGlobal(self.parent.rect().center())
            dialog_rect = self.rect()
            self.move(
                center_point.x() - dialog_rect.width() // 2,
                center_point.y() - dialog_rect.height() // 2,
            )

    def exec_(self):
        self.center_on_parent()
        return super().exec_()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)

    def reject(self):
        self.timer.stop()
        super().reject()


class AIPromptDialog(QDialog):
    """AI prompt dialog for QLineEdit components"""

    def __init__(self, parent=None, current_text=""):
        super().__init__(parent)
        self.parent = parent
        self.current_text = current_text
        self.setWindowTitle(self.tr("AI Assistance"))
        self.setMinimumWidth(600)
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI interface"""
        self.setStyleSheet(get_ui_style())

        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(24, 24, 24, 24)
        dialog_layout.setSpacing(12)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(AI_PROMPT_PLACEHOLDER)
        self.prompt_input.setStyleSheet(get_prompt_input_style())
        self.prompt_input.setAcceptRichText(True)
        self.prompt_input.textChanged.connect(self.on_text_changed)
        self.prompt_input.setMinimumHeight(120)
        self.prompt_input.setMaximumHeight(160)
        self.prompt_input.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        dialog_layout.addWidget(self.prompt_input)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 8, 0, 0)
        button_layout.setSpacing(12)

        template_btn = QPushButton(self.tr("Templates"))
        template_btn.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        template_btn.setCursor(Qt.PointingHandCursor)
        template_btn.clicked.connect(self.open_template_library)

        button_layout.addWidget(template_btn)
        button_layout.addStretch()

        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)

        confirm_btn = QPushButton(self.tr("Generate"))
        confirm_btn.setStyleSheet(get_dialog_button_style("primary", "medium"))
        confirm_btn.setCursor(Qt.PointingHandCursor)
        confirm_btn.clicked.connect(self.accept)

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(confirm_btn)
        dialog_layout.addLayout(button_layout)

        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setWindowFlags(self.windowFlags() & ~Qt.FramelessWindowHint)

    def open_template_library(self):
        """Open template library dialog"""
        dialog = PromptTemplateDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            selected_template = dialog.get_selected_template()
            if selected_template:
                self.prompt_input.setPlainText(selected_template["content"])

    def center_on_parent(self):
        """Center the dialog on the parent window"""
        if self.parent:
            center_point = self.parent.mapToGlobal(self.parent.rect().center())
            dialog_rect = self.rect()
            self.move(
                center_point.x() - dialog_rect.width() // 2,
                center_point.y() - dialog_rect.height() // 2,
            )

    def get_prompt(self):
        """Get the user input prompt"""
        return self.prompt_input.toPlainText().strip()

    def exec_(self):
        """Override exec_ method to adjust position before showing the dialog"""
        self.adjustSize()
        self.center_on_parent()
        return super().exec_()

    def on_text_changed(self):
        """Handle text changes in the message input to highlight @image and @text tags"""
        cursor = self.prompt_input.textCursor()
        current_position = cursor.position()
        document = self.prompt_input.document()
        text = document.toPlainText()

        self.prompt_input.blockSignals(True)

        cursor.select(QTextCursor.Document)
        format = QTextCharFormat()
        cursor.setCharFormat(format)

        tags = ["@image", "@text"]
        if hasattr(self.parent, "custom_components"):
            tags.extend(
                [
                    f"@widget.{comp['title']}"
                    for comp in self.parent.custom_components
                    if comp["type"] == "QLineEdit"
                ]
            )

        tags.extend(
            [
                "@label.shapes",
                "@label.imagePath",
                "@label.imageHeight",
                "@label.imageWidth",
                "@label.flags",
            ]
        )

        for tag in tags:
            start_index = 0
            while True:
                start_index = text.find(tag, start_index)
                if start_index == -1:
                    break

                highlight_format = QTextCharFormat()
                highlight_format.setBackground(QColor("#E3F2FD"))
                highlight_format.setForeground(QColor("#1976D2"))

                cursor.setPosition(start_index)
                cursor.setPosition(
                    start_index + len(tag), QTextCursor.KeepAnchor
                )
                cursor.setCharFormat(highlight_format)

                start_index += len(tag)

        cursor.setPosition(current_position)
        self.prompt_input.setTextCursor(cursor)
        self.prompt_input.blockSignals(False)


class ComponentDialog(QDialog):
    """
    A dialog for creating or editing a form component with a title, type, and optional options.

    This modal dialog allows users to define or update a form component such as a QLineEdit,
    QRadioButton, QComboBox, or QCheckBox field. It validates user input and prevents
    duplicate titles or empty option lists for applicable component types.

    Attributes:
        parent_dialog (QWidget): The parent dialog or widget.
        edit_data (dict or None): Data for editing an existing component, or None for adding a new one.
        title_input (QLineEdit): Input field for the component title.
        type_combo (QComboBox): Dropdown for selecting the component type.
        options_input (QTextEdit): Text area for entering options (one per line).
        ok_button (QPushButton): Button to confirm the dialog and validate input.
        cancel_button (QPushButton): Button to cancel and close the dialog.
    """

    def __init__(self, parent=None, edit_data=None):
        """
        Initializes the ComponentDialog with optional edit data.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
            edit_data (dict, optional): Component data for editing. If provided, fields are pre-filled.
        """
        super().__init__(parent)
        self.parent_dialog = parent
        self.edit_data = edit_data
        self.setWindowTitle(
            self.tr("Edit Component")
            if edit_data
            else self.tr("Add Component")
        )
        self.setModal(True)
        self.setFixedSize(*DEFAULT_COMPONENT_WINDOW_SIZE)

        layout = QVBoxLayout(self)

        title_label = QLabel(self.tr("Component Title:"))
        title_label.setStyleSheet(get_filename_label_style())
        self.title_input = QLineEdit()
        self.title_input.setStyleSheet(get_page_input_style())
        if edit_data:
            self.title_input.setText(edit_data["title"])
        layout.addWidget(title_label)
        layout.addWidget(self.title_input)

        type_label = QLabel(self.tr("Component Type:"))
        type_label.setStyleSheet(get_filename_label_style())
        self.type_combo = QComboBox()
        self.type_combo.addItems(SUPPORTED_WIDGETS)
        if edit_data:
            self.type_combo.setCurrentText(edit_data["type"])
            self.type_combo.setEnabled(False)
        layout.addWidget(type_label)
        layout.addWidget(self.type_combo)

        options_label = QLabel(self.tr("Options (one per line):"))
        options_label.setStyleSheet(get_filename_label_style())
        self.options_input = QTextEdit()
        self.options_input.setFixedHeight(100)
        if edit_data and edit_data["options"]:
            self.options_input.setText("\n".join(edit_data["options"]))
        layout.addWidget(options_label)
        layout.addWidget(self.options_input)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.ok_button = QPushButton(self.tr("OK"))
        self.ok_button.setStyleSheet(
            get_dialog_button_style("primary", "medium")
        )
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )

        self.ok_button.clicked.connect(self.validate_and_accept)
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        self.on_type_changed()

    def on_type_changed(self):
        """
        Enables or disables the options input based on the selected component type.

        If the selected type is "QLineEdit", the options field is disabled since it's not applicable.
        """
        text_type = self.type_combo.currentText() == "QLineEdit"
        self.options_input.setEnabled(not text_type)

    def validate_and_accept(self):
        """
        Validates user input before accepting the dialog.

        Performs checks for:
            - Non-empty title.
            - Unique title within the parent dialog, if applicable.
            - Required non-empty and non-duplicate options for non-text components.

        If validation passes, the dialog is accepted; otherwise, a warning is shown.
        """
        title = self.title_input.text().strip()
        if not title:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Component title cannot be empty!"),
            )
            return

        if hasattr(self.parent_dialog, "check_duplicate_title"):
            original_title = (
                self.edit_data["title"] if self.edit_data else None
            )
            if self.parent_dialog.check_duplicate_title(title, original_title):
                QMessageBox.warning(
                    self,
                    self.tr("Warning"),
                    self.tr("Component title already exists!"),
                )
                return

        comp_type = self.type_combo.currentText()
        if comp_type != "QLineEdit":
            options_text = self.options_input.toPlainText().strip()
            if not options_text:
                QMessageBox.warning(
                    self,
                    self.tr("Warning"),
                    self.tr(
                        "Options cannot be empty for this component type!"
                    ),
                )
                return

            options = [
                opt.strip() for opt in options_text.split("\n") if opt.strip()
            ]
            if len(options) != len(set(options)):
                QMessageBox.warning(
                    self,
                    self.tr("Warning"),
                    self.tr("Duplicate options are not allowed!"),
                )
                return

        self.accept()

    def get_component_data(self):
        """
        Returns the data entered in the dialog as a dictionary.

        Returns:
            dict: A dictionary containing the component's title, type, and options list.
        """
        if self.options_input.toPlainText():
            options = self.options_input.toPlainText().split("\n")
        else:
            options = []
        return {
            "title": self.title_input.text(),
            "type": self.type_combo.currentText(),
            "options": options,
        }


class DeleteComponentDialog(QDialog):
    """
    A dialog that allows users to select and delete components from a table.

    This modal dialog presents a table of existing components with columns for
    type, title, and selection checkbox. Users can select multiple components
    to delete using checkboxes.
    """

    def __init__(self, components, parent=None):
        """
        Initializes the DeleteComponentDialog with a list of components.

        Args:
            components (list): A list of component dictionaries, each with 'title' and 'type' keys.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.components = components
        self.setWindowTitle(self.tr("Delete Components"))
        self.setModal(True)

        base_height = 200
        row_height = 35
        table_height = max(150, len(components) * row_height + 50)
        total_height = min(600, base_height + table_height)

        self.setFixedSize(520, total_height)

        layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        top_layout.addStretch()

        select_all_layout = QHBoxLayout()
        select_all_label = QLabel(self.tr("Select All:"))
        select_all_label.setStyleSheet(get_filename_label_style())

        self.select_all_checkbox = QCheckBox()
        self.select_all_checkbox.setToolTip(
            self.tr("Select/Deselect All Components")
        )
        self.select_all_checkbox.stateChanged.connect(
            self.on_select_all_changed
        )

        select_all_layout.addWidget(select_all_label)
        select_all_layout.addWidget(self.select_all_checkbox)
        top_layout.addLayout(select_all_layout)

        layout.addLayout(top_layout)

        self.component_table = QTableWidget()
        self.component_table.setRowCount(len(components))
        self.component_table.setColumnCount(3)

        headers = [self.tr("Type"), self.tr("Title"), self.tr("Select")]
        self.component_table.setHorizontalHeaderLabels(headers)

        self.component_table.setSelectionBehavior(QTableWidget.SelectItems)
        self.component_table.setSelectionMode(QTableWidget.NoSelection)
        self.component_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.component_table.setAlternatingRowColors(True)
        self.component_table.verticalHeader().setVisible(False)
        self.component_table.setFocusPolicy(Qt.NoFocus)
        self.component_table.verticalHeader().setDefaultSectionSize(32)
        min_table_height = len(components) * 32 + 30
        self.component_table.setMinimumHeight(min_table_height)
        self.component_table.setStyleSheet(
            get_component_dialog_combobox_style()
        )

        header = self.component_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.component_table.setColumnWidth(2, 80)

        self.populate_table()
        self.component_table.resizeRowsToContents()
        layout.addWidget(self.component_table, 1)
        button_layout = QHBoxLayout()

        self.status_label = QLabel(self.tr("No components selected"))
        self.status_label.setStyleSheet(get_status_label_style())
        button_layout.addWidget(self.status_label)
        button_layout.addStretch()

        self.delete_button = QPushButton(self.tr("Delete"))
        self.delete_button.setStyleSheet(
            get_dialog_button_style("danger", "medium")
        )
        self.delete_button.setEnabled(False)

        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )

        self.delete_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

    def populate_table(self):
        """Populate table data"""
        for row, component in enumerate(self.components):
            # First column: Component type
            type_item = QTableWidgetItem(component["type"])
            type_item.setTextAlignment(Qt.AlignCenter)
            type_colors = {
                "QLineEdit": QColor("#4299e1"),  # Blue
                "QRadioButton": QColor("#48bb78"),  # Green
                "QComboBox": QColor("#ed8936"),  # Orange
                "QCheckBox": QColor("#9f7aea"),  # Purple
            }
            color = type_colors.get(component["type"], QColor("#718096"))
            type_item.setData(Qt.ForegroundRole, color)
            self.component_table.setItem(row, 0, type_item)

            # Second column: Component title
            title_item = QTableWidgetItem(component["title"])
            title_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.component_table.setItem(row, 1, title_item)

            # Third column: Checkbox with default style
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.setAlignment(Qt.AlignCenter)

            checkbox = QCheckBox()
            checkbox.stateChanged.connect(self.on_item_selection_changed)
            checkbox_layout.addWidget(checkbox)

            self.component_table.setCellWidget(row, 2, checkbox_widget)

    def on_select_all_changed(self, state):
        """Handle select all checkbox state change"""
        is_checked = state == Qt.Checked

        for row in range(self.component_table.rowCount()):
            checkbox_widget = self.component_table.cellWidget(row, 2)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(is_checked)
                    checkbox.blockSignals(False)

        self.update_ui_state()

    def on_item_selection_changed(self):
        """Handle individual checkbox state change"""
        selected_count = self.get_selected_count()
        total_count = self.component_table.rowCount()

        self.select_all_checkbox.blockSignals(True)
        if selected_count == 0:
            self.select_all_checkbox.setCheckState(Qt.Unchecked)
        elif selected_count == total_count:
            self.select_all_checkbox.setCheckState(Qt.Checked)
        else:
            self.select_all_checkbox.setCheckState(Qt.PartiallyChecked)
        self.select_all_checkbox.blockSignals(False)

        self.update_ui_state()

    def get_selected_count(self):
        """Get count of selected checkboxes"""
        count = 0
        for row in range(self.component_table.rowCount()):
            checkbox_widget = self.component_table.cellWidget(row, 2)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    count += 1
        return count

    def update_ui_state(self):
        """Update UI state (buttons and status label)"""
        selected_count = self.get_selected_count()
        has_selection = selected_count > 0

        self.delete_button.setEnabled(has_selection)

        if selected_count == 0:
            status_text = self.tr("No components selected")
        elif selected_count == 1:
            status_text = self.tr("1 component selected")
        else:
            status_text = self.tr("%d components selected") % selected_count

        self.status_label.setText(status_text)

    def get_selected_indices(self):
        """Get indices of all selected components"""
        indices = []
        for row in range(self.component_table.rowCount()):
            checkbox_widget = self.component_table.cellWidget(row, 2)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    indices.append(row)

        return sorted(indices, reverse=True)

    def get_selected_index(self):
        """Get index of first selected component (for backwards compatibility)"""
        indices = self.get_selected_indices()
        return indices[-1] if indices else -1


class ExportLabelsDialog(QDialog):
    """Dialog for selecting and configuring label export options"""

    def __init__(self, components, parent=None):
        super().__init__(parent)
        self.components = components
        self.setWindowTitle(self.tr("Export Labels"))
        self.setModal(True)

        # Calculate dialog size based on content
        base_height = 280
        row_height = 35
        field_count = len(components) + 4
        table_height = max(150, field_count * row_height + 50)
        total_height = min(500, base_height + table_height)

        self.setFixedSize(520, total_height)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Top controls
        top_layout = QHBoxLayout()
        top_layout.addStretch()

        select_all_layout = QHBoxLayout()
        select_all_label = QLabel(self.tr("Select All:"))
        select_all_label.setStyleSheet(get_filename_label_style())

        self.select_all_checkbox = QCheckBox()
        self.select_all_checkbox.setToolTip(
            self.tr("Select/Deselect All Fields")
        )
        self.select_all_checkbox.stateChanged.connect(
            self.on_select_all_changed
        )

        select_all_layout.addWidget(select_all_label)
        select_all_layout.addWidget(self.select_all_checkbox)
        top_layout.addLayout(select_all_layout)

        layout.addLayout(top_layout)

        # Export fields table
        self.export_table = QTableWidget()
        field_count = len(components) + 4
        self.export_table.setRowCount(field_count)
        self.export_table.setColumnCount(4)

        headers = [
            self.tr("Type"),
            self.tr("Original Key"),
            self.tr("Export Key"),
            self.tr("Select"),
        ]
        self.export_table.setHorizontalHeaderLabels(headers)

        self.export_table.setSelectionBehavior(QTableWidget.SelectItems)
        self.export_table.setSelectionMode(QTableWidget.NoSelection)
        self.export_table.setAlternatingRowColors(True)
        self.export_table.verticalHeader().setVisible(False)
        self.export_table.setFocusPolicy(Qt.NoFocus)
        self.export_table.verticalHeader().setDefaultSectionSize(32)
        min_table_height = field_count * 32 + 30
        max_table_height = total_height - 120
        self.export_table.setMaximumHeight(max_table_height)
        self.export_table.setMinimumHeight(
            min(min_table_height, max_table_height)
        )
        self.export_table.setStyleSheet(get_component_dialog_combobox_style())

        header = self.export_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        self.export_table.setColumnWidth(3, 80)

        layout.addWidget(self.export_table)

        button_layout = QHBoxLayout()

        self.status_label = QLabel(self.tr("No fields selected"))
        self.status_label.setStyleSheet(get_status_label_style())
        button_layout.addWidget(self.status_label)
        button_layout.addStretch()

        self.export_button = QPushButton(self.tr("Export"))
        self.export_button.setStyleSheet(
            get_dialog_button_style("primary", "medium")
        )
        self.export_button.setEnabled(False)

        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )

        self.export_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Populate table after all widgets are created
        self.populate_table()
        self.export_table.resizeRowsToContents()

    def populate_table(self):
        """Populate table with basic fields and component fields"""
        row = 0

        # Basic fields
        basic_fields = [
            ("Basic", "image", "image"),
            ("Basic", "width", "width"),
            ("Basic", "height", "height"),
            ("Basic", "shapes", "shapes"),
        ]

        for field_type, original_key, export_key in basic_fields:
            type_item = QTableWidgetItem(field_type)
            type_item.setTextAlignment(Qt.AlignCenter)
            type_item.setData(Qt.ForegroundRole, QColor("#718096"))
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            self.export_table.setItem(row, 0, type_item)

            original_item = QTableWidgetItem(original_key)
            original_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            original_item.setFlags(original_item.flags() & ~Qt.ItemIsEditable)
            self.export_table.setItem(row, 1, original_item)

            export_input = QLineEdit()
            export_input.setText(export_key)
            export_input.setStyleSheet(get_page_input_style())
            self.export_table.setCellWidget(row, 2, export_input)

            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.setAlignment(Qt.AlignCenter)

            checkbox = QCheckBox()
            checkbox.setChecked(original_key != "shapes")
            checkbox.stateChanged.connect(self.on_item_selection_changed)
            checkbox_layout.addWidget(checkbox)

            self.export_table.setCellWidget(row, 3, checkbox_widget)
            row += 1

        type_colors = {
            "QLineEdit": QColor("#4299e1"),
            "QRadioButton": QColor("#48bb78"),
            "QComboBox": QColor("#ed8936"),
            "QCheckBox": QColor("#9f7aea"),
        }

        for component in self.components:

            type_item = QTableWidgetItem(component["type"])
            type_item.setTextAlignment(Qt.AlignCenter)
            color = type_colors.get(component["type"], QColor("#718096"))
            type_item.setData(Qt.ForegroundRole, color)
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            self.export_table.setItem(row, 0, type_item)

            original_item = QTableWidgetItem(component["title"])
            original_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            original_item.setFlags(original_item.flags() & ~Qt.ItemIsEditable)
            self.export_table.setItem(row, 1, original_item)

            export_input = QLineEdit()
            export_input.setText(component["title"])
            export_input.setStyleSheet(get_page_input_style())
            self.export_table.setCellWidget(row, 2, export_input)

            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.setAlignment(Qt.AlignCenter)

            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.on_item_selection_changed)
            checkbox_layout.addWidget(checkbox)

            self.export_table.setCellWidget(row, 3, checkbox_widget)
            row += 1

        self.update_ui_state()

    def on_select_all_changed(self, state):
        """Handle select all checkbox state change"""
        is_checked = state == Qt.Checked

        for row in range(self.export_table.rowCount()):
            checkbox_widget = self.export_table.cellWidget(row, 3)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(is_checked)
                    checkbox.blockSignals(False)

        self.update_ui_state()

    def on_item_selection_changed(self):
        """Handle individual checkbox state change"""
        selected_count = self.get_selected_count()
        total_count = self.export_table.rowCount()

        self.select_all_checkbox.blockSignals(True)
        if selected_count == 0:
            self.select_all_checkbox.setCheckState(Qt.Unchecked)
        elif selected_count == total_count:
            self.select_all_checkbox.setCheckState(Qt.Checked)
        else:
            self.select_all_checkbox.setCheckState(Qt.PartiallyChecked)
        self.select_all_checkbox.blockSignals(False)

        self.update_ui_state()

    def get_selected_count(self):
        """Get count of selected checkboxes"""
        count = 0
        for row in range(self.export_table.rowCount()):
            checkbox_widget = self.export_table.cellWidget(row, 3)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    count += 1
        return count

    def update_ui_state(self):
        """Update UI state (buttons and status label)"""
        selected_count = self.get_selected_count()
        has_selection = selected_count > 0

        self.export_button.setEnabled(has_selection)

        if selected_count == 0:
            status_text = self.tr("No fields selected")
        elif selected_count == 1:
            status_text = self.tr("1 field selected")
        else:
            status_text = self.tr("%d fields selected") % selected_count

        self.status_label.setText(status_text)

    def get_export_config(self):
        """Get export configuration based on user selections"""
        config = {}

        for row in range(self.export_table.rowCount()):
            checkbox_widget = self.export_table.cellWidget(row, 3)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    original_key = self.export_table.item(row, 1).text()
                    export_input = self.export_table.cellWidget(row, 2)
                    export_key = (
                        export_input.text().strip()
                        if export_input
                        else original_key
                    )

                    if export_key:
                        config[original_key] = export_key

        return config
