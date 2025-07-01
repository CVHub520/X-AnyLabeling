from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QCheckBox,
    QHeaderView,
    QWidget,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from anylabeling.views.labeling.vqa.config import (
    DEFAULT_COMPONENT_WINDOW_SIZE,
    SUPPORTED_WIDGETS,
)
from anylabeling.views.labeling.vqa.style import (
    get_cancel_button_style,
    get_danger_button_style,
    get_filename_label_style,
    get_page_input_style,
    get_primary_button_style,
    get_status_label_style,
    get_component_dialog_combobox_style,
)


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
        self.ok_button = QPushButton(self.tr("OK"))
        self.ok_button.setStyleSheet(get_primary_button_style())
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(get_cancel_button_style())

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
        self.delete_button.setStyleSheet(get_danger_button_style())
        self.delete_button.setEnabled(False)

        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(get_cancel_button_style())

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
