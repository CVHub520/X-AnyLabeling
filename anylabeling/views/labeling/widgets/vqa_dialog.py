import os
import json

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QShortcut,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import (
    QIcon,
    QIntValidator,
    QKeySequence,
    QPixmap,
)

from anylabeling.views.labeling.vqa import *
from anylabeling.views.labeling.utils.qt import new_icon
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.vqa.dialogs import ExportLabelsDialog
from anylabeling.views.labeling.widgets.popup import Popup


class VQADialog(QDialog):

    def __init__(self, parent=None):
        """
        Initialize the VQA dialog with default settings and UI components.

        Args:
            parent: Parent widget, defaults to None
        """
        super().__init__(parent)
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setModal(False)
        self.resize(*DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(PANEL_SIZE, PANEL_SIZE)

        self.is_enlarged = False

        self.image_files = []
        self.custom_components = []
        self.current_image_index = 0
        self.switching_image = False

        self.init_ui()
        self.setup_shortcuts()
        self.load_config()
        self.load_initial_image_data()

    def init_ui(self):
        """
        Initialize and setup the user interface layout and components.
        Creates the main layout with left panel (image display) and right panel (controls).
        """
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(20)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(3)
        self.main_splitter.setStyleSheet(get_main_splitter_style())

        ################################
        #          Left panel          #
        ################################
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.setSpacing(10)

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        self.filename_label = QLabel(self.tr("No image loaded"))
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setFixedHeight(DEFAULT_COMPONENT_HEIGHT)
        self.filename_label.setStyleSheet(get_filename_label_style())

        self.refresh_button = QPushButton()
        self.refresh_button.setIcon(QIcon(new_icon("refresh", "svg")))
        self.refresh_button.setFixedSize(*ICON_SIZE_NORMAL)
        self.refresh_button.setStyleSheet(get_button_style())
        self.refresh_button.setToolTip(self.tr("Refresh Data"))
        self.refresh_button.clicked.connect(self.refresh_data)

        self.toggle_panel_button = QPushButton()
        self.toggle_panel_button.setIcon(QIcon(new_icon("sidebar", "svg")))
        self.toggle_panel_button.setFixedSize(*ICON_SIZE_NORMAL)
        self.toggle_panel_button.setStyleSheet(get_button_style())
        self.toggle_panel_button.setToolTip(self.tr("Toggle Sidebar"))
        self.toggle_panel_button.clicked.connect(self.toggle_left_panel)

        header_layout.addWidget(self.filename_label, 1)
        header_layout.addWidget(self.refresh_button)
        header_layout.addWidget(self.toggle_panel_button)

        left_layout.addWidget(header_widget)

        self.image_container = QWidget()
        self.image_container.setStyleSheet(get_image_container_style())
        container_layout = QVBoxLayout(self.image_container)
        container_layout.setContentsMargins(8, 8, 8, 8)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(get_image_label_style())
        container_layout.addWidget(self.image_label, 1)

        left_layout.addWidget(self.image_container, 1)

        ################################
        #          Right panel         #
        ################################
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        action_widget = QWidget()
        action_widget.setFixedHeight(DEFAULT_COMPONENT_HEIGHT)
        action_layout = QHBoxLayout(action_widget)
        action_layout.setContentsMargins(10, 0, 10, 0)
        action_layout.setSpacing(8)

        self.refresh_button_right = QPushButton()
        self.refresh_button_right.setIcon(QIcon(new_icon("refresh", "svg")))
        self.refresh_button_right.setFixedSize(*ICON_SIZE_NORMAL)
        self.refresh_button_right.setStyleSheet(get_button_style())
        self.refresh_button_right.setToolTip(self.tr("Refresh Data"))
        self.refresh_button_right.clicked.connect(self.refresh_data)
        self.refresh_button_right.setVisible(False)

        self.toggle_panel_button_right = QPushButton()
        self.toggle_panel_button_right.setIcon(
            QIcon(new_icon("sidebar", "svg"))
        )
        self.toggle_panel_button_right.setFixedSize(*ICON_SIZE_NORMAL)
        self.toggle_panel_button_right.setStyleSheet(get_button_style())
        self.toggle_panel_button_right.setToolTip(self.tr("Toggle Sidebar"))
        self.toggle_panel_button_right.clicked.connect(self.toggle_left_panel)
        self.toggle_panel_button_right.setVisible(False)

        self.export_button = QPushButton(self.tr("Export Labels"))
        self.export_button.setStyleSheet(
            get_dialog_button_style("success", "medium")
        )
        self.export_button.clicked.connect(self.export_labels)

        self.clear_button = QPushButton(self.tr("Clear All"))
        self.clear_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        self.clear_button.clicked.connect(self.clear_current)

        self.add_component_button = QPushButton(self.tr("Add Compo"))
        self.add_component_button.setStyleSheet(
            get_dialog_button_style("primary", "medium")
        )
        self.add_component_button.clicked.connect(self.add_custom_component)

        self.delete_component_button = QPushButton(self.tr("Del Compo"))
        self.delete_component_button.setStyleSheet(
            get_dialog_button_style("danger", "medium")
        )
        self.delete_component_button.clicked.connect(
            self.delete_custom_component
        )

        action_layout.addWidget(self.refresh_button_right)
        action_layout.addWidget(self.toggle_panel_button_right)
        action_layout.addWidget(self.export_button, 1)
        action_layout.addWidget(self.clear_button, 1)
        action_layout.addWidget(self.add_component_button, 1)
        action_layout.addWidget(self.delete_component_button, 1)

        right_layout.addWidget(action_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameStyle(QFrame.NoFrame)

        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll_layout.setSpacing(15)

        self.scroll_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_widget)
        right_layout.addWidget(self.scroll_area, 1)

        nav_widget = QWidget()
        nav_widget.setFixedHeight(DEFAULT_COMPONENT_HEIGHT)
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(10, 0, 10, 0)
        nav_layout.setSpacing(8)

        self.prev_button = QPushButton()
        self.prev_button.setIcon(QIcon(new_icon("arrow-left", "svg")))
        self.prev_button.setFixedSize(*ICON_SIZE_NORMAL)
        self.prev_button.setStyleSheet(get_button_style())
        self.prev_button.clicked.connect(lambda _: self.switch_image("prev"))

        page_widget = QWidget()
        page_widget.setFixedWidth(82)
        page_layout = QHBoxLayout(page_widget)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(2)

        self.page_input = PageInputLineEdit()
        self.page_input.vqa_dialog = self
        self.page_input.setFixedSize(68, DEFAULT_COMPONENT_HEIGHT)
        self.page_input.setAlignment(Qt.AlignCenter)
        self.page_input.setStyleSheet(get_page_input_style())
        self.page_input.textChanged.connect(self.validate_page_input)
        self.page_input.editingFinished.connect(self.on_page_input_finished)

        page_layout.addWidget(self.page_input)

        self.next_button = QPushButton()
        self.next_button.setIcon(QIcon(new_icon("arrow-right", "svg")))
        self.next_button.setFixedSize(*ICON_SIZE_NORMAL)
        self.next_button.setStyleSheet(get_button_style())
        self.next_button.clicked.connect(lambda _: self.switch_image("next"))

        nav_layout.addWidget(self.prev_button)
        nav_layout.addStretch()
        nav_layout.addWidget(page_widget)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_button)

        right_layout.addWidget(nav_widget)

        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(right_widget)

        total_width = DEFAULT_WINDOW_SIZE[0] - 40
        left_width = total_width // 2
        right_width = total_width - left_width
        self.main_splitter.setSizes([left_width, right_width])
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(self.main_splitter)

        self.update_navigation_state()

    def setup_shortcuts(self):
        prev_shortcut = QShortcut(QKeySequence("A"), self)
        prev_shortcut.activated.connect(lambda: self.switch_image("prev"))

        next_shortcut = QShortcut(QKeySequence("D"), self)
        next_shortcut.activated.connect(lambda: self.switch_image("next"))

    def toggle_left_panel(self):
        sizes = self.main_splitter.sizes()
        if sizes[0] == 0:
            total_width = self.main_splitter.width()
            left_width = total_width // 2
            right_width = total_width - left_width
            self.main_splitter.setSizes([left_width, right_width])
            self.refresh_button_right.setVisible(False)
            self.toggle_panel_button_right.setVisible(False)
        else:
            total_width = self.main_splitter.width()
            self.main_splitter.setSizes([0, total_width])
            self.refresh_button_right.setVisible(True)
            self.toggle_panel_button_right.setVisible(True)

    def load_config(self):
        """
        Load component configuration from the config file.
        """
        if not os.path.exists(COMPONENTS_CONFIG_PATH):
            logger.info(
                f"Config file not found at {COMPONENTS_CONFIG_PATH}, creating new one"
            )
            self.save_config()
            return

        try:
            with open(COMPONENTS_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.load_components_from_config(config)
            logger.info(
                f"Successfully loaded config from {COMPONENTS_CONFIG_PATH}"
            )

        except (json.JSONDecodeError, KeyError):
            logger.error(
                f"Failed to parse config file {COMPONENTS_CONFIG_PATH}, creating new one"
            )
            os.remove(COMPONENTS_CONFIG_PATH)
            self.save_config()

    def save_config(self):
        """
        Save current component configuration to the config file.
        """
        config = dict(components=[])

        for comp in self.custom_components:
            component_data = dict(
                title=comp["title"], type=comp["type"], options=comp["options"]
            )
            config["components"].append(component_data)

        os.makedirs(os.path.dirname(COMPONENTS_CONFIG_PATH), exist_ok=True)

        with open(COMPONENTS_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def load_images_folder(self):
        """
        Load images from a selected folder and initialize navigation.
        Updates the image list and displays the first image.
        """
        self.parent().open_folder_dialog()
        if self.parent().image_list:
            self.image_files = self.parent().image_list
            self.current_image_index = 0
            self.update_image_display()
            self.update_navigation_state()
            self.load_current_image_data()
            self.page_input.setValidator(
                QIntValidator(1, len(self.image_files) - 1)
            )

    def switch_image(self, mode):
        """
        Switch between images in the collection.

        Args:
            mode (str): Switch mode - 'prev', 'next', or 'jump'
        """
        if not self.parent().image_list:
            return

        if mode == "jump":
            user_input = int(self.page_input.text())
            current_index = user_input - 1
            if current_index < 0 or current_index >= len(
                self.parent().image_list
            ):
                return
        elif mode == "prev":
            current_index = self.parent().image_list.index(
                self.parent().filename
            )
            if current_index <= 0:
                return
            current_index -= 1
        elif mode == "next":
            current_index = self.parent().image_list.index(
                self.parent().filename
            )
            if current_index >= len(self.parent().image_list) - 1:
                return
            current_index += 1

        try:
            self.switching_image = True

            original_sizes = self.main_splitter.sizes()
            was_collapsed = original_sizes[0] == 0

            self.save_current_image_data()
            new_file = self.parent().image_list[current_index]
            self.parent().load_file(new_file)
            self.update_image_display()
            self.update_navigation_state()
            self.clear_all_components_silent()
            self.load_current_image_data()

            def restore_panel_state():
                if was_collapsed:
                    self.main_splitter.setSizes([0, PANEL_SIZE])
                    self.toggle_panel_button_right.setVisible(True)
                else:
                    self.main_splitter.setSizes([PANEL_SIZE, PANEL_SIZE])
                    self.toggle_panel_button_right.setVisible(False)

            QTimer.singleShot(10, restore_panel_state)

            self.switching_image = False
        except (ValueError, AttributeError):
            self.switching_image = False

    def update_image_display(self):
        """
        Update the image display with current image and navigation info.
        """
        if self.parent().filename:
            filename = os.path.basename(self.parent().filename)

            if self.parent().image_list:
                try:
                    current_index = self.parent().image_list.index(
                        self.parent().filename
                    )
                    total_count = len(self.parent().image_list)
                    self.filename_label.setText(
                        f"{filename} ({current_index + 1}/{total_count})"
                    )
                    self.page_input.setText(str(current_index + 1))
                except ValueError:
                    self.filename_label.setText(f"{filename} (1/1)")
                    self.page_input.setText("1")
            else:
                self.filename_label.setText(filename)
                self.page_input.setText("1")

            pixmap = QPixmap(self.parent().filename)
            if not pixmap.isNull():
                available_width = self.image_label.width() - 10
                available_height = self.image_label.height() - 10

                if available_width > 0 and available_height > 0:
                    scaled_pixmap = pixmap.scaled(
                        available_width,
                        available_height,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    self.image_label.setPixmap(scaled_pixmap)
        else:
            self.filename_label.setText(self.tr("No image loaded"))
            self.image_label.clear()
            self.page_input.setText("0")

    def save_current_image_data(self):
        """
        Save annotation data for the currently displayed image.
        """
        if getattr(self, "switching_image", False):
            return

        if not hasattr(self.parent(), "other_data"):
            self.parent().other_data = {}

        if "vqaData" not in self.parent().other_data:
            self.parent().other_data["vqaData"] = {}

        vqa_data = self.parent().other_data["vqaData"]

        for comp in self.custom_components:
            title = comp["title"]
            widget = comp["widget"]
            comp_type = comp["type"]
            value = self.get_component_value(widget, comp_type)
            vqa_data[title] = value

        if self.parent().filename:
            self.parent().set_dirty()

    def load_current_image_data(self):
        """
        Load annotation data for the currently displayed image.
        """
        if (
            not hasattr(self.parent(), "other_data")
            or not self.parent().other_data
        ):
            self.set_default_values()
            self.adjust_all_text_widgets_height()
            return

        vqa_data = self.parent().other_data.get("vqaData", {})

        for comp in self.custom_components:
            title = comp["title"]
            widget = comp["widget"]
            comp_type = comp["type"]

            if title in vqa_data:
                value = vqa_data[title]
                self.set_component_value(widget, comp_type, value)
            else:
                self.set_component_default_value(
                    widget, comp_type, comp["options"]
                )

        self.adjust_all_text_widgets_height()

    def update_navigation_state(self):
        """
        Update navigation button states based on current position.
        """
        if self.parent().image_list:
            try:
                current_index = self.parent().image_list.index(
                    self.parent().filename
                )
                self.prev_button.setEnabled(current_index > 0)
                self.next_button.setEnabled(
                    current_index < len(self.parent().image_list) - 1
                )
                self.image_files = self.parent().image_list
            except (ValueError, AttributeError):
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(False)
        else:
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)

    def check_duplicate_title(self, title, original_title=None):
        """
        Check if a component title already exists to prevent duplicates.

        Args:
            title (str): Title to check for duplicates
            original_title (str, optional): Original title when editing

        Returns:
            bool: True if duplicate exists, False otherwise
        """
        for comp in self.custom_components:
            if comp["title"] == title and title != original_title:
                return True
        return False

    def load_components_from_config(self, config):
        """
        Load and create components from configuration data.

        Args:
            config (dict): Configuration dictionary containing component data
        """
        if "components" not in config:
            return

        for comp_data in config["components"]:
            component_data = {
                "title": comp_data["title"],
                "type": comp_data["type"],
                "options": comp_data.get("options", []),
            }
            self.create_component(component_data, from_config=True)

    def add_custom_component(self):
        """
        Open dialog to add a new custom component.
        """
        dialog = ComponentDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            component_data = dialog.get_component_data()
            self.create_component(component_data)

    def edit_custom_component_by_object(self, component_obj):
        """
        Edit a custom component using its object reference.

        Args:
            component_obj (dict): Component object to edit
        """
        try:
            index = self.custom_components.index(component_obj)
            self.edit_custom_component(index)
        except ValueError:
            return

    def edit_custom_component(self, index):
        """
        Edit a custom component at the specified index.

        Args:
            index (int): Index of the component to edit
        """
        if index < 0 or index >= len(self.custom_components):
            return

        self.save_current_image_data()

        component_data = self.custom_components[index]
        edit_data = {
            "title": component_data["title"],
            "type": component_data["type"],
            "options": component_data["options"],
        }

        dialog = ComponentDialog(self, edit_data)
        if dialog.exec_() == QDialog.Accepted:
            new_data = dialog.get_component_data()
            self.update_component(index, new_data)

    def update_component(self, index, new_data):
        """
        Update an existing component with new configuration data.

        Args:
            index (int): Index of the component to update
            new_data (dict): New component configuration data
        """
        if index < 0 or index >= len(self.custom_components):
            return

        component_data = self.custom_components[index]
        old_title = component_data["title"]
        new_title = new_data["title"]
        old_options = component_data["options"]
        new_options = new_data["options"]

        if old_title != new_title:
            self.update_labels_title(old_title, new_title)

        if (
            old_options != new_options
            and component_data["type"] != "QLineEdit"
        ):
            change_result = self.handle_options_change(
                new_title, old_options, new_options, component_data["type"]
            )
            if change_result == False:
                return

        component_data["title"] = new_title
        component_data["options"] = new_options
        component_data["title_label"].setText(f"{new_title}")

        if old_options != new_options:
            if component_data["type"] == "QComboBox":
                self.update_dropdown_options(
                    component_data, old_options, new_options
                )
            elif component_data["type"] in ["QRadioButton", "QCheckBox"]:
                self.rebuild_component_options(component_data, new_options)
                self.load_current_image_data()

        self.save_config()

    def handle_options_change(
        self, component_title, old_options, new_options, comp_type
    ):
        """
        Handle changes to component options and update related data.

        Args:
            component_title (str): Title of the component
            old_options (list): Previous option values
            new_options (list): New option values
            comp_type (str): Type of the component

        Returns:
            bool: True if changes should proceed, False to cancel
        """
        old_set = set(old_options)
        new_set = set(new_options)

        deleted_options = old_set - new_set
        added_options = new_set - old_set

        if deleted_options == set() and added_options != set():
            return True

        modified_options = get_real_modified_options(
            old_options, new_options, old_set & new_set
        )

        if modified_options:
            self.update_modified_options(component_title, modified_options)

        if deleted_options:
            template = self.tr(
                "Removing options %s will reset related annotation data to default values.\n"
                "Do you want to continue?"
            )
            msg_text = template % list(deleted_options)
            reply = QMessageBox.question(
                self,
                self.tr("Confirm Option Delete"),
                msg_text,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return False

            self.reset_deleted_options_to_default(
                component_title, deleted_options, comp_type, new_options
            )

        return True

    def reset_deleted_options_to_default(
        self, component_title, deleted_options, comp_type, new_options
    ):
        """
        Reset values to default when options are deleted from a component.

        Args:
            component_title (str): Title of the component
            deleted_options (set): Set of deleted option values
            comp_type (str): Type of the component
            new_options (list): Remaining option values
        """
        default_value = (
            None
            if comp_type == "QComboBox"
            else get_default_value(comp_type, new_options)
        )

        if hasattr(self.parent(), "other_data") and self.parent().other_data:
            vqa_data = self.parent().other_data.get("vqaData", {})
            if component_title in vqa_data:
                current_value = vqa_data[component_title]
                if value_contains_deleted_options(
                    current_value, deleted_options
                ):
                    vqa_data[component_title] = default_value
                    self.parent().set_dirty()

        self.update_all_labels_for_deleted_options(
            component_title, deleted_options, default_value
        )

    def update_all_labels_for_deleted_options(
        self, component_title, deleted_options, default_value
    ):
        """
        Update all label files when component options are deleted.

        Args:
            component_title (str): Title of the component
            deleted_options (set): Set of deleted option values
            default_value: Default value to set for affected entries
        """
        self.apply_to_all_label_files(
            lambda label_file: self.reset_deleted_options_in_json(
                label_file, component_title, deleted_options, default_value
            )
        )

    def reset_deleted_options_in_json(
        self, label_file, component_title, deleted_options, default_value
    ):
        """
        Reset deleted options to default values in a specific JSON file.

        Args:
            label_file (str): Path to the JSON label file
            component_title (str): Title of the component
            deleted_options (set): Set of deleted option values
            default_value: Default value to set
        """
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "vqaData" in data and component_title in data["vqaData"]:
                current_value = data["vqaData"][component_title]
                if value_contains_deleted_options(
                    current_value, deleted_options
                ):
                    data["vqaData"][component_title] = default_value
                    with open(label_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            pass

    def update_modified_options(self, component_title, modified_mapping):
        """
        Update current image data when component options are modified.

        Args:
            component_title (str): Title of the component
            modified_mapping (dict): Mapping of old to new option values
        """
        if hasattr(self.parent(), "other_data") and self.parent().other_data:
            vqa_data = self.parent().other_data.get("vqaData", {})
            if component_title in vqa_data:
                current_value = vqa_data[component_title]
                updated_value = apply_option_mapping(
                    current_value, modified_mapping
                )
                if updated_value != current_value:
                    vqa_data[component_title] = updated_value
                    self.parent().set_dirty()

        self.update_all_labels_for_modified_options(
            component_title, modified_mapping
        )

    def update_all_labels_for_modified_options(
        self, component_title, modified_mapping
    ):
        """
        Update all label files when component options are modified.

        Args:
            component_title (str): Title of the component
            modified_mapping (dict): Mapping of old to new option values
        """
        self.apply_to_all_label_files(
            lambda label_file: self.update_modified_options_in_json(
                label_file, component_title, modified_mapping
            )
        )

    def update_modified_options_in_json(
        self, label_file, component_title, modified_mapping
    ):
        """
        Update modified options in a specific JSON file.

        Args:
            label_file (str): Path to the JSON label file
            component_title (str): Title of the component
            modified_mapping (dict): Mapping of old to new option values
        """
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "vqaData" in data and component_title in data["vqaData"]:
                current_value = data["vqaData"][component_title]
                updated_value = apply_option_mapping(
                    current_value, modified_mapping
                )
                if updated_value != current_value:
                    data["vqaData"][component_title] = updated_value
                    with open(label_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            pass

    def rebuild_component_options(self, component_data, new_options):
        """
        Rebuild component UI elements when options change.

        Args:
            component_data (dict): Component data dictionary
            new_options (list): New list of option values
        """
        widget = component_data["widget"]
        comp_type = component_data["type"]

        if comp_type == "QRadioButton":
            layout = widget.layout()
            layout.clear()

            button_group = QButtonGroup(widget)
            for option in new_options:
                if option.strip():
                    radio = create_truncated_widget(
                        option.strip(), QRadioButton
                    )
                    radio.toggled.connect(self.save_current_image_data)
                    button_group.addButton(radio)
                    layout.addWidget(radio)

            widget.button_group = button_group
            widget.setContentsMargins(0, 0, 0, 4)

        elif comp_type == "QComboBox":
            widget.clear()
            widget.addItem("-- Select --")
            for option in new_options:
                if option.strip():
                    widget.addItem(option.strip())
            widget.currentTextChanged.connect(self.save_current_image_data)

        elif comp_type == "QCheckBox":
            layout = widget.layout()
            layout.clear()

            for option in new_options:
                if option.strip():
                    checkbox = create_truncated_widget(
                        option.strip(), QCheckBox
                    )
                    checkbox.toggled.connect(self.save_current_image_data)
                    layout.addWidget(checkbox)

            widget.setContentsMargins(0, 0, 0, 4)

    def delete_custom_component(self):
        """
        Open dialog to select and delete custom components.
        """
        if not self.custom_components:
            QMessageBox.information(
                self,
                self.tr("Info"),
                self.tr("No custom components to delete!"),
            )
            return

        dialog = DeleteComponentDialog(self.custom_components, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_indices = dialog.get_selected_indices()

            if not selected_indices:
                return

            component_titles = [
                self.custom_components[i]["title"] for i in selected_indices
            ]
            if len(selected_indices) == 1:
                template = self.tr(
                    "Deleting component '%s' will remove all related annotation data from the current task.\n"
                    "Do you want to continue?"
                )
                msg_text = template % component_titles[0]
            else:
                template = self.tr(
                    "Deleting %d components (%s) will remove all related annotation data from the current task.\n"
                    "Do you want to continue?"
                )
                titles_str = ", ".join(
                    f"'{title}'" for title in component_titles
                )
                msg_text = template % (len(selected_indices), titles_str)

            reply = QMessageBox.question(
                self,
                self.tr("Confirm Delete"),
                msg_text,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                for index in selected_indices:
                    component_title = self.custom_components[index]["title"]
                    self.remove_component(index)
                    self.clean_all_labels_for_component(component_title)

    def remove_component(self, index):
        """
        Remove a component from the UI and component list.

        Args:
            index (int): Index of the component to remove
        """
        if index < 0 or index >= len(self.custom_components):
            return

        component_data = self.custom_components[index]
        widget = component_data["widget"]
        title_widget = component_data["title_widget"]

        self.scroll_layout.removeWidget(title_widget)
        self.scroll_layout.removeWidget(widget)
        title_widget.deleteLater()
        widget.deleteLater()

        self.custom_components.pop(index)
        self.save_config()

    def open_ai_assistant(self, component_obj):
        """
        Open AI assistant dialog for QLineEdit component.

        Args:
            component_obj (dict): Component object containing widget and metadata
        """
        widget = component_obj["widget"]
        current_text = widget.toPlainText().strip()

        dialog = AIPromptDialog(self, current_text)
        if dialog.exec_() == QDialog.Accepted:
            prompt = dialog.get_prompt()
            if prompt:
                self.loading_msg = AILoadingDialog(self)

                current_image_path = None
                if (
                    hasattr(self.parent(), "filename")
                    and self.parent().filename
                ):
                    current_image_path = self.parent().filename

                self.ai_worker = AIWorkerThread(
                    prompt,
                    current_text,
                    {},
                    current_image_path,
                    self.custom_components,
                    self.parent(),
                )
                self.ai_worker.finished.connect(
                    lambda result, success, error: self.handle_ai_result(
                        result, success, error, widget
                    )
                )

                self.loading_msg.cancel_button.clicked.connect(
                    self.cancel_ai_processing
                )
                self.ai_worker.start()
                if self.loading_msg.exec_() == QDialog.Rejected:
                    self.cancel_ai_processing()

    def cancel_ai_processing(self):
        """Cancel the AI processing"""
        if hasattr(self, "ai_worker") and self.ai_worker.isRunning():
            self.ai_worker.terminate()
            self.ai_worker.wait(1000)
        if hasattr(self, "loading_msg"):
            self.loading_msg.close()

    def handle_ai_result(self, result, success, error_message, widget):
        """
        Handle AI API result.

        Args:
            result (str): Generated text result
            success (bool): Whether the API call was successful
            error_message (str): Error message if failed
            widget: Target text widget
        """
        if hasattr(self, "loading_msg"):
            self.loading_msg.close()

        if success:
            dialog = QDialog(self)
            dialog.setWindowTitle(self.tr("AI Generated Result"))
            dialog.setModal(True)
            dialog.setWindowFlags(
                dialog.windowFlags() | Qt.WindowStaysOnTopHint
            )
            dialog.resize(500, 400)

            layout = QVBoxLayout(dialog)
            layout.setSpacing(10)

            text_edit = QTextEdit()
            text_edit.setPlainText(result)
            text_edit.setReadOnly(True)
            layout.addWidget(text_edit)

            button_layout = QHBoxLayout()
            button_layout.addStretch()

            apply_button = QPushButton(self.tr("Apply"))
            apply_button.setStyleSheet(
                get_dialog_button_style("primary", "medium")
            )
            cancel_button = QPushButton(self.tr("Cancel"))
            cancel_button.setStyleSheet(
                get_dialog_button_style("secondary", "medium")
            )

            button_layout.addWidget(apply_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout)

            apply_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            apply_button.setDefault(True)

            reply = dialog.exec_()

            if reply == QDialog.Accepted:
                widget.blockSignals(True)
                widget.setPlainText(result)
                widget.blockSignals(False)
                if hasattr(widget, "adjust_height"):
                    widget.adjust_height()
                self.save_current_image_data()
        else:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Failed to generate content:\n") + error_message,
            )

    def create_component(self, component_data, from_config=False):
        """
        Create and add a new component to the UI.

        Args:
            component_data (dict): Component configuration data
            from_config (bool): Whether component is being loaded from config
        """
        title = component_data["title"]
        comp_type = component_data["type"]
        options = component_data["options"]

        item_count = self.scroll_layout.count()
        if item_count > 0:
            stretch_item = self.scroll_layout.itemAt(item_count - 1)
            if stretch_item.spacerItem():
                self.scroll_layout.removeItem(stretch_item)

        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel(f"{title}")
        title_label.setStyleSheet(get_filename_label_style())
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        if comp_type == "QLineEdit":
            ai_button = QPushButton()
            ai_button.setIcon(QIcon(new_icon("wand", "svg")))
            ai_button.setFixedSize(*ICON_SIZE_SMALL)
            ai_button.setStyleSheet(get_button_style())
            ai_button.setToolTip(self.tr("AI Assistant"))
            title_layout.addWidget(ai_button)

        edit_button = QPushButton()
        edit_button.setIcon(QIcon(new_icon("edit", "svg")))
        edit_button.setFixedSize(*ICON_SIZE_SMALL)
        edit_button.setStyleSheet(get_button_style())
        edit_button.setToolTip(self.tr("Edit Content"))
        title_layout.addWidget(edit_button)

        self.scroll_layout.addWidget(title_widget)

        if comp_type == "QLineEdit":
            widget = AutoResizeTextEdit()
            widget.textChanged.connect(self.save_current_image_data)
            self.scroll_layout.addWidget(widget)

        elif comp_type == "QRadioButton":
            widget = QWidget()
            widget.setContentsMargins(0, 0, 0, 4)
            layout = FlowLayout(widget)
            layout.setSpacing(10)
            button_group = QButtonGroup(widget)

            for i, option in enumerate(options):
                if option.strip():
                    radio = create_truncated_widget(
                        option.strip(), QRadioButton
                    )
                    button_group.addButton(radio)
                    layout.addWidget(radio)

            widget.button_group = button_group
            self.scroll_layout.addWidget(widget)

            if from_config:
                widget.blockSignals(True)
                for item in layout.itemList:
                    radio_widget = item.widget()
                    if radio_widget:
                        radio_widget.toggled.connect(
                            self.save_current_image_data
                        )
                widget.blockSignals(False)
            else:
                for item in layout.itemList:
                    radio_widget = item.widget()
                    if radio_widget:
                        radio_widget.toggled.connect(
                            self.save_current_image_data
                        )
                widget.blockSignals(True)
                if layout.itemList:
                    first_radio = layout.itemList[0].widget()
                    if first_radio:
                        first_radio.setChecked(True)
                widget.blockSignals(False)

        elif comp_type == "QComboBox":
            widget = QComboBox()
            widget.addItem("-- Select --")
            for option in options:
                if option.strip():
                    widget.addItem(option.strip())
            widget.currentTextChanged.connect(self.save_current_image_data)
            self.scroll_layout.addWidget(widget)

        elif comp_type == "QCheckBox":
            widget = QWidget()
            widget.setContentsMargins(0, 0, 0, 4)
            layout = FlowLayout(widget)
            layout.setSpacing(10)

            for option in options:
                if option.strip():
                    checkbox = create_truncated_widget(
                        option.strip(), QCheckBox
                    )
                    checkbox.toggled.connect(self.save_current_image_data)
                    layout.addWidget(checkbox)

            self.scroll_layout.addWidget(widget)

        self.scroll_layout.addStretch()

        component_obj = {
            "title": title,
            "type": comp_type,
            "options": options,
            "widget": widget,
            "title_widget": title_widget,
            "title_label": title_label,
        }

        edit_button.clicked.connect(
            lambda: self.edit_custom_component_by_object(component_obj)
        )

        if comp_type == "QLineEdit":
            ai_button.clicked.connect(
                lambda: self.open_ai_assistant(component_obj)
            )

        self.custom_components.append(component_obj)

        if not from_config:
            self.save_config()
            self.add_component_to_all_labels(component_data)

    def add_component_to_all_labels(self, component_data):
        """
        Add a new component to all existing label files.

        Args:
            component_data (dict): Component configuration data
        """
        title = component_data["title"]
        comp_type = component_data["type"]
        options = component_data["options"]

        default_value = None
        if comp_type == "QRadioButton" and options:
            default_value = options[0]
        elif comp_type == "QCheckBox":
            default_value = []
        elif comp_type == "QComboBox":
            default_value = None

        self.apply_to_all_label_files(
            lambda label_file: self.add_component_to_json_file(
                label_file, title, default_value
            )
        )

    def add_component_to_json_file(
        self, label_file, component_title, default_value
    ):
        """
        Add a component with default value to a specific JSON file.

        Args:
            label_file (str): Path to the JSON label file
            component_title (str): Title of the component
            default_value: Default value for the component
        """
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "vqaData" not in data:
                data["vqaData"] = {}

            if component_title not in data["vqaData"]:
                data["vqaData"][component_title] = default_value

                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            pass

    def clear_all_components(self):
        """
        Clear all component values in the UI.
        """
        for component_data in self.custom_components:
            widget = component_data["widget"]
            comp_type = component_data["type"]

            if comp_type == "QLineEdit":
                widget.clear()
            elif comp_type == "QRadioButton":
                layout = widget.layout()
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item and item.widget():
                        item.widget().setChecked(False)
            elif comp_type in ["QCheckBox"]:
                layout = widget.layout()
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item and item.widget():
                        item.widget().setChecked(False)
            elif comp_type == "QComboBox":
                widget.setCurrentIndex(0)

    def clear_all_components_silent(self):
        """
        Clear all component values without triggering change signals.
        """
        for component_data in self.custom_components:
            widget = component_data["widget"]
            widget.blockSignals(True)

        self.clear_all_components()

        for component_data in self.custom_components:
            widget = component_data["widget"]
            widget.blockSignals(False)

    def get_component_value(self, widget, comp_type):
        """
        Get the current value from a component based on its type.

        Args:
            widget: The component widget
            comp_type (str): Type of the component

        Returns:
            Current value of the component
        """
        if comp_type == "QLineEdit":
            text = widget.toPlainText().strip()
            return text
        elif comp_type == "QRadioButton":
            layout = widget.layout()
            for item in layout.itemList:
                radio_widget = item.widget()
                if radio_widget and radio_widget.isChecked():
                    return radio_widget.toolTip() or radio_widget.text()
            return None
        elif comp_type == "QComboBox":
            current_text = widget.currentText()
            return current_text if current_text != "-- Select --" else None
        elif comp_type == "QCheckBox":
            selected = []
            layout = widget.layout()
            for item in layout.itemList:
                checkbox_widget = item.widget()
                if checkbox_widget and checkbox_widget.isChecked():
                    selected.append(
                        checkbox_widget.toolTip() or checkbox_widget.text()
                    )
            return selected
        return None

    def set_component_value(self, widget, comp_type, value):
        """
        Set a value to a component based on its type.

        Args:
            widget: The component widget
            comp_type (str): Type of the component
            value: Value to set
        """
        if comp_type == "QLineEdit":
            widget.blockSignals(True)
            widget.setPlainText(str(value) if value else "")
            widget.blockSignals(False)
            if hasattr(widget, "adjust_height"):
                widget.adjust_height()

        elif comp_type == "QRadioButton":
            layout = widget.layout()
            for item in layout.itemList:
                radio_widget = item.widget()
                if radio_widget:
                    radio_widget.blockSignals(True)

            for item in layout.itemList:
                radio_widget = item.widget()
                if radio_widget:
                    original_text = (
                        radio_widget.toolTip() or radio_widget.text()
                    )
                    is_checked = original_text == value
                    radio_widget.setChecked(is_checked)

            for item in layout.itemList:
                radio_widget = item.widget()
                if radio_widget:
                    radio_widget.blockSignals(False)

        elif comp_type == "QComboBox":
            widget.blockSignals(True)
            if value:
                index = widget.findText(str(value))
                widget.setCurrentIndex(index if index >= 0 else 0)
            else:
                widget.setCurrentIndex(0)
            widget.blockSignals(False)

        elif comp_type == "QCheckBox":
            layout = widget.layout()
            for item in layout.itemList:
                checkbox_widget = item.widget()
                if checkbox_widget:
                    checkbox_widget.blockSignals(True)

            if isinstance(value, list):
                for item in layout.itemList:
                    checkbox_widget = item.widget()
                    if checkbox_widget:
                        original_text = (
                            checkbox_widget.toolTip() or checkbox_widget.text()
                        )
                        checkbox_widget.setChecked(original_text in value)

            for item in layout.itemList:
                checkbox_widget = item.widget()
                if checkbox_widget:
                    checkbox_widget.blockSignals(False)

    def adjust_all_text_widgets_height(self):
        """
        Trigger height adjustment for all AutoResizeTextEdit widgets.
        """
        for comp in self.custom_components:
            if comp["type"] == "QLineEdit":
                widget = comp["widget"]
                if hasattr(widget, "adjust_height"):
                    widget.adjust_height()

    def export_labels(self):
        """
        Export all annotation data to a JSONL file.
        """
        if not self.image_files:
            QMessageBox.warning(
                self, self.tr("Warning"), self.tr("No images loaded!")
            )
            return

        export_dialog = ExportLabelsDialog(self.custom_components, self)
        if export_dialog.exec_() != QDialog.Accepted:
            return

        export_config = export_dialog.get_export_config()
        if not export_config:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("No fields selected for export!"),
            )
            return

        export_path = QFileDialog.getSaveFileName(
            self,
            self.tr("Export Labels"),
            self.tr("vqa_labels.jsonl"),
            "JSONL files (*.jsonl)",
        )[0]
        if not export_path:
            return

        self.save_current_image_data()

        with open(export_path, "w", encoding="utf-8") as f:
            for i, image_file in enumerate(self.image_files):
                # Basic image info
                pixmap = QPixmap(image_file)
                width = pixmap.width() if not pixmap.isNull() else 0
                height = pixmap.height() if not pixmap.isNull() else 0

                # Build data dict with all available fields
                all_data = {
                    "image": os.path.basename(image_file),
                    "width": width,
                    "height": height,
                }

                label_file_path = get_label_file_path(
                    image_file, self.parent().output_dir
                )
                if os.path.exists(label_file_path):
                    try:
                        with open(
                            label_file_path, "r", encoding="utf-8"
                        ) as label_f:
                            label_data_json = json.load(label_f)
                            vqa_data = label_data_json.get("vqaData", {})
                            all_data.update(vqa_data)
                            all_data["shapes"] = label_data_json.get(
                                "shapes", []
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load label file {label_file_path}: {e}"
                        )

                for component in self.custom_components:
                    comp_title = component["title"]
                    if comp_title not in all_data:
                        comp_type = component["type"]
                        if comp_type == "QLineEdit":
                            all_data[comp_title] = ""
                        elif comp_type == "QRadioButton":
                            all_data[comp_title] = None
                        elif comp_type == "QComboBox":
                            all_data[comp_title] = None
                        elif comp_type == "QCheckBox":
                            all_data[comp_title] = []

                if "shapes" not in all_data:
                    all_data["shapes"] = []

                # Filter and rename fields based on export config
                label_data = {}
                for original_key, export_key in export_config.items():
                    if original_key in all_data:
                        label_data[export_key] = all_data[original_key]

                f.write(json.dumps(label_data, ensure_ascii=False) + "\n")

        template = self.tr("Labels exported to %s.")
        msg_text = template % export_path
        QMessageBox.information(self, self.tr("Success"), msg_text)

    def clear_current(self):
        """
        Clear all annotations for the current image.
        """
        reply = QMessageBox.question(
            self,
            self.tr("Confirm Clear"),
            self.tr("Are you sure you want to clear all current annotations?"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        for component_data in self.custom_components:
            widget = component_data["widget"]
            comp_type = component_data["type"]

            if comp_type == "QLineEdit":
                widget.clear()
            elif comp_type in ["QRadioButton", "QCheckBox"]:
                layout = widget.layout()
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item and item.widget():
                        item.widget().setChecked(False)
            elif comp_type == "QComboBox":
                widget.setCurrentIndex(0)

        self.save_current_image_data()

    def clean_all_labels_for_component(self, component_title):
        """
        Remove component data from all label files when component is deleted.

        Args:
            component_title (str): Title of the component to remove
        """
        self.apply_to_all_label_files(
            lambda label_file: self.remove_component_from_json_file(
                label_file, component_title
            )
        )
        self.load_current_image_data()

    def set_default_values(self):
        """
        Set default values for all components.
        """
        for comp in self.custom_components:
            widget = comp["widget"]
            comp_type = comp["type"]
            options = comp["options"]
            self.set_component_default_value(widget, comp_type, options)

    def set_component_default_value(self, widget, comp_type, options):
        """
        Set default value for a specific component based on its type.

        Args:
            widget: The component widget
            comp_type (str): Type of the component
            options (list): Available options for the component
        """
        widget.blockSignals(True)

        if comp_type == "QLineEdit":
            widget.clear()
        elif comp_type == "QRadioButton":
            if options:
                target_option = options[0]
                for i in range(widget.layout().count()):
                    item = widget.layout().itemAt(i)
                    if item and item.widget():
                        item.widget().setChecked(
                            item.widget().text() == target_option
                        )
        elif comp_type == "QComboBox":
            widget.setCurrentIndex(0)
        elif comp_type == "QCheckBox":
            for i in range(widget.layout().count()):
                item = widget.layout().itemAt(i)
                if item and item.widget():
                    item.widget().setChecked(False)

        widget.blockSignals(False)

    def update_labels_title(self, old_title, new_title):
        """
        Update component title in both current and all label files.

        Args:
            old_title (str): Previous component title
            new_title (str): New component title
        """
        self.update_current_labels_title(old_title, new_title)
        self.update_all_json_labels_title(old_title, new_title)

    def update_current_labels_title(self, old_title, new_title):
        """
        Update component title in the current image's label data.

        Args:
            old_title (str): Previous component title
            new_title (str): New component title
        """
        if hasattr(self.parent(), "other_data") and self.parent().other_data:
            vqa_data = self.parent().other_data.get("vqaData", {})
            if old_title in vqa_data:
                vqa_data[new_title] = vqa_data.pop(old_title)
                self.parent().set_dirty()

    def update_all_json_labels_title(self, old_title, new_title):
        """
        Update component title in all JSON label files.

        Args:
            old_title (str): Previous component title
            new_title (str): New component title
        """
        self.apply_to_all_label_files(
            lambda label_file: self.update_title_in_json_file(
                label_file, old_title, new_title
            )
        )

    def update_title_in_json_file(self, label_file, old_title, new_title):
        """
        Update component title in a specific JSON file.

        Args:
            label_file (str): Path to the JSON label file
            old_title (str): Previous component title
            new_title (str): New component title
        """
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "vqaData" in data and old_title in data["vqaData"]:
                data["vqaData"][new_title] = data["vqaData"].pop(old_title)
                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            pass

    def remove_component_from_json_file(self, label_file, component_title):
        """
        Remove component data from a specific JSON file.

        Args:
            label_file (str): Path to the JSON label file
            component_title (str): Title of the component to remove
        """
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "vqaData" in data and component_title in data["vqaData"]:
                del data["vqaData"][component_title]
                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            pass

    def apply_to_all_label_files(self, callback):
        """
        Apply a callback function to all label files in the project.

        Args:
            callback (function): Function to apply to each label file
        """
        if self.parent().output_dir:
            label_dir = self.parent().output_dir
            for file_name in os.listdir(label_dir):
                if file_name.endswith(".json"):
                    label_file = os.path.join(label_dir, file_name)
                    callback(label_file)
        elif self.parent().image_list:
            for image_file in self.parent().image_list:
                label_file = os.path.splitext(image_file)[0] + ".json"
                if os.path.exists(label_file):
                    callback(label_file)

    def update_dropdown_options(
        self, component_data, old_options, new_options
    ):
        """
        Update dropdown component options while preserving current selection.

        Args:
            component_data (dict): Component data dictionary
            old_options (list): Previous option values
            new_options (list): New option values
        """
        widget = component_data["widget"]
        current_selection = widget.currentText()

        old_set = set(old_options)
        new_set = set(new_options)
        deleted_options = old_set - new_set

        if current_selection in deleted_options:
            current_selection = None

        widget.blockSignals(True)
        widget.clear()
        widget.addItem("-- Select --")
        for option in new_options:
            if option.strip():
                widget.addItem(option.strip())

        if current_selection and current_selection != "-- Select --":
            index = widget.findText(current_selection)
            widget.setCurrentIndex(index if index >= 0 else 0)

        widget.blockSignals(False)
        widget.currentTextChanged.connect(self.save_current_image_data)

    def validate_page_input(self, text):
        """Real-time validation of page input"""
        if not text:
            return

        try:
            if not text.isdigit():
                cursor_pos = self.page_input.cursorPosition()
                clean_text = "".join(c for c in text if c.isdigit())
                self.page_input.setText(clean_text)
                self.page_input.setCursorPosition(
                    min(cursor_pos, len(clean_text))
                )
                return

            page_num = int(text)
            max_pages = (
                len(self.parent().image_list)
                if self.parent().image_list
                else 1
            )

            if page_num > max_pages:
                self.page_input.setText(str(max_pages))
                self.page_input.setCursorPosition(len(str(max_pages)))

        except ValueError:
            pass

    def on_page_input_finished(self):
        """Handle when user finishes editing page input"""
        text = self.page_input.text().strip()

        if not text:
            self.restore_current_page_number()
            return

        try:
            page_num = int(text)
            max_pages = (
                len(self.parent().image_list)
                if self.parent().image_list
                else 1
            )

            if page_num < 1:
                self.page_input.setText("1")
            elif page_num > max_pages:
                self.page_input.setText(str(max_pages))

        except ValueError:
            self.restore_current_page_number()

    def restore_current_page_number(self):
        """Restore the current page number in the input"""
        if self.parent().filename and self.parent().image_list:
            try:
                current_index = self.parent().image_list.index(
                    self.parent().filename
                )
                self.page_input.setText(str(current_index + 1))
            except (ValueError, AttributeError):
                self.page_input.setText("1")
        else:
            self.page_input.setText("1")

    def closeEvent(self, event):
        """
        Handle dialog close event and save configuration.

        Args:
            event: Close event object
        """
        self.save_config()
        self.hide()
        event.ignore()

    def resizeEvent(self, event):
        """
        Handle window resize event and update image display.

        Args:
            event: Resize event object
        """
        super().resizeEvent(event)

        current_width = event.size().width()
        current_height = event.size().height()
        default_width, default_height = DEFAULT_WINDOW_SIZE

        if current_width > default_width or current_height > default_height:
            self.is_enlarged = True
        elif (
            current_width <= default_width and current_height <= default_height
        ):
            if self.is_enlarged:
                self.resize(default_width, default_height)
            self.is_enlarged = False

        self.update_image_display()

    def load_initial_image_data(self):
        """
        Load initial image data if parent already has an image loaded.
        This fixes the issue where labels don't show when dialog is first opened.
        """
        if (
            hasattr(self.parent(), "filename")
            and self.parent().filename
            and hasattr(self.parent(), "image_list")
            and self.parent().image_list
        ):
            self.image_files = self.parent().image_list
            self.update_image_display()
            self.update_navigation_state()
            self.load_current_image_data()
            self.page_input.setValidator(
                QIntValidator(1, len(self.image_files))
            )
            QTimer.singleShot(100, self.adjust_all_text_widgets_height)

    def refresh_data(self):
        """
        Refresh VQA dialog data to sync with main window changes.
        """
        if (
            not hasattr(self.parent(), "image_list")
            or not self.parent().image_list
        ):
            QMessageBox.information(
                self,
                self.tr("Info"),
                self.tr("No images loaded in main window!"),
            )
            return

        try:
            self.switching_image = True

            if (
                hasattr(self.parent(), "other_data")
                and self.parent().other_data
            ):
                self.parent().other_data.pop("vqaData", None)

            current_file = self.parent().filename
            self.image_files = self.parent().image_list

            if current_file and current_file in self.image_files:
                self.parent().load_file(current_file)

            self.update_image_display()
            self.update_navigation_state()
            self.clear_all_components_silent()
            self.load_current_image_data()

            if self.image_files:
                self.page_input.setValidator(
                    QIntValidator(1, len(self.image_files))
                )

            def finalize_refresh():
                self.adjust_all_text_widgets_height()
                self.switching_image = False
                logger.info("VQA data refreshed successfully!")
                popup = Popup(
                    self.tr("VQA data refreshed successfully!"),
                    self,
                    icon=new_icon("copy-green", "svg"),
                )
                popup.show_popup(self, position="default")

            QTimer.singleShot(100, finalize_refresh)

        except Exception as e:
            self.switching_image = False
            logger.error(f"Error in refresh_data: {e}")

    def showEvent(self, event):
        """Adjust text widget heights after dialog shown"""
        super().showEvent(event)
        QTimer.singleShot(200, self.adjust_all_text_widgets_height)
