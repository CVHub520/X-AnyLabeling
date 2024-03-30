import os

from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QFileDialog

from anylabeling.services.auto_labeling.model_manager import ModelManager
from anylabeling.services.auto_labeling.types import AutoLabelingMode


class AutoLabelingWidget(QWidget):
    new_model_selected = pyqtSignal(str)
    new_custom_model_selected = pyqtSignal(str)
    auto_segmentation_requested = pyqtSignal()
    auto_segmentation_disabled = pyqtSignal()
    auto_labeling_mode_changed = pyqtSignal(AutoLabelingMode)
    clear_auto_labeling_action_requested = pyqtSignal()
    finish_auto_labeling_object_action_requested = pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        current_dir = os.path.dirname(__file__)
        uic.loadUi(os.path.join(current_dir, "auto_labeling.ui"), self)

        self.model_manager = ModelManager()
        self.model_manager.model_configs_changed.connect(
            lambda model_list: self.update_model_configs(model_list)
        )
        self.model_manager.new_model_status.connect(self.on_new_model_status)
        self.new_model_selected.connect(self.model_manager.load_model)
        self.new_custom_model_selected.connect(
            self.model_manager.load_custom_model
        )
        self.model_manager.model_loaded.connect(self.update_visible_widgets)
        self.model_manager.model_loaded.connect(self.on_new_model_loaded)
        self.model_manager.new_auto_labeling_result.connect(
            lambda auto_labeling_result: self.parent.new_shapes_from_auto_labeling(
                auto_labeling_result
            )
        )
        self.model_manager.auto_segmentation_model_selected.connect(
            self.auto_segmentation_requested
        )
        self.model_manager.auto_segmentation_model_unselected.connect(
            self.auto_segmentation_disabled
        )
        self.model_manager.output_modes_changed.connect(
            self.on_output_modes_changed
        )
        self.output_select_combobox.currentIndexChanged.connect(
            lambda: self.model_manager.set_output_mode(
                self.output_select_combobox.currentData()
            )
        )

        self.update_model_configs(self.model_manager.get_model_configs())

        # Disable tools when inference is running
        def set_enable_tools(enable):
            self.model_select_combobox.setEnabled(enable)
            self.output_select_combobox.setEnabled(enable)
            self.button_add_point.setEnabled(enable)
            self.button_remove_point.setEnabled(enable)
            self.button_add_rect.setEnabled(enable)
            self.button_clear.setEnabled(enable)
            self.button_finish_object.setEnabled(enable)

        self.model_manager.prediction_started.connect(
            lambda: set_enable_tools(False)
        )
        self.model_manager.prediction_finished.connect(
            lambda: set_enable_tools(True)
        )

        # Auto labeling buttons
        self.button_run.setShortcut("I")
        self.button_run.clicked.connect(self.run_prediction)
        self.edit_text.setPlaceholderText(
            "Enter text prompt here and press Enter to confirm."
        )
        self.edit_text.returnPressed.connect(self.run_vl_prediction)
        self.button_add_point.setShortcut("Q")
        self.button_add_point.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.ADD, AutoLabelingMode.POINT
            )
        )
        self.button_remove_point.setShortcut("E")
        self.button_remove_point.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.REMOVE, AutoLabelingMode.POINT
            )
        )
        self.button_add_rect.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.ADD, AutoLabelingMode.RECTANGLE
            )
        )
        self.button_clear.clicked.connect(
            self.clear_auto_labeling_action_requested
        )
        self.button_clear.setShortcut("B")
        self.button_finish_object.clicked.connect(
            self.finish_auto_labeling_object_action_requested
        )
        self.button_finish_object.setShortcut("F")

        # Hide labeling widgets by default
        self.hide_labeling_widgets()

        # Handle close button
        self.button_close.clicked.connect(self.unload_and_hide)

        # Handle model select combobox
        self.model_select_combobox.currentIndexChanged.connect(
            self.on_model_select_combobox_changed
        )

        self.auto_labeling_mode_changed.connect(self.update_button_colors)
        self.auto_labeling_mode = AutoLabelingMode.NONE
        self.auto_labeling_mode_changed.emit(self.auto_labeling_mode)

    def update_model_configs(self, model_list):
        """Update model list"""
        # Add models to combobox
        self.model_select_combobox.clear()
        self.model_select_combobox.addItem(self.tr("No Model"), userData=None)
        self.model_select_combobox.addItem(
            self.tr("...Load Custom Model"), userData="load_custom_model"
        )
        for model_config in model_list:
            self.model_select_combobox.addItem(
                (
                    self.tr("(User) ")
                    if model_config.get("is_custom_model", False)
                    else ""
                )
                + model_config["display_name"],
                userData=model_config["config_file"],
            )

    @pyqtSlot()
    def update_button_colors(self):
        """Update button colors"""
        style_sheet = """
            text-align: center;
            margin-right: 3px;
            border-radius: 5px;
            padding: 4px 8px;
            border: 1px solid #999999;
        """

        for button in [
            self.button_add_point,
            self.button_remove_point,
            self.button_add_rect,
            self.button_clear,
            self.button_finish_object,
        ]:
            button.setStyleSheet(style_sheet + "background-color: #ffffff;")
        if self.auto_labeling_mode == AutoLabelingMode.NONE:
            return
        if self.auto_labeling_mode.edit_mode == AutoLabelingMode.ADD:
            if self.auto_labeling_mode.shape_type == AutoLabelingMode.POINT:
                self.button_add_point.setStyleSheet(
                    style_sheet + "background-color: #00ff00;"
                )
            elif (
                self.auto_labeling_mode.shape_type
                == AutoLabelingMode.RECTANGLE
            ):
                self.button_add_rect.setStyleSheet(
                    style_sheet + "background-color: #00ff00;"
                )
        elif self.auto_labeling_mode.edit_mode == AutoLabelingMode.REMOVE:
            if self.auto_labeling_mode.shape_type == AutoLabelingMode.POINT:
                self.button_remove_point.setStyleSheet(
                    style_sheet + "background-color: #ff0000;"
                )

    def set_auto_labeling_mode(self, edit_mode, shape_type=None):
        """Set auto labeling mode"""
        if edit_mode is None:
            self.auto_labeling_mode = AutoLabelingMode.NONE
        else:
            self.auto_labeling_mode = AutoLabelingMode(edit_mode, shape_type)
        self.auto_labeling_mode_changed.emit(self.auto_labeling_mode)

    def run_prediction(self):
        """Run prediction"""
        if self.parent.filename is not None:
            self.model_manager.predict_shapes_threading(
                self.parent.image, self.parent.filename
            )

    def run_vl_prediction(self):
        """Run visual-language prediction"""
        if self.parent.filename is not None and self.edit_text:
            self.model_manager.predict_shapes_threading(
                self.parent.image, self.parent.filename, self.edit_text.text()
            )

    def unload_and_hide(self):
        """Unload model and hide widget"""
        self.model_select_combobox.setCurrentIndex(0)
        self.hide()

    def on_new_model_status(self, status):
        self.model_status_label.setText(status)

    def on_new_model_loaded(self, model_config):
        """Enable model select combobox"""
        self.model_select_combobox.currentIndexChanged.disconnect()
        if "config_file" not in model_config:
            self.model_select_combobox.setCurrentIndex(0)
        else:
            config_file = model_config["config_file"]
            self.model_select_combobox.setCurrentIndex(
                self.model_select_combobox.findData(config_file)
            )
        self.model_select_combobox.currentIndexChanged.connect(
            self.on_model_select_combobox_changed
        )
        self.model_select_combobox.setEnabled(True)

    def on_output_modes_changed(self, output_modes, default_output_mode):
        """Handle output modes changed"""
        # Disconnect onIndexChanged signal to prevent triggering
        # on model select combobox change
        self.output_select_combobox.currentIndexChanged.disconnect()

        self.output_select_combobox.clear()
        for output_mode, display_name in output_modes.items():
            self.output_select_combobox.addItem(
                display_name, userData=output_mode
            )
        self.output_select_combobox.setCurrentIndex(
            self.output_select_combobox.findData(default_output_mode)
        )

        # Reconnect onIndexChanged signal
        self.output_select_combobox.currentIndexChanged.connect(
            lambda: self.model_manager.set_output_mode(
                self.output_select_combobox.currentData()
            )
        )

    def on_model_select_combobox_changed(self, index):
        """Handle model select combobox change"""
        self.clear_auto_labeling_action_requested.emit()
        config_path = self.model_select_combobox.itemData(index)

        # Load custom model?
        if config_path == "load_custom_model":
            # Unload current model
            self.model_manager.unload_model()
            # Open file dialog to select "config.yaml" file for model
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter("Config file (*.yaml)")
            if file_dialog.exec_():
                config_file = file_dialog.selectedFiles()[0]
                # Disable combobox while loading model
                if config_path:
                    self.model_select_combobox.setEnabled(False)
                self.hide_labeling_widgets()
                self.model_manager.load_custom_model(config_file)
            else:
                self.model_select_combobox.setCurrentIndex(0)
            return

        # Disable combobox while loading model
        if config_path:
            self.model_select_combobox.setEnabled(False)
        self.hide_labeling_widgets()
        self.new_model_selected.emit(config_path)

    def update_visible_widgets(self, model_config):
        """Update widget status"""
        if not model_config or "model" not in model_config:
            return
        widgets = model_config["model"].get_required_widgets()
        for widget in widgets:
            getattr(self, widget).show()

    def hide_labeling_widgets(self):
        """Hide labeling widgets by default"""
        widgets = [
            "output_label",
            "output_select_combobox",
            "button_run",
            "edit_text",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
        ]
        for widget in widgets:
            getattr(self, widget).hide()

    def on_new_marks(self, marks):
        """Handle new marks"""
        self.model_manager.set_auto_labeling_marks(marks)
        self.run_prediction()

    def on_open(self):
        pass

    def on_close(self):
        return True
