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
    cache_auto_label_changed = pyqtSignal()

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
        self.upn_select_combobox.currentIndexChanged.connect(
            self.on_upn_mode_changed
        )
        self.florence2_select_combobox.currentIndexChanged.connect(
            self.on_florence2_mode_changed
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
            self.upn_select_combobox.setEnabled(enable)
            self.florence2_select_combobox.setEnabled(enable)

        self.model_manager.prediction_started.connect(
            lambda: set_enable_tools(False)
        )
        self.model_manager.prediction_finished.connect(
            lambda: set_enable_tools(True)
        )

        # Init value
        self.initial_conf_value = 0
        self.initial_iou_value = 0
        self.initial_preserve_annotations_state = False

        # Auto labeling buttons
        self.button_run.setShortcut("I")
        self.button_run.clicked.connect(self.run_prediction)
        self.button_send.clicked.connect(self.run_vl_prediction)
        self.edit_conf.valueChanged.connect(self.on_conf_value_changed)
        self.edit_iou.valueChanged.connect(self.on_iou_value_changed)
        self.button_reset_tracker.clicked.connect(self.on_reset_tracker)
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
        self.button_finish_object.clicked.connect(self.add_new_prompt)
        self.button_finish_object.clicked.connect(
            self.finish_auto_labeling_object_action_requested
        )
        self.button_finish_object.clicked.connect(
            self.cache_auto_label_changed
        )
        self.button_finish_object.setShortcut("F")
        self.toggle_preserve_existing_annotations.stateChanged.connect(
            self.on_preserve_existing_annotations_state_changed
        )
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

        # Populate UPN select combobox with modes
        self.populate_upn_combobox()
        # Populate Florence2 select combobox with modes
        self.populate_florence2_combobox()

    def populate_upn_combobox(self):
        """Populate UPN combobox with available modes"""
        self.upn_select_combobox.clear()
        # Define modes with display names
        modes = {
            "coarse_grained_prompt": self.tr("Coarse Grained"),
            "fine_grained_prompt": self.tr("Fine Grained"),
        }
        # Add modes to combobox
        for mode, display_name in modes.items():
            self.upn_select_combobox.addItem(display_name, userData=mode)

    def populate_florence2_combobox(self):
        """Populate Florence2 combobox with available modes"""
        self.florence2_select_combobox.clear()
        # Define modes with display names
        modes = {
            "caption": self.tr("Caption"),
            "detailed_cap": self.tr("Detailed Caption"),
            "more_detailed_cap": self.tr("More Detailed Caption"),
            "od": self.tr("Object Detection"),
            "region_proposal": self.tr("Region Proposal"),
            "dense_region_cap": self.tr("Dense Region Caption"),
            "refer_exp_seg": self.tr("Refer-Exp Segmentation"),
            "region_to_seg": self.tr("Region to Segmentation"),
            "ovd": self.tr("OVD"),
            "cap_to_pg": self.tr("Caption to Parse Grounding"),
            "region_to_cat": self.tr("Region to Category"),
            "region_to_desc": self.tr("Region to Description"),
            "ocr": self.tr("OCR"),
            "ocr_with_region": self.tr("OCR with Region"),
        }
        # Add modes to combobox
        for mode, display_name in modes.items():
            self.florence2_select_combobox.addItem(display_name, userData=mode)

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
            color: #000000;  /* black */
            background-color: #e0e0e0;  /* light gray */
        """

        for button in [
            self.button_add_point,
            self.button_remove_point,
            self.button_add_rect,
            self.button_clear,
            self.button_finish_object,
        ]:
            button.setStyleSheet(style_sheet)
        if self.auto_labeling_mode == AutoLabelingMode.NONE:
            return
        if self.auto_labeling_mode.edit_mode == AutoLabelingMode.ADD:
            if self.auto_labeling_mode.shape_type == AutoLabelingMode.POINT:
                self.button_add_point.setStyleSheet(
                    style_sheet + "background-color: #90EE90;"  # light green color
                )
            elif self.auto_labeling_mode.shape_type == AutoLabelingMode.RECTANGLE:
                self.button_add_rect.setStyleSheet(
                    style_sheet + "background-color: #90EE90;"  # light green color
                )
        elif self.auto_labeling_mode.edit_mode == AutoLabelingMode.REMOVE:
            if self.auto_labeling_mode.shape_type == AutoLabelingMode.POINT:
                self.button_remove_point.setStyleSheet(
                    style_sheet + "background-color: #FFB6C1;"  # light red color
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
                self.parent.image,
                self.parent.filename,
                text_prompt=self.edit_text.text(),
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

        # Reset controls to initial values when the model changes
        self.on_conf_value_changed(self.initial_conf_value)
        self.on_iou_value_changed(self.initial_iou_value)
        self.on_preserve_existing_annotations_state_changed(
            self.initial_preserve_annotations_state
        )
        self.on_reset_tracker()

        # Update UPN mode in UI if UPN model is loaded
        if model_config.get("type") == "upn":
            self.update_upn_mode_ui()
        # Update Florence2 mode in UI if Florence2 model is loaded
        elif model_config.get("type") == "florence2":
            self.update_florence2_mode_ui()

    def update_upn_mode_ui(self):
        """Update UPN mode combobox to reflect current backend state"""
        current_mode = self.model_manager.loaded_model_config[
            "model"
        ].prompt_type
        index = self.upn_select_combobox.findData(current_mode)
        if index != -1:
            self.upn_select_combobox.setCurrentIndex(index)

    def update_florence2_mode_ui(self):
        """Update Florence2 mode combobox to reflect current backend state"""
        current_mode = self.model_manager.loaded_model_config[
            "model"
        ].prompt_type
        index = self.florence2_select_combobox.findData(current_mode)
        if index != -1:
            self.florence2_select_combobox.setCurrentIndex(index)
        self.update_florence2_widgets(current_mode)

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
            "button_run",
            "button_add_point",
            "button_remove_point",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
            "button_send",
            "edit_text",
            "edit_conf",
            "edit_iou",
            "input_box_thres",
            "input_conf",
            "input_iou",
            "output_label",
            "output_select_combobox",
            "toggle_preserve_existing_annotations",
            "button_reset_tracker",
            "upn_select_combobox",
            "florence2_select_combobox",
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

    def on_conf_value_changed(self, value):
        self.initial_conf_value = value
        self.model_manager.set_auto_labeling_conf(value)

    def on_iou_value_changed(self, value):
        self.initial_iou_value = value
        self.model_manager.set_auto_labeling_iou(value)

    def on_preserve_existing_annotations_state_changed(self, state):
        self.initial_preserve_annotations_state = state
        self.model_manager.set_auto_labeling_preserve_existing_annotations_state(
            state
        )

    def on_reset_tracker(self):
        self.model_manager.set_auto_labeling_reset_tracker()

    def on_cache_auto_label_changed(self, text, gid):
        self.model_manager.set_cache_auto_label(text, gid)

    def add_new_prompt(self):
        self.model_manager.set_auto_labeling_prompt()

    @pyqtSlot()
    def on_upn_mode_changed(self):
        """Handle UPN mode change"""
        mode = self.upn_select_combobox.currentData()
        self.model_manager.set_upn_mode(mode)

    @pyqtSlot()
    def on_florence2_mode_changed(self):
        """Handle Florence2 mode change"""
        mode = self.florence2_select_combobox.currentData()
        self.model_manager.set_florence2_mode(mode)
        self.update_florence2_widgets(mode)

    def update_florence2_widgets(self, mode):
        """Update widget visibility based on Florence2 mode"""
        # Check if Florence2 model is loaded
        if (
            not self.model_manager.loaded_model_config
            or self.model_manager.loaded_model_config.get("type")
            != "florence2"
        ):
            return

        # Define which widgets are needed for each mode
        mode_widgets = {
            # Only need run button
            "caption": ["button_run"],
            "detailed_cap": ["button_run"],
            "more_detailed_cap": ["button_run"],
            "ocr": ["button_run"],
            "ocr_with_region": ["button_run"],
            "od": ["button_run"],
            "region_proposal": ["button_run"],
            "dense_region_cap": ["button_run"],
            # Region-based modes need rectangle tools
            "region_to_cat": [
                "button_add_rect",
                "button_clear",
                "button_finish_object",
            ],
            "region_to_desc": [
                "button_add_rect",
                "button_clear",
                "button_finish_object",
            ],
            "region_to_seg": [
                "button_add_rect",
                "button_clear",
                "button_finish_object",
            ],
            # Other modes
            "refer_exp_seg": ["edit_text", "button_send"],
            "cap_to_pg": ["edit_text", "button_send"],
            "ovd": ["edit_text", "button_send"],
        }

        # Define which modes should preserve existing annotations by default
        preserve_annotations_modes = {
            # Modes that should preserve existing annotations (replace=False)
            "region_to_cat": True,
            "region_to_desc": True,
            "region_to_seg": True,
            "refer_exp_seg": True,
            # Modes that should replace existing annotations (replace=True)
            "caption": False,
            "detailed_cap": False,
            "more_detailed_cap": False,
            "od": False,
            "region_proposal": False,
            "dense_region_cap": False,
            "ovd": False,
            "cap_to_pg": False,
            "ocr": False,
            "ocr_with_region": False,
        }

        # Hide all widgets first
        widgets_to_manage = [
            "edit_text",
            "button_run",
            "button_send",
            "button_add_rect",
            "button_clear",
            "button_finish_object",
        ]

        for widget_name in widgets_to_manage:
            getattr(self, widget_name).hide()

        if mode in ["ovd", "cap_to_pg", "refer_exp_seg"]:
            self.edit_text.setPlaceholderText("Enter prompt here...")

        # Show only the widgets needed for current mode
        if mode in mode_widgets:
            for widget_name in mode_widgets[mode]:
                getattr(self, widget_name).show()

            # Show preserve annotations toggle for all modes
            self.toggle_preserve_existing_annotations.show()
            # Set the default state for preserve annotations
            if mode in preserve_annotations_modes:
                # Temporarily disconnect the signal to avoid triggering the callback
                self.toggle_preserve_existing_annotations.stateChanged.disconnect()
                # Set the state
                self.toggle_preserve_existing_annotations.setChecked(
                    preserve_annotations_modes[mode]
                )
                # Reconnect the signal
                self.toggle_preserve_existing_annotations.stateChanged.connect(
                    self.on_preserve_existing_annotations_state_changed
                )
                # Manually trigger the state change to update the model
                self.on_preserve_existing_annotations_state_changed(
                    preserve_annotations_modes[mode]
                )
