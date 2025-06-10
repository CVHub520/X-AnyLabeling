import os
import yaml
import collections

from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QPoint
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QWidget,
)

from anylabeling.services.auto_labeling.model_manager import ModelManager
from anylabeling.services.auto_labeling.types import AutoLabelingMode
from anylabeling.services.auto_labeling import (
    _AUTO_LABELING_IOU_MODELS,
    _AUTO_LABELING_CONF_MODELS,
    _SKIP_PREDICTION_ON_NEW_MARKS_MODELS,
)
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.style import (
    get_lineedit_style,
    get_double_spinbox_style,
    get_normal_button_style,
    get_highlight_button_style,
    get_toggle_button_style,
)
from anylabeling.views.labeling.widgets.api_token_dialog import ApiTokenDialog
from anylabeling.views.labeling.widgets.searchable_model_dropdown import (
    load_json,
    save_json,
    _MODELS_CONFIG_PATH,
    SearchableModelDropdownPopup,
)


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
        self.gd_select_combobox.currentIndexChanged.connect(
            self.on_gd_mode_changed
        )

        # Disable tools when inference is running
        def set_enable_tools(enable):
            self.model_selection_button.setEnabled(enable)
            self.output_select_combobox.setEnabled(enable)
            self.button_add_point.setEnabled(enable)
            self.button_remove_point.setEnabled(enable)
            self.button_add_rect.setEnabled(enable)
            self.button_clear.setEnabled(enable)
            self.button_finish_object.setEnabled(enable)
            self.upn_select_combobox.setEnabled(enable)
            self.gd_select_combobox.setEnabled(enable)
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

        # ===================================
        #  Auto labeling buttons
        # ===================================

        # --- Configuration for: model_selection_button ---
        model_data = self.init_model_data()
        self.model_dropdown = SearchableModelDropdownPopup(model_data)
        self.model_dropdown.hide()
        self.model_dropdown.modelSelected.connect(self.on_model_selected)
        self.model_selection_button.setStyleSheet(get_normal_button_style())
        self.model_selection_button.clicked.connect(self.show_model_dropdown)

        # --- Configuration for: button_run ---
        self.button_run.setShortcut("I")
        self.button_run.setStyleSheet(get_highlight_button_style())
        self.button_run.clicked.connect(self.run_prediction)

        # --- Configuration for: button_reset_tracker ---
        self.button_reset_tracker.setStyleSheet(get_normal_button_style())
        self.button_reset_tracker.clicked.connect(self.on_reset_tracker)

        # --- Configuration for: button_set_api_token ---
        self.button_set_api_token.setStyleSheet(get_normal_button_style())
        self.button_set_api_token.setToolTip(
            self.tr(
                "You can set the API token via the GROUNDING_DINO_API_TOKEN environment variable"
            )
        )
        self.button_set_api_token.clicked.connect(self.on_set_api_token)

        # --- Configuration for: button_send ---
        self.button_send.setStyleSheet(get_highlight_button_style())
        self.button_send.clicked.connect(self.run_vl_prediction)

        # --- Configuration for: edit_conf ---
        self.edit_conf.setStyleSheet(get_double_spinbox_style())
        self.edit_conf.valueChanged.connect(self.on_conf_value_changed)

        # --- Configuration for: edit_iou ---
        self.edit_iou.setStyleSheet(get_double_spinbox_style())
        self.edit_iou.valueChanged.connect(self.on_iou_value_changed)

        # --- Configuration for: edit_text ---
        self.edit_text.setStyleSheet(get_lineedit_style())

        # --- Configuration for: button_add_point ---
        self.button_add_point.setShortcut("Q")
        self.button_add_point.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.ADD, AutoLabelingMode.POINT
            )
        )

        # --- Configuration for: button_remove_point ---
        self.button_remove_point.setShortcut("E")
        self.button_remove_point.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.REMOVE, AutoLabelingMode.POINT
            )
        )

        # --- Configuration for: button_add_rect ---
        self.button_add_rect.clicked.connect(
            lambda: self.set_auto_labeling_mode(
                AutoLabelingMode.ADD, AutoLabelingMode.RECTANGLE
            )
        )

        # --- Configuration for: button_clear ---
        self.button_clear.clicked.connect(
            self.clear_auto_labeling_action_requested
        )
        self.button_clear.setShortcut("B")

        # --- Configuration for: button_finish_object ---
        self.button_finish_object.clicked.connect(self.add_new_prompt)
        self.button_finish_object.clicked.connect(
            self.finish_auto_labeling_object_action_requested
        )
        self.button_finish_object.clicked.connect(
            self.cache_auto_label_changed
        )
        self.button_finish_object.setShortcut("F")

        # --- Configuration for: toggle_preserve_existing_annotations ---
        self.toggle_preserve_existing_annotations.setChecked(False)
        self.toggle_preserve_existing_annotations.setCheckable(True)
        self.toggle_preserve_existing_annotations.setStyleSheet(
            get_normal_button_style()
        )
        tooltip_on = self.tr(
            "Existing shapes will be preserved during updates. Click to switch to overwriting."
        )
        tooltip_off = self.tr(
            "Existing shapes will be overwritten by new shapes during updates. Click to switch to preserving."
        )
        self.toggle_preserve_existing_annotations.setToolTip(tooltip_off)
        self.toggle_preserve_existing_annotations.clicked.connect(
            lambda checked: (
                self.toggle_preserve_existing_annotations.setToolTip(
                    tooltip_on if checked else tooltip_off
                ),
                self.toggle_preserve_existing_annotations.setText(
                    self.tr("Replace (Off)")
                    if checked
                    else self.tr("Replace (On)")
                ),
            )
        )
        self.toggle_preserve_existing_annotations.toggled.connect(
            self.on_preserve_existing_annotations_state_changed
        )

        # ===================================
        #  End of Auto labeling buttons
        # ===================================

        # Hide labeling widgets by default
        self.hide_labeling_widgets()

        # Handle close button
        self.button_close.clicked.connect(self.unload_and_hide)

        self.auto_labeling_mode_changed.connect(self.update_button_colors)
        self.auto_labeling_mode = AutoLabelingMode.NONE
        self.auto_labeling_mode_changed.emit(self.auto_labeling_mode)

        # Populate select combobox with modes
        self.populate_upn_combobox()
        self.populate_florence2_combobox()
        self.populate_gd_combobox()

    def init_model_data(self):
        """Get models data"""
        model_data = {
            "Custom": {
                "load_custom_model": {
                    "selected": False,
                    "favorite": False,
                    "display_name": "...Load Custom Model",
                }
            }
        }
        self.model_info = {
            "load_custom_model": {
                "display_name": "...Load Custom Model",
                "config_path": None,
            }
        }

        try:
            local_model_data = load_json(_MODELS_CONFIG_PATH)["models_data"]
            for model_name, model_dict in local_model_data["Custom"].items():
                if model_name == "load_custom_model":
                    continue
                elif not os.path.exists(model_dict["config_path"]):
                    continue

                if not model_name.startswith("_custom_"):
                    model_name = f"_custom_{model_name}"

                model_data["Custom"][model_name] = {
                    "selected": False,
                    "favorite": model_dict["favorite"],
                    "display_name": model_dict["display_name"],
                    "config_path": model_dict["config_path"],
                }

                self.model_info[model_name] = {
                    "display_name": model_dict["display_name"],
                    "config_path": model_dict["config_path"],
                }

        except Exception as _:
            local_model_data = {}

        model_list = self.model_manager.get_model_configs()
        for model_dict in model_list:
            model_name = model_dict["name"]
            if model_dict.get("is_custom_model", False):
                provider_name = "Custom"
            else:
                provider_name = model_dict.get("provider", "Others")

            if provider_name not in model_data:
                model_data[provider_name] = {}

            if (
                provider_name in local_model_data
                and model_name in local_model_data[provider_name]
            ):
                local_model_data[provider_name][model_name]["selected"] = False
                model_data[provider_name].update(
                    local_model_data[provider_name]
                )
            else:
                model_data[provider_name][model_name] = {
                    "selected": False,
                    "favorite": False,
                    "display_name": model_dict["display_name"],
                }

            self.model_info[model_name] = {
                "display_name": model_dict["display_name"],
                "config_path": (
                    None
                    if model_name == "load_custom_model"
                    else model_dict["config_file"]
                ),
            }

        # Sort the collected model_data
        sorted_model_data = self._sort_model_data(model_data)

        return sorted_model_data

    def _sort_model_data(self, model_data: dict) -> collections.OrderedDict:
        """Sorts the model data dictionary"""

        def top_level_sort_key(key: str):
            if key == "Custom":
                return (0,)
            if key == "Others":
                return (2,)
            return (1, key)

        def inner_sort_key(item: tuple[str, dict]):
            _, model_details = item
            display_name = model_details.get("display_name", "")
            if display_name == "...Load Custom Model":
                return (0,)
            return (1, display_name)

        sorted_top_keys = sorted(model_data.keys(), key=top_level_sort_key)
        sorted_data = collections.OrderedDict()
        for key in sorted_top_keys:
            inner_dict = model_data[key]
            sorted_inner_items = sorted(inner_dict.items(), key=inner_sort_key)
            sorted_data[key] = collections.OrderedDict(sorted_inner_items)
        return sorted_data

    def show_model_dropdown(self):
        """Show the model dropdown"""
        button_pos = self.model_selection_button.mapToGlobal(QPoint(0, 0))
        self.model_dropdown.move(int(button_pos.x()), int(button_pos.y()))
        self.model_dropdown.adjustSize()
        self.model_dropdown.show()

    def on_model_selected(self, provider, model_name):
        """Handle the model selected event"""

        if model_name == "load_custom_model":
            # Unload current model first
            self.model_manager.unload_model()

            # Open file dialog to select "config.yaml" file for model
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter("Config file (*.yaml)")

            if file_dialog.exec_():
                self.hide_labeling_widgets()
                config_file = file_dialog.selectedFiles()[0]
                flag = self.model_manager.load_custom_model(config_file)
                if not flag:
                    self.model_selection_button.setText("No Model")
                    return

                # update model_info
                with open(config_file, "r", encoding="utf-8") as f:
                    config_info = yaml.safe_load(f)

                if not config_info["name"].startswith("_custom_"):
                    config_info["name"] = f"_custom_{config_info['name']}"

                self.model_info[config_info["name"]] = {
                    "display_name": config_info["display_name"],
                    "config_path": config_file,
                }

                # update model_data
                models_data = self.init_model_data()
                models_data["Custom"]["load_custom_model"]["selected"] = False
                models_data["Custom"][config_info["name"]] = {
                    "selected": True,
                    "favorite": False,
                    "display_name": config_info["display_name"],
                    "config_path": config_file,
                }
                save_json({"models_data": models_data}, _MODELS_CONFIG_PATH)
                self.model_dropdown.update_models_data(models_data)

                self.clear_auto_labeling_action_requested.emit()
                self.model_selection_button.setText(
                    config_info["display_name"]
                )
                self.model_selection_button.setEnabled(False)

            return

        self.clear_auto_labeling_action_requested.emit()
        self.model_selection_button.setText(
            self.model_info[model_name]["display_name"]
        )

        self.model_selection_button.setEnabled(False)
        self.hide_labeling_widgets()

        if provider == "Custom":
            self.model_manager.load_custom_model(
                self.model_info[model_name]["config_path"]
            )
        else:
            self.new_model_selected.emit(
                self.model_info[model_name]["config_path"]
            )

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

    def populate_gd_combobox(self):
        """Populate GroundingDino combobox with available modes"""
        self.gd_select_combobox.clear()
        # Define modes with display names
        modes = {
            "GroundingDino_1_6_Pro": "GroundingDino-1.6-Pro",
            "GroundingDino_1_6_Edge": "GroundingDino-1.6-Edge",
            "GroundingDino_1_5_Pro": "GroundingDino-1.5-Pro",
            "GroundingDino_1_5_Edge": "GroundingDino-1.5-Edge",
        }
        # Add modes to combobox
        for mode, display_name in modes.items():
            self.gd_select_combobox.addItem(display_name, userData=mode)

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

    @pyqtSlot()
    def update_button_colors(self):
        """Update button colors"""
        for button in [
            self.button_add_point,
            self.button_remove_point,
            self.button_add_rect,
            self.button_clear,
            self.button_finish_object,
        ]:
            button.setStyleSheet(get_normal_button_style())
        if self.auto_labeling_mode == AutoLabelingMode.NONE:
            return
        if self.auto_labeling_mode.edit_mode == AutoLabelingMode.ADD:
            if self.auto_labeling_mode.shape_type == AutoLabelingMode.POINT:
                self.button_add_point.setStyleSheet(
                    get_toggle_button_style(button_color="#90EE90")
                )
            elif (
                self.auto_labeling_mode.shape_type
                == AutoLabelingMode.RECTANGLE
            ):
                self.button_add_rect.setStyleSheet(
                    get_toggle_button_style(button_color="#90EE90")
                )
        elif self.auto_labeling_mode.edit_mode == AutoLabelingMode.REMOVE:
            if self.auto_labeling_mode.shape_type == AutoLabelingMode.POINT:
                self.button_remove_point.setStyleSheet(
                    get_toggle_button_style(button_color="#FFB6C1")
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
        self.hide()

    def on_new_model_status(self, status):
        self.model_status_label.setText(status)

    def on_new_model_loaded(self, model_config):
        """Enable model select combobox"""
        self.model_selection_button.setEnabled(True)

        # Reset controls to initial values when the model changes
        try:
            if (
                self.model_manager.loaded_model_config["type"]
                in _AUTO_LABELING_IOU_MODELS
            ):
                initial_iou_value = self.model_manager.loaded_model_config[
                    "iou_threshold"
                ]
                self.edit_iou.setValue(initial_iou_value)
            else:
                initial_iou_value = 0.0
                self.edit_iou.setValue(initial_iou_value)
        except Exception as _:
            initial_iou_value = 0.0
            self.edit_iou.setValue(initial_iou_value)

        try:
            if (
                self.model_manager.loaded_model_config["type"]
                in _AUTO_LABELING_CONF_MODELS
            ):
                initial_conf_value = self.model_manager.loaded_model_config[
                    "conf_threshold"
                ]
                self.edit_conf.setValue(initial_conf_value)
            else:
                initial_conf_value = 0.0
                self.edit_conf.setValue(initial_conf_value)
        except Exception as _:
            initial_conf_value = 0.0
            self.edit_conf.setValue(initial_conf_value)

        self.on_reset_tracker()
        self.on_iou_value_changed(initial_iou_value)
        self.on_conf_value_changed(initial_conf_value)
        self.on_preserve_existing_annotations_state_changed(
            self.initial_preserve_annotations_state
        )

        # Update specific mode in UI if specific model is loaded
        if model_config.get("type") == "upn":
            self.update_upn_mode_ui()
        elif model_config.get("type") == "florence2":
            self.update_florence2_mode_ui()
        elif model_config.get("type") == "groundingdino":
            self.update_groundingdino_mode_ui()

    def update_upn_mode_ui(self):
        """Update UPN mode combobox to reflect current backend state"""
        current_mode = self.model_manager.loaded_model_config[
            "model"
        ].prompt_type
        index = self.upn_select_combobox.findData(current_mode)
        if index != -1:
            self.upn_select_combobox.setCurrentIndex(index)

    def update_groundingdino_mode_ui(self):
        """Update GroundingDino mode combobox to reflect current backend state"""
        current_mode = self.model_manager.loaded_model_config[
            "model"
        ].prompt_type
        index = self.gd_select_combobox.findData(current_mode)
        if index != -1:
            self.gd_select_combobox.setCurrentIndex(index)

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

    def update_visible_widgets(self, model_config):
        """Update widget status"""
        if not model_config or "model" not in model_config:
            return
        widgets = model_config["model"].get_required_widgets()
        for widget_name in widgets:
            if hasattr(self, widget_name):
                getattr(self, widget_name).show()
            else:
                logger.warning(
                    f"Warning: Widget '{widget_name}' not found in AutoLabelingWidget."
                )

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
            "button_set_api_token",
            "button_reset_tracker",
            "upn_select_combobox",
            "gd_select_combobox",
            "florence2_select_combobox",
        ]
        for widget in widgets:
            getattr(self, widget).hide()

    def on_new_marks(self, marks):
        """Handle new marks"""
        self.model_manager.set_auto_labeling_marks(marks)
        current_model_name = self.model_manager.loaded_model_config["type"]
        if current_model_name not in _SKIP_PREDICTION_ON_NEW_MARKS_MODELS:
            self.run_prediction()

    def on_open(self):
        pass

    def on_close(self):
        return True

    def on_conf_value_changed(self, value):
        """Handle conf value changed"""
        self.model_manager.set_auto_labeling_conf(value)

    def on_iou_value_changed(self, value):
        """Handle iou value changed"""
        self.model_manager.set_auto_labeling_iou(value)

    def on_preserve_existing_annotations_state_changed(self, state):
        """Handle preserve existing annotations state changed"""
        self.initial_preserve_annotations_state = state
        self.model_manager.set_auto_labeling_preserve_existing_annotations_state(
            state
        )

    def on_reset_tracker(self):
        """Handle reset tracker"""
        self.model_manager.set_auto_labeling_reset_tracker()

    def on_set_api_token(self):
        """Show a dialog to input the API token."""
        dialog = ApiTokenDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            token = dialog.get_token()
            try:
                self.model_manager.set_auto_labeling_api_token(token)
            except Exception as e:
                logger.error(f"Error setting API token: {e}")

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
    def on_gd_mode_changed(self):
        """Handle GroundingDino mode change"""
        mode = self.gd_select_combobox.currentData()
        self.model_manager.set_groundingdino_mode(mode)

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
            "region_to_cat": "Replace (Off)",
            "region_to_desc": "Replace (Off)",
            "region_to_seg": "Replace (Off)",
            "refer_exp_seg": "Replace (Off)",
            # Modes that should replace existing annotations (replace=True)
            "caption": "Replace (On)",
            "detailed_cap": "Replace (On)",
            "more_detailed_cap": "Replace (On)",
            "od": "Replace (On)",
            "region_proposal": "Replace (On)",
            "dense_region_cap": "Replace (On)",
            "ovd": "Replace (On)",
            "cap_to_pg": "Replace (On)",
            "ocr": "Replace (On)",
            "ocr_with_region": "Replace (On)",
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
                self.toggle_preserve_existing_annotations.toggled.disconnect()
                # Set the state
                self.toggle_preserve_existing_annotations.setText(
                    preserve_annotations_modes[mode]
                )
                # Reconnect the signal
                self.toggle_preserve_existing_annotations.toggled.connect(
                    self.on_preserve_existing_annotations_state_changed
                )
                # Manually trigger the state change to update the model
                self.on_preserve_existing_annotations_state_changed(
                    preserve_annotations_modes[mode]
                )
