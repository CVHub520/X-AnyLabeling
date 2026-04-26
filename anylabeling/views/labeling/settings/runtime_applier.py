from __future__ import annotations

import math
from typing import Any

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from anylabeling.views.common.device_manager import device_manager

from .. import utils
from ..logger import logger
from ..shape import Shape
from ..widgets import LabelDialog

LABEL_OPACITY = 128


class SettingsRuntimeApplier:
    def __init__(self, widget: QtWidgets.QWidget):
        self._widget = widget
        self._shortcut_action_map: dict[str, QtGui.QAction] = {}
        self._hidden_actions: dict[str, QtGui.QAction] = {}

    @staticmethod
    def shortcut_value_to_text(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value if v)
        if value in (None, ""):
            return ""
        return str(value)

    def set_auto_switch_to_edit_mode(self, enabled: bool) -> None:
        signal_connected = bool(
            getattr(self._widget, "_auto_switch_signal_connected", False)
        )
        if enabled and not signal_connected:
            self._widget.canvas.mode_changed.connect(
                self._widget.set_edit_mode
            )
            self._widget._auto_switch_signal_connected = True
            return
        if not enabled and signal_connected:
            try:
                self._widget.canvas.mode_changed.disconnect(
                    self._widget.set_edit_mode
                )
            except TypeError:
                pass
            self._widget._auto_switch_signal_connected = False

    def build_shortcut_action_map(self) -> None:
        shortcut_map = {
            "shortcuts.close": self._widget.actions.close,
            "shortcuts.open": self._widget.actions.open,
            "shortcuts.open_video": self._widget.actions.open_video,
            "shortcuts.open_dir": self._widget.actions.open_dir,
            "shortcuts.open_chatbot": self._widget.actions.open_chatbot,
            "shortcuts.open_vqa": self._widget.actions.open_vqa,
            "shortcuts.open_classifier": self._widget.actions.open_classifier,
            "shortcuts.open_paddleocr": self._widget.actions.open_paddleocr,
            "shortcuts.save": self._widget.actions.save,
            "shortcuts.save_as": self._widget.actions.save_as,
            "shortcuts.save_to": self._widget.actions.change_output_dir,
            "shortcuts.delete_file": self._widget.actions.delete_file,
            "shortcuts.delete_image_file": self._widget.actions.delete_image_file,
            "shortcuts.open_next": self._widget.actions.open_next_image,
            "shortcuts.open_prev": self._widget.actions.open_prev_image,
            "shortcuts.open_next_unchecked": self._widget.actions.open_next_unchecked_image,
            "shortcuts.open_prev_unchecked": self._widget.actions.open_prev_unchecked_image,
            "shortcuts.toggle_annotation_checked": self._widget.actions.toggle_annotation_checked,
            "shortcuts.zoom_in": self._widget.actions.zoom_in,
            "shortcuts.zoom_out": self._widget.actions.zoom_out,
            "shortcuts.zoom_to_original": self._widget.actions.zoom_org,
            "shortcuts.fit_window": self._widget.actions.fit_window,
            "shortcuts.fit_width": self._widget.actions.fit_width,
            "shortcuts.show_navigator": self._widget.actions.show_navigator,
            "shortcuts.create_polygon": self._widget.actions.create_mode,
            "shortcuts.create_brush_polygon": self._widget.actions.create_brush_polygon_mode,
            "shortcuts.create_rectangle": self._widget.actions.create_rectangle_mode,
            "shortcuts.create_cuboid": self._widget.actions.create_cuboid_mode,
            "shortcuts.create_rotation": self._widget.actions.create_rotation_mode,
            "shortcuts.create_quadrilateral": self._widget.actions.create_quadrilateral_mode,
            "shortcuts.create_circle": self._widget.actions.create_circle_mode,
            "shortcuts.create_line": self._widget.actions.create_line_mode,
            "shortcuts.create_point": self._widget.actions.create_point_mode,
            "shortcuts.create_linestrip": self._widget.actions.create_line_strip_mode,
            "shortcuts.edit_polygon": self._widget.actions.edit_mode,
            "shortcuts.delete_polygon": self._widget.actions.delete,
            "shortcuts.duplicate_polygon": self._widget.actions.duplicate,
            "shortcuts.copy_polygon": self._widget.actions.copy,
            "shortcuts.paste_polygon": self._widget.actions.paste,
            "shortcuts.undo": self._widget.actions.undo,
            "shortcuts.undo_last_point": self._widget.actions.undo_last_point,
            "shortcuts.edit_label": self._widget.actions.edit,
            "shortcuts.edit_digit_shortcut": self._widget.actions.digit_shortcut_manager,
            "shortcuts.edit_group_id": self._widget.actions.gid_manager,
            "shortcuts.edit_labels": self._widget.actions.label_manager,
            "shortcuts.edit_shapes": self._widget.actions.shape_manager,
            "shortcuts.toggle_keep_prev_mode": self._widget.actions.keep_prev_mode,
            "shortcuts.remove_selected_point": self._widget.actions.remove_point,
            "shortcuts.group_selected_shapes": self._widget.actions.group_selected_shapes,
            "shortcuts.ungroup_selected_shapes": self._widget.actions.ungroup_selected_shapes,
            "shortcuts.hide_selected_polygons": self._widget.actions.hide_selected_polygons,
            "shortcuts.show_hidden_polygons": self._widget.actions.show_hidden_polygons,
            "shortcuts.show_overview": self._widget.actions.overview,
            "shortcuts.show_masks": self._widget.actions.show_masks,
            "shortcuts.show_texts": self._widget.actions.show_texts,
            "shortcuts.show_labels": self._widget.actions.show_labels,
            "shortcuts.show_linking": self._widget.actions.show_linking,
            "shortcuts.show_attributes": self._widget.actions.show_attributes,
            "shortcuts.union_selected_shapes": self._widget.actions.union_selection,
            "shortcuts.toggle_auto_use_last_label": self._widget.actions.auto_use_last_label_mode,
            "shortcuts.toggle_auto_use_last_gid": self._widget.actions.auto_use_last_gid_mode,
            "shortcuts.toggle_visibility_shapes": self._widget.actions.visibility_shapes_mode,
            "shortcuts.toggle_compare_view": self._widget.actions.toggle_compare_view,
            "shortcuts.auto_label": self._widget.actions.toggle_auto_labeling_widget,
            "shortcuts.auto_run": self._widget.actions.run_all_images,
            "shortcuts.loop_thru_labels": self._widget.actions.loop_thru_labels,
            "shortcuts.loop_select_labels": self._widget.actions.loop_select_labels,
        }
        shortcut_map["shortcuts.quit"] = self._ensure_hidden_shortcut_action(
            "quit",
            self._widget.tr("Quit"),
            self._quit_application,
        )
        shortcut_map["shortcuts.open_settings"] = (
            self._ensure_hidden_shortcut_action(
                "open_settings",
                self._widget.tr("Open Settings"),
                self._widget.open_settings_dialog,
            )
        )
        shortcut_map["shortcuts.add_point_to_edge"] = (
            self._ensure_hidden_shortcut_action(
                "add_point_to_edge",
                self._widget.tr("Add Point To Edge"),
                self._widget.add_point_to_edge,
            )
        )
        shortcut_map["shortcuts.auto_labeling_add_point"] = (
            self._ensure_hidden_shortcut_action(
                "auto_labeling_add_point",
                self._widget.tr("Auto Labeling Add Point"),
                lambda: self._trigger_auto_labeling_button("button_add_point"),
            )
        )
        shortcut_map["shortcuts.auto_labeling_remove_point"] = (
            self._ensure_hidden_shortcut_action(
                "auto_labeling_remove_point",
                self._widget.tr("Auto Labeling Remove Point"),
                lambda: self._trigger_auto_labeling_button(
                    "button_remove_point"
                ),
            )
        )
        shortcut_map["shortcuts.auto_labeling_run"] = (
            self._ensure_hidden_shortcut_action(
                "auto_labeling_run",
                self._widget.tr("Auto Labeling Run"),
                lambda: self._trigger_auto_labeling_button("button_run"),
            )
        )
        shortcut_map["shortcuts.auto_labeling_clear"] = (
            self._ensure_hidden_shortcut_action(
                "auto_labeling_clear",
                self._widget.tr("Auto Labeling Clear"),
                lambda: self._trigger_auto_labeling_button("button_clear"),
            )
        )
        shortcut_map["shortcuts.auto_labeling_finish_object"] = (
            self._ensure_hidden_shortcut_action(
                "auto_labeling_finish_object",
                self._widget.tr("Auto Labeling Finish Object"),
                lambda: self._trigger_auto_labeling_button(
                    "button_finish_object"
                ),
            )
        )
        self._shortcut_action_map = shortcut_map
        for key, action in shortcut_map.items():
            short_key = key.split(".", 1)[1]
            value = self._widget._config.get("shortcuts", {}).get(short_key)
            self._set_action_shortcut(action, value)
        self.update_zoom_shortcut_hint()

    def update_zoom_shortcut_hint(self) -> None:
        zoom_in = self.shortcut_value_to_text(
            self._widget._config.get("shortcuts", {}).get("zoom_in")
        )
        zoom_out = self.shortcut_value_to_text(
            self._widget._config.get("shortcuts", {}).get("zoom_out")
        )
        combo = ",".join([v for v in (zoom_in, zoom_out) if v])
        self._widget.zoom_widget.setWhatsThis(
            str(
                self._widget.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmt_shortcut(combo or "-"),
                utils.fmt_shortcut(self._widget.tr("Ctrl+Wheel")),
            )
        )

    def apply_change(self, key: str, value: Any) -> None:
        if key in {
            "canvas.epsilon",
            "canvas.double_click",
            "canvas.double_click_edit_label",
            "canvas.num_backups",
        }:
            self.apply_canvas_basic()
            return
        if key.startswith("canvas.wheel_rectangle_editing."):
            self.apply_canvas_wheel_edit()
            return
        if key.startswith("canvas.crosshair."):
            self.apply_canvas_crosshair()
            return
        if key == "canvas.brush.point_distance":
            self.apply_canvas_brush()
            return
        if key.startswith("canvas.attributes."):
            self.apply_canvas_attributes()
            return
        if key.startswith("canvas.rotation."):
            self.apply_canvas_rotation()
            return
        if key.startswith("canvas.cuboid."):
            self.apply_canvas_cuboid()
            return
        if key == "canvas.mask.opacity":
            self.apply_canvas_mask()
            return
        if key == "shift_auto_shape_color":
            self._widget._runtime_shape_color_shift = int(
                self._widget._config.get("shift_auto_shape_color", 0)
            )
            return
        if key.startswith("shape.") or key in {
            "shape_color",
            "default_shape_color",
        }:
            self.apply_shape_style(key)
            return
        if (
            key.startswith("flag_dock.")
            or key.startswith("label_dock.")
            or key.startswith("shape_dock.")
            or key.startswith("description_dock.")
            or key.startswith("file_dock.")
        ):
            self.apply_dock_features()
            return
        if key in {
            "display_label_popup",
            "auto_highlight_shape",
            "auto_switch_to_edit_mode",
            "exif_scan_enabled",
            "switch_to_checked",
            "file_list_checkbox_editable",
            "system_clipboard",
        }:
            self.apply_behavior_flags(key)
            return
        if key in {
            "show_label_text_field",
            "label_completion",
            "fit_to_content.column",
            "fit_to_content.row",
            "sort_labels",
            "validate_label",
            "move_mode",
        }:
            self.apply_label_dialog_runtime(key)
            return
        if key in {
            "model_hub",
            "device",
            "logger_level",
            "remote_server_settings.timeout",
            "training.ultralytics.project_readonly",
            "file_search",
        }:
            self.apply_runtime_advanced(key, value)
            return
        if key.startswith("shortcuts."):
            self.apply_shortcuts(key, value)
            return
        logger.debug("No runtime handler for settings key: %s", key)

    def apply_canvas_basic(self) -> None:
        self._widget.canvas.epsilon = float(
            self._widget._config["canvas"]["epsilon"]
        )
        self._widget.canvas.double_click = self._widget._config["canvas"][
            "double_click"
        ]
        self._widget.canvas.double_click_edit_label = self._widget._config[
            "canvas"
        ]["double_click_edit_label"]
        self._widget.canvas.num_backups = int(
            self._widget._config["canvas"]["num_backups"]
        )
        self._widget.canvas.update()

    def apply_canvas_wheel_edit(self) -> None:
        wheel_config = self._widget._config["canvas"][
            "wheel_rectangle_editing"
        ]
        self._widget.canvas.wheel_rectangle_editing = wheel_config
        self._widget.canvas.enable_wheel_rectangle_editing = wheel_config[
            "enable"
        ]
        self._widget.canvas.rect_adjust_step = float(
            wheel_config["adjust_step"]
        )
        self._widget.canvas.rect_scale_step = float(wheel_config["scale_step"])

    def apply_canvas_crosshair(self) -> None:
        crosshair = self._widget._config["canvas"]["crosshair"]
        self._widget.canvas.set_cross_line(
            bool(crosshair["show"]),
            float(crosshair["width"]),
            str(crosshair["color"]),
            float(crosshair["opacity"]),
        )
        self._widget.crosshair_settings = dict(crosshair)

    def apply_canvas_brush(self) -> None:
        brush = self._widget._config["canvas"]["brush"]
        self._widget.canvas.brush_point_distance = float(
            brush["point_distance"]
        )

    def apply_canvas_attributes(self) -> None:
        attrs = self._widget._config["canvas"]["attributes"]
        self._widget.canvas.attr_background_color = list(
            attrs["background_color"]
        )
        self._widget.canvas.attr_border_color = list(attrs["border_color"])
        self._widget.canvas.attr_text_color = list(attrs["text_color"])
        self._widget.canvas.update()

    def apply_canvas_rotation(self) -> None:
        rotation = self._widget._config["canvas"]["rotation"]
        self._widget.canvas.large_rotation_increment = math.radians(
            float(rotation["large_increment"])
        )
        self._widget.canvas.small_rotation_increment = math.radians(
            float(rotation["small_increment"])
        )

    def apply_canvas_cuboid(self) -> None:
        cuboid = self._widget._config["canvas"]["cuboid"]
        default_depth_vector = cuboid["default_depth_vector"]
        self._widget.canvas.cuboid_default_depth_vector = [
            float(default_depth_vector[0]),
            float(default_depth_vector[1]),
        ]
        self._widget.canvas.cuboid_min_depth = float(cuboid["min_depth"])

    def apply_canvas_mask(self) -> None:
        self._widget.canvas.mask_opacity = int(
            self._widget._config["canvas"]["mask"]["opacity"]
        )
        self._widget.canvas.update()

    def apply_shape_style(self, key: str) -> None:
        shape_config = self._widget._config["shape"]
        Shape.line_color = QtGui.QColor(*shape_config["line_color"])
        Shape.fill_color = QtGui.QColor(*shape_config["fill_color"])
        Shape.vertex_fill_color = QtGui.QColor(
            *shape_config["vertex_fill_color"]
        )
        Shape.select_line_color = QtGui.QColor(
            *shape_config["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *shape_config["select_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *shape_config["hvertex_fill_color"]
        )
        Shape.point_size = int(shape_config["point_size"])
        Shape.line_width = float(shape_config["line_width"])

        strategy_keys = {"shape_color", "default_shape_color"}
        if key in strategy_keys:
            for label in self._widget.label_info:
                self._widget.label_info[label]["color"] = list(
                    self._widget._get_rgb_by_label(label, skip_label_info=True)
                )
            for shape in self._widget.canvas.shapes:
                self._widget._update_shape_color(shape)
            self._refresh_label_item_colors()
            self._widget.canvas.update()
            return

        color_key_map = {
            "shape.line_color": "line_color",
            "shape.fill_color": "fill_color",
            "shape.vertex_fill_color": "vertex_fill_color",
            "shape.select_line_color": "select_line_color",
            "shape.select_fill_color": "select_fill_color",
            "shape.hvertex_fill_color": "hvertex_fill_color",
        }
        if key in color_key_map:
            color_value = shape_config[key.split(".", 1)[1]]
            for shape in self._widget.canvas.shapes:
                setattr(shape, color_key_map[key], QtGui.QColor(*color_value))
        self._widget.canvas.update()

    def apply_dock_features(self) -> None:
        for dock_name in (
            "flag_dock",
            "label_dock",
            "shape_dock",
            "description_dock",
            "file_dock",
        ):
            features = QtWidgets.QDockWidget.DockWidgetFeature(0)
            dock_config = self._widget._config[dock_name]
            if dock_config["closable"]:
                features |= (
                    QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
                )
            if dock_config["floatable"]:
                features |= (
                    QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
                )
            if dock_config["movable"]:
                features |= (
                    QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                )
            getattr(self._widget, dock_name).setFeatures(features)

    def apply_behavior_flags(self, key: str) -> None:
        if key == "auto_highlight_shape":
            value = self._widget._config.get("auto_highlight_shape", False)
            self._widget.canvas.auto_highlight_shape = value
            self._widget.canvas.h_shape_is_hovered = value
        elif key == "auto_switch_to_edit_mode":
            self.set_auto_switch_to_edit_mode(
                self._widget._config.get("auto_switch_to_edit_mode", False)
            )
        elif key == "file_list_checkbox_editable":
            self._apply_file_list_checkbox_editable()
        elif key == "system_clipboard":
            self._widget.toggle_system_clipboard(
                self._widget._config.get("system_clipboard", False)
            )

    def apply_label_dialog_runtime(self, key: str) -> None:
        if key in {
            "show_label_text_field",
            "label_completion",
            "fit_to_content.column",
            "fit_to_content.row",
            "sort_labels",
        }:
            self._rebuild_label_dialog()

    def apply_runtime_advanced(self, key: str, value: Any) -> None:
        if key == "logger_level":
            logger.set_level(str(value).upper())
            return
        if key == "device":
            if value is None:
                device_manager.reset_device_preference()
                return
            device_manager.set_device(str(value))
            return
        if key == "remote_server_settings.timeout":
            timeout = int(
                self._widget._config["remote_server_settings"]["timeout"]
            )
            model_config = (
                self._widget.auto_labeling_widget.model_manager.loaded_model_config
            )
            if model_config:
                model_config["timeout"] = timeout
                model = model_config.get("model")
                if model is not None and hasattr(model, "timeout"):
                    model.timeout = timeout
            return
        if key == "file_search":
            value_text = "" if value is None else str(value)
            self._widget.file_search.setText(value_text)

    def apply_shortcuts(self, key: str, value: Any) -> None:
        action = self._shortcut_action_map.get(key)
        if action is None:
            return
        self._set_action_shortcut(action, value)
        if key in {"shortcuts.zoom_in", "shortcuts.zoom_out"}:
            self.update_zoom_shortcut_hint()
        self._widget.auto_labeling_widget.update_shortcut_button_texts(
            self._widget._config.get("shortcuts", {})
        )
        self._widget.label_instruction.setText(
            self._widget.get_labeling_instruction()
        )

    def _ensure_hidden_shortcut_action(
        self, name: str, text: str, callback
    ) -> QtGui.QAction:
        action = self._hidden_actions.get(name)
        if action is not None:
            return action
        action = QtGui.QAction(text, self._widget)
        action.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        action.triggered.connect(callback)
        self._widget.addAction(action)
        self._hidden_actions[name] = action
        return action

    def _trigger_auto_labeling_button(self, button_name: str) -> None:
        widget = getattr(self._widget, "auto_labeling_widget", None)
        if widget is None:
            return
        button = getattr(widget, button_name, None)
        if button is None or not button.isEnabled():
            return
        button.click()

    def _set_action_shortcut(self, action: QtGui.QAction, value: Any) -> None:
        if isinstance(value, (list, tuple)):
            shortcuts = [QtGui.QKeySequence(str(v)) for v in value if v]
            action.setShortcuts(shortcuts)
            return
        if value in (None, ""):
            action.setShortcuts([])
            action.setShortcut(QtGui.QKeySequence())
            return
        action.setShortcut(QtGui.QKeySequence(str(value)))

    def _refresh_label_item_colors(self) -> None:
        for item in self._widget.label_list:
            shape = item.shape()
            if shape is None:
                continue
            color = shape.fill_color.getRgb()[:3]
            item.setBackground(QtGui.QColor(*color, LABEL_OPACITY))
            self._widget.unique_label_list.update_item_color(
                shape.label, color, LABEL_OPACITY
            )

    def _apply_file_list_checkbox_editable(self) -> None:
        editable = self._widget._config.get(
            "file_list_checkbox_editable", False
        )
        for i in range(self._widget.file_list_widget.count()):
            item = self._widget.file_list_widget.item(i)
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if editable:
                flags |= Qt.ItemFlag.ItemIsUserCheckable
            item.setFlags(flags)

    def _rebuild_label_dialog(self) -> None:
        old_dialog = self._widget.label_dialog
        labels = [
            old_dialog.label_list.item(i).text()
            for i in range(old_dialog.label_list.count())
        ]
        last_label = old_dialog.get_last_label()
        last_gid = old_dialog.get_last_gid()
        self._widget.label_dialog = LabelDialog(
            parent=self._widget,
            labels=labels,
            sort_labels=self._widget._config["sort_labels"],
            show_text_field=self._widget._config["show_label_text_field"],
            completion=self._widget._config["label_completion"],
            fit_to_content=self._widget._config["fit_to_content"],
            flags=self._widget.label_flags,
        )
        self._widget.label_dialog._last_label = last_label
        self._widget.label_dialog._last_gid = last_gid
        old_dialog.deleteLater()

    def _quit_application(self) -> None:
        window = self._widget.window()
        if window is not None:
            window.close()
            return
        self._widget.close()
