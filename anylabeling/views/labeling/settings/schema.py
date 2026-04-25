from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.resources as pkg_resources
from typing import Any

import yaml

try:
    from PyQt6.QtCore import QCoreApplication, QT_TRANSLATE_NOOP
except Exception:

    class QCoreApplication:
        @staticmethod
        def translate(_context: str, text: str) -> str:
            return text

    def QT_TRANSLATE_NOOP(_context: str, text: str) -> str:
        return text


from anylabeling import configs as anylabeling_configs

SETTINGS_TRANSLATION_CONTEXT = "SettingsDialog"

SETTINGS_PRIMARY_ORDER = (
    QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Shortcuts"),
    QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "General"),
    QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Shape"),
    QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Canvas"),
)
SETTINGS_SHORTCUT_SECTIONS = (
    QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "AI"),
    QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Dialog"),
    QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "File"),
    QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Shape"),
    QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "View"),
)


def _settings_translation_markers() -> None:
    QCoreApplication.translate("SettingsDialog", "Shortcuts")
    QCoreApplication.translate("SettingsDialog", "General")
    QCoreApplication.translate("SettingsDialog", "Shape")
    QCoreApplication.translate("SettingsDialog", "Canvas")
    QCoreApplication.translate("SettingsDialog", "AI")
    QCoreApplication.translate("SettingsDialog", "Dialog")
    QCoreApplication.translate("SettingsDialog", "File")
    QCoreApplication.translate("SettingsDialog", "View")
    QCoreApplication.translate("SettingsDialog", "Display Label Popup")
    QCoreApplication.translate("SettingsDialog", "Auto Highlight Shape")
    QCoreApplication.translate(
        "SettingsDialog",
        "In edit mode, automatically highlight vertices of selected objects.",
    )
    QCoreApplication.translate("SettingsDialog", "Auto Switch To Edit Mode")
    QCoreApplication.translate(
        "SettingsDialog",
        "Automatically switch selected objects into edit mode.",
    )
    QCoreApplication.translate("SettingsDialog", "Enable EXIF Scan")
    QCoreApplication.translate(
        "SettingsDialog",
        "Scan EXIF metadata when loading directories; this adds overhead.",
    )
    QCoreApplication.translate("SettingsDialog", "Toggle Annotation Checked")
    QCoreApplication.translate("SettingsDialog", "File List Checkbox Editable")
    QCoreApplication.translate("SettingsDialog", "Use System Clipboard")
    QCoreApplication.translate("SettingsDialog", "Shape Color Strategy")
    QCoreApplication.translate("SettingsDialog", "Default Shape Color")
    QCoreApplication.translate("SettingsDialog", "Auto Color Shift")
    QCoreApplication.translate("SettingsDialog", "Line Color")
    QCoreApplication.translate("SettingsDialog", "Fill Color")
    QCoreApplication.translate("SettingsDialog", "Vertex Fill Color")
    QCoreApplication.translate("SettingsDialog", "Select Line Color")
    QCoreApplication.translate("SettingsDialog", "Select Fill Color")
    QCoreApplication.translate("SettingsDialog", "Hover Vertex Fill Color")
    QCoreApplication.translate("SettingsDialog", "Point Size")
    QCoreApplication.translate("SettingsDialog", "Line Width")
    QCoreApplication.translate("SettingsDialog", "Selection Epsilon")
    QCoreApplication.translate(
        "SettingsDialog",
        "Distance threshold in pixels for selecting nearby vertices or edges.",
    )
    QCoreApplication.translate("SettingsDialog", "Double Click")
    QCoreApplication.translate(
        "SettingsDialog",
        "Set to 'close' to finish the current shape with a double click.",
    )
    QCoreApplication.translate("SettingsDialog", "Double Click Edit Label")
    QCoreApplication.translate("SettingsDialog", "Undo Backups")
    QCoreApplication.translate(
        "SettingsDialog", "Enable Wheel Rectangle Editing"
    )
    QCoreApplication.translate("SettingsDialog", "Adjust Step")
    QCoreApplication.translate("SettingsDialog", "Scale Step")
    QCoreApplication.translate("SettingsDialog", "Show Crosshair")
    QCoreApplication.translate("SettingsDialog", "Crosshair Width")
    QCoreApplication.translate("SettingsDialog", "Crosshair Color")
    QCoreApplication.translate("SettingsDialog", "Crosshair Opacity")
    QCoreApplication.translate("SettingsDialog", "Background Color")
    QCoreApplication.translate("SettingsDialog", "Border Color")
    QCoreApplication.translate("SettingsDialog", "Text Color")
    QCoreApplication.translate("SettingsDialog", "Large Increment")
    QCoreApplication.translate("SettingsDialog", "Small Increment")
    QCoreApplication.translate("SettingsDialog", "Brush Point Distance")
    QCoreApplication.translate("SettingsDialog", "Default Depth Vector")
    QCoreApplication.translate("SettingsDialog", "Min Depth")
    QCoreApplication.translate("SettingsDialog", "Mask Opacity")
    QCoreApplication.translate("SettingsDialog", "Model Hub")
    QCoreApplication.translate("SettingsDialog", "Model download source.")
    QCoreApplication.translate("SettingsDialog", "Logger Level")
    QCoreApplication.translate("SettingsDialog", "Qt Image Allocation Limit")
    QCoreApplication.translate(
        "SettingsDialog",
        "Qt default is 256 MB. Use 0 to disable the limit.",
    )


SETTINGS_GENERAL_KEYS = (
    "auto_highlight_shape",
    "auto_switch_to_edit_mode",
    "exif_scan_enabled",
    "file_list_checkbox_editable",
    "system_clipboard",
    "model_hub",
    "logger_level",
    "qt_image_allocation_limit",
)

SETTINGS_SHAPE_KEYS = (
    "shift_auto_shape_color",
    "shape.line_color",
    "shape.fill_color",
    "shape.vertex_fill_color",
    "shape.select_line_color",
    "shape.select_fill_color",
    "shape.hvertex_fill_color",
    "shape.point_size",
    "shape.line_width",
)

SETTINGS_SHORTCUT_KEYS_CORE = (
    "shortcuts.open",
    "shortcuts.open_dir",
    "shortcuts.open_video",
    "shortcuts.save",
    "shortcuts.save_as",
    "shortcuts.close",
    "shortcuts.delete_file",
    "shortcuts.open_next",
    "shortcuts.open_prev",
    "shortcuts.open_next_unchecked",
    "shortcuts.open_prev_unchecked",
    "shortcuts.zoom_in",
    "shortcuts.zoom_out",
    "shortcuts.zoom_to_original",
    "shortcuts.fit_window",
    "shortcuts.fit_width",
    "shortcuts.show_navigator",
    "shortcuts.create_polygon",
    "shortcuts.create_rectangle",
    "shortcuts.edit_polygon",
    "shortcuts.delete_polygon",
    "shortcuts.copy_polygon",
    "shortcuts.paste_polygon",
    "shortcuts.undo",
)

SHORTCUT_DUPLICATE_WHITELIST = {
    frozenset({"shortcuts.undo", "shortcuts.undo_last_point"}),
}

EXCLUDED_KEYS = frozenset(
    {
        "language",
        "theme",
        "auto_save",
        "store_data",
        "keep_prev",
        "keep_prev_scale",
        "keep_prev_brightness",
        "keep_prev_contrast",
        "auto_use_last_label",
        "auto_use_last_gid",
        "show_groups",
        "show_masks",
        "show_texts",
        "show_labels",
        "show_scores",
        "show_degrees",
        "show_shapes",
        "show_linking",
        "show_attributes",
        "flags",
        "label_flags",
        "labels",
        "label_colors",
        "flag_dock.show",
        "label_dock.show",
        "shape_dock.show",
        "description_dock.show",
        "file_dock.show",
        "custom_models",
        "remote_server_settings.server_url",
        "remote_server_settings.api_key",
        "digit_shortcuts",
    }
)


@dataclass(frozen=True)
class SettingField:
    key: str
    label: str
    control: str
    primary: str
    secondary: str
    group: str
    options: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    decimals: int = 0
    allow_none: bool = False
    channels: int = 0
    description: str | None = None


@lru_cache(maxsize=1)
def load_template_config() -> dict[str, Any]:
    with pkg_resources.open_text(
        anylabeling_configs, "xanylabeling_config.yaml"
    ) as f:
        return yaml.safe_load(f)


def get_nested_value(data: dict[str, Any], key_path: str) -> Any:
    current = data
    for part in key_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(key_path)
        current = current[part]
    return current


def set_nested_value(data: dict[str, Any], key_path: str, value: Any) -> None:
    parts = key_path.split(".")
    current = data
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _shortcut_label(short_key: str) -> str:
    label_overrides = {
        "open_classifier": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Open Classifier Dialog"
        ),
        "open_chatbot": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Open Chatbot Dialog"
        ),
        "open_paddleocr": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Open PaddleOCR Dialog"
        ),
        "edit_digit_shortcut": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT,
            "Open Digit Shortcut Manager Dialog",
        ),
        "edit_group_id": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Open Group ID Manager"
        ),
        "edit_labels": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Open Label Manager Dialog"
        ),
        "show_navigator": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Show Navigator Dialog"
        ),
        "show_overview": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Open Overview Dialog"
        ),
        "edit_shapes": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Open Shape Manager Dialog"
        ),
        "open_settings": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Open Settings Dialog"
        ),
        "open_vqa": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Open VQA Dialog"
        ),
        "open_next": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Switch Next Image"
        ),
        "open_next_unchecked": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Switch Next Unchecked Image"
        ),
        "open_prev": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Switch Prev Image"
        ),
        "open_prev_unchecked": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Switch Prev Unchecked Image"
        ),
        "toggle_annotation_checked": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Toggle Annotation Checked"
        ),
        "auto_labeling_add_point": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Add Point"
        ),
        "auto_labeling_clear": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Clear"
        ),
        "auto_labeling_finish_object": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Finish Object"
        ),
        "auto_labeling_remove_point": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Remove Point"
        ),
        "auto_labeling_run": QT_TRANSLATE_NOOP(
            SETTINGS_TRANSLATION_CONTEXT, "Run"
        ),
    }
    if short_key in label_overrides:
        return label_overrides[short_key]
    chunks = short_key.split("_")
    return " ".join(chunk.capitalize() for chunk in chunks)


def _non_shortcut_fields() -> list[SettingField]:
    return [
        SettingField(
            "display_label_popup",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Display Label Popup"
            ),
            "bool",
            "General",
            "Behavior",
            "Basic",
        ),
        SettingField(
            "auto_highlight_shape",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Auto Highlight Shape"
            ),
            "bool",
            "General",
            "Behavior",
            "Basic",
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "In edit mode, automatically highlight vertices of selected objects.",
            ),
        ),
        SettingField(
            "auto_switch_to_edit_mode",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Auto Switch To Edit Mode"
            ),
            "bool",
            "General",
            "Behavior",
            "Basic",
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Automatically switch selected objects into edit mode.",
            ),
        ),
        SettingField(
            "exif_scan_enabled",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Enable EXIF Scan"
            ),
            "bool",
            "General",
            "Behavior",
            "Basic",
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Scan EXIF metadata when loading directories; this adds overhead.",
            ),
        ),
        SettingField(
            "file_list_checkbox_editable",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "File List Checkbox Editable",
            ),
            "bool",
            "General",
            "File List",
            "Navigation",
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Allow checked state changes directly from the file list.",
            ),
        ),
        SettingField(
            "system_clipboard",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Use System Clipboard"
            ),
            "bool",
            "General",
            "Behavior",
            "Basic",
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Use the operating system clipboard for copy and paste actions.",
            ),
        ),
        SettingField(
            "sort_labels",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Sort Labels"),
            "bool",
            "General",
            "Label Dialog",
            "Behavior",
        ),
        SettingField(
            "validate_label",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Validate Label"),
            "enum",
            "General",
            "Label Dialog",
            "Validation",
            options=(None, "exact"),
            allow_none=True,
        ),
        SettingField(
            "show_label_text_field",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Show Label Text Field"
            ),
            "bool",
            "General",
            "Label Dialog",
            "Layout",
        ),
        SettingField(
            "label_completion",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Label Completion"
            ),
            "enum",
            "General",
            "Label Dialog",
            "Behavior",
            options=("startswith", "contains"),
        ),
        SettingField(
            "move_mode",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Move Mode"),
            "enum",
            "General",
            "Label Dialog",
            "Behavior",
            options=("auto", "center"),
        ),
        SettingField(
            "fit_to_content.column",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Fit To Content Column"
            ),
            "bool",
            "General",
            "Label Dialog",
            "Layout",
        ),
        SettingField(
            "fit_to_content.row",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Fit To Content Row"
            ),
            "bool",
            "General",
            "Label Dialog",
            "Layout",
        ),
        SettingField(
            "shape_color",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Shape Color Strategy"
            ),
            "enum",
            "Shape",
            "Color Strategy",
            "Mode",
            options=(None, "auto", "manual"),
            allow_none=True,
        ),
        SettingField(
            "default_shape_color",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Default Shape Color"
            ),
            "color",
            "Shape",
            "Color Strategy",
            "Mode",
            channels=3,
        ),
        SettingField(
            "shift_auto_shape_color",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Auto Color Shift"
            ),
            "int",
            "Shape",
            "Color Strategy",
            "Mode",
            minimum=-1000,
            maximum=1000,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Shift the generated color index when automatic coloring is enabled.",
            ),
        ),
        SettingField(
            "shape.line_color",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Line Color"),
            "color",
            "Shape",
            "Drawing Style",
            "Color",
            channels=4,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the default outline color for shapes.",
            ),
        ),
        SettingField(
            "shape.fill_color",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Fill Color"),
            "color",
            "Shape",
            "Drawing Style",
            "Color",
            channels=4,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the default fill color for shapes.",
            ),
        ),
        SettingField(
            "shape.vertex_fill_color",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Vertex Fill Color"
            ),
            "color",
            "Shape",
            "Drawing Style",
            "Color",
            channels=4,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the default fill color for shape vertices.",
            ),
        ),
        SettingField(
            "shape.select_line_color",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Select Line Color"
            ),
            "color",
            "Shape",
            "Selection Style",
            "Color",
            channels=4,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the outline color for selected shapes.",
            ),
        ),
        SettingField(
            "shape.select_fill_color",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Select Fill Color"
            ),
            "color",
            "Shape",
            "Selection Style",
            "Color",
            channels=4,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the fill color for selected shapes.",
            ),
        ),
        SettingField(
            "shape.hvertex_fill_color",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Hover Vertex Fill Color"
            ),
            "color",
            "Shape",
            "Selection Style",
            "Color",
            channels=4,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the highlight color for hovered vertices.",
            ),
        ),
        SettingField(
            "shape.point_size",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Point Size"),
            "int",
            "Shape",
            "Geometry",
            "Basic",
            minimum=1,
            maximum=50,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Control the displayed size of shape vertices.",
            ),
        ),
        SettingField(
            "shape.line_width",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Line Width"),
            "int",
            "Shape",
            "Geometry",
            "Basic",
            minimum=1,
            maximum=20,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Control the default stroke width for shapes.",
            ),
        ),
        SettingField(
            "canvas.epsilon",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Selection Epsilon"
            ),
            "float",
            "Canvas",
            "Interaction",
            "Basic Interaction",
            minimum=0.1,
            maximum=100.0,
            decimals=1,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Distance threshold in pixels for selecting nearby vertices or edges.",
            ),
        ),
        SettingField(
            "canvas.double_click",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Double Click"),
            "enum",
            "Canvas",
            "Interaction",
            "Basic Interaction",
            options=(None, "close"),
            allow_none=True,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set to 'close' to finish the current shape with a double click.",
            ),
        ),
        SettingField(
            "canvas.double_click_edit_label",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Double Click Edit Label"
            ),
            "bool",
            "Canvas",
            "Interaction",
            "Basic Interaction",
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Open label editing when a shape is double-clicked in edit mode.",
            ),
        ),
        SettingField(
            "canvas.num_backups",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Undo Backups"),
            "int",
            "Canvas",
            "Interaction",
            "Undo/Backup",
            minimum=0,
            maximum=200,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set how many undo history snapshots are kept in memory.",
            ),
        ),
        SettingField(
            "canvas.wheel_rectangle_editing.enable",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Enable Wheel Rectangle Editing",
            ),
            "bool",
            "Canvas",
            "Wheel Editing",
            "Basic",
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Use the mouse wheel to adjust rectangle geometry while editing.",
            ),
        ),
        SettingField(
            "canvas.wheel_rectangle_editing.adjust_step",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Adjust Step"),
            "float",
            "Canvas",
            "Wheel Editing",
            "Step",
            minimum=0.1,
            maximum=20.0,
            decimals=2,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the step size for wheel-based rectangle adjustments.",
            ),
        ),
        SettingField(
            "canvas.wheel_rectangle_editing.scale_step",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Scale Step"),
            "float",
            "Canvas",
            "Wheel Editing",
            "Step",
            minimum=0.01,
            maximum=1.0,
            decimals=3,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the scale ratio applied by each wheel adjustment.",
            ),
        ),
        SettingField(
            "canvas.crosshair.show",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Show Crosshair"),
            "bool",
            "Canvas",
            "Interaction",
            "Crosshair",
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Show crosshair guides on the canvas.",
            ),
        ),
        SettingField(
            "canvas.crosshair.width",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Crosshair Width"),
            "float",
            "Canvas",
            "Interaction",
            "Crosshair",
            minimum=1.0,
            maximum=10.0,
            decimals=1,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the stroke width of the crosshair guides.",
            ),
        ),
        SettingField(
            "canvas.crosshair.color",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Crosshair Color"),
            "str",
            "Canvas",
            "Interaction",
            "Crosshair",
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the display color of the crosshair guides.",
            ),
        ),
        SettingField(
            "canvas.crosshair.opacity",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Crosshair Opacity"
            ),
            "float",
            "Canvas",
            "Interaction",
            "Crosshair",
            minimum=0.0,
            maximum=1.0,
            decimals=2,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the opacity of the crosshair guides.",
            ),
        ),
        SettingField(
            "canvas.attributes.background_color",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Background Color"
            ),
            "color",
            "Canvas",
            "Attributes Overlay",
            "Color",
            channels=4,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the background color of attribute overlays.",
            ),
        ),
        SettingField(
            "canvas.attributes.border_color",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Border Color"),
            "color",
            "Canvas",
            "Attributes Overlay",
            "Color",
            channels=4,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the border color of attribute overlays.",
            ),
        ),
        SettingField(
            "canvas.attributes.text_color",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Text Color"),
            "color",
            "Canvas",
            "Attributes Overlay",
            "Color",
            channels=4,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the text color of attribute overlays.",
            ),
        ),
        SettingField(
            "canvas.rotation.large_increment",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Large Increment"),
            "float",
            "Canvas",
            "Rotation",
            "Increment",
            minimum=0.01,
            maximum=45.0,
            decimals=2,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the larger step used for rotation adjustments.",
            ),
        ),
        SettingField(
            "canvas.rotation.small_increment",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Small Increment"),
            "float",
            "Canvas",
            "Rotation",
            "Increment",
            minimum=0.01,
            maximum=10.0,
            decimals=2,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the smaller step used for rotation adjustments.",
            ),
        ),
        SettingField(
            "canvas.brush.point_distance",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Brush Point Distance"
            ),
            "float",
            "Canvas",
            "Interaction",
            "Brush",
            minimum=1.0,
            maximum=200.0,
            decimals=1,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the spacing between sampled brush points.",
            ),
        ),
        SettingField(
            "canvas.cuboid.default_depth_vector",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Default Depth Vector"
            ),
            "vector2",
            "Canvas",
            "Cuboid",
            "Geometry",
            decimals=2,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the default depth direction for newly created cuboids.",
            ),
        ),
        SettingField(
            "canvas.cuboid.min_depth",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Min Depth"),
            "float",
            "Canvas",
            "Cuboid",
            "Geometry",
            minimum=1.0,
            maximum=200.0,
            decimals=2,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the minimum depth allowed for cuboid shapes.",
            ),
        ),
        SettingField(
            "canvas.mask.opacity",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Mask Opacity"),
            "int",
            "Canvas",
            "Mask",
            "Rendering",
            minimum=0,
            maximum=255,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the opacity used when rendering masks.",
            ),
        ),
        SettingField(
            "model_hub",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Model Hub"),
            "enum",
            "General",
            "Behavior",
            "Runtime",
            options=("github", "modelscope"),
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Model download source."
            ),
        ),
        SettingField(
            "logger_level",
            QT_TRANSLATE_NOOP(SETTINGS_TRANSLATION_CONTEXT, "Logger Level"),
            "enum",
            "General",
            "Behavior",
            "Runtime",
            options=("debug", "info", "warning", "error", "fatal"),
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Set the minimum log level shown in the application.",
            ),
        ),
        SettingField(
            "qt_image_allocation_limit",
            QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT, "Qt Image Allocation Limit"
            ),
            "int",
            "General",
            "Behavior",
            "Startup",
            minimum=0,
            maximum=16384,
            allow_none=True,
            description=QT_TRANSLATE_NOOP(
                SETTINGS_TRANSLATION_CONTEXT,
                "Qt default is 256 MB. Use 0 to disable the limit.",
            ),
        ),
    ]


def _shortcut_category_map() -> dict[str, tuple[str, ...]]:
    return {
        "AI": (
            "auto_run",
            "auto_label",
            "auto_labeling_add_point",
            "auto_labeling_clear",
            "auto_labeling_finish_object",
            "auto_labeling_remove_point",
            "auto_labeling_run",
        ),
        "Dialog": (
            "open_settings",
            "open_chatbot",
            "open_vqa",
            "open_classifier",
            "open_paddleocr",
            "show_overview",
            "show_navigator",
            "edit_digit_shortcut",
            "edit_group_id",
            "edit_labels",
            "edit_shapes",
        ),
        "File": (
            "close",
            "delete_file",
            "delete_image_file",
            "open",
            "open_dir",
            "open_video",
            "quit",
            "save",
            "save_as",
            "save_to",
            "open_next",
            "open_next_unchecked",
            "open_prev",
            "open_prev_unchecked",
            "toggle_annotation_checked",
        ),
        "Shape": (
            "add_point_to_edge",
            "copy_polygon",
            "create_brush_polygon",
            "create_circle",
            "create_cuboid",
            "create_line",
            "create_linestrip",
            "create_point",
            "create_polygon",
            "create_quadrilateral",
            "create_rectangle",
            "create_rotation",
            "delete_polygon",
            "duplicate_polygon",
            "edit_label",
            "edit_polygon",
            "group_selected_shapes",
            "ungroup_selected_shapes",
            "loop_thru_labels",
            "loop_select_labels",
            "hide_selected_polygons",
            "paste_polygon",
            "remove_selected_point",
            "show_hidden_polygons",
            "undo",
            "undo_last_point",
            "union_selected_shapes",
        ),
        "View": (
            "fit_width",
            "fit_window",
            "show_attributes",
            "show_labels",
            "show_linking",
            "show_masks",
            "show_texts",
            "toggle_auto_use_last_gid",
            "toggle_auto_use_last_label",
            "toggle_compare_view",
            "toggle_keep_prev_mode",
            "toggle_visibility_shapes",
            "zoom_in",
            "zoom_out",
            "zoom_to_original",
        ),
    }


def _shortcut_fields() -> list[SettingField]:
    shortcuts = load_template_config().get("shortcuts", {})
    category_map = _shortcut_category_map()
    fields: list[SettingField] = []
    consumed: set[str] = set()
    for secondary in SETTINGS_SHORTCUT_SECTIONS:
        for short_key in category_map.get(secondary, ()):
            full_key = f"shortcuts.{short_key}"
            if short_key not in shortcuts:
                continue
            consumed.add(short_key)
            control = (
                "multi_shortcut" if short_key == "zoom_in" else "shortcut"
            )
            fields.append(
                SettingField(
                    key=full_key,
                    label=_shortcut_label(short_key),
                    control=control,
                    primary="Shortcuts",
                    secondary=secondary,
                    group="Binding",
                    allow_none=True,
                )
            )
    for short_key in shortcuts.keys():
        if short_key in consumed:
            continue
        full_key = f"shortcuts.{short_key}"
        fields.append(
            SettingField(
                key=full_key,
                label=_shortcut_label(short_key),
                control="shortcut",
                primary="Shortcuts",
                secondary="View",
                group="Binding",
                allow_none=True,
            )
        )
    return fields


def _build_fields() -> tuple[SettingField, ...]:
    fields = _non_shortcut_fields() + _shortcut_fields()
    return tuple(fields)


SETTING_FIELDS = _build_fields()
SETTING_FIELD_MAP = {field.key: field for field in SETTING_FIELDS}
SETTINGS_KEYS = frozenset(SETTING_FIELD_MAP.keys())


def fields_for_page(primary: str, secondary: str) -> list[SettingField]:
    return [
        field
        for field in SETTING_FIELDS
        if field.primary == primary and field.secondary == secondary
    ]


def fields_for_primary(primary: str) -> list[SettingField]:
    if primary == "General":
        return [SETTING_FIELD_MAP[key] for key in SETTINGS_GENERAL_KEYS]
    if primary == "Shape":
        return [SETTING_FIELD_MAP[key] for key in SETTINGS_SHAPE_KEYS]
    if primary == "Shortcuts":
        return [
            field for field in SETTING_FIELDS if field.primary == "Shortcuts"
        ]
    if primary == "Canvas":
        return [field for field in SETTING_FIELDS if field.primary == "Canvas"]
    return []


def defaults_map() -> dict[str, Any]:
    template = load_template_config()
    return {key: get_nested_value(template, key) for key in SETTINGS_KEYS}
