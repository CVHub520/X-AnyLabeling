_EXPORTS = {
    "AboutDialog": (".about_dialog", "AboutDialog"),
    "AutoLabelingWidget": (".auto_labeling", "AutoLabelingWidget"),
    "BrightnessContrastDialog": (
        ".brightness_contrast_dialog",
        "BrightnessContrastDialog",
    ),
    "Canvas": (".canvas", "Canvas"),
    "ChatbotDialog": (".chatbot_dialog", "ChatbotDialog"),
    "ClassifierDialog": (".classifier_dialog", "ClassifierDialog"),
    "ColorDialog": (".color_dialog", "ColorDialog"),
    "CrosshairSettingsDialog": (
        ".crosshair_settings_dialog",
        "CrosshairSettingsDialog",
    ),
    "FileDialogPreview": (".file_dialog_preview", "FileDialogPreview"),
    "GroupIDFilterComboBox": (
        ".filter_label_widget",
        "GroupIDFilterComboBox",
    ),
    "LabelFilterComboBox": (
        ".filter_label_widget",
        "LabelFilterComboBox",
    ),
    "ShapeModifyDialog": (".shape_dialog", "ShapeModifyDialog"),
    "DigitShortcutDialog": (".label_dialog", "DigitShortcutDialog"),
    "GroupIDModifyDialog": (".label_dialog", "GroupIDModifyDialog"),
    "LabelDialog": (".label_dialog", "LabelDialog"),
    "LabelModifyDialog": (".label_dialog", "LabelModifyDialog"),
    "LabelQLineEdit": (".label_dialog", "LabelQLineEdit"),
    "LabelListWidget": (".label_list_widget", "LabelListWidget"),
    "LabelListWidgetItem": (".label_list_widget", "LabelListWidgetItem"),
    "SearchBar": (".model_dropdown_widget", "SearchBar"),
    "NavigatorDialog": (".navigator_widget", "NavigatorDialog"),
    "OverviewDialog": (".overview_dialog", "OverviewDialog"),
    "PolygonSidesDialog": (".polygon_sides_dialog", "PolygonSidesDialog"),
    "Popup": (".popup", "Popup"),
    "ToolBar": (".toolbar", "ToolBar"),
    "UniqueLabelQListWidget": (
        ".unique_label_qlist_widget",
        "UniqueLabelQListWidget",
    ),
    "VQADialog": (".vqa_dialog", "VQADialog"),
    "ZoomWidget": (".zoom_widget", "ZoomWidget"),
    "CompareViewManager": (".compare_view", "CompareViewManager"),
    "CompareViewSlider": (".compare_view", "CompareViewSlider"),
}


def __getattr__(name: str):
    if name in _EXPORTS:
        module_path, attr = _EXPORTS[name]
        from importlib import import_module

        module = import_module(module_path, __name__)
        return getattr(module, attr)
    raise AttributeError(name)


__all__ = tuple(_EXPORTS.keys())
