from typing import Dict

from anylabeling.views.labeling.classifier.config import (
    BORDER_RADIUS,
    BUTTON_COLORS,
    FONT_SIZE_SMALL,
    FONT_SIZE_NORMAL,
    THEME,
)


def get_filename_label_style(theme: Dict[str, str] = None) -> str:
    theme = theme or THEME
    return f"""
        QLabel {{
            color: {theme["text"]};
            font-size: {FONT_SIZE_NORMAL};
            font-weight: 500;
            background-color: transparent;
            border: none;
        }}
    """


def get_image_label_style() -> str:
    return """
        QLabel {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 4px;
        }
    """


def get_image_container_style() -> str:
    return f"""
        QWidget {{
            background-color: transparent;
            border: none;
        }}
    """


def get_button_style(theme: Dict[str, str] = None) -> str:
    """Style for common buttion"""
    theme = theme or THEME
    return f"""
        QPushButton {{
            border: none;
            background: transparent;
        }}
        QPushButton:hover {{
            background-color: {theme["background_hover"]};
            border-radius: {BORDER_RADIUS};
        }}
    """


def get_dialog_button_style(
    variant: str = "primary", size: str = "medium"
) -> str:
    colors = BUTTON_COLORS.get(variant, BUTTON_COLORS["primary"])

    height = (
        "32px" if size == "medium" else "28px" if size == "small" else "36px"
    )
    font_size = FONT_SIZE_NORMAL if size == "medium" else FONT_SIZE_SMALL

    border = (
        f"border: 1px solid {colors.get('border', 'transparent')};"
        if variant in ["secondary", "outline"]
        else "border: none;"
    )

    return f"""
        QPushButton {{
            background-color: {colors["background"]};
            color: {colors["text"]};
            font-size: {font_size};
            font-weight: 500;
            {border}
            border-radius: {BORDER_RADIUS};
            padding: 0 16px;
            height: {height};
        }}
        QPushButton:hover {{
            background-color: {colors["hover"]};
        }}
        QPushButton:pressed {{
            background-color: {colors["pressed"]};
        }}
        QPushButton:disabled {{
            background-color: #f3f4f6;
            color: #9ca3af;
            border: 1px solid #e5e7eb;
        }}
    """


def get_main_splitter_style() -> str:
    return """
        QSplitter::handle {
            background-color: #e2e8f0;
            width: 3px;
            border-radius: 1px;
        }
        QSplitter::handle:hover {
            background-color: #cbd5e1;
        }
    """


def get_overlay_text_style() -> str:
    return """
        QLabel {
            background-color: rgba(0, 0, 0, 255);
            color: rgb(255, 255, 255);
            font-family: Arial;
            font-size: 12px;
            font-weight: bold;
            padding: 4px;
            border-radius: 4px;
        }
    """
