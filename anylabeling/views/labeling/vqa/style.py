from typing import Dict

from anylabeling.views.labeling.vqa.config import (
    BORDER_RADIUS,
    BUTTON_COLORS,
    FONT_SIZE_SMALL,
    FONT_SIZE_NORMAL,
    FONT_SIZE_LARGE,
    THEME,
)
from anylabeling.views.labeling.utils.qt import new_icon_path


def get_filename_label_style(theme: Dict[str, str] = None) -> str:
    """Style for filename label"""
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
    """Style for image display label"""
    return """
        QLabel {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 4px;
        }
    """


def get_image_container_style() -> str:
    """Style for image container widget"""
    return f"""
        QWidget {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: {BORDER_RADIUS};
        }}
    """


def get_page_label_style(theme: Dict[str, str] = None) -> str:
    """Style for page labels"""
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


def get_message_label_style(theme: Dict[str, str] = None) -> str:
    """Style for message labels"""
    theme = theme or THEME
    return f"""
        QLabel {{
            font-size: {FONT_SIZE_NORMAL};
            color: {theme["text"]};
            background: none;
            border: none;
        }}
    """


def get_title_label_style(theme: Dict[str, str] = None) -> str:
    """Style for title labels"""
    theme = theme or THEME
    return f"""
        QLabel {{
            font-size: {FONT_SIZE_LARGE};
            font-weight: 500;
            color: {theme["highlight_text"]};
            background: none;
            border: none;
        }}
    """


def get_status_label_style() -> str:
    """Style for status labels in dialogs"""
    return """
        QLabel {
            color: #718096;
            font-size: 12px;
            font-style: italic;
        }
    """


def get_ui_style(theme: Dict[str, str] = None) -> str:
    """Style for setup ui"""
    theme = theme or THEME
    return f"""
        QDialog {{
            background-color: {theme["background"]};
            border-radius: 0px;
        }}
    """


def get_main_splitter_style(theme: Dict[str, str] = None) -> str:
    theme = theme or THEME
    return f"""
        QSplitter::handle {{
            background-color: {theme["border"]};
        }}
    """


def get_component_dialog_combobox_style() -> str:
    """Style for combobox in component dialogs"""
    return """
        QTableWidget {
            gridline-color: #e2e8f0;
            background-color: #ffffff;
            alternate-background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            outline: none;
        }
        QTableWidget::item {
            padding: 6px 8px;
            border: none;
            min-height: 20px;
            outline: none;
        }
        QTableWidget::item:focus {
            outline: none;
            border: none;
        }
        QHeaderView::section {
            background-color: #f7fafc;
            padding: 8px;
            border: 0px;
            border-bottom: 2px solid #e2e8f0;
            border-right: 1px solid #e2e8f0;
            font-weight: 600;
            color: #4a5568;
            min-height: 25px;
            outline: none;
        }
        QHeaderView::section:last {
            border-right: none;
        }
    """


def get_content_input_style(theme: Dict[str, str] = None) -> str:
    theme = theme or THEME
    return f"""
        QTextEdit {{
            background-color: {theme["background"]};
            border-radius: {BORDER_RADIUS};
            padding: 8px 12px;
            font-size: {FONT_SIZE_NORMAL};
            border: 1px solid #e2e8f0;
        }}
        QTextEdit:focus {{
            border: 2px solid #60A5FA;
        }}
    """


def get_name_input_style(theme: Dict[str, str] = None) -> str:
    theme = theme or THEME
    return f"""
        QLineEdit {{
            background-color: {theme["background"]};
            border-radius: {BORDER_RADIUS};
            padding: 8px 12px;
            font-size: {FONT_SIZE_NORMAL};
            border: 1px solid #e2e8f0;
        }}
        QLineEdit:focus {{
            border: 2px solid #60A5FA;
        }}
    """


def get_page_input_style(theme: Dict[str, str] = None) -> str:
    """Style for page input field"""
    theme = theme or THEME
    return f"""
        QLineEdit {{
            border: 1px solid {theme["border"]};
            border-radius: {BORDER_RADIUS};
            padding: 6px 8px;
            background-color: {theme["background_secondary"]};
            color: #718096;
            font-size: {FONT_SIZE_NORMAL};
            font-weight: 500;
        }}
        QLineEdit:hover {{
            background-color: {theme["background_hover"]};
            border-color: {theme["border"]};
        }}
        QLineEdit:focus {{
            border: 2px solid {theme["primary"]};
            background-color: {theme["background_secondary"]};
            outline: none;
        }}
    """


def get_prompt_input_style(theme: Dict[str, str] = None) -> str:
    """Style for prompt input field"""
    theme = theme or THEME
    return f"""
        QTextEdit {{
            border: 1px solid #E5E7EB;
            border-radius: {BORDER_RADIUS};
            background-color: #F9FAFB;
            color: {theme["text"]};
            font-size: {FONT_SIZE_NORMAL};
            line-height: 1.5;
            padding: 12px;
        }}
        QTextEdit:focus {{
            border: 1px solid #6366F1;
        }}
        QScrollBar:vertical {{
            width: 8px;
            background: transparent;
        }}
        QScrollBar::handle:vertical {{
            background: #D1D5DB;
            border-radius: 4px;
            min-height: 30px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
    """


def get_table_style():
    return f"""
        QTableWidget {{
            border: 1px solid #E5E7EB;
            border-radius: 0px;
            background-color: white;
            gridline-color: transparent;
            outline: none;
        }}
        QTableWidget::item {{
            padding: 6px 12px;
            border: none;
            border-bottom: 1px solid #F3F4F6;
            color: #374151;
            font-size: 13px;
            outline: none;
        }}
        QTableWidget::item:hover {{
            background-color: #F9FAFB;
        }}
        QHeaderView::section {{
            background-color: #F8FAFC;
            color: #6B7280;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 8px 12px;
            border: none;
            border-bottom: 2px solid #E5E7EB;
            border-right: 1px solid #F3F4F6;
            outline: none;
            height: 28px;
        }}
        QCheckBox {{
            spacing: 6px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
            border: 1px solid #D2D2D7;
            background-color: white;
        }}
        QCheckBox::indicator:checked {{
            background-color: white;
            border: 1px solid #D2D2D7;
            image: url({new_icon_path("checkmark", "svg")});
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
    variant="primary", size="medium", disabled=False
) -> str:
    """
    Unified button style for dialogs

    Args:
        variant: Button variant - "primary", "secondary", "success", "danger"
        size: Button size - "small", "medium", "large"
        disabled: Whether the button is disabled
    """
    colors = BUTTON_COLORS.get(variant, BUTTON_COLORS["primary"])
    sizes = {
        "small": {
            "height": "28px",
            "padding": "0 8px",
            "font_size": FONT_SIZE_SMALL,
            "min_width": "50px",
        },
        "medium": {
            "height": "36px",
            "padding": "0 10px",
            "font_size": FONT_SIZE_NORMAL,
            "min_width": "70px",
        },
        "large": {
            "height": "44px",
            "padding": "0 12px",
            "font_size": FONT_SIZE_LARGE,
            "min_width": "90px",
        },
    }
    size_config = sizes.get(size, sizes["medium"])

    # Handle disabled state
    if disabled:
        style = f"""
            QPushButton {{
                background-color: #F3F4F6;
                color: #9CA3AF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                font-weight: 500;
                font-size: {size_config["font_size"]};
                height: {size_config["height"]};
                min-width: {size_config["min_width"]};
                padding: {size_config["padding"]};
            }}
        """
        return style

    # Base style for enabled buttons
    style = f"""
        QPushButton {{
            background-color: {colors["background"]};
            color: {colors["text"]};
            border: {"none" if "border" not in colors else f"1px solid {colors['border']}"};
            border-radius: 8px;
            font-weight: 500;
            font-size: {size_config["font_size"]};
            height: {size_config["height"]};
            min-width: {size_config["min_width"]};
            padding: {size_config["padding"]};
        }}
        QPushButton:hover {{
            background-color: {colors["hover"]};
        }}
        QPushButton:pressed {{
            background-color: {colors["pressed"]};
        }}
    """

    if variant == "secondary":
        style = style.replace(
            "border: none", f"border: 1px solid {colors['border']}"
        )

    return style
