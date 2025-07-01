from typing import Dict

from anylabeling.views.labeling.vqa.config import (
    BORDER_RADIUS,
    FONT_SIZE_NORMAL,
    THEME,
)


def get_image_container_style() -> str:
    """Style for image container widget"""
    return f"""
        QWidget {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: {BORDER_RADIUS};
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


def get_status_label_style() -> str:
    """Style for status labels in dialogs"""
    return """
        QLabel {
            color: #718096;
            font-size: 12px;
            font-style: italic;
        }
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


def get_primary_button_style() -> str:
    """Style for primary action buttons"""
    return f"""
        QPushButton {{
            background-color: #4299e1;
            color: #f9fbfd;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            font-size: {FONT_SIZE_NORMAL};
            height: 28px;
            padding: 0 12px;
        }}
        QPushButton:hover {{
            background-color: #3182ce;
        }}
        QPushButton:pressed {{
            background-color: #2c5282;
        }}
    """


def get_cancel_button_style(theme: Dict[str, str] = None) -> str:
    """Style for cancel button"""
    theme = theme or THEME
    return f"""
        QPushButton {{
            background-color: {theme["background"]};
            color: {theme["text"]};
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-weight: bold;
            font-size: {FONT_SIZE_NORMAL};
            height: 28px;
            padding: 0 12px;
        }}
        QPushButton:hover {{
            background-color: #f9fafb;
            border-color: #9ca3af;
        }}
        QPushButton:pressed {{
            background-color: #f3f4f6;
        }}
    """


def get_save_button_style(theme: Dict[str, str] = None):
    """Style for edit message save button"""
    theme = theme or THEME
    return f"""
        QPushButton {{
            background-color: {theme["primary"]};
            border: 1px solid {theme["primary"]};
            border-radius: 4px;
            padding: 4px 10px;
            color: white;
            font-size: {FONT_SIZE_NORMAL};
        }}
        QPushButton:hover {{
            background-color: {theme["primary"]};
        }}
    """


def get_danger_button_style() -> str:
    """Style for danger/destructive action buttons"""
    return f"""
        QPushButton {{
            background-color: #fed7d7;
            color: #c53030;
            border: 1px solid #feb2b2;
            border-radius: 6px;
            font-weight: bold;
            font-size: {FONT_SIZE_NORMAL};
            height: 28px;
            padding: 0 12px;
        }}
        QPushButton:hover {{
            background-color: #fbb6ce;
        }}
        QPushButton:pressed {{
            background-color: #f687b3;
        }}
    """


def get_export_button_style() -> str:
    """Style for export action buttons"""
    return f"""
        QPushButton {{
            background-color: #48bb78;
            color: #ffffff;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            font-size: {FONT_SIZE_NORMAL};
            height: 28px;
            padding: 0 12px;
        }}
        QPushButton:hover {{
            background-color: #38a169;
        }}
        QPushButton:pressed {{
            background-color: #2f855a;
        }}
    """
