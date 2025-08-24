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


def get_primary_button_style(theme: Dict[str, str] = None) -> str:
    """Style for primary action buttons"""
    theme = theme or THEME
    return f"""
        QPushButton {{
            background-color: {theme["primary"]};
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            font-size: {FONT_SIZE_NORMAL};
            height: 32px;
            padding: 0 16px;
            min-width: 80px;
        }}
        QPushButton:hover {{
            background-color: #3b82f6;
            transform: translateY(-1px);
        }}
        QPushButton:pressed {{
            background-color: #2563eb;
            transform: translateY(0px);
        }}
    """


def get_secondary_button_style(theme: Dict[str, str] = None) -> str:
    """Style for secondary action buttons"""
    theme = theme or THEME
    return f"""
        QPushButton {{
            background-color: {theme["background"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
            border-radius: 8px;
            font-weight: 500;
            font-size: {FONT_SIZE_NORMAL};
            height: 32px;
            padding: 0 16px;
            min-width: 80px;
        }}
        QPushButton:hover {{
            background-color: #f8fafc;
            border-color: #cbd5e1;
            transform: translateY(-1px);
        }}
        QPushButton:pressed {{
            background-color: #f1f5f9;
            transform: translateY(0px);
        }}
    """


def get_success_button_style(theme: Dict[str, str] = None) -> str:
    """Style for success/export action buttons"""
    theme = theme or THEME
    return f"""
        QPushButton {{
            background-color: #10b981;
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            font-size: {FONT_SIZE_NORMAL};
            height: 32px;
            padding: 0 16px;
            min-width: 80px;
        }}
        QPushButton:hover {{
            background-color: #059669;
            transform: translateY(-1px);
        }}
        QPushButton:pressed {{
            background-color: #047857;
            transform: translateY(0px);
        }}
    """


def get_danger_button_style(theme: Dict[str, str] = None) -> str:
    """Style for danger/destructive action buttons"""
    theme = theme or THEME
    return f"""
        QPushButton {{
            background-color: {theme["background"]};
            color: #dc2626;
            border: 1px solid #fecaca;
            border-radius: 8px;
            font-weight: 500;
            font-size: {FONT_SIZE_NORMAL};
            height: 32px;
            padding: 0 16px;
            min-width: 80px;
        }}
        QPushButton:hover {{
            background-color: #fef2f2;
            border-color: #fca5a5;
            color: #b91c1c;
            transform: translateY(-1px);
        }}
        QPushButton:pressed {{
            background-color: #fee2e2;
            transform: translateY(0px);
        }}
    """


def get_export_button_style() -> str:
    """Style for export action buttons - alias for success style"""
    return get_success_button_style()
