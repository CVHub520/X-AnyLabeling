from typing import Dict, List, Tuple

import markdown
import markdown.extensions.fenced_code
import markdown.extensions.codehilite
import markdown.extensions.tables
import markdown.extensions.toc
import markdown.extensions.attr_list
import markdown.extensions.smarty

from PyQt5.QtWidgets import (
    QWidget,
    QFrame,
    QVBoxLayout,
    QLabel,
    QGridLayout,
)
from PyQt5.QtCore import Qt

from anylabeling.views.labeling.chatbot.config import *


class CustomTooltip(QWidget):
    """Custom tooltip widget with Mac-style appearance"""

    def __init__(
        self,
        parent=None,
        text_color: str = "#7f7f88",
        background_color: str = "#ffffff",
        title: str = None,
        value_pairs: List[Tuple[str, str]] = None,
    ):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self.container = QFrame(self)
        self.container.setObjectName("tooltip_container")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: #000000;")
        layout.addWidget(self.title_label)

        if value_pairs:
            grid_layout = QGridLayout()
            grid_layout.setHorizontalSpacing(20)
            grid_layout.setVerticalSpacing(4)
            for i, (label, value) in enumerate(value_pairs):
                name_label = QLabel(label)
                name_label.setStyleSheet(f"color: {text_color};")
                value_label = QLabel(value)
                value_label.setStyleSheet(f"color: {text_color};")
                value_label.setAlignment(Qt.AlignRight)
                grid_layout.addWidget(name_label, i, 0)
                grid_layout.addWidget(value_label, i, 1)

            layout.addLayout(grid_layout)

        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: transparent;
            }}
            #tooltip_container {{
                background-color: {background_color}; 
                border-radius: 16px;
                border: 1px solid rgba(0, 0, 0, 0.1);
            }}
        """
        )

    def show_at(self, pos):
        """Show tooltip at specified position"""
        self.move(pos)
        self.adjustSize()
        self.show()


class ChatbotDialogStyle:
    def get_menu_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QMenu {{
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                padding: 2px;
            }}
            QMenu::item {{
                padding: 6px 28px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: #F5F5F5;
            }}
            QMenu::separator {{
                height: 1px;
                background-color: #E0E0E0;
                margin: 4px 2px;
            }}
        """

    def get_dialog_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
        QDialog {{
            background-color: {theme["background"]};
            border: none;
            font-family: {FONT_FAMILY};
        }}
        QLabel {{
            border: none;
            border-left: none;
        }}
        QLineEdit, QTextEdit {{
            border: 1px solid {theme["border"]};
            border-radius: {BORDER_RADIUS};
            padding: 8px;
            background-color: {theme["background_secondary"]};
            selection-background-color: {theme["primary"]};
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_NORMAL};
            transition: border {ANIMATION_DURATION} ease;
        }}
        QLineEdit:focus, QTextEdit:focus {{
            border: 2px solid {theme["primary"]};
        }}
        QPushButton {{
            border: none;
            background: transparent;
        }}
        QPushButton:hover {{
            background-color: {theme["background_secondary"]};
            border-radius: {BORDER_RADIUS};
        }}
        QScrollBar:vertical {{
            border: none;
            background: {theme["background"]};
            width: 8px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: {theme["border"]};
            min-height: 20px;
            border-radius: 4px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar:horizontal {{
            border: none;
            background: {theme["background"]};
            height: 8px;
            margin: 0px;
        }}
        QScrollBar::handle:horizontal {{
            background: {theme["border"]};
            min-width: 20px;
            border-radius: 4px;
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
    """

    def get_main_splitter_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QSplitter::handle {{
                background-color: {theme["border"]};
            }}
        """

    def get_provider_button_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: {BORDER_RADIUS};
                text-align: left;
                padding: 8px;
                color: {theme["text"]};
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE_NORMAL};
                font-weight: 450;
                transition: all {ANIMATION_DURATION} ease;
            }}
            QPushButton:focus {{
                outline: none;
                border: none;
            }}
            QPushButton:checked {{
                background-color: #d1d0d4;
                color: white;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            QPushButton:hover:!checked {{
                background-color: #dcdbdf;
            }}
        """

    def get_left_widget_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QWidget {{
                background-color: #e3e2e6;
                border-right: 1px solid {theme["border"]};
            }}
        """

    def get_middle_widget_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QWidget {{
                background-color: {theme["background"]};
            }}
        """

    def get_chat_container_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QWidget {{
                background-color: {theme["background"]};
            }}
        """

    def get_chat_scroll_area_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QScrollArea {{
                background-color: {theme["background"]};
                border: none;
            }}
        """

    def get_input_container_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QWidget {{
                background-color: transparent;
            }}
        """

    def get_input_frame_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QFrame#inputFrame {{
                border: 1px solid {theme["border"]};
                border-radius: {BORDER_RADIUS};
                background-color: {theme["background_secondary"]};
            }}
            QFrame#inputFrame:focus-within {{
                border: 1px solid {theme["primary"]};
            }}
        """

    def get_message_input_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QTextEdit {{
                border: none;
                background-color: {theme["background_secondary"]};
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE_NORMAL};
                padding: 0px;
                color: {theme["text"]};  /* Set text color */
            }}
            QTextEdit::frame {{
                border: none;
            }}
            QTextEdit::viewport {{
                border: none;
                background-color: {theme["background_secondary"]};
            }}
            QScrollBar:vertical {{
                width: 8px;
                background: {theme["background_secondary"]};
            }}
            QScrollBar::handle:vertical {{
                background: {theme["border"]};
                border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """

    def get_send_button_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QPushButton {{
                border: none;
                background-color: {theme["background_secondary"]};
                padding: 0px;
                margin: 0px;
            }}
            QPushButton:hover {{
                background-color: {theme["background_secondary"]};
                border-radius: 10px;
            }}
            QPushButton:disabled {{
                opacity: 0.5;
            }}
        """

    def get_image_preview_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QLabel {{
                background-color: {theme["background"]};
                border: 1px solid {theme["border"]};
                border-radius: {BORDER_RADIUS};
                padding: 8px;
            }}
        """

    def get_button_style(theme: Dict[str, str] = None) -> str:
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

    def get_model_button_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QPushButton {{
                background-color: {theme["background_secondary"]};
                border-radius: {BORDER_RADIUS};
                border: 1px solid {theme["border"]};
                padding: 8px;
                color: {theme["text"]};
                font-size: {FONT_SIZE_NORMAL};
            }}
            QPushButton:hover {{
                background-color: {theme["background_hover"]};
                border-radius: {BORDER_RADIUS};
            }}
            QPushButton:focus {{
                outline: none;
                border: none;
            }}
        """

    def get_settings_label_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QLabel {{
                background-color: transparent;
                border-left: none;
            }}
        """

    def get_help_btn_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QPushButton {{
                border: none;
                background: transparent;
            }}
            QPushButton:hover {{
                background-color: {theme["background_secondary"]};
                border-radius: {BORDER_RADIUS};
            }}
        """

    def get_settings_edit_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QLineEdit {{
                border: 1px solid {theme["border"]};
                border-radius: {BORDER_RADIUS};
                padding: 8px;
                background-color: {theme["background_secondary"]};
                color: {theme["text"]};
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE_NORMAL};
            }}
            QLineEdit:hover {{
                background-color: {theme["background_hover"]};
                border-radius: {BORDER_RADIUS};
            }}
            QLineEdit:focus {{
                border: 3px solid {theme["primary"]};
                background-color: {theme["background_secondary"]};
            }}
        """

    def get_right_splitter_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QSplitter::handle {{
                background-color: {theme["border"]};
            }}
        """

    def get_right_widget_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QWidget {{
                background-color: {theme["background"]};
                border-left: none;
            }}
        """

    def get_temperature_label_style():
        return """
        QLabel {{
            background-color: transparent;
            font-size: FONT_SIZE_TINY;
            color: #727273;
        }}
        """

    def get_tab_widget_style(theme: Dict[str, str] = None):
        theme = theme or THEME
        return f"""
        QTabWidget::pane {{
            border: none;
            background: transparent;
        }}
        
        QTabWidget::tab-bar {{
            alignment: left;
            border-left: none;
        }}
        
        QTabBar::tab {{
            background: #f5f5f5;
            color: #333;
            padding: 8px 16px;
            border: none;
            border-left: none;
            border-bottom: 2px solid transparent;
            min-width: 100px;
            font-weight: 500;
        }}
        
        QTabBar::tab:selected {{
            background: #ffffff;
            border-bottom: 3px solid {theme["primary"]};
            border-left: none;
            color: {theme["primary"]};
        }}
        
        QTabBar::tab:hover:!selected {{
            background: #e0e0e0;
        }}
        """

    def get_slider_style(theme: Dict[str, str] = None):
        theme = theme or THEME
        return f"""
        QSlider {{
            height: 20px;
        }}

        QSlider::groove:horizontal {{
            border: none;
            height: 4px;
            background: {theme["border"]};
            margin: 0px;
            border-radius: 2px;
        }}

        QSlider::handle:horizontal {{
            background: {theme["primary"]};
            border: none;
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: {BORDER_RADIUS};
        }}

        QSlider::sub-page:horizontal {{
            background: {theme["primary"]};
            border-radius: 2px;
        }}
        """

    def get_settings_tabs_style():
        return """
        QTabBar::tab {
            text-align: center;
            padding-left: 0px;
            padding-right: 0px;
            border-left: none;
        }
        """

    def get_spinbox_style(
        up_arrow_url: str, down_arrow_url: str, theme: Dict[str, str] = None
    ) -> str:
        theme = theme or THEME
        return f"""
            QSpinBox {{
                border: 1px solid {theme["border"]};
                border-radius: {BORDER_RADIUS};
                padding: 2px 30px 2px 8px;
                background-color: {theme["background_secondary"]};
                color: {theme["text"]};
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE_NORMAL};
                min-height: 36px;
            }}
            QSpinBox:hover {{
                background-color: {theme["background_hover"]};
            }}
            QSpinBox:focus {{
                border: 2px solid {theme["primary"]};
                background-color: {theme["background_secondary"]};
            }}
            QSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                top: 5px;
                width: 22px;
                height: 18px;
                background-color: transparent;
                border: none;
                margin: 0px;
                margin-right: 10px;
            }}
            QSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                bottom: 5px;
                width: 22px;
                height: 18px;
                background-color: transparent;
                border: none;
                margin: 0px;
                margin-right: 10px;
            }}
            QSpinBox::up-arrow {{
                width: 22px;
                height: 16px;
                image: url({up_arrow_url});
            }}
            QSpinBox::down-arrow {{
                width: 22px;
                height: 16px;
                image: url({down_arrow_url});
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background-color: rgba(0, 0, 0, 0.1);
                border-radius: 3px;
            }}
            QSpinBox:disabled {{
                color: {theme["border"]};
                background-color: {theme["background"]};
            }}
        """

    def get_progress_dialog_style(theme: Dict[str, str] = None):
        theme = theme or THEME
        return f"""
            QProgressDialog {{
                background-color: {theme["background"]};
                border: 1px solid {theme["border"]};
                border-radius: 12px;
                min-width: 350px;
                min-height: 120px;
                padding: 10px;
                font-family: {FONT_FAMILY};
            }}

            QProgressDialog QLabel {{
                color: {theme["text"]};
                font-size: {FONT_SIZE_NORMAL};
                font-weight: 500;
                margin-bottom: 10px;
            }}

            QProgressDialog QProgressBar {{
                border: none;
                border-radius: 8px;
                background-color: {theme["background_secondary"]};
                text-align: center;
                color: {theme["text"]};
                font-weight: 500;
                height: 16px;
                min-width: 300px;
                margin: 16px 0;
            }}

            QProgressDialog QProgressBar::chunk {{
                background-color: {theme["primary"]};
                border-radius: 8px;
            }}

            QProgressDialog QPushButton {{
                border: 1px solid {theme["border"]};
                border-radius: {BORDER_RADIUS};
                padding: 8px 16px;
                background-color: {theme["background_secondary"]};
                color: {theme["text"]};
                font-weight: 500;
                min-width: 100px;
            }}

            QProgressDialog QPushButton:hover {{
                background-color: {theme["background_hover"]};
            }}

            QProgressDialog QPushButton:pressed {{
                background-color: {theme["border"]};
            }}
        """

    def get_option_dialog_style(theme: Dict[str, str] = None):
        theme = theme or THEME
        return f"""
            QDialog {{
                background-color: {theme["background"]};
                border-radius: {BORDER_RADIUS};
                font-family: {FONT_FAMILY};
            }}
            QPushButton {{
                background-color: {theme["primary"]};
                color: white;
                border: none;
                border-radius: {BORDER_RADIUS};
                padding: 8px 16px;
                font-size: {FONT_SIZE_NORMAL};
                min-width: 150px;
            }}
            QPushButton:hover {{
                background-color: {theme["primary"]};
            }}
            QPushButton:pressed {{
                background-color: #3D7FE3;
            }}
            """


class ChatMessageStyle:
    def get_bubble_style(is_user: bool, theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        bubble_color = "#f6f2f2" if is_user else None
        return f"""
            QWidget#messageBubble {{
                background-color: {bubble_color};
                border-radius: {BORDER_RADIUS};
                padding: 4px;
            }}
        """

    def get_role_label_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QLabel {{
                background-color: transparent;
            }}
        """

    def get_role_label_background_style() -> str:
        return f"""
            #roleLabelContainer {{
                background-color: #D1D0D4;
                border-radius: 8px;
                padding: 4px;
            }}
        """

    def get_button_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QPushButton {{
                border: none;
                background: transparent;
            }}
        """

    def get_content_label_style(is_error, theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        text_color = theme["error"] if is_error else theme["text"]
        return f"""
            QLabel {{
                color: {text_color};
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE_NORMAL};
                background-color: transparent;
                padding: 0px 0px;
                font-weight: normal;
            }}
        """

    def get_edit_area_style(theme: Dict[str, str] = None):
        theme = theme or THEME
        return f"""
            QTextEdit {{
                border: 3px solid {theme["primary"]};
                border-radius: {BORDER_RADIUS};
                padding: 4px 2px;
                color: {theme["text"]};
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE_NORMAL};
            }}
        """

    def get_edit_button_wdiget_style():
        """Style for edit buttons widget"""
        return """
            QWidget {
               border: none;
                background: transparent;
            }
        """

    def get_resend_button_style(theme: Dict[str, str] = None):
        """Style for edit message resend button"""
        theme = theme or THEME
        return f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: {FONT_SIZE_SMALL};
            }}
            QPushButton:hover {{
                background-color: {theme["background"]};
                border: 1px solid {theme["border"]};
                color: {theme["primary"]};
            }}
        """

    def get_cancel_button_style(theme: Dict[str, str] = None):
        """Style for edit message cancel button"""
        theme = theme or THEME
        return f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: {FONT_SIZE_SMALL};
            }}
            QPushButton:hover {{
                background-color: {theme["background"]};
                border: 1px solid {theme["border"]};
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
                font-size: {FONT_SIZE_SMALL};
            }}
            QPushButton:hover {{
                background-color: {theme["primary"]};
            }}
        """


def set_html_style(content):
    """Set the HTML style for the content label with GitHub-style markdown rendering and LaTeX support"""
    extension_configs = {
        "codehilite": {"linenums": False, "guess_lang": False},
        "fenced_code": {"lang_prefix": "language-"},
    }

    # Convert markdown to HTML with extensions
    html_content = markdown.markdown(
        content,
        extensions=[
            "fenced_code",
            "codehilite",
            "tables",
            "toc",
            "attr_list",
            "smarty",
        ],
        extension_configs=extension_configs,
    )

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GitHub-Style Markdown Renderer with LaTeX</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
        
        <!-- MathJax for LaTeX rendering -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"></script>

        <style>
            :root {{
                --primary: #60A5FA;
                --background: #FFFFFF;
                --background-secondary: #F9F9F9;
                --background-hover: #DBDBDB;
                --border: #E5E5E5;
                --text: #2C2C2E;
                --highlight-text: #2196F3;
                --success: #30D158;
                --warning: #FF9F0A;
                --error: #FF453A;
                --font-size-normal: 13px;
                --font-size-h1: 24px;
                --font-size-h2: 20px;
                --font-size-h3: 16px;
                --font-size-h4: 14px;
                --font-size-code: 12px;
            }}

            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                font-size: var(--font-size-normal);
                line-height: 1.5;
                color: var(--text);
                background-color: var(--background);
                max-width: 99%;
                margin: 0;
                padding: 0;
            }}

            .markdown-body {{
                padding: 5px;
                border: none;
                border-radius: 0px;
                background-color: var(--background);
                scrollbar-width: none; /* Firefox */
                -ms-overflow-style: none; /* IE and Edge */
            }}

            .markdown-body::-webkit-scrollbar {{
                display: none; /* Chrome, Safari, and Opera */
            }}
            
            /* Typography */
            .markdown-body h1, 
            .markdown-body h2, 
            .markdown-body h3, 
            .markdown-body h4, 
            .markdown-body h5, 
            .markdown-body h6 {{
                margin-top: 24px;
                margin-bottom: 16px;
                font-weight: 600;
                line-height: 1.25;
            }}

            .markdown-body h1 {{
                font-size: var(--font-size-h1);
                padding-bottom: 0.3em;
            }}

            .markdown-body h2 {{
                font-size: var(--font-size-h2);
                padding-bottom: 0.3em;
            }}

            .markdown-body h3 {{
                font-size: var(--font-size-h3);
            }}

            .markdown-body h4 {{
                font-size: var(--font-size-h4);
            }}

            .markdown-body p {{
                margin-bottom: 16px;
            }}

            .markdown-body a {{
                color: var(--highlight-text);
                text-decoration: none;
            }}

            .markdown-body a:hover {{
                text-decoration: underline;
            }}

            /* Lists */
            .markdown-body ul, 
            .markdown-body ol {{
                padding-left: 2em;
                margin-bottom: 16px;
            }}

            .markdown-body li {{
                margin-bottom: 4px;
            }}

            .markdown-body li + li {{
                margin-top: 4px;
            }}

            /* Blockquotes */
            .markdown-body blockquote {{
                padding: 0 1em;
                color: #6a737d;
                border-left: 0.25em solid var(--border);
                margin-bottom: 16px;
            }}

            /* Tables */
            .markdown-body table {{
                border-collapse: collapse;
                margin-bottom: 16px;
                width: auto;
                overflow: auto;
            }}

            .markdown-body table th,
            .markdown-body table td {{
                padding: 6px 13px;
                border: 1px solid var(--border);
            }}

            .markdown-body table tr {{
                background-color: var(--background);
                border-top: 1px solid var(--border);
            }}

            .markdown-body table tr:nth-child(2n) {{
                background-color: var(--background-secondary);
            }}

            .markdown-body table th {{
                font-weight: 600;
                background-color: var(--background-secondary);
            }}

            /* Code Blocks */
            .markdown-body pre {{
                position: relative;
                background-color: var(--background-secondary);
                border-radius: 6px;
                margin-bottom: 16px;
                overflow: auto;
            }}

            .markdown-body pre code {{
                display: block;
                padding: 16px;
                overflow-x: auto;
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
                font-size: var(--font-size-code);
                line-height: 1.45;
                background-color: transparent;
                border-radius: 0;
            }}

            .markdown-body code {{
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
                font-size: var(--font-size-code);
                padding: 0.2em 0.4em;
                margin: 0;
                background-color: rgba(27, 31, 35, 0.05);
                border-radius: 3px;
            }}

            /* Copy Button */
            .copy-button {{
                position: absolute;
                top: 8px;
                right: 8px;
                padding: 4px 8px;
                font-size: 12px;
                color: #666;
                background-color: var(--background);
                border: 1px solid var(--border);
                border-radius: 4px;
                cursor: pointer;
                opacity: 0;
                transition: opacity 0.2s;
            }}

            .markdown-body pre:hover .copy-button {{
                opacity: 1;
            }}

            .copy-button:hover {{
                background-color: var(--background-hover);
            }}

            .copy-button.copied {{
                color: var(--success);
                border-color: var(--success);
            }}

            /* Status colors */
            .success {{
                color: var(--success);
            }}

            .warning {{
                color: var(--warning);
            }}

            .error {{
                color: var(--error);
            }}

            /* TOC */
            .toc {{
                background-color: var(--background-secondary);
                border: 1px solid var(--border);
                border-radius: 6px;
                padding: 16px;
                margin-bottom: 16px;
            }}

            .toc-title {{
                font-weight: 600;
                margin-bottom: 8px;
            }}

            /* GitHub-style horizontal rule */
            .markdown-body hr {{
                display: none;
            }}
            
            /* Styles for LaTeX blocks */
            .math-block {{
                overflow-x: auto;
                margin: 16px 0;
                padding: 8px;
                background-color: var(--background-secondary);
                border-radius: 6px;
            }}
        </style>
    </head>
    <body>
        <div class="markdown-body" id="content">
            {html_content}
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                hljs.configure({{
                    ignoreUnescapedHTML: true
                }});

                // Add copy buttons to code blocks
                document.querySelectorAll('pre').forEach(function(codeBlock) {{
                    if (!codeBlock.querySelector('.copy-button')) {{
                        const copyButton = document.createElement('button');
                        copyButton.className = 'copy-button';
                        copyButton.textContent = 'Copy';
                        codeBlock.appendChild(copyButton);

                        copyButton.addEventListener('click', function() {{
                            const code = codeBlock.querySelector('code');
                            const range = document.createRange();
                            range.selectNode(code);
                            window.getSelection().removeAllRanges();
                            window.getSelection().addRange(range);
                            
                            try {{
                                document.execCommand('copy');
                                copyButton.textContent = 'Copied!';
                                copyButton.classList.add('copied');
                                
                                setTimeout(function() {{
                                    copyButton.textContent = 'Copy';
                                    copyButton.classList.remove('copied');
                                }}, 2000);
                            }} catch (err) {{
                                console.error('Failed to copy: ', err);
                                copyButton.textContent = 'Failed';
                            }}
                            
                            window.getSelection().removeAllRanges();
                        }});
                    }}
                }});

                // Initialize syntax highlighting
                document.querySelectorAll('pre code').forEach((el) => {{
                    hljs.highlightElement(el);
                }});
                
                // Configure MathJax
                window.MathJax = {{
                    tex: {{
                        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                        processEscapes: true,
                        tags: 'ams'
                    }},
                    options: {{
                        enableMenu: false,
                        renderActions: {{
                            addMenu: [0, '', '']
                        }}
                    }},
                    startup: {{
                        pageReady: () => {{
                            return MathJax.startup.defaultPageReady().then(() => {{
                                // Wrap display math in styled divs
                                document.querySelectorAll('.MathJax_Display').forEach(element => {{
                                    const wrapper = document.createElement('div');
                                    wrapper.className = 'math-block';
                                    element.parentNode.insertBefore(wrapper, element);
                                    wrapper.appendChild(element);
                                }});
                            }});
                        }}
                    }}
                }};
            }});
        </script>
    </body>
    </html>
    """
