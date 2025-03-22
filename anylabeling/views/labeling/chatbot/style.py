from typing import Dict, List, Tuple

import markdown
import markdown.extensions.fenced_code
import markdown.extensions.codehilite
import markdown.extensions.tables
import markdown.extensions.toc

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
    def __init__(self,
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
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: {text_color};")
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

        self.setStyleSheet(f"""
            QWidget {{
                background-color: transparent;
            }}
            #tooltip_container {{
                background-color: {background_color}; 
                border-radius: 16px;
                border: 1px solid rgba(0, 0, 0, 0.1);
            }}
        """)

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

    def get_spinbox_style(up_arrow_url: str, down_arrow_url: str, theme: Dict[str, str] = None) -> str:
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
    """Set the HTML style for the content label"""
    extension_configs = {
        'codehilite': {
            'linenums': False,
            'guess_lang': False
        }
    }

    # Convert markdown to HTML with extensions
    html = markdown.markdown(
        content, 
        extensions=[
            'fenced_code',
            'codehilite',
            'tables', 
            'toc'
        ],
        extension_configs=extension_configs
    )

    return f"""
    <html>
    <head>
        <style>
            /* GitHub-like styling */
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                font-size: 16px;
                line-height: 1.5;
                color: #24292e;
                max-width: 100%;
                margin: 0;
                padding: 0;
            }}
            
            h1, h2, h3, h4, h5, h6 {{
                margin-top: 24px;
                margin-bottom: 16px;
                font-weight: 600;
                line-height: 1.25;
            }}
            
            /* Remove border-bottom from h1 and h2 as requested */
            h1 {{ font-size: 2em; padding-bottom: 0.3em; }}
            h2 {{ font-size: 1.5em; padding-bottom: 0.3em; }}
            
            a {{ color: #0366d6; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            
            code {{
                font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace;
                background-color: rgba(27, 31, 35, 0.05);
                padding: 0.2em 0.4em;
                border-radius: 3px;
                font-size: 85%;
            }}
            
            pre {{
                word-wrap: normal;
                position: relative;
                margin: 0;
            }}
            
            .code-block {{
                margin-bottom: 16px;
                background-color: #f6f8fa;
                border-radius: 6px;
                overflow: hidden;
            }}
            
            .code-header {{
                position: relative;
                padding: 8px 16px;
                color: #24292e;
                background-color: #f6f8fa;
                border-bottom: 1px solid #e1e4e8;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
                font-size: 85%;
                line-height: 1.4;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}
            
            .language-label {{
                color: #24292e;
                font-weight: 600;
            }}
            
            .copy-btn {{
                position: relative;
                display: inline-block;
                cursor: pointer;
                background: none;
                border: 0;
                color: #586069;
                padding: 2px 6px;
            }}
            
            .copy-btn:hover {{
                color: #0366d6;
            }}
            
            .copy-btn svg {{
                fill: currentColor;
                display: inline-block;
                vertical-align: text-top;
                overflow: visible;
            }}
            
            pre > code {{
                padding: 16px;
                display: block;
                overflow: auto;
                line-height: 1.45;
                background-color: #f6f8fa;
                border-radius: 0;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
                font-size: 85%;
            }}
            
            blockquote {{
                padding: 0 1em;
                color: #6a737d;
                border-left: 0.25em solid #dfe2e5;
                margin: 0 0 16px 0;
            }}
            
            hr {{ height: 0.25em; padding: 0; margin: 24px 0; background-color: #e1e4e8; border: 0; }}
            
            table {{
                border-spacing: 0;
                border-collapse: collapse;
                margin-top: 0;
                margin-bottom: 16px;
                width: 100%;
                overflow: auto;
            }}
            
            table th {{
                font-weight: 600;
                padding: 6px 13px;
                border: 1px solid #dfe2e5;
            }}
            
            table td {{
                padding: 6px 13px;
                border: 1px solid #dfe2e5;
            }}
            
            table tr {{
                background-color: #fff;
                border-top: 1px solid #c6cbd1;
            }}
            
            table tr:nth-child(2n) {{
                background-color: #f6f8fa;
            }}
            
            img {{ max-width: 100%; box-sizing: content-box; }}
            
            ul, ol {{
                padding-left: 2em;
                margin-top: 0;
                margin-bottom: 16px;
            }}
            
            li + li {{ margin-top: 0.25em; }}
            
            /* Additional styling for code blocks with syntax highlighting */
            .codehilite {{ margin-bottom: 0; }}
            .codehilite pre {{ margin-bottom: 0; }}
        </style>
        
        <!-- MathJax for formula rendering -->
        <script type="text/javascript" async
            src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
        </script>
        <script type="text/x-mathjax-config">
            MathJax.Hub.Config({{
                tex2jax: {{
                    inlineMath: [['$','$'], ['\\\\(','\\\\)']],
                    displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
                    processEscapes: true
                }},
                "HTML-CSS": {{ fonts: ["TeX"] }}
            }});
        </script>
        
        <!-- JavaScript for copy button functionality -->
        <script>
            document.addEventListener('DOMContentLoaded', () => {{
                // Wrap all code blocks and add copy button
                document.querySelectorAll('pre > code').forEach((codeBlock, index) => {{
                    const pre = codeBlock.parentNode;
                    const wrapper = document.createElement('div');
                    wrapper.className = 'code-block';
                    
                    // Get language if available
                    let language = 'Text';
                    if (codeBlock.className) {{
                        const match = codeBlock.className.match(/language-([\\w-]+)/);
                        if (match) language = match[1].charAt(0).toUpperCase() + match[1].slice(1);
                    }}
                    
                    // Create header with language label and copy button
                    const header = document.createElement('div');
                    header.className = 'code-header';
                    
                    const languageLabel = document.createElement('span');
                    languageLabel.className = 'language-label';
                    languageLabel.textContent = language;
                    
                    const copyButton = document.createElement('button');
                    copyButton.className = 'copy-btn';
                    copyButton.setAttribute('data-index', index);
                    copyButton.title = 'Copy code to clipboard';
                    
                    // Create SVG icon for copy button
                    copyButton.innerHTML = `
                        <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16">
                            <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path>
                            <path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
                        </svg>
                    `;
                    
                    header.appendChild(languageLabel);
                    header.appendChild(copyButton);
                    
                    // Replace pre with wrapper and move pre inside wrapper
                    pre.parentNode.insertBefore(wrapper, pre);
                    wrapper.appendChild(header);
                    wrapper.appendChild(pre);
                    
                    // Add click event for copy button
                    copyButton.addEventListener('click', () => {{
                        const textarea = document.createElement('textarea');
                        textarea.value = codeBlock.textContent;
                        document.body.appendChild(textarea);
                        textarea.select();
                        document.execCommand('copy');
                        document.body.removeChild(textarea);
                        
                        // Change the icon to a checkmark temporarily
                        const originalInnerHTML = copyButton.innerHTML;
                        copyButton.innerHTML = `
                            <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16">
                                <path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path>
                            </svg>
                        `;
                        
                        setTimeout(() => {{
                            copyButton.innerHTML = originalInnerHTML;
                        }}, 2000);
                    }});
                }});
            }});
        </script>
    </head>
    <body>
        {html}
    </body>
    </html>
    """