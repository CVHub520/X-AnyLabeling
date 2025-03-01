from typing import Dict

from PyQt5.QtWidgets import QWidget, QPushButton

# Global design system
ANIMATION_DURATION = "200ms"
BORDER_RADIUS = "8px"
FONT_FAMILY = "SF Pro Text, -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif"
FONT_SIZE_NORMAL = "13px"
FONT_SIZE_SMALL = "12px"
FONT_SIZE_LARGE = "15px"

# Provider configurations
DEFAULT_PROVIDER = "ollama"
PROVIDER_CONFIGS = {
    "deepseek": {
        "icon": "anylabeling/resources/images/deepseek.svg",
        "api_address": "https://api.deepseek.com/v1",
        "api_key" : None,
        "model_name": "deepseek-vision",
        "api_key_url": "https://platform.deepseek.com/api_keys",
        "api_docs_url": "https://platform.deepseek.com/docs",
        "model_docs_url": "https://platform.deepseek.com/models"
    },
    "ollama": {
        "icon": "anylabeling/resources/images/ollama.svg",
        "api_address": "http://localhost:11434/v1",
        "model_name": "llava",
        "api_key": "ollama",
        "api_key_url": None,
        "api_docs_url": "https://github.com/ollama/ollama/blob/main/docs/api.md",
        "model_docs_url": "https://ollama.com/search"
    },
    "qwen": {
        "icon": "anylabeling/resources/images/qwen.svg",
        "api_address": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-vl-max-latest",
        "api_key": None,
        "api_key_url": "https://bailian.console.aliyun.com/?apiKey=1#/api-key",
        "api_docs_url": "https://help.aliyun.com/document_detail/2590237.html",
        "model_docs_url": "https://help.aliyun.com/document_detail/2590257.html"
    },
}

# Theme configuration - Apple-inspired
THEME = {
    "primary": "#0071e3",           # Apple-style blue
    "primary_hover": "#0077ed",     # Slightly lighter blue
    "background": "#ffffff",        # Pure white
    "sidebar": "#f5f5f7",           # Light gray sidebar
    "border": "#d2d2d7",            # Light border
    "text": "#1d1d1f",              # Almost black
    "text_secondary": "#86868b",    # Medium gray
    "success": "#34c759",           # Green
    "warning": "#ff9500",           # Orange
    "error": "#ff3b30",             # Red
    "user_bubble": "#e1f2ff",       # Light blue for user messages
    "bot_bubble": "#f5f5f7",        # Light gray for bot messages
    "input_bg": "#ffffff",          # White input background
    "hover": "#f5f5f7",             # Light hover state
}


def set_left_widget(left_panel):
    left_widget = QWidget()
    left_widget.setLayout(left_panel)
    left_widget.setFixedWidth(240)  # Slightly wider for better spacing
    left_widget.setStyleSheet(f"""
        QWidget {{
            background-color: {THEME["sidebar"]};
            border-right: 1px solid {THEME["border"]};
        }}
    """)
    return left_widget


def set_button_style(btn: QPushButton, theme: Dict[str, str]) -> str:
    btn.setFixedSize(150, 36)  # Uniform size
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {theme["primary"]};
            color: white;
            border: none;
            border-radius: {BORDER_RADIUS};
            padding: 8px 16px;
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_NORMAL};
            font-weight: 500;
            transition: background-color {ANIMATION_DURATION} ease;
        }}
        QPushButton:hover {{
            background-color: {theme["primary_hover"]};
        }}
        QPushButton:pressed {{
            background-color: {theme["primary"]};
            opacity: 0.8;
        }}
        QPushButton:disabled {{
            background-color: {theme["border"]};
            color: {theme["text_secondary"]};
        }}
    """)


# Style configurations
def get_dialog_style(theme: Dict[str, str]) -> str:
    return f"""
        QDialog {{
            background-color: {theme["background"]};
            border: none;
            font-family: {FONT_FAMILY};
        }}
        QLabel {{
            color: {theme["text"]};
            font-size: {FONT_SIZE_NORMAL};
            font-weight: 500;
            margin-bottom: 4px;
        }}
        QLineEdit, QTextEdit {{
            border: 1px solid {theme["border"]};
            border-radius: {BORDER_RADIUS};
            padding: 10px;
            background-color: {theme["input_bg"]};
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
            border-radius: {BORDER_RADIUS};
            padding: 8px 16px;
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_NORMAL};
            font-weight: 500;
            transition: all {ANIMATION_DURATION} ease;
        }}
        QPushButton:hover {{
            background-color: {theme["primary_hover"]};
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


def get_provider_button_style(theme: Dict[str, str]) -> str:
    return f"""
        QPushButton {{
            background-color: transparent;
            border: none;
            border-radius: {BORDER_RADIUS};
            text-align: left;
            padding: 12px 16px;
            color: {theme["text"]};
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_NORMAL};
            font-weight: 500;
            transition: all {ANIMATION_DURATION} ease;
        }}
        QPushButton:checked {{
            background-color: {theme["primary"]};
            color: white;
        }}
        QPushButton:hover:!checked {{
            background-color: {theme["hover"]};
        }}
    """


def get_link_style(theme: Dict[str, str]) -> str:
    return f"""
        QPushButton {{
            border: none;
            color: {theme["primary"]};
            text-align: left;
            padding: 0;
            background: transparent;
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_SMALL};
            transition: color {ANIMATION_DURATION} ease;
        }}
        QPushButton:hover {{
            color: {theme["primary_hover"]};
            text-decoration: underline;
        }}
    """


def get_image_preview_style(theme: Dict[str, str]) -> str:
    return f"""
        QLabel {{
            background-color: {theme["background"]};
            border: 1px solid {theme["border"]};
            border-radius: {BORDER_RADIUS};
            padding: 8px;
        }}
    """


def get_chat_bubble_style(theme: Dict[str, str], is_user: bool = False, is_error: bool = False) -> str:
    if is_error:
        bubble_color = theme["bot_bubble"]
        text_color = theme["error"]
    else:
        bubble_color = theme["user_bubble"] if is_user else theme["bot_bubble"]
        text_color = theme["text"]
    
    align = "right" if is_user else "left"
    
    return f"""
        <div style="
            margin: 12px 0; 
            text-align: {align};
        ">
            <div style="
                display: inline-block; 
                background-color: {bubble_color}; 
                padding: 12px 16px; 
                border-radius: {BORDER_RADIUS}; 
                max-width: 80%;
                font-family: {FONT_FAMILY.split(',')[0]};
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                color: {text_color};
            ">
    """


def get_chat_input_style(theme: Dict[str, str]) -> str:
    return f"""
        QTextEdit {{
            border: 1px solid {theme["border"]};
            border-radius: {BORDER_RADIUS};
            padding: 12px;
            background-color: {theme["input_bg"]};
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_NORMAL};
            min-height: 24px;
            max-height: 120px;
        }}
        QTextEdit:focus {{
            border: 2px solid {theme["primary"]};
        }}
    """


def get_send_button_style(theme: Dict[str, str]) -> str:
    return f"""
        QPushButton {{
            background-color: {theme["primary"]};
            color: white;
            border: none;
            border-radius: 15px; /* Make it circular */
            padding: 4px;
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_NORMAL};
            font-weight: 500;
            icon-size: 16px;
            min-width: 30px;
            min-height: 30px;
            max-width: 30px;
            max-height: 30px;
            transition: background-color {ANIMATION_DURATION} ease;
        }}
        QPushButton:hover {{
            background-color: {theme["primary_hover"]};
        }}
        QPushButton:pressed {{
            background-color: {theme["primary"]};
            opacity: 0.8;
        }}
        QPushButton:disabled {{
            background-color: {theme["border"]};
            color: {theme["text_secondary"]};
        }}
    """
