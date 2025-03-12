from typing import Dict, List, Tuple

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
            padding: 10px;
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
            border-radius: {BORDER_RADIUS};
            padding: 8px 16px;
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_NORMAL};
            font-weight: 500;
            transition: all {ANIMATION_DURATION} ease;
        }}
        QPushButton:hover {{
            background-color: {theme["primary"]};
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
                padding: 14px 18px;
                color: {theme["text"]};
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE_NORMAL};
                font-weight: 450;
                transition: all {ANIMATION_DURATION} ease;
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

    def get_navigation_btn_style(theme: Dict[str, str] = None) -> str:
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
            QLineEdit:focus {{
                border: 3px solid {theme["primary"]};
                background-color: {theme["background_secondary"]};
            }}
        """

    def get_toggle_visibility_btn_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QPushButton {{
                border: none;
                background: transparent;
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

    def get_combobox_style(img_url, theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QComboBox {{
                border: 1px solid {theme["border"]};
                border-radius: {BORDER_RADIUS};
                padding: 4px 10px;
                background-color: {theme["background_secondary"]};
                color: {theme["text"]};
            }}
            QComboBox:hover {{
                border: 3px solid {theme["primary"]};
            }}
            QComboBox:focus {{
                background-color: {theme["background_secondary"]};
            }}
            QComboBox::down-arrow {{
                image: url("{img_url}");
                width: 24px;
                height: 24px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 24px;
                border-left: 1px solid transparent;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
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
            height: {TEMPERATURE_SLIDER_HEIGHT}px;
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

    def get_spinbox_style(theme: Dict[str, str] = None) -> str:
        theme = theme or THEME
        return f"""
            QSpinBox {{
                border: 1px solid {theme["border"]};
                border-radius: {BORDER_RADIUS};
                padding: 8px;
                background-color: {theme["background_secondary"]};
                color: {theme["text"]};
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE_NORMAL};
            }}
            QSpinBox:focus {{
                border: 3px solid {theme["primary"]};
                background-color: {theme["background_secondary"]};
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                width: 20px;
                border: none;
                background: transparent;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background-color: {theme["background_secondary"]};
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
                font-weight: bold;
                color: {theme["text"]};
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE_NORMAL};
                background-color: transparent;
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
                background-color: {theme["background_secondary"]};
                border-radius: {BORDER_RADIUS};
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
                padding: 4px 0px;
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
                background-coldfor: {theme["primary"]};
            }}
        """
