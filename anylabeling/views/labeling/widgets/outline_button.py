from PyQt6 import QtCore, QtWidgets

from ..utils.theme import get_mode, get_theme


class OutlineButton(QtWidgets.QPushButton):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(32)
        self.setIconSize(QtCore.QSize(16, 16))
        theme = get_theme()
        dark = get_mode() == "dark"
        background = (
            theme["background_secondary"] if dark else theme["background"]
        )
        hover = theme["surface_hover"] if dark else "#f5f5f5"
        self.setStyleSheet(f"""
            QPushButton {{
                background: {background}; color: {theme["text"]};
                border: 1px solid {theme["border"]}; border-radius: 10px;
                min-height: 30px; max-height: 30px; min-width: 0;
                padding: 0 10px; margin: 0; font-size: 14px; font-weight: 500;
            }}
            QPushButton:hover {{ background: {hover}; }}
            QPushButton:pressed {{ background: {theme["surface_pressed"]}; }}
            QPushButton:disabled {{ color: {theme["text_secondary"]}; }}
        """)
