from PyQt6 import QtCore, QtGui, QtWidgets

from ..utils.theme import get_theme


def paint_input_frame(widget, focused):
    painter = QtGui.QPainter(widget)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    theme = get_theme()
    if focused:
        ring = QtGui.QColor(theme["text_secondary"])
        ring.setAlpha(110)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(ring)
        painter.drawRoundedRect(QtCore.QRectF(widget.rect()), 9, 9)
    rect = QtCore.QRectF(widget.rect()).adjusted(2.5, 2.5, -2.5, -2.5)
    painter.setBrush(QtGui.QColor(theme["background"]))
    painter.setPen(QtGui.QPen(QtGui.QColor(theme["border_light"]), 1))
    painter.drawRoundedRect(rect, 6, 6)


class TextInput(QtWidgets.QLineEdit):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(32)
        theme = get_theme()
        self.setStyleSheet(f"""
            QLineEdit {{
                background: transparent; color: {theme["text"]};
                border: 3px solid transparent; border-radius: 8px;
                min-height: 26px; max-height: 26px;
                padding: 0 7px; font-size: 14px;
                selection-background-color: {theme["selection"]};
                selection-color: {theme["selection_text"]};
            }}
        """)

    def paintEvent(self, event):
        paint_input_frame(self, self.hasFocus())
        super().paintEvent(event)
