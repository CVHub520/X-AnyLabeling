import os

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QGraphicsDropShadowEffect,
    QApplication,
)
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPainter, QPainterPath, QColor


def is_wsl():
    """ Check if running in WSL """
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                return True
    return False


class Popup(QWidget):
    def __init__(self, text, parent=None, msec=5000):
        super().__init__(
            parent, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )

        self.setStyleSheet(
            """
            QWidget {
                background-color: #f2edec;
                border-radius: 16px;
            }
        """
        )

        layout = QVBoxLayout()

        self.label = QLabel(text)
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignLeft)
        layout.addWidget(self.label)

        self.setLayout(layout)

        # Set window properties
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # Add drop shadow effect
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(16)
        self.shadow.setColor(QColor(0, 0, 0, 80))
        self.shadow.setOffset(0, 3)
        self.setGraphicsEffect(self.shadow)

        # Create auto-close timer
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.close)
        self.timer.start(msec)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        path = QPainterPath()
        rect = QRectF(self.rect())
        path.addRoundedRect(rect, 10, 10)

        painter.fillPath(path, QColor("#f2edec"))

    def show_popup(self, parent_widget, copy_msg="", popup_width=350, popup_height=50):

        if copy_msg:
            if is_wsl():
                # Use clip.exe for WSL environment
                escaped_msg = copy_msg.replace('"', '\\"')
                os.system(f'echo "{escaped_msg}" | clip.exe')
            else:
                # Use Qt clipboard for Windows/other environments
                clipboard = QApplication.clipboard()
                clipboard.setText(self.copy_msg)
            self.close()

        # Calculate the position to place the popup at the top center
        parent_geo = parent_widget.geometry()

        # Position popup at the top center with a small margin from the top
        x = parent_geo.x() + (parent_geo.width() - popup_width) // 2
        y = parent_geo.y() + 20

        self.setGeometry(x, y, popup_width, popup_height)
        self.show()
