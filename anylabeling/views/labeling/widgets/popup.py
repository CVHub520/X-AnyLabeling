import os

from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QGraphicsDropShadowEffect,
    QApplication,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, QRectF, QSize
from PyQt5.QtGui import QPainter, QPainterPath, QColor, QIcon


def is_wsl():
    """Check if running in WSL"""
    if os.path.exists("/proc/version"):
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                return True
    return False


class Popup(QWidget):
    def __init__(self, text, parent=None, msec=3000, icon=None):
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

        # Use horizontal layout to place icon and text side by side
        hbox = QHBoxLayout()
        hbox.setContentsMargins(12, 8, 12, 8)  # Add spacing on both sides

        # Add icon if provided
        self.icon_label = None
        if icon:
            self.icon_label = QLabel()
            self.icon_label.setPixmap(QIcon(icon).pixmap(QSize(16, 16)))
            self.icon_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            hbox.addWidget(self.icon_label)
            hbox.addSpacing(1)  # Space between icon and text

        # Add text label
        self.label = QLabel(text)
        self.label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        hbox.addWidget(self.label)

        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(hbox)
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

    def show_popup(
        self, parent_widget, copy_msg="", popup_height=36, position="default"
    ):
        if copy_msg:
            if is_wsl():
                # Use clip.exe for WSL environment
                escaped_msg = copy_msg.replace('"', '\\"')
                os.system(f'echo "{escaped_msg}" | clip.exe')
            else:
                # Use Qt clipboard for Windows/other environments
                clipboard = QApplication.clipboard()
                clipboard.setText(copy_msg)

        # Calculate position based on preference
        parent_geo = parent_widget.geometry()

        # Auto-adjust width based on content
        self.adjustSize()
        popup_width = self.sizeHint().width()

        # Set position based on specified option
        if position == "center":
            x = parent_geo.x() + (parent_geo.width() - popup_width) // 2
            y = parent_geo.y() + (parent_geo.height() - popup_height) // 2
        elif position == "bottom":
            x = parent_geo.x() + (parent_geo.width() - popup_width) // 2
            y = parent_geo.y() + parent_geo.height() - popup_height - 20
        else:  # "default" - top position
            x = parent_geo.x() + (parent_geo.width() - popup_width) // 2
            y = parent_geo.y() + 100

        self.setGeometry(x, y, popup_width, popup_height)
        self.show()
