import os
import shutil
import subprocess
import sys

from anylabeling.views.labeling.utils.theme import get_theme
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QGraphicsDropShadowEffect,
    QApplication,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QSize
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QIcon


def is_wsl():
    """Check if running in WSL"""
    if os.path.exists("/proc/version"):
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                return True
    return False


def _copy_via_command(command, text):
    if isinstance(command, str):
        command = [command]
    executable = command[0]
    if os.sep in executable:
        exists = os.path.exists(executable)
    else:
        exists = shutil.which(executable) is not None
    if not exists:
        return False
    try:
        result = subprocess.run(
            command,
            input=text,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def copy_text_to_system_clipboard(text):
    if not text:
        return

    clipboard = QApplication.clipboard()
    if clipboard is not None:
        try:
            clipboard.setText(text)
            if clipboard.text() == text:
                return
        except Exception:
            pass

    if is_wsl():
        if _copy_via_command(["clip.exe"], text):
            return
        _copy_via_command(["/mnt/c/Windows/System32/clip.exe"], text)
        return

    if sys.platform.startswith("win"):
        _copy_via_command(["clip"], text)
        return

    if sys.platform == "darwin":
        _copy_via_command(["pbcopy"], text)
        return

    if _copy_via_command(["wl-copy"], text):
        return
    if _copy_via_command(["xclip", "-selection", "clipboard"], text):
        return
    _copy_via_command(["xsel", "--clipboard", "--input"], text)


class Popup(QWidget):
    def __init__(self, text, parent=None, msec=3000, icon=None):
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )

        t = get_theme()
        self._bg_color = t["surface_hover"]
        self._text_color = t["text"]
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self._bg_color};
                border-radius: 16px;
            }}
            QLabel {{
                background-color: transparent;
                color: {self._text_color};
            }}
        """)

        # Use horizontal layout to place icon and text side by side
        hbox = QHBoxLayout()
        hbox.setContentsMargins(12, 8, 12, 8)  # Add spacing on both sides

        # Add icon if provided
        self.icon_label = None
        if icon:
            self.icon_label = QLabel()
            self.icon_label.setPixmap(QIcon(icon).pixmap(QSize(16, 16)))
            self.icon_label.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
            )
            hbox.addWidget(self.icon_label)
            hbox.addSpacing(1)  # Space between icon and text

        # Add text label
        self.label = QLabel(text)
        self.label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        hbox.addWidget(self.label)

        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(hbox)
        self.setLayout(layout)

        # Set window properties
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )

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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        rect = QRectF(self.rect())
        path.addRoundedRect(rect, 10, 10)

        painter.fillPath(path, QColor(self._bg_color))

    def show_popup(
        self, parent_widget, copy_msg="", popup_height=36, position="default"
    ):
        if copy_msg:
            copy_text_to_system_clipboard(copy_msg)

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
