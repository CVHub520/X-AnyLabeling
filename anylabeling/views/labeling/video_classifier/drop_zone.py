import os

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QVBoxLayout,
)

from .config import SUPPORTED_VIDEO_EXTS, SUPPORTED_VIDEO_FILTER
from .icons import themed_icon, theme_icon_color
from .style import get_drop_zone_style


class DropZone(QFrame):
    """Empty-state widget: dashed border + drag/drop + click-to-load."""

    videoSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("XvaDropZone")
        self.setStyleSheet(get_drop_zone_style())
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("hover", "false")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.icon_label = QLabel()
        self.icon_label.setObjectName("XvaDropZoneIcon")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon: QIcon = themed_icon(
            "video", "svg", theme_icon_color("primary"), 64
        )
        if icon is not None and not icon.isNull():
            self.icon_label.setPixmap(icon.pixmap(QSize(64, 64)))
        else:
            self.icon_label.setText("Video")
        layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignCenter)

        self.title_label = QLabel(
            self.tr("Drop a video here or click to load")
        )
        self.title_label.setObjectName("XvaDropZoneTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        ext_text = ", ".join(e.lstrip(".") for e in SUPPORTED_VIDEO_EXTS)
        self.hint_label = QLabel(self.tr("Supported: ") + ext_text)
        self.hint_label.setObjectName("XvaDropZoneHint")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

    # Drag & drop
    def dragEnterEvent(self, event):
        if self._has_acceptable_url(event):
            event.acceptProposedAction()
            self._set_hover(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._set_hover(False)
        super().dragLeaveEvent(event)

    def dragMoveEvent(self, event):
        if self._has_acceptable_url(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        self._set_hover(False)
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local and self._is_supported(local):
                event.acceptProposedAction()
                self.videoSelected.emit(local)
                return
        event.ignore()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._open_file_dialog()
            return
        super().mousePressEvent(event)

    # Helpers
    def _has_acceptable_url(self, event):
        if not event.mimeData().hasUrls():
            return False
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local and self._is_supported(local):
                return True
        return False

    def _is_supported(self, path):
        if not path or not os.path.isfile(path):
            return False
        return path.lower().endswith(SUPPORTED_VIDEO_EXTS)

    def _set_hover(self, hover):
        self.setProperty("hover", "true" if hover else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open video"),
            "",
            SUPPORTED_VIDEO_FILTER,
        )
        if path:
            self.videoSelected.emit(path)
