"""Compare View widget for split-screen image comparison."""

import os
import os.path as osp
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from anylabeling.views.labeling.chatbot.style import ChatbotDialogStyle
from anylabeling.views.labeling.utils.theme import get_theme

SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


class CompareViewManager(QtCore.QObject):
    """
    Manager for Compare View functionality.

    Handles loading and displaying comparison images alongside the main image
    using a split-view interface with a draggable divider.

    Attributes:
        status_message: Signal emitted to display status bar messages.
        compare_closed: Signal emitted when compare view is closed.
    """

    status_message = QtCore.pyqtSignal(str, int)
    compare_closed = QtCore.pyqtSignal()

    def __init__(self, canvas, parent: Optional[QtWidgets.QWidget] = None):
        """
        Initialize the CompareViewManager.

        Args:
            canvas: The Canvas widget to render compare images on.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._canvas = canvas
        self._compare_dir: Optional[str] = None
        self._current_compare_path: Optional[str] = None
        self._file_cache: dict = {}
        self._init_canvas_state()

    def _init_canvas_state(self):
        """Initialize canvas compare view state."""
        self._canvas.compare_pixmap = None
        self._canvas.split_position = 0.5

    def is_active(self) -> bool:
        """
        Check if compare view is currently active.

        Returns:
            bool: True if compare view has an active directory set.
        """
        return self._compare_dir is not None

    def get_compare_directory(self) -> Optional[str]:
        """
        Get the current compare directory.

        Returns:
            Optional[str]: The compare directory path or None.
        """
        return self._compare_dir

    def set_compare_directory(self, directory: str) -> bool:
        """
        Set the compare image directory and build file cache.

        Args:
            directory: Path to the directory containing compare images.

        Returns:
            bool: True if directory was set successfully.
        """
        if not directory or not osp.isdir(directory):
            return False

        self._compare_dir = directory
        self._build_file_cache()
        return True

    def _build_file_cache(self):
        """Build a cache of basename -> filepath mappings for fast lookup."""
        self._file_cache.clear()
        if not self._compare_dir:
            return

        for root, _, files in os.walk(self._compare_dir):
            for file in files:
                ext = osp.splitext(file)[1].lower()
                if ext in SUPPORTED_FORMATS:
                    basename = osp.splitext(file)[0]
                    if basename not in self._file_cache:
                        self._file_cache[basename] = osp.join(root, file)

    def load_compare_for_file(self, filename: str) -> bool:
        """
        Load the compare image corresponding to the given main image file.

        Args:
            filename: Path to the main image file.

        Returns:
            bool: True if compare image was loaded successfully.
        """
        if not self._compare_dir or not filename:
            self._clear_compare_pixmap()
            return False

        basename = osp.splitext(osp.basename(filename))[0]
        compare_path = self._file_cache.get(basename)

        if not compare_path:
            self._clear_compare_pixmap()
            self.status_message.emit(
                f"No matching compare image for: {basename}", 3000
            )
            return False

        return self._load_compare_image(compare_path)

    def _load_compare_image(self, image_path: str) -> bool:
        """
        Load a compare image and validate its dimensions.

        Args:
            image_path: Path to the compare image.

        Returns:
            bool: True if image was loaded and dimensions match.
        """
        if not osp.isfile(image_path):
            self._clear_compare_pixmap()
            return False

        pixmap = QtGui.QPixmap(image_path)
        if pixmap.isNull():
            self._clear_compare_pixmap()
            self.status_message.emit(
                f"Failed to load compare image: {osp.basename(image_path)}",
                3000,
            )
            return False

        main_pixmap = self._canvas.pixmap
        if main_pixmap is None or main_pixmap.isNull():
            self._clear_compare_pixmap()
            return False

        if (
            pixmap.width() != main_pixmap.width()
            or pixmap.height() != main_pixmap.height()
        ):
            self._clear_compare_pixmap()
            self.status_message.emit(
                f"Size mismatch: compare image {pixmap.width()}x{pixmap.height()} "
                f"vs main {main_pixmap.width()}x{main_pixmap.height()}",
                5000,
            )
            return False

        self._current_compare_path = image_path
        self._canvas.compare_pixmap = pixmap
        self._canvas.update()
        return True

    def _clear_compare_pixmap(self):
        """Clear the compare pixmap from canvas."""
        self._current_compare_path = None
        self._canvas.compare_pixmap = None
        self._canvas.update()

    def set_split_position(self, position: float):
        """
        Set the split line position.

        Args:
            position: Position value from 0.0 (full compare) to 1.0 (full main).
        """
        self._canvas.split_position = max(0.0, min(1.0, position))
        self._canvas.update()

    def get_split_position(self) -> float:
        """
        Get the current split line position.

        Returns:
            float: Position value from 0.0 to 1.0.
        """
        return self._canvas.split_position

    def close(self):
        """Close the compare view and clean up resources."""
        self._compare_dir = None
        self._current_compare_path = None
        self._file_cache.clear()
        self._clear_compare_pixmap()
        self._canvas.split_position = 0.5
        self.compare_closed.emit()

    def reset(self):
        """Reset compare view state without closing (for image switching)."""
        self._clear_compare_pixmap()


class CompareViewSlider(QtWidgets.QWidget):
    """
    Slider widget for controlling the compare view split position.

    Provides a horizontal slider with close button for compare view control.
    """

    position_changed = QtCore.pyqtSignal(float)
    close_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """
        Initialize the CompareViewSlider.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the slider UI components."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)

        self._left_label = QtWidgets.QLabel("Original")
        self._left_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._left_label)

        self._slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(50)
        self._slider.setMinimumWidth(200)
        self._slider.setStyleSheet(ChatbotDialogStyle.get_slider_style())
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, 1)

        self._right_label = QtWidgets.QLabel("Compare")
        self._right_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._right_label)

        self._value_label = QtWidgets.QLabel("50%")
        self._value_label.setMinimumWidth(40)
        layout.addWidget(self._value_label)

        t = get_theme()
        self._close_btn = QtWidgets.QToolButton()
        self._close_btn.setText("\u00d7")
        self._close_btn.setToolTip("Close Compare View")
        self._close_btn.setStyleSheet(f"""
            QToolButton {{
                background-color: {t["surface"]};
                border: none;
                border-radius: 10px;
                padding: 2px 6px;
                font-size: 14px;
                font-weight: bold;
                color: {t["text_secondary"]};
            }}
            QToolButton:hover {{
                background-color: {t["background_hover"]};
                color: {t["text"]};
            }}
            QToolButton:pressed {{
                background-color: {t["border_light"]};
            }}
        """)
        self._close_btn.setFixedSize(20, 20)
        self._close_btn.clicked.connect(self.close_requested.emit)
        layout.addWidget(self._close_btn)

        self.setVisible(False)

    def _on_slider_changed(self, value: int):
        """Handle slider value change."""
        self._value_label.setText(f"{value}%")
        self.position_changed.emit(value / 100.0)

    def set_position(self, position: float):
        """
        Set the slider position without emitting signal.

        Args:
            position: Position value from 0.0 to 1.0.
        """
        self._slider.blockSignals(True)
        self._slider.setValue(int(position * 100))
        self._value_label.setText(f"{int(position * 100)}%")
        self._slider.blockSignals(False)

    def show_slider(self):
        """Show the slider widget."""
        self.setVisible(True)

    def hide_slider(self):
        """Hide the slider widget."""
        self.setVisible(False)
