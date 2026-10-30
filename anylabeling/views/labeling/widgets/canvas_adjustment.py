"""A compact panel docked at the bottom-left of the canvas.

It groups three sliders that let the user fine tune how the annotations and the
underlying image are displayed:

- ``Opacity``    : transparency of the labels/shapes (0-100%, default 100%).
- ``Brightness`` : brightness of the image, mirroring the Brightness/Contrast
  dialog (slider 0-150, displayed as a 0.00-3.00 factor, neutral 1.00).
- ``Contrast``   : contrast of the image, same range/mapping as brightness.

The widget only emits signals; the actual rendering is handled by the owner
(:class:`LabelingWidget`), which reuses the brightness/contrast processor so
that 16-bit grayscale images keep working.
"""

import PIL.Image
import PIL.ImageEnhance
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from ..pointcloud.icons import get_icon
from ..utils.image import pil_to_qimage
from ..utils.qt import new_icon
from ..utils.theme import get_theme

GRAYSCALE_16BIT_MODES = {"I;16", "I;16L", "I;16B"}


class BrightnessContrastProcessor:
    """Apply display-only brightness and contrast adjustments to an image."""

    def __init__(self):
        self.img = None
        self._grayscale16_data = None
        self._grayscale16_mean = None
        self._grayscale16_qimage_data = None

    def update_image(self, image):
        """Update the source image."""
        assert isinstance(image, PIL.Image.Image)
        self.clear_image()
        self.img = image

    def clear_image(self):
        """Release resources associated with the current image."""
        if self.img is not None:
            self.img.close()
        self.img = None
        self._grayscale16_data = None
        self._grayscale16_mean = None
        self._grayscale16_qimage_data = None

    def adjust(self, brightness, contrast):
        """Return an adjusted image for the requested display factors."""
        brightness = brightness / 50.0
        contrast = contrast / 50.0

        if (
            self.img.mode in GRAYSCALE_16BIT_MODES
            and self._grayscale16_data is None
        ):
            self._initialize_grayscale16()
        if self._grayscale16_data is not None:
            return self._enhance_grayscale16(brightness, contrast)

        img = self.img
        if brightness != 1:
            img = PIL.ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1:
            img = PIL.ImageEnhance.Contrast(img).enhance(contrast)
        return pil_to_qimage(img)

    def _initialize_grayscale16(self):
        """Initialize data used for 16-bit grayscale enhancement."""
        data = np.asarray(self.img).astype(np.uint16, copy=False)
        if data.ndim == 2:
            self._grayscale16_data = np.ascontiguousarray(data)
            self._grayscale16_mean = float(self._grayscale16_data.mean())

    def _enhance_grayscale16(self, brightness, contrast):
        """Enhance a 16-bit grayscale image and return a QImage."""
        if brightness == 1 and contrast == 1:
            self._grayscale16_qimage_data = self._grayscale16_data
        else:
            data = self._grayscale16_data.astype(np.float32)
            data *= brightness * contrast
            data += self._grayscale16_mean * brightness * (1 - contrast)
            np.clip(data, 0, np.iinfo(np.uint16).max, out=data)
            self._grayscale16_qimage_data = data.astype(np.uint16)

        height, width = self._grayscale16_qimage_data.shape
        return QtGui.QImage(
            self._grayscale16_qimage_data,
            width,
            height,
            self._grayscale16_qimage_data.strides[0],
            QtGui.QImage.Format.Format_Grayscale16,
        )


class CanvasAdjustmentWidget(QtWidgets.QWidget):
    """Three-slider panel for label opacity, image brightness and contrast."""

    # Emitted when the opacity slider changes. Value is in range 0-100.
    opacity_changed = QtCore.pyqtSignal(int)
    # Emitted when brightness or contrast changes. Values are in range 0-150
    # (factor = value / 50).
    brightness_contrast_changed = QtCore.pyqtSignal(int, int)
    geometry_changed = QtCore.pyqtSignal()
    status_changed = QtCore.pyqtSignal(object)

    SLIDER_WIDTH = 90
    BC_UPDATE_INTERVAL_MS = 33

    # Opacity: 0-100%, fully opaque by default.
    OPACITY_MIN = 0
    OPACITY_MAX = 100
    OPACITY_DEFAULT = 100

    # Brightness/contrast factors use 50 as the neutral value.
    BC_MIN = 0
    BC_MAX = 150
    BC_DEFAULT = 50  # 50 / 50 == 1.00 (neutral)

    _LABEL_CSS = (
        "QLabel { color: #333; font-size: 11px; background: transparent; }"
    )
    _ROW_BUTTON_CSS = (
        "QToolButton { background: transparent;"
        " border: none; border-radius: 3px; padding: 0; }"
        "QToolButton:hover { background: rgba(120, 120, 120, 110); }"
        "QToolButton:pressed { background: rgba(120, 120, 120, 150); }"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("canvas_adjustment")
        self._collapsed = False
        self.setProperty("collapsed", False)
        # A bare QWidget ignores stylesheet background/border unless it is
        # told to paint a styled background; without this the translucent
        # card behind the sliders never shows.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(self._build_stylesheet())

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        self.toggle_button = QtWidgets.QToolButton()
        self.toggle_button.setIcon(new_icon("caret-down", "svg"))
        self.toggle_button.setIconSize(QtCore.QSize(10, 10))
        self.toggle_button.setFixedSize(20, 20)
        self.toggle_button.setToolTip(self.tr("Collapse adjustments"))
        self.toggle_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.toggle_button.setStyleSheet(
            "QToolButton { background: transparent; border: none;"
            " border-radius: 3px; padding: 4px; }"
            "QToolButton:hover { background: rgba(120, 120, 120, 60); }"
            "QToolButton:pressed { background: rgba(120, 120, 120, 100); }"
        )
        header_layout.addWidget(self.toggle_button)
        header_layout.addStretch()

        self.content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(5)
        self.content_widget.setLayout(content_layout)

        self.opacity_slider, self.opacity_value_label, self.opacity_pin = (
            self._build_row(
                content_layout,
                self.tr("Opacity"),
                self.OPACITY_MIN,
                self.OPACITY_MAX,
                self.OPACITY_DEFAULT,
                self.tr(
                    "Adjust the transparency of annotation shapes and masks. "
                    "Label text remains fully visible."
                ),
                display="opacity",
            )
        )
        (
            self.brightness_slider,
            self.brightness_value_label,
            self.brightness_pin,
        ) = self._build_row(
            content_layout,
            self.tr("Brightness"),
            self.BC_MIN,
            self.BC_MAX,
            self.BC_DEFAULT,
            self.tr("Adjust the brightness of the underlying image."),
            display="factor",
        )
        self.contrast_slider, self.contrast_value_label, self.contrast_pin = (
            self._build_row(
                content_layout,
                self.tr("Contrast"),
                self.BC_MIN,
                self.BC_MAX,
                self.BC_DEFAULT,
                self.tr("Adjust the contrast of the underlying image."),
                display="factor",
            )
        )
        status_layout = QtWidgets.QHBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(4)
        status_label = QtWidgets.QLabel(self.tr("Status"))
        status_label.setFixedWidth(64)
        status_label.setStyleSheet(self._LABEL_CSS)
        status_layout.addWidget(status_label)
        self.status_buttons = {}
        for status, text, name in (
            (False, self.tr("Pending"), "pending"),
            (True, self.tr("Passed"), "passed"),
            ("failed", self.tr("Failed"), "failed"),
        ):
            button = QtWidgets.QRadioButton(text)
            button.setObjectName(name)
            button.clicked.connect(
                lambda _checked, value=status: self.status_changed.emit(value)
            )
            if self.status_buttons:
                status_layout.addStretch()
            self.status_buttons[status] = button
            status_layout.addWidget(button)
        content_layout.addLayout(status_layout)
        self.set_status(False)
        layout.addWidget(self.content_widget)
        layout.addLayout(header_layout)

        self.setLayout(layout)
        self.adjustSize()

        self._bc_update_timer = QtCore.QTimer(self)
        self._bc_update_timer.setSingleShot(True)
        self._bc_update_timer.setInterval(self.BC_UPDATE_INTERVAL_MS)
        self._bc_update_timer.timeout.connect(self._emit_bc_changed)

        self.toggle_button.clicked.connect(self._toggle_collapsed)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.brightness_slider.valueChanged.connect(self._on_bc_changed)
        self.contrast_slider.valueChanged.connect(self._on_bc_changed)

    def _build_stylesheet(self):
        """Translucent white card with theme-blue sliders."""
        theme = get_theme()
        primary = theme["primary"]
        return f"""
        #canvas_adjustment {{
            background: rgba(255, 255, 255, 220);
            border: none;
            border-radius: 6px;
        }}
        #canvas_adjustment[collapsed="true"] {{
            background: transparent;
        }}
        #canvas_adjustment QRadioButton {{
            color: #333;
            font-size: 11px;
            background: transparent;
            spacing: 3px;
        }}
        #canvas_adjustment QRadioButton::indicator {{
            width: 12px;
            height: 12px;
            border: 1px solid #666;
            border-radius: 7px;
            background: white;
        }}
        #canvas_adjustment QRadioButton::indicator:checked {{
            border: 4px solid {primary};
            width: 6px;
            height: 6px;
            border-radius: 7px;
            background: white;
        }}
        #canvas_adjustment QRadioButton#passed::indicator:checked {{
            border-color: {theme["success"]};
        }}
        #canvas_adjustment QRadioButton#failed::indicator:checked {{
            border-color: {theme["error"]};
        }}
        #canvas_adjustment QSlider::groove:horizontal {{
            height: 4px;
            background: rgba(0, 0, 0, 55);
            border-radius: 2px;
        }}
        #canvas_adjustment QSlider::handle:horizontal {{
            background: {primary};
            border: none;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}
        #canvas_adjustment QSlider::sub-page:horizontal {{
            background: {primary};
            border-radius: 2px;
        }}
        """

    def _build_row(
        self,
        layout,
        title,
        minimum,
        maximum,
        default,
        tooltip,
        display="percent",
    ):
        """Create one labelled slider row and append it to ``layout``."""
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        name_label = QtWidgets.QLabel(title)
        name_label.setFixedWidth(64)
        name_label.setStyleSheet(self._LABEL_CSS)
        name_label.setToolTip(tooltip)

        slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(default)
        slider.setFixedWidth(self.SLIDER_WIDTH)
        slider.setTracking(True)
        slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        slider.setToolTip(tooltip)

        value_label = QtWidgets.QLabel()
        value_label.setProperty("display", display)
        value_label.setFixedWidth(36)
        value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        value_label.setStyleSheet(self._LABEL_CSS)
        value_label.setToolTip(tooltip)
        self._set_value_text(value_label, default)

        reset_btn = QtWidgets.QToolButton()
        reset_btn.setFixedSize(22, 20)
        reset_btn.setIcon(get_icon("rotate-cw", "#333333", "#333333"))
        reset_btn.setIconSize(QtCore.QSize(14, 14))
        reset_btn.setStyleSheet(self._ROW_BUTTON_CSS)
        reset_btn.setToolTip(self.tr("Reset to default"))
        reset_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        reset_btn.clicked.connect(
            lambda _=False, s=slider: s.setValue(default)
        )

        pin_btn = QtWidgets.QToolButton()
        pin_btn.setCheckable(True)
        pin_btn.setFixedSize(22, 20)
        pin_btn.setIconSize(QtCore.QSize(14, 14))
        pin_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        pin_btn.setToolTip(self.tr("Keep this value when switching images"))
        pin_btn.setAccessibleName(pin_btn.toolTip())
        pin_btn.setStyleSheet(self._ROW_BUTTON_CSS)
        pin_btn.setIcon(get_icon("unlock", "#333333", "#333333"))
        pin_btn.toggled.connect(
            lambda checked: pin_btn.setIcon(
                get_icon("lock" if checked else "unlock", "#333333", "#333333")
            )
        )

        row.addWidget(name_label)
        row.addWidget(slider)
        row.addWidget(value_label)
        row.addWidget(reset_btn)
        row.addWidget(pin_btn)
        row.addSpacing(4)
        layout.addLayout(row)
        return slider, value_label, pin_btn

    @staticmethod
    def _set_value_text(label, value):
        """Format the value for display based on the label's display mode."""
        if label.property("display") == "factor":
            label.setText(f"{value / 50:.2f}")
        elif label.property("display") == "opacity":
            label.setText(f"{value / 100:.2f}")
        else:
            label.setText(f"{value}%")

    def _on_opacity_changed(self, value):
        self._set_value_text(self.opacity_value_label, value)
        self.opacity_changed.emit(value)

    def _on_bc_changed(self, _=None):
        self._set_value_text(
            self.brightness_value_label, self.brightness_slider.value()
        )
        self._set_value_text(
            self.contrast_value_label, self.contrast_slider.value()
        )
        if not self._bc_update_timer.isActive():
            self._bc_update_timer.start()

    def _emit_bc_changed(self):
        self.brightness_contrast_changed.emit(
            self.brightness_slider.value(), self.contrast_slider.value()
        )

    def _toggle_collapsed(self):
        self._collapsed = not self._collapsed
        self.setProperty("collapsed", self._collapsed)
        self.content_widget.setVisible(not self._collapsed)
        icon = "caret-up" if self._collapsed else "caret-down"
        self.toggle_button.setIcon(new_icon(icon, "svg"))
        tooltip = (
            self.tr("Expand adjustments")
            if self._collapsed
            else self.tr("Collapse adjustments")
        )
        self.toggle_button.setToolTip(tooltip)
        self.style().unpolish(self)
        self.style().polish(self)
        self.layout().activate()
        self.adjustSize()
        self.geometry_changed.emit()

    def set_status(self, status):
        self.status_buttons[status].setChecked(True)

    def set_opacity(self, value):
        """Set the opacity slider without emitting ``opacity_changed``."""
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(value)
        self.opacity_slider.blockSignals(False)
        self._set_value_text(self.opacity_value_label, value)

    def set_brightness_contrast(self, brightness, contrast):
        """Set the brightness/contrast sliders without emitting signals."""
        self._bc_update_timer.stop()
        for slider, value, label in (
            (self.brightness_slider, brightness, self.brightness_value_label),
            (self.contrast_slider, contrast, self.contrast_value_label),
        ):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
            self._set_value_text(label, value)
