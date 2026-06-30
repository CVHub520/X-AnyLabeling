"""A compact panel docked at the bottom-left of the canvas.

It groups three sliders that let the user fine tune how the annotations and the
underlying image are displayed:

- ``Opacity``    : transparency of the labels/shapes (0-100%, default 50%).
- ``Brightness`` : brightness of the image, mirroring the Brightness/Contrast
  dialog (slider 0-150, displayed as a 0.00-3.00 factor, neutral 1.00).
- ``Contrast``   : contrast of the image, same range/mapping as brightness.

The widget only emits signals; the actual rendering is handled by the owner
(:class:`LabelingWidget`), which reuses the existing brightness/contrast
pipeline so that 16-bit grayscale images keep working.
"""

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt

from ..utils.theme import get_theme


class CanvasAdjustmentWidget(QtWidgets.QWidget):
    """Three-slider panel for label opacity, image brightness and contrast."""

    # Emitted when the opacity slider changes. Value is in range 0-100.
    opacity_changed = QtCore.pyqtSignal(int)
    # Emitted when brightness or contrast changes. Values are in range 0-150
    # (same scale as BrightnessContrastDialog; factor = value / 50).
    brightness_contrast_changed = QtCore.pyqtSignal(int, int)

    SLIDER_WIDTH = 120

    # Opacity: 0-100%, neutral default centred on the bar.
    OPACITY_MIN = 0
    OPACITY_MAX = 100
    OPACITY_DEFAULT = 50

    # Brightness/contrast: identical to BrightnessContrastDialog's sliders so
    # the values and initial handle position match the menu-driven dialog.
    BC_MIN = 0
    BC_MAX = 150
    BC_DEFAULT = 50  # 50 / 50 == 1.00 (neutral)

    _LABEL_CSS = (
        "QLabel { color: #333; font-size: 11px; background: transparent; }"
    )
    _RESET_CSS = (
        "QPushButton { background: rgba(120, 120, 120, 60);"
        " border-radius: 3px; font-size: 11px; color: #333; padding: 0; }"
        "QPushButton:hover { background: rgba(120, 120, 120, 110); }"
        "QPushButton:pressed { background: rgba(120, 120, 120, 150); }"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("canvas_adjustment")
        # A bare QWidget ignores stylesheet background/border unless it is
        # told to paint a styled background; without this the translucent
        # card behind the sliders never shows.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(self._build_stylesheet())

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)

        self.opacity_slider, self.opacity_value_label = self._build_row(
            layout,
            self.tr("Opacity"),
            self.OPACITY_MIN,
            self.OPACITY_MAX,
            self.OPACITY_DEFAULT,
            display="percent",
        )
        self.brightness_slider, self.brightness_value_label = self._build_row(
            layout,
            self.tr("Brightness"),
            self.BC_MIN,
            self.BC_MAX,
            self.BC_DEFAULT,
            display="factor",
        )
        self.contrast_slider, self.contrast_value_label = self._build_row(
            layout,
            self.tr("Contrast"),
            self.BC_MIN,
            self.BC_MAX,
            self.BC_DEFAULT,
            display="factor",
        )

        self.setLayout(layout)
        self.adjustSize()

        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.brightness_slider.valueChanged.connect(self._on_bc_changed)
        self.contrast_slider.valueChanged.connect(self._on_bc_changed)

    def _build_stylesheet(self):
        """Translucent white card with theme-blue sliders."""
        primary = get_theme().get("primary", "#0071e3")
        return f"""
        #canvas_adjustment {{
            background: rgba(255, 255, 255, 220);
            border: none;
            border-radius: 6px;
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
        self, layout, title, minimum, maximum, default, display="percent"
    ):
        """Create one labelled slider row and append it to ``layout``."""
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        name_label = QtWidgets.QLabel(title)
        name_label.setFixedWidth(64)
        name_label.setStyleSheet(self._LABEL_CSS)

        slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(default)
        slider.setFixedWidth(self.SLIDER_WIDTH)
        slider.setTracking(True)
        slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        value_label = QtWidgets.QLabel()
        value_label.setProperty("display", display)
        value_label.setFixedWidth(36)
        value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        value_label.setStyleSheet(self._LABEL_CSS)
        self._set_value_text(value_label, default)

        reset_btn = QtWidgets.QPushButton("↺")
        reset_btn.setFixedSize(22, 20)
        reset_btn.setStyleSheet(self._RESET_CSS)
        reset_btn.setToolTip(self.tr("Reset to default"))
        reset_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        reset_btn.clicked.connect(
            lambda _=False, s=slider: s.setValue(default)
        )

        row.addWidget(name_label)
        row.addWidget(slider)
        row.addWidget(value_label)
        row.addWidget(reset_btn)
        layout.addLayout(row)
        return slider, value_label

    @staticmethod
    def _set_value_text(label, value):
        """Format the value for display based on the label's display mode."""
        if label.property("display") == "factor":
            label.setText(f"{value / 50:.2f}")
        else:
            label.setText(f"{value}%")

    def _on_opacity_changed(self, value):
        self._set_value_text(self.opacity_value_label, value)
        self.opacity_changed.emit(value)

    def _on_bc_changed(self, _=None):
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()
        self._set_value_text(self.brightness_value_label, brightness)
        self._set_value_text(self.contrast_value_label, contrast)
        self.brightness_contrast_changed.emit(brightness, contrast)

    def set_opacity(self, value):
        """Set the opacity slider without emitting ``opacity_changed``."""
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(value)
        self.opacity_slider.blockSignals(False)
        self._set_value_text(self.opacity_value_label, value)

    def set_brightness_contrast(self, brightness, contrast):
        """Set the brightness/contrast sliders without emitting signals."""
        for slider, value, label in (
            (self.brightness_slider, brightness, self.brightness_value_label),
            (self.contrast_slider, contrast, self.contrast_value_label),
        ):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
            self._set_value_text(label, value)
