"""This module defines brightness/contrast dialog"""

import PIL.Image
import PIL.ImageEnhance
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from ..utils.image import pil_to_qimage


class BrightnessContrastDialog(QtWidgets.QDialog):
    """Dialog for adjusting brightness and contrast of current image"""

    def __init__(self, img, callback, parent=None):
        super(BrightnessContrastDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle(self.tr("Brightness/Contrast"))

        # Brightness slider and label
        self.slider_brightness = self._create_slider()
        self.brightness_label = QtWidgets.QLabel(
            f"{self.slider_brightness.value() / 50:.2f}"
        )
        self.slider_brightness.valueChanged.connect(
            self.update_brightness_label
        )

        brightness_layout = QtWidgets.QHBoxLayout()
        brightness_layout.addWidget(QtWidgets.QLabel(self.tr("Brightness: ")))
        brightness_layout.addWidget(self.slider_brightness)
        brightness_layout.addWidget(self.brightness_label)

        brightness_widget = QtWidgets.QWidget()
        brightness_widget.setLayout(brightness_layout)

        # Contrast slider and label
        self.slider_contrast = self._create_slider()
        self.contrast_label = QtWidgets.QLabel(
            f"{self.slider_contrast.value() / 50:.2f}"
        )
        self.slider_contrast.valueChanged.connect(self.update_contrast_label)

        contrast_layout = QtWidgets.QHBoxLayout()
        contrast_layout.addWidget(QtWidgets.QLabel(self.tr("Contrast:    ")))
        contrast_layout.addWidget(self.slider_contrast)
        contrast_layout.addWidget(self.contrast_label)

        contrast_widget = QtWidgets.QWidget()
        contrast_widget.setLayout(contrast_layout)

        # Reset button
        self.reset_button = QtWidgets.QPushButton(self.tr("Reset"))
        self.reset_button.clicked.connect(self.reset_values)

        # Confirm button
        self.confirm_button = QtWidgets.QPushButton(self.tr("Confirm"))
        self.confirm_button.clicked.connect(self.confirm_values)

        # Buttons layout
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.confirm_button)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(brightness_widget)
        main_layout.addWidget(contrast_widget)
        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)

        assert isinstance(img, PIL.Image.Image)
        self.img = img
        self.callback = callback

    def update_brightness_label(self, value):
        """Update brightness label"""
        self.brightness_label.setText(f"{value / 50:.2f}")
        self.on_new_value()

    def update_contrast_label(self, value):
        """Update contrast label"""
        self.contrast_label.setText(f"{value / 50:.2f}")
        self.on_new_value()

    def on_new_value(self):
        """On new value event"""
        brightness = self.slider_brightness.value() / 50.0
        contrast = self.slider_contrast.value() / 50.0

        img = self.img
        if brightness != 1:
            img = PIL.ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1:
            img = PIL.ImageEnhance.Contrast(img).enhance(contrast)

        qimage = pil_to_qimage(img)
        self.callback(qimage)

    def reset_values(self):
        """Reset sliders to default values"""
        self.slider_brightness.setValue(50)
        self.slider_contrast.setValue(50)
        self.on_new_value()

    def confirm_values(self):
        """Confirm the current values and close the dialog"""
        self.accept()

    def _create_slider(self):
        """Create brightness/contrast slider"""
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setRange(0, 150)
        slider.setValue(50)
        return slider
