"""This module defines brightness/contrast dialog"""

import PIL.Image
import PIL.ImageEnhance
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage


class BrightnessContrastDialog(QtWidgets.QDialog):
    """Dialog for adjusting brightness and contrast of current image"""

    def __init__(self, img, callback, parent=None):
        super(BrightnessContrastDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle(self.tr("Brightness/Contrast"))

        self.slider_brightness = self._create_slider()
        brightness_label = QtWidgets.QLabel(
            f"{self.slider_brightness.value() / 50:.2f}"
        )
        self.slider_brightness.valueChanged.connect(
            lambda value: brightness_label.setText(f"{value / 50:.2f}")
        )
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.slider_brightness)
        layout.addWidget(brightness_label)
        brightness_widget = QtWidgets.QWidget()
        brightness_widget.setLayout(layout)

        self.slider_contrast = self._create_slider()
        contrast_label = QtWidgets.QLabel(
            f"{self.slider_contrast.value() / 50:.2f}"
        )
        self.slider_contrast.valueChanged.connect(
            lambda value: contrast_label.setText(f"{value / 50:.2f}")
        )
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.slider_contrast)
        layout.addWidget(contrast_label)
        contrast_widget = QtWidgets.QWidget()
        contrast_widget.setLayout(layout)

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow(self.tr("Brightness"), brightness_widget)
        form_layout.addRow(self.tr("Contrast"), contrast_widget)
        self.setLayout(form_layout)

        assert isinstance(img, PIL.Image.Image)
        self.img = img
        self.callback = callback

    def on_new_value(self, _):
        """On new value event"""
        brightness = self.slider_brightness.value() / 50.0
        contrast = self.slider_contrast.value() / 50.0

        img = self.img
        if brightness != 1:
            img = PIL.ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1:
            img = PIL.ImageEnhance.Contrast(img).enhance(contrast)

        qimage = QImage(
            img.tobytes(), img.width, img.height, QImage.Format_RGB888
        )
        self.callback(qimage)

    def _create_slider(self):
        """Create brightness/contrast slider"""
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setRange(0, 150)
        slider.setValue(50)
        slider.valueChanged.connect(self.on_new_value)
        return slider
