"""This module defines brightness/contrast dialog"""

import PIL.Image
import PIL.ImageEnhance
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt

from .. import utils


class BrightnessContrastDialog(QtWidgets.QDialog):
    """Dialog for adjusting brightness and contrast of current image"""

    def __init__(self, img, callback, parent=None):
        super(BrightnessContrastDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle(self.tr("Brightness/Contrast"))

        self.slider_brightness = self._create_slider()
        self.slider_contrast = self._create_slider()

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow(self.tr("Brightness"), self.slider_brightness)
        form_layout.addRow(self.tr("Contrast"), self.slider_contrast)
        self.setLayout(form_layout)

        assert isinstance(img, PIL.Image.Image)
        self.img = img
        self.callback = callback

    def on_new_value(self, value):
        """On new value event"""
        brightness = self.slider_brightness.value() / 50.0
        contrast = self.slider_contrast.value() / 50.0

        img = self.img
        img = PIL.ImageEnhance.Brightness(img).enhance(brightness)
        img = PIL.ImageEnhance.Contrast(img).enhance(contrast)

        img_data = utils.img_pil_to_data(img)
        qimage = QtGui.QImage.fromData(img_data)
        self.callback(qimage)

    def _create_slider(self):
        """Create brightness/contrast slider"""
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setRange(0, 150)
        slider.setValue(50)
        slider.valueChanged.connect(self.on_new_value)
        return slider
