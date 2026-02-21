"""This module defines brightness/contrast dialog"""

import PIL.Image
import PIL.ImageEnhance
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from ..utils.image import pil_to_qimage
from ..utils.style import (
    get_dialog_style,
    get_ok_btn_style,
    get_cancel_btn_style,
)


class BrightnessContrastDialog(QtWidgets.QDialog):
    """Dialog for adjusting brightness and contrast of current image"""

    def __init__(self, callback, parent=None):
        super(BrightnessContrastDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle(self.tr("Brightness/Contrast"))
        self.setFixedSize(400, 160)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )

        self.setStyleSheet(get_dialog_style())

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        self.slider_brightness = self._create_slider()
        self.slider_contrast = self._create_slider()

        self.brightness_label = QtWidgets.QLabel(
            f"{self.slider_brightness.value() / 50:.2f}"
        )
        self.brightness_label.setFixedWidth(40)
        self.brightness_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.contrast_label = QtWidgets.QLabel(
            f"{self.slider_contrast.value() / 50:.2f}"
        )
        self.contrast_label.setFixedWidth(40)
        self.contrast_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.slider_brightness.valueChanged.connect(
            self.update_brightness_label
        )
        self.slider_contrast.valueChanged.connect(self.update_contrast_label)

        # Reset button
        self.reset_button = QtWidgets.QPushButton(self.tr("Reset"))
        self.reset_button.clicked.connect(self.reset_values)
        self.reset_button.setStyleSheet(get_cancel_btn_style())

        # Confirm button
        self.confirm_button = QtWidgets.QPushButton(self.tr("Confirm"))
        self.confirm_button.clicked.connect(self.confirm_values)
        self.confirm_button.setStyleSheet(get_ok_btn_style())

        # Grid: col 0 = fixed labels, col 1 = sliders + buttons (same column)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(16)
        grid.setColumnStretch(1, 1)

        brightness_name = QtWidgets.QLabel(self.tr("Brightness:"))
        brightness_name.setFixedWidth(85)
        contrast_name = QtWidgets.QLabel(self.tr("Contrast:"))
        contrast_name.setFixedWidth(85)

        b_row = QtWidgets.QHBoxLayout()
        b_row.setSpacing(10)
        b_row.addWidget(self.slider_brightness)
        b_row.addWidget(self.brightness_label)

        c_row = QtWidgets.QHBoxLayout()
        c_row.setSpacing(10)
        c_row.addWidget(self.slider_contrast)
        c_row.addWidget(self.contrast_label)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()
        btn_row.addWidget(self.reset_button)
        btn_row.addWidget(self.confirm_button)

        grid.addWidget(brightness_name, 0, 0, Qt.AlignVCenter)
        grid.addLayout(b_row, 0, 1)
        grid.addWidget(contrast_name, 1, 0, Qt.AlignVCenter)
        grid.addLayout(c_row, 1, 1)
        grid.addLayout(btn_row, 2, 1)

        main_layout.addLayout(grid)
        self.setLayout(main_layout)
        self.callback = callback

        # Center the dialog on the screen
        self.move_to_center()

    def move_to_center(self):
        """Move dialog to center of the screen"""
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def update_image(self, image):
        """Update image instance"""
        assert isinstance(image, PIL.Image.Image)
        self.img = image

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
        slider.setTracking(True)
        slider.setFixedHeight(28)
        return slider
