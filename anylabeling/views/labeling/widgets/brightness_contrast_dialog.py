"""This module defines brightness/contrast dialog"""

import PIL.Image
import PIL.ImageEnhance
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from ..utils.image import pil_to_qimage


class BrightnessContrastDialog(QtWidgets.QDialog):
    """Dialog for adjusting brightness and contrast of current image"""

    def __init__(self, callback, parent=None):
        super(BrightnessContrastDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle(self.tr("Brightness/Contrast"))
        self.setFixedSize(350, 160)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )

        self.setStyleSheet(
            """
            QDialog {
                background-color: #f5f5f7;
                border-radius: 10px;
            }
            QLabel {
                color: #1d1d1f;
                font-size: 13px;
            }
            QSlider {
                height: 28px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #d2d2d7;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0071e3;
                border: none;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #0071e3;
                border-radius: 2px;
            }
        """
        )

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Brightness slider and label
        brightness_layout = QtWidgets.QHBoxLayout()
        brightness_layout.setSpacing(10)

        brightness_label = QtWidgets.QLabel(self.tr("Brightness:"))
        brightness_label.setMinimumWidth(85)
        brightness_layout.addWidget(brightness_label)

        self.slider_brightness = self._create_slider()
        brightness_layout.addWidget(self.slider_brightness)

        self.brightness_label = QtWidgets.QLabel(
            f"{self.slider_brightness.value() / 50:.2f}"
        )
        self.brightness_label.setFixedWidth(40)
        self.brightness_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        brightness_layout.addWidget(self.brightness_label)

        self.slider_brightness.valueChanged.connect(
            self.update_brightness_label
        )

        # Contrast slider and label
        contrast_layout = QtWidgets.QHBoxLayout()
        contrast_layout.setSpacing(10)

        contrast_label = QtWidgets.QLabel(self.tr("Contrast:"))
        contrast_label.setMinimumWidth(85)
        contrast_layout.addWidget(contrast_label)

        self.slider_contrast = self._create_slider()
        contrast_layout.addWidget(self.slider_contrast)

        self.contrast_label = QtWidgets.QLabel(
            f"{self.slider_contrast.value() / 50:.2f}"
        )
        self.contrast_label.setFixedWidth(40)
        self.contrast_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        contrast_layout.addWidget(self.contrast_label)

        self.slider_contrast.valueChanged.connect(self.update_contrast_label)

        # Add layouts to main layout
        main_layout.addLayout(brightness_layout)
        main_layout.addLayout(contrast_layout)
        main_layout.addSpacing(5)

        # Buttons layout
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.setSpacing(8)

        # Reset button
        self.reset_button = QtWidgets.QPushButton(self.tr("Reset"))
        self.reset_button.setFixedSize(100, 32)
        self.reset_button.clicked.connect(self.reset_values)
        self.reset_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f5f5f7;
                color: #1d1d1f;
                border: none;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #e5e5e5;
            }
            QPushButton:pressed {
                background-color: #d5d5d5;
            }
        """
        )

        # Confirm button
        self.confirm_button = QtWidgets.QPushButton(self.tr("Confirm"))
        self.confirm_button.setFixedSize(100, 32)
        self.confirm_button.clicked.connect(self.confirm_values)
        self.confirm_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0071e3;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0077ED;
            }
            QPushButton:pressed {
                background-color: #0068D0;
            }
        """
        )

        buttons_layout.addStretch()
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.confirm_button)

        main_layout.addLayout(buttons_layout)
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
