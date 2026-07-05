import os
import unittest

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtGui, QtWidgets

    from anylabeling.views.labeling.widgets.brightness_contrast_dialog import (
        BrightnessContrastDialog,
    )

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for brightness/contrast tests"
)
class TestBrightnessContrastDialog(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self._widgets = []

    def tearDown(self):
        for widget in self._widgets:
            widget.close()
        self.app.processEvents()

    def test_adjusts_16_bit_grayscale_image_without_crashing(self):
        for slider_name in ("slider_brightness", "slider_contrast"):
            with self.subTest(slider_name=slider_name):
                images = []
                dialog = BrightnessContrastDialog(images.append)
                self._widgets.append(dialog)
                image = Image.fromarray(
                    np.array([[0, 1024], [32768, 65535]], dtype=np.uint16)
                )

                dialog.update_image(image)
                self.assertIsNone(dialog._grayscale16_data)
                self.assertIsNone(dialog._grayscale16_mean)
                slider = getattr(dialog, slider_name)
                slider.blockSignals(True)
                slider.setValue(60)
                slider.blockSignals(False)
                dialog.on_new_value()

                self.assertEqual(len(images), 1)
                self.assertFalse(images[0].isNull())
                self.assertEqual(
                    images[0].format(),
                    QtGui.QImage.Format.Format_Grayscale16,
                )
                self.assertIsNotNone(dialog._grayscale16_data)
                self.assertIsNotNone(dialog._grayscale16_mean)

    def test_set_values_synchronizes_sliders_and_labels(self):
        dialog = BrightnessContrastDialog(lambda _: None)
        self._widgets.append(dialog)

        dialog.set_values(60, 70)

        self.assertEqual(dialog.slider_brightness.value(), 60)
        self.assertEqual(dialog.slider_contrast.value(), 70)
        self.assertEqual(dialog.brightness_label.text(), "1.20")
        self.assertEqual(dialog.contrast_label.text(), "1.40")

    def test_clear_image_releases_image_resources(self):
        dialog = BrightnessContrastDialog(lambda _: None)
        self._widgets.append(dialog)
        image = Image.fromarray(
            np.array([[0, 1024], [32768, 65535]], dtype=np.uint16)
        )
        dialog.update_image(image)

        dialog.clear_image()

        self.assertIsNone(dialog.img)
        self.assertIsNone(dialog._grayscale16_data)
        self.assertIsNone(dialog._grayscale16_mean)
        self.assertIsNone(dialog._grayscale16_qimage_data)


if __name__ == "__main__":
    unittest.main()
