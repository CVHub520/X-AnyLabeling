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
                    np.array(
                        [[0, 1024], [32768, 65535]], dtype=np.uint16
                    )
                )

                dialog.update_image(image)
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


if __name__ == "__main__":
    unittest.main()
