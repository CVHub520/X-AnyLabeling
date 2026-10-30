import os
import unittest

import numpy as np
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtGui, QtTest, QtWidgets

    from anylabeling.resources import resources  # noqa: F401
    from anylabeling.views.labeling.widgets.canvas_adjustment import (
        BrightnessContrastProcessor,
        CanvasAdjustmentWidget,
    )

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for canvas adjustment tests"
)
class TestCanvasAdjustmentWidget(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )

    def setUp(self):
        self.widget = CanvasAdjustmentWidget()
        self.widget.show()
        self.app.processEvents()

    def tearDown(self):
        self.widget.close()
        self.widget.deleteLater()
        self.app.processEvents()

    def test_defaults_and_tooltips_describe_adjustment_targets(self):
        self.assertEqual(self.widget.opacity_slider.value(), 100)
        self.assertIn("shapes", self.widget.opacity_slider.toolTip())
        self.assertIn("Label text", self.widget.opacity_slider.toolTip())
        self.assertIn("image", self.widget.brightness_slider.toolTip())
        self.assertIn("image", self.widget.contrast_slider.toolTip())

    def test_status_selection_and_programmatic_sync(self):
        signal_spy = QtTest.QSignalSpy(self.widget.status_changed)
        self.assertTrue(self.widget.status_buttons[False].isChecked())
        self.widget.status_buttons["failed"].click()
        self.assertEqual(list(signal_spy[0]), ["failed"])
        self.assertTrue(self.widget.status_buttons["failed"].isChecked())
        self.widget.set_status(True)
        self.assertTrue(self.widget.status_buttons[True].isChecked())
        self.assertFalse(self.widget.status_buttons["failed"].isChecked())
        self.assertEqual(len(signal_spy), 1)

    def test_toggle_collapses_to_button_and_restores_content(self):
        geometry_spy = QtTest.QSignalSpy(self.widget.geometry_changed)
        expanded_size = self.widget.sizeHint()

        self.widget.toggle_button.click()
        self.app.processEvents()

        self.assertTrue(self.widget.content_widget.isHidden())
        self.assertTrue(self.widget.toggle_button.isVisible())
        self.assertTrue(self.widget.property("collapsed"))
        margins = self.widget.layout().contentsMargins()
        self.assertEqual(
            self.widget.sizeHint().width(),
            self.widget.toggle_button.width()
            + margins.left()
            + margins.right(),
        )
        self.assertEqual(
            self.widget.sizeHint().height(),
            self.widget.toggle_button.height()
            + margins.top()
            + margins.bottom(),
        )
        self.assertEqual(len(geometry_spy), 1)
        self.assertIn("Expand", self.widget.toggle_button.toolTip())

        self.widget.toggle_button.click()
        self.app.processEvents()

        self.assertTrue(self.widget.content_widget.isVisible())
        self.assertFalse(self.widget.property("collapsed"))
        self.assertEqual(self.widget.sizeHint(), expanded_size)
        self.assertIn("Collapse", self.widget.toggle_button.toolTip())

    def test_brightness_contrast_updates_are_throttled(self):
        signal_spy = QtTest.QSignalSpy(self.widget.brightness_contrast_changed)

        self.widget.brightness_slider.setValue(51)
        self.widget.brightness_slider.setValue(52)
        self.widget.brightness_slider.setValue(53)

        self.assertEqual(len(signal_spy), 0)
        QtTest.QTest.qWait(self.widget.BC_UPDATE_INTERVAL_MS + 10)
        self.assertEqual(len(signal_spy), 1)
        self.assertEqual(list(signal_spy[0]), [53, 50])

    def test_programmatic_update_cancels_pending_adjustment(self):
        signal_spy = QtTest.QSignalSpy(self.widget.brightness_contrast_changed)

        self.widget.brightness_slider.setValue(51)
        self.widget.set_brightness_contrast(60, 70)
        QtTest.QTest.qWait(self.widget.BC_UPDATE_INTERVAL_MS + 10)

        self.assertEqual(len(signal_spy), 0)
        self.assertEqual(self.widget.brightness_value_label.text(), "1.20")
        self.assertEqual(self.widget.contrast_value_label.text(), "1.40")


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for brightness/contrast tests"
)
class TestBrightnessContrastProcessor(unittest.TestCase):

    def test_adjusts_16_bit_grayscale_image(self):
        for brightness, contrast in ((60, 50), (50, 60)):
            with self.subTest(brightness=brightness, contrast=contrast):
                processor = BrightnessContrastProcessor()
                image = Image.fromarray(
                    np.array([[0, 1024], [32768, 65535]], dtype=np.uint16)
                )

                processor.update_image(image)
                adjusted = processor.adjust(brightness, contrast)

                self.assertFalse(adjusted.isNull())
                self.assertEqual(
                    adjusted.format(), QtGui.QImage.Format.Format_Grayscale16
                )
                self.assertIsNotNone(processor._grayscale16_data)
                self.assertIsNotNone(processor._grayscale16_mean)

    def test_clear_image_releases_image_resources(self):
        processor = BrightnessContrastProcessor()
        image = Image.fromarray(
            np.array([[0, 1024], [32768, 65535]], dtype=np.uint16)
        )
        processor.update_image(image)

        processor.clear_image()

        self.assertIsNone(processor.img)
        self.assertIsNone(processor._grayscale16_data)
        self.assertIsNone(processor._grayscale16_mean)
        self.assertIsNone(processor._grayscale16_qimage_data)


if __name__ == "__main__":
    unittest.main()
