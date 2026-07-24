import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtTest, QtWidgets

    from anylabeling.resources import resources  # noqa: F401
    from anylabeling.views.labeling.widgets.canvas_adjustment import (
        CanvasAdjustmentWidget,
    )

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for canvas adjustment tests"
)
class TestCanvasAdjustmentWidget(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.widget = CanvasAdjustmentWidget()
        self.widget.show()
        self.app.processEvents()

    def tearDown(self):
        self.widget.close()
        self.app.processEvents()

    def test_defaults_and_tooltips_describe_adjustment_targets(self):
        self.assertEqual(self.widget.opacity_slider.value(), 100)
        self.assertIn("shapes", self.widget.opacity_slider.toolTip())
        self.assertIn("Label text", self.widget.opacity_slider.toolTip())
        self.assertIn("image", self.widget.brightness_slider.toolTip())
        self.assertIn("image", self.widget.contrast_slider.toolTip())

    def test_toggle_collapses_to_button_and_restores_content(self):
        geometry_spy = QtTest.QSignalSpy(self.widget.geometry_changed)

        self.widget.toggle_button.click()
        self.app.processEvents()

        self.assertTrue(self.widget.content_widget.isHidden())
        self.assertTrue(self.widget.title_label.isHidden())
        self.assertTrue(self.widget.toggle_button.isVisible())
        self.assertEqual(len(geometry_spy), 1)
        self.assertIn("Expand", self.widget.toggle_button.toolTip())

        self.widget.toggle_button.click()
        self.app.processEvents()

        self.assertTrue(self.widget.content_widget.isVisible())
        self.assertTrue(self.widget.title_label.isVisible())
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


if __name__ == "__main__":
    unittest.main()
