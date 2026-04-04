import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtGui, QtWidgets

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for label widget metrics tests"
)
class TestLabelWidgetMetrics(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

    def test_measure_text_width_matches_horizontal_advance(self):
        from anylabeling.views.labeling.label_widget import _measure_text_width

        font = self.app.font()
        metrics = QtGui.QFontMetrics(font)

        self.assertEqual(
            _measure_text_width(metrics, "bodyColor"),
            metrics.horizontalAdvance("bodyColor"),
        )

    def test_measure_text_width_falls_back_to_width(self):
        from anylabeling.views.labeling.label_widget import _measure_text_width

        class LegacyFontMetrics:
            def width(self, text):
                return len(text) * 7

        metrics = LegacyFontMetrics()
        self.assertEqual(_measure_text_width(metrics, "vehicle"), 49)
