import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtGui, QtWidgets

    from anylabeling.views.labeling.settings.runtime_applier import (
        SettingsRuntimeApplier,
    )

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for runtime tests")
class TestSettingsRuntimeApplier(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.original_font = QtGui.QFont(self.app.font())
        self.app._xanylabeling_default_font = QtGui.QFont(self.original_font)

    def tearDown(self):
        self.app.setFont(self.original_font)
        self.app._xanylabeling_default_font = QtGui.QFont(self.original_font)

    def test_font_family_applies_and_restores_without_restart(self):
        families = QtGui.QFontDatabase.families()
        if not families:
            self.skipTest("No font families are available")
        widget = QtWidgets.QWidget()
        applier = SettingsRuntimeApplier(widget)

        applier.apply_change("font_family", families[0])
        self.app.processEvents()
        self.assertEqual(self.app.font().family(), families[0])
        self.assertEqual(widget.font().family(), families[0])
        self.assertEqual(
            self.app.font().pointSizeF(), self.original_font.pointSizeF()
        )

        applier.apply_change("font_family", None)
        self.app.processEvents()
        self.assertEqual(self.app.font().family(), self.original_font.family())


if __name__ == "__main__":
    unittest.main()
