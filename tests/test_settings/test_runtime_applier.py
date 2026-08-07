import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

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


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for runtime tests")
class TestMagicWandSettingsRuntimeApplier(unittest.TestCase):

    def test_magic_wand_settings_apply_without_restart(self):
        canvas = SimpleNamespace(
            magic_wand_luminance_weight=0.5,
            _magic_wand_active=True,
            _magic_wand_distance=object(),
            _magic_wand_threshold=31,
            _update_magic_wand_preview=Mock(),
            update=Mock(),
        )
        widget = SimpleNamespace(
            canvas=canvas,
            _config={
                "canvas": {
                    "magic_wand": {
                        "default_threshold": 25,
                        "drag_sensitivity": 4.5,
                        "luminance_weight": 0.25,
                        "simplify_epsilon": 1.5,
                        "opacity": 0.8,
                    }
                }
            },
        )

        SettingsRuntimeApplier(widget).apply_change(
            "canvas.magic_wand.luminance_weight", 0.25
        )

        self.assertEqual(canvas.magic_wand_default_threshold, 25)
        self.assertEqual(canvas.magic_wand_drag_sensitivity, 4.5)
        self.assertEqual(canvas.magic_wand_luminance_weight, 0.25)
        self.assertIsNone(canvas._magic_wand_distance)
        self.assertEqual(canvas.magic_wand_simplify_epsilon_px, 1.5)
        self.assertEqual(canvas.magic_wand_opacity, 0.8)
        canvas._update_magic_wand_preview.assert_called_once_with(31)
        canvas.update.assert_not_called()

    def test_magic_wand_defaults_refresh_when_preview_is_inactive(self):
        canvas = SimpleNamespace(
            magic_wand_luminance_weight=0.5,
            _magic_wand_active=False,
            _magic_wand_distance=None,
            _magic_wand_threshold=15,
            _update_magic_wand_preview=Mock(),
            update=Mock(),
        )
        widget = SimpleNamespace(
            canvas=canvas,
            _config={
                "canvas": {
                    "magic_wand": {
                        "default_threshold": 22,
                        "drag_sensitivity": 3.0,
                        "luminance_weight": 0.5,
                        "simplify_epsilon": 0.5,
                        "opacity": 0.6,
                    }
                }
            },
        )

        SettingsRuntimeApplier(widget).apply_change(
            "canvas.magic_wand.default_threshold", 22
        )

        self.assertEqual(canvas._magic_wand_threshold, 22)
        canvas._update_magic_wand_preview.assert_not_called()
        canvas.update.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
