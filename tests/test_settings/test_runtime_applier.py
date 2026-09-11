import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtGui, QtWidgets

    from anylabeling.views.labeling.shape import Shape
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

    def test_label_font_size_applies_without_restart(self):
        canvas = SimpleNamespace(
            label_font_size=8,
            epsilon=10.0,
            double_click="close",
            double_click_edit_label=True,
            num_backups=10,
            update=Mock(),
        )
        widget = SimpleNamespace(
            canvas=canvas,
            _config={
                "canvas": {
                    "label_font_size": 12,
                    "epsilon": 10.0,
                    "double_click": "close",
                    "double_click_edit_label": True,
                    "num_backups": 10,
                }
            },
        )

        SettingsRuntimeApplier(widget).apply_change(
            "canvas.label_font_size", 12
        )

        self.assertEqual(canvas.label_font_size, 12)
        canvas.update.assert_called_once_with()

    def test_width_previews_use_pending_values(self):
        config = {
            "shape": {
                "line_color": [0, 255, 0, 128],
                "fill_color": [0, 255, 0, 128],
                "vertex_fill_color": [0, 255, 0, 255],
                "select_line_color": [255, 255, 255, 255],
                "select_fill_color": [0, 255, 0, 155],
                "hvertex_fill_color": [255, 255, 255, 255],
                "point_size": 4,
                "line_width": 4,
            },
            "canvas": {
                "crosshair": {
                    "show": True,
                    "width": 2.0,
                    "color": "#00FF00",
                    "opacity": 0.5,
                }
            },
        }
        canvas = SimpleNamespace(
            set_cross_line=Mock(), update=Mock(), shapes=[]
        )
        widget = SimpleNamespace(
            canvas=canvas,
            _config=config,
            crosshair_settings=dict(config["canvas"]["crosshair"]),
        )
        applier = SettingsRuntimeApplier(widget)
        original_line_width = Shape.line_width

        try:
            applier.apply_change("shape.line_width", 0.5)
            self.assertEqual(applier._widget.canvas.update.call_count, 1)
            self.assertEqual(Shape.line_width, 0.5)

            applier.apply_change("canvas.crosshair.width", 0.5)
            canvas.set_cross_line.assert_called_once_with(
                True, 0.5, "#00FF00", 0.5
            )
            self.assertEqual(widget.crosshair_settings["width"], 0.5)
        finally:
            Shape.line_width = original_line_width

    def test_pixel_precision_settings_apply_without_restart(self):
        pixel_precision = {
            "enabled": True,
            "disable_smoothing_scale": 5.0,
            "show_pixel_grid": False,
            "pixel_grid_min_scale": 5.0,
            "snap_enabled": False,
            "snap_step": 0.5,
            "max_zoom_percent": 6400,
        }
        canvas = SimpleNamespace(
            label_font_size=8,
            epsilon=10.0,
            double_click="close",
            double_click_edit_label=True,
            num_backups=10,
            configure_pixel_precision=Mock(),
            update=Mock(),
        )
        actions = SimpleNamespace(
            show_pixel_grid=QtGui.QAction(),
            pixel_snap=QtGui.QAction(),
        )
        widget = SimpleNamespace(
            canvas=canvas,
            actions=actions,
            zoom_widget=QtWidgets.QSpinBox(),
            navigator_dialog=SimpleNamespace(set_maximum_zoom=Mock()),
            _config={
                "canvas": {
                    "label_font_size": 8,
                    "epsilon": 10.0,
                    "double_click": "close",
                    "double_click_edit_label": True,
                    "num_backups": 10,
                    "pixel_precision": pixel_precision,
                }
            },
        )

        SettingsRuntimeApplier(widget).apply_change(
            "canvas.pixel_precision.snap_step", 0.5
        )

        canvas.configure_pixel_precision.assert_called_once_with(
            pixel_precision
        )
        self.assertEqual(widget.zoom_widget.maximum(), 6400)
        widget.navigator_dialog.set_maximum_zoom.assert_called_once_with(6400)
        self.assertFalse(actions.show_pixel_grid.isChecked())
        self.assertFalse(actions.pixel_snap.isChecked())


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


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for runtime tests")
class TestEdgeRefinementSettingsRuntimeApplier(unittest.TestCase):

    def test_edge_refinement_switches_apply_without_restart(self):
        config = {
            "threshold": 140,
            "double_click_enabled": True,
            "auto_label_enabled": True,
        }
        canvas = SimpleNamespace(configure_edge_refinement=Mock())
        panel = SimpleNamespace(apply_settings=Mock())
        widget = SimpleNamespace(
            canvas=canvas,
            pixel_edge_widget=panel,
            _config={"canvas": {"edge_refinement": config}},
        )

        SettingsRuntimeApplier(widget).apply_change(
            "canvas.edge_refinement.threshold", 140
        )

        canvas.configure_edge_refinement.assert_called_once_with(config)
        panel.apply_settings.assert_called_once_with(config)


if __name__ == "__main__":
    unittest.main()
