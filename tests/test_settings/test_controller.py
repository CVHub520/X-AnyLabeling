import copy
import unittest

try:
    from PyQt6 import QtCore

    from anylabeling.views.labeling.settings.controller import (
        SettingsController,
        SettingsValidationError,
    )
    from anylabeling.views.labeling.settings.schema import load_template_config

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for settings controller tests")
class TestSettingsController(unittest.TestCase):

    def setUp(self):
        self.app = QtCore.QCoreApplication.instance()
        if self.app is None:
            self.app = QtCore.QCoreApplication([])
        self.config = copy.deepcopy(load_template_config())
        self.applied = []
        self.saved = []

        def apply_callback(key, value):
            self.applied.append((key, value))

        def save_callback(config):
            self.saved.append(copy.deepcopy(config))
            return True

        self.controller = SettingsController(
            config=self.config,
            apply_callback=apply_callback,
            save_callback=save_callback,
            save_delay_ms=1000,
        )

    def tearDown(self):
        self.controller.flush()

    def test_numeric_validation_and_apply(self):
        old_value = self.controller.get_value("canvas.epsilon")
        with self.assertRaises(SettingsValidationError):
            self.controller.update_field(
                "canvas.epsilon", 0.05, schedule_save=False
            )
        self.assertEqual(self.controller.get_value("canvas.epsilon"), old_value)

        changed = self.controller.update_field(
            "canvas.epsilon", 12.5, schedule_save=False
        )
        self.assertTrue(changed)
        self.assertEqual(self.controller.get_value("canvas.epsilon"), 12.5)
        self.assertIn(("canvas.epsilon", 12.5), self.applied)

    def test_optional_integer_accepts_none_and_non_negative_values(self):
        self.assertIsNone(
            self.controller.get_value("qt_image_allocation_limit")
        )

        changed = self.controller.update_field(
            "qt_image_allocation_limit", 1024, schedule_save=False
        )
        self.assertTrue(changed)
        self.assertEqual(
            self.controller.get_value("qt_image_allocation_limit"), 1024
        )

        changed = self.controller.update_field(
            "qt_image_allocation_limit", None, schedule_save=False
        )
        self.assertTrue(changed)
        self.assertIsNone(
            self.controller.get_value("qt_image_allocation_limit")
        )

        with self.assertRaises(SettingsValidationError):
            self.controller.update_field(
                "qt_image_allocation_limit", -1, schedule_save=False
            )

    def test_canvas_field_validation(self):
        self.controller.update_field(
            "canvas.crosshair.width",
            1.0,
            schedule_save=False,
        )
        self.controller.update_field(
            "canvas.crosshair.width",
            10.0,
            schedule_save=False,
        )
        with self.assertRaises(SettingsValidationError):
            self.controller.update_field(
                "canvas.crosshair.width",
                0.5,
                schedule_save=False,
            )

        self.controller.update_field(
            "canvas.crosshair.opacity",
            0.0,
            schedule_save=False,
        )
        self.controller.update_field(
            "canvas.crosshair.opacity",
            1.0,
            schedule_save=False,
        )
        with self.assertRaises(SettingsValidationError):
            self.controller.update_field(
                "canvas.crosshair.opacity",
                1.1,
                schedule_save=False,
            )

        self.controller.update_field(
            "canvas.brush.point_distance",
            1.0,
            schedule_save=False,
        )
        self.controller.update_field(
            "canvas.brush.point_distance",
            200.0,
            schedule_save=False,
        )
        with self.assertRaises(SettingsValidationError):
            self.controller.update_field(
                "canvas.brush.point_distance",
                250.0,
                schedule_save=False,
            )

        self.controller.update_field(
            "canvas.crosshair.color",
            "#1a2b3c",
            schedule_save=False,
        )
        self.assertEqual(
            self.controller.get_value("canvas.crosshair.color"),
            "#1A2B3C",
        )
        with self.assertRaises(SettingsValidationError):
            self.controller.update_field(
                "canvas.crosshair.color",
                "bad-color",
                schedule_save=False,
            )

        self.controller.update_field(
            "canvas.attributes.background_color",
            [1, 2, 3, 4],
            schedule_save=False,
        )
        self.assertEqual(
            self.controller.get_value("canvas.attributes.background_color"),
            [1, 2, 3, 4],
        )
        self.controller.update_field(
            "canvas.cuboid.default_depth_vector",
            [10, -5.5],
            schedule_save=False,
        )
        self.assertEqual(
            self.controller.get_value("canvas.cuboid.default_depth_vector"),
            [10.0, -5.5],
        )

    def test_shape_field_validation(self):
        self.controller.update_field(
            "shape_color",
            None,
            schedule_save=False,
        )
        self.assertIsNone(self.controller.get_value("shape_color"))
        self.controller.update_field(
            "shape_color",
            "auto",
            schedule_save=False,
        )
        self.controller.update_field(
            "shape_color",
            "manual",
            schedule_save=False,
        )
        with self.assertRaises(SettingsValidationError):
            self.controller.update_field(
                "shape_color",
                "bad-mode",
                schedule_save=False,
            )

        self.controller.update_field(
            "shape.line_color",
            [0, 255, 0, 128],
            schedule_save=False,
        )
        self.assertEqual(
            self.controller.get_value("shape.line_color"),
            [0, 255, 0, 128],
        )
        with self.assertRaises(SettingsValidationError):
            self.controller.update_field(
                "shape.line_color",
                [0, 255, 0],
                schedule_save=False,
            )
        with self.assertRaises(SettingsValidationError):
            self.controller.update_field(
                "shape.line_color",
                [0, 255, 0, 300],
                schedule_save=False,
            )

        self.controller.update_field(
            "default_shape_color",
            [10, 20, 30],
            schedule_save=False,
        )
        self.assertEqual(
            self.controller.get_value("default_shape_color"),
            [10, 20, 30],
        )
        with self.assertRaises(SettingsValidationError):
            self.controller.update_field(
                "default_shape_color",
                [10, 20, 30, 40],
                schedule_save=False,
            )

    def test_shortcut_conflict_and_whitelist(self):
        with self.assertRaises(SettingsValidationError) as ctx:
            self.controller.update_field(
                "shortcuts.open",
                "Ctrl+S",
                schedule_save=False,
            )
        self.assertIn("'Ctrl+S'", str(ctx.exception))
        self.assertIn("conflicts with", str(ctx.exception))
        self.assertIn("(File)", str(ctx.exception))
        self.assertIn("shortcuts.open", ctx.exception.conflict_keys)
        self.assertIn("shortcuts.save", ctx.exception.conflict_keys)

        self.controller.update_field(
            "shortcuts.undo_last_point",
            "Alt+Shift+Z",
            schedule_save=False,
        )
        changed = self.controller.update_field(
            "shortcuts.undo_last_point",
            "Ctrl+Z",
            schedule_save=False,
        )
        self.assertTrue(changed)
        self.assertEqual(
            self.controller.get_value("shortcuts.undo_last_point"),
            "Ctrl+Z",
        )

    def test_shortcut_clear_behavior(self):
        changed = self.controller.update_field(
            "shortcuts.open",
            "",
            schedule_save=False,
        )
        self.assertTrue(changed)
        self.assertIsNone(self.controller.get_value("shortcuts.open"))

        self.controller.update_field(
            "shortcuts.create_circle",
            "Ctrl+Shift+9",
            schedule_save=False,
        )
        changed = self.controller.update_field(
            "shortcuts.create_circle",
            None,
            schedule_save=False,
        )
        self.assertTrue(changed)
        self.assertIsNone(self.controller.get_value("shortcuts.create_circle"))

    def test_shortcut_list_value_uses_first_entry(self):
        self.controller.update_field(
            "shortcuts.open_next",
            "Ctrl+Alt+D",
            schedule_save=False,
        )
        changed = self.controller.update_field(
            "shortcuts.open_next",
            ["D", "Ctrl+Shift+D"],
            schedule_save=False,
        )
        self.assertTrue(changed)
        self.assertEqual(self.controller.get_value("shortcuts.open_next"), "D")

    def test_multi_shortcut_conflict(self):
        with self.assertRaises(SettingsValidationError) as ctx:
            self.controller.update_field(
                "shortcuts.zoom_in",
                ["Ctrl+S"],
                schedule_save=False,
            )
        self.assertIn("shortcuts.zoom_in", ctx.exception.conflict_keys)
        self.assertIn("shortcuts.save", ctx.exception.conflict_keys)

    def test_reset_page_and_reset_all(self):
        default_model_hub = load_template_config()["model_hub"]
        default_epsilon = load_template_config()["canvas"]["epsilon"]

        self.controller.update_field("model_hub", "modelscope", schedule_save=False)
        self.controller.update_field("canvas.epsilon", 9.9, schedule_save=False)

        self.controller.reset_page("Canvas", "Interaction")
        self.assertEqual(
            self.controller.get_value("canvas.epsilon"), default_epsilon
        )

        self.controller.reset_all()
        self.assertEqual(self.controller.get_value("model_hub"), default_model_hub)

    def test_debounce_save_and_flush(self):
        self.controller.update_field("model_hub", "modelscope")
        self.assertTrue(self.controller.has_pending_save())
        self.assertEqual(len(self.saved), 0)

        self.controller.flush()
        self.assertFalse(self.controller.has_pending_save())
        self.assertEqual(len(self.saved), 1)

    def test_deferred_runtime_apply_and_discard(self):
        config = copy.deepcopy(load_template_config())
        applied = []
        saved = []

        controller = SettingsController(
            config=config,
            apply_callback=lambda key, value: applied.append((key, value)),
            save_callback=lambda payload: saved.append(copy.deepcopy(payload))
            or True,
            save_delay_ms=1000,
            defer_runtime_apply=True,
        )

        runtime_before = config["canvas"]["epsilon"]
        changed = controller.update_field(
            "canvas.epsilon", 12.5, schedule_save=False
        )
        self.assertTrue(changed)
        self.assertEqual(config["canvas"]["epsilon"], runtime_before)
        self.assertEqual(controller.get_value("canvas.epsilon"), 12.5)
        self.assertEqual(applied, [])
        self.assertTrue(controller.has_unsaved_changes())

        controller.discard_changes()
        self.assertFalse(controller.has_unsaved_changes())
        self.assertEqual(controller.get_value("canvas.epsilon"), runtime_before)
        self.assertEqual(config["canvas"]["epsilon"], runtime_before)

        controller.update_field("canvas.epsilon", 13.5, schedule_save=False)
        controller.save_now()
        self.assertEqual(config["canvas"]["epsilon"], 13.5)
        self.assertEqual(controller.get_value("canvas.epsilon"), 13.5)
        self.assertEqual(controller.last_saved_keys, ["canvas.epsilon"])
        self.assertIn(("canvas.epsilon", 13.5), applied)
        self.assertEqual(len(saved), 1)


if __name__ == "__main__":
    unittest.main()
