import unittest

try:
    from anylabeling.views.labeling.settings.schema import (
        EXCLUDED_KEYS,
        SETTING_FIELDS,
        SETTINGS_KEYS,
        SETTINGS_GENERAL_KEYS,
        SETTINGS_SHAPE_KEYS,
        SETTINGS_PRIMARY_ORDER,
        SETTINGS_SHORTCUT_KEYS_CORE,
        defaults_map,
        fields_for_primary,
    )

    SCHEMA_AVAILABLE = True
except Exception:
    SCHEMA_AVAILABLE = False


@unittest.skipUnless(SCHEMA_AVAILABLE, "Settings schema dependencies are unavailable")
class TestSettingsSchema(unittest.TestCase):

    def test_field_count(self):
        self.assertEqual(len(SETTING_FIELDS), 120)

    def test_shortcut_and_non_shortcut_count(self):
        shortcut_fields = [
            field for field in SETTING_FIELDS if field.primary == "Shortcuts"
        ]
        self.assertEqual(len(shortcut_fields), 73)
        self.assertEqual(len(SETTING_FIELDS) - len(shortcut_fields), 47)

    def test_defaults_cover_all_keys(self):
        defaults = defaults_map()
        self.assertEqual(set(defaults.keys()), set(SETTINGS_KEYS))

    def test_included_and_excluded_keys(self):
        expected_keys = {
            "display_label_popup",
            "auto_switch_to_edit_mode",
            "system_clipboard",
            "shape.line_color",
            "canvas.mask.opacity",
            "canvas.crosshair.show",
            "canvas.crosshair.width",
            "canvas.crosshair.color",
            "canvas.crosshair.opacity",
            "canvas.brush.point_distance",
            "model_hub",
            "logger_level",
            "shortcuts.open",
            "shortcuts.zoom_in",
            "shortcuts.add_point_to_edge",
            "shortcuts.quit",
            "shortcuts.open_settings",
            "shortcuts.auto_labeling_add_point",
            "shortcuts.auto_labeling_finish_object",
        }
        for key in expected_keys:
            self.assertIn(key, SETTINGS_KEYS)

        for key in EXCLUDED_KEYS:
            self.assertNotIn(key, SETTINGS_KEYS)

    def test_primary_and_key_sets(self):
        self.assertEqual(
            SETTINGS_PRIMARY_ORDER,
            ("Shortcuts", "General", "Shape", "Canvas"),
        )
        self.assertEqual(len(SETTINGS_GENERAL_KEYS), 8)
        self.assertEqual(len(SETTINGS_SHAPE_KEYS), 9)
        self.assertEqual(len(SETTINGS_SHORTCUT_KEYS_CORE), 24)
        for key in SETTINGS_GENERAL_KEYS:
            self.assertIn(key, SETTINGS_KEYS)
        for key in SETTINGS_SHAPE_KEYS:
            self.assertIn(key, SETTINGS_KEYS)
        for key in SETTINGS_SHORTCUT_KEYS_CORE:
            self.assertIn(key, SETTINGS_KEYS)

    def test_fields_for_primary(self):
        general_fields = fields_for_primary("General")
        shape_fields = fields_for_primary("Shape")
        shortcut_fields = fields_for_primary("Shortcuts")
        canvas_fields = fields_for_primary("Canvas")
        self.assertEqual(
            [field.key for field in general_fields], list(SETTINGS_GENERAL_KEYS)
        )
        self.assertEqual(
            [field.key for field in shape_fields], list(SETTINGS_SHAPE_KEYS)
        )
        shape_keys = {field.key for field in shape_fields}
        self.assertIn("shape.line_color", shape_keys)
        self.assertIn("shape.point_size", shape_keys)
        self.assertIn("shape.line_width", shape_keys)
        self.assertEqual(
            len(shortcut_fields),
            73,
        )
        for key in SETTINGS_SHORTCUT_KEYS_CORE:
            self.assertIn(key, [field.key for field in shortcut_fields])
        self.assertEqual(len(canvas_fields), 20)
        canvas_keys = {field.key for field in canvas_fields}
        self.assertIn("canvas.crosshair.show", canvas_keys)
        self.assertIn("canvas.crosshair.width", canvas_keys)
        self.assertIn("canvas.crosshair.color", canvas_keys)
        self.assertIn("canvas.crosshair.opacity", canvas_keys)
        self.assertIn("canvas.brush.point_distance", canvas_keys)

    def test_visible_non_shortcut_fields_have_descriptions(self):
        fields = (
            fields_for_primary("General")
            + fields_for_primary("Shape")
            + fields_for_primary("Canvas")
        )
        self.assertTrue(all(field.description for field in fields))
