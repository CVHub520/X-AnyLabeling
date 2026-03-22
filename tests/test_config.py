import unittest
import importlib.util
import sys
import types
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "anylabeling/config.py"
INSERTED_MODULES = []


def _inject_module(name, module):
    if name not in sys.modules:
        sys.modules[name] = module
        INSERTED_MODULES.append(name)

fake_anylabeling = types.ModuleType("anylabeling")
fake_anylabeling_configs = types.ModuleType("anylabeling.configs")
fake_anylabeling.configs = fake_anylabeling_configs
_inject_module("anylabeling", fake_anylabeling)
_inject_module("anylabeling.configs", fake_anylabeling_configs)

fake_views = types.ModuleType("anylabeling.views")
fake_labeling = types.ModuleType("anylabeling.views.labeling")
fake_logger_module = types.ModuleType("anylabeling.views.labeling.logger")


class _DummyLogger:
    def warning(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None


fake_logger_module.logger = _DummyLogger()
_inject_module("anylabeling.views", fake_views)
_inject_module("anylabeling.views.labeling", fake_labeling)
_inject_module("anylabeling.views.labeling.logger", fake_logger_module)

SPEC = importlib.util.spec_from_file_location("config_module", MODULE_PATH)
CONFIG_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIG_MODULE)
normalize_user_config = CONFIG_MODULE.normalize_user_config

for module_name in INSERTED_MODULES:
    sys.modules.pop(module_name, None)


class TestConfigNormalization(unittest.TestCase):

    def test_normalize_legacy_keys_and_shortcuts(self):
        user_config = {
            "epsilon": 8.5,
            "show_cross_line": True,
            "ui": {"legacy": True},
            "shortcuts": {
                "open_next": ["D", "Ctrl+Shift+D"],
                "open_prev": ["A", "Ctrl+Shift+A"],
                "zoom_in": "Ctrl+=",
            },
        }

        normalized = normalize_user_config(user_config)

        self.assertNotIn("epsilon", normalized)
        self.assertNotIn("show_cross_line", normalized)
        self.assertNotIn("ui", normalized)
        self.assertEqual(normalized["canvas"]["epsilon"], 8.5)
        self.assertTrue(normalized["canvas"]["crosshair"]["show"])
        self.assertEqual(normalized["shortcuts"]["open_next"], "D")
        self.assertEqual(normalized["shortcuts"]["open_prev"], "A")
        self.assertEqual(normalized["shortcuts"]["zoom_in"], ["Ctrl+="])

    def test_legacy_key_does_not_override_new_path(self):
        user_config = {
            "show_cross_line": True,
            "canvas": {"crosshair": {"show": False}},
        }
        normalized = normalize_user_config(user_config)
        self.assertFalse(normalized["canvas"]["crosshair"]["show"])


if __name__ == "__main__":
    unittest.main()
