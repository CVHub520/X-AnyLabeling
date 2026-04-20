"""Registry guard tests for `anylabeling.services.auto_labeling`.

The auto-labeling registry is a set of plain Python lists in
``anylabeling/services/auto_labeling/__init__.py`` that decide which UI
widgets a model exposes (marks, mask fineness, cropping mode, ...). The
file has historically picked up duplicate entries when several model
backends were added in parallel, which is silent at import time but
shows up later as duplicated controls in the UI. This module pins the
invariant in CI.
"""

import importlib.util
import unittest
from pathlib import Path


REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "anylabeling/services/auto_labeling/__init__.py"
)


def _load_registry_module():
    """Import the registry module without pulling Qt or model deps."""
    spec = importlib.util.spec_from_file_location(
        "auto_labeling_registry", REGISTRY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAutoLabelingRegistry(unittest.TestCase):
    """Sanity checks on the registry list constants."""

    @classmethod
    def setUpClass(cls):
        cls.module = _load_registry_module()

    def test_no_duplicates_in_any_registry_list(self):
        """Every ``_*_MODELS`` list must contain unique entries."""
        for name in dir(self.module):
            if not name.endswith("_MODELS"):
                continue
            value = getattr(self.module, name)
            if not isinstance(value, list):
                continue
            self.assertEqual(
                len(value),
                len(set(value)),
                msg=(
                    f"{name} contains duplicates: "
                    f"{[v for v in value if value.count(v) > 1]}"
                ),
            )

    def test_sam3cpp_video_registered_in_video_lists(self):
        """``sam3cpp_video`` is wired into the video-relevant registries."""
        expected_lists = [
            "_CUSTOM_MODELS",
            "_CACHED_AUTO_LABELING_MODELS",
            "_AUTO_LABELING_MARKS_MODELS",
            "_AUTO_LABELING_MASK_FINENESS_MODELS",
            "_AUTO_LABELING_CROPPING_MODE_MODELS",
            "_AUTO_LABELING_RESET_TRACKER_MODELS",
            "_AUTO_LABELING_PRESERVE_EXISTING_ANNOTATIONS_STATE_MODELS",
            "_AUTO_LABELING_PROMPT_MODELS",
            "_BATCH_PROCESSING_VIDEO_MODELS",
        ]
        for name in expected_lists:
            self.assertIn(
                "sam3cpp_video",
                getattr(self.module, name),
                msg=f"sam3cpp_video missing from {name}",
            )


if __name__ == "__main__":
    unittest.main()
