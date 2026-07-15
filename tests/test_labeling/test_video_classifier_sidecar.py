import json
import os
import tempfile
import unittest

from anylabeling.views.labeling.video_classifier.config import (
    SCHEMA_VERSION,
    SIDECAR_TYPE,
)
from anylabeling.views.labeling.video_classifier.sidecar import (
    IncompatibleSidecarError,
    InvalidSidecarError,
    backup_sidecar,
    load_sidecar,
    sidecar_path_for,
)


class TestVideoClassifierSidecar(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.video_path = os.path.join(self.directory.name, "video.mp4")
        self.sidecar_path = sidecar_path_for(self.video_path)

    def _write_text(self, text):
        with open(self.sidecar_path, "w", encoding="utf-8") as file:
            file.write(text)

    def _write_json(self, payload):
        self._write_text(json.dumps(payload))

    def test_missing_sidecar_returns_none(self):
        self.assertIsNone(load_sidecar(self.video_path))

    def test_invalid_json_reports_location(self):
        self._write_text('{"segments": [}')

        with self.assertRaisesRegex(InvalidSidecarError, "line 1, column 15"):
            load_sidecar(self.video_path)

    def test_non_object_root_is_invalid(self):
        self._write_json([])

        with self.assertRaisesRegex(
            InvalidSidecarError, "must be a JSON object"
        ):
            load_sidecar(self.video_path)

    def test_invalid_schema_is_reported(self):
        self._write_json(
            {
                "version": SCHEMA_VERSION,
                "type": SIDECAR_TYPE,
                "segments": {},
            }
        )

        with self.assertRaisesRegex(
            InvalidSidecarError, "'segments' must be a JSON array"
        ):
            load_sidecar(self.video_path)

    def test_unsupported_version_is_incompatible(self):
        self._write_json(
            {
                "version": "2.0.0",
                "type": SIDECAR_TYPE,
                "segments": [],
            }
        )

        with self.assertRaisesRegex(
            IncompatibleSidecarError, "Unsupported sidecar version"
        ):
            load_sidecar(self.video_path)

    def test_backup_preserves_invalid_sidecar(self):
        content = "invalid sidecar"
        self._write_text(content)

        backup_path = backup_sidecar(self.video_path)

        self.assertFalse(os.path.exists(self.sidecar_path))
        self.assertTrue(os.path.exists(backup_path))
        with open(backup_path, "r", encoding="utf-8") as file:
            self.assertEqual(file.read(), content)


if __name__ == "__main__":
    unittest.main()
