import base64
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from anylabeling.services.auto_labeling.remote_server import RemoteServer


class TestRemoteArtifacts(unittest.TestCase):
    def test_multiple_cutouts_preview_uses_maximum_alpha(self):
        first = np.array(
            [[[10, 20, 30, 200], [40, 50, 60, 0]]], dtype=np.uint8
        )
        second = np.array(
            [[[10, 20, 30, 100], [40, 50, 60, 170]]], dtype=np.uint8
        )
        artifacts = []
        for pixels in (first, second):
            _, encoded = cv2.imencode(".png", pixels)
            artifacts.append(
                {
                    "kind": "cutout",
                    "png_base64": base64.b64encode(encoded).decode(),
                }
            )
        with tempfile.TemporaryDirectory() as folder:
            output = RemoteServer._save_artifacts(
                artifacts, str(Path(folder) / "image.jpg")
            )
            preview = cv2.imread(
                str(output / "preview.png"), cv2.IMREAD_UNCHANGED
            )
            expected = first.copy()
            expected[0, 1] = second[0, 1]
            np.testing.assert_array_equal(preview, expected)

    def test_preview_is_scoped_to_image_model_and_task_and_cleared_on_empty_result(
        self,
    ):
        with patch(
            "anylabeling.services.auto_labeling.model.get_config",
            return_value={"remote_server_settings": {}},
        ):
            model = RemoteServer(
                {"type": "remote_server", "display_name": "Remote Server"},
                Mock(),
            )
        model.set_model_id("wevisone_vl")
        model.set_task("matting")
        _, encoded = cv2.imencode(
            ".png", np.full((2, 2, 4), 128, dtype=np.uint8)
        )
        response = Mock()
        response.json.return_value = {
            "data": {
                "shapes": [],
                "artifacts": [
                    {
                        "kind": "cutout",
                        "png_base64": base64.b64encode(encoded).decode(),
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "image.png"
            source.write_bytes(encoded.tobytes())
            with patch(
                "anylabeling.services.auto_labeling.remote_server.requests.post",
                return_value=response,
            ):
                model.predict_shapes(object(), str(source))
                self.assertTrue(
                    Path(model.get_preview_path(str(source))).is_file()
                )
                self.assertIsNone(
                    model.get_preview_path(str(source.with_name("other.png")))
                )
                model.set_task("detection")
                self.assertIsNone(model.get_preview_path(str(source)))
                model.set_task("matting")
                model.set_model_id("other_model")
                self.assertIsNone(model.get_preview_path(str(source)))
                model.set_model_id("wevisone_vl")
                response.json.return_value = {"data": {"shapes": []}}
                model.predict_shapes(object(), str(source))
                self.assertIsNone(model.get_preview_path(str(source)))

    def test_png_alpha_and_existing_exports_are_preserved(self):
        pixels = np.array(
            [[[10, 20, 30, 0], [30, 40, 50, 127]]], dtype=np.uint8
        )
        ok, encoded = cv2.imencode(".png", pixels)
        self.assertTrue(ok)
        artifact = {
            "kind": "cutout",
            "png_base64": base64.b64encode(encoded).decode(),
        }
        with tempfile.TemporaryDirectory() as folder:
            source = str(Path(folder) / "image.jpg")
            first = RemoteServer._save_artifacts([artifact], source)
            second = RemoteServer._save_artifacts([artifact], source)
            self.assertNotEqual(first, second)
            for output in (first, second):
                np.testing.assert_array_equal(
                    cv2.imread(
                        str(output / "000_cutout.png"), cv2.IMREAD_UNCHANGED
                    ),
                    pixels,
                )

    def test_invalid_artifacts_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = str(Path(folder) / "image.jpg")
            for artifact in (
                {"kind": "../outside", "png_base64": ""},
                {
                    "kind": "alpha",
                    "png_base64": base64.b64encode(b"invalid").decode(),
                },
            ):
                with self.assertRaises(ValueError):
                    RemoteServer._save_artifacts([artifact], source)
            self.assertEqual(list(Path(folder).iterdir()), [])
