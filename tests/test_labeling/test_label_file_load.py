import json
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from anylabeling.views.labeling import label_file


class TestLabelFileLoad(unittest.TestCase):
    def test_image_path_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            image_dir = os.path.join(directory, "images", "nested")
            label_dir = os.path.join(directory, "labels")
            os.makedirs(image_dir)
            os.makedirs(label_dir)
            image_path = os.path.join(image_dir, "image.png")
            Image.new("RGB", (2, 3), "white").save(image_path)
            label_path = os.path.join(label_dir, "image.json")
            relative_path = os.path.relpath(image_path, label_dir)
            local_image_path = os.path.join(label_dir, "image.png")
            Image.new("RGB", (4, 5), "black").save(local_image_path)
            for stored_path, kwargs in (
                (relative_path, {}),
                (image_path, {}),
                ("../old/renamed.png", {"image_path": image_path}),
                ("../old/image.png", {"image_dir": image_dir}),
            ):
                with self.subTest(stored_path=stored_path, kwargs=kwargs):
                    with open(label_path, "w", encoding="utf-8") as stream:
                        json.dump(
                            {
                                "version": "test",
                                "shapes": [],
                                "imagePath": stored_path,
                                "imageData": None,
                            },
                            stream,
                        )
                    loaded = label_file.LabelFile(label_path, **kwargs)
                    self.assertEqual(loaded.image_path, stored_path)
                    self.assertEqual(
                        os.path.abspath(loaded.resolved_image_path), image_path
                    )
                    self.assertEqual(
                        label_file.utils.get_pil_img_dim(loaded.image_data),
                        (2, 3),
                    )

            loaded = label_file.LabelFile(label_path)
            self.assertEqual(loaded.resolved_image_path, local_image_path)

    def test_missing_image_reports_path(self):
        with tempfile.TemporaryDirectory() as directory:
            label_path = os.path.join(directory, "image.json")
            with open(label_path, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "version": "test",
                        "shapes": [],
                        "imagePath": "missing.png",
                        "imageData": None,
                    },
                    stream,
                )
            with self.assertRaises(label_file.LabelFileError) as error:
                label_file.LabelFile(label_path)
            self.assertIn(
                os.path.join(directory, "missing.png"), str(error.exception)
            )

    def test_external_image_uses_header_without_pixel_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "image.png")
            label_path = os.path.join(directory, "image.json")
            Image.new("RGB", (2, 3), "white").save(image_path)
            data = {
                "version": "test",
                "flags": {},
                "shapes": [],
                "imagePath": "image.png",
                "imageData": None,
                "imageHeight": 3,
                "imageWidth": 2,
            }
            with open(label_path, "w", encoding="utf-8") as label_stream:
                json.dump(data, label_stream)

            with (
                mock.patch.object(
                    label_file.base64,
                    "b64encode",
                    side_effect=AssertionError("unexpected base64 encoding"),
                ),
                mock.patch.object(
                    label_file.utils,
                    "img_data_to_arr",
                    side_effect=AssertionError("unexpected pixel conversion"),
                ),
            ):
                loaded_label = label_file.LabelFile(label_path)

            with open(image_path, "rb") as image_stream:
                self.assertEqual(loaded_label.image_data, image_stream.read())


if __name__ == "__main__":
    unittest.main()
