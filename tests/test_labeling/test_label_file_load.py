import json
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from anylabeling.views.labeling import label_file


class TestLabelFileLoad(unittest.TestCase):
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
