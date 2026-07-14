import base64
import importlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from anylabeling.services.auto_labeling.utils.sahi.utils.cv import (
    read_image_as_pil,
)
from anylabeling.views.labeling import label_file


class TestPillowLimits(unittest.TestCase):

    def test_label_file_import_preserves_pillow_pixel_limit(self):
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 1):
            importlib.reload(label_file)

            self.assertEqual(Image.MAX_IMAGE_PIXELS, 1)

    def test_label_file_rejects_decompression_bomb(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (2, 2), "white").save(image_buffer, format="PNG")
        data = {
            "version": "test",
            "flags": {},
            "shapes": [],
            "imagePath": "image.png",
            "imageData": base64.b64encode(image_buffer.getvalue()).decode(),
            "imageHeight": 2,
            "imageWidth": 2,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            label_path = os.path.join(temp_dir, "image.json")
            with open(label_path, "w", encoding="utf-8") as label_stream:
                json.dump(data, label_stream)

            with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 1):
                with self.assertRaisesRegex(
                    label_file.LabelFileError, "decompression bomb"
                ):
                    label_file.LabelFile(label_path)

    def test_sahi_reader_preserves_pillow_pixel_limit(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 1):
            result = read_image_as_pil(image)

            self.assertEqual(result.size, (2, 2))
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 1)


if __name__ == "__main__":
    unittest.main()
