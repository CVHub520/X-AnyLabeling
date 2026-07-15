import base64
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from anylabeling.views.labeling import label_file


class TestLabelFileSave(unittest.TestCase):
    def test_image_dimensions_are_read_without_pixel_conversion(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (2, 3), "white").save(image_buffer, format="PNG")

        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "annotation.json")
            label = label_file.LabelFile()
            with mock.patch.object(
                label_file.utils,
                "img_data_to_arr",
                side_effect=AssertionError("unexpected pixel conversion"),
            ):
                label.save(
                    filename=filename,
                    shapes=[],
                    image_path="image.png",
                    image_height=1,
                    image_width=1,
                    image_data=image_buffer.getvalue(),
                )

            with open(filename, "r", encoding="utf-8") as label_stream:
                data = json.load(label_stream)
            self.assertEqual(data["imageHeight"], 3)
            self.assertEqual(data["imageWidth"], 2)
            self.assertEqual(
                base64.b64decode(data["imageData"]), image_buffer.getvalue()
            )

    def test_failed_save_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "annotation.json")
            original_data = '{"original": true}\n'
            with open(filename, "w", encoding="utf-8") as f:
                f.write(original_data)

            def fail_after_partial_write(data, f, **kwargs):
                f.write('{"partial":')
                raise OSError("simulated write failure")

            label = label_file.LabelFile()
            with mock.patch.object(
                label_file.json,
                "dump",
                side_effect=fail_after_partial_write,
            ):
                with self.assertRaises(label_file.LabelFileError):
                    label.save(
                        filename=filename,
                        shapes=[],
                        image_path="image.jpg",
                        image_height=1,
                        image_width=1,
                    )

            with open(filename, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), original_data)
            self.assertEqual(os.listdir(directory), ["annotation.json"])
            self.assertIsNone(label.filename)


if __name__ == "__main__":
    unittest.main()
