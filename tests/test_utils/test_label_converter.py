import json
import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import yaml
from PIL import Image

from anylabeling.views.labeling.label_converter import LabelConverter


class TestLabelConverterPoseConfig(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _write_pose_cfg(self, data):
        cfg_path = os.path.join(self.temp_dir, "pose.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return cfg_path

    def test_missing_has_visible_defaults_to_true(self):
        cfg_path = self._write_pose_cfg(
            {"classes": {"person": ["nose", "left_eye"]}}
        )

        converter = LabelConverter(pose_cfg_file=cfg_path)

        self.assertTrue(converter.has_visible)
        self.assertEqual(converter.classes, ["person"])

    def test_explicit_has_visible_false_is_respected(self):
        cfg_path = self._write_pose_cfg(
            {
                "has_visible": False,
                "classes": {"person": ["nose", "left_eye"]},
            }
        )

        converter = LabelConverter(pose_cfg_file=cfg_path)

        self.assertFalse(converter.has_visible)

    def test_missing_classes_raises_value_error(self):
        cfg_path = self._write_pose_cfg({"has_visible": True})

        with self.assertRaises(ValueError):
            LabelConverter(pose_cfg_file=cfg_path)


class TestLabelConverterObbBounds(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.classes_file = os.path.join(self.temp_dir, "classes.txt")
        with open(self.classes_file, "w", encoding="utf-8") as f:
            f.write("plane\n")
        self.converter = LabelConverter(classes_file=self.classes_file)

    def tearDown(self):
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _write_label_file(self, points):
        label_file = os.path.join(self.temp_dir, "label.json")
        data = {
            "imagePath": "image.jpg",
            "imageWidth": 100,
            "imageHeight": 50,
            "shapes": [
                {
                    "label": "plane",
                    "shape_type": "rotation",
                    "points": points,
                }
            ],
        }
        with open(label_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return label_file

    def test_dota_skips_rotation_shape_with_any_out_of_bounds_point(self):
        label_file = self._write_label_file(
            [[-1, 10], [20, 10], [20, 20], [10, 20]]
        )
        output_file = os.path.join(self.temp_dir, "label.txt")

        self.converter.custom_to_dota(label_file, output_file)

        with open(output_file, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "")

    def test_yolo_obb_skips_rotation_shape_with_any_out_of_bounds_point(self):
        label_file = self._write_label_file(
            [[-1, 10], [20, 10], [20, 20], [10, 20]]
        )
        output_file = os.path.join(self.temp_dir, "label.txt")

        self.converter.custom_to_yolo(label_file, output_file, "obb")

        with open(output_file, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "")


class TestLabelConverterMaskExport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.converter = LabelConverter()
        self.mapping_table = {"type": "grayscale", "colors": {"cat": 1}}

    def _write_label_file(self, shapes):
        label_file = os.path.join(self.temp_dir.name, "label.json")
        data = {
            "imagePath": "image.jpg",
            "imageWidth": 4,
            "imageHeight": 3,
            "shapes": shapes,
        }
        with open(label_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return label_file

    def test_custom_to_mask_writes_blank_mask_for_empty_labels(self):
        label_file = self._write_label_file([])
        output_file = os.path.join(self.temp_dir.name, "mask.png")

        self.converter.custom_to_mask(
            label_file, output_file, self.mapping_table
        )

        mask = cv2.imread(output_file, cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (3, 4))
        self.assertTrue(np.all(mask == 0))

    def test_custom_image_to_empty_mask_uses_source_image_size(self):
        image_file = os.path.join(self.temp_dir.name, "image.png")
        output_file = os.path.join(self.temp_dir.name, "mask.png")
        Image.new("RGB", (5, 2), color=(255, 255, 255)).save(image_file)

        self.converter.custom_image_to_empty_mask(
            image_file, output_file, self.mapping_table
        )

        mask = cv2.imread(output_file, cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (2, 5))
        self.assertTrue(np.all(mask == 0))

    def test_custom_image_to_empty_mask_supports_rgb_output(self):
        image_file = os.path.join(self.temp_dir.name, "image.png")
        output_file = os.path.join(self.temp_dir.name, "mask.png")
        mapping_table = {"type": "rgb", "colors": {"cat": [1, 2, 3]}}
        Image.new("RGB", (5, 2), color=(255, 255, 255)).save(image_file)

        self.converter.custom_image_to_empty_mask(
            image_file, output_file, mapping_table
        )

        mask = cv2.imread(output_file, cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (2, 5, 3))
        self.assertTrue(np.all(mask == 0))


class TestLabelConverterVocValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.converter = LabelConverter()
        self.input_file = os.path.join(self.temp_dir.name, "input.xml")
        self.output_file = os.path.join(self.temp_dir.name, "output.json")

    def _convert(self, objects, mode="rectangle"):
        xml = (
            "<annotation><filename>image.jpg</filename>"
            "<size><width>100</width><height>50</height></size>"
            f"{objects}</annotation>"
        )
        with open(self.input_file, "w", encoding="utf-8") as f:
            f.write(xml)
        self.converter.voc_to_custom(
            self.input_file, self.output_file, "image.jpg", mode
        )
        with open(self.output_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_missing_geometry_is_skipped(self):
        objects = (
            "<object><name>missing</name></object>"
            "<object><name>valid</name><bndbox>"
            "<xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax>"
            "</bndbox></object>"
        )

        with mock.patch(
            "anylabeling.views.labeling.label_converter.logger.warning"
        ) as warning:
            data = self._convert(objects)

        self.assertEqual(
            [shape["label"] for shape in data["shapes"]], ["valid"]
        )
        warning.assert_called_once()
        self.assertIn("VOC object 1", warning.call_args.args[0])
        self.assertIn(self.input_file, warning.call_args.args[0])

    def test_incomplete_geometry_is_skipped(self):
        objects = (
            "<object><name>incomplete</name><bndbox>"
            "<xmin>1</xmin><ymin>2</ymin><xmax>3</xmax>"
            "</bndbox></object>"
        )

        with mock.patch(
            "anylabeling.views.labeling.label_converter.logger.warning"
        ) as warning:
            data = self._convert(objects)

        self.assertEqual(data["shapes"], [])
        warning.assert_called_once()
        self.assertIn("bndbox/ymax", warning.call_args.args[0])
