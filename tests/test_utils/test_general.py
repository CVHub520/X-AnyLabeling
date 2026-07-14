import json
import unittest

import cv2
import numpy as np
import pytest

from anylabeling.views.labeling.classifier.utils import (
    export_image_to_category,
)
from anylabeling.views.labeling.utils.crop import (
    crop_and_save,
    process_single_image,
)
from anylabeling.views.labeling.utils.general import is_possible_rectangle
from anylabeling.views.labeling.utils.general import (
    resolve_export_directory,
)


class TestIsRectangle(unittest.TestCase):

    def test_normal_rectangle(self):
        points = [[0, 0], [1000, 0], [1000, 1], [0, 1]]
        self.assertEqual(is_possible_rectangle(points), True)

    def test_irregular_shape(self):
        points = [[0, 0], [2, 3], [4, 5], [6, 7]]
        self.assertEqual(is_possible_rectangle(points), False)

    def test_rectangle_with_square_shape(self):
        points = [[0, 0], [0, 1], [1, 1], [1, 0]]
        self.assertEqual(is_possible_rectangle(points), True)

    def test_rectangle_with_diagonal_points(self):
        points = [[1, 1], [1, 2], [2, 1], [2, 2]]
        self.assertEqual(is_possible_rectangle(points), True)

    def test_lese_than_four_points(self):
        points = [[0, 0], [1, 1], [1, 0]]
        self.assertEqual(is_possible_rectangle(points), False)

    def test_more_than_four_points(self):
        points = [[0, 0], [1, 1], [1, 0], [2, 0], [1, 2]]
        self.assertEqual(is_possible_rectangle(points), False)


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "../outside", "nested/path", r"nested\path"],
)
def test_resolve_export_directory_rejects_unsafe_names(tmp_path, name):
    with pytest.raises(ValueError):
        resolve_export_directory(tmp_path / "output", name)


def test_resolve_export_directory_preserves_safe_name(tmp_path):
    output_dir = tmp_path / "output"

    path = resolve_export_directory(output_dir, "cat 猫")

    assert path == output_dir / "cat 猫"


def test_crop_export_rejects_label_path_traversal(tmp_path):
    image_file = tmp_path / "image.jpg"
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image_file.write_bytes(cv2.imencode(".jpg", image)[1].tobytes())

    with pytest.raises(ValueError):
        crop_and_save(
            str(image_file),
            "../outside",
            np.array([[0, 0], [9, 0], [9, 9], [0, 9]]),
            str(tmp_path / "output"),
            {},
            "rectangle",
            0,
            0,
        )

    assert not (tmp_path / "outside").exists()


def test_batch_crop_rejects_label_path_traversal(tmp_path):
    image_file = tmp_path / "image.jpg"
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image_file.write_bytes(cv2.imencode(".jpg", image)[1].tobytes())
    label_file = tmp_path / "image.json"
    label_file.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "../outside",
                        "points": [[0, 0], [9, 0], [9, 9], [0, 9]],
                        "shape_type": "rectangle",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = process_single_image(
        (
            str(image_file),
            str(tmp_path),
            str(tmp_path / "output"),
            0,
            0,
            {"../outside": 1},
        )
    )

    assert not result
    assert not (tmp_path / "outside").exists()


def test_classification_export_rejects_category_path_traversal(tmp_path):
    image_file = tmp_path / "image.jpg"
    image_file.write_bytes(b"image")

    with pytest.raises(ValueError):
        export_image_to_category(
            str(image_file), "../outside", str(tmp_path / "output")
        )

    assert not (tmp_path / "outside").exists()
