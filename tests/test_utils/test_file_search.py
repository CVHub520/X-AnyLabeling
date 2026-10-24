import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "anylabeling/views/labeling/utils/file_search.py"
)
SPEC = importlib.util.spec_from_file_location(
    "file_search_module", MODULE_PATH
)
FILE_SEARCH_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FILE_SEARCH_MODULE)

filter_image_files = FILE_SEARCH_MODULE.filter_image_files
parse_search_pattern = FILE_SEARCH_MODULE.parse_search_pattern


class TestFileSearch(unittest.TestCase):

    @staticmethod
    def _write_label(path, shapes):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"shapes": shapes}, f)

    @staticmethod
    def _write_label_data(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def test_text_search(self):
        image_files = ["a_test.jpg", "b.jpg", "c_test.png"]
        pattern = parse_search_pattern("test")
        result = filter_image_files(image_files, pattern)

        self.assertEqual(result, ["a_test.jpg", "c_test.png"])

    def test_regex_search(self):
        image_files = ["a_test.jpg", "b.JPG", "c.png"]
        pattern = parse_search_pattern("<\\.jpg$>")
        result = filter_image_files(image_files, pattern)

        self.assertEqual(result, ["a_test.jpg", "b.JPG"])

    def test_index_search(self):
        image_files = ["a.jpg", "b.jpg", "c.jpg"]
        pattern = parse_search_pattern("#2")
        result = filter_image_files(image_files, pattern)

        self.assertEqual(result, ["b.jpg"])

    def test_index_search_out_of_range(self):
        image_files = ["a.jpg", "b.jpg", "c.jpg"]
        pattern = parse_search_pattern("#10")
        result = filter_image_files(image_files, pattern)

        self.assertEqual(result, [])

    def test_invalid_index_falls_back_to_text_search(self):
        pattern = parse_search_pattern("#0")

        self.assertEqual(pattern.mode, "normal")
        self.assertEqual(pattern.pattern, "#0")

    def test_attribute_search_modes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_files = [
                str(Path(temp_dir) / "a.jpg"),
                str(Path(temp_dir) / "b.jpg"),
                str(Path(temp_dir) / "c.jpg"),
            ]
            self._write_label(
                Path(temp_dir) / "a.json",
                [
                    {
                        "label": "person",
                        "shape_type": "rectangle",
                        "score": 0.4,
                        "description": "front view",
                        "difficult": True,
                        "group_id": 0,
                    }
                ],
            )
            self._write_label(
                Path(temp_dir) / "b.json",
                [
                    {
                        "label": "car",
                        "shape_type": "polygon",
                        "score": 0.8,
                        "description": "",
                        "difficult": False,
                        "group_id": 1,
                    }
                ],
            )
            self._write_label(Path(temp_dir) / "c.json", [])

            test_cases = [
                ("difficult::1", [image_files[0]]),
                ("gid::1", [image_files[1]]),
                ("shape::1", [image_files[0], image_files[1]]),
                ("label::person", [image_files[0]]),
                ("type::polygon", [image_files[1]]),
                ("score::[0,0.5]", [image_files[0]]),
                ("description::yes", [image_files[0]]),
            ]
            for query, expected in test_cases:
                with self.subTest(query=query):
                    pattern = parse_search_pattern(query)
                    result = filter_image_files(image_files, pattern)
                    self.assertEqual(result, expected)

    def test_checked_status_search_modes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_files = [
                str(Path(temp_dir) / "checked.jpg"),
                str(Path(temp_dir) / "unchecked.jpg"),
                str(Path(temp_dir) / "missing.jpg"),
                str(Path(temp_dir) / "failed.jpg"),
            ]
            self._write_label_data(
                Path(temp_dir) / "checked.json",
                {"checked": True, "shapes": []},
            )
            self._write_label_data(
                Path(temp_dir) / "unchecked.json",
                {"checked": False, "shapes": []},
            )

            self._write_label_data(
                Path(temp_dir) / "failed.json",
                {"checked": "failed", "shapes": []},
            )

            checked_pattern = parse_search_pattern("checked::1")
            unchecked_pattern = parse_search_pattern("checked::0")

            self.assertEqual(
                filter_image_files(image_files, checked_pattern),
                [image_files[0], image_files[3]],
            )
            self.assertEqual(
                filter_image_files(image_files, unchecked_pattern),
                [image_files[1], image_files[2]],
            )

    def test_three_state_check_search_with_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "labels"
            output_dir.mkdir()
            image_files = [
                str(Path(temp_dir) / f"{name}.jpg")
                for name in (
                    "passed",
                    "failed",
                    "unchecked",
                    "legacy",
                    "missing",
                )
            ]
            for name, data in (
                ("passed", {"checked": True}),
                ("failed", {"checked": "failed"}),
                ("unchecked", {"checked": False}),
                ("legacy", {}),
            ):
                self._write_label_data(output_dir / f"{name}.json", data)
            for value, indices in (
                ("passed", [0]),
                ("failed", [1]),
                ("unchecked", [2, 3, 4]),
                ("PASSED", [0]),
                ("FAILED", [1]),
                ("UNCHECKED", [2, 3, 4]),
                ("1", [0, 1]),
                ("true", [0, 1]),
                ("yes", [0, 1]),
                ("0", [2, 3, 4]),
                ("false", [2, 3, 4]),
            ):
                with self.subTest(value=value):
                    pattern = parse_search_pattern(f"checked::{value}")
                    self.assertEqual(
                        filter_image_files(
                            image_files, pattern, str(output_dir)
                        ),
                        [image_files[index] for index in indices],
                    )
