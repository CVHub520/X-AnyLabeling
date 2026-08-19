import unittest
from unittest import mock

import numpy as np

from anylabeling.services.auto_labeling.__base__.ram import (
    RecognizeAnything,
    normalize_tag_mode,
)


class RamTagsTest(unittest.TestCase):
    def setUp(self):
        self.model = RecognizeAnything.__new__(RecognizeAnything)
        self.model.delete_tag_index = []
        self.model.tag_list = np.array(["person", "car", "street"])
        self.model.tag_list_chinese = np.array(["人", "汽车", "街道"])

    def test_postprocess_returns_ordered_tag_lists(self):
        outputs = (np.array([[1, 0, 1]]), np.array([1]))

        english, chinese = self.model.postprocess(outputs)

        self.assertEqual(english, [["person", "street"]])
        self.assertEqual(chinese, [["人", "街道"]])

    def test_get_results_defaults_to_english(self):
        tags = ([["person", "street"]], [["人", "街道"]])
        self.model.tag_mode = "en"
        self.assertEqual(self.model.get_results(tags), tags[0][0])

    def test_get_results_supports_chinese(self):
        tags = ([["person", "street"]], [["人", "街道"]])
        self.model.tag_mode = "zh"
        self.assertEqual(self.model.get_results(tags), tags[1][0])

    def test_postprocess_keeps_empty_result(self):
        outputs = (np.array([[0, 0, 0]]), np.array([1]))

        english, chinese = self.model.postprocess(outputs)

        self.assertEqual(english, [[]])
        self.assertEqual(chinese, [[]])

    def test_tag_mode_defaults_to_english(self):
        self.assertEqual(normalize_tag_mode(""), "en")
        self.assertEqual(normalize_tag_mode(None), "en")
        self.assertEqual(normalize_tag_mode("en"), "en")
        self.assertEqual(normalize_tag_mode("zh"), "zh")

    def test_invalid_tag_mode_warns_and_falls_back_to_english(self):
        with mock.patch(
            "anylabeling.services.auto_labeling.__base__.ram.logger.warning"
        ) as warning:
            mode = normalize_tag_mode("cn")

        self.assertEqual(mode, "en")
        warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
