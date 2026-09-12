import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from PIL import Image
from PyQt6 import QtGui

from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.views.labeling.label_file import LabelFile
from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.utils.batch import save_auto_labeling_result
from anylabeling.views.labeling.utils.image_tags import (
    normalize_image_tag,
    normalize_image_tags,
)


class ImageTagsTest(unittest.TestCase):
    def test_normalize_tag_text(self):
        self.assertEqual(normalize_image_tag("  street  "), "street")
        self.assertEqual(normalize_image_tag("标签"), "标签")
        self.assertIsNone(normalize_image_tag(""))
        self.assertIsNone(normalize_image_tag("a\nb"))
        self.assertIsNone(normalize_image_tag(1))

    def test_normalize_tag_list(self):
        value = [" car ", "car", "Car", "", "a\nb", 1, "street"]
        self.assertEqual(normalize_image_tags(value), ["car", "Car", "street"])

    def test_auto_labeling_result_tags_are_optional(self):
        self.assertIsNone(AutoLabelingResult([]).tags)
        self.assertEqual(AutoLabelingResult([], tags=[]).tags, [])

    def test_label_file_normalizes_tags_and_preserves_other_data(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "image.png")
            label_path = os.path.join(directory, "image.json")
            Image.new("RGB", (8, 6)).save(image_path)
            data = {
                "version": "4.0.0",
                "flags": {"reviewed": True},
                "shapes": [],
                "imagePath": "image.png",
                "imageData": None,
                "imageHeight": 6,
                "imageWidth": 8,
                "description": "unchanged",
                "tags": [" car ", "car", "Car", None],
                "custom": {"value": 1},
            }
            with open(label_path, "w", encoding="utf-8") as stream:
                json.dump(data, stream)

            with mock.patch(
                "anylabeling.views.labeling.utils.image_tags.logger.warning"
            ) as warning:
                label_file = LabelFile(label_path)

            self.assertEqual(label_file.other_data["tags"], ["car", "Car"])
            self.assertEqual(label_file.other_data["description"], "unchanged")
            self.assertEqual(label_file.other_data["custom"], {"value": 1})
            self.assertEqual(label_file.flags, {"reviewed": True})
            warning.assert_called_once()

    def test_label_file_accepts_invalid_tags_field(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "image.png")
            label_path = os.path.join(directory, "image.json")
            Image.new("RGB", (8, 6)).save(image_path)
            data = {
                "version": "4.0.0",
                "flags": {},
                "shapes": [],
                "imagePath": "image.png",
                "imageData": None,
                "imageHeight": 6,
                "imageWidth": 8,
                "tags": "car",
            }
            with open(label_path, "w", encoding="utf-8") as stream:
                json.dump(data, stream)

            label_file = LabelFile(label_path)

            self.assertIn("tags", label_file.other_data)
            self.assertEqual(label_file.other_data["tags"], [])

    def test_tag_only_empty_list_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "image.png")
            label_path = os.path.join(directory, "image.json")
            Image.new("RGB", (8, 6)).save(image_path)

            LabelFile().save(
                filename=label_path,
                shapes=[],
                image_path="image.png",
                image_height=6,
                image_width=8,
                image_data=None,
                other_data={"tags": [], "custom": "keep"},
                flags={"reviewed": False},
            )
            loaded = LabelFile(label_path)

            self.assertEqual(loaded.shapes, [])
            self.assertEqual(loaded.other_data["tags"], [])
            self.assertEqual(loaded.other_data["custom"], "keep")
            self.assertEqual(loaded.flags, {"reviewed": False})


class ImageTagsLabelingWidgetTest(unittest.TestCase):
    def make_widget(self, tags_marker=True):
        other_data = {"description": "keep", "checked": True}
        if tags_marker:
            other_data["tags"] = ["person"]
        return SimpleNamespace(
            image=True,
            image_path="/tmp/image.png",
            filename="/tmp/image.png",
            canvas=SimpleNamespace(shapes=[]),
            label_list=mock.Mock(),
            load_shapes=mock.Mock(),
            other_data=other_data,
            image_tags_widget=mock.Mock(),
            _auto_show_image_tags=mock.Mock(),
            set_dirty=mock.Mock(),
            _sync_annotation_checked_state=mock.Mock(),
            shape_text_edit=mock.Mock(),
            shape_text_label=mock.Mock(),
            tr=lambda text: text,
        )

    def test_equal_auto_tags_do_not_mark_dirty(self):
        widget = self.make_widget()
        result = AutoLabelingResult([], replace=False, tags=[" person "])

        LabelingWidget.new_shapes_from_auto_labeling(widget, result)

        self.assertEqual(widget.other_data["tags"], ["person"])
        widget.image_tags_widget.set_tags.assert_called_once_with(["person"])
        widget._auto_show_image_tags.assert_called_once_with()
        widget.set_dirty.assert_not_called()
        self.assertTrue(widget.other_data["checked"])
        widget._sync_annotation_checked_state.assert_not_called()

    def test_auto_shapes_reset_checked_before_saving(self):
        for replace in (False, True):
            with self.subTest(replace=replace):
                widget = self.make_widget()
                widget.set_dirty.side_effect = lambda: self.assertFalse(
                    widget.other_data["checked"]
                )
                result = AutoLabelingResult([object()], replace=replace)

                LabelingWidget.new_shapes_from_auto_labeling(widget, result)

                self.assertFalse(widget.other_data["checked"])
                widget._sync_annotation_checked_state.assert_called_once_with()
                widget.set_dirty.assert_called_once_with()

    def test_auto_replacement_removing_shapes_resets_checked(self):
        widget = self.make_widget()
        widget.canvas.shapes = [SimpleNamespace(locked=False)]

        LabelingWidget.new_shapes_from_auto_labeling(
            widget, AutoLabelingResult([], replace=True)
        )

        self.assertFalse(widget.other_data["checked"])
        widget._sync_annotation_checked_state.assert_called_once_with()

    def test_toggle_image_tags_visibility_tracks_explicit_state(self):
        widget = SimpleNamespace(
            _image_tags_visibility="auto",
            image_tags_widget=mock.Mock(),
        )

        LabelingWidget.toggle_image_tags_visibility(widget, True)
        self.assertEqual(widget._image_tags_visibility, "explicit_visible")
        widget.image_tags_widget.setVisible.assert_called_once_with(True)

        LabelingWidget.toggle_image_tags_visibility(widget, False)
        self.assertEqual(widget._image_tags_visibility, "explicit_hidden")
        widget.image_tags_widget.setVisible.assert_called_with(False)

    def test_batch_result_replaces_tags_without_changing_other_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "image.png")
            label_path = os.path.join(directory, "image.json")
            Image.new("RGB", (8, 6)).save(image_path)
            data = {
                "version": "4.0.0",
                "flags": {"reviewed": True},
                "shapes": [],
                "imagePath": "image.png",
                "imageData": None,
                "imageHeight": 6,
                "imageWidth": 8,
                "description": "keep",
                "tags": ["old"],
                "checked": True,
            }
            with open(label_path, "w", encoding="utf-8") as stream:
                json.dump(data, stream)
            widget = SimpleNamespace(
                output_dir=None, _config={"store_data": False}
            )

            save_auto_labeling_result(
                widget,
                image_path,
                AutoLabelingResult(
                    [], replace=False, tags=[" street ", "street", "Car"]
                ),
            )

            with open(label_path, encoding="utf-8") as stream:
                saved = json.load(stream)
            self.assertEqual(saved["tags"], ["street", "Car"])
            self.assertEqual(saved["flags"], {"reviewed": True})
            self.assertEqual(saved["description"], "keep")
            self.assertFalse(saved["checked"])

    def test_batch_result_with_none_tags_preserves_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "image.png")
            label_path = os.path.join(directory, "image.json")
            Image.new("RGB", (8, 6)).save(image_path)
            data = {
                "version": "4.0.0",
                "flags": {},
                "shapes": [],
                "imagePath": "image.png",
                "imageData": None,
                "imageHeight": 6,
                "imageWidth": 8,
                "description": "",
                "tags": ["keep"],
                "checked": True,
            }
            with open(label_path, "w", encoding="utf-8") as stream:
                json.dump(data, stream)
            widget = SimpleNamespace(
                output_dir=None, _config={"store_data": False}
            )

            save_auto_labeling_result(
                widget, image_path, AutoLabelingResult([], replace=False)
            )

            with open(label_path, encoding="utf-8") as stream:
                saved = json.load(stream)
            self.assertEqual(saved["tags"], ["keep"])
            self.assertTrue(saved["checked"])

    def test_empty_auto_tags_create_field_and_mark_dirty(self):
        widget = self.make_widget(tags_marker=False)
        result = AutoLabelingResult([], replace=False, tags=[])

        LabelingWidget.new_shapes_from_auto_labeling(widget, result)

        self.assertEqual(widget.other_data["tags"], [])
        self.assertEqual(widget.other_data["description"], "keep")
        widget.set_dirty.assert_called_once_with()
        self.assertFalse(widget.other_data["checked"])
        widget._sync_annotation_checked_state.assert_called_once_with()

    def test_none_auto_tags_preserve_existing_field(self):
        widget = self.make_widget()
        result = AutoLabelingResult([], replace=False)

        LabelingWidget.new_shapes_from_auto_labeling(widget, result)

        self.assertEqual(widget.other_data["tags"], ["person"])
        widget.image_tags_widget.set_tags.assert_not_called()
        widget.set_dirty.assert_called_once_with()
        self.assertTrue(widget.other_data["checked"])
        widget._sync_annotation_checked_state.assert_not_called()

    def test_stale_auto_tags_are_ignored(self):
        widget = self.make_widget()
        result = AutoLabelingResult(
            [],
            replace=False,
            image_path="/tmp/another.png",
            tags=["street"],
        )

        LabelingWidget.new_shapes_from_auto_labeling(widget, result)

        self.assertEqual(widget.other_data["tags"], ["person"])
        widget.image_tags_widget.set_tags.assert_not_called()
        widget.set_dirty.assert_not_called()
        self.assertTrue(widget.other_data["checked"])
        widget._sync_annotation_checked_state.assert_not_called()

    def test_image_tag_colors_use_shape_manual_and_stable_priorities(self):
        shape = SimpleNamespace(
            label="shape", line_color=QtGui.QColor(1, 2, 3)
        )
        widget = SimpleNamespace(
            canvas=SimpleNamespace(shapes=[shape]),
            _config={"label_colors": {"manual": [4, 5, 6]}},
        )

        shape_color = LabelingWidget._get_rgb_by_image_tag(widget, "shape")
        manual_color = LabelingWidget._get_rgb_by_image_tag(widget, "manual")
        stable_color = LabelingWidget._get_rgb_by_image_tag(widget, "stable")

        self.assertEqual(shape_color, (1, 2, 3))
        self.assertEqual(manual_color, (4, 5, 6))
        self.assertEqual(
            stable_color,
            LabelingWidget._get_rgb_by_image_tag(widget, "stable"),
        )

        formerly_colliding_tags = [
            "almond",
            "container",
            "lunch",
            "plastic",
            "food",
            "fruit",
            "tray",
            "vegetable",
        ]
        fallback_colors = {
            LabelingWidget._get_rgb_by_image_tag(widget, tag)
            for tag in formerly_colliding_tags
        }
        self.assertEqual(len(fallback_colors), len(formerly_colliding_tags))


if __name__ == "__main__":
    unittest.main()
