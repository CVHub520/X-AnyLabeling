"""Regression tests for non-destructive special image exports."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from anylabeling.views.labeling.utils.special_image_export import (
    ExportCancelled,
    export_marked_pairs,
    read_export_mark,
)
from anylabeling.views.labeling.utils.file_search import (
    filter_image_files,
    parse_search_pattern,
)


class SpecialImageExportTest(unittest.TestCase):
    """Exercise filesystem safety and persisted search semantics."""

    def setUp(self) -> None:
        """Create isolated source and destination folders."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.root.mkdir()
        self.destination = Path(self.temp.name) / "export"

    def pair(self, name: str, marked: bool = True) -> Path:
        """Create an image and annotation fixture.

        Args:
            name: Relative image path.
            marked: Persisted export mark.

        Returns:
            Created image path.
        """
        image = self.root / name
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image content")
        image.with_suffix(".json").write_text(json.dumps({
            "imagePath": "../original/" + image.name,
            "imageData": None,
            "export_marked": marked,
            "checked": True,
            "shapes": [{"label": "defect", "points": [[1, 2], [3, 4]]}],
            "custom": {"description": "review"},
        }), encoding="utf-8")
        return image

    def export(self, images: list[Path], **kwargs: object) -> int:
        """Export fixtures with optional helper arguments.

        Args:
            images: Source images.
            **kwargs: Additional export options.

        Returns:
            Exported pair count.
        """
        return export_marked_pairs(
            [str(p) for p in images], str(self.root), str(self.destination), **kwargs
        )

    def test_preserves_tree_annotations_and_sources(self) -> None:
        """Same basenames in different directories must remain distinct."""
        first = self.pair("farm-a/图片.png")
        second = self.pair("farm-b/图片.png")
        unmarked = self.pair("other.png", False)
        original = first.with_suffix(".json").read_bytes()
        self.assertEqual(self.export([first, second, unmarked]), 2)
        for image in (first, second):
            exported = self.destination / image.relative_to(self.root)
            self.assertEqual(exported.read_bytes(), image.read_bytes())
            data = json.loads(exported.with_suffix(".json").read_text(encoding="utf-8"))
            source = json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))
            source["imagePath"] = exported.name
            self.assertEqual(data, source)
        self.assertEqual(first.with_suffix(".json").read_bytes(), original)
        self.assertFalse((self.destination / unmarked.name).exists())

    def test_flat_output_directory(self) -> None:
        """Use the configured annotation directory rather than adjacent labels."""
        image = self.pair("nested/a.png")
        labels = Path(self.temp.name) / "labels"
        labels.mkdir()
        image.with_suffix(".json").rename(labels / "a.json")
        self.assertEqual(self.export([image], output_dir=str(labels)), 1)
        self.assertTrue((self.destination / "nested/a.json").exists())

    def test_conflict_preflight_does_not_copy_any_pairs(self) -> None:
        """An existing target must abort before any other pair is copied."""
        first, second = self.pair("a.png"), self.pair("b.png")
        self.destination.mkdir()
        existing = self.destination / "b.json"
        existing.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.export([first, second])
        self.assertEqual(existing.read_text(), "keep")
        self.assertFalse((self.destination / "a.png").exists())

    def test_rejects_source_and_nested_destination(self) -> None:
        """Export cannot pollute the source dataset."""
        image = self.pair("a.png")
        for destination in (self.root, self.root / "export"):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                export_marked_pairs([str(image)], str(self.root), str(destination))

    def test_rejects_images_outside_source(self) -> None:
        """Relative parent traversal must never escape the destination."""
        image = Path(self.temp.name) / "outside.png"
        image.write_bytes(b"image")
        with self.assertRaises(ValueError):
            self.export([image])

    def test_missing_unmarked_and_corrupt_annotations(self) -> None:
        """Missing labels are unmarked; corrupt labels fail explicitly."""
        image = self.pair("a.png", False)
        label = image.with_suffix(".json")
        self.assertFalse(read_export_mark(str(label)))
        self.assertEqual(self.export([image]), 0)
        label.unlink()
        self.assertEqual(self.export([image]), 0)
        label.write_text("{broken", encoding="utf-8")
        self.assertFalse(read_export_mark(str(label)))
        with self.assertRaises(ValueError):
            self.export([image])

    def test_rejects_ambiguous_same_stem_annotations(self) -> None:
        """Two images sharing one JSON cannot be silently mispaired."""
        first = self.pair("a.png")
        second = self.root / "a.jpg"
        second.write_bytes(b"image")
        with self.assertRaisesRegex(ValueError, "same annotation"):
            self.export([first, second])

    def test_rejects_flat_output_label_collisions(self) -> None:
        """A flat output directory cannot disambiguate duplicate basenames."""
        first = self.pair("one/a.png")
        second = self.pair("two/a.png")
        with self.assertRaisesRegex(ValueError, "same annotation"):
            self.export([first, second], output_dir=str(first.parent))

    def test_cancellation_rolls_back_completed_pairs(self) -> None:
        """Cancellation removes only files created by this export."""
        images = [self.pair("a.png"), self.pair("b.png")]

        def cancel(value: int, total: int) -> None:
            """Cancel after one pair has been copied.

            Args:
                value: Completed steps.
                total: Total steps.
            """
            if value == 3:
                raise ExportCancelled()

        with self.assertRaises(ExportCancelled):
            self.export(images, progress=cancel)
        self.assertEqual(list(self.destination.rglob("*.*")), [])
        self.assertTrue(all(image.exists() for image in images))

    def test_copy_error_rolls_back_partial_file(self) -> None:
        """A failed copy cannot leave an incomplete exported image."""
        image = self.pair("a.png")
        with mock.patch(
            "anylabeling.views.labeling.utils.special_image_export.shutil.copyfileobj",
            side_effect=OSError("disk full"),
        ), self.assertRaises(OSError):
            self.export([image])
        self.assertEqual(list(self.destination.rglob("*.*")), [])

    def test_search_marks_and_missing_labels(self) -> None:
        """Search supports marked/unmarked independently of checked status."""
        marked, unmarked = self.pair("yes.png"), self.pair("no.png", False)
        missing = self.root / "missing.png"
        missing.write_bytes(b"image")
        images = list(map(str, (marked, unmarked, missing)))
        self.assertEqual(filter_image_files(images, parse_search_pattern("export::1")), [str(marked)])
        self.assertEqual(filter_image_files(images, parse_search_pattern("export::0")), [str(unmarked), str(missing)])
        self.assertEqual(filter_image_files(images, parse_search_pattern("checked::1")), images[:2])

    def test_export_mark_requires_json_boolean(self) -> None:
        """Truthy strings and integers are not valid export marks."""
        image = self.pair("a.png")
        for value in ("true", 1, None, False):
            image.with_suffix(".json").write_text(json.dumps({"export_marked": value}), encoding="utf-8")
            self.assertFalse(read_export_mark(str(image.with_suffix(".json"))))
            self.assertEqual(self.export([image]), 0)

    def test_file_created_after_preflight_is_not_overwritten(self) -> None:
        """Exclusive creation also protects files appearing during the export."""
        image = self.pair("a.png")

        def create_conflict(value: int, total: int) -> None:
            """Simulate another process creating the target after scanning.

            Args:
                value: Completed steps.
                total: Total steps.
            """
            if value == 1:
                self.destination.mkdir()
                (self.destination / image.name).write_bytes(b"keep concurrent file")

        with self.assertRaises(FileExistsError):
            self.export([image], progress=create_conflict)
        self.assertEqual((self.destination / image.name).read_bytes(), b"keep concurrent file")

    def test_invalid_json_structure_does_not_break_export_search(self) -> None:
        """A non-object annotation is excluded from search instead of crashing."""
        image = self.pair("a.png")
        image.with_suffix(".json").write_text("[]", encoding="utf-8")
        self.assertEqual(filter_image_files([str(image)], parse_search_pattern("export::1")), [])
