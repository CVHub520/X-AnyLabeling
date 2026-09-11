"""Offscreen integration tests using the real labeling widget."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt6 import QtCore, QtWidgets

import anylabeling.resources.resources
from anylabeling.config import get_default_config
from anylabeling.views.labeling.label_widget import FILE_EXPORT_MARK_ROLE
from anylabeling.views.labeling.label_wrapper import LabelingWrapper


class SpecialImageExportWidgetTest(unittest.TestCase):
    """Exercise mark persistence, real list rebuilding, actions, and export."""

    @classmethod
    def setUpClass(cls) -> None:
        """Keep a QApplication alive for all widget tests."""
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        """Construct the real widget with isolated application settings."""
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "images"
        self.root.mkdir()
        settings = QtCore.QSettings(
            str(Path(self.temp.name) / "settings.ini"),
            QtCore.QSettings.Format.IniFormat,
        )
        self.window = QtWidgets.QMainWindow()
        config = get_default_config()
        config["exif_scan_enabled"] = False
        config["auto_save"] = False
        config["store_data"] = False
        config["file_search"] = ""
        with mock.patch(
            "anylabeling.views.labeling.label_widget.QtCore.QSettings",
            return_value=settings,
        ), mock.patch(
            "anylabeling.config.current_config_file",
            str(Path(__file__).resolve().parents[2] / "anylabeling/configs/xanylabeling_config.yaml"),
        ):
            wrapper = LabelingWrapper(self.window, config=config)
        self.window.setCentralWidget(wrapper)
        self.widget = wrapper.view
        self.widget.error_message = mock.Mock()

    def tearDown(self) -> None:
        """Release widgets without writing the user's application settings."""
        self.widget.auto_labeling_widget.model_manager.unload_model()
        self.window.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.temp.cleanup()

    def image(self, name: str, label: str = "") -> Path:
        """Create a real PNG and optionally a valid labeled annotation.

        Args:
            name: Image filename.
            label: Optional rectangle label.

        Returns:
            Created image path.
        """
        image = self.root / name
        Image.new("RGB", (32, 24), "white").save(image)
        if label:
            data = {
                "version": "4.0.0", "flags": {}, "checked": False,
                "shapes": [{"label": label, "points": [[1, 2], [20, 2], [20, 15], [1, 15]],
                            "shape_type": "rectangle", "flags": {}}],
                "imagePath": name, "imageData": None,
                "imageHeight": 24, "imageWidth": 32,
                "custom": {"keep": "original"},
            }
            image.with_suffix(".json").write_text(json.dumps(data), encoding="utf-8")
        return image

    def read_label(self, image: Path) -> dict:
        """Read a fixture annotation.

        Args:
            image: Image whose adjacent JSON is requested.

        Returns:
            Persisted annotation dictionary.
        """
        return json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))

    def test_mark_save_reload_and_unmark(self) -> None:
        """The actual QAction persists marks without changing shapes/custom data."""
        image = self.image("a.png", "defect")
        self.widget.import_image_folder(str(self.root))
        shapes = self.read_label(image)["shapes"]
        self.widget.actions.toggle_export_mark.trigger()
        data = self.read_label(image)
        self.assertTrue(data["export_marked"])
        self.assertEqual(data["shapes"][0]["points"], shapes[0]["points"])
        self.assertEqual(data["custom"], {"keep": "original"})
        self.assertFalse(data["checked"])
        self.widget.load_file(str(image))
        self.assertTrue(self.widget.actions.toggle_export_mark.isChecked())
        self.assertTrue(self.widget._current_file_item().data(FILE_EXPORT_MARK_ROLE))
        self.widget.actions.toggle_export_mark.trigger()
        self.assertFalse(self.read_label(image)["export_marked"])
        self.assertFalse(self.widget._current_file_item().font().bold())
        self.widget.error_message.assert_not_called()

    def test_unannotated_image_can_be_marked(self) -> None:
        """Marking a negative image creates an empty but loadable annotation."""
        image = self.image("negative.png")
        self.widget.import_image_folder(str(self.root))
        self.widget.set_export_marked(True)
        data = self.read_label(image)
        self.assertEqual(data["shapes"], [])
        self.assertTrue(data["export_marked"])
        self.widget.load_file(str(image))
        self.assertTrue(self.widget.actions.toggle_export_mark.isChecked())

    def test_failed_save_rolls_back_mark_and_keeps_dirty_state(self) -> None:
        """A failed save must not display an unpersisted mark as successful."""
        image = self.image("a.png")
        self.widget.import_image_folder(str(self.root))
        self.widget.set_dirty()
        with mock.patch.object(self.widget, "save_labels", return_value=False):
            self.widget.actions.toggle_export_mark.trigger()
        self.assertTrue(self.widget.dirty)
        self.assertNotIn("export_marked", self.widget.other_data)
        self.assertFalse(self.widget.actions.toggle_export_mark.isChecked())
        self.assertFalse(image.with_suffix(".json").exists())
        self.widget.set_clean()

    def test_failed_unmark_restores_persisted_mark(self) -> None:
        """Save failure when unmarking restores the existing true state."""
        image = self.image("a.png")
        self.widget.import_image_folder(str(self.root))
        self.widget.set_export_marked(True)
        with mock.patch.object(self.widget, "save_labels", return_value=False):
            self.widget.actions.toggle_export_mark.trigger()
        self.assertTrue(self.widget.actions.toggle_export_mark.isChecked())
        self.assertTrue(self.read_label(image)["export_marked"])

    def test_search_mark_clear_preserves_current_image_and_indicator(self) -> None:
        """Reproduce label search -> mark -> clear without mismatching the canvas."""
        first = self.image("a.png", "other")
        selected = self.image("b.png", "WR")
        self.widget.import_image_folder(str(self.root))
        self.widget.file_search.setText("label::WR")
        self.widget.file_search_changed()
        self.assertEqual(self.widget.image_path, str(selected))
        self.widget.set_export_marked(True)
        self.widget.file_search.setText("")
        self.widget.file_search_changed()
        self.assertEqual(self.widget.filename, str(selected))
        self.assertEqual(self.widget.image_path, str(selected))
        self.assertEqual(self.widget.file_list_widget.currentItem().text(), str(selected))
        self.assertTrue(self.widget._current_file_item().data(FILE_EXPORT_MARK_ROLE))
        self.assertTrue(self.widget.actions.toggle_export_mark.isChecked())
        self.assertNotIn("export_marked", self.read_label(first))
        self.widget.import_image_folder(str(self.root))
        item = self.widget.file_list_widget.item(self.widget.fn_to_index[str(selected)])
        self.assertTrue(item.data(FILE_EXPORT_MARK_ROLE))

    def test_empty_filter_disables_mark_and_clears_stale_indices(self) -> None:
        """An empty result cannot mark the previously displayed image."""
        self.image("a.png")
        self.widget.import_image_folder(str(self.root))
        self.widget.file_search.setText("export::1")
        self.widget.file_search_changed()
        self.assertEqual(self.widget.file_list_widget.count(), 0)
        self.assertEqual(self.widget.fn_to_index, {})
        self.assertIsNone(self.widget.filename)
        self.assertFalse(self.widget.actions.toggle_export_mark.isEnabled())
        self.widget.file_search.setText("")
        self.widget.file_search_changed()
        self.assertEqual(self.widget.file_list_widget.count(), 1)
        self.assertTrue(self.widget.actions.toggle_export_mark.isEnabled())

    def test_checked_status_is_independent(self) -> None:
        """Checking an annotation must retain its export selection."""
        image = self.image("a.png")
        self.widget.import_image_folder(str(self.root))
        self.widget.set_export_marked(True)
        self.widget.set_annotation_checked(True)
        self.widget.set_export_marked(False)
        data = self.read_label(image)
        self.assertTrue(data["checked"])
        self.assertFalse(data["export_marked"])

    def test_export_ignores_search_and_uses_output_directory(self) -> None:
        """The UI exports all marks, including files hidden by the active search."""
        first, second = self.image("a.png"), self.image("b.png")
        labels = Path(self.temp.name) / "labels"
        labels.mkdir()
        self.widget.output_dir = str(labels)
        self.widget.import_image_folder(str(self.root))
        self.widget.set_export_marked(True)
        self.widget.load_file(str(second))
        self.widget.set_export_marked(True)
        self.widget.file_search.setText("a.png")
        self.widget.file_search_changed()
        self.assertEqual(self.widget.file_list_widget.count(), 1)
        destination = Path(self.temp.name) / "export"
        with mock.patch.object(QtWidgets.QFileDialog, "getExistingDirectory", return_value=str(destination)), mock.patch.object(QtWidgets.QMessageBox, "information") as notice:
            self.widget.actions.export_marked_images.trigger()
        self.assertTrue((destination / first.name).exists())
        self.assertTrue((destination / second.name).exists())
        self.assertEqual(json.loads((destination / "b.json").read_text())["imagePath"], "b.png")
        self.widget.error_message.assert_not_called()
        notice.assert_called_once()

    def test_shortcuts_are_configurable_at_runtime(self) -> None:
        """The new shortcuts are registered in the existing settings mechanism."""
        self.assertEqual(self.widget.actions.toggle_export_mark.shortcut().toString(), "Ctrl+Alt+E")
        self.widget._settings_runtime_applier.apply_change("shortcuts.toggle_export_mark", "Ctrl+Shift+M")
        self.assertEqual(self.widget.actions.toggle_export_mark.shortcut().toString(), "Ctrl+Shift+M")

    def test_output_directory_change_reloads_the_current_mark(self) -> None:
        """Changing annotation roots must not reuse the old current mark."""
        image = self.image("a.png")
        self.widget.import_image_folder(str(self.root))
        self.widget.set_export_marked(True)
        labels = Path(self.temp.name) / "other-labels"
        labels.mkdir()
        self.widget.output_dir = str(labels)
        self.widget.import_image_folder(str(self.root), load=False)
        self.assertEqual(self.widget.image_path, str(image))
        self.assertFalse(self.widget.actions.toggle_export_mark.isChecked())
        self.assertFalse(self.widget._current_file_item().data(FILE_EXPORT_MARK_ROLE))
        self.assertTrue(self.read_label(image)["export_marked"])

    def test_remembered_folder_is_not_an_active_export_source(self) -> None:
        """A folder remembered from another session is not an export target."""
        self.widget.last_open_dir = str(self.root)
        with mock.patch.object(QtWidgets.QFileDialog, "getExistingDirectory") as dialog:
            self.widget.export_marked_images()
        dialog.assert_not_called()
        self.widget.error_message.assert_called_once()
