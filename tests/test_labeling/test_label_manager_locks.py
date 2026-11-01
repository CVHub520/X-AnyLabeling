import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

from anylabeling.views.labeling.widgets.label_dialog import LabelModifyDialog
from anylabeling.views.labeling.widgets.unique_label_qlist_widget import (
    UniqueLabelQListWidget,
)


class TestLabelManagerLocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.parent = QtWidgets.QWidget()
        self.parent.file_list_widget = QtWidgets.QListWidget()
        self.parent.unique_label_list = UniqueLabelQListWidget()
        self.parent.label_info = {}
        self.parent.output_dir = None
        self.parent._get_rgb_by_label = Mock(return_value=(0, 255, 0))
        self.parent.label_dialog = Mock()
        self.parent.toggle_unique_label_items_lock = Mock()
        self.parent.apply_label_visibility = Mock()
        self.paths = []
        for i in range(2):
            image = Path(self.temp.name) / f"{i}.png"
            self.parent.file_list_widget.addItem(str(image))
            path = image.with_suffix(".json")
            path.write_text(
                json.dumps(
                    {
                        "shapes": [
                            {"label": "cat", "locked": True},
                            {"label": "cat"},
                        ]
                    }
                )
            )
            self.paths.append(path)
        self.dialog = LabelModifyDialog(self.parent)

    def tearDown(self):
        self.dialog.close()
        self.parent.close()
        self.temp.cleanup()

    def test_manager_lock_only_updates_session_policy(self):
        before = [path.read_bytes() for path in self.paths]
        self.dialog._get_checkbox(0, 4).setChecked(True)
        with patch("anylabeling.views.labeling.widgets.label_dialog.Popup"):
            self.dialog.confirm_changes(1, 1)
        self.assertEqual([path.read_bytes() for path in self.paths], before)
        self.assertFalse(self.dialog.annotations_changed)
        self.assertTrue(self.parent.unique_label_list.lock_states["cat"])
        self.parent.toggle_unique_label_items_lock.assert_called_once()

    def test_unchanged_lock_preserves_individual_states(self):
        with patch("anylabeling.views.labeling.widgets.label_dialog.Popup"):
            self.dialog.confirm_changes(1, 2)
        for path in self.paths:
            shapes = json.loads(path.read_text())["shapes"]
            self.assertTrue(shapes[0]["locked"])
            self.assertNotIn("locked", shapes[1])

    def test_delete_checkbox_is_centered_and_has_no_icon(self):
        self.dialog.show()
        self.app.processEvents()
        checkbox = self.dialog._get_checkbox(0, 1)
        container = self.dialog.table_widget.cellWidget(0, 1)
        self.assertTrue(checkbox.icon().isNull())
        self.assertLessEqual(
            abs(
                checkbox.geometry().center().x()
                - container.rect().center().x()
            ),
            1,
        )
        checkbox.setChecked(True)
        self.assertTrue(self.dialog._get_value_edit(0).isReadOnly())

    def test_explicit_unlock_does_not_rewrite_json(self):
        before = [path.read_bytes() for path in self.paths]
        checkbox = self.dialog._get_checkbox(0, 4)
        checkbox.setChecked(True)
        checkbox.setChecked(False)
        with patch("anylabeling.views.labeling.widgets.label_dialog.Popup"):
            self.dialog.confirm_changes(1, 2)
        self.assertEqual([path.read_bytes() for path in self.paths], before)
        self.assertFalse(self.parent.unique_label_list.lock_states["cat"])
