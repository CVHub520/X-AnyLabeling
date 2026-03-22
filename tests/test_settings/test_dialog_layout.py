import copy
import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtWidgets

    from anylabeling.views.labeling.settings.controller import SettingsController
    from anylabeling.views.labeling.settings.dialog import SettingsDialog
    from anylabeling.views.labeling.settings.schema import load_template_config

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for settings dialog tests")
class TestSettingsDialogLayout(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self._resources = []

    def tearDown(self):
        for dialog, controller in self._resources:
            controller.flush()
            dialog.close()
        self.app.processEvents()

    def _create_dialog(self):
        config = copy.deepcopy(load_template_config())
        controller = SettingsController(
            config=config,
            apply_callback=lambda _key, _value: None,
            save_callback=lambda _config: True,
            save_delay_ms=1000,
            defer_runtime_apply=True,
        )
        dialog = SettingsDialog(None, controller)
        dialog.show()
        self.app.processEvents()
        self._resources.append((dialog, controller))
        return dialog

    def test_general_uses_viewport_gap_and_fixed_bottom_height(self):
        dialog = self._create_dialog()
        dialog._render_primary("General")
        self.app.processEvents()

        viewport_margins = dialog.content_scroll.viewportMargins()
        self.assertEqual(viewport_margins.top(), 8)
        self.assertEqual(viewport_margins.bottom(), 8)

        body_margins = dialog.content_body_layout.contentsMargins()
        self.assertEqual(body_margins.left(), 0)
        self.assertEqual(body_margins.top(), 0)
        self.assertEqual(body_margins.right(), 0)
        self.assertEqual(body_margins.bottom(), 0)
        self.assertEqual(dialog._content_bottom_spacer.height(), 16)
        bottom_margins = dialog.shortcuts_bottom_layout.contentsMargins()
        self.assertEqual(bottom_margins.left(), 0)
        self.assertEqual(bottom_margins.right(), 0)
        self.assertEqual(bottom_margins.top(), 8)
        self.assertEqual(bottom_margins.bottom(), 8)

        self.assertEqual(dialog.shortcuts_bottom_panel.minimumHeight(), 48)
        self.assertEqual(dialog.shortcuts_bottom_panel.maximumHeight(), 48)
        self.assertEqual(dialog.shortcuts_bottom_panel.height(), 48)

    def test_canvas_uses_same_viewport_gap(self):
        dialog = self._create_dialog()
        dialog._render_primary("Canvas")
        self.app.processEvents()

        viewport_margins = dialog.content_scroll.viewportMargins()
        self.assertEqual(viewport_margins.top(), 8)
        self.assertEqual(viewport_margins.bottom(), 8)

        body_margins = dialog.content_body_layout.contentsMargins()
        self.assertEqual(body_margins.left(), 0)
        self.assertEqual(body_margins.top(), 0)
        self.assertEqual(body_margins.right(), 0)
        self.assertEqual(body_margins.bottom(), 0)
        self.assertEqual(dialog._content_bottom_spacer.height(), 16)
        bottom_margins = dialog.shortcuts_bottom_layout.contentsMargins()
        self.assertEqual(bottom_margins.left(), 0)
        self.assertEqual(bottom_margins.right(), 0)
        self.assertEqual(bottom_margins.top(), 8)
        self.assertEqual(bottom_margins.bottom(), 8)

    def test_shape_uses_same_viewport_gap(self):
        dialog = self._create_dialog()
        dialog._render_primary("Shape")
        self.app.processEvents()

        viewport_margins = dialog.content_scroll.viewportMargins()
        self.assertEqual(viewport_margins.top(), 8)
        self.assertEqual(viewport_margins.bottom(), 8)

        body_margins = dialog.content_body_layout.contentsMargins()
        self.assertEqual(body_margins.left(), 0)
        self.assertEqual(body_margins.top(), 0)
        self.assertEqual(body_margins.right(), 0)
        self.assertEqual(body_margins.bottom(), 0)
        self.assertEqual(dialog._content_bottom_spacer.height(), 16)
        bottom_margins = dialog.shortcuts_bottom_layout.contentsMargins()
        self.assertEqual(bottom_margins.left(), 0)
        self.assertEqual(bottom_margins.right(), 0)
        self.assertEqual(bottom_margins.top(), 8)
        self.assertEqual(bottom_margins.bottom(), 8)

    def test_general_titles_remain_english_while_descriptions_translate(self):
        with mock.patch.object(
            SettingsDialog,
            "tr",
            lambda _self, text: f"zh:{text}",
        ):
            dialog = self._create_dialog()
            dialog._render_primary("General")
            self.app.processEvents()

        first_row = dialog.content_body_layout.itemAt(0).widget()
        self.assertIsNotNone(first_row)
        first_row_texts = [
            label.text() for label in first_row.findChildren(QtWidgets.QLabel)
        ]
        self.assertIn("Auto Highlight Shape", first_row_texts)
        self.assertNotIn("zh:Auto Highlight Shape", first_row_texts)
        self.assertIn(
            "zh:In edit mode, automatically highlight vertices of selected objects.",
            first_row_texts,
        )

    def test_shortcuts_reset_viewport_margins(self):
        dialog = self._create_dialog()
        dialog._render_primary("General")
        self.app.processEvents()

        viewport_margins = dialog.content_scroll.viewportMargins()
        self.assertEqual(viewport_margins.top(), 8)
        self.assertEqual(viewport_margins.bottom(), 8)

        dialog._render_primary("Shortcuts")
        self.app.processEvents()
        viewport_margins = dialog.content_scroll.viewportMargins()
        self.assertEqual(viewport_margins.top(), 0)
        self.assertEqual(viewport_margins.bottom(), 0)
        self.assertEqual(dialog._content_bottom_spacer.height(), 0)
        bottom_margins = dialog.shortcuts_bottom_layout.contentsMargins()
        self.assertEqual(bottom_margins.left(), 16)
        self.assertEqual(bottom_margins.right(), 16)
        self.assertEqual(bottom_margins.top(), 8)
        self.assertEqual(bottom_margins.bottom(), 8)

    def test_shortcuts_reset_handles_transient_conflict(self):
        dialog = self._create_dialog()
        dialog._render_primary("Shortcuts")
        self.app.processEvents()

        group_list = dialog._shortcut_group_list
        self.assertIsNotNone(group_list)
        for row in range(group_list.count()):
            if group_list.item(row).text().startswith("View"):
                group_list.setCurrentRow(row)
                break
        self.app.processEvents()

        controller = dialog._controller
        controller.update_field(
            "shortcuts.show_masks",
            "Alt+M",
            schedule_save=False,
        )
        controller.update_field(
            "shortcuts.toggle_compare_view",
            "Ctrl+M",
            schedule_save=False,
        )
        dialog._confirm_reset = lambda *_args, **_kwargs: True

        dialog._on_shortcuts_reset_clicked()
        self.app.processEvents()

        self.assertEqual(controller.get_value("shortcuts.show_masks"), "Ctrl+M")
        self.assertEqual(
            controller.get_value("shortcuts.toggle_compare_view"),
            "Ctrl+Alt+C",
        )

    def test_close_discards_unsaved_changes(self):
        dialog = self._create_dialog()
        controller = dialog._controller
        initial_value = controller.get_value("model_hub")
        next_value = (
            "modelscope" if initial_value != "modelscope" else "huggingface"
        )

        controller.update_field("model_hub", next_value, schedule_save=False)
        self.assertEqual(controller.get_value("model_hub"), next_value)

        dialog.close()
        self.app.processEvents()

        self.assertEqual(controller.get_value("model_hub"), initial_value)
