import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.widgets.image_tags_widget import (
    IMAGE_TAG_CLOSE_SIZE,
    IMAGE_TAG_HEIGHT,
    IMAGE_TAG_RADIUS,
    IMAGE_TAG_SPACING,
    ImageTagsWidget,
)
from anylabeling.views.labeling.utils.theme import (
    get_app_stylesheet,
    get_dark_palette,
    get_mode,
    init_theme,
)


class ImageTagsWidgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def setUp(self):
        self.widget = ImageTagsWidget(lambda _tag: (32, 120, 200))
        self.widget.set_interactions_enabled(True)
        self.changes = []
        self.widget.tags_changed.connect(self.changes.append)

    def tearDown(self):
        self.widget.close()
        self.widget.deleteLater()
        self.app.processEvents()

    def test_add_edit_and_cancel(self):
        self.widget.start_add()
        self.widget._input.setText("  street  ")
        self.assertTrue(self.widget.commit_add())
        self.assertEqual(self.widget.tags, ["street"])

        self.widget.start_edit(0)
        self.widget._input.setText("road")
        self.assertTrue(self.widget.commit_edit())
        self.assertEqual(self.widget.tags, ["road"])

        self.widget.start_edit(0)
        self.widget._input.setText("ignored")
        self.widget.cancel_input()
        self.assertEqual(self.widget.tags, ["road"])
        self.assertEqual(self.changes, [["street"], ["road"]])

    def test_add_uses_next_available_default_tag(self):
        self.widget.set_tags(["tag", "tag1"])
        self.widget.start_add()

        self.assertEqual(self.widget._input.text(), "tag2")
        self.widget._input.clear()
        self.assertTrue(self.widget.commit_add())
        self.assertEqual(self.widget.tags, ["tag", "tag1"])
        self.assertEqual(self.widget.mode, "normal")
        self.assertEqual(self.changes, [])

    def test_empty_edit_removes_tag(self):
        self.widget.set_tags(["person", "car"])
        self.widget.start_edit(0)
        self.widget._input.clear()

        self.assertTrue(self.widget.commit_edit())
        self.assertEqual(self.widget.tags, ["car"])
        self.assertEqual(self.changes, [["car"]])

    def test_edit_focuses_cursor_at_end_without_selecting_text(self):
        self.widget.set_tags(["person"])
        self.widget.start_edit(0)
        self.app.processEvents()

        self.assertFalse(self.widget._input.hasSelectedText())
        self.assertEqual(self.widget._input.cursorPosition(), len("person"))

    def test_chip_uses_arrow_cursor_and_shows_edit_hint(self):
        self.widget.set_tags(["person"])
        messages = []
        self.widget.status_message.connect(messages.append)
        chip = self.widget._chips[0]

        enter = QtGui.QEnterEvent(
            QtCore.QPointF(1, 1),
            QtCore.QPointF(1, 1),
            QtCore.QPointF(1, 1),
        )
        chip.enterEvent(enter)

        self.assertEqual(
            chip.cursor().shape(), QtCore.Qt.CursorShape.ArrowCursor
        )
        self.assertEqual(messages[-1], "Double-click a tag to edit its text.")

    def test_invalid_and_duplicate_inputs_stay_editable(self):
        self.widget.set_tags(["car", "Car"])
        self.widget.start_add()
        self.widget._input.setText(" car ")
        self.assertFalse(self.widget.commit_add())
        self.assertEqual(self.widget.mode, "add")

        self.widget._input.setText("a\nb")
        self.assertFalse(self.widget.commit_add())
        self.assertEqual(self.widget.tags, ["car", "Car"])

    def test_single_delete_requires_confirmation(self):
        self.widget.set_tags(["person", "street"])
        self.widget._confirm_single_delete = lambda _index: False
        self.widget.delete_tag(0)
        self.assertEqual(self.widget.tags, ["person", "street"])

        self.widget._confirm_single_delete = lambda _index: True
        self.widget.delete_tag(0)
        self.assertEqual(self.widget.tags, ["street"])
        self.assertEqual(self.changes, [["street"]])

    def test_batch_delete_preserves_relative_order(self):
        self.widget.set_tags(["a", "b", "c", "d"])
        self.widget.start_batch_mode()
        self.widget.toggle_selection(1)
        self.widget.toggle_selection(3)
        self.widget._confirm_batch_delete = lambda: True

        self.widget.delete_selected()

        self.assertEqual(self.widget.tags, ["a", "c"])
        self.assertEqual(self.widget.mode, "normal")
        self.assertEqual(self.changes, [["a", "c"]])

    def test_batch_mode_uses_strike_selection_without_leading_space(self):
        self.widget.set_tags(["person", "street"])
        normal_widths = [
            chip.sizeHint().width() for chip in self.widget._chips
        ]

        self.widget.start_batch_mode()
        batch_widths = [chip.sizeHint().width() for chip in self.widget._chips]
        self.widget.toggle_selection(0)
        self.widget.show()
        self.app.processEvents()

        self.assertEqual(batch_widths, normal_widths)
        self.assertFalse(hasattr(self.widget._chips[0], "selection_indicator"))
        self.assertEqual(self.widget._chips[0].text_label.geometry().left(), 9)
        self.assertTrue(self.widget._chips[0]._selected)

    def test_batch_buttons_share_tag_flow_and_wrap(self):
        self.widget.setFixedSize(600, 200)
        self.widget.set_tags(["person"])
        self.widget.start_batch_mode()
        self.widget.show()
        self.app.processEvents()

        items = [
            self.widget._flow.itemAt(index).widget()
            for index in range(self.widget._flow.count())
        ]
        self.assertEqual(
            items[1:],
            [
                self.widget._select_all_button,
                self.widget._delete_selected_button,
                self.widget._cancel_button,
            ],
        )
        self.assertTrue(all(item.y() == items[0].y() for item in items))
        self.assertTrue(
            all(item.height() == IMAGE_TAG_HEIGHT for item in items)
        )
        self.assertIn(
            "border-radius: 14px",
            self.widget._select_all_button.styleSheet(),
        )

        select_width = self.widget._select_all_button.width()
        self.widget.toggle_select_all()
        self.app.processEvents()
        self.assertEqual(self.widget._select_all_button.width(), select_width)
        self.assertEqual(self.widget._delete_selected_button.text(), "Delete")

        required_width = (
            sum(item.sizeHint().width() for item in items)
            + (len(items) - 1) * IMAGE_TAG_SPACING
            + 2 * IMAGE_TAG_SPACING
        )
        self.widget._flow.setGeometry(
            QtCore.QRect(0, 0, required_width - 1, 200)
        )
        self.assertEqual(
            items[-1].y() - items[0].y(),
            IMAGE_TAG_HEIGHT + IMAGE_TAG_SPACING,
        )

    def test_batch_confirmation_uses_compact_qt_question(self):
        self.widget.set_tags(["person", "street"])
        self.widget.start_batch_mode()
        self.widget.toggle_select_all()

        with mock.patch.object(
            QtWidgets.QMessageBox,
            "question",
            return_value=QtWidgets.QMessageBox.StandardButton.Yes,
        ) as question:
            self.assertTrue(self.widget._confirm_batch_delete())

        arguments = question.call_args.args
        self.assertEqual(arguments[1], "Delete Image Tags")
        self.assertEqual(arguments[2], "Delete 2 selected tags?")
        self.assertNotIn("person", arguments[2])
        self.assertNotIn("street", arguments[2])

    def test_drag_reorders_only_when_position_changes(self):
        self.widget.set_tags(["a", "b", "c"])

        self.widget.finish_drag(0, QtCore.QPoint(10000, 10000))
        self.assertEqual(self.widget.tags, ["b", "c", "a"])
        self.assertEqual(self.changes, [["b", "c", "a"]])

        self.widget.finish_drag(2, QtCore.QPoint(10000, 10000))
        self.assertEqual(self.changes, [["b", "c", "a"]])

    def test_image_change_commits_valid_input_and_cancels_invalid_input(self):
        self.widget.start_add()
        self.widget._input.setText("person")
        self.widget.finish_for_image_change()
        self.assertEqual(self.widget.tags, ["person"])

        self.widget.start_add()
        self.widget._input.setText("person")
        self.widget.finish_for_image_change()
        self.assertEqual(self.widget.mode, "normal")
        self.assertEqual(self.widget.tags, ["person"])

    def test_batch_cancel_does_not_emit_change(self):
        self.widget.set_tags(["a", "b"])
        self.widget.start_batch_mode()
        self.widget.toggle_select_all()

        self.assertTrue(self.widget.cancel_active_mode())
        self.assertEqual(self.widget.tags, ["a", "b"])
        self.assertEqual(self.changes, [])

    def test_large_tag_set_uses_bounded_internal_scroll_area(self):
        self.widget.resize(260, 200)
        self.widget.set_tags([f"tag-{index}" for index in range(30)])
        self.widget.show()
        self.app.processEvents()
        self.widget._refresh_height()

        self.assertLessEqual(self.widget._scroll.height(), 112)
        self.assertGreater(
            self.widget._content.minimumHeight(),
            self.widget._scroll.viewport().height(),
        )

    def test_long_tag_is_elided_without_tooltip(self):
        tag = "a" * 200
        self.widget.set_tags([tag])

        chip = self.widget._chips[0]
        self.assertEqual(chip.toolTip(), "")
        self.assertEqual(chip.accessibleName(), tag)
        self.assertNotEqual(chip.text_label.text(), tag)

    def test_chip_add_button_and_editor_share_height(self):
        self.widget.set_tags(["person"])
        self.widget.show()
        self.app.processEvents()

        chip = self.widget._chips[0]
        add_button = self.widget._flow.itemAt(1).widget()
        self.assertEqual(chip.height(), IMAGE_TAG_HEIGHT)
        self.assertEqual(add_button.height(), IMAGE_TAG_HEIGHT)
        self.assertEqual(
            self.widget.frameShape(), QtWidgets.QFrame.Shape.NoFrame
        )

        width_before = chip.sizeHint().width()
        chip._set_delete_visible(True)
        self.assertEqual(chip.sizeHint().width(), width_before)
        self.assertEqual(chip.delete_button.geometry().top(), 0)
        self.assertEqual(
            chip.delete_button.geometry().right(), chip.width() - 1
        )

        self.widget.start_add()
        self.assertEqual(self.widget._input.height(), IMAGE_TAG_HEIGHT)

    def test_add_button_and_editor_keep_stable_visual_style(self):
        self.widget.set_tags(["person"])
        add_button = self.widget._flow.itemAt(1).widget()
        self.assertNotIn("dashed", add_button.styleSheet())
        self.assertIn("border-radius: 14px", add_button.styleSheet())

        self.widget.start_add()
        style_before = self.widget._input.styleSheet()
        width_before = self.widget._input.width()
        self.widget._input.setText("a completely different label")

        self.assertEqual(self.widget._input.styleSheet(), style_before)
        self.assertEqual(self.widget._input.width(), width_before)

    def test_minus_button_replaces_batch_delete_context_menu(self):
        self.widget.set_tags(["person"])
        add_button = self.widget._flow.itemAt(1).widget()
        batch_button = self.widget._flow.itemAt(2).widget()
        copy_button = self.widget._flow.itemAt(3).widget()

        self.assertEqual(
            self.widget._content.contextMenuPolicy(),
            QtCore.Qt.ContextMenuPolicy.DefaultContextMenu,
        )
        self.assertEqual(add_button.text(), "+")
        self.assertEqual(batch_button.text(), "−")
        self.assertEqual(copy_button.text(), "C")
        self.assertIn("font-size: 14px", copy_button.styleSheet())
        self.assertEqual(add_button.styleSheet(), batch_button.styleSheet())

        batch_button.click()
        self.assertEqual(self.widget.mode, "batch")

    @mock.patch("anylabeling.views.labeling.widgets.image_tags_widget.Popup")
    def test_copy_all_button_uses_popup_component(self, popup):
        self.widget.set_tags(["person", "street", "traffic light"])
        button = self.widget._copy_all_button

        self.assertIs(
            self.widget._flow.itemAt(self.widget._flow.count() - 1).widget(),
            button,
        )
        self.assertEqual(button.toolTip(), "Copy All Tags")
        button.click()

        popup.assert_called_once()
        self.assertEqual(popup.call_args.args[0], "Copy Successful")
        popup.return_value.show_popup.assert_called_once_with(
            self.widget.window(),
            copy_msg="person,street,traffic light",
            position="default",
        )
        self.assertEqual(self.changes, [])

    def test_action_buttons_preserve_subtle_background_style(self):
        self.widget.set_tags(["person"])
        add_button = self.widget._flow.itemAt(1).widget()
        self.assertIn(
            "background-color: rgba(128, 128, 128, 35)",
            add_button.styleSheet(),
        )

        self.widget.start_batch_mode()
        for button in (
            self.widget._select_all_button,
            self.widget._delete_selected_button,
            self.widget._cancel_button,
        ):
            self.assertIn(
                "background-color: rgba(128, 128, 128, 35)",
                button.styleSheet(),
            )
        self.assertIn(
            "color: palette(placeholder-text)",
            self.widget._delete_selected_button.styleSheet(),
        )

    def test_dark_theme_preserves_button_and_close_geometry(self):
        original_mode = get_mode()
        original_palette = QtGui.QPalette(self.app.palette())
        original_style = self.app.style().objectName()
        original_stylesheet = self.app.styleSheet()
        try:
            init_theme("dark")
            self.app.setStyle("Fusion")
            self.app.setPalette(get_dark_palette())
            self.app.setStyleSheet(get_app_stylesheet())
            self.widget.set_tags(["person"])
            self.widget.show()
            self.app.processEvents()

            chip = self.widget._chips[0]
            chip._set_delete_visible(True)
            self.app.processEvents()
            self.assertEqual(
                chip.delete_button.size(),
                QtCore.QSize(IMAGE_TAG_CLOSE_SIZE, IMAGE_TAG_CLOSE_SIZE),
            )
            self.assertEqual(chip.delete_button.y(), 0)
            self.assertEqual(
                chip.delete_button.geometry().right(), chip.width() - 1
            )

            for index in (1, 2, 3):
                button = self.widget._flow.itemAt(index).widget()
                self.assertEqual(
                    button.size(),
                    QtCore.QSize(IMAGE_TAG_HEIGHT, IMAGE_TAG_HEIGHT),
                )
                self.assertIn(
                    f"border-radius: {IMAGE_TAG_RADIUS}px",
                    button.styleSheet(),
                )

            self.widget.start_add()
            self.app.processEvents()
            self.assertEqual(self.widget._input.height(), IMAGE_TAG_HEIGHT)
            self.widget.cancel_input()
            self.widget.start_batch_mode()
            self.app.processEvents()
            for button in (
                self.widget._select_all_button,
                self.widget._delete_selected_button,
                self.widget._cancel_button,
            ):
                self.assertEqual(button.height(), IMAGE_TAG_HEIGHT)
        finally:
            init_theme(original_mode)
            self.app.setStyle(original_style)
            self.app.setPalette(original_palette)
            self.app.setStyleSheet(original_stylesheet)

    def test_flow_uses_equal_row_spacing_without_bottom_margin(self):
        self.widget.resize(90, 240)
        self.widget.set_tags(["one", "two", "three"])
        self.widget.show()
        self.app.processEvents()
        self.widget._refresh_height()
        self.app.processEvents()

        items = [
            self.widget._flow.itemAt(index).widget()
            for index in range(self.widget._flow.count())
        ]
        rows = sorted({item.geometry().top() for item in items})
        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(
            all(
                next_row - row == IMAGE_TAG_HEIGHT + IMAGE_TAG_SPACING
                for row, next_row in zip(rows, rows[1:])
            )
        )
        self.assertEqual(rows[0], 0)
        last_bottom = rows[-1] + IMAGE_TAG_HEIGHT
        self.assertEqual(
            self.widget._content.minimumHeight() - last_bottom,
            0,
        )


if __name__ == "__main__":
    unittest.main()
