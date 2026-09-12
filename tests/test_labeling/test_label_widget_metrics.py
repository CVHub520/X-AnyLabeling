import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for label widget metrics tests"
)
class TestLabelWidgetMetrics(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

    def test_measure_text_width_matches_horizontal_advance(self):
        from anylabeling.views.labeling.label_widget import _measure_text_width

        font = self.app.font()
        metrics = QtGui.QFontMetrics(font)

        self.assertEqual(
            _measure_text_width(metrics, "bodyColor"),
            metrics.horizontalAdvance("bodyColor"),
        )

    def test_measure_text_width_falls_back_to_width(self):
        from anylabeling.views.labeling.label_widget import _measure_text_width

        class LegacyFontMetrics:
            def width(self, text):
                return len(text) * 7

        metrics = LegacyFontMetrics()
        self.assertEqual(_measure_text_width(metrics, "vehicle"), 49)

    def test_format_label_list_text_includes_group_id(self):
        from anylabeling.views.labeling.label_widget import (
            _format_label_list_text,
        )

        self.assertEqual(
            _format_label_list_text("towel_clamp", 1), "towel_clamp (1)"
        )
        self.assertEqual(
            _format_label_list_text("a<b", None),
            "a&lt;b",
        )

    def test_locked_item_uses_svg_icon(self):
        from anylabeling.resources import resources  # noqa: F401
        from anylabeling.views.labeling.label_widget import (
            _set_label_list_item_lock,
        )
        from anylabeling.views.labeling.widgets import LabelListWidgetItem

        item = LabelListWidgetItem("vehicle")

        _set_label_list_item_lock(item, True)
        self.assertTrue(item.is_locked())

        _set_label_list_item_lock(item, False)
        self.assertFalse(item.is_locked())

    def test_selected_object_keeps_label_background_in_each_mode(self):
        from anylabeling.resources import resources  # noqa: F401
        from anylabeling.views.labeling.utils.style import get_dock_style
        from anylabeling.views.labeling.utils.theme import (
            get_theme,
            init_theme,
        )
        from anylabeling.views.labeling.widgets import (
            LabelListWidget,
            LabelListWidgetItem,
        )

        for mode in ("light", "dark"):
            init_theme(mode)
            widget = LabelListWidget()
            widget.setStyleSheet(get_dock_style())
            item = LabelListWidgetItem("vase")
            item.setBackground(QtGui.QColor("#FF8A8A"))
            widget.add_iem(item)
            delegate = widget.itemDelegate()

            def render(check_state):
                item.setCheckState(check_state)
                image = QtGui.QImage(
                    200, 24, QtGui.QImage.Format.Format_ARGB32
                )
                image.fill(QtGui.QColor("#FFFFFF"))
                painter = QtGui.QPainter(image)
                option = QtWidgets.QStyleOptionViewItem()
                option.rect = QtCore.QRect(0, 0, 200, 24)
                option.state = (
                    QtWidgets.QStyle.StateFlag.State_Enabled
                    | QtWidgets.QStyle.StateFlag.State_Active
                    | QtWidgets.QStyle.StateFlag.State_Selected
                )
                option.palette = widget.palette()
                option.font = widget.font()
                option.widget = widget
                delegate.paint(painter, option, item.index())
                painter.end()
                return image

            visible = render(QtCore.Qt.CheckState.Checked)
            hidden = render(QtCore.Qt.CheckState.Unchecked)
            primary = QtGui.QColor(get_theme()["primary"])

            for image in (visible, hidden):
                for y in (0, 12, 23):
                    self.assertEqual(
                        image.pixelColor(195, y), QtGui.QColor("#FF8A8A")
                    )
                self.assertEqual(image.pixelColor(7, 8), primary)
                if mode == "dark":
                    self.assertEqual(image.pixelColor(3, 5), primary)
                else:
                    self.assertNotEqual(image.pixelColor(3, 5), primary)
            self.assertNotEqual(visible.pixelColor(10, 12), primary)
            self.assertEqual(hidden.pixelColor(10, 12), primary)
            widget.close()
        init_theme("light")

    def test_file_selection_uses_hover_colors_in_each_mode(self):
        from anylabeling.views.labeling.utils.style import get_dock_style
        from anylabeling.views.labeling.utils.theme import init_theme

        init_theme("light")
        style = get_dock_style()
        self.assertIn("background-color: #e5e5e5", style)
        self.assertIn("color: #1d1d1f", style)

        init_theme("dark")
        dark_style = get_dock_style()
        self.assertIn("background-color: #3a3a3c", dark_style)
        self.assertIn("color: #f5f5f7", dark_style)
        init_theme("light")

    def test_unchecked_review_status_icon_is_hollow_circle(self):
        from anylabeling.views.labeling.label_widget import (
            FILE_CHECKED_COLOR,
            FILE_UNCHECKED_COLOR,
            _create_file_status_icon,
        )

        checked = _create_file_status_icon(FILE_CHECKED_COLOR)
        unchecked = _create_file_status_icon(
            FILE_UNCHECKED_COLOR, filled=False
        )
        checked_image = checked.pixmap(12, 12).toImage()
        unchecked_image = unchecked.pixmap(12, 12).toImage()

        self.assertGreater(checked_image.pixelColor(6, 6).alpha(), 0)
        self.assertEqual(unchecked_image.pixelColor(6, 6).alpha(), 0)
        self.assertGreater(unchecked_image.pixelColor(6, 2).alpha(), 0)

    def test_window_title_includes_annotation_checked_status(self):
        from anylabeling.views.labeling.label_widget import LabelingWidget

        widget = SimpleNamespace(
            filename="/tmp/image.jpg",
            dirty=False,
            image=SimpleNamespace(
                isNull=lambda: False,
                width=lambda: 640,
                height=lambda: 480,
            ),
            get_image_progress_info=lambda: (2, 10),
            tr=lambda text: text,
        )

        for checked, status in ((True, "Checked"), (False, "Unchecked")):
            widget._annotation_checked = lambda value=checked: value
            title = LabelingWidget._window_title(widget)
            self.assertEqual(
                title,
                f"X-AnyLabeling - image.jpg [{status}] [640x480] [2/10]",
            )

    def test_checked_state_sync_refreshes_window_title(self):
        from anylabeling.views.labeling.label_widget import LabelingWidget

        calls = []
        widget = SimpleNamespace(
            _update_annotation_checked_action=lambda: calls.append("action"),
            _update_current_file_checked_item=lambda: calls.append("item"),
            update_progress_title=lambda: calls.append("title"),
        )

        LabelingWidget._sync_annotation_checked_state(widget)

        self.assertEqual(calls, ["action", "item", "title"])

    def test_right_double_click_does_not_emit_item_double_clicked(self):
        from anylabeling.views.labeling.widgets import (
            LabelListWidget,
            LabelListWidgetItem,
        )

        widget = LabelListWidget()
        widget.resize(200, 100)
        widget.add_iem(LabelListWidgetItem("vehicle"))
        widget.show()
        self.app.processEvents()
        emitted = []
        widget.item_double_clicked.connect(emitted.append)
        index = widget.model().index(0, 0)
        position = widget.visualRect(index).center()

        QtTest.QTest.mouseDClick(
            widget.viewport(),
            QtCore.Qt.MouseButton.RightButton,
            pos=position,
        )
        self.app.processEvents()

        self.assertEqual(emitted, [])

        QtTest.QTest.mouseDClick(
            widget.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            pos=position,
        )
        self.app.processEvents()

        self.assertEqual(len(emitted), 1)
        widget.close()

    def test_right_click_requests_lock_for_selected_items(self):
        from anylabeling.views.labeling.widgets import (
            LabelListWidget,
            LabelListWidgetItem,
        )

        widget = LabelListWidget()
        widget.resize(200, 100)
        first_item = LabelListWidgetItem("vehicle")
        second_item = LabelListWidgetItem("person")
        widget.add_iem(first_item)
        widget.add_iem(second_item)
        widget.select_item(first_item)
        widget.select_item(second_item)
        widget.show()
        self.app.processEvents()
        emitted = []
        widget.items_lock_requested.connect(emitted.append)
        index = widget.model().index(0, 0)
        position = widget.visualRect(index).center()

        QtTest.QTest.mouseClick(
            widget.viewport(),
            QtCore.Qt.MouseButton.RightButton,
            pos=position,
        )
        self.app.processEvents()

        self.assertEqual(emitted, [[first_item, second_item]])
        self.assertEqual(widget.selected_items(), [first_item, second_item])
        widget.close()

    def test_canvas_lock_action_uses_checkbox_without_icon(self):
        from anylabeling.views.labeling.label_widget import LabelingWidget

        shape = SimpleNamespace(locked=True)
        item = SimpleNamespace(shape=lambda: shape)
        action = QtGui.QAction("Lock Shape")
        action.setCheckable(True)
        widget = SimpleNamespace(
            label_list=SimpleNamespace(selected_items=lambda: [item]),
            actions=SimpleNamespace(toggle_shape_lock=action),
        )

        LabelingWidget.refresh_shape_lock_action(widget)

        self.assertTrue(action.isChecked())
        self.assertEqual(action.text(), "Lock Shape")
        self.assertTrue(action.icon().isNull())
