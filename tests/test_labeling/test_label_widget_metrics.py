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
