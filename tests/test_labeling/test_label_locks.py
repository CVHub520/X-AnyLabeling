import os
from types import SimpleNamespace
import unittest
from functools import partial
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtTest import QTest

import anylabeling.resources.resources  # noqa: F401

from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets.label_list_widget import LOCKED_ROLE
from anylabeling.views.labeling.widgets.panel import PanelHeader
from anylabeling.views.labeling.widgets.unique_label_qlist_widget import (
    UniqueLabelQListWidget,
)


class TestLabelLocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )

    def setUp(self):
        self.labels = UniqueLabelQListWidget()
        for label in ("cat", "dog"):
            item = self.labels.create_item_from_label(label)
            self.labels.addItem(item)
            self.labels.set_item_label(item, label)
        self.shapes = [Shape(label=label) for label in ("cat", "cat", "dog")]
        self.widget = SimpleNamespace(
            unique_label_list=self.labels,
            canvas=SimpleNamespace(shapes=self.shapes),
            _update_shapes_lock=Mock(),
            refresh_panel_controls=Mock(),
            _session_shape_locks={},
        )
        self.widget.toggle_unique_label_items_lock = partial(
            LabelingWidget.toggle_unique_label_items_lock, self.widget
        )
        self.widget._set_shapes_locked = partial(
            LabelingWidget._set_shapes_locked, self.widget
        )
        self.labels.items_lock_requested.connect(
            lambda items: LabelingWidget.toggle_unique_label_items_lock(
                self.widget, items
            )
        )

    def tearDown(self):
        self.labels.close()

    def test_right_click_toggles_whole_category(self):
        self.labels.show()
        self.app.processEvents()
        position = self.labels.visualItemRect(self.labels.item(0)).center()
        for expected in (True, False):
            QTest.mouseClick(
                self.labels.viewport(),
                QtCore.Qt.MouseButton.RightButton,
                pos=position,
            )
            self.assertEqual(
                [s.locked for s in self.shapes], [expected, expected, False]
            )
            self.assertEqual(self.labels.lock_states["cat"], expected)
            self.assertEqual(
                self.labels.item(0).data(LOCKED_ROLE),
                expected,
            )

    def test_bulk_inverts_mixed_categories(self):
        LabelingWidget.toggle_unique_label_items_lock(
            self.widget, [self.labels.item(0)]
        )
        LabelingWidget.toggle_unique_label_items_lock(
            self.widget, [self.labels.item(0), self.labels.item(1)]
        )
        self.assertEqual([s.locked for s in self.shapes], [False, False, True])

    def test_empty_category_keeps_policy_when_list_is_rebuilt(self):
        self.widget.canvas.shapes = []
        LabelingWidget.toggle_unique_label_items_lock(
            self.widget, [self.labels.item(0)]
        )
        self.widget._update_shapes_lock.assert_not_called()
        self.labels.clear()
        item = self.labels.create_item_from_label("cat")
        self.labels.addItem(item)
        self.labels.set_item_label(item, "cat")
        self.assertTrue(self.labels.lock_states["cat"])
        self.assertTrue(item.data(LOCKED_ROLE))

    def test_bulk_labels_replaces_previous_individual_lock(self):
        self.widget.toggle_unique_label_items_lock([self.labels.item(0)])
        for expected in (True, False):
            LabelingWidget.toggle_all_labels_lock(self.widget)
            self.assertEqual([s.locked for s in self.shapes], [expected] * 3)
            self.assertEqual(
                self.labels.lock_states, {"cat": expected, "dog": expected}
            )

    def test_bulk_shapes_clears_previous_category_policy(self):
        self.widget.toggle_unique_label_items_lock([self.labels.item(0)])
        for expected in (True, False):
            LabelingWidget.toggle_all_shapes_lock(self.widget)
            self.assertEqual([s.locked for s in self.shapes], [expected] * 3)
            self.assertEqual(self.labels.lock_states, {})
            self.assertFalse(self.labels.item(0).data(LOCKED_ROLE))

    def test_header_title_is_centered_with_compact_controls(self):
        for count in (1, 2):
            title = QtWidgets.QLabel("Image Description")
            controls = [QtWidgets.QToolButton() for _ in range(count)]
            header = PanelHeader(title, controls)
            header.resize(280, 26)
            header.show()
            self.app.processEvents()
            self.assertLessEqual(
                abs(
                    title.geometry().center().x() - header.rect().center().x()
                ),
                1,
            )
            if count == 2:
                self.assertEqual(
                    controls[1].x() - controls[0].geometry().right(), 3
                )
            header.close()

    def test_category_checkbox_controls_all_matching_shapes(self):
        from anylabeling.views.labeling.widgets.label_list_widget import (
            LabelListWidget,
            LabelListWidgetItem,
        )

        objects = LabelListWidget()
        for shape in self.shapes:
            objects.add_iem(LabelListWidgetItem(shape.label, shape))
        self.widget.label_list = objects
        self.widget.label_info = {}
        self.widget.canvas.visible = {}
        self.widget.canvas.update = Mock()
        self.widget._get_rgb_by_label = Mock(return_value=(0, 255, 0))
        self.widget._update_select_toggle_button_tooltip = Mock()
        self.widget._sync_label_list_visibility = partial(
            LabelingWidget._sync_label_list_visibility, self.widget
        )
        self.widget._refresh_shape_filters = partial(
            LabelingWidget.apply_label_visibility, self.widget
        )
        self.labels.label_visibility_changed.connect(
            partial(LabelingWidget.set_label_visibility, self.widget)
        )
        self.labels.show()
        self.app.processEvents()
        item = self.labels.item(0)
        option = QtWidgets.QStyleOptionViewItem()
        option.initFrom(self.labels)
        option.rect = self.labels.visualItemRect(item)
        self.labels.itemDelegate().initStyleOption(
            option, self.labels.indexFromItem(item)
        )
        rect = self.labels.style().subElementRect(
            QtWidgets.QStyle.SubElement.SE_ItemViewItemCheckIndicator,
            option,
            self.labels,
        )
        for visible in (False, True):
            QTest.mouseClick(
                self.labels.viewport(),
                QtCore.Qt.MouseButton.LeftButton,
                pos=rect.center(),
            )
            self.assertEqual(
                [s.visible for s in self.shapes], [visible, visible, True]
            )
            self.assertEqual(self.widget.label_info["cat"]["visible"], visible)
            self.assertEqual(
                [
                    i.checkState() == QtCore.Qt.CheckState.Checked
                    for i in objects
                ],
                [visible, visible, True],
            )
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.labels.set_item_locked(item, True)
        self.assertFalse(self.widget.label_info["cat"]["visible"])
        self.labels.clear()
        restored = self.labels.create_item_from_label("cat")
        self.labels.addItem(restored)
        self.widget._refresh_shape_filters()
        self.assertEqual(restored.checkState(), QtCore.Qt.CheckState.Unchecked)
        objects.close()

    def test_lock_and_color_changes_do_not_emit_visibility(self):
        changed = Mock()
        self.labels.label_visibility_changed.connect(changed)
        item = self.labels.item(0)
        self.labels.set_item_locked(item, True)
        self.labels.update_item_color("cat", (255, 0, 0))
        changed.assert_not_called()

    def test_header_visibility_toggles_all_categories(self):
        self.widget.label_info = {
            "cat": {"visible": False},
            "dog": {"visible": True},
        }
        self.widget._refresh_shape_filters = Mock()
        for visible in (False, True):
            LabelingWidget.toggle_all_labels_visibility(self.widget)
            self.assertEqual(
                [info["visible"] for info in self.widget.label_info.values()],
                [visible, visible],
            )

    def test_header_controls_reflect_current_state(self):
        for name in (
            "labels_visibility_button",
            "shapes_visibility_button",
            "labels_lock_button",
            "shapes_lock_button",
        ):
            button = QtWidgets.QToolButton()
            button.setCheckable(True)
            setattr(self.widget, name, button)
        self.widget.label_info = {}
        self.widget.toggle_unique_label_items_lock(
            [self.labels.item(0), self.labels.item(1)]
        )
        LabelingWidget.refresh_panel_controls(self.widget)
        self.assertTrue(self.widget.labels_lock_button.isChecked())
        self.assertTrue(self.widget.shapes_lock_button.isChecked())
        self.assertFalse(self.widget.labels_visibility_button.icon().isNull())
        self.shapes[0].locked = False
        LabelingWidget.refresh_panel_controls(self.widget)
        self.assertFalse(self.widget.shapes_lock_button.isChecked())

    def test_object_locks_survive_image_switch_in_memory(self):
        from anylabeling.views.labeling.widgets.label_list_widget import (
            LabelListWidget,
            LabelListWidgetItem,
        )

        self.widget.image_path = "/tmp/session-image.png"
        self.widget._shape_lock_key = LabelingWidget._shape_lock_key
        self.shapes[0].locked = True
        LabelingWidget._remember_shape_locks(self.widget)
        restored = [
            Shape().load_from_dict(shape.to_dict()) for shape in self.shapes
        ]
        objects = LabelListWidget()
        for shape in restored:
            objects.add_iem(LabelListWidgetItem(shape.label, shape))
        self.widget.canvas.shapes = restored
        self.widget.label_list = objects
        LabelingWidget._restore_shape_locks(self.widget)
        self.assertEqual([s.locked for s in restored], [True, False, False])
        self.widget._session_shape_locks.clear()
        for shape in restored:
            shape.locked = False
        LabelingWidget._restore_shape_locks(self.widget)
        self.assertFalse(any(s.locked for s in restored))
        objects.close()

    def test_locking_does_not_mark_annotations_dirty(self):
        self.widget.label_list = Mock()
        self.widget.canvas.store_shapes = Mock()
        self.widget.canvas.update = Mock()
        self.widget.canvas.editing = Mock(return_value=False)
        self.widget.refresh_shape_lock_action = Mock()
        self.widget.set_dirty = Mock()
        LabelingWidget._update_shapes_lock(self.widget, self.shapes)
        self.widget.set_dirty.assert_not_called()
