import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from anylabeling.services.auto_labeling.types import AutoLabelingMode
    from anylabeling.views.labeling.label_widget import LabelingWidget

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for label widget brush mode tests"
)
class TestLabelWidgetBrushMode(unittest.TestCase):

    def make_selection_widget(self):
        action_names = [
            "delete",
            "duplicate",
            "copy",
            "edit",
            "copy_coordinates",
            "edit_brush_mode",
            "union_selection",
        ]
        return SimpleNamespace(
            canvas=SimpleNamespace(
                is_brush_mode=False,
                _brush_target_shape=None,
                selected_shapes=[],
                _active_group_shapes=Mock(return_value=[]),
            ),
            label_list=Mock(),
            actions=SimpleNamespace(**{name: Mock() for name in action_names}),
            _no_selection_slot=False,
            attributes=None,
            refresh_shape_lock_action=Mock(),
            set_text_editing=Mock(),
            hide_attributes_panel=Mock(),
        )

    def test_active_brush_mode_disables_shape_list(self):
        brush_action = Mock()
        label_list = Mock()
        widget = SimpleNamespace(
            actions=SimpleNamespace(edit_brush_mode=brush_action),
            label_list=label_list,
        )

        LabelingWidget.on_brush_mode_changed(widget, True)

        brush_action.setChecked.assert_called_once_with(True)
        label_list.setEnabled.assert_called_once_with(False)

    def test_active_brush_mode_rejects_selection_change(self):
        target = object()
        target_item = object()
        label_list = Mock()
        label_list.find_item_by_shape.return_value = target_item
        widget = SimpleNamespace(
            canvas=SimpleNamespace(
                is_brush_mode=True,
                _brush_target_shape=target,
                selected_shapes=[target],
            ),
            label_list=label_list,
            _no_selection_slot=False,
        )

        LabelingWidget.shape_selection_changed(widget, [object()])

        self.assertEqual(widget.canvas.selected_shapes, [target])
        self.assertFalse(widget._no_selection_slot)
        label_list.clearSelection.assert_called_once()
        label_list.select_item.assert_called_once_with(target_item)
        label_list.scroll_to_item.assert_called_once_with(target_item)

    def test_brush_action_is_disabled_without_selection(self):
        widget = self.make_selection_widget()

        LabelingWidget.shape_selection_changed(widget, [])

        widget.actions.edit_brush_mode.setEnabled.assert_called_once_with(
            False
        )

    def test_brush_action_is_enabled_for_one_polygon(self):
        widget = self.make_selection_widget()
        polygon = SimpleNamespace(
            shape_type="polygon", selected=False, locked=False
        )
        widget.label_list.find_item_by_shape.return_value = None

        LabelingWidget.shape_selection_changed(widget, [polygon])

        widget.actions.edit_brush_mode.setEnabled.assert_called_once_with(True)

    def test_locked_polygon_disables_destructive_actions(self):
        widget = self.make_selection_widget()
        polygon = SimpleNamespace(
            shape_type="polygon", selected=False, locked=True
        )
        widget.label_list.find_item_by_shape.return_value = None

        LabelingWidget.shape_selection_changed(widget, [polygon])

        widget.actions.delete.setEnabled.assert_called_once_with(False)
        widget.actions.edit_brush_mode.setEnabled.assert_called_once_with(
            False
        )
        widget.actions.union_selection.setEnabled.assert_called_once_with(
            False
        )

    def test_item_lock_request_inverts_each_selected_shape(self):
        unlocked_shape = SimpleNamespace(locked=False)
        locked_shape = SimpleNamespace(locked=True)
        items = [
            SimpleNamespace(shape=lambda: unlocked_shape),
            SimpleNamespace(shape=lambda: locked_shape),
        ]
        widget = SimpleNamespace(_update_shapes_lock=Mock())

        LabelingWidget.toggle_label_items_lock(widget, items)

        self.assertTrue(unlocked_shape.locked)
        self.assertFalse(locked_shape.locked)
        widget._update_shapes_lock.assert_called_once_with(
            [unlocked_shape, locked_shape]
        )

    def test_magic_wand_mode_uses_polygon_creation(self):
        canvas = SimpleNamespace(
            drawing=Mock(return_value=False),
            is_magic_wand_mode=False,
            set_magic_wand_mode=Mock(),
        )
        widget = SimpleNamespace(
            canvas=canvas,
            toggle_draw_mode=Mock(),
            actions=SimpleNamespace(
                create_mode=Mock(),
                create_brush_polygon_mode=Mock(),
                create_magic_wand_mode=Mock(),
            ),
        )

        LabelingWidget.toggle_magic_wand_mode(widget)

        widget.toggle_draw_mode.assert_called_once_with(
            False, create_mode="polygon"
        )
        canvas.set_magic_wand_mode.assert_called_once_with(True)
        widget.actions.create_mode.setEnabled.assert_called_once_with(True)
        widget.actions.create_magic_wand_mode.setEnabled.assert_called_once_with(
            False
        )

    def test_draw_mode_disables_active_action(self):
        action_names = [
            "create_mode",
            "create_brush_polygon_mode",
            "create_magic_wand_mode",
            "create_rectangle_mode",
            "create_cuboid_mode",
            "create_rotation_mode",
            "create_quadrilateral_mode",
            "create_circle_mode",
            "create_line_mode",
            "create_point_mode",
            "create_line_strip_mode",
            "edit_mode",
            "edit_brush_mode",
            "union_selection",
        ]
        actions = SimpleNamespace(**{name: Mock() for name in action_names})
        actions.edit_brush_mode.isChecked.return_value = False
        canvas = SimpleNamespace(
            is_brush_mode=False,
            set_magic_wand_mode=Mock(),
            set_editing=Mock(),
            create_mode="polygon",
            _brush_drawing=False,
        )
        widget = SimpleNamespace(
            canvas=canvas,
            actions=actions,
            auto_labeling_widget=SimpleNamespace(
                auto_labeling_mode=AutoLabelingMode.NONE
            ),
            set_text_editing=Mock(),
            hide_attributes_panel=Mock(),
            update_labeling_instruction=Mock(),
        )

        LabelingWidget.toggle_draw_mode(
            widget, edit=False, create_mode="rectangle"
        )

        self.assertEqual(
            actions.create_rectangle_mode.setEnabled.call_count, 2
        )
        actions.create_rectangle_mode.setEnabled.assert_called_with(False)
        actions.edit_mode.setEnabled.assert_called_once_with(True)
