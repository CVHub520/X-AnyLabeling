import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
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
            ),
            label_list=Mock(),
            actions=SimpleNamespace(**{name: Mock() for name in action_names}),
            _no_selection_slot=False,
            attributes=None,
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
        polygon = SimpleNamespace(shape_type="polygon", selected=False)
        widget.label_list.find_item_by_shape.return_value = None

        LabelingWidget.shape_selection_changed(widget, [polygon])

        widget.actions.edit_brush_mode.setEnabled.assert_called_once_with(True)
