import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore

    from anylabeling.services.auto_labeling.types import (
        AutoLabelingMode,
        AutoLabelingResult,
    )
    from anylabeling.views.labeling.label_widget import LabelingWidget
    from anylabeling.views.labeling.shape import Shape

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

    def make_model_edge_receiver_widget(self):
        shape = Shape(label="model", shape_type="polygon")
        shape.points = [
            QtCore.QPointF(1, 1),
            QtCore.QPointF(5, 1),
            QtCore.QPointF(5, 5),
            QtCore.QPointF(1, 5),
        ]
        shape.close()
        candidate = np.asarray(
            [[1.5, 1.5], [5.5, 1.5], [5.5, 5.5], [1.5, 5.5]]
        )
        result = AutoLabelingResult(
            [shape], replace=False, image_path="image.png"
        )
        panel = SimpleNamespace(set_result_status=Mock())
        widget = SimpleNamespace(
            _edge_tasks=set(),
            _model_edge_result_requests={7: (result, [shape])},
            filename="image.png",
            pixel_edge_widget=panel,
            tr=lambda text: text,
            _edge_refinement_settings=lambda: {"auto_label_enabled": True},
            _set_shape_edge_points=LabelingWidget._set_shape_edge_points,
            new_shapes_from_auto_labeling=Mock(),
        )
        return widget, result, shape, candidate

    def test_model_edge_postprocess_happens_before_original_receiver(self):
        widget, result, shape, candidate = (
            self.make_model_edge_receiver_widget()
        )

        LabelingWidget._on_auto_label_edge_postprocess_ready(
            widget, 7, [candidate], None
        )

        np.testing.assert_allclose(
            [[point.x(), point.y()] for point in shape.points], candidate + 0.5
        )
        self.assertEqual(
            shape.other_data["pixel_edge_source"], "auto_label_pre_display"
        )
        widget.new_shapes_from_auto_labeling.assert_called_once_with(result)
        self.assertNotIn(7, widget._model_edge_result_requests)

    def test_model_edge_postprocess_failure_forwards_original_result(self):
        widget, result, shape, _candidate = (
            self.make_model_edge_receiver_widget()
        )
        original = [[point.x(), point.y()] for point in shape.points]

        LabelingWidget._on_auto_label_edge_postprocess_ready(
            widget, 7, None, RuntimeError("edge worker failed")
        )

        self.assertEqual(
            [[point.x(), point.y()] for point in shape.points], original
        )
        self.assertNotIn("pixel_edge_source", shape.other_data)
        widget.new_shapes_from_auto_labeling.assert_called_once_with(result)

    def make_existing_edge_preview_widget(self):
        shape = Shape(label="existing", shape_type="polygon")
        shape.points = [
            QtCore.QPointF(10.0, 10.0),
            QtCore.QPointF(20.0, 10.0),
            QtCore.QPointF(20.0, 20.0),
            QtCore.QPointF(10.0, 20.0),
        ]
        shape.close()
        original = [QtCore.QPointF(point) for point in shape.points]
        undo = Mock()
        undo.isEnabled.return_value = True
        canvas = SimpleNamespace(
            shapes=[shape],
            shapes_backups=[[shape.copy()]],
            update=Mock(),
            clear_edge_preview_shapes=Mock(),
            select_shapes=Mock(),
            store_shapes=Mock(),
            shape_moved=SimpleNamespace(emit=Mock()),
            is_shape_restorable=True,
        )
        panel = SimpleNamespace(
            set_pending=Mock(), set_result_status=Mock(), set_candidates=Mock()
        )
        widget = SimpleNamespace(
            _edge_tasks=set(),
            _edge_request_id=11,
            filename="image.png",
            _edge_existing_request={
                "request_id": 11,
                "shape": shape,
                "original_points": original,
                "original_other_data": {},
                "backups": [[shape.copy()]],
                "undo_enabled": True,
            },
            _edge_existing_preview=None,
            _edge_preview_timer=Mock(),
            _edge_box_mode=False,
            _edge_pending_box=None,
            _edge_pending_candidate=None,
            canvas=canvas,
            pixel_edge_widget=panel,
            actions=SimpleNamespace(
                undo=undo,
                delete=Mock(),
                duplicate=Mock(),
                edit=Mock(),
                edit_brush_mode=Mock(),
                union_selection=Mock(),
            ),
            tr=lambda text: text,
            status=Mock(),
            toggle_draw_mode=Mock(),
            _copy_shape_backups=LabelingWidget._copy_shape_backups,
            _shape_points_array=LabelingWidget._shape_points_array,
            _set_shape_edge_points=LabelingWidget._set_shape_edge_points,
            _edge_refinement_settings=lambda: {},
            image=object(),
        )
        return widget, shape, original

    def test_existing_annotation_uses_editable_preview_before_commit(self):
        widget, shape, original = self.make_existing_edge_preview_widget()
        candidate = np.asarray(
            [[10.5, 10.5], [20.5, 10.5], [20.5, 20.5], [10.5, 20.5]]
        )
        result = SimpleNamespace(
            succeeded=True,
            points=candidate,
            fit_error=0.25,
            reason="",
        )

        ready = LabelingWidget._on_existing_edge_preview_ready(
            widget, 11, result, None
        )

        self.assertTrue(ready)
        np.testing.assert_allclose(
            [
                [point.x(), point.y()]
                for point in widget.canvas.shapes[0].points
            ],
            candidate + 0.5,
        )
        self.assertNotIn("pixel_edge_refined", shape.other_data)
        self.assertIsNotNone(widget._edge_existing_preview)
        widget.pixel_edge_widget.set_pending.assert_called_with(True)
        widget.canvas.store_shapes.assert_not_called()
        self.assertEqual(shape.points, original)

    def test_existing_annotation_cancel_restores_original_after_manual_edit(
        self,
    ):
        widget, shape, original = self.make_existing_edge_preview_widget()
        widget._edge_existing_preview = widget._edge_existing_request
        widget._edge_existing_request = None
        shape.points[0] = QtCore.QPointF(13.25, 14.75)

        LabelingWidget.cancel_pixel_edge_preview(widget, switch_mode=False)

        self.assertEqual(shape.points, original)
        self.assertIsNone(widget._edge_existing_preview)
        widget.actions.undo.setEnabled.assert_called_with(True)

    def test_existing_annotation_confirm_validates_then_commits_once(self):
        widget, shape, _original = self.make_existing_edge_preview_widget()
        widget._edge_existing_preview = widget._edge_existing_request
        widget._edge_existing_request = None
        validated = SimpleNamespace(
            succeeded=True,
            points=np.asarray(
                [[10.5, 10.5], [20.5, 10.5], [20.5, 20.5], [10.5, 20.5]]
            ),
            fit_error=0.2,
            reason="",
        )

        with patch(
            "anylabeling.views.labeling.label_widget.validate_polygon_edge_fit",
            return_value=validated,
        ):
            confirmed = LabelingWidget.confirm_pixel_edge_preview(widget)

        self.assertTrue(confirmed)
        self.assertIsNone(widget._edge_existing_preview)
        self.assertEqual(
            shape.other_data["pixel_edge_source"],
            "double_click_confirmed",
        )
        widget.canvas.store_shapes.assert_called_once()
        widget.canvas.shape_moved.emit.assert_called_once()

    def test_existing_annotation_rejects_manual_point_outside_tolerance(self):
        widget, _shape, _original = self.make_existing_edge_preview_widget()
        widget._edge_existing_preview = widget._edge_existing_request
        widget._edge_existing_request = None
        rejected = SimpleNamespace(
            succeeded=False,
            points=None,
            fit_error=0.75,
            reason="One vertex is outside tolerance",
        )

        with patch(
            "anylabeling.views.labeling.label_widget.validate_polygon_edge_fit",
            return_value=rejected,
        ):
            confirmed = LabelingWidget.confirm_pixel_edge_preview(widget)

        self.assertFalse(confirmed)
        self.assertIsNotNone(widget._edge_existing_preview)
        widget.canvas.store_shapes.assert_not_called()
