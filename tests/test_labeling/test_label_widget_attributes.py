import gc
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtWidgets

    from anylabeling.views.labeling.label_widget import LabelingWidget
    from anylabeling.views.labeling.shape import Shape

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for label widget attribute tests"
)
class TestLabelWidgetAttributes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def test_reset_attribute_uses_first_option(self):
        widget = SimpleNamespace(
            attributes={"car": {"visibility": ["low", "high"]}}
        )
        shape = SimpleNamespace(attributes={})

        text = LabelingWidget.reset_attribute(widget, "car", shape)

        self.assertEqual(text, "car")
        self.assertEqual(shape.attributes, {"visibility": "low"})

    def test_reset_attribute_preserves_scalar_value(self):
        widget = SimpleNamespace(
            attributes={"car": {"vehicle_id": "vehicle-001"}}
        )
        shape = SimpleNamespace(attributes={})

        LabelingWidget.reset_attribute(widget, "car", shape)

        self.assertEqual(shape.attributes, {"vehicle_id": "vehicle-001"})

    def test_reset_attribute_preserves_empty_values(self):
        widget = SimpleNamespace(
            attributes={"car": {"vehicle_id": "", "occluded_by": []}}
        )
        shape = SimpleNamespace(attributes={})

        LabelingWidget.reset_attribute(widget, "car", shape)

        self.assertEqual(
            shape.attributes, {"vehicle_id": "", "occluded_by": []}
        )

    @patch("anylabeling.views.labeling.label_widget.LabelFile")
    def test_save_attributes_uses_standard_shape_serialization(
        self, label_file_class
    ):
        shape = Shape(
            label="car",
            score=0.91,
            shape_type="rotation",
            direction=0.25,
            attributes={"color": "red"},
        )
        shape.points = [QtCore.QPointF(1, 2), QtCore.QPointF(3, 4)]
        shape.locked = True
        shape.other_data = {"custom": "value"}
        widget = SimpleNamespace(
            image_path="/tmp/image.jpg",
            output_dir=None,
            flag_widget=SimpleNamespace(count=Mock(return_value=0)),
            other_data={},
            _annotation_checked=Mock(return_value=False),
            _config={"store_data": False},
            image_data=None,
            image=SimpleNamespace(
                height=Mock(return_value=100),
                width=Mock(return_value=200),
            ),
            file_list_widget=SimpleNamespace(findItems=Mock(return_value=[])),
            error_message=Mock(),
        )

        result = LabelingWidget.save_attributes(widget, [shape])

        self.assertTrue(result)
        saved_shape = label_file_class.return_value.save.call_args.kwargs[
            "shapes"
        ][0]
        self.assertEqual(saved_shape["score"], 0.91)
        self.assertTrue(saved_shape["locked"])
        self.assertEqual(saved_shape["direction"], 0.25)
        self.assertEqual(saved_shape["custom"], "value")

    def test_update_attributes_does_not_save_unchanged_shape(self):
        shape = SimpleNamespace(
            label="car", attributes={"color": "red"}
        )
        widget = SimpleNamespace(
            canvas=SimpleNamespace(shapes=[shape]),
            attributes={"car": {"color": ["red", "blue"]}},
            attribute_widget_types={},
            scroll_area=QtWidgets.QScrollArea(),
            save_attributes=Mock(),
            show_attributes_panel=Mock(),
            hide_attributes_panel=Mock(),
        )

        LabelingWidget.update_attributes(widget, 0)

        widget.save_attributes.assert_not_called()

    def test_edit_label_preserves_attributes_for_same_label(self):
        shape = SimpleNamespace(
            label="car",
            flags={},
            group_id=None,
            description="old",
            difficult=False,
            kie_linking=[],
            attributes={"color": "blue"},
            fill_color=SimpleNamespace(
                getRgb=Mock(return_value=(255, 0, 0, 255))
            ),
        )
        item = SimpleNamespace(
            shape=Mock(return_value=shape),
            setText=Mock(),
            setBackground=Mock(),
        )
        widget = SimpleNamespace(
            canvas=SimpleNamespace(
                editing=Mock(return_value=True),
                selected_shapes=[shape],
                shapes=[shape],
            ),
            current_item=Mock(return_value=item),
            label_dialog=SimpleNamespace(
                pop_up=Mock(
                    return_value=("car", {}, None, "new", False, [])
                ),
                add_label_history=Mock(),
            ),
            _config={"move_mode": "auto"},
            validate_label=Mock(return_value=True),
            attributes={"car": {"color": ["red", "blue"]}},
            reset_attribute=Mock(),
            unique_label_list=SimpleNamespace(
                find_items_by_label=Mock(return_value=[object()])
            ),
            _update_shape_color=Mock(),
            set_dirty=Mock(),
            _refresh_shape_filters=Mock(),
            update_attributes=Mock(),
        )

        LabelingWidget.edit_label(widget)

        self.assertEqual(shape.attributes, {"color": "blue"})
        widget.reset_attribute.assert_not_called()

    def test_batch_edit_resets_only_shapes_with_changed_labels(self):
        unchanged_shape = SimpleNamespace(
            label="car",
            flags={},
            group_id=None,
            description="",
            difficult=False,
            kie_linking=[],
            attributes={"color": "blue"},
        )
        changed_shape = SimpleNamespace(
            label="person",
            flags={},
            group_id=None,
            description="",
            difficult=False,
            kie_linking=[],
            attributes={"age": "adult"},
        )

        def reset_attribute(text, shape):
            shape.attributes = {"color": "red"}
            return text

        widget = SimpleNamespace(
            _batch_edit_warning_shown=True,
            label_dialog=SimpleNamespace(
                pop_up=Mock(
                    return_value=("car", {}, None, "new", False, [])
                ),
                add_label_history=Mock(),
            ),
            validate_label=Mock(return_value=True),
            attributes={"car": {"color": ["red", "blue"]}},
            reset_attribute=Mock(side_effect=reset_attribute),
            _update_shape_color=Mock(),
            label_list=SimpleNamespace(
                find_item_by_shape=Mock(return_value=None)
            ),
            unique_label_list=SimpleNamespace(
                find_items_by_label=Mock(return_value=[object()])
            ),
            set_dirty=Mock(),
            _refresh_shape_filters=Mock(),
        )

        LabelingWidget.batch_edit_labels(
            widget, [unchanged_shape, changed_shape]
        )

        self.assertEqual(unchanged_shape.attributes, {"color": "blue"})
        self.assertEqual(changed_shape.attributes, {"color": "red"})
        widget.reset_attribute.assert_called_once_with("car", changed_shape)

    def test_radio_buttons_remain_exclusive_across_rows(self):
        options = [f"long-option-{index}" for index in range(6)]
        shape = SimpleNamespace(
            label="car", attributes={"visibility": options[0]}
        )
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.resize(130, 200)
        widget = SimpleNamespace(
            canvas=SimpleNamespace(shapes=[shape]),
            attributes={"car": {"visibility": options}},
            attribute_widget_types={
                "car": {"visibility": "radiobutton"}
            },
            scroll_area=scroll_area,
            attribute_radio_changed=Mock(),
            save_attributes=Mock(),
            show_attributes_panel=Mock(),
            hide_attributes_panel=Mock(),
        )

        LabelingWidget.update_attributes(widget, 0)
        gc.collect()

        buttons = widget.grid_layout_container.findChildren(
            QtWidgets.QRadioButton
        )
        self.assertEqual(len(buttons), len(options))
        self.assertIsNot(buttons[0].parentWidget(), buttons[-1].parentWidget())
        self.assertIsNotNone(buttons[0].group())
        self.assertTrue(
            all(button.group() == buttons[0].group() for button in buttons)
        )

        buttons[0].setChecked(True)
        buttons[-1].setChecked(True)

        self.assertFalse(buttons[0].isChecked())
        self.assertTrue(buttons[-1].isChecked())
        widget.save_attributes.assert_not_called()
