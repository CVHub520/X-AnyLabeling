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
