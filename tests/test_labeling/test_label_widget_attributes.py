import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from anylabeling.views.labeling.label_widget import LabelingWidget

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for label widget attribute tests"
)
class TestLabelWidgetAttributes(unittest.TestCase):

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
