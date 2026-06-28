import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtWidgets

    from anylabeling.views.labeling.label_widget import LabelingWidget

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for label widget group tests"
)
class TestLabelWidgetGroups(unittest.TestCase):

    def test_group_delete_requires_confirmation(self):
        shapes = [object(), object()]
        canvas = SimpleNamespace(
            _active_group_shapes=Mock(return_value=shapes),
            delete_selected=Mock(return_value=shapes),
            selected_shapes=[],
        )
        widget = SimpleNamespace(
            canvas=canvas,
            tr=lambda text: text,
            remove_labels=Mock(),
            shape_selection_changed=Mock(),
            set_dirty=Mock(),
            no_shape=Mock(return_value=False),
            actions=SimpleNamespace(on_shapes_present=[]),
        )

        with patch.object(
            QtWidgets.QMessageBox,
            "warning",
            return_value=QtWidgets.QMessageBox.StandardButton.No,
        ):
            LabelingWidget.delete_selected_shape(widget)

        canvas.delete_selected.assert_not_called()

        with patch.object(
            QtWidgets.QMessageBox,
            "warning",
            return_value=QtWidgets.QMessageBox.StandardButton.Yes,
        ):
            LabelingWidget.delete_selected_shape(widget)

        canvas.delete_selected.assert_called_once_with()

    def test_group_copy_and_paste_preserves_group_context(self):
        shapes = [Mock(), Mock()]
        copied_shapes = [object(), object()]
        pasted_shapes = [object(), object()]
        for shape, copied_shape in zip(shapes, copied_shapes):
            shape.copy.return_value = copied_shape
        canvas = SimpleNamespace(
            _active_group_shapes=Mock(return_value=shapes),
            _selected_group_id=3,
            selected_shapes=shapes,
            prepare_pasted_shapes=Mock(return_value=pasted_shapes),
        )
        widget = SimpleNamespace(
            canvas=canvas,
            _config={"system_clipboard": False},
            _copied_shapes=None,
            _copied_group_id=None,
            actions=SimpleNamespace(paste=Mock()),
            load_shapes=Mock(),
            set_dirty=Mock(),
        )

        LabelingWidget.copy_selected_shape(widget)
        LabelingWidget.paste_selected_shape(widget)

        self.assertEqual(widget._copied_group_id, 3)
        self.assertEqual(widget._copied_shapes, copied_shapes)
        canvas.prepare_pasted_shapes.assert_called_once_with(copied_shapes, 3)
        widget.load_shapes.assert_called_once_with(
            pasted_shapes, replace=False
        )
