import os
import importlib.util
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtWidgets

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


if PYQT_AVAILABLE:
    MODULE_PATH = (
        Path(__file__).resolve().parents[1]
        / "anylabeling"
        / "views"
        / "labeling"
        / "widgets"
        / "label_list_widget.py"
    )
    MODULE_SPEC = importlib.util.spec_from_file_location(
        "test_label_list_widget_module", MODULE_PATH
    )
    MODULE = importlib.util.module_from_spec(MODULE_SPEC)
    assert MODULE_SPEC.loader is not None
    MODULE_SPEC.loader.exec_module(MODULE)
    LabelListWidget = MODULE.LabelListWidget
    LabelListWidgetItem = MODULE.LabelListWidgetItem


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for label list widget tests"
)
class TestLabelListWidget(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

    def test_remove_item_does_not_emit_item_dropped(self):
        widget = LabelListWidget()
        for name in ["car", "bus", "truck"]:
            widget.add_iem(LabelListWidgetItem(name, shape=name))

        dropped = []
        widget.item_dropped.connect(lambda: dropped.append(True))

        widget.remove_item(widget[1])

        self.assertEqual(dropped, [])
        self.assertEqual([widget[i].text() for i in range(len(widget))], ["car", "truck"])

    def test_internal_move_uses_insert_mode_instead_of_overwrite(self):
        widget = LabelListWidget()

        self.assertFalse(widget.dragDropOverwriteMode())
