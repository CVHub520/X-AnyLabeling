import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtWidgets

    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.widgets.label_dialog import LabelDialog

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for KIE linking tests")
class TestKieLinking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def setUp(self):
        self.dialog = LabelDialog()

    def tearDown(self):
        self.dialog.close()
        self.app.processEvents()

    @staticmethod
    def shape_data(kie_linking):
        return {
            "label": "key",
            "points": [[0, 0], [1, 0], [1, 1]],
            "shape_type": "polygon",
            "kie_linking": kie_linking,
        }

    def test_shape_accepts_integer_linking_pairs(self):
        shape = Shape().load_from_dict(self.shape_data([[1, 2], [3, 4]]))

        self.assertEqual(shape.kie_linking, [[1, 2], [3, 4]])

    def test_shape_rejects_invalid_linking_data(self):
        invalid_values = [
            "__import__('builtins').sum([20, 22])",
            [[1, 2, 3]],
            [[1, 2.0]],
            [[1, {"value": 2}]],
            [[True, 2]],
        ]

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "kie_linking must be a list of integer pairs",
                ):
                    Shape().load_from_dict(self.shape_data(value))

    def test_dialog_keeps_linking_data_separate_from_display_text(self):
        self.dialog.reset_linking([[1, 2]])
        item = self.dialog.linking_list.item(0)

        self.assertEqual(
            item.data(QtCore.Qt.ItemDataRole.UserRole), [1, 2]
        )
        item.setText("__import__('builtins').sum([20, 22])")

        self.assertEqual(self.dialog.get_kie_linking(), [[1, 2]])

    def test_dialog_stores_new_linking_pair_as_data(self):
        self.dialog.linking_input.setText("[3, 4]")

        self.dialog.add_linking_pair()

        item = self.dialog.linking_list.item(0)
        self.assertEqual(item.text(), "[3, 4]")
        self.assertEqual(
            item.data(QtCore.Qt.ItemDataRole.UserRole), [3, 4]
        )
        self.assertEqual(self.dialog.get_kie_linking(), [[3, 4]])

    def test_dialog_rejects_boolean_linking_values(self):
        self.dialog.linking_input.setText("[True, 2]")

        with patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.dialog.add_linking_pair()

        warning.assert_called_once()
        self.assertEqual(self.dialog.linking_list.count(), 0)


if __name__ == "__main__":
    unittest.main()
