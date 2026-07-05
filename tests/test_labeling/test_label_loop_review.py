import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtWidgets

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for label loop review tests"
)
class TestLabelLoopReview(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.widgets = []

    def tearDown(self):
        for widget in self.widgets:
            widget.close()
        self.app.processEvents()

    def test_find_next_shape_skips_deleted_entries_by_identity(self):
        from anylabeling.views.labeling.label_widget import (
            _find_next_label_loop_shape,
        )

        class EqualShape:
            def __eq__(self, other):
                return True

        first = EqualShape()
        deleted = EqualShape()
        added = EqualShape()
        last = EqualShape()
        shapes = [first, deleted, last]
        canvas_shapes = [first, added, last]

        index, shape = _find_next_label_loop_shape(
            shapes, 1, canvas_shapes
        )

        self.assertEqual(index, 2)
        self.assertIs(shape, last)

    def test_find_next_shape_reports_completed_snapshot(self):
        from anylabeling.views.labeling.label_widget import (
            _find_next_label_loop_shape,
        )

        shapes = [object(), object()]

        index, shape = _find_next_label_loop_shape(shapes, 0, [])

        self.assertEqual(index, len(shapes))
        self.assertIsNone(shape)

    def test_popup_reuses_text_timer_and_nested_widget_position(self):
        from anylabeling.views.labeling.widgets.popup import Popup

        host = QtWidgets.QWidget()
        host.setGeometry(100, 120, 400, 300)
        child = QtWidgets.QWidget(host)
        child.setGeometry(20, 30, 200, 100)
        host.show()
        child.show()
        self.widgets.append(host)
        self.app.processEvents()
        icon_path = os.path.join(
            os.path.dirname(__file__),
            "../..",
            "anylabeling",
            "resources",
            "images",
            "copy-green.svg",
        )

        popup = Popup(
            "Initial",
            parent=host,
            msec=1800,
            icon=icon_path,
        )
        self.widgets.append(popup)
        popup.set_text("Reviewing 2 / 3")
        popup.show_popup(child, popup_height=36, top_offset=24)

        origin = child.mapToGlobal(QtCore.QPoint(0, 0))
        expected_x = origin.x() + (
            child.width() - popup.sizeHint().width()
        ) // 2
        self.assertEqual(popup.label.text(), "Reviewing 2 / 3")
        self.assertIsNotNone(popup.icon_label)
        self.assertFalse(popup.icon_label.pixmap().isNull())
        self.assertEqual(popup.geometry().x(), expected_x)
        self.assertEqual(popup.geometry().y(), origin.y() + 24)
        self.assertTrue(popup.timer.isActive())


if __name__ == "__main__":
    unittest.main()
