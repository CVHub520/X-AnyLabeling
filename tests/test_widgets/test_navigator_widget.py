import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtWidgets

    from anylabeling.resources import resources  # noqa: F401
    from anylabeling.views.labeling.widgets.navigator_widget import (
        NavigatorWidget,
    )

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for navigator widget tests"
)
class TestNavigatorWidget(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

    def test_set_image_with_invalid_type_does_not_crash(self):
        widget = NavigatorWidget()
        widget.set_image(None)
        widget.set_image(12345)
        widget.set_image(object())

    def test_set_image_with_invalid_path_does_not_crash(self):
        widget = NavigatorWidget()
        widget.set_image("/nonexistent/path/to/image.png")
