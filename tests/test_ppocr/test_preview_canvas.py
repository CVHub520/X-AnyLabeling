import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtWidgets
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QPixmap

    from anylabeling.views.labeling.ppocr.widgets import PPOCRPreviewCanvas
    from anylabeling.views.labeling.utils.theme import get_theme, init_theme

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for preview canvas tests"
)
class TestPPOCRPreviewCanvas(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.widget = PPOCRPreviewCanvas()
        self.widget.show()
        self.app.processEvents()

    def tearDown(self):
        self.widget.close()
        init_theme("light")
        self.app.processEvents()

    def test_empty_canvas_uses_dark_theme_background(self):
        init_theme("dark")
        self.widget.update()
        self.app.processEvents()

        image = self.widget.grab().toImage()
        self.assertEqual(
            image.pixelColor(20, 20), QColor(get_theme()["background"])
        )

    def test_page_canvas_keeps_original_white_background(self):
        init_theme("dark")
        pixmap = QPixmap(80, 80)
        pixmap.fill(Qt.GlobalColor.transparent)
        self.widget.set_page(pixmap, [])
        self.app.processEvents()

        image = self.widget.grab().toImage()
        self.assertEqual(image.pixelColor(20, 20), QColor(255, 255, 255))
