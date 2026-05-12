import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtWidgets
    from PyQt6.QtGui import QColor

    import anylabeling.resources.resources  # noqa: F401
    from anylabeling.views.labeling.utils.theme import get_theme, init_theme
    from anylabeling.views.labeling.widgets.ppocr_dialog import (
        PPOCR_RESULT_ACTION_ICON_SIZE,
        PPOCR_RESULT_ACTION_ICON_RENDER_SCALE,
        _tinted_icon_pixmap,
    )

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for result action icon tests"
)
class TestPPOCRResultActionIcons(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

    def tearDown(self):
        init_theme("light")

    def test_result_action_icons_share_theme_color(self):
        init_theme("dark")
        target = QColor(get_theme()["text_secondary"])
        target_rgb = (target.red(), target.green(), target.blue())

        for icon_name in ("settings", "refresh", "copy", "download"):
            pixmap = _tinted_icon_pixmap(
                icon_name,
                get_theme()["text_secondary"],
                PPOCR_RESULT_ACTION_ICON_SIZE,
            )
            self.assertFalse(pixmap.isNull(), icon_name)
            self.assertEqual(
                pixmap.width(),
                PPOCR_RESULT_ACTION_ICON_SIZE
                * PPOCR_RESULT_ACTION_ICON_RENDER_SCALE,
            )
            self.assertEqual(pixmap.devicePixelRatio(), 1.0)
            image = pixmap.toImage()
            high_alpha_colors = set()
            for y in range(image.height()):
                for x in range(image.width()):
                    color = image.pixelColor(x, y)
                    if color.alpha() > 128:
                        high_alpha_colors.add(
                            (color.red(), color.green(), color.blue())
                        )
            self.assertTrue(high_alpha_colors, icon_name)
            for color_rgb in high_alpha_colors:
                self.assertLessEqual(
                    max(
                        abs(color_rgb[0] - target_rgb[0]),
                        abs(color_rgb[1] - target_rgb[1]),
                        abs(color_rgb[2] - target_rgb[2]),
                    ),
                    1,
                    icon_name,
                )
