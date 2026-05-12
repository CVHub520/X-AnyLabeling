import unittest

from anylabeling.views.labeling.ppocr.style import get_icon_button_style


class TestPPOCRStyle(unittest.TestCase):

    def test_icon_button_style_clears_global_button_padding(self):
        style = get_icon_button_style()
        self.assertIn("padding: 0px", style)
        self.assertIn("min-width: 0px", style)
        self.assertIn("min-height: 0px", style)
