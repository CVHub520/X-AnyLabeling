import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import Qt
    from PyQt6 import QtWidgets

    from anylabeling.views.labeling.ppocr.editors import (
        PPOCRLatexBlockEditor,
        PPOCRRichTextBlockEditor,
        _configure_latex_preview_rcparams,
        _normalized_latex_source,
        _sanitize_latex_preview_source,
        create_ppocr_block_editor,
        render_latex_preview_pixmap,
    )
    from anylabeling.views.labeling.ppocr.render import PPOCRBlockData
    from anylabeling.views.labeling.ppocr.widgets import PPOCRBlockCard

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for PPOCR LaTeX tests")
class TestPPOCRLatexEditor(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self._widgets = []

    def tearDown(self):
        for widget in self._widgets:
            widget.close()
        self.app.processEvents()

    def _track(self, widget):
        self._widgets.append(widget)
        return widget

    def test_formula_labels_route_to_latex_editor(self):
        for label in (
            "display_formula",
            "formula_number",
            "algorithm",
            "formula",
            "Formula",
        ):
            editor = self._track(create_ppocr_block_editor(label, r"\frac{a}{b}"))
            self.assertIsInstance(editor, PPOCRLatexBlockEditor)

        inline_editor = self._track(
            create_ppocr_block_editor("inline_formula", r"\frac{a}{b}")
        )
        self.assertIsInstance(inline_editor, PPOCRRichTextBlockEditor)

    def test_normalized_latex_source_supports_parenthesis_delimiters(self):
        self.assertEqual(
            _normalized_latex_source(r"\(\angle HBC = \pi / 2\)"),
            r"\angle HBC = \pi / 2",
        )

    def test_sanitize_latex_preview_source_preserves_arrow_commands(self):
        self.assertEqual(
            _sanitize_latex_preview_source(
                r"\left(\xrightarrow[\mathrm{cool}]{heat}\right)"
            ),
            r"(\underset{\mathrm{cool}}{\overset{heat}{\rightarrow}})",
        )
        self.assertEqual(
            _sanitize_latex_preview_source(r"\xleftarrow{back}"),
            r"\overset{back}{\leftarrow}",
        )

    def test_render_latex_preview_pixmap_supports_single_and_multiline(self):
        single_pixmap = render_latex_preview_pixmap(
            r"x=\frac{-b\pm\sqrt{b^{2}-4ac}}{2aa}"
        )
        self.assertFalse(single_pixmap.isNull())
        image = single_pixmap.toImage()
        top_rows_with_content = 0
        for y in range(min(8, image.height())):
            for x in range(image.width()):
                color = image.pixelColor(x, y)
                if (color.red(), color.green(), color.blue(), color.alpha()) != (
                    255,
                    255,
                    255,
                    255,
                ):
                    top_rows_with_content += 1
                    break
        self.assertGreater(top_rows_with_content, 0)

        multiline_pixmap = render_latex_preview_pixmap(
            "\n\n".join(
                [r"x=\frac{-b\pm\sqrt{b^{2}-4ac}}{2aa}"] * 3
            )
        )
        self.assertFalse(multiline_pixmap.isNull())
        self.assertGreater(multiline_pixmap.height(), single_pixmap.height())

        aligned_pixmap = render_latex_preview_pixmap(
            r"\begin{aligned}"
            r"&=\lim_{x\to0}\frac{\left(1+\int_{0}^{x}\mathrm{e}^{t^{2}}\mathrm{d}t\right)\sin x-\mathrm{e}^{x}+1}{x^{2}}"
            r"\\&=\lim_{x\to0}\mathrm{e}^{x^{2}}-\lim_{x\to0}\frac{\mathrm{e}^{x}-1}{2x}"
            r"\\&=1-\frac{1}{2}=\frac{1}{2}."
            r"\end{aligned}"
        )
        self.assertFalse(aligned_pixmap.isNull())
        self.assertGreater(aligned_pixmap.height(), single_pixmap.height())

        big_pixmap = render_latex_preview_pixmap(
            r"\begin{aligned}"
            r"\lim_{x\to0}\Big(\frac{1}{x}-\frac{1}{\sin x}\Big)"
            r"&=\frac{1}{2}"
            r"\end{aligned}"
        )
        self.assertFalse(big_pixmap.isNull())

        array_pixmap = render_latex_preview_pixmap(
            r"$$ \begin{array}{rcl}"
            r"a_{n}&=&\frac{\sum\limits_{k=1}^{n-1}a_{k}}{n-1}+1\\"
            r"&=&\frac{\sum\limits_{k=1}^{n-2}a_{k}+a_{n-1}}{n-1}+1\\"
            r"&=&\frac{(a_{n-1}-1)(n-2)+a_{n-1}}{n-1}+1\\"
            r"&=&a_{n-1}+\frac{1}{n-1}"
            r"\end{array} $$"
        )
        self.assertFalse(array_pixmap.isNull())

        arrow_pixmap = render_latex_preview_pixmap(
            r"$$ \mathrm{H_{2}S+D_{2}O\xrightarrow[\mathrm{cool}]{heat}D_{2}S+H_{2}O} $$"
        )
        self.assertFalse(arrow_pixmap.isNull())

        long_equal_aligned_pixmap = render_latex_preview_pixmap(
            r"$$ \begin{aligned}"
            r"&3\mathrm{H}_{2}+\mathrm{N}_{2}\xlongequal{\quad}2\mathrm{NH}_{3}\\"
            r"&\mathrm{H}_{2}(\mathrm{~g})+2\mathrm{Na}(\mathrm{s})\xlongequal{\quad}2\mathrm{NaH}(\mathrm{s})\\"
            r"&\mathrm{H}_{2}(\mathrm{~g})+\mathrm{Ca}(\mathrm{s})\xlongequal{\quad}\mathrm{CaH}_{2}(\mathrm{~s})\\"
            r"&4\mathrm{H}_{2}(\mathrm{~g})+\mathrm{Fe}_{3}\mathrm{O}_{4}(\mathrm{~s})\xlongequal{\quad}3\mathrm{Fe}(\mathrm{s})+4\mathrm{H}_{2}\mathrm{O}(\mathrm{l})\\"
            r"&3\mathrm{H}_{2}(\mathrm{~g})+\mathrm{WO}_{3}(\mathrm{~s})\xlongequal{\quad}\mathrm{W}(\mathrm{s})+3\mathrm{H}_{2}\mathrm{O}(\mathrm{l})\\"
            r"&2\mathrm{H}_{2}(\mathrm{~g})+\mathrm{TiCl}_{4}(\mathrm{~l})\xlongequal{\quad}\mathrm{Ti}(\mathrm{s})+4\mathrm{HCl}(\mathrm{g})"
            r"\end{aligned} $$"
        )
        self.assertFalse(long_equal_aligned_pixmap.isNull())

        long_equal_temp_pixmap = render_latex_preview_pixmap(
            r"$$ \mathrm{C}(\mathrm{glow})+\mathrm{H}_{2}\mathrm{O}(\mathrm{g})\xlongequal{1.273\mathrm{~K}}\mathrm{H}_{2}(\mathrm{~g})+\mathrm{CO}(\mathrm{g}) $$"
        )
        self.assertFalse(long_equal_temp_pixmap.isNull())

    def test_configure_latex_preview_rcparams_uses_custom_font_for_cjk(self):
        rc_params = {}
        with patch(
            "anylabeling.views.labeling.ppocr.editors._resolve_latex_preview_cjk_font",
            return_value="Microsoft YaHei",
        ):
            _configure_latex_preview_rcparams(rc_params, r"阳极 + x")
        self.assertEqual(rc_params["mathtext.fontset"], "custom")
        self.assertEqual(rc_params["mathtext.fallback"], "stix")
        self.assertEqual(rc_params["mathtext.rm"], "Microsoft YaHei")
        self.assertEqual(rc_params["mathtext.sf"], "Microsoft YaHei")
        self.assertEqual(rc_params["font.family"], "Microsoft YaHei")

    def test_render_latex_preview_pixmap_reports_multiline_error_line(self):
        with self.assertRaisesRegex(ValueError, r"Line 3:"):
            render_latex_preview_pixmap(
                r"\frac{a}{b}" "\n\n" r"\badcommand"
            )

    def test_latex_editor_disables_save_when_preview_is_invalid(self):
        editor = self._track(PPOCRLatexBlockEditor(r"\frac{a}{b}"))
        editor.show()
        self.app.processEvents()
        editor._update_preview()
        self.app.processEvents()

        self.assertTrue(editor.save_button.isEnabled())

        editor.source_editor.setPlainText(r"\badcommand")
        editor._update_preview()
        self.app.processEvents()
        self.assertFalse(editor.save_button.isEnabled())
        self.assertIn("badcommand", editor.preview_content.text())

        editor.source_editor.setPlainText(
            "\n\n".join([r"\frac{a}{b}"] * 2)
        )
        editor._update_preview()
        self.app.processEvents()
        self.assertTrue(editor.save_button.isEnabled())
        pixmap = editor.preview_content.pixmap()
        self.assertIsNotNone(pixmap)
        self.assertFalse(pixmap.isNull())

    def test_latex_editor_preview_expands_without_vertical_scrollbar(self):
        editor = self._track(PPOCRLatexBlockEditor(r"\frac{a}{b}"))
        editor.resize(720, 360)
        editor.show()
        self.app.processEvents()

        single_height = editor.preview_scroll.height()
        self.assertEqual(
            editor.preview_scroll.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )

        editor.source_editor.setPlainText(
            "\n\n".join([r"x=\frac{-b\pm\sqrt{b^{2}-4ac}}{2aa}"] * 4)
        )
        editor._update_preview()
        self.app.processEvents()

        self.assertGreater(editor.preview_scroll.height(), single_height)
        self.assertEqual(
            editor.preview_scroll.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )

    def test_formula_block_card_renders_formula_pixmap_instead_of_raw_text(self):
        block = PPOCRBlockData(
            page_no=1,
            block_uid="block_1",
            block_key="page_1:block_1",
            label="display_formula",
            display_label="Display formula",
            content="\n\n".join([r"\frac{a}{b}"] * 2),
            points=[],
            category_color="rgb(250, 219, 20)",
        )
        card = self._track(PPOCRBlockCard(block, Path(".")))
        card.resize(480, 220)
        card.show()
        self.app.processEvents()

        self.assertTrue(card.formula_label.isVisible())
        pixmap = card.formula_label.pixmap()
        self.assertIsNotNone(pixmap)
        self.assertFalse(pixmap.isNull())
        self.assertFalse(card.content_label.isVisible())
