import os
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtWidgets

    from anylabeling.views.labeling.ppocr.editors import (
        PPOCRTableBlockEditor,
        create_ppocr_block_editor,
    )
    from anylabeling.views.labeling.ppocr.render import PPOCRBlockData
    from anylabeling.views.labeling.ppocr.widgets import PPOCRBlockCard

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


HTML_TABLE_CONTENT = (
    "<table><tr><td></td><td>符号</td><td>图象的特征</td></tr>"
    "<tr><td rowspan=\"2\">a</td><td>a&gt;0</td><td>开口向上</td></tr>"
    "<tr><td>a&lt;0</td><td>开口向下</td></tr>"
    "<tr><td rowspan=\"3\">b</td><td>b=0</td><td>对称轴为y轴</td></tr>"
    "<tr><td>a、b同号</td><td>对称轴在y轴左侧</td></tr>"
    "<tr><td>a、b异号</td><td>对称轴在y轴右侧</td></tr>"
    "<tr><td rowspan=\"3\">c</td><td>c=0</td><td>图象过原点</td></tr>"
    "<tr><td>c&gt;0</td><td>与y轴的正半轴相交</td></tr>"
    "<tr><td>c&lt;0</td><td>与y轴的负半轴相交</td></tr>"
    "<tr><td rowspan=\"3\">b²-4ac</td><td>b²-4ac=0</td>"
    "<td>与x轴有唯一交点（顶点）</td></tr>"
    "<tr><td>b²-4ac&gt;0</td><td>与x轴有两个不同的交点</td></tr>"
    "<tr><td>b²-4ac&lt;0</td><td>与x轴无交点</td></tr></table>"
)


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for PPOCR table tests")
class TestPPOCRTableEditor(unittest.TestCase):
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

    def test_table_editor_parses_html_table_with_rowspan(self):
        editor = self._track(PPOCRTableBlockEditor(HTML_TABLE_CONTENT))
        editor.show()
        self.app.processEvents()

        self.assertEqual(editor.table.rowCount(), 12)
        self.assertEqual(editor.table.columnCount(), 3)
        self.assertEqual(editor.table.item(0, 1).text(), "符号")
        self.assertEqual(editor.table.item(1, 1).text(), "a>0")
        self.assertEqual(editor.table.item(2, 1).text(), "a<0")
        self.assertEqual(editor.table.rowSpan(1, 0), 2)
        self.assertEqual(editor.table.rowSpan(3, 0), 3)
        self.assertEqual(editor.table.rowSpan(6, 0), 3)
        self.assertEqual(editor.table.rowSpan(9, 0), 3)
        serialized = editor._serialize_table_tokens()
        self.assertIn("<nl>", serialized)

    def test_table_editor_supports_undo_and_redo_for_cell_change(self):
        editor = self._track(PPOCRTableBlockEditor(HTML_TABLE_CONTENT))
        editor.show()
        self.app.processEvents()

        original = editor._serialize_table_tokens()
        item = editor.table.item(1, 1)
        self.assertIsNotNone(item)
        item.setText("a>1")
        self.app.processEvents()
        changed = editor._serialize_table_tokens()
        self.assertNotEqual(changed, original)

        editor._undo_table_change()
        self.app.processEvents()
        self.assertEqual(editor._serialize_table_tokens(), original)

        editor._redo_table_change()
        self.app.processEvents()
        self.assertEqual(editor._serialize_table_tokens(), changed)

    def test_table_card_renders_html_table_in_initial_state(self):
        block = PPOCRBlockData(
            page_no=1,
            block_uid="table_block",
            block_key="page_1:table_block",
            label="table",
            display_label="Table",
            content=HTML_TABLE_CONTENT,
            points=[],
            category_color="rgb(96, 129, 255)",
        )
        card = self._track(PPOCRBlockCard(block, Path(".")))
        card.resize(720, 340)
        card.show()
        self.app.processEvents()

        self.assertTrue(card.content_label.isVisible())
        plain = card.content_label.toPlainText()
        self.assertIn("符号", plain)
        self.assertIn("图象的特征", plain)
        self.assertNotIn("<table", plain.casefold())

    def test_html_table_content_uses_table_editor_even_if_label_is_text(self):
        editor = self._track(
            create_ppocr_block_editor("text", HTML_TABLE_CONTENT)
        )
        self.assertIsInstance(editor, PPOCRTableBlockEditor)
