import os
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtWidgets

    from anylabeling.views.labeling.ppocr.config import (
        PPOCR_FILE_TYPE_IMAGE,
        PPOCR_STATUS_PARSED,
    )
    from anylabeling.views.labeling.ppocr.data_manager import PPOCRFileRecord
    from anylabeling.views.labeling.ppocr.widgets import PPOCRRecentsListWidget

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for PPOCR recents list tests")
class TestPPOCRRecentsListWidget(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.widget = PPOCRRecentsListWidget()
        self.widget.show()
        self.app.processEvents()

    def tearDown(self):
        self.widget.close()
        self.app.processEvents()

    @staticmethod
    def _record(filename: str) -> PPOCRFileRecord:
        return PPOCRFileRecord(
            filename=filename,
            source_path=Path(f"/tmp/{filename}"),
            json_path=Path(f"/tmp/{filename}.json"),
            file_type=PPOCR_FILE_TYPE_IMAGE,
            status=PPOCR_STATUS_PARSED,
            mtime=0.0,
            timestamp="2026-01-01 00:00:00",
            size_bytes=1,
        )

    def test_click_updates_selected_state_without_rerender(self):
        records = [self._record("a.png"), self._record("b.png")]
        self.widget.render_records(records, "a.png")
        self.app.processEvents()

        first_widget = self.widget.itemWidget(self.widget.item(0))
        second_widget = self.widget.itemWidget(self.widget.item(1))
        self.assertTrue(first_widget._selected)
        self.assertFalse(second_widget._selected)

        selected_names = []
        self.widget.fileSelected.connect(selected_names.append)

        self.widget._on_item_clicked(self.widget.item(1))
        self.app.processEvents()

        self.assertEqual(selected_names, ["b.png"])
        self.assertFalse(first_widget._selected)
        self.assertTrue(second_widget._selected)
        self.assertEqual(self.widget.count(), 3)
