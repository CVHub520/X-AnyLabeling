import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from anylabeling.views.labeling.ppocr.data_manager import PPOCRDataManager
from anylabeling.views.labeling.ppocr.pipeline import PPOCRPipeline


class TestDocumentPipeline(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.pipeline = PPOCRPipeline(PPOCRDataManager(str(self.root)))
        self.pipeline.pipeline_model = "unlimited_ocr_2"
        self.paths = []
        for index in range(2):
            path = self.root / f"{index}.png"
            Image.new("RGB", (20, 10), (index * 255, 0, 0)).save(path)
            self.paths.append(path)

    def test_whole_document_uses_one_request_in_page_order(self):
        response = Mock()
        response.json.return_value = {"success": True, "data": {"pages": 2}}
        with patch.object(
            self.pipeline, "_post_json_with_cancel", return_value=response
        ) as post:
            self.assertEqual(
                self.pipeline._predict_remote_document(self.paths),
                {"pages": 2},
            )
        self.assertEqual(post.call_count, 1)
        self.assertTrue(
            post.call_args.args[0].endswith("/v1/predict/document")
        )
        payload = post.call_args.kwargs["payload"]
        self.assertEqual(payload["model"], "unlimited_ocr_2")
        import base64
        from io import BytesIO

        images = [
            Image.open(BytesIO(base64.b64decode(v))) for v in payload["images"]
        ]
        self.assertEqual(
            [im.getpixel((0, 0)) for im in images], [(0, 0, 0), (255, 0, 0)]
        )

    def test_cancelled_document_does_not_send_request(self):
        cancelled = threading.Event()
        cancelled.set()
        with patch.object(self.pipeline, "_post_json_with_cancel") as post:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.pipeline._predict_remote_document(self.paths, cancelled)
        post.assert_not_called()

    def test_page_count_mismatch_is_not_saved(self):
        with self.assertRaisesRegex(RuntimeError, "page count"):
            self.pipeline._normalize_document_result(
                None,
                self.paths,
                {"layoutParsingResults": [{"prunedResult": {}}]},
                job_id="",
                remote=True,
            )
