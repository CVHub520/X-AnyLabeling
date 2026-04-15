import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from anylabeling.views.labeling.ppocr.config import (
        PPOCR_PIPELINE_CAPABILITY_KEY,
    )
    from anylabeling.views.labeling.ppocr.pipeline import PPOCRPipeline

    PIPELINE_AVAILABLE = True
except Exception:
    PIPELINE_AVAILABLE = False


@unittest.skipUnless(
    PIPELINE_AVAILABLE,
    "PPOCR pipeline dependencies are required for pipeline contract tests",
)
class TestPPOCRPipelineTaskMapping(unittest.TestCase):

    def setUp(self):
        self.pipeline = PPOCRPipeline.__new__(PPOCRPipeline)

    def test_ppocr_pipeline_capability_detection(self):
        key = PPOCR_PIPELINE_CAPABILITY_KEY
        self.assertTrue(
            self.pipeline._is_ppocr_pipeline_model(
                {"capabilities": {key: True}}
            )
        )
        self.assertTrue(
            self.pipeline._is_ppocr_pipeline_model(
                {"capabilities": {key: {"enabled": True}}}
            )
        )
        self.assertFalse(
            self.pipeline._is_ppocr_pipeline_model(
                {"capabilities": {key: {"enabled": False}}}
            )
        )
        self.assertFalse(
            self.pipeline._is_ppocr_pipeline_model(
                {"capabilities": {"other_feature": True}}
            )
        )

    def test_collect_pipeline_models_uses_display_name_fallback(self):
        models = self.pipeline._collect_pipeline_models(
            {
                "ppocr_document_parser": {
                    "display_name": "PPOCR Document Parser",
                    "capabilities": {
                        PPOCR_PIPELINE_CAPABILITY_KEY: True,
                    },
                },
                "generic_ocr": {
                    "display_name": "Generic OCR",
                },
                "ppocr_pipeline_2": {
                    "capabilities": {
                        PPOCR_PIPELINE_CAPABILITY_KEY: {"enabled": True},
                    },
                },
            }
        )
        self.assertEqual(
            [model.model_id for model in models],
            ["ppocr_document_parser", "ppocr_pipeline_2"],
        )
        self.assertEqual(
            [model.display_name for model in models],
            ["PPOCR Document Parser", "ppocr_pipeline_2"],
        )

    def test_extract_prediction_page_supports_single_page_payload(self):
        payload = {"prunedResult": {"parsing_res_list": []}}
        page_data = PPOCRPipeline._extract_prediction_page(payload)
        self.assertIs(page_data, payload)

    def test_extract_prediction_page_supports_layout_results_payload(self):
        payload = {
            "layoutParsingResults": [
                {"prunedResult": {"parsing_res_list": []}},
            ]
        }
        page_data = PPOCRPipeline._extract_prediction_page(payload)
        self.assertEqual(
            page_data,
            {"prunedResult": {"parsing_res_list": []}},
        )

    def test_extract_prediction_page_rejects_invalid_payload(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "missing prunedResult or layoutParsingResults",
        ):
            PPOCRPipeline._extract_prediction_page({"data": {}})
