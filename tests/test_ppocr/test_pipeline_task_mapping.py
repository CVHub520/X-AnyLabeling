import json
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from anylabeling.views.labeling.ppocr.config import (
        PPOCR_API_DEFAULT_MODEL,
        PPOCR_API_JOB_URL,
        build_ppocr_api_model_id,
        is_ppocr_api_model_id,
        ppocr_api_model_label,
        PPOCR_PIPELINE_CAPABILITY_KEY,
        resolve_ppocr_api_model,
    )
    from anylabeling.views.labeling.ppocr.data_manager import (
        PPOCRDataManager,
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

    def test_api_model_id_helpers_keep_legacy_id_compatible(self):
        model_id = build_ppocr_api_model_id("PaddleOCR-VL")
        self.assertTrue(is_ppocr_api_model_id(model_id))
        self.assertTrue(is_ppocr_api_model_id("__ppocr_api__"))
        self.assertEqual(resolve_ppocr_api_model(model_id), "PaddleOCR-VL")
        self.assertEqual(
            resolve_ppocr_api_model("__ppocr_api__"),
            PPOCR_API_DEFAULT_MODEL,
        )
        self.assertEqual(
            ppocr_api_model_label("PaddleOCR-VL"), "PaddleOCR-VL (API)"
        )

    def test_parse_ppocr_jsonl_result_merges_layout_results(self):
        text = "\n".join(
            [
                '{"result": {"layoutParsingResults": [{"prunedResult": {"width": 10}}]}}',
                '{"result": {"layoutParsingResults": [{"prunedResult": {"width": 20}}]}}',
            ]
        )
        result = PPOCRPipeline._parse_ppocr_jsonl_result(text)
        pages = result["layoutParsingResults"]
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["prunedResult"]["width"], 10)
        self.assertEqual(pages[1]["prunedResult"]["width"], 20)

    def test_extract_job_metadata_supports_nested_payloads(self):
        payload = {
            "result": {
                "jobId": "39373553546153984",
                "resultUrl": {
                    "jsonUrl": "https://example.test/result.jsonl",
                },
            }
        }
        job_data = PPOCRPipeline._extract_job_payload(payload)
        self.assertEqual(
            PPOCRPipeline._extract_job_id(payload),
            "39373553546153984",
        )
        self.assertEqual(
            PPOCRPipeline._extract_job_json_url(job_data),
            "https://example.test/result.jsonl",
        )

    def test_api_settings_migrates_legacy_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_manager = PPOCRDataManager(tmp_dir)
            data_manager.api_settings_path.write_text(
                json.dumps(
                    {
                        "api_url": "https://legacy.example/layout-parsing",
                        "api_key": "test-key",
                    }
                ),
                encoding="utf-8",
            )
            settings = data_manager.load_api_settings()
        self.assertEqual(
            settings["api_url"],
            "https://legacy.example/layout-parsing",
        )
        self.assertEqual(settings["api_key"], "test-key")
        self.assertEqual(settings["api_model"], PPOCR_API_DEFAULT_MODEL)

    def test_update_api_settings_preserves_selected_api_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_manager = PPOCRDataManager(tmp_dir)
            pipeline = PPOCRPipeline(data_manager)
            pipeline.set_pipeline_model(
                build_ppocr_api_model_id("PaddleOCR-VL")
            )
            pipeline.update_api_settings(PPOCR_API_JOB_URL, "test-key")
            settings = data_manager.load_api_settings()
        self.assertEqual(settings["api_model"], "PaddleOCR-VL")
