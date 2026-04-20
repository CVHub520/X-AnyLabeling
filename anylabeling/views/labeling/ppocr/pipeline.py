from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from PIL import Image, ImageOps
from PyQt6.QtCore import QObject, pyqtSignal
import requests

from anylabeling.views.labeling.logger import logger

from .config import (
    PPOCR_API_JOB_URL,
    PPOCR_API_MODE_ASYNC_JOBS,
    PPOCR_FILE_TYPE_IMAGE,
    PPOCR_PIPELINE_CAPABILITY_KEY,
    PPOCRPipelineModel,
    PPOCRServiceProbe,
    build_ppocr_api_model_id,
    is_ppocr_api_model_id,
    normalize_ppocr_api_model,
    resolve_ppocr_api_model,
)
from .data_manager import PPOCRDataManager, PPOCRFileRecord
from .render import (
    build_block_key,
    build_unique_block_key,
    normalize_block_label,
)

IMAGE_ONLY_LABELS = {
    "image",
    "header_image",
    "footer_image",
}

_SUPPORTED_API_PARAM_KEYS = {
    "fileType",
    "promptLabel",
    "markdownIgnoreLabels",
    "useDocOrientationClassify",
    "useDocUnwarping",
    "useLayoutDetection",
    "useChartRecognition",
    "useSealRecognition",
    "useOcrForImageBlock",
    "mergeTables",
    "relevelTitles",
    "layoutShapeMode",
    "repetitionPenalty",
    "temperature",
    "topP",
    "minPixels",
    "maxPixels",
    "layoutNms",
    "restructurePages",
}


@dataclass
class PPOCRParsingProgress:
    filename: str
    index: int
    total: int
    page_no: int
    page_total: int


class PPOCRPipeline:
    def __init__(
        self,
        data_manager: PPOCRDataManager,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.data_manager = data_manager
        remote_settings = (config or {}).get("remote_server_settings") or {}
        self.server_url = str(
            remote_settings.get(
                "server_url",
                "http://127.0.0.1:8000",
            )
            or ""
        ).strip()
        self.server_api_key = str(remote_settings.get("api_key") or "").strip()
        self.timeout = int(remote_settings.get("timeout", 180) or 180)
        api_settings = self.data_manager.load_api_settings()
        self.api_url = PPOCR_API_JOB_URL
        self.api_key = api_settings.get("api_key", "")
        self.api_model = normalize_ppocr_api_model(
            api_settings.get("api_model")
        )
        self.pipeline_model = build_ppocr_api_model_id(self.api_model)
        self.pipeline_models: list[PPOCRPipelineModel] = []
        self.auth_schemes = ["token", "bearer"]
        self.client_platform = "x-anylabeling-client"
        self.default_api_payload = {
            "fileType": 1,
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useLayoutDetection": True,
            "useChartRecognition": False,
            "useSealRecognition": True,
            "useOcrForImageBlock": False,
            "mergeTables": True,
            "relevelTitles": True,
            "layoutShapeMode": "auto",
            "repetitionPenalty": 1,
            "temperature": 0,
            "topP": 1,
            "minPixels": 147384,
            "maxPixels": 2822400,
            "layoutNms": True,
            "restructurePages": True,
        }
        self.service_probe = PPOCRServiceProbe(
            is_online=False,
            server_url=self.server_url,
            pipeline_model=self.pipeline_model,
            pipeline_models=tuple(),
            error_message="Service probing has not run yet.",
        )

    def update_api_settings(
        self,
        api_url: str,
        api_key: str,
        api_model: str | None = None,
    ) -> None:
        self.api_url = str(api_url or PPOCR_API_JOB_URL).strip()
        self.api_key = str(api_key or "").strip()
        if api_model is not None:
            self.api_model = normalize_ppocr_api_model(api_model)
        if is_ppocr_api_model_id(self.pipeline_model):
            self.pipeline_model = build_ppocr_api_model_id(self.api_model)
        self.data_manager.save_api_settings(
            self.api_url,
            self.api_key,
            self.api_model,
            PPOCR_API_MODE_ASYNC_JOBS,
        )

    def update_api_model(self, api_model: str) -> None:
        self.api_model = normalize_ppocr_api_model(api_model)
        if is_ppocr_api_model_id(self.pipeline_model):
            self.pipeline_model = build_ppocr_api_model_id(self.api_model)
        self.data_manager.save_api_settings(
            self.api_url,
            self.api_key,
            self.api_model,
            PPOCR_API_MODE_ASYNC_JOBS,
        )

    def has_required_api_settings(self) -> bool:
        return bool(self.api_key)

    def has_remote_server_settings(self) -> bool:
        return bool(str(self.server_url or "").strip())

    def _server_headers(self) -> dict[str, str]:
        if self.server_api_key:
            return {"Token": self.server_api_key}
        return {}

    def _candidate_server_base_urls(self) -> list[str]:
        raw_url = str(self.server_url or "").strip()
        if not raw_url:
            return []
        candidates: list[str] = []

        def append_candidate(url_text: str) -> None:
            normalized = str(url_text or "").strip().rstrip("/")
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        append_candidate(raw_url)
        try:
            parsed = urlsplit(raw_url)
        except Exception:
            return candidates
        if not parsed.scheme or not parsed.netloc:
            return candidates

        path = (parsed.path or "").rstrip("/")
        suffixes = ("/v1/predict", "/predict", "/v1/models")
        for suffix in suffixes:
            if not path.endswith(suffix):
                continue
            base_path = path[: -len(suffix)]
            append_candidate(
                urlunsplit(
                    (
                        parsed.scheme,
                        parsed.netloc,
                        base_path,
                        "",
                        "",
                    )
                )
            )
        return candidates

    @staticmethod
    def _is_ppocr_pipeline_model(model_info: Any) -> bool:
        if not isinstance(model_info, dict):
            return False
        capabilities = model_info.get("capabilities")
        if not isinstance(capabilities, dict):
            return False
        capability = capabilities.get(PPOCR_PIPELINE_CAPABILITY_KEY)
        if isinstance(capability, bool):
            return capability
        if isinstance(capability, dict):
            enabled = capability.get("enabled")
            if isinstance(enabled, bool):
                return enabled
            return True
        if isinstance(capability, str):
            return capability.casefold() in {
                "1",
                "true",
                "yes",
                "enabled",
            }
        return False

    @staticmethod
    def _model_display_name(model_id: str, model_info: Any) -> str:
        if isinstance(model_info, dict):
            display_name = str(model_info.get("display_name") or "").strip()
            if display_name:
                return display_name
        return model_id

    def _collect_pipeline_models(
        self,
        models_data: dict[str, Any],
    ) -> list[PPOCRPipelineModel]:
        pipeline_models: list[PPOCRPipelineModel] = []
        for model_id, model_info in models_data.items():
            if not isinstance(model_id, str):
                continue
            if not self._is_ppocr_pipeline_model(model_info):
                continue
            pipeline_models.append(
                PPOCRPipelineModel(
                    model_id=model_id,
                    display_name=self._model_display_name(
                        model_id, model_info
                    ),
                )
            )
        return pipeline_models

    def _select_pipeline_model(
        self, pipeline_models: list[PPOCRPipelineModel]
    ) -> str:
        model_ids = {model.model_id for model in pipeline_models}
        if self.pipeline_model in model_ids:
            return self.pipeline_model
        return pipeline_models[0].model_id

    def probe_service(self) -> PPOCRServiceProbe:
        if not self.has_remote_server_settings():
            self.pipeline_models = []
            self.service_probe = PPOCRServiceProbe(
                is_online=False,
                server_url="",
                pipeline_model=self.pipeline_model,
                pipeline_models=tuple(),
                error_message="Remote server URL is not configured.",
            )
            return self.service_probe
        last_error = "Server probing has not run yet."
        for base_url in self._candidate_server_base_urls():
            normalized_base_url = base_url.rstrip("/")
            if normalized_base_url.endswith("/v1/models"):
                normalized_base_url = normalized_base_url[: -len("/v1/models")]
            models_url = f"{normalized_base_url}/v1/models"
            try:
                response = requests.get(
                    models_url,
                    headers=self._server_headers(),
                    timeout=min(self.timeout, 10),
                )
                response.raise_for_status()
                payload = response.json()
                models_data = payload.get("data") or {}
                if not isinstance(models_data, dict):
                    raise RuntimeError("Invalid /v1/models response data")

                pipeline_models = self._collect_pipeline_models(models_data)
                if not pipeline_models:
                    raise RuntimeError(
                        "No PPOCR pipeline model found. "
                        "Expected capabilities.ppocr_pipeline = true."
                    )

                self.pipeline_models = pipeline_models
                selected_model = self._select_pipeline_model(pipeline_models)
                self.service_probe = PPOCRServiceProbe(
                    is_online=True,
                    server_url=normalized_base_url,
                    pipeline_model=selected_model,
                    pipeline_models=tuple(self.pipeline_models),
                    error_message="",
                )
                return self.service_probe
            except Exception as exc:
                last_error = str(exc)
                continue
        logger.debug(f"Failed to probe PaddleOCR service: {last_error}")
        self.service_probe = PPOCRServiceProbe(
            is_online=False,
            server_url=self.server_url,
            pipeline_model=self.pipeline_model,
            pipeline_models=tuple(self.pipeline_models),
            error_message=last_error,
        )
        return self.service_probe

    def set_pipeline_model(self, model_id: str) -> None:
        if model_id:
            if is_ppocr_api_model_id(model_id):
                self.api_model = resolve_ppocr_api_model(
                    model_id, self.api_model
                )
                self.pipeline_model = build_ppocr_api_model_id(self.api_model)
                return
            self.pipeline_model = model_id

    def _current_result_pipeline_model(self) -> str:
        if is_ppocr_api_model_id(self.pipeline_model):
            return self.api_model
        return self.pipeline_model

    def parse_record(
        self,
        record: PPOCRFileRecord,
        cancel_event: threading.Event | None = None,
        progress_callback=None,
        index: int = 1,
        total: int = 1,
    ) -> dict[str, Any]:
        is_official_api = is_ppocr_api_model_id(self.pipeline_model)
        if not is_official_api:
            service_probe = self.probe_service()
            if not service_probe.is_online:
                raise RuntimeError(
                    service_probe.error_message
                    or "PaddleOCR service is offline"
                )
        if record.file_type == PPOCR_FILE_TYPE_IMAGE:
            page_paths = [record.source_path]
        else:
            page_paths = self.data_manager.ensure_pdf_pages(record, force=True)
        self.data_manager.clear_block_images(record)
        if is_official_api:
            return self._predict_ppocr_api_async_job(
                record,
                page_paths,
                cancel_event=cancel_event,
                progress_callback=progress_callback,
                index=index,
                total=total,
            )

        layout_results = []
        page_sizes = []
        block_image_paths = {}
        page_total = len(page_paths)

        for page_index, page_path in enumerate(page_paths, start=1):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Parsing cancelled")
            if progress_callback is not None:
                progress_callback(
                    PPOCRParsingProgress(
                        filename=record.filename,
                        index=index,
                        total=total,
                        page_no=page_index,
                        page_total=page_total,
                    )
                )
            page_result, page_size, page_block_image_paths = self._parse_page(
                record,
                page_path,
                page_index,
                cancel_event=cancel_event,
            )
            layout_results.append(page_result)
            page_sizes.append({"width": page_size[0], "height": page_size[1]})
            block_image_paths.update(page_block_image_paths)

        document_data = {
            "layoutParsingResults": layout_results,
            "preprocessedImages": [],
            "dataInfo": {
                "type": record.file_type,
                "numPages": len(page_paths),
                "pages": page_sizes,
            },
            "_ppocr_meta": {
                "status": "parsed",
                "source_path": f"files/{record.filename}",
                "updated_at": "",
                "error_message": "",
                "edited_blocks": [],
                "block_image_paths": block_image_paths,
                "pipeline_model": self._current_result_pipeline_model(),
            },
        }
        self.data_manager.save_record_data(record, document_data)
        return document_data

    def _parse_page(
        self,
        record: PPOCRFileRecord,
        page_path: Path,
        page_no: int,
        cancel_event: threading.Event | None = None,
    ) -> tuple[dict[str, Any], tuple[int, int], dict[str, str]]:
        with Image.open(page_path) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            ):
                rgba_image = image.convert("RGBA")
                background = Image.new(
                    "RGBA",
                    rgba_image.size,
                    (255, 255, 255, 255),
                )
                image = Image.alpha_composite(background, rgba_image).convert(
                    "RGB"
                )
            else:
                image = image.convert("RGB")
            params = {
                "page_no": page_no,
                "source_file_type": record.file_type,
            }
            if is_ppocr_api_model_id(self.pipeline_model):
                page_result_data = self._predict_ppocr_api(
                    image,
                    params,
                    cancel_event=cancel_event,
                )
            else:
                page_result_data = self._predict_remote(
                    self.pipeline_model,
                    image,
                    params=params,
                    cancel_event=cancel_event,
                )
            page_result = self._normalize_page_result(
                page_result_data,
                page_path,
                image.width,
                image.height,
            )
            parsing_res_list = (page_result.get("prunedResult") or {}).get(
                "parsing_res_list"
            ) or []
            block_image_paths = self._collect_block_image_paths(
                record,
                image,
                page_no,
                parsing_res_list,
                page_result_data,
            )

        markdown = page_result.get("markdown")
        if not isinstance(markdown, dict):
            markdown = {}
            page_result["markdown"] = markdown
        existing_images = markdown.get("images")
        if not isinstance(existing_images, dict):
            existing_images = {}
        markdown["images"] = {**existing_images, **block_image_paths}
        if not markdown.get("text"):
            markdown["text"] = "\n\n".join(
                block.get("block_content", "")
                for block in parsing_res_list
                if block.get("block_content")
            )

        pruned_result = page_result.get("prunedResult") or {}
        page_width = int(pruned_result.get("width") or image.width)
        page_height = int(pruned_result.get("height") or image.height)
        return page_result, (page_width, page_height), markdown["images"]

    def _normalize_page_result(
        self,
        page_result_data: dict[str, Any],
        page_path: Path,
        image_width: int,
        image_height: int,
        pipeline_model: str | None = None,
        api_mode: str = "",
        fit_to_image_size: bool = False,
    ) -> dict[str, Any]:
        prediction_page = self._extract_prediction_page(page_result_data)
        pruned_result = prediction_page.get("prunedResult") or {}
        raw_blocks = pruned_result.get("parsing_res_list") or []
        source_page_width = int(pruned_result.get("width") or image_width)
        source_page_height = int(pruned_result.get("height") or image_height)
        normalized_page_width = (
            image_width
            if fit_to_image_size
            else int(pruned_result.get("width") or image_width)
        )
        normalized_page_height = (
            image_height
            if fit_to_image_size
            else int(pruned_result.get("height") or image_height)
        )

        normalized_blocks = []
        if isinstance(raw_blocks, list):
            for index, block_data in enumerate(raw_blocks, start=1):
                normalized = self._normalize_block_data(
                    block_data,
                    index,
                    source_page_width,
                    source_page_height,
                )
                if normalized is None:
                    continue
                if fit_to_image_size:
                    normalized = self._scale_block_to_page_size(
                        normalized,
                        source_page_width,
                        source_page_height,
                        normalized_page_width,
                        normalized_page_height,
                    )
                normalized_blocks.append(normalized)

        model_settings = pruned_result.get("model_settings")
        if not isinstance(model_settings, dict):
            model_settings = {}
        model_settings.setdefault(
            "pipeline_model",
            pipeline_model or self._current_result_pipeline_model(),
        )
        if api_mode:
            model_settings.setdefault("api_mode", api_mode)

        markdown = prediction_page.get("markdown")
        if not isinstance(markdown, dict):
            markdown = {}

        output_images = prediction_page.get("outputImages")
        if not isinstance(output_images, dict):
            output_images = {}

        return {
            "prunedResult": {
                "page_count": int(pruned_result.get("page_count") or 1),
                "width": normalized_page_width,
                "height": normalized_page_height,
                "model_settings": model_settings,
                "parsing_res_list": normalized_blocks,
            },
            "markdown": markdown,
            "outputImages": output_images,
            "inputImage": str(
                page_path.relative_to(self.data_manager.root_dir)
            ),
        }

    @staticmethod
    def _extract_prediction_page(
        page_result_data: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(page_result_data, dict):
            raise RuntimeError(
                "Invalid PPOCR pipeline response: expected object data"
            )
        layout_results = page_result_data.get("layoutParsingResults")
        if isinstance(layout_results, list):
            first_page = next(
                (
                    page_result
                    for page_result in layout_results
                    if isinstance(page_result, dict)
                ),
                None,
            )
            if first_page is None:
                raise RuntimeError("PPOCR pipeline returned no page results")
            return first_page
        if isinstance(page_result_data.get("prunedResult"), dict):
            return page_result_data
        raise RuntimeError(
            "Invalid PPOCR pipeline response: missing prunedResult or layoutParsingResults"
        )

    def _normalize_block_data(
        self,
        block_data: Any,
        index: int,
        image_width: int,
        image_height: int,
    ) -> dict[str, Any] | None:
        if not isinstance(block_data, dict):
            return None
        points = self._normalize_points(
            block_data.get("block_polygon_points")
            or block_data.get("points")
            or []
        )
        bbox = self._sanitize_bbox(
            block_data.get("block_bbox"),
            image_width,
            image_height,
        )
        if bbox is None and points:
            bbox = self._points_to_bbox(points, image_width, image_height)
        if bbox is None:
            bbox = [0, 0, max(1, image_width), max(1, image_height)]

        block_order = int(block_data.get("block_order") or index)
        group_id = int(block_data.get("group_id") or block_order)
        global_block_id = int(block_data.get("global_block_id") or block_order)
        global_group_id = int(block_data.get("global_group_id") or group_id)

        return {
            "block_label": normalize_block_label(
                block_data.get("block_label") or block_data.get("label")
            ),
            "block_content": str(
                block_data.get("block_content")
                or block_data.get("description")
                or ""
            ),
            "block_bbox": bbox,
            "block_id": int(block_data.get("block_id") or block_order),
            "block_order": block_order,
            "group_id": group_id,
            "global_block_id": global_block_id,
            "global_group_id": global_group_id,
            "block_polygon_points": points,
        }

    @staticmethod
    def _scale_block_to_page_size(
        block_data: dict[str, Any],
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
    ) -> dict[str, Any]:
        if source_width <= 0 or source_height <= 0:
            return block_data
        if source_width == target_width and source_height == target_height:
            return block_data
        scale_x = target_width / float(source_width)
        scale_y = target_height / float(source_height)
        bbox = block_data.get("block_bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            block_data["block_bbox"] = [
                max(
                    0, min(target_width, int(round(float(bbox[0]) * scale_x)))
                ),
                max(
                    0,
                    min(target_height, int(round(float(bbox[1]) * scale_y))),
                ),
                max(
                    0, min(target_width, int(round(float(bbox[2]) * scale_x)))
                ),
                max(
                    0,
                    min(target_height, int(round(float(bbox[3]) * scale_y))),
                ),
            ]
        points = block_data.get("block_polygon_points")
        if isinstance(points, list):
            scaled_points = []
            for point in points:
                if not isinstance(point, list) or len(point) < 2:
                    continue
                scaled_points.append(
                    [
                        max(
                            0,
                            min(
                                target_width,
                                int(round(float(point[0]) * scale_x)),
                            ),
                        ),
                        max(
                            0,
                            min(
                                target_height,
                                int(round(float(point[1]) * scale_y)),
                            ),
                        ),
                    ]
                )
            block_data["block_polygon_points"] = scaled_points
        return block_data

    def _collect_block_image_paths(
        self,
        record: PPOCRFileRecord,
        page_image: Image.Image,
        page_no: int,
        parsing_res_list: list[dict[str, Any]],
        page_result_data: dict[str, Any],
    ) -> dict[str, str]:
        block_image_paths: dict[str, str] = {}
        server_images = self._resolve_server_images(page_result_data)
        seen_keys: dict[str, int] = {}

        for index, block_data in enumerate(parsing_res_list, start=1):
            block_key = build_unique_block_key(
                page_no,
                block_data,
                index - 1,
                seen_keys,
            )
            base_block_key = build_block_key(page_no, block_data, index - 1)
            block_label = normalize_block_label(block_data.get("block_label"))
            if block_label not in IMAGE_ONLY_LABELS:
                continue
            existing_path = server_images.get(block_key) or server_images.get(
                base_block_key
            )
            if existing_path:
                block_image_paths[block_key] = existing_path
                continue
            bbox = self._sanitize_bbox(
                block_data.get("block_bbox"),
                page_image.width,
                page_image.height,
            )
            if bbox is None:
                points = self._normalize_points(
                    block_data.get("block_polygon_points") or []
                )
                if points:
                    bbox = self._points_to_bbox(
                        points,
                        page_image.width,
                        page_image.height,
                    )
            if bbox is None:
                continue
            crop_image = self._crop_image(page_image, bbox).convert("RGB")
            if crop_image.width <= 0 or crop_image.height <= 0:
                continue
            relative_path = self._save_block_crop(
                record,
                crop_image,
                page_no,
                index,
            )
            block_image_paths[block_key] = relative_path
        return block_image_paths

    def _resolve_server_images(
        self,
        page_result_data: dict[str, Any],
    ) -> dict[str, str]:
        server_images: dict[str, str] = {}
        if not isinstance(page_result_data, dict):
            return server_images
        image_maps = [
            page_result_data.get("block_image_paths"),
            (page_result_data.get("_ppocr_meta") or {}).get(
                "block_image_paths"
            ),
        ]
        for image_map in image_maps:
            if not isinstance(image_map, dict):
                continue
            for key, value in image_map.items():
                if not isinstance(key, str):
                    continue
                resolved_path = self._resolve_local_asset_path(value)
                if not resolved_path:
                    continue
                server_images[key] = resolved_path
        return server_images

    def _resolve_local_asset_path(self, path_value: Any) -> str:
        if not path_value:
            return ""
        candidate = Path(str(path_value))
        if not candidate.is_absolute():
            candidate = self.data_manager.root_dir / candidate
        if not candidate.exists() or not candidate.is_file():
            return ""
        try:
            return str(candidate.relative_to(self.data_manager.root_dir))
        except ValueError:
            return str(candidate)

    @staticmethod
    def _normalize_auth_schemes(raw_schemes: Any) -> list[str]:
        if isinstance(raw_schemes, str):
            return [
                item.strip() for item in raw_schemes.split(",") if item.strip()
            ]
        if isinstance(raw_schemes, list):
            return [
                str(item).strip() for item in raw_schemes if str(item).strip()
            ]
        return []

    def _resolve_auth_schemes(
        self,
        params: dict[str, Any],
        api_key: str,
    ) -> list[str]:
        schemes = self._normalize_auth_schemes(params.get("auth_schemes"))
        if not schemes:
            schemes = self._normalize_auth_schemes(params.get("auth_scheme"))
        if not schemes:
            schemes = list(self.auth_schemes)
        if api_key:
            lowered = {item.casefold() for item in schemes}
            if "token" not in lowered:
                schemes.append("token")
            if "bearer" not in lowered:
                schemes.append("bearer")
        return [item for item in schemes if item] or ["token"]

    def _build_api_headers_candidates(
        self,
        api_key: str,
        auth_schemes: list[str],
    ) -> list[dict[str, str]]:
        base_headers = {
            "Content-Type": "application/json",
            "Client-Platform": self.client_platform,
        }
        if not api_key:
            return [base_headers]
        candidates: list[dict[str, str]] = []
        used: set[str] = set()
        for scheme in auth_schemes:
            key = scheme.casefold()
            if key in used:
                continue
            used.add(key)
            headers = dict(base_headers)
            headers["Authorization"] = f"{scheme} {api_key}"
            candidates.append(headers)
        return candidates or [base_headers]

    def _build_ppocr_api_payload(
        self,
        image_b64: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"file": image_b64}
        payload.update(self.default_api_payload)
        for key in _SUPPORTED_API_PARAM_KEYS:
            if key in params:
                payload[key] = params[key]
        if not payload.get("useLayoutDetection", True):
            prompt_label = str(payload.get("promptLabel") or "ocr").strip()
            payload["promptLabel"] = prompt_label or "ocr"
        else:
            payload.pop("promptLabel", None)
        return payload

    def _build_ppocr_async_optional_payload(
        self,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = dict(self.default_api_payload)
        payload.pop("fileType", None)
        params = params or {}
        for key in _SUPPORTED_API_PARAM_KEYS:
            if key == "fileType" or key not in params:
                continue
            payload[key] = params[key]
        if not payload.get("useLayoutDetection", True):
            prompt_label = str(payload.get("promptLabel") or "ocr").strip()
            payload["promptLabel"] = prompt_label or "ocr"
        else:
            payload.pop("promptLabel", None)
        return payload

    def _api_job_headers(self, api_key: str) -> dict[str, str]:
        headers = {"Client-Platform": self.client_platform}
        if api_key:
            headers["Authorization"] = f"bearer {api_key}"
        return headers

    def _predict_ppocr_api_async_job(
        self,
        record: PPOCRFileRecord,
        page_paths: list[Path],
        cancel_event: threading.Event | None = None,
        progress_callback=None,
        index: int = 1,
        total: int = 1,
    ) -> dict[str, Any]:
        page_total = max(1, len(page_paths))
        if progress_callback is not None:
            progress_callback(
                PPOCRParsingProgress(
                    filename=record.filename,
                    index=index,
                    total=total,
                    page_no=1,
                    page_total=page_total,
                )
            )
        job_id = self._submit_ppocr_job(record, cancel_event=cancel_event)
        job_data = self._poll_ppocr_job(
            job_id,
            record,
            page_total,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            index=index,
            total=total,
        )
        json_url = self._extract_job_json_url(job_data)
        result_text = self._download_ppocr_job_result(
            json_url,
            cancel_event=cancel_event,
        )
        result_data = self._parse_ppocr_jsonl_result(result_text)
        return self._normalize_document_result(
            record,
            page_paths,
            result_data,
            job_id,
        )

    def _submit_ppocr_job(
        self,
        record: PPOCRFileRecord,
        cancel_event: threading.Event | None = None,
    ) -> str:
        api_key = str(self.api_key or "").strip()
        if not api_key:
            raise RuntimeError("API_KEY is empty.")
        data = {
            "model": self.api_model,
            "optionalPayload": json.dumps(
                self._build_ppocr_async_optional_payload(),
                ensure_ascii=False,
            ),
        }
        with open(record.source_path, "rb") as file_handle:
            files = {"file": (record.source_path.name, file_handle)}
            response = self._request_with_cancel(
                "post",
                PPOCR_API_JOB_URL,
                timeout_seconds=self.timeout,
                cancel_event=cancel_event,
                headers=self._api_job_headers(api_key),
                data=data,
                files=files,
            )
        self._raise_for_ppocr_response(
            response,
            "PPOCR API job submission failed",
        )
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("PPOCR API job response is invalid.")
        self._raise_for_ppocr_payload(payload)
        job_id = self._extract_job_id(payload)
        if not job_id:
            raise RuntimeError("PPOCR API job response missing job id.")
        return job_id

    def _poll_ppocr_job(
        self,
        job_id: str,
        record: PPOCRFileRecord,
        page_total: int,
        cancel_event: threading.Event | None = None,
        progress_callback=None,
        index: int = 1,
        total: int = 1,
    ) -> dict[str, Any]:
        poll_url = f"{PPOCR_API_JOB_URL.rstrip('/')}/{job_id}"
        timeout_seconds = max(1, int(self.timeout or 1))
        deadline = time.monotonic() + timeout_seconds
        interval = min(5.0, max(1.0, timeout_seconds / 60.0))
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Parsing cancelled")
            response = self._request_with_cancel(
                "get",
                poll_url,
                timeout_seconds=min(timeout_seconds, 30),
                cancel_event=cancel_event,
                headers=self._api_job_headers(str(self.api_key or "").strip()),
            )
            self._raise_for_ppocr_response(
                response,
                "PPOCR API job polling failed",
            )
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(
                    "PPOCR API job polling response is invalid."
                )
            self._raise_for_ppocr_payload(payload)
            job_data = self._extract_job_payload(payload)
            self._emit_job_progress(
                job_data,
                record,
                page_total,
                progress_callback,
                index,
                total,
            )
            state = str(
                job_data.get("state") or job_data.get("status") or ""
            ).casefold()
            if state in {
                "done",
                "completed",
                "complete",
                "success",
                "succeeded",
            }:
                return job_data
            if state in {
                "failed",
                "failure",
                "error",
                "cancelled",
                "canceled",
            }:
                raise RuntimeError(self._extract_job_error(job_data))
            if self._extract_job_json_url(job_data, required=False):
                return job_data
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"PPOCR API job {job_id} did not finish within "
                    f"{timeout_seconds} seconds."
                )
            if cancel_event is None:
                time.sleep(interval)
            elif cancel_event.wait(interval):
                raise RuntimeError("Parsing cancelled")

    def _download_ppocr_job_result(
        self,
        json_url: str,
        cancel_event: threading.Event | None = None,
    ) -> str:
        response = self._request_with_cancel(
            "get",
            json_url,
            timeout_seconds=self.timeout,
            cancel_event=cancel_event,
        )
        self._raise_for_ppocr_response(
            response,
            "PPOCR API job result download failed",
        )
        return response.text

    def _normalize_document_result(
        self,
        record: PPOCRFileRecord,
        page_paths: list[Path],
        result_data: dict[str, Any],
        job_id: str,
    ) -> dict[str, Any]:
        raw_layout_results = result_data.get("layoutParsingResults") or []
        if not isinstance(raw_layout_results, list) or not raw_layout_results:
            raise RuntimeError(
                "PPOCR API job result contains no page results."
            )

        layout_results = []
        page_sizes = []
        block_image_paths = {}
        fallback_page_path = (
            page_paths[-1] if page_paths else record.source_path
        )
        for page_index, page_result_data in enumerate(
            raw_layout_results,
            start=1,
        ):
            page_path = (
                page_paths[page_index - 1]
                if page_index <= len(page_paths)
                else fallback_page_path
            )
            with Image.open(page_path) as image:
                image = ImageOps.exif_transpose(image)
                if image.mode in {"RGBA", "LA"} or (
                    image.mode == "P" and "transparency" in image.info
                ):
                    rgba_image = image.convert("RGBA")
                    background = Image.new(
                        "RGBA",
                        rgba_image.size,
                        (255, 255, 255, 255),
                    )
                    image = Image.alpha_composite(
                        background,
                        rgba_image,
                    ).convert("RGB")
                else:
                    image = image.convert("RGB")
                page_result = self._normalize_page_result(
                    page_result_data,
                    page_path,
                    image.width,
                    image.height,
                    pipeline_model=self.api_model,
                    api_mode=PPOCR_API_MODE_ASYNC_JOBS,
                    fit_to_image_size=True,
                )
                parsing_res_list = (page_result.get("prunedResult") or {}).get(
                    "parsing_res_list"
                ) or []
                page_block_image_paths = self._collect_block_image_paths(
                    record,
                    image,
                    page_index,
                    parsing_res_list,
                    page_result_data,
                )

            markdown = page_result.get("markdown")
            if not isinstance(markdown, dict):
                markdown = {}
                page_result["markdown"] = markdown
            existing_images = markdown.get("images")
            if not isinstance(existing_images, dict):
                existing_images = {}
            markdown["images"] = {
                **existing_images,
                **page_block_image_paths,
            }
            if not markdown.get("text"):
                markdown["text"] = "\n\n".join(
                    block.get("block_content", "")
                    for block in parsing_res_list
                    if block.get("block_content")
                )

            pruned_result = page_result.get("prunedResult") or {}
            page_width = int(pruned_result.get("width") or 0)
            page_height = int(pruned_result.get("height") or 0)
            layout_results.append(page_result)
            page_sizes.append({"width": page_width, "height": page_height})
            block_image_paths.update(page_block_image_paths)

        preprocessed_images = result_data.get("preprocessedImages")
        if not isinstance(preprocessed_images, list):
            preprocessed_images = []
        document_data = {
            "layoutParsingResults": layout_results,
            "preprocessedImages": preprocessed_images,
            "dataInfo": {
                "type": record.file_type,
                "numPages": len(layout_results),
                "pages": page_sizes,
            },
            "_ppocr_meta": {
                "status": "parsed",
                "source_path": f"files/{record.filename}",
                "updated_at": "",
                "error_message": "",
                "edited_blocks": [],
                "block_image_paths": block_image_paths,
                "pipeline_model": self.api_model,
                "api_mode": PPOCR_API_MODE_ASYNC_JOBS,
                "api_job_id": job_id,
            },
        }
        self.data_manager.save_record_data(record, document_data)
        return document_data

    @staticmethod
    def _extract_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
        for key in ("result", "data", "job"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        return payload

    @classmethod
    def _extract_job_id(cls, payload: dict[str, Any]) -> str:
        candidates = [payload, cls._extract_job_payload(payload)]
        for candidate in candidates:
            for key in ("jobId", "job_id", "id", "taskId", "task_id"):
                value = candidate.get(key)
                if value:
                    return str(value).strip()
        return ""

    @staticmethod
    def _extract_job_json_url(
        job_data: dict[str, Any],
        required: bool = True,
    ) -> str:
        containers = [job_data]
        for key in ("resultUrl", "result_url", "result"):
            value = job_data.get(key)
            if isinstance(value, dict):
                containers.append(value)
            elif isinstance(value, str) and value.strip():
                return value.strip()
        for container in containers:
            for key in ("jsonUrl", "json_url", "url", "downloadUrl"):
                value = container.get(key)
                if value:
                    return str(value).strip()
        if required:
            raise RuntimeError("PPOCR API job result missing json url.")
        return ""

    @staticmethod
    def _extract_job_error(job_data: dict[str, Any]) -> str:
        error = job_data.get("error") or {}
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            if message:
                return message
        message = str(
            job_data.get("errorMsg")
            or job_data.get("error_message")
            or job_data.get("message")
            or "PPOCR API job failed"
        ).strip()
        return message or "PPOCR API job failed"

    @staticmethod
    def _int_or_default(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _emit_job_progress(
        self,
        job_data: dict[str, Any],
        record: PPOCRFileRecord,
        page_total: int,
        progress_callback,
        index: int,
        total: int,
    ) -> None:
        if progress_callback is None:
            return
        progress = job_data.get("extractProgress") or job_data.get(
            "extract_progress"
        )
        total_pages = page_total
        extracted_pages = 1
        if isinstance(progress, dict):
            total_pages = self._int_or_default(
                progress.get("totalPages") or progress.get("total_pages"),
                page_total,
            )
            extracted_pages = self._int_or_default(
                progress.get("extractedPages")
                or progress.get("extracted_pages"),
                1,
            )
        total_pages = max(1, total_pages)
        extracted_pages = max(1, min(total_pages, extracted_pages))
        progress_callback(
            PPOCRParsingProgress(
                filename=record.filename,
                index=index,
                total=total,
                page_no=extracted_pages,
                page_total=total_pages,
            )
        )

    @staticmethod
    def _parse_ppocr_jsonl_result(text: str) -> dict[str, Any]:
        stripped = str(text or "").strip()
        if not stripped:
            raise RuntimeError("PPOCR API job result is empty.")
        records: list[Any]
        try:
            parsed = json.loads(stripped)
            records = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            records = []
            for line_no, line in enumerate(stripped.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"PPOCR API job result jsonl line {line_no} is invalid."
                    ) from exc

        layout_results = []
        preprocessed_images = []
        data_info: dict[str, Any] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            result = record.get("result") if "result" in record else record
            if not isinstance(result, dict):
                continue
            pages = result.get("layoutParsingResults")
            if isinstance(pages, list):
                layout_results.extend(
                    page for page in pages if isinstance(page, dict)
                )
            elif isinstance(result.get("prunedResult"), dict):
                layout_results.append(result)
            images = result.get("preprocessedImages")
            if isinstance(images, list):
                preprocessed_images.extend(images)
            result_data_info = result.get("dataInfo")
            if isinstance(result_data_info, dict):
                data_info.update(result_data_info)
        if not layout_results:
            raise RuntimeError(
                "PPOCR API job result contains no page results."
            )
        return {
            "layoutParsingResults": layout_results,
            "preprocessedImages": preprocessed_images,
            "dataInfo": data_info,
        }

    def _resolve_server_predict_url(self) -> str:
        if self.service_probe.is_online and self.service_probe.server_url:
            return (
                f"{str(self.service_probe.server_url).rstrip('/')}/v1/predict"
            )
        server_url = str(self.server_url or "").strip().rstrip("/")
        if not server_url:
            raise RuntimeError("Server URL is empty.")
        if server_url.endswith("/predict"):
            return server_url
        base_urls = self._candidate_server_base_urls()
        if base_urls:
            for base_url in base_urls:
                normalized = base_url.rstrip("/")
                if normalized.endswith("/v1/models"):
                    normalized = normalized[: -len("/v1/models")]
                if normalized.endswith("/predict"):
                    return normalized
                if normalized:
                    return f"{normalized}/v1/predict"
        return f"{server_url}/v1/predict"

    def _predict_ppocr_api(
        self,
        image: Image.Image,
        params: dict[str, Any],
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        api_url = str(self.api_url or "").strip()
        if not api_url:
            raise RuntimeError("API_URL is empty.")
        api_key = str(
            params.get("api_key") or params.get("token") or self.api_key or ""
        ).strip()
        if not api_key:
            raise RuntimeError("API_KEY is empty.")

        auth_schemes = self._resolve_auth_schemes(params, api_key)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        payload = self._build_ppocr_api_payload(
            base64.b64encode(buffer.getvalue()).decode("utf-8"),
            params,
        )
        headers_candidates = self._build_api_headers_candidates(
            api_key=api_key,
            auth_schemes=auth_schemes,
        )

        response = None
        for index, headers in enumerate(headers_candidates):
            response = self._post_json_with_cancel(
                api_url,
                payload=payload,
                headers=headers,
                timeout_seconds=self.timeout,
                cancel_event=cancel_event,
            )
            if response.status_code != 401:
                break
            if index < len(headers_candidates) - 1:
                logger.warning(
                    "PPOCR API unauthorized with current auth scheme, retrying with next scheme."
                )
        if response is None:
            raise RuntimeError("PPOCR API request failed: empty response")
        if response.status_code == 401:
            raise RuntimeError(
                "PPOCR API unauthorized (401). Please set a valid API_KEY."
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            message = ""
            try:
                error_data = response.json()
                message = str(
                    error_data.get("errorMsg")
                    or error_data.get("message")
                    or ""
                ).strip()
            except Exception:
                message = str(response.text or "").strip()
            detail = f" ({message})" if message else ""
            raise RuntimeError(
                f"PPOCR API request failed with status={response.status_code}{detail}"
            ) from exc

        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("PPOCR API response is invalid.")

        if "result" in data:
            error_code = data.get("errorCode", -1)
            try:
                error_code_value = int(error_code)
            except (TypeError, ValueError):
                error_code_value = -1
            if error_code_value != 0:
                raise RuntimeError(
                    str(
                        data.get("errorMsg")
                        or data.get("message")
                        or "PPOCR API failed"
                    )
                )
            result = data.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("PPOCR API response missing result object")
            return result

        if "success" in data:
            if not data.get("success", True):
                error = data.get("error") or {}
                message = error.get("message") or "Remote inference failed"
                raise RuntimeError(str(message))
            payload_data = data.get("data") or {}
            if isinstance(payload_data, dict):
                return payload_data
            raise RuntimeError("Remote inference returned invalid data")

        if isinstance(data.get("layoutParsingResults"), list) or isinstance(
            data.get("prunedResult"), dict
        ):
            return data
        raise RuntimeError("PPOCR API response missing result object")

    def _predict_remote(
        self,
        model_id: str,
        image: Image.Image,
        params: dict[str, Any],
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        if not model_id:
            raise RuntimeError("No PPOCR pipeline model selected")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        payload = {
            "model": model_id,
            "image": f"data:image/png;base64,{image_b64}",
            "params": params,
        }
        predict_url = self._resolve_server_predict_url()
        response = self._post_json_with_cancel(
            predict_url,
            payload=payload,
            headers=self._server_headers(),
            timeout_seconds=self.timeout,
            cancel_event=cancel_event,
        )
        response.raise_for_status()
        response_data = response.json()
        if not response_data.get("success", True):
            error = response_data.get("error") or {}
            message = error.get("message") or "Remote inference failed"
            raise RuntimeError(message)
        return response_data.get("data") or {}

    @staticmethod
    def _raise_for_ppocr_payload(payload: dict[str, Any]) -> None:
        if "errorCode" not in payload:
            return
        error_code = payload.get("errorCode", 0)
        try:
            error_code_value = int(error_code)
        except (TypeError, ValueError):
            error_code_value = -1
        if error_code_value == 0:
            return
        raise RuntimeError(
            str(
                payload.get("errorMsg")
                or payload.get("message")
                or "PPOCR API failed"
            )
        )

    @staticmethod
    def _raise_for_ppocr_response(response, message_prefix: str) -> None:
        if response.status_code == 401:
            raise RuntimeError(
                "PPOCR API unauthorized (401). Please set a valid API_KEY."
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            message = ""
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    message = str(
                        error_data.get("errorMsg")
                        or error_data.get("message")
                        or ""
                    ).strip()
            except Exception:
                message = str(response.text or "").strip()
            detail = f" ({message})" if message else ""
            raise RuntimeError(
                f"{message_prefix} with status={response.status_code}{detail}"
            ) from exc

    @staticmethod
    def _request_with_cancel(
        method: str,
        url: str,
        timeout_seconds: int,
        cancel_event: threading.Event | None = None,
        **kwargs,
    ):
        timeout_value = max(1, int(timeout_seconds or 1))
        if cancel_event is None:
            return requests.request(
                method,
                url,
                timeout=timeout_value,
                **kwargs,
            )
        if cancel_event.is_set():
            raise RuntimeError("Parsing cancelled")

        result_holder: dict[str, Any] = {}
        error_holder: dict[str, Exception] = {}
        completed = threading.Event()

        def _runner() -> None:
            try:
                result_holder["response"] = requests.request(
                    method,
                    url,
                    timeout=timeout_value,
                    **kwargs,
                )
            except Exception as exc:  # pragma: no cover - requests path
                error_holder["error"] = exc
            finally:
                completed.set()

        request_thread = threading.Thread(target=_runner, daemon=True)
        request_thread.start()
        while not completed.wait(0.1):
            if cancel_event.is_set():
                raise RuntimeError("Parsing cancelled")
        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder["response"]

    @staticmethod
    def _post_json_with_cancel(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: int,
        cancel_event: threading.Event | None = None,
    ):
        return PPOCRPipeline._request_with_cancel(
            "post",
            url,
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
            json=payload,
            headers=headers,
        )

    @staticmethod
    def _normalize_points(points: list[Any]) -> list[list[int]]:
        normalized = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            normalized.append(
                [
                    int(round(float(point[0]))),
                    int(round(float(point[1]))),
                ]
            )
        return normalized

    @staticmethod
    def _sanitize_bbox(
        bbox: Any,
        image_width: int,
        image_height: int,
    ) -> list[int] | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        try:
            x1 = int(round(float(bbox[0])))
            y1 = int(round(float(bbox[1])))
            x2 = int(round(float(bbox[2])))
            y2 = int(round(float(bbox[3])))
        except Exception:
            return None
        x1 = max(0, min(image_width, x1))
        y1 = max(0, min(image_height, y1))
        x2 = max(0, min(image_width, x2))
        y2 = max(0, min(image_height, y2))
        if x2 <= x1:
            x2 = min(image_width, x1 + 1)
        if y2 <= y1:
            y2 = min(image_height, y1 + 1)
        return [x1, y1, x2, y2]

    @staticmethod
    def _points_to_bbox(
        points: list[list[int]],
        image_width: int,
        image_height: int,
    ) -> list[int]:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1 = max(0, min(xs))
        y1 = max(0, min(ys))
        x2 = min(image_width, max(xs))
        y2 = min(image_height, max(ys))
        if x2 <= x1:
            x2 = min(image_width, x1 + 1)
        if y2 <= y1:
            y2 = min(image_height, y1 + 1)
        return [x1, y1, x2, y2]

    @staticmethod
    def _crop_image(image: Image.Image, bbox: list[int]) -> Image.Image:
        return image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

    def _save_block_crop(
        self,
        record: PPOCRFileRecord,
        image: Image.Image,
        page_no: int,
        block_index: int,
    ) -> str:
        block_dir = self.data_manager.get_block_images_dir(record.filename)
        block_dir.mkdir(parents=True, exist_ok=True)
        image_path = (
            block_dir / f"page_{page_no:03d}_block_{block_index:04d}.png"
        )
        image.save(image_path, format="PNG")
        return str(image_path.relative_to(self.data_manager.root_dir))


class PPOCRPipelineWorker(QObject):
    progressChanged = pyqtSignal(object)
    recordStarted = pyqtSignal(str)
    recordFinished = pyqtSignal(str, object)
    recordFailed = pyqtSignal(str, str)
    batchFinished = pyqtSignal()
    batchCancelled = pyqtSignal()

    def __init__(
        self,
        pipeline: PPOCRPipeline,
        records: list[PPOCRFileRecord],
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.records = records
        self.cancel_event = cancel_event

    def run(self) -> None:
        total = max(1, len(self.records))
        active_record: PPOCRFileRecord | None = None
        try:
            for index, record in enumerate(self.records, start=1):
                active_record = record
                if self.cancel_event.is_set():
                    self.batchCancelled.emit()
                    return
                self.recordStarted.emit(record.filename)
                try:
                    document_data = self.pipeline.parse_record(
                        record,
                        cancel_event=self.cancel_event,
                        progress_callback=self.progressChanged.emit,
                        index=index,
                        total=total,
                    )
                    if self.cancel_event.is_set():
                        self.pipeline.data_manager.reset_to_pending(record)
                        self.batchCancelled.emit()
                        return
                    self.recordFinished.emit(record.filename, document_data)
                    active_record = None
                except Exception as exc:
                    if self.cancel_event.is_set():
                        self.pipeline.data_manager.reset_to_pending(record)
                        self.batchCancelled.emit()
                        return
                    error_message = str(exc)
                    try:
                        self.pipeline.data_manager.mark_error(
                            record,
                            error_message,
                        )
                    except Exception as mark_exc:
                        error_message = (
                            f"{error_message} (mark_error failed: {mark_exc})"
                        )
                    self.recordFailed.emit(record.filename, error_message)
                    active_record = None
            self.batchFinished.emit()
        except Exception as exc:
            if self.cancel_event.is_set():
                self.batchCancelled.emit()
                return
            if active_record is not None:
                error_message = f"Unexpected worker error: {exc}"
                try:
                    self.pipeline.data_manager.mark_error(
                        active_record,
                        error_message,
                    )
                except Exception:
                    pass
                self.recordFailed.emit(active_record.filename, error_message)
            self.batchFinished.emit()
