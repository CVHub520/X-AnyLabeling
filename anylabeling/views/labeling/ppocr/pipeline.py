from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import threading
from typing import Any

from PIL import Image, ImageOps
from PyQt6.QtCore import QObject, pyqtSignal
import requests

from anylabeling.views.labeling.logger import logger

from .config import (
    PPOCR_FILE_TYPE_IMAGE,
    PPOCR_OFFLINE_MODEL_LABEL,
    PPOCR_PIPELINE_CAPABILITY_KEY,
    PPOCRPipelineModel,
    PPOCRServiceProbe,
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
        self.server_url = remote_settings.get(
            "server_url",
            "http://127.0.0.1:8000",
        )
        self.api_key = remote_settings.get("api_key", "") or ""
        self.timeout = int(remote_settings.get("timeout", 180) or 180)
        self.pipeline_model = ""
        self.pipeline_models: list[PPOCRPipelineModel] = []
        self.service_probe = PPOCRServiceProbe(
            is_online=False,
            server_url=self.server_url,
            pipeline_model=PPOCR_OFFLINE_MODEL_LABEL,
            pipeline_models=tuple(),
            error_message="Service probing has not run yet.",
        )

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
        headers = {"Token": self.api_key}
        models_url = f"{self.server_url.rstrip('/')}/v1/models"
        try:
            response = requests.get(
                models_url,
                headers=headers,
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
            self.pipeline_model = self._select_pipeline_model(pipeline_models)
            self.service_probe = PPOCRServiceProbe(
                is_online=True,
                server_url=self.server_url,
                pipeline_model=self.pipeline_model,
                pipeline_models=tuple(self.pipeline_models),
                error_message="",
            )
            return self.service_probe
        except Exception as exc:
            logger.warning(f"Failed to probe PaddleOCR service: {exc}")
            self.service_probe = PPOCRServiceProbe(
                is_online=False,
                server_url=self.server_url,
                pipeline_model=PPOCR_OFFLINE_MODEL_LABEL,
                pipeline_models=tuple(self.pipeline_models),
                error_message=str(exc),
            )
            return self.service_probe

    def set_pipeline_model(self, model_id: str) -> None:
        if model_id and model_id != PPOCR_OFFLINE_MODEL_LABEL:
            self.pipeline_model = model_id

    def parse_record(
        self,
        record: PPOCRFileRecord,
        cancel_event: threading.Event | None = None,
        progress_callback=None,
        index: int = 1,
        total: int = 1,
    ) -> dict[str, Any]:
        service_probe = self.probe_service()
        if not service_probe.is_online:
            raise RuntimeError(
                service_probe.error_message or "PaddleOCR service is offline"
            )
        if record.file_type == PPOCR_FILE_TYPE_IMAGE:
            page_paths = [record.source_path]
        else:
            page_paths = self.data_manager.ensure_pdf_pages(record, force=True)
        self.data_manager.clear_block_images(record)

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
                "pipeline_model": self.pipeline_model,
            },
        }
        self.data_manager.save_record_data(record, document_data)
        return document_data

    def _parse_page(
        self,
        record: PPOCRFileRecord,
        page_path: Path,
        page_no: int,
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
            page_result_data = self._predict_remote(
                self.pipeline_model,
                image,
                params={
                    "page_no": page_no,
                    "source_file_type": record.file_type,
                },
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
    ) -> dict[str, Any]:
        prediction_page = self._extract_prediction_page(page_result_data)
        pruned_result = prediction_page.get("prunedResult") or {}
        raw_blocks = pruned_result.get("parsing_res_list") or []

        normalized_blocks = []
        if isinstance(raw_blocks, list):
            for index, block_data in enumerate(raw_blocks, start=1):
                normalized = self._normalize_block_data(
                    block_data,
                    index,
                    image_width,
                    image_height,
                )
                if normalized is None:
                    continue
                normalized_blocks.append(normalized)

        normalized_page_width = int(pruned_result.get("width") or image_width)
        normalized_page_height = int(
            pruned_result.get("height") or image_height
        )

        model_settings = pruned_result.get("model_settings")
        if not isinstance(model_settings, dict):
            model_settings = {}
        model_settings.setdefault("pipeline_model", self.pipeline_model)

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

    def _predict_remote(
        self,
        model_id: str,
        image: Image.Image,
        params: dict[str, Any],
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
        response = requests.post(
            f"{self.server_url.rstrip('/')}/v1/predict",
            json=payload,
            headers={"Token": self.api_key},
            timeout=self.timeout,
        )
        response.raise_for_status()
        response_data = response.json()
        if not response_data.get("success", True):
            error = response_data.get("error") or {}
            message = error.get("message") or "Remote inference failed"
            raise RuntimeError(message)
        return response_data.get("data") or {}

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
        for index, record in enumerate(self.records, start=1):
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
            except Exception as exc:
                if self.cancel_event.is_set():
                    self.pipeline.data_manager.reset_to_pending(record)
                    self.batchCancelled.emit()
                    return
                error_message = str(exc)
                self.pipeline.data_manager.mark_error(record, error_message)
                self.recordFailed.emit(record.filename, error_message)
        self.batchFinished.emit()
