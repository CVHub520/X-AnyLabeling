from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from typing import Any

from PIL import Image, ImageOps
from PyQt6.QtCore import QSize

try:
    from PyQt6.QtPdf import QPdfDocument
except Exception:
    QPdfDocument = None

from anylabeling.config import get_work_directory
from anylabeling.views.labeling.logger import logger

from .config import (
    PPOCR_BLOCK_IMAGES_DIR_PREFIX,
    PPOCR_FILE_TYPE_ALL,
    PPOCR_FILE_TYPE_IMAGE,
    PPOCR_FILE_TYPE_PDF,
    PPOCR_FILES_DIRNAME,
    PPOCR_JSONS_DIRNAME,
    PPOCR_PDF_DIR_PREFIX,
    PPOCR_ROOT_DIRNAME,
    PPOCR_SORT_NEWEST,
    PPOCR_STATUS_ERROR,
    PPOCR_STATUS_PARSED,
    PPOCR_STATUS_PENDING,
    PPOCR_SUPPORTED_IMAGE_SUFFIXES,
    PPOCR_SUPPORTED_PDF_SUFFIXES,
    PPOCR_SUPPORTED_SUFFIXES,
)
from .render import build_unique_block_key


@dataclass
class PPOCRFileRecord:
    filename: str
    source_path: Path
    json_path: Path
    file_type: str
    status: str
    mtime: float
    timestamp: str
    size_bytes: int
    page_count: int = 1
    error_message: str = ""
    is_parsing: bool = False


class PPOCRDataManager:
    def __init__(self, work_dir: str | None = None) -> None:
        base_dir = Path(work_dir or get_work_directory()).expanduser()
        self.root_dir = base_dir / PPOCR_ROOT_DIRNAME
        self.files_dir = self.root_dir / PPOCR_FILES_DIRNAME
        self.jsons_dir = self.root_dir / PPOCR_JSONS_DIRNAME
        self.api_settings_path = self.root_dir / "api_settings.json"
        self.state_path = self.root_dir / "ui_state.json"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.jsons_dir.mkdir(parents=True, exist_ok=True)

    def load_api_settings(self) -> dict[str, str]:
        if not self.api_settings_path.exists():
            return {
                "api_url": "",
                "api_key": "",
            }
        try:
            with open(self.api_settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(
                f"Failed to load PaddleOCR API settings: {self.api_settings_path}, {exc}"
            )
            return {
                "api_url": "",
                "api_key": "",
            }
        if not isinstance(data, dict):
            return {
                "api_url": "",
                "api_key": "",
            }
        return {
            "api_url": str(data.get("api_url") or "").strip(),
            "api_key": str(data.get("api_key") or "").strip(),
        }

    def save_api_settings(
        self,
        api_url: str,
        api_key: str,
    ) -> None:
        payload = {
            "api_url": str(api_url or "").strip(),
            "api_key": str(api_key or "").strip(),
        }
        with open(self.api_settings_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def scan_records(self) -> list[PPOCRFileRecord]:
        records = []
        for source_path in sorted(
            self.files_dir.iterdir(), key=lambda p: p.name
        ):
            if not source_path.is_file():
                continue
            file_type = self.get_file_type(source_path)
            if file_type is None:
                continue
            json_path = self.get_json_path(source_path.name)
            json_data = (
                self.load_json_data(json_path) if json_path.exists() else {}
            )
            meta = json_data.get("_ppocr_meta") or {}
            status = str(meta.get("status") or "")
            if not status:
                status = (
                    PPOCR_STATUS_PARSED
                    if json_path.exists()
                    else PPOCR_STATUS_PENDING
                )
            if status not in {
                PPOCR_STATUS_PENDING,
                PPOCR_STATUS_PARSED,
                PPOCR_STATUS_ERROR,
            }:
                status = PPOCR_STATUS_PENDING
            page_count = self.get_page_count(source_path, json_data)
            mtime = source_path.stat().st_mtime
            records.append(
                PPOCRFileRecord(
                    filename=source_path.name,
                    source_path=source_path,
                    json_path=json_path,
                    file_type=file_type,
                    status=status,
                    mtime=mtime,
                    timestamp=datetime.fromtimestamp(mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    size_bytes=source_path.stat().st_size,
                    page_count=page_count,
                    error_message=str(meta.get("error_message") or ""),
                )
            )
        return records

    def query_records(
        self,
        search_text: str = "",
        sort_mode: str = PPOCR_SORT_NEWEST,
        file_type: str = PPOCR_FILE_TYPE_ALL,
        status: str = PPOCR_FILE_TYPE_ALL,
    ) -> list[PPOCRFileRecord]:
        records = self.scan_records()
        if search_text:
            text = search_text.casefold()
            records = [
                record
                for record in records
                if text in record.filename.casefold()
            ]
        if file_type != PPOCR_FILE_TYPE_ALL:
            records = [
                record for record in records if record.file_type == file_type
            ]
        if status != PPOCR_FILE_TYPE_ALL:
            records = [record for record in records if record.status == status]
        records.sort(
            key=lambda record: record.mtime,
            reverse=(sort_mode == PPOCR_SORT_NEWEST),
        )
        return records

    def get_file_type(self, path: Path) -> str | None:
        suffix = path.suffix.lower()
        if suffix in PPOCR_SUPPORTED_IMAGE_SUFFIXES:
            return PPOCR_FILE_TYPE_IMAGE
        if suffix in PPOCR_SUPPORTED_PDF_SUFFIXES:
            return PPOCR_FILE_TYPE_PDF
        return None

    def get_json_path(self, filename: str) -> Path:
        return self.jsons_dir / f"{filename}.json"

    def get_pdf_pages_dir(self, filename: str) -> Path:
        return self.files_dir / f"{PPOCR_PDF_DIR_PREFIX}{Path(filename).stem}"

    def get_block_images_dir(self, filename: str) -> Path:
        return self.files_dir / f"{PPOCR_BLOCK_IMAGES_DIR_PREFIX}{filename}"

    def load_json_data(self, json_path: Path) -> dict[str, Any]:
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            logger.warning(
                f"Failed to load PaddleOCR json: {json_path}, {exc}"
            )
            return {}

    def load_record_data(self, record: PPOCRFileRecord) -> dict[str, Any]:
        if not record.json_path.exists():
            return {}
        return self.load_json_data(record.json_path)

    def save_record_data(
        self,
        record: PPOCRFileRecord,
        data: dict[str, Any],
        status: str = PPOCR_STATUS_PARSED,
        error_message: str = "",
    ) -> None:
        meta = data.setdefault("_ppocr_meta", {})
        meta["status"] = status
        meta["source_path"] = str(Path(PPOCR_FILES_DIRNAME) / record.filename)
        meta["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta["error_message"] = error_message
        meta.setdefault("edited_blocks", [])
        meta.setdefault("block_image_paths", {})
        with open(record.json_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def save_block_content(
        self,
        record: PPOCRFileRecord,
        page_no: int,
        block_uid: str,
        content: str,
    ) -> dict[str, Any]:
        data = self.load_record_data(record)
        layout_results = data.get("layoutParsingResults") or []
        if not 1 <= page_no <= len(layout_results):
            raise ValueError("Invalid page index")
        pruned_result = layout_results[page_no - 1].get("prunedResult") or {}
        block_list = pruned_result.get("parsing_res_list") or []
        matched = False
        block_key = ""
        seen_keys: dict[str, int] = {}
        for index, block_data in enumerate(block_list):
            block_key = build_unique_block_key(
                page_no,
                block_data,
                index,
                seen_keys,
            )
            if block_key == f"page_{page_no}:{block_uid}":
                block_data["block_content"] = content
                matched = True
                break
        if not matched:
            raise ValueError("Block not found")
        meta = data.setdefault("_ppocr_meta", {})
        edited_blocks = set(meta.get("edited_blocks") or [])
        edited_blocks.add(block_key)
        meta["edited_blocks"] = sorted(edited_blocks)
        self.save_record_data(record, data, status=PPOCR_STATUS_PARSED)
        return data

    def clear_error_and_edits(self, record: PPOCRFileRecord) -> None:
        if not record.json_path.exists():
            return
        data = self.load_record_data(record)
        meta = data.setdefault("_ppocr_meta", {})
        meta["status"] = PPOCR_STATUS_PARSED
        meta["error_message"] = ""
        meta["edited_blocks"] = []
        self.save_record_data(record, data, status=PPOCR_STATUS_PARSED)

    def mark_error(self, record: PPOCRFileRecord, error_message: str) -> None:
        data = (
            self.load_record_data(record)
            if record.json_path.exists()
            else {
                "layoutParsingResults": [],
                "preprocessedImages": [],
                "dataInfo": {
                    "type": record.file_type,
                    "numPages": record.page_count,
                    "pages": [],
                },
            }
        )
        self.save_record_data(
            record,
            data,
            status=PPOCR_STATUS_ERROR,
            error_message=error_message,
        )

    def reset_to_pending(self, record: PPOCRFileRecord) -> None:
        if record.json_path.exists():
            record.json_path.unlink()

    def import_files(
        self, source_paths: list[str]
    ) -> tuple[list[PPOCRFileRecord], list[str]]:
        imported_records = []
        errors = []
        for raw_path in source_paths:
            source_path = Path(raw_path)
            if source_path.is_dir():
                errors.append(
                    f"{source_path}: directory import is not supported"
                )
                continue
            if not source_path.is_file():
                errors.append(f"{source_path}: file does not exist")
                continue
            if source_path.suffix.lower() not in PPOCR_SUPPORTED_SUFFIXES:
                errors.append(f"{source_path.name}: unsupported file type")
                continue
            is_valid, message = self._validate_basename(source_path.name)
            if not is_valid:
                errors.append(f"{source_path.name}: {message}")
                continue
            target_name = self._deduplicated_filename(source_path.name)
            target_path = self.files_dir / target_name
            shutil.copy2(source_path, target_path)
            record = PPOCRFileRecord(
                filename=target_name,
                source_path=target_path,
                json_path=self.get_json_path(target_name),
                file_type=self.get_file_type(target_path)
                or PPOCR_FILE_TYPE_IMAGE,
                status=PPOCR_STATUS_PENDING,
                mtime=target_path.stat().st_mtime,
                timestamp=datetime.fromtimestamp(
                    target_path.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S"),
                size_bytes=target_path.stat().st_size,
                page_count=1,
            )
            imported_records.append(record)
        return imported_records, errors

    def delete_record(self, record: PPOCRFileRecord) -> None:
        self.set_favorite(record.filename, False)
        if record.source_path.exists():
            record.source_path.unlink()
        if record.json_path.exists():
            record.json_path.unlink()
        if record.file_type == PPOCR_FILE_TYPE_PDF:
            pdf_pages_dir = self.get_pdf_pages_dir(record.filename)
            if pdf_pages_dir.exists():
                shutil.rmtree(pdf_pages_dir)
        block_images_dir = self.get_block_images_dir(record.filename)
        if block_images_dir.exists():
            shutil.rmtree(block_images_dir)

    def get_preview_pages(self, record: PPOCRFileRecord) -> list[Path]:
        if record.file_type == PPOCR_FILE_TYPE_IMAGE:
            return [record.source_path]
        return self.ensure_pdf_pages(record)

    def ensure_pdf_pages(
        self,
        record: PPOCRFileRecord,
        force: bool = False,
    ) -> list[Path]:
        pages_dir = self.get_pdf_pages_dir(record.filename)
        if pages_dir.exists() and not force:
            page_paths = sorted(pages_dir.glob("page_*.png"))
            if page_paths:
                return page_paths
        if QPdfDocument is None:
            raise RuntimeError("QtPdf is not available")
        if pages_dir.exists():
            shutil.rmtree(pages_dir)
        pages_dir.mkdir(parents=True, exist_ok=True)
        document = QPdfDocument(None)
        error = document.load(str(record.source_path))
        if error != QPdfDocument.Error.None_:
            raise RuntimeError(f"Failed to load PDF: {record.filename}")
        page_paths = []
        for page_index in range(document.pageCount()):
            page_size = document.pagePointSize(page_index)
            width = max(1, int(round(page_size.width() * 2)))
            height = max(1, int(round(page_size.height() * 2)))
            rendered = document.render(page_index, QSize(width, height))
            if rendered.isNull():
                raise RuntimeError(
                    f"Failed to render PDF page {page_index + 1}: "
                    f"{record.filename}"
                )
            page_path = pages_dir / f"page_{page_index + 1:03d}.png"
            rendered.save(str(page_path), "PNG")
            page_paths.append(page_path)
        return page_paths

    def get_page_count(
        self,
        source_path: Path,
        json_data: dict[str, Any],
    ) -> int:
        data_info = json_data.get("dataInfo") or {}
        num_pages = data_info.get("numPages")
        if isinstance(num_pages, int) and num_pages > 0:
            return num_pages
        if source_path.suffix.lower() not in PPOCR_SUPPORTED_PDF_SUFFIXES:
            return 1
        pages_dir = self.get_pdf_pages_dir(source_path.name)
        if pages_dir.exists():
            page_paths = list(pages_dir.glob("page_*.png"))
            if page_paths:
                return len(page_paths)
        return 1

    def read_page_image_size(self, image_path: Path) -> tuple[int, int]:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            return image.width, image.height

    def clear_block_images(self, record: PPOCRFileRecord) -> None:
        block_images_dir = self.get_block_images_dir(record.filename)
        if block_images_dir.exists():
            shutil.rmtree(block_images_dir)

    def list_favorites(self) -> set[str]:
        state = self._load_ui_state()
        favorites = state.get("favorites") or []
        return {item for item in favorites if isinstance(item, str) and item}

    def is_favorite(self, filename: str) -> bool:
        return filename in self.list_favorites()

    def set_favorite(self, filename: str, favorite: bool) -> None:
        state = self._load_ui_state()
        favorites = {
            item
            for item in (state.get("favorites") or [])
            if isinstance(item, str) and item
        }
        if favorite:
            favorites.add(filename)
        else:
            favorites.discard(filename)
        state["favorites"] = sorted(favorites)
        self._save_ui_state(state)

    def _validate_basename(self, filename: str) -> tuple[bool, str]:
        if filename in {"", ".", ".."}:
            return False, "invalid filename"
        if "\x00" in filename:
            return False, "filename contains null byte"
        if os.sep in filename or (os.altsep and os.altsep in filename):
            return False, "filename contains path separator"
        if any(ord(ch) < 32 for ch in filename):
            return False, "filename contains control character"
        return True, ""

    def _deduplicated_filename(self, filename: str) -> str:
        path = Path(filename)
        if not self._target_name_exists(path.name):
            return path.name
        index = 1
        while True:
            candidate = f"{path.stem} ({index}){path.suffix}"
            if not self._target_name_exists(candidate):
                return candidate
            index += 1

    def _target_name_exists(self, filename: str) -> bool:
        return (self.files_dir / filename).exists() or self.get_json_path(
            filename
        ).exists()

    def _load_ui_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            logger.warning(f"Failed to load PaddleOCR state: {exc}")
            return {}

    def _save_ui_state(self, state: dict[str, Any]) -> None:
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
