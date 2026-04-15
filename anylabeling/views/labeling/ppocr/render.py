from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import re
from typing import Any

from .config import (
    PPOCR_COLOR_FORMULA,
    PPOCR_COLOR_HEADER,
    PPOCR_COLOR_IMAGE,
    PPOCR_COLOR_TABLE,
    PPOCR_COLOR_TEXT,
)

TEXT_BLOCK_LABELS = {
    "text",
    "doc_title",
    "vision_footnote",
    "figure_title",
    "inline_formula",
    "paragraph_title",
    "seal",
    "content",
    "vertical_text",
    "aside_text",
    "abstract",
    "footnote",
    "reference",
    "reference_content",
    "footer",
    "number",
}
TABLE_BLOCK_LABELS = {"table"}
IMAGE_BLOCK_LABELS = {
    "image",
    "chart",
    "header_image",
    "footer_image",
}
HEADER_BLOCK_LABELS = {"header"}
FORMULA_BLOCK_LABELS = {
    "display_formula",
    "formula_number",
    "algorithm",
    "formula",
}
HTML_TAG_PATTERN = re.compile(
    r"</?(?:p|div|span|h[1-6]|strong|b|em|i|u|s|strike|del|img|br|ul|ol|li)\b",
    flags=re.IGNORECASE,
)


def _block_top_y(block: PPOCRBlockData) -> float | None:
    if not block.points:
        return None
    return min(point[1] for point in block.points)


def _block_left_x(block: PPOCRBlockData) -> float | None:
    if not block.points:
        return None
    return min(point[0] for point in block.points)


def _block_right_x(block: PPOCRBlockData) -> float | None:
    if not block.points:
        return None
    return max(point[0] for point in block.points)


def _normalize_group_token(group_id: Any) -> str:
    if group_id is None:
        return ""
    token = str(group_id).strip()
    return token


def _repair_spilled_text_content(page_blocks: list[PPOCRBlockData]) -> None:
    for index in range(len(page_blocks) - 1):
        current = page_blocks[index]
        following = page_blocks[index + 1]
        if (
            current.label not in TEXT_BLOCK_LABELS
            or following.label not in TEXT_BLOCK_LABELS
        ):
            continue
        if not current.content.strip() or following.content.strip():
            continue

        current_top = _block_top_y(current)
        following_top = _block_top_y(following)
        current_left = _block_left_x(current)
        current_right = _block_right_x(current)
        following_left = _block_left_x(following)
        if (
            current_top is None
            or following_top is None
            or current_left is None
            or current_right is None
            or following_left is None
        ):
            continue
        if following_top + 24 >= current_top:
            continue
        # Restrict to cross-column handoff cases:
        # the next block sits to the right and starts significantly higher.
        if following_left <= current_right - 8:
            continue

        split_index = current.content.rfind("\n\n")
        if split_index < 0:
            split_index = current.content.rfind("\n")
        if split_index < 0:
            continue
        prefix = current.content[:split_index].rstrip()
        suffix = current.content[split_index:].strip()
        if not prefix or len(suffix) < 16:
            continue
        current.content = prefix
        following.content = suffix


def _link_grouped_text_blocks(page_blocks: list[PPOCRBlockData]) -> None:
    group_map: dict[str, list[PPOCRBlockData]] = {}
    for block in page_blocks:
        block.representative_block_key = block.block_key
        block.linked_block_keys = (block.block_key,)
        block.hidden_in_panel = False
        if block.label not in TEXT_BLOCK_LABELS:
            continue
        group_token = _normalize_group_token(block.group_id)
        if not group_token:
            continue
        group_map.setdefault(group_token, []).append(block)

    for grouped_blocks in group_map.values():
        if len(grouped_blocks) < 2:
            continue
        non_empty_blocks = [
            block for block in grouped_blocks if block.content.strip()
        ]
        if len(non_empty_blocks) != 1:
            continue
        representative = non_empty_blocks[0]
        linked_keys = tuple(
            block.block_key
            for block in sorted(
                grouped_blocks,
                key=lambda block: block.block_order,
            )
        )
        for block in grouped_blocks:
            block.representative_block_key = representative.block_key
            block.linked_block_keys = linked_keys
            block.hidden_in_panel = (
                block.block_key != representative.block_key
                and not block.content.strip()
            )


@dataclass
class PPOCRBlockData:
    page_no: int
    block_uid: str
    block_key: str
    label: str
    display_label: str
    content: str
    points: list[tuple[float, float]]
    category_color: str
    image_path: str = ""
    block_order: int = 0
    edited: bool = False
    group_id: int | str | None = None
    linked_block_keys: tuple[str, ...] = ()
    representative_block_key: str = ""
    hidden_in_panel: bool = False


@dataclass(frozen=True)
class PPOCRDocumentImageAsset:
    source_path: Path
    output_name: str


def normalize_block_uid(block_data: dict[str, Any], index: int) -> str:
    global_block_id = block_data.get("global_block_id")
    if global_block_id is not None:
        return f"global_block_{global_block_id}"
    block_id = block_data.get("block_id")
    if block_id is not None:
        return f"block_{block_id}"
    return f"block_{index + 1}"


def build_block_key(
    page_no: int, block_data: dict[str, Any], index: int
) -> str:
    return f"page_{page_no}:{normalize_block_uid(block_data, index)}"


def build_unique_block_key(
    page_no: int,
    block_data: dict[str, Any],
    index: int,
    seen_keys: dict[str, int],
) -> str:
    base_key = build_block_key(page_no, block_data, index)
    duplicate_index = seen_keys.get(base_key, 0) + 1
    seen_keys[base_key] = duplicate_index
    if duplicate_index <= 1:
        return base_key
    return f"{base_key}__dup_{duplicate_index}"


def normalize_block_label(label: str | None) -> str:
    normalized = re.sub(
        r"[\s\-]+",
        "_",
        str(label or "").strip(),
    ).casefold()
    return normalized or "text"


def label_to_display_text(label: str) -> str:
    normalized_label = normalize_block_label(label)
    if not normalized_label:
        return "Text"
    if normalized_label in FORMULA_BLOCK_LABELS:
        return "Display formula"
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", normalized_label)
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(chunk.capitalize() for chunk in text.split())


def is_rich_text_html(content: str) -> bool:
    if not content or "<" not in content or ">" not in content:
        return False
    return bool(HTML_TAG_PATTERN.search(content))


def rich_text_to_markdown(content: str) -> str:
    if not is_rich_text_html(content):
        return content
    from PyQt6.QtGui import QTextDocument

    document = QTextDocument()
    document.setHtml(content)
    to_markdown = getattr(document, "toMarkdown", None)
    if callable(to_markdown):
        return to_markdown().rstrip("\n")
    return document.toPlainText()


def category_color(label: str) -> str:
    normalized_label = normalize_block_label(label)
    if normalized_label in TABLE_BLOCK_LABELS:
        return PPOCR_COLOR_TABLE
    if normalized_label in IMAGE_BLOCK_LABELS:
        return PPOCR_COLOR_IMAGE
    if normalized_label in HEADER_BLOCK_LABELS:
        return PPOCR_COLOR_HEADER
    if normalized_label in FORMULA_BLOCK_LABELS:
        return PPOCR_COLOR_FORMULA
    return PPOCR_COLOR_TEXT


def block_points(block_data: dict[str, Any]) -> list[tuple[float, float]]:
    polygon = block_data.get("block_polygon_points") or []
    if polygon:
        return [
            (float(point[0]), float(point[1]))
            for point in polygon
            if len(point) >= 2
        ]
    bbox = block_data.get("block_bbox") or []
    if len(bbox) >= 4:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return []


def extract_blocks(document_data: dict[str, Any]) -> list[PPOCRBlockData]:
    layout_results = document_data.get("layoutParsingResults") or []
    meta = document_data.get("_ppocr_meta") or {}
    edited_blocks = set(meta.get("edited_blocks") or [])
    image_paths = meta.get("block_image_paths") or {}
    parsed_blocks = []
    for page_index, page_result in enumerate(layout_results, start=1):
        pruned_result = page_result.get("prunedResult") or {}
        block_list = pruned_result.get("parsing_res_list") or []
        page_blocks = []
        seen_keys: dict[str, int] = {}
        for index, block_data in enumerate(block_list):
            block_key = build_unique_block_key(
                page_index,
                block_data,
                index,
                seen_keys,
            )
            base_block_key = build_block_key(page_index, block_data, index)
            label = normalize_block_label(block_data.get("block_label"))
            page_blocks.append(
                PPOCRBlockData(
                    page_no=page_index,
                    block_uid=block_key.split(":", 1)[1],
                    block_key=block_key,
                    label=label,
                    display_label=label_to_display_text(label),
                    content=str(block_data.get("block_content") or ""),
                    points=block_points(block_data),
                    category_color=category_color(label),
                    image_path=(
                        str(
                            image_paths.get(block_key)
                            or image_paths.get(base_block_key)
                            or ""
                        )
                        if label in IMAGE_BLOCK_LABELS
                        else ""
                    ),
                    block_order=int(
                        block_data.get("block_order") or index + 1
                    ),
                    edited=(block_key in edited_blocks),
                    group_id=block_data.get("group_id"),
                )
            )
        page_blocks.sort(key=lambda block: block.block_order)
        _repair_spilled_text_content(page_blocks)
        _link_grouped_text_blocks(page_blocks)
        parsed_blocks.extend(page_blocks)
    return parsed_blocks


def page_blocks(
    document_data: dict[str, Any], page_no: int
) -> list[PPOCRBlockData]:
    return [
        block
        for block in extract_blocks(document_data)
        if block.page_no == page_no
    ]


def document_page_count(
    document_data: dict[str, Any], default: int = 1
) -> int:
    data_info = document_data.get("dataInfo") or {}
    num_pages = data_info.get("numPages")
    if isinstance(num_pages, int) and num_pages > 0:
        return num_pages
    layout_results = document_data.get("layoutParsingResults") or []
    if layout_results:
        return len(layout_results)
    return default


def document_page_size(
    document_data: dict[str, Any],
    page_no: int,
) -> tuple[int, int]:
    data_info = document_data.get("dataInfo") or {}
    pages = data_info.get("pages") or []
    if 1 <= page_no <= len(pages):
        page_info = pages[page_no - 1] or {}
        width = int(page_info.get("width") or 0)
        height = int(page_info.get("height") or 0)
        if width > 0 and height > 0:
            return width, height
    layout_results = document_data.get("layoutParsingResults") or []
    if 1 <= page_no <= len(layout_results):
        pruned_result = (layout_results[page_no - 1] or {}).get(
            "prunedResult"
        ) or {}
        width = int(pruned_result.get("width") or 0)
        height = int(pruned_result.get("height") or 0)
        if width > 0 and height > 0:
            return width, height
    return 0, 0


def get_block_copy_text(block: PPOCRBlockData, root_dir: Path) -> str:
    content = rich_text_to_markdown(block.content)
    if block.image_path:
        image_path = (root_dir / block.image_path).resolve()
        if content.strip():
            return f"{content.strip()}\n\n![]({image_path})"
        return f"![]({image_path})"
    return content


def _block_bbox_int(block: PPOCRBlockData) -> tuple[int, int, int, int]:
    if not block.points:
        return (0, 0, 0, 0)
    xs = [point[0] for point in block.points]
    ys = [point[1] for point in block.points]
    x1 = int(round(min(xs)))
    y1 = int(round(min(ys)))
    x2 = int(round(max(xs)))
    y2 = int(round(max(ys)))
    return (max(0, x1), max(0, y1), max(0, x2), max(0, y2))


def _deduplicate_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        used_names.add(name)
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    index = 1
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def _image_width_percent(
    block: PPOCRBlockData,
    page_width: int,
) -> int:
    if page_width <= 0 or not block.points:
        return 100
    xs = [point[0] for point in block.points]
    width_px = max(xs) - min(xs)
    if width_px <= 0:
        return 100
    percent = int(round((width_px / float(page_width)) * 100))
    return max(1, min(100, percent))


def _format_block_markdown(
    block: PPOCRBlockData,
    page_width: int,
    image_ref: str = "",
) -> tuple[str, str]:
    chunks = []
    text = rich_text_to_markdown(block.content)
    if text and text.strip():
        chunks.append(text)
    if image_ref:
        width_percent = _image_width_percent(block, page_width)
        chunks.append(
            '<div style="text-align: center;"><img src="'
            f"{escape(image_ref, quote=True)}"
            '" alt="Image" width="'
            f"{width_percent}%"
            '" /></div>'
        )
    return ("\n\n".join(chunks).strip(), image_ref)


def build_document_markdown_assets(
    blocks: list[PPOCRBlockData],
    document_data: dict[str, Any],
    root_dir: Path,
    image_dir_name: str = "imgs",
) -> tuple[str, list[PPOCRDocumentImageAsset]]:
    if not blocks:
        return ("", [])
    page_widths: dict[int, int] = {}
    for block in blocks:
        if block.page_no not in page_widths:
            page_widths[block.page_no] = document_page_size(
                document_data,
                block.page_no,
            )[0]
    used_image_names: set[str] = set()
    assets: list[PPOCRDocumentImageAsset] = []
    parts = []
    for block in blocks:
        image_ref = ""
        if block.image_path:
            x1, y1, x2, y2 = _block_bbox_int(block)
            output_name = _deduplicate_name(
                f"img_in_image_box_{x1}_{y1}_{x2}_{y2}.jpg",
                used_image_names,
            )
            image_ref = f"{image_dir_name}/{output_name}"
            source_path = (root_dir / block.image_path).resolve()
            if source_path.exists():
                assets.append(
                    PPOCRDocumentImageAsset(
                        source_path=source_path,
                        output_name=output_name,
                    )
                )
        block_text, _ = _format_block_markdown(
            block,
            page_widths.get(block.page_no, 0),
            image_ref,
        )
        if block_text:
            parts.append(block_text)
    return ("\n\n\n".join(parts).strip(), assets)


def get_document_copy_text(
    blocks: list[PPOCRBlockData],
    document_data: dict[str, Any],
    root_dir: Path,
) -> str:
    markdown_text, _ = build_document_markdown_assets(
        blocks,
        document_data,
        root_dir,
        image_dir_name="imgs",
    )
    return markdown_text
