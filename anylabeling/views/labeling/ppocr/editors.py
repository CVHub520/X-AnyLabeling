from __future__ import annotations

from html import escape as html_escape, unescape as html_unescape
from html.parser import HTMLParser
import hashlib
from io import BytesIO
import os
from functools import lru_cache
from pathlib import Path
import re
import shutil
import subprocess
import sys

from PyQt6.QtCore import (
    QEvent,
    QItemSelectionModel,
    QMimeData,
    QPoint,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPixmapCache,
    QShortcut,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.utils.theme import get_theme

from .config import PPOCR_BLOCK_CARD_MAX_HEIGHT_PX, PPOCR_COLOR_TEXT
from .render import (
    FORMULA_BLOCK_LABELS,
    TABLE_BLOCK_LABELS,
    is_rich_text_html,
    normalize_block_label,
)
from .style import get_primary_button_style, get_secondary_button_style

PPOCR_EDITOR_BODY_PT = 13
PPOCR_EDITOR_IMAGE_MAX_WIDTH = 280
PPOCR_EDITOR_IMAGE_MAX_HEIGHT = 220

PPOCR_EDITOR_HEADING_LEVELS = {
    1: {"label": "H1", "point_size": 24, "weight": QFont.Weight.DemiBold},
    2: {"label": "H2", "point_size": 20, "weight": QFont.Weight.DemiBold},
    3: {"label": "H3", "point_size": 18, "weight": QFont.Weight.Medium},
    4: {"label": "H4", "point_size": 16, "weight": QFont.Weight.Medium},
    5: {"label": "H5", "point_size": 14, "weight": QFont.Weight.Normal},
    6: {"label": "H6", "point_size": 13, "weight": QFont.Weight.Normal},
}

PPOCR_RICH_EDITOR_ICON_SIZE = 20
PPOCR_RICH_EDITOR_ICON_DPR = 2
PPOCR_RICH_EDITOR_BODY_PT = 10
PPOCR_RICH_EDITOR_TEXT_COLOR = QColor(0, 0, 0, 217)
PPOCR_RICH_EDITOR_ICON_COLOR = "#6b6f76"
PPOCR_RICH_EDITOR_SERIF = "Georgia"
PPOCR_RICH_EDITOR_BODY_LINE_HEIGHT = 100
PPOCR_RICH_EDITOR_BODY_MARGIN_BOTTOM = 6
PPOCR_RICH_EDITOR_IMAGE_TEXT_SPACING_PX = 6
PPOCR_RICH_EDITOR_MIN_HEIGHT_PX = 72
PPOCR_RICH_EDITOR_MAX_HEIGHT_RATIO = 0.75
PPOCR_RICH_EDITOR_HEIGHT_PADDING_PX = 40
PPOCR_RICH_EDITOR_BODY_OVERHEAD_PX = 32

PPOCR_RICH_EDITOR_HEADING_LEVELS = {
    1: {
        "label": "H1",
        "point_size": 29,
        "weight": QFont.Weight.DemiBold,
        "line_height": 123,
        "margin_bottom": 19,
        "css_size": 38,
        "css_line_height": "1.23",
    },
    2: {
        "label": "H2",
        "point_size": 23,
        "weight": QFont.Weight.DemiBold,
        "line_height": 135,
        "margin_bottom": 15,
        "css_size": 30,
        "css_line_height": "1.35",
    },
    3: {
        "label": "H3",
        "point_size": 18,
        "weight": QFont.Weight.DemiBold,
        "line_height": 135,
        "margin_bottom": 12,
        "css_size": 24,
        "css_line_height": "1.35",
    },
    4: {
        "label": "H4",
        "point_size": 15,
        "weight": QFont.Weight.DemiBold,
        "line_height": 140,
        "margin_bottom": 10,
        "css_size": 20,
        "css_line_height": "1.4",
    },
    5: {
        "label": "H5",
        "point_size": 12,
        "weight": QFont.Weight.DemiBold,
        "line_height": 150,
        "margin_bottom": 8,
        "css_size": 16,
        "css_line_height": "1.5",
    },
    6: {
        "label": "H6",
        "point_size": 11,
        "weight": QFont.Weight.DemiBold,
        "line_height": 157,
        "margin_bottom": 7,
        "css_size": 14,
        "css_line_height": "1.5715",
    },
}

PPOCR_RICH_EDITOR_CONTAINER_STYLE = """
QFrame#PPOCRRichTextBlockEditor {
    background-color: #f5f6f7;
    border-radius: 10px;
    border: 1px solid #e5e6eb;
}
QFrame#PPOCRRichTextBlockEditorToolbar {
    background-color: #f5f6f7;
    border: none;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    border-bottom: 1px solid #e8e9ec;
}
QFrame#PPOCRRichTextBlockEditorBody {
    background-color: #ffffff;
    border: none;
    border-bottom-left-radius: 10px;
    border-bottom-right-radius: 10px;
}
QLabel#PPOCRRichTextBlockEditorImage {
    background: transparent;
    border: none;
    padding: 0px;
}
"""

PPOCR_RICH_EDITOR_FORMAT_BUTTON_STYLE = """
QToolButton {
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 4px;
    min-width: 32px;
    min-height: 32px;
}
QToolButton:hover {
    background-color: rgb(241, 245, 255);
}
QToolButton:checked {
    background-color: rgb(241, 245, 255);
}
"""

PPOCR_RICH_EDITOR_HEADING_BUTTON_STYLE = """
QToolButton {
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 4px;
    min-width: 36px;
    min-height: 36px;
    color: rgba(0, 0, 0, 0.45);
}
QToolButton:hover {
    background-color: #f5f5f5;
    color: rgba(0, 0, 0, 0.85);
}
"""

PPOCR_RICH_EDITOR_HEADING_BUTTON_ACTIVE_STYLE = """
QToolButton {
    background-color: #f0f3ff;
    border: none;
    border-radius: 4px;
    padding: 4px;
    min-width: 36px;
    min-height: 36px;
    color: #2932e1;
}
"""

PPOCR_RICH_EDITOR_STYLE = """
QTextEdit {
    background-color: #ffffff;
    border: none;
    border-bottom-left-radius: 10px;
    border-bottom-right-radius: 10px;
    padding: 16px;
    font-size: 13px;
    selection-background-color: #c4c9f5;
    selection-color: #ffffff;
}
QScrollBar:vertical {
    width: 6px;
    background: transparent;
    margin: 4px 2px;
}
QScrollBar::handle:vertical {
    background: #d0d1d6;
    border-radius: 3px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #a0a1a6;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

PPOCR_LATEX_EDITOR_CONTAINER_STYLE = """
QFrame#PPOCRLatexBlockEditor {
    background-color: #f5f6f7;
    border-radius: 10px;
    border: 1px solid #e5e6eb;
}
QFrame#PPOCRLatexBlockEditorToolbar {
    background-color: #f5f6f7;
    border: none;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    border-bottom: 1px solid #e8e9ec;
}
QFrame#PPOCRLatexBlockEditorBody {
    background-color: #ffffff;
    border: none;
    border-bottom-left-radius: 10px;
    border-bottom-right-radius: 10px;
}
QFrame#PPOCRLatexPreviewDivider {
    background: transparent;
    border: none;
    border-top: 1px solid #e8e9ec;
}
QLabel#PPOCRLatexPreviewTitle {
    background: #d9e1ff;
    color: #ffffff;
    font-size: 12px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 0px;
}
QLabel#PPOCRLatexPreviewContent {
    background: transparent;
    color: rgba(0, 0, 0, 0.85);
    border: none;
}
"""

PPOCR_LATEX_SOURCE_STYLE = """
QPlainTextEdit {
    background-color: #ffffff;
    color: rgba(0, 0, 0, 0.85);
    border: none;
    border-radius: 0px;
    padding: 8px 10px 14px 10px;
    font-family: 'SFMono-Regular', 'Menlo', 'Consolas', monospace;
    font-size: 13px;
    selection-background-color: #c4c9f5;
}
QScrollBar:vertical {
    width: 6px;
    background: transparent;
    margin: 4px 2px;
}
QScrollBar::handle:vertical {
    background: #d0d1d6;
    border-radius: 3px;
    min-height: 30px;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

PPOCR_LATEX_PREVIEW_SCROLL_STYLE = """
QScrollArea {
    background: transparent;
    border: none;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: transparent;
}
QScrollBar:vertical {
    width: 6px;
    margin: 4px 2px;
}
QScrollBar:horizontal {
    height: 6px;
    margin: 2px 4px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #d0d1d6;
    border-radius: 3px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
    height: 0px;
}
"""

PPOCR_LATEX_PREVIEW_FONT_PT = 12
PPOCR_LATEX_PREVIEW_DPI = 110
PPOCR_LATEX_PREVIEW_FONTSET = "stix"
PPOCR_LATEX_PREVIEW_TRIM_PADDING_PX = 8
PPOCR_LATEX_PREVIEW_LINE_GAP_PX = 16
PPOCR_LATEX_PREVIEW_BLANK_LINE_GAP_PX = 12
PPOCR_LATEX_PREVIEW_CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Arial Unicode MS",
)

PPOCR_LATEX_RENDER_SCRIPT = f"""
from io import BytesIO
import sys

try:
    from matplotlib import font_manager, rcParams
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image
except Exception as exc:
    sys.stderr.write(str(exc))
    raise SystemExit(3)

rcParams["mathtext.fontset"] = "{PPOCR_LATEX_PREVIEW_FONTSET}"
rcParams["mathtext.default"] = "it"
rcParams["font.family"] = "serif"

expression = sys.argv[1]
def _cjk_chars(value):
    return "".join(
        sorted(set(char for char in value if "\u3400" <= char <= "\u9fff"))
    )


def _font_supports_chars(font_path, chars):
    if not chars:
        return True
    try:
        from matplotlib.ft2font import FT2Font
        charmap = FT2Font(font_path).get_charmap()
    except Exception:
        return False
    return all(ord(char) in charmap for char in chars)


def _select_cjk_font(chars):
    if not chars:
        return ""
    try:
        font_entries = font_manager.fontManager.ttflist
    except Exception:
        return ""
    for candidate in {repr(PPOCR_LATEX_PREVIEW_CJK_FONT_CANDIDATES)}:
        candidate_entries = [
            entry for entry in font_entries if entry.name == candidate
        ]
        if not candidate_entries:
            continue
        for entry in candidate_entries:
            font_path = getattr(entry, "fname", "")
            if font_path and _font_supports_chars(font_path, chars):
                return candidate
    return ""


selected_cjk_font = _select_cjk_font(_cjk_chars(expression))
if selected_cjk_font:
    rcParams["mathtext.fontset"] = "custom"
    rcParams["mathtext.fallback"] = "{PPOCR_LATEX_PREVIEW_FONTSET}"
    rcParams["mathtext.rm"] = selected_cjk_font
    rcParams["mathtext.sf"] = selected_cjk_font
    rcParams["font.family"] = selected_cjk_font
    rcParams["font.sans-serif"] = [selected_cjk_font, "DejaVu Sans"]
buffer = BytesIO()
prop = FontProperties(size={PPOCR_LATEX_PREVIEW_FONT_PT})
try:
    math_to_image(
        f"${{expression}}$",
        buffer,
        prop=prop,
        dpi={PPOCR_LATEX_PREVIEW_DPI},
        format="png",
        color="black",
    )
except Exception as exc:
    sys.stderr.write(str(exc))
    raise SystemExit(2)

sys.stdout.buffer.write(buffer.getvalue())
"""

_LATEX_RENDERER_PYTHON: str | None = None
_LATEX_PREVIEW_PIXMAP_CACHE_CONFIGURED = False
_LATEX_ALIGN_ENV_PATTERN = re.compile(
    r"^\\begin\{(?P<env>aligned\*?|align\*?|gathered\*?|gather\*?)\}"
    r"(?P<body>.*)"
    r"\\end\{(?P=env)\}$",
    re.DOTALL,
)
_LATEX_ARRAY_ENV_PATTERN = re.compile(
    r"^\\begin\{(?P<env>array\*?)\}(?:\{[^{}]*\})?"
    r"(?P<body>.*)"
    r"\\end\{(?P=env)\}$",
    re.DOTALL,
)
_LATEX_DELIMITER_SIZE_COMMAND_PATTERN = re.compile(
    r"\\(?:big|Big|bigg|Bigg)(?:l|r|m)?"
)
_LATEX_NULL_AUTO_DELIMITER_PATTERN = re.compile(r"\\(?:left|right)\s*\.")
_LATEX_AUTO_DELIMITER_COMMAND_PATTERN = re.compile(r"\\(?:left|right)\s*")
_LATEX_LIMITS_COMMAND_PATTERN = re.compile(r"\\limits")
_LATEX_ARRAY_RULE_PATTERN = re.compile(
    r"\\(?:hline|toprule|midrule|bottomrule|cline\{[^{}]*\})"
)
_LATEX_CJK_CHAR_PATTERN = re.compile(r"[\u3400-\u9fff]")
_TABLE_TOKEN_PATTERN = re.compile(
    r"<(fcel|ecel|lcel|ucel|xcel|nl)>",
    flags=re.IGNORECASE,
)
_TABLE_STYLE_TAG_PATTERN = re.compile(
    r"</?(?:b|strong|i|em|s|strike|del)>",
    flags=re.IGNORECASE,
)
_HTML_TABLE_OPEN_PATTERN = re.compile(r"<\s*table\b", re.IGNORECASE)
_HTML_TABLE_ROW_PATTERN = re.compile(r"<\s*tr\b", re.IGNORECASE)
_HTML_TABLE_CELL_PATTERN = re.compile(r"<\s*t[dh]\b", re.IGNORECASE)
PPOCR_LATEX_PREVIEW_MIN_HEIGHT_PX = 72
PPOCR_TABLE_CELL_LINE_HEIGHT = 1.45
PPOCR_TABLE_CELL_VERTICAL_PADDING_PX = 6
PPOCR_TABLE_CELL_HORIZONTAL_PADDING_PX = 10
PPOCR_TABLE_MIN_ROW_HEIGHT_PX = 28
PPOCR_TABLE_MIN_COLUMN_WIDTH_PX = 36
PPOCR_TABLE_MAX_COLUMN_WIDTH_PX = 560
PPOCR_TABLE_SELECTION_COLOR = "rgb(228, 236, 255)"
PPOCR_TABLE_VIEWPORT_BOTTOM_MARGIN_PX = 20


def _ensure_latex_preview_pixmap_cache_limit() -> None:
    global _LATEX_PREVIEW_PIXMAP_CACHE_CONFIGURED
    if _LATEX_PREVIEW_PIXMAP_CACHE_CONFIGURED:
        return
    target_limit_kb = max(QPixmapCache.cacheLimit(), 64 * 1024)
    QPixmapCache.setCacheLimit(target_limit_kb)
    _LATEX_PREVIEW_PIXMAP_CACHE_CONFIGURED = True


def _latex_preview_pixmap_cache_key(normalized_source: str) -> str:
    _, _, _, cjk_font = _latex_preview_cache_token_for_source(
        normalized_source
    )
    digest = hashlib.sha1(
        normalized_source.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    return (
        "ppocr_latex_preview:"
        f"{PPOCR_LATEX_PREVIEW_FONTSET}:"
        f"{PPOCR_LATEX_PREVIEW_FONT_PT}:"
        f"{PPOCR_LATEX_PREVIEW_DPI}:"
        f"{PPOCR_LATEX_PREVIEW_TRIM_PADDING_PX}:"
        f"{cjk_font}:"
        f"{digest}"
    )


def has_cached_latex_preview_pixmap(source: str) -> bool:
    normalized_source = _normalized_latex_source(source)
    if not normalized_source:
        return False
    _ensure_latex_preview_pixmap_cache_limit()
    cached_pixmap = QPixmapCache.find(
        _latex_preview_pixmap_cache_key(normalized_source)
    )
    return cached_pixmap is not None and not cached_pixmap.isNull()


def _apply_rich_block_metrics(
    block_format: QTextBlockFormat,
    heading_level: int,
    image_only: bool = False,
) -> None:
    if image_only:
        block_format.setTopMargin(0)
        block_format.setBottomMargin(PPOCR_RICH_EDITOR_IMAGE_TEXT_SPACING_PX)
        block_format.setLineHeight(
            0,
            QTextBlockFormat.LineHeightTypes.SingleHeight.value,
        )
        return
    if heading_level in PPOCR_RICH_EDITOR_HEADING_LEVELS:
        heading_info = PPOCR_RICH_EDITOR_HEADING_LEVELS[heading_level]
        block_format.setTopMargin(0)
        block_format.setBottomMargin(heading_info["margin_bottom"])
        block_format.setLineHeight(
            heading_info["line_height"],
            QTextBlockFormat.LineHeightTypes.ProportionalHeight.value,
        )
        return
    block_format.setTopMargin(0)
    block_format.setBottomMargin(PPOCR_RICH_EDITOR_BODY_MARGIN_BOTTOM)
    block_format.setLineHeight(
        0,
        QTextBlockFormat.LineHeightTypes.SingleHeight.value,
    )


def _block_contains_only_image(block) -> bool:
    image_found = False
    iterator = block.begin()
    while not iterator.atEnd():
        fragment = iterator.fragment()
        if fragment.isValid():
            char_format = fragment.charFormat()
            if char_format.isImageFormat():
                image_found = True
            elif fragment.text().strip():
                return False
        iterator += 1
    return image_found


def _block_is_effectively_empty(block) -> bool:
    if _block_contains_only_image(block):
        return False
    iterator = block.begin()
    while not iterator.atEnd():
        fragment = iterator.fragment()
        if fragment.isValid():
            text = (
                fragment.text()
                .replace("\u2028", "")
                .replace("\u2029", "")
                .strip()
            )
            if text:
                return False
        iterator += 1
    return True


def _normalized_latex_source(source: str) -> str:
    normalized = (source or "").strip()
    if normalized.startswith("$$") and normalized.endswith("$$"):
        return normalized[2:-2].strip()
    if normalized.startswith(r"\[") and normalized.endswith(r"\]"):
        return normalized[2:-2].strip()
    if normalized.startswith(r"\(") and normalized.endswith(r"\)"):
        return normalized[2:-2].strip()
    if normalized.startswith("$") and normalized.endswith("$"):
        return normalized[1:-1].strip()
    return normalized


def _consume_balanced_latex_group(
    source: str,
    start_index: int,
    open_char: str,
    close_char: str,
) -> tuple[str, int] | None:
    if start_index >= len(source) or source[start_index] != open_char:
        return None
    depth = 0
    index = start_index
    while index < len(source):
        current = source[index]
        if current == "\\" and index + 1 < len(source):
            index += 2
            continue
        if current == open_char:
            depth += 1
        elif current == close_char:
            depth -= 1
            if depth == 0:
                return source[start_index + 1 : index], index + 1
        index += 1
    return None


def _normalize_extensible_arrow_commands(source: str) -> str:
    command_fallbacks = (
        (r"\xrightarrow", r"\rightarrow"),
        (r"\xleftarrow", r"\leftarrow"),
        (r"\xlongequal", "="),
    )
    if not any(command in source for command, _ in command_fallbacks):
        return source

    normalized_parts: list[str] = []
    index = 0
    source_len = len(source)
    while index < source_len:
        matched_command = None
        fallback_arrow = ""
        for command, fallback in command_fallbacks:
            if source.startswith(command, index):
                matched_command = command
                fallback_arrow = fallback
                break
        if matched_command is None:
            normalized_parts.append(source[index])
            index += 1
            continue

        cursor = index + len(matched_command)
        while cursor < source_len and source[cursor].isspace():
            cursor += 1

        below_text = ""
        if cursor < source_len and source[cursor] == "[":
            below_group = _consume_balanced_latex_group(
                source,
                cursor,
                "[",
                "]",
            )
            if below_group is None:
                normalized_parts.append(source[index])
                index += 1
                continue
            below_text, cursor = below_group
            while cursor < source_len and source[cursor].isspace():
                cursor += 1

        if cursor >= source_len or source[cursor] != "{":
            normalized_parts.append(source[index])
            index += 1
            continue

        above_group = _consume_balanced_latex_group(source, cursor, "{", "}")
        if above_group is None:
            normalized_parts.append(source[index])
            index += 1
            continue
        above_text, cursor = above_group

        above_text = above_text.strip()
        below_text = below_text.strip()
        if above_text and below_text:
            normalized_parts.append(
                rf"\underset{{{below_text}}}{{\overset{{{above_text}}}{{{fallback_arrow}}}}}"
            )
        elif above_text:
            normalized_parts.append(
                rf"\overset{{{above_text}}}{{{fallback_arrow}}}"
            )
        elif below_text:
            normalized_parts.append(
                rf"\underset{{{below_text}}}{{{fallback_arrow}}}"
            )
        else:
            normalized_parts.append(fallback_arrow)
        index = cursor

    return "".join(normalized_parts)


def _sanitize_latex_preview_source(source: str) -> str:
    sanitized = _normalize_extensible_arrow_commands(source)
    sanitized = _LATEX_DELIMITER_SIZE_COMMAND_PATTERN.sub("", sanitized)
    sanitized = _LATEX_NULL_AUTO_DELIMITER_PATTERN.sub("", sanitized)
    sanitized = _LATEX_AUTO_DELIMITER_COMMAND_PATTERN.sub("", sanitized)
    return _LATEX_LIMITS_COMMAND_PATTERN.sub("", sanitized)


def _latex_preview_cjk_chars(source: str) -> str:
    if not source:
        return ""
    cjk_chars = _LATEX_CJK_CHAR_PATTERN.findall(source)
    if not cjk_chars:
        return ""
    return "".join(sorted(set(cjk_chars)))


def _font_supports_latex_cjk_chars(font_path: str, cjk_chars: str) -> bool:
    if not font_path or not cjk_chars:
        return False
    try:
        from matplotlib.ft2font import FT2Font
    except Exception:
        return False
    try:
        charmap = FT2Font(font_path).get_charmap()
    except Exception:
        return False
    return all(ord(char) in charmap for char in cjk_chars)


@lru_cache(maxsize=64)
def _resolve_latex_preview_cjk_font(cjk_chars: str) -> str:
    if not cjk_chars:
        return ""
    try:
        from matplotlib import font_manager
    except Exception:
        return ""
    font_entries = font_manager.fontManager.ttflist
    for candidate in PPOCR_LATEX_PREVIEW_CJK_FONT_CANDIDATES:
        candidate_entries = [
            entry for entry in font_entries if entry.name == candidate
        ]
        if not candidate_entries:
            continue
        for entry in candidate_entries:
            font_path = str(getattr(entry, "fname", "") or "")
            if _font_supports_latex_cjk_chars(font_path, cjk_chars):
                return candidate
    return ""


def _latex_preview_cache_token_for_source(
    normalized_source: str,
) -> tuple[str, int, int, str]:
    cjk_chars = _latex_preview_cjk_chars(normalized_source)
    cjk_font = _resolve_latex_preview_cjk_font(cjk_chars)
    return (
        PPOCR_LATEX_PREVIEW_FONTSET,
        PPOCR_LATEX_PREVIEW_FONT_PT,
        PPOCR_LATEX_PREVIEW_DPI,
        cjk_font,
    )


def _configure_latex_preview_rcparams(
    rc_params, normalized_source: str
) -> None:
    rc_params["mathtext.fontset"] = PPOCR_LATEX_PREVIEW_FONTSET
    rc_params["mathtext.default"] = "it"
    rc_params["font.family"] = "serif"
    cjk_chars = _latex_preview_cjk_chars(normalized_source)
    if not cjk_chars:
        return
    cjk_font = _resolve_latex_preview_cjk_font(cjk_chars)
    if not cjk_font:
        return
    rc_params["mathtext.fontset"] = "custom"
    rc_params["mathtext.fallback"] = PPOCR_LATEX_PREVIEW_FONTSET
    rc_params["mathtext.rm"] = cjk_font
    rc_params["mathtext.sf"] = cjk_font
    rc_params["font.family"] = cjk_font
    rc_params["font.sans-serif"] = [cjk_font, "DejaVu Sans"]


def is_table_like_content(content: str) -> bool:
    raw = (content or "").strip()
    if not raw:
        return False
    if _TABLE_TOKEN_PATTERN.search(raw):
        return True
    return bool(
        _HTML_TABLE_OPEN_PATTERN.search(raw)
        and _HTML_TABLE_ROW_PATTERN.search(raw)
        and _HTML_TABLE_CELL_PATTERN.search(raw)
    )


def _parse_span_int(value: str | None) -> int:
    if value is None:
        return 1
    try:
        parsed = int(str(value).strip())
    except Exception:
        return 1
    return max(1, parsed)


class _PPOCRHTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, object]]] = []
        self._table_depth = 0
        self._current_row: list[dict[str, object]] | None = None
        self._current_cell: dict[str, object] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        lowered = tag.lower()
        if lowered == "table":
            self._table_depth += 1
            return
        if self._table_depth <= 0:
            return
        if lowered == "tr":
            self._current_row = []
            return
        if lowered in {"td", "th"}:
            if self._current_row is None:
                self._current_row = []
            attr_map = {
                str(key).lower(): str(value or "") for key, value in attrs
            }
            self._current_cell = {
                "text_parts": [],
                "row_span": _parse_span_int(attr_map.get("rowspan")),
                "col_span": _parse_span_int(attr_map.get("colspan")),
                "bold": False,
                "italic": False,
                "strike": False,
            }
            return
        if self._current_cell is None:
            return
        if lowered in {"b", "strong"}:
            self._current_cell["bold"] = True
        elif lowered in {"i", "em"}:
            self._current_cell["italic"] = True
        elif lowered in {"s", "strike", "del"}:
            self._current_cell["strike"] = True
        elif lowered == "br":
            text_parts = self._current_cell.get("text_parts")
            if isinstance(text_parts, list):
                text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "table":
            if self._table_depth > 0:
                self._table_depth -= 1
            return
        if self._table_depth <= 0:
            return
        if lowered in {"td", "th"}:
            if self._current_cell is None:
                return
            text_parts = self._current_cell.get("text_parts")
            text = (
                "".join(text_parts).strip()
                if isinstance(text_parts, list)
                else ""
            )
            self._current_cell["text"] = text
            if self._current_row is None:
                self._current_row = []
            self._current_row.append(self._current_cell)
            self._current_cell = None
            return
        if lowered == "tr":
            if self._current_row is not None:
                self.rows.append(self._current_row)
            self._current_row = None
            self._current_cell = None

    def handle_data(self, data: str) -> None:
        if self._table_depth <= 0 or self._current_cell is None:
            return
        text_parts = self._current_cell.get("text_parts")
        if isinstance(text_parts, list):
            text_parts.append(data)


def _latex_renderer_candidates() -> list[str]:
    candidates = [
        os.environ.get("CONDA_PYTHON_EXE"),
        sys.executable,
        shutil.which("python"),
        shutil.which("python3"),
        "/usr/bin/python3",
    ]
    resolved_candidates = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = str(Path(candidate).expanduser())
        if (
            candidate_path not in resolved_candidates
            and Path(candidate_path).exists()
        ):
            resolved_candidates.append(candidate_path)
    return resolved_candidates


def _render_latex_preview_png_bytes_with_matplotlib(
    normalized_source: str,
) -> bytes:
    from matplotlib import rcParams
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image

    _configure_latex_preview_rcparams(rcParams, normalized_source)

    buffer = BytesIO()
    prop = FontProperties(size=PPOCR_LATEX_PREVIEW_FONT_PT)
    math_to_image(
        f"${normalized_source}$",
        buffer,
        prop=prop,
        dpi=PPOCR_LATEX_PREVIEW_DPI,
        format="png",
        color="black",
    )
    return buffer.getvalue()


def _render_latex_preview_png_bytes_with_subprocess(
    normalized_source: str,
) -> bytes:
    global _LATEX_RENDERER_PYTHON

    candidates = (
        [_LATEX_RENDERER_PYTHON]
        if _LATEX_RENDERER_PYTHON
        else _latex_renderer_candidates()
    )
    missing_matplotlib = False
    render_env = dict(os.environ)
    render_env["MPLBACKEND"] = "Agg"

    for candidate in candidates:
        try:
            completed = subprocess.run(
                [
                    candidate,
                    "-c",
                    PPOCR_LATEX_RENDER_SCRIPT,
                    normalized_source,
                ],
                capture_output=True,
                timeout=8,
                env=render_env,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

        stderr_text = completed.stderr.decode(
            "utf-8", errors="replace"
        ).strip()
        if completed.returncode == 0 and completed.stdout:
            _LATEX_RENDERER_PYTHON = candidate
            return completed.stdout
        if "No module named 'matplotlib'" in stderr_text:
            missing_matplotlib = True
            continue
        if stderr_text:
            raise ValueError(stderr_text)

    if missing_matplotlib:
        raise ModuleNotFoundError(
            "LaTeX preview requires matplotlib in an available Python runtime."
        )
    raise ValueError("Failed to render LaTeX preview.")


@lru_cache(maxsize=384)
def _render_latex_preview_png_bytes_cached(
    normalized_source: str,
    cache_token: tuple[str, int, int, str],
) -> bytes:
    del cache_token
    local_error = None
    try:
        return _render_latex_preview_png_bytes_with_matplotlib(
            normalized_source
        )
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        local_error = exc

    try:
        return _render_latex_preview_png_bytes_with_subprocess(
            normalized_source
        )
    except Exception:
        if local_error is not None:
            raise local_error
        raise


def _render_latex_preview_png_bytes(source: str) -> bytes:
    normalized = _sanitize_latex_preview_source(
        _normalized_latex_source(source)
    )
    if not normalized:
        raise ValueError("Enter LaTeX source to preview.")
    return _render_latex_preview_png_bytes_cached(
        normalized,
        _latex_preview_cache_token_for_source(normalized),
    )


def _trim_latex_preview_png_bytes(png_bytes: bytes) -> bytes:
    try:
        from PIL import Image, ImageChops
    except Exception:
        return png_bytes

    image_buffer = BytesIO(png_bytes)
    with Image.open(image_buffer) as image:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bbox = ImageChops.difference(image, background).getbbox()
        if bbox is None:
            return png_bytes
        padding = PPOCR_LATEX_PREVIEW_TRIM_PADDING_PX
        left = max(0, bbox[0] - padding)
        top = max(0, bbox[1] - padding)
        right = min(image.width, bbox[2] + padding)
        bottom = min(image.height, bbox[3] + padding)
        cropped = image.crop((left, top, right, bottom))
        output_buffer = BytesIO()
        cropped.save(output_buffer, format="PNG")
        return output_buffer.getvalue()


@lru_cache(maxsize=384)
def _render_trimmed_latex_preview_png_bytes(
    normalized_source: str,
    cache_token: tuple[str, int, int, str, int],
) -> bytes:
    del cache_token
    return _trim_latex_preview_png_bytes(
        _render_latex_preview_png_bytes_cached(
            normalized_source,
            _latex_preview_cache_token_for_source(normalized_source),
        )
    )


def _render_single_latex_preview_pixmap(source: str) -> QPixmap:
    normalized = _sanitize_latex_preview_source(
        _normalized_latex_source(source)
    )
    if not normalized:
        raise ValueError("Enter LaTeX source to preview.")
    image_buffer = BytesIO(
        _render_trimmed_latex_preview_png_bytes(
            normalized,
            (
                *_latex_preview_cache_token_for_source(normalized),
                PPOCR_LATEX_PREVIEW_TRIM_PADDING_PX,
            ),
        )
    )
    pixmap = QPixmap()
    if not pixmap.loadFromData(image_buffer.getvalue(), "PNG"):
        raise ValueError("Failed to render LaTeX preview.")
    return pixmap


def _latex_preview_line_specs(
    source: str,
) -> list[tuple[int, str, int]]:
    normalized = _normalized_latex_source(source)
    if not normalized:
        return []

    aligned_match = _LATEX_ALIGN_ENV_PATTERN.fullmatch(normalized)
    if aligned_match:
        body = aligned_match.group("body").strip()
        if not body:
            return []
        line_specs = []
        for line_number, raw_line in enumerate(
            re.split(r"\\\\\s*", body),
            start=1,
        ):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            stripped_line = re.sub(r"\s*&\s*", "", stripped_line).strip()
            if stripped_line:
                line_specs.append((line_number, stripped_line, 0))
        return line_specs

    array_match = _LATEX_ARRAY_ENV_PATTERN.fullmatch(normalized)
    if array_match:
        body = array_match.group("body").strip()
        if not body:
            return []
        line_specs = []
        for line_number, raw_line in enumerate(
            re.split(r"\\\\\s*", body),
            start=1,
        ):
            stripped_line = _LATEX_ARRAY_RULE_PATTERN.sub("", raw_line).strip()
            if not stripped_line:
                continue
            stripped_line = re.sub(r"\s*&\s*", "", stripped_line).strip()
            if stripped_line:
                line_specs.append((line_number, stripped_line, 0))
        return line_specs

    line_specs: list[tuple[int, str, int]] = []
    blank_run = 0
    for line_number, raw_line in enumerate(normalized.splitlines(), start=1):
        stripped_line = _normalized_latex_source(raw_line.strip())
        if not stripped_line:
            blank_run += 1
            continue
        line_specs.append((line_number, stripped_line, blank_run))
        blank_run = 0
    return line_specs


def render_latex_preview_pixmap(source: str) -> QPixmap:
    normalized_source = _normalized_latex_source(source)
    if not normalized_source:
        raise ValueError("Enter LaTeX source to preview.")

    _ensure_latex_preview_pixmap_cache_limit()
    cache_key = _latex_preview_pixmap_cache_key(normalized_source)
    cached_pixmap = QPixmapCache.find(cache_key)
    if cached_pixmap is not None and not cached_pixmap.isNull():
        return cached_pixmap

    line_specs = _latex_preview_line_specs(normalized_source)
    if not line_specs:
        raise ValueError("Enter LaTeX source to preview.")

    sanitized_source = _sanitize_latex_preview_source(normalized_source)
    has_tex_environment = r"\begin{" in normalized_source
    if len(line_specs) > 1 and sanitized_source and not has_tex_environment:
        try:
            pixmap = _render_single_latex_preview_pixmap(sanitized_source)
            if not pixmap.isNull():
                QPixmapCache.insert(cache_key, pixmap)
            return pixmap
        except Exception:
            pass

    if len(line_specs) == 1 and line_specs[0][2] == 0:
        pixmap = _render_single_latex_preview_pixmap(line_specs[0][1])
        if not pixmap.isNull():
            QPixmapCache.insert(cache_key, pixmap)
        return pixmap

    rendered_lines: list[tuple[QPixmap, int]] = []
    max_width = 0
    total_height = 0
    for index, (line_number, line_text, blank_before) in enumerate(line_specs):
        try:
            line_pixmap = _render_single_latex_preview_pixmap(line_text)
        except Exception as exc:
            raise ValueError(f"Line {line_number}: {exc}") from exc
        gap_before = 0
        if index > 0:
            gap_before = PPOCR_LATEX_PREVIEW_LINE_GAP_PX + (
                blank_before * PPOCR_LATEX_PREVIEW_BLANK_LINE_GAP_PX
            )
        rendered_lines.append((line_pixmap, gap_before))
        max_width = max(max_width, line_pixmap.width())
        total_height += gap_before + line_pixmap.height()

    composite = QPixmap(max_width, total_height)
    composite.fill(Qt.GlobalColor.transparent)
    painter = QPainter(composite)
    try:
        y_offset = 0
        for line_pixmap, gap_before in rendered_lines:
            y_offset += gap_before
            painter.drawPixmap(0, y_offset, line_pixmap)
            y_offset += line_pixmap.height()
    finally:
        painter.end()
    if not composite.isNull():
        QPixmapCache.insert(cache_key, composite)
    return composite


def _apply_heading_char_format(
    cursor: QTextCursor, heading_level: int
) -> None:
    char_format = QTextCharFormat()
    if heading_level <= 0:
        char_format.setFontPointSize(PPOCR_RICH_EDITOR_BODY_PT)
        char_format.setFontWeight(QFont.Weight.Normal)
    else:
        heading_info = PPOCR_RICH_EDITOR_HEADING_LEVELS[heading_level]
        char_format.setFontPointSize(heading_info["point_size"])
        char_format.setFontWeight(heading_info["weight"])
    char_format.setForeground(PPOCR_RICH_EDITOR_TEXT_COLOR)
    cursor.mergeCharFormat(char_format)


def _heading_font_point_size(heading_level: int) -> float:
    if heading_level <= 0:
        return float(PPOCR_RICH_EDITOR_BODY_PT)
    return float(PPOCR_RICH_EDITOR_HEADING_LEVELS[heading_level]["point_size"])


def _heading_font_weight(heading_level: int) -> int:
    if heading_level <= 0:
        return int(QFont.Weight.Normal)
    return int(PPOCR_RICH_EDITOR_HEADING_LEVELS[heading_level]["weight"])


def _apply_heading_fragment_formats(
    cursor: QTextCursor,
    previous_heading_level: int,
    heading_level: int,
) -> None:
    target_point_size = _heading_font_point_size(heading_level)
    previous_weight = _heading_font_weight(previous_heading_level)
    target_weight = _heading_font_weight(heading_level)
    block = cursor.block()
    selection_start = cursor.selectionStart()
    selection_end = cursor.selectionEnd()
    iterator = block.begin()
    while not iterator.atEnd():
        fragment = iterator.fragment()
        iterator += 1
        if not fragment.isValid() or fragment.charFormat().isImageFormat():
            continue
        fragment_start = fragment.position()
        fragment_end = fragment_start + len(fragment.text())
        if fragment_end <= selection_start or fragment_start >= selection_end:
            continue
        fragment_cursor = QTextCursor(cursor.document())
        fragment_cursor.setPosition(max(fragment_start, selection_start))
        fragment_cursor.setPosition(
            min(fragment_end, selection_end),
            QTextCursor.MoveMode.KeepAnchor,
        )
        fragment_char_format = fragment.charFormat()
        char_format = QTextCharFormat()
        char_format.setFontPointSize(target_point_size)
        char_format.setFontWeight(
            fragment_char_format.fontWeight()
            if fragment_char_format.fontWeight() > previous_weight
            else target_weight
        )
        char_format.setForeground(PPOCR_RICH_EDITOR_TEXT_COLOR)
        fragment_cursor.mergeCharFormat(char_format)


class _PPOCRLatexResizeHandle(QWidget):
    def __init__(self, owner) -> None:
        super().__init__(owner.viewport())
        self._owner = owner
        self.setFixedSize(14, 14)
        self.setCursor(Qt.CursorShape.SizeVerCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._owner.start_resize_drag(event.globalPosition().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._owner.continue_resize_drag(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._owner.finish_resize_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#c6cad6"))
        pen.setWidth(1)
        painter.setPen(pen)
        for offset in (0, 4, 8):
            painter.drawLine(
                self.width() - 8 - offset,
                self.height() - 2,
                self.width() - 2,
                self.height() - 8 - offset,
            )


class _PPOCRResizableLatexSourceEdit(QPlainTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._minimum_editor_height = 120
        self._maximum_editor_height = 360
        self._manual_height = 0
        self._resize_drag_active = False
        self._resize_drag_origin_y = 0
        self._resize_drag_origin_height = 0
        self._resize_handle = _PPOCRLatexResizeHandle(self)
        self.document().documentLayout().documentSizeChanged.connect(
            lambda _size: self._sync_height_to_content()
        )
        self.textChanged.connect(self._sync_height_to_content)

    def set_height_bounds(
        self, minimum_height: int, maximum_height: int
    ) -> None:
        self._minimum_editor_height = minimum_height
        self._maximum_editor_height = max(maximum_height, minimum_height)
        self._sync_height_to_content()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._sync_height_to_content)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_resize_handle()
        QTimer.singleShot(0, self._sync_height_to_content)

    def start_resize_drag(self, global_pos: QPoint) -> None:
        self._resize_drag_active = True
        self._resize_drag_origin_y = global_pos.y()
        self._resize_drag_origin_height = self.height()

    def continue_resize_drag(self, global_pos: QPoint) -> None:
        if not self._resize_drag_active:
            return
        delta_y = global_pos.y() - self._resize_drag_origin_y
        self._manual_height = max(
            self._minimum_editor_height,
            min(
                self._maximum_editor_height,
                self._resize_drag_origin_height + delta_y,
            ),
        )
        self.setFixedHeight(self._manual_height)
        self._position_resize_handle()

    def finish_resize_drag(self) -> None:
        self._resize_drag_active = False

    def _content_height(self) -> int:
        return int(self.document().size().height()) + 30

    def _sync_height_to_content(self) -> None:
        if self._resize_drag_active:
            return
        auto_height = max(
            self._minimum_editor_height,
            min(self._maximum_editor_height, self._content_height()),
        )
        target_height = max(auto_height, self._manual_height)
        if self.height() != target_height:
            self.setFixedHeight(target_height)
        self._position_resize_handle()

    def _position_resize_handle(self) -> None:
        margin = 4
        self._resize_handle.move(
            max(
                margin,
                self.viewport().width() - self._resize_handle.width() - margin,
            ),
            max(
                margin,
                self.viewport().height()
                - self._resize_handle.height()
                - margin,
            ),
        )
        self._resize_handle.raise_()


class _PPOCRRichTextEdit(QTextEdit):
    def keyPressEvent(self, event) -> None:
        if event.key() not in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            super().keyPressEvent(event)
            return

        current_cursor = self.textCursor()
        should_reset_heading = current_cursor.blockFormat().headingLevel() > 0
        super().keyPressEvent(event)
        if not should_reset_heading:
            return

        cursor = self.textCursor()
        block_format = cursor.blockFormat()
        block_format.setHeadingLevel(0)
        cursor.setBlockFormat(block_format)

        char_format = QTextCharFormat()
        font = QFont()
        font.setPointSize(PPOCR_EDITOR_BODY_PT)
        font.setWeight(QFont.Weight.Normal)
        char_format.setFont(font)
        cursor.setCharFormat(char_format)
        self.setCurrentCharFormat(char_format)
        self.setTextCursor(cursor)


class PPOCRTextBlockEditor(QFrame):
    saveRequested = pyqtSignal(str)
    cancelRequested = pyqtSignal()

    def __init__(
        self,
        content: str,
        parent=None,
        image_path: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._heading_actions: dict[int, QAction] = {}
        self._image_height = 0

        self._build_ui()
        self._load_content(content)
        self._load_image_preview(image_path)
        self._connect_signals()
        self._update_toolbar_state()

    def _build_ui(self) -> None:
        self.setObjectName("PPOCRTextBlockEditor")
        self.setStyleSheet(self._frame_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QFrame(self)
        toolbar.setObjectName("PPOCRTextBlockEditorToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 10, 12, 10)
        toolbar_layout.setSpacing(6)

        self.heading_button = self._build_heading_button()
        self.bold_button = self._build_format_button("B")
        bold_font = self.bold_button.font()
        bold_font.setWeight(QFont.Weight.Bold)
        self.bold_button.setFont(bold_font)

        self.italic_button = self._build_format_button("I")
        italic_font = self.italic_button.font()
        italic_font.setItalic(True)
        self.italic_button.setFont(italic_font)

        self.strike_button = self._build_format_button("S")
        strike_font = self.strike_button.font()
        strike_font.setStrikeOut(True)
        self.strike_button.setFont(strike_font)

        toolbar_layout.addWidget(self.heading_button)
        toolbar_layout.addWidget(self.bold_button)
        toolbar_layout.addWidget(self.italic_button)
        toolbar_layout.addWidget(self.strike_button)
        toolbar_layout.addStretch()

        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(get_secondary_button_style())
        self.save_button = QPushButton(self.tr("Save"))
        self.save_button.setStyleSheet(get_primary_button_style())

        toolbar_layout.addWidget(self.cancel_button)
        toolbar_layout.addWidget(self.save_button)

        body = QFrame(self)
        body.setObjectName("PPOCRTextBlockEditorBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(16, 16, 16, 16)
        body_layout.setSpacing(12)

        self.image_label = QLabel(body)
        self.image_label.setObjectName("PPOCRTextBlockEditorImage")
        self.image_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.image_label.hide()

        self.editor = _PPOCRRichTextEdit(body)
        self.editor.setAcceptRichText(True)
        self.editor.setMinimumHeight(180)
        self.editor.setPlaceholderText(self.tr("Enter content"))
        self.editor.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.editor.document().setDocumentMargin(0)
        default_font = QFont()
        default_font.setPointSize(PPOCR_EDITOR_BODY_PT)
        self.editor.setFont(default_font)
        self.editor.document().setDefaultFont(default_font)
        self.editor.setStyleSheet(self._editor_style())

        body_layout.addWidget(self.image_label)
        body_layout.addWidget(self.editor)

        layout.addWidget(toolbar)
        layout.addWidget(body)

    def _build_heading_button(self) -> QToolButton:
        button = QToolButton(self)
        button.setText("Tt")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setStyleSheet(self._heading_button_style(False))

        menu = QMenu(button)
        menu.setStyleSheet(self._menu_style())
        for level in range(1, 7):
            action = QAction(PPOCR_EDITOR_HEADING_LEVELS[level]["label"], menu)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, heading_level=level: self._apply_heading(
                    heading_level
                )
            )
            menu.addAction(action)
            self._heading_actions[level] = action
        button.setMenu(menu)
        return button

    def _build_format_button(self, text: str) -> QToolButton:
        button = QToolButton(self)
        button.setText(text)
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setStyleSheet(self._format_button_style())
        return button

    def _connect_signals(self) -> None:
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        self.save_button.clicked.connect(self._emit_save)
        self.bold_button.clicked.connect(self._toggle_bold)
        self.italic_button.clicked.connect(self._toggle_italic)
        self.strike_button.clicked.connect(self._toggle_strikethrough)
        self.editor.cursorPositionChanged.connect(self._update_toolbar_state)
        self.editor.currentCharFormatChanged.connect(
            lambda _fmt: self._update_toolbar_state()
        )

        shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        shortcut.activated.connect(self._toggle_bold)
        shortcut = QShortcut(QKeySequence("Ctrl+I"), self)
        shortcut.activated.connect(self._toggle_italic)
        shortcut = QShortcut(QKeySequence("Ctrl+Shift+X"), self)
        shortcut.activated.connect(self._toggle_strikethrough)
        shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        shortcut.activated.connect(self._emit_save)

    def _load_content(self, content: str) -> None:
        set_markdown = getattr(self.editor, "setMarkdown", None)
        if callable(set_markdown):
            set_markdown(content or "")
        else:
            self.editor.setPlainText(content or "")
        self.editor.moveCursor(QTextCursor.MoveOperation.End)

    def _load_image_preview(self, image_path: str | Path | None) -> None:
        if not image_path:
            return
        image_file = Path(image_path)
        if not image_file.exists():
            return
        pixmap = QPixmap(str(image_file))
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(
            PPOCR_EDITOR_IMAGE_MAX_WIDTH,
            PPOCR_EDITOR_IMAGE_MAX_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_height = scaled.height()
        self.image_label.setPixmap(scaled)
        self.image_label.show()

    def _emit_save(self) -> None:
        self.saveRequested.emit(self._markdown_text())

    def _markdown_text(self) -> str:
        to_markdown = getattr(self.editor, "toMarkdown", None)
        if callable(to_markdown):
            return to_markdown().rstrip("\n")
        document = self.editor.document()
        if hasattr(document, "toMarkdown"):
            return document.toMarkdown().rstrip("\n")
        return self.editor.toPlainText()

    def _merge_char_format(self, char_format: QTextCharFormat) -> None:
        cursor = self.editor.textCursor()
        if not cursor.hasSelection():
            cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        cursor.mergeCharFormat(char_format)
        self.editor.mergeCurrentCharFormat(char_format)
        self.editor.setFocus()

    def _toggle_bold(self) -> None:
        char_format = QTextCharFormat()
        is_bold = (
            self.editor.currentCharFormat().fontWeight()
            >= QFont.Weight.Bold.value
        )
        char_format.setFontWeight(
            QFont.Weight.Normal if is_bold else QFont.Weight.Bold
        )
        self._merge_char_format(char_format)

    def _toggle_italic(self) -> None:
        char_format = QTextCharFormat()
        char_format.setFontItalic(
            not self.editor.currentCharFormat().fontItalic()
        )
        self._merge_char_format(char_format)

    def _toggle_strikethrough(self) -> None:
        char_format = QTextCharFormat()
        char_format.setFontStrikeOut(
            not self.editor.currentCharFormat().fontStrikeOut()
        )
        self._merge_char_format(char_format)

    def _apply_heading(self, level: int) -> None:
        current_level = self._detect_heading_level()
        target_level = 0 if current_level == level else level

        cursor = self.editor.textCursor()
        cursor.beginEditBlock()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        cursor.movePosition(
            QTextCursor.MoveOperation.EndOfBlock,
            QTextCursor.MoveMode.KeepAnchor,
        )

        block_format = QTextBlockFormat()
        block_format.setHeadingLevel(target_level)
        cursor.mergeBlockFormat(block_format)

        char_format = QTextCharFormat()
        font = QFont()
        if target_level == 0:
            font.setPointSize(PPOCR_EDITOR_BODY_PT)
            font.setWeight(QFont.Weight.Normal)
        else:
            heading_info = PPOCR_EDITOR_HEADING_LEVELS[target_level]
            font.setPointSize(heading_info["point_size"])
            font.setWeight(heading_info["weight"])
        char_format.setFont(font)
        cursor.mergeCharFormat(char_format)
        cursor.endEditBlock()

        self.editor.setTextCursor(cursor)
        self.editor.mergeCurrentCharFormat(char_format)
        self.editor.setFocus()
        self._update_toolbar_state()

    def _detect_heading_level(self) -> int:
        block_level = self.editor.textCursor().blockFormat().headingLevel()
        if block_level in PPOCR_EDITOR_HEADING_LEVELS:
            return block_level

        point_size = self.editor.currentCharFormat().font().pointSize()
        for level, heading_info in PPOCR_EDITOR_HEADING_LEVELS.items():
            if abs(point_size - heading_info["point_size"]) <= 1:
                return level
        return 0

    def _update_toolbar_state(self) -> None:
        char_format = self.editor.currentCharFormat()
        self.bold_button.setChecked(
            char_format.fontWeight() >= QFont.Weight.Bold.value
        )
        self.italic_button.setChecked(char_format.fontItalic())
        self.strike_button.setChecked(char_format.fontStrikeOut())

        heading_level = self._detect_heading_level()
        self.heading_button.setStyleSheet(
            self._heading_button_style(heading_level > 0)
        )
        for level, action in self._heading_actions.items():
            action.setChecked(level == heading_level)

    def content_height_valid(self) -> bool:
        text = self._markdown_text().strip()
        text_height = 0
        if text:
            text_height = max(
                24, int(self.editor.document().size().height()) + 6
            )
        image_spacing = 12 if self._image_height and text_height else 0
        content_height = 32 + self._image_height + image_spacing + text_height
        return content_height <= PPOCR_BLOCK_CARD_MAX_HEIGHT_PX

    def _frame_style(self) -> str:
        theme = get_theme()
        return (
            "QFrame#PPOCRTextBlockEditor {"
            f"background: {theme['background']};"
            "border: 1px solid rgb(220, 228, 241);"
            "border-radius: 12px;"
            "}"
            "QFrame#PPOCRTextBlockEditorToolbar {"
            "background: rgb(246, 248, 252);"
            "border: none;"
            "border-top-left-radius: 12px;"
            "border-top-right-radius: 12px;"
            "border-bottom: 1px solid rgb(229, 234, 244);"
            "}"
            "QFrame#PPOCRTextBlockEditorBody {"
            f"background: {theme['background']};"
            "border: none;"
            "border-bottom-left-radius: 12px;"
            "border-bottom-right-radius: 12px;"
            "}"
            "QLabel#PPOCRTextBlockEditorImage {"
            "background: transparent;"
            "border: none;"
            "padding: 0px;"
            "}"
        )

    def _editor_style(self) -> str:
        theme = get_theme()
        return (
            "QTextEdit {"
            f"background: {theme['background']};"
            f"color: {theme['text']};"
            "border: none;"
            "padding: 0px;"
            "selection-background-color: rgb(70, 88, 255);"
            "}"
            "QScrollBar:vertical {"
            "width: 6px;"
            "background: transparent;"
            "margin: 2px 0px;"
            "}"
            "QScrollBar::handle:vertical {"
            "background: rgb(204, 212, 229);"
            "border-radius: 3px;"
            "min-height: 24px;"
            "}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
            "height: 0px;"
            "}"
        )

    def _format_button_style(self) -> str:
        theme = get_theme()
        return (
            "QToolButton {"
            "background: transparent;"
            "border: none;"
            "border-radius: 6px;"
            f"color: {theme['text_secondary']};"
            "min-width: 30px;"
            "min-height: 30px;"
            "font-size: 15px;"
            "}"
            "QToolButton:hover {"
            "background: rgb(241, 245, 255);"
            f"color: {PPOCR_COLOR_TEXT};"
            "}"
            "QToolButton:checked {"
            "background: rgb(241, 245, 255);"
            f"color: {PPOCR_COLOR_TEXT};"
            "}"
        )

    def _heading_button_style(self, active: bool) -> str:
        theme = get_theme()
        background = "rgb(241, 245, 255)" if active else "transparent"
        color = PPOCR_COLOR_TEXT if active else theme["text_secondary"]
        return (
            "QToolButton {"
            f"background: {background};"
            "border: none;"
            "border-radius: 6px;"
            f"color: {color};"
            "min-width: 34px;"
            "min-height: 30px;"
            "font-size: 15px;"
            "font-weight: 600;"
            "}"
            "QToolButton:hover {"
            "background: rgb(241, 245, 255);"
            f"color: {PPOCR_COLOR_TEXT};"
            "}"
            "QToolButton::menu-indicator {"
            "width: 0px;"
            "}"
        )

    def _menu_style(self) -> str:
        theme = get_theme()
        return (
            "QMenu {"
            f"background: {theme['background']};"
            f"color: {theme['text']};"
            "border: 1px solid rgb(220, 228, 241);"
            "padding: 6px;"
            "}"
            "QMenu::item {"
            "padding: 6px 18px;"
            "border-radius: 6px;"
            "}"
            "QMenu::item:selected {"
            "background: rgb(241, 245, 255);"
            f"color: {PPOCR_COLOR_TEXT};"
            "}"
            "QMenu::indicator {"
            "width: 12px;"
            "height: 12px;"
            "}"
        )


def _new_rich_editor_pixmap() -> QPixmap:
    real_size = PPOCR_RICH_EDITOR_ICON_SIZE * PPOCR_RICH_EDITOR_ICON_DPR
    pixmap = QPixmap(real_size, real_size)
    pixmap.setDevicePixelRatio(PPOCR_RICH_EDITOR_ICON_DPR)
    pixmap.fill(Qt.GlobalColor.transparent)
    return pixmap


def _begin_rich_editor_painter(pixmap: QPixmap) -> QPainter:
    painter = QPainter(pixmap)
    painter.setRenderHints(
        QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
    )
    painter.setPen(QColor(PPOCR_RICH_EDITOR_ICON_COLOR))
    return painter


def _make_rich_editor_char_icon(
    char: str,
    *,
    bold: bool = False,
    italic: bool = False,
) -> QIcon:
    pixmap = _new_rich_editor_pixmap()
    painter = _begin_rich_editor_painter(pixmap)
    font = QFont(PPOCR_RICH_EDITOR_SERIF)
    font.setPixelSize(18)
    if bold:
        font.setWeight(QFont.Weight.Bold)
    if italic:
        font.setItalic(True)
    painter.setFont(font)
    painter.drawText(
        QRectF(
            0,
            0,
            PPOCR_RICH_EDITOR_ICON_SIZE,
            PPOCR_RICH_EDITOR_ICON_SIZE,
        ),
        Qt.AlignmentFlag.AlignCenter,
        char,
    )
    painter.end()
    return QIcon(pixmap)


def _make_rich_editor_heading_icon() -> QIcon:
    size = PPOCR_RICH_EDITOR_ICON_SIZE
    pixmap = _new_rich_editor_pixmap()
    painter = _begin_rich_editor_painter(pixmap)

    primary_font = QFont(PPOCR_RICH_EDITOR_SERIF)
    primary_font.setPixelSize(17)
    primary_font.setWeight(QFont.Weight.Medium)
    painter.setFont(primary_font)
    primary_metrics = painter.fontMetrics()
    primary_width = primary_metrics.horizontalAdvance("T")

    secondary_font = QFont(PPOCR_RICH_EDITOR_SERIF)
    secondary_font.setPixelSize(12)
    painter.setFont(secondary_font)
    secondary_width = painter.fontMetrics().horizontalAdvance("T")

    total_width = primary_width + secondary_width - 2
    start_x = (size - total_width) / 2
    baseline = (
        size + primary_metrics.ascent() - primary_metrics.descent()
    ) / 2

    painter.setFont(primary_font)
    painter.drawText(QPointF(start_x, baseline), "T")
    painter.setFont(secondary_font)
    painter.drawText(QPointF(start_x + primary_width - 2, baseline), "T")
    painter.end()
    return QIcon(pixmap)


def _make_rich_editor_strike_icon() -> QIcon:
    size = PPOCR_RICH_EDITOR_ICON_SIZE
    pixmap = _new_rich_editor_pixmap()
    painter = _begin_rich_editor_painter(pixmap)
    font = QFont(PPOCR_RICH_EDITOR_SERIF)
    font.setPixelSize(18)
    painter.setFont(font)
    painter.drawText(
        QRectF(0, 0, size, size),
        Qt.AlignmentFlag.AlignCenter,
        "S",
    )
    painter.setPen(QPen(QColor(PPOCR_RICH_EDITOR_ICON_COLOR), 1.3))
    middle_y = size / 2
    painter.drawLine(QPointF(3, middle_y), QPointF(size - 3, middle_y))
    painter.end()
    return QIcon(pixmap)


class _PPOCRStyledRichTextEdit(QTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._selected_image_pos = -1

    def keyPressEvent(self, event) -> None:
        if event.key() not in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            super().keyPressEvent(event)
            return

        cursor = self.textCursor()
        current_block = cursor.block()
        if _block_contains_only_image(current_block):
            next_block = current_block.next()
            if (
                next_block.isValid()
                and not next_block.text()
                and not _block_contains_only_image(next_block)
            ):
                next_cursor = QTextCursor(next_block)
                next_cursor.movePosition(
                    QTextCursor.MoveOperation.StartOfBlock
                )
                block_format = next_cursor.blockFormat()
                block_format.setHeadingLevel(0)
                _apply_rich_block_metrics(block_format, 0)
                next_cursor.setBlockFormat(block_format)
                self.setTextCursor(next_cursor)
                self._apply_body_char_format(next_cursor)
                event.accept()
                return
        should_reset_heading = cursor.blockFormat().headingLevel() > 0
        should_exit_image_block = _block_contains_only_image(current_block)
        super().keyPressEvent(event)

        cursor = self.textCursor()
        block_format = cursor.blockFormat()
        if should_reset_heading or should_exit_image_block:
            block_format.setHeadingLevel(0)
        _apply_rich_block_metrics(block_format, block_format.headingLevel())
        cursor.setBlockFormat(block_format)

        if should_reset_heading or should_exit_image_block:
            self._apply_body_char_format(cursor)

        self.setTextCursor(cursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            image_pos = self._image_at(event.pos())
            previous_image_pos = self._selected_image_pos
            self._selected_image_pos = image_pos
            if image_pos != previous_image_pos:
                self.viewport().update()
            if image_pos >= 0:
                event.accept()
                return
        super().mousePressEvent(event)

    def canInsertFromMimeData(self, source) -> bool:
        sanitized_source = self._sanitized_mime_source(source)
        if sanitized_source is None:
            return False
        return super().canInsertFromMimeData(sanitized_source)

    def insertFromMimeData(self, source) -> None:
        sanitized_source = self._sanitized_mime_source(source)
        if sanitized_source is None:
            return
        pasted_headings = (
            self._pasted_block_headings(sanitized_source.html())
            if sanitized_source.hasHtml()
            else []
        )
        start_block_number = self.textCursor().block().blockNumber()
        if self._should_start_new_block_for_paste(sanitized_source):
            cursor = self.textCursor()
            cursor.insertBlock()
            block_format = cursor.blockFormat()
            block_format.setHeadingLevel(0)
            _apply_rich_block_metrics(block_format, 0)
            cursor.setBlockFormat(block_format)
            self._apply_body_char_format(cursor)
            self.setTextCursor(cursor)
            start_block_number = cursor.block().blockNumber()
        super().insertFromMimeData(sanitized_source)
        self._normalize_document_blocks()
        if pasted_headings:
            self._restore_pasted_block_headings(
                start_block_number, pasted_headings
            )
        cursor = self.textCursor()
        if not cursor.charFormat().isImageFormat():
            self._apply_body_char_format(cursor)
        self.setTextCursor(cursor)

    def _contains_image_urls(self, source) -> bool:
        if not source.hasUrls():
            return False
        for url in source.urls():
            local_file = url.toLocalFile()
            if not local_file:
                continue
            if Path(local_file).suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".gif",
                ".webp",
                ".tif",
                ".tiff",
            }:
                return True
        return False

    def _sanitized_mime_source(self, source):
        if self._contains_image_urls(source):
            return None
        if not source.hasHtml():
            if not source.hasImage():
                return source
            if not source.hasText() or not source.text().strip():
                return None
            sanitized_source = QMimeData()
            sanitized_source.setText(source.text())
            return sanitized_source
        html_text = source.html()
        if source.hasImage() or re.search(
            r"<img\b", html_text, flags=re.IGNORECASE
        ):
            sanitized_html = re.sub(
                r"<img\b[^>]*>", "", html_text, flags=re.IGNORECASE
            )
            if not re.sub(r"<[^>]+>", "", sanitized_html).strip():
                if not source.hasText() or not source.text().strip():
                    return None
                sanitized_source = QMimeData()
                sanitized_source.setText(source.text())
                return sanitized_source
            sanitized_source = QMimeData()
            sanitized_source.setHtml(sanitized_html)
            if source.hasText():
                sanitized_source.setText(source.text())
            return sanitized_source
        return source

    def _image_at(self, point) -> int:
        target_point = QPointF(point)
        block = self.document().begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if (
                    fragment.isValid()
                    and fragment.charFormat().isImageFormat()
                ):
                    image_rect = self._image_rect(fragment.position())
                    if image_rect and image_rect.contains(target_point):
                        return fragment.position()
                iterator += 1
            block = block.next()
        return -1

    def _image_rect(self, pos: int) -> QRectF | None:
        document = self.document()
        if pos + 1 >= document.characterCount():
            return None
        end_cursor = QTextCursor(document)
        end_cursor.setPosition(pos + 1)
        if not end_cursor.charFormat().isImageFormat():
            return None
        image_format = end_cursor.charFormat().toImageFormat()

        start_cursor = QTextCursor(document)
        start_cursor.setPosition(pos)
        start_rect = self.cursorRect(start_cursor)

        end_rect = self.cursorRect(end_cursor)
        width = (
            (end_rect.x() - start_rect.x())
            if abs(end_rect.y() - start_rect.y()) < 4
            else image_format.width()
        )
        return QRectF(
            start_rect.x(), start_rect.y(), width, start_rect.height()
        )

    def _apply_body_char_format(
        self, cursor: QTextCursor | None = None
    ) -> None:
        target_cursor = (
            QTextCursor(cursor) if cursor is not None else self.textCursor()
        )
        char_format = QTextCharFormat()
        font = QFont()
        font.setPointSize(PPOCR_RICH_EDITOR_BODY_PT)
        font.setWeight(QFont.Weight.Normal)
        char_format.setFont(font)
        char_format.setForeground(PPOCR_RICH_EDITOR_TEXT_COLOR)
        target_cursor.setCharFormat(char_format)
        self.setCurrentCharFormat(char_format)
        self.setTextCursor(target_cursor)

    def _normalize_document_blocks(self) -> None:
        cursor = QTextCursor(self.document())
        block = self.document().begin()
        while block.isValid():
            block_format = block.blockFormat()
            _apply_rich_block_metrics(
                block_format,
                block_format.headingLevel(),
                image_only=_block_contains_only_image(block),
            )
            cursor.setPosition(block.position())
            cursor.mergeBlockFormat(block_format)
            block = block.next()

    def _should_start_new_block_for_paste(self, source) -> bool:
        cursor = self.textCursor()
        if not (cursor.positionInBlock() > 0 or cursor.block().text()):
            return False
        if source.hasHtml():
            html_text = source.html()
            return bool(
                re.search(
                    r"<(?:h[1-6]|p|div|ul|ol|li|blockquote|img)\b",
                    html_text,
                    flags=re.IGNORECASE,
                )
            )
        if not source.hasText():
            return False
        text = source.text().lstrip()
        return bool(
            "\n" in text or re.match(r"(#{1,6}\s|!\[|\d+\.\s|[-*+]\s)", text)
        )

    def _pasted_block_headings(self, html_text: str) -> list[int]:
        block_tags = re.findall(
            r"<(h[1-6]|p|div|li|blockquote)\b",
            html_text,
            flags=re.IGNORECASE,
        )
        heading_levels = []
        for tag in block_tags:
            lowered_tag = tag.lower()
            if lowered_tag.startswith("h") and lowered_tag[1:].isdigit():
                heading_levels.append(int(lowered_tag[1:]))
            else:
                heading_levels.append(0)
        return heading_levels

    def _restore_pasted_block_headings(
        self,
        start_block_number: int,
        heading_levels: list[int],
    ) -> None:
        document = self.document()
        for index, heading_level in enumerate(heading_levels):
            if heading_level <= 0:
                continue
            block = document.findBlockByNumber(start_block_number + index)
            if not block.isValid():
                break
            cursor = QTextCursor(block)
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            cursor.movePosition(
                QTextCursor.MoveOperation.EndOfBlock,
                QTextCursor.MoveMode.KeepAnchor,
            )
            block_format = cursor.blockFormat()
            previous_heading_level = block_format.headingLevel()
            block_format.setHeadingLevel(heading_level)
            _apply_rich_block_metrics(block_format, heading_level)
            cursor.setBlockFormat(block_format)
            _apply_heading_fragment_formats(
                cursor,
                previous_heading_level,
                heading_level,
            )

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._selected_image_pos < 0:
            return
        image_rect = self._image_rect(self._selected_image_pos)
        if image_rect is None:
            self._selected_image_pos = -1
            return
        frame_padding = 4.0
        frame_width = 3.0
        frame_radius = 10.0
        frame_rect = image_rect.adjusted(
            -frame_padding,
            -frame_padding,
            frame_padding,
            frame_padding,
        )
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#2932e1"), frame_width))
        painter.drawRoundedRect(frame_rect, frame_radius, frame_radius)
        painter.end()


class _PPOCRRichHeadingMenuItem(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, level: int, parent=None) -> None:
        super().__init__(parent)
        self._level = level
        self._hovered = False
        heading_info = PPOCR_RICH_EDITOR_HEADING_LEVELS[level]

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(44)
        self.setMinimumWidth(130)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(0)

        self._label = QLabel(heading_info["label"])
        font = QFont()
        font.setPointSize(heading_info["point_size"])
        font.setWeight(heading_info["weight"])
        self._label.setFont(font)
        self._label.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(self._label)
        layout.addStretch()

    def set_selected(self, selected: bool) -> None:
        color = "#2932e1" if selected else "rgba(0, 0, 0, 0.85)"
        self._label.setStyleSheet(
            f"color: {color}; background: transparent; border: none;"
        )

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        if self._hovered:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#f5f5f5"))
            painter.drawRoundedRect(self.rect(), 4, 4)
            painter.end()
        super().paintEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._level)
        super().mousePressEvent(event)


class _PPOCRRichHeadingDropdown(QWidget):
    headingSelected = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Popup
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self._items: list[_PPOCRRichHeadingMenuItem] = []
        for level in range(1, 7):
            item = _PPOCRRichHeadingMenuItem(level, self)
            item.clicked.connect(self._on_item_clicked)
            layout.addWidget(item)
            self._items.append(item)

        self.adjustSize()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#e8e8e8"), 1))
        painter.setBrush(QColor("#ffffff"))
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.drawRoundedRect(rect, 6, 6)
        painter.end()

    def set_current_level(self, level: int) -> None:
        for item in self._items:
            item.set_selected(item._level == level)

    def _on_item_clicked(self, level: int) -> None:
        self.headingSelected.emit(level)
        self.close()

    def show_below(self, widget: QWidget) -> None:
        self.move(widget.mapToGlobal(QPoint(0, widget.height() + 2)))
        self.show()


class PPOCRRichTextBlockEditor(QFrame):
    saveRequested = pyqtSignal(str)
    cancelRequested = pyqtSignal()

    def __init__(
        self,
        content: str,
        parent=None,
        image_path: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._image_height = 0
        self._current_heading_level = 0
        self._tracked_scroll_viewport = None

        self._build_ui()
        self._load_content(content)
        self._load_image_preview(image_path)
        self._connect_signals()
        self._update_toolbar_state()
        QTimer.singleShot(0, self._update_editor_height)

    def _build_ui(self) -> None:
        self.setObjectName("PPOCRRichTextBlockEditor")
        self.setStyleSheet(PPOCR_RICH_EDITOR_CONTAINER_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QFrame(self)
        toolbar.setObjectName("PPOCRRichTextBlockEditorToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 0, 12, 0)
        toolbar_layout.setSpacing(4)

        icon_size = QSize(
            PPOCR_RICH_EDITOR_ICON_SIZE,
            PPOCR_RICH_EDITOR_ICON_SIZE,
        )

        self.heading_button = QToolButton(toolbar)
        self.heading_button.setIcon(_make_rich_editor_heading_icon())
        self.heading_button.setIconSize(icon_size)
        self.heading_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.heading_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.heading_button.setStyleSheet(
            PPOCR_RICH_EDITOR_HEADING_BUTTON_STYLE
        )

        self.heading_menu = _PPOCRRichHeadingDropdown()

        self.bold_button = self._build_format_button(
            _make_rich_editor_char_icon("B", bold=True)
        )
        self.italic_button = self._build_format_button(
            _make_rich_editor_char_icon("I", italic=True)
        )
        self.strike_button = self._build_format_button(
            _make_rich_editor_strike_icon()
        )

        toolbar_layout.addWidget(self.heading_button)
        toolbar_layout.addWidget(self.bold_button)
        toolbar_layout.addWidget(self.italic_button)
        toolbar_layout.addWidget(self.strike_button)
        toolbar_layout.addStretch()

        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(get_secondary_button_style())
        self.save_button = QPushButton(self.tr("Save"))
        self.save_button.setStyleSheet(get_primary_button_style())

        toolbar_layout.addWidget(self.cancel_button)
        toolbar_layout.addSpacing(12)
        toolbar_layout.addWidget(self.save_button)

        body = QFrame(self)
        body.setObjectName("PPOCRRichTextBlockEditorBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.image_label = QLabel(body)
        self.image_label.setObjectName("PPOCRRichTextBlockEditorImage")
        self.image_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.image_label.setContentsMargins(24, 24, 24, 0)
        self.image_label.hide()

        self.editor = _PPOCRStyledRichTextEdit(body)
        self.editor.setAcceptRichText(True)
        self.editor.setMinimumHeight(PPOCR_RICH_EDITOR_MIN_HEIGHT_PX)
        self.editor.setPlaceholderText(self.tr("Enter content"))
        self.editor.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.editor.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.editor.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.editor.document().setDocumentMargin(0)
        self.editor.setStyleSheet(PPOCR_RICH_EDITOR_STYLE)
        default_font = QFont()
        default_font.setPointSize(PPOCR_RICH_EDITOR_BODY_PT)
        default_font.setWeight(QFont.Weight.Normal)
        self.editor.setFont(default_font)
        self.editor.document().setDefaultFont(default_font)
        self.editor.setTextColor(PPOCR_RICH_EDITOR_TEXT_COLOR)

        body_layout.addWidget(self.image_label)
        body_layout.addWidget(self.editor)

        layout.addWidget(toolbar)
        layout.addWidget(body)

    def _build_format_button(self, icon: QIcon) -> QToolButton:
        button = QToolButton(self)
        button.setIcon(icon)
        button.setIconSize(
            QSize(
                PPOCR_RICH_EDITOR_ICON_SIZE,
                PPOCR_RICH_EDITOR_ICON_SIZE,
            )
        )
        button.setCheckable(True)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setStyleSheet(PPOCR_RICH_EDITOR_FORMAT_BUTTON_STYLE)
        return button

    def _connect_signals(self) -> None:
        self.heading_button.clicked.connect(self._show_heading_menu)
        self.heading_menu.headingSelected.connect(self._apply_heading)
        self.bold_button.clicked.connect(self._toggle_bold)
        self.italic_button.clicked.connect(self._toggle_italic)
        self.strike_button.clicked.connect(self._toggle_strikethrough)
        self.save_button.clicked.connect(self._emit_save)
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        self.editor.cursorPositionChanged.connect(self._update_toolbar_state)
        self.editor.currentCharFormatChanged.connect(
            lambda _fmt: self._update_toolbar_state()
        )
        self.editor.document().documentLayout().documentSizeChanged.connect(
            lambda _size: self._update_editor_height()
        )
        shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        shortcut.activated.connect(self._toggle_bold)
        shortcut = QShortcut(QKeySequence("Ctrl+I"), self)
        shortcut.activated.connect(self._toggle_italic)
        shortcut = QShortcut(QKeySequence("Ctrl+Shift+X"), self)
        shortcut.activated.connect(self._toggle_strikethrough)
        shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        shortcut.activated.connect(self._emit_save)

    def _load_content(self, content: str) -> None:
        if is_rich_text_html(content):
            self.editor.setHtml(content)
        else:
            set_markdown = getattr(self.editor, "setMarkdown", None)
            if callable(set_markdown):
                set_markdown(content or "")
            else:
                self.editor.setPlainText(content or "")
        self._trim_edge_empty_blocks()
        self.editor.moveCursor(QTextCursor.MoveOperation.End)
        self._update_editor_height()

    def _load_image_preview(self, image_path: str | Path | None) -> None:
        if not image_path:
            return
        image_file = Path(image_path)
        if not image_file.exists():
            return
        pixmap = QPixmap(str(image_file))
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(
            PPOCR_EDITOR_IMAGE_MAX_WIDTH,
            PPOCR_EDITOR_IMAGE_MAX_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_height = scaled.height()
        self.image_label.setPixmap(scaled)
        self.image_label.show()
        self._update_editor_height()

    def _show_heading_menu(self) -> None:
        if self.heading_menu.isVisible():
            self.heading_menu.close()
            return
        self.heading_menu.set_current_level(self._current_heading_level)
        self.heading_button.setStyleSheet(
            PPOCR_RICH_EDITOR_HEADING_BUTTON_ACTIVE_STYLE
        )
        self.heading_menu.show_below(self.heading_button)
        QTimer.singleShot(50, self._poll_heading_menu_close)

    def _poll_heading_menu_close(self) -> None:
        if self.heading_menu.isVisible():
            QTimer.singleShot(50, self._poll_heading_menu_close)
            return
        self._update_heading_button_style()

    def _emit_save(self) -> None:
        self.saveRequested.emit(self._serialized_text())

    def _serialized_text(self) -> str:
        markdown_text = self._markdown_text()
        if not self._document_has_strikeout():
            return markdown_text
        return self._html_fragment_text()

    def _markdown_text(self) -> str:
        to_markdown = getattr(self.editor, "toMarkdown", None)
        if callable(to_markdown):
            return to_markdown().rstrip("\n")
        document = self.editor.document()
        if hasattr(document, "toMarkdown"):
            return document.toMarkdown().rstrip("\n")
        return self.editor.toPlainText()

    def _document_has_strikeout(self) -> bool:
        block = self.editor.document().begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if (
                    fragment.isValid()
                    and fragment.charFormat().fontStrikeOut()
                ):
                    return True
                iterator += 1
            block = block.next()
        return False

    def _html_fragment_text(self) -> str:
        cursor = QTextCursor(self.editor.document())
        cursor.select(QTextCursor.SelectionType.Document)
        fragment = cursor.selection().toHtml().strip()
        body_match = re.search(
            r"<body[^>]*>(.*)</body>",
            fragment,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if body_match:
            fragment = body_match.group(1).strip()
        fragment = fragment.replace("<!--StartFragment-->", "")
        fragment = fragment.replace("<!--EndFragment-->", "")
        return fragment.strip()

    def _normalize_document_block_formats(self) -> None:
        cursor = QTextCursor(self.editor.document())
        block = self.editor.document().begin()
        while block.isValid():
            block_format = block.blockFormat()
            _apply_rich_block_metrics(
                block_format,
                block_format.headingLevel(),
                image_only=_block_contains_only_image(block),
            )
            cursor.setPosition(block.position())
            cursor.mergeBlockFormat(block_format)
            if not _block_contains_only_image(block):
                block_cursor = QTextCursor(block)
                block_cursor.movePosition(
                    QTextCursor.MoveOperation.StartOfBlock
                )
                block_cursor.movePosition(
                    QTextCursor.MoveOperation.EndOfBlock,
                    QTextCursor.MoveMode.KeepAnchor,
                )
                _apply_heading_fragment_formats(
                    block_cursor,
                    block_format.headingLevel(),
                    block_format.headingLevel(),
                )
            block = block.next()

    def _trim_edge_empty_blocks(self) -> None:
        document = self.editor.document()
        while document.blockCount() > 1:
            first_block = document.begin()
            if not _block_is_effectively_empty(first_block):
                break
            next_block = first_block.next()
            cursor = QTextCursor(document)
            cursor.setPosition(first_block.position())
            cursor.setPosition(
                next_block.position(),
                QTextCursor.MoveMode.KeepAnchor,
            )
            cursor.removeSelectedText()
        while document.blockCount() > 1:
            last_block = document.lastBlock()
            if not _block_is_effectively_empty(last_block):
                break
            previous_block = last_block.previous()
            cursor = QTextCursor(document)
            cursor.setPosition(
                previous_block.position() + previous_block.length() - 1
            )
            cursor.setPosition(
                last_block.position() + last_block.length() - 1,
                QTextCursor.MoveMode.KeepAnchor,
            )
            cursor.removeSelectedText()

    def _find_cards_scroll_area(self) -> QScrollArea | None:
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                return parent
            parent = parent.parentWidget()
        return None

    def _bind_scroll_viewport(self) -> None:
        scroll_area = self._find_cards_scroll_area()
        viewport = scroll_area.viewport() if scroll_area is not None else None
        if viewport is self._tracked_scroll_viewport:
            return
        if self._tracked_scroll_viewport is not None:
            self._tracked_scroll_viewport.removeEventFilter(self)
        self._tracked_scroll_viewport = viewport
        if self._tracked_scroll_viewport is not None:
            self._tracked_scroll_viewport.installEventFilter(self)

    def _max_content_height(self) -> int:
        self._bind_scroll_viewport()
        if self._tracked_scroll_viewport is None:
            return PPOCR_BLOCK_CARD_MAX_HEIGHT_PX
        viewport_height = self._tracked_scroll_viewport.height()
        if viewport_height <= 0:
            return PPOCR_BLOCK_CARD_MAX_HEIGHT_PX
        return max(
            PPOCR_RICH_EDITOR_MIN_HEIGHT_PX,
            int(viewport_height * PPOCR_RICH_EDITOR_MAX_HEIGHT_RATIO),
        )

    def _document_text_height(self) -> int:
        return max(
            PPOCR_RICH_EDITOR_MIN_HEIGHT_PX,
            int(self.editor.document().size().height())
            + PPOCR_RICH_EDITOR_HEIGHT_PADDING_PX,
        )

    def _update_editor_height(self) -> None:
        desired_height = self._document_text_height()
        image_spacing = 12 if self._image_height and desired_height else 0
        available_height = max(
            PPOCR_RICH_EDITOR_MIN_HEIGHT_PX,
            self._max_content_height()
            - PPOCR_RICH_EDITOR_BODY_OVERHEAD_PX
            - self._image_height
            - image_spacing,
        )
        target_height = min(desired_height, available_height)
        if self.editor.height() != target_height:
            self.editor.setFixedHeight(target_height)

    def _merge_char_format(self, char_format: QTextCharFormat) -> None:
        cursor = self.editor.textCursor()
        if not cursor.hasSelection():
            cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        cursor.mergeCharFormat(char_format)
        self.editor.mergeCurrentCharFormat(char_format)
        self.editor.setFocus()

    def _toggle_bold(self) -> None:
        char_format = QTextCharFormat()
        is_bold = (
            self.editor.currentCharFormat().fontWeight()
            >= QFont.Weight.Bold.value
        )
        char_format.setFontWeight(
            QFont.Weight.Normal if is_bold else QFont.Weight.Bold
        )
        self._merge_char_format(char_format)

    def _toggle_italic(self) -> None:
        char_format = QTextCharFormat()
        char_format.setFontItalic(
            not self.editor.currentCharFormat().fontItalic()
        )
        self._merge_char_format(char_format)

    def _toggle_strikethrough(self) -> None:
        char_format = QTextCharFormat()
        char_format.setFontStrikeOut(
            not self.editor.currentCharFormat().fontStrikeOut()
        )
        self._merge_char_format(char_format)

    def _apply_heading(self, level: int) -> None:
        current_level = self._detect_heading_level()
        target_level = 0 if current_level == level else level

        selected_blocks = [
            block
            for block in self._selected_blocks()
            if not _block_contains_only_image(block)
        ]
        if not selected_blocks:
            return

        original_cursor = QTextCursor(self.editor.textCursor())
        edit_cursor = QTextCursor(self.editor.textCursor())
        edit_cursor.beginEditBlock()
        for block in selected_blocks:
            block_cursor = QTextCursor(block)
            block_cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            block_cursor.movePosition(
                QTextCursor.MoveOperation.EndOfBlock,
                QTextCursor.MoveMode.KeepAnchor,
            )
            block_format = block_cursor.blockFormat()
            previous_heading_level = block_format.headingLevel()
            block_format.setHeadingLevel(target_level)
            _apply_rich_block_metrics(block_format, target_level)
            block_cursor.setBlockFormat(block_format)
            _apply_heading_fragment_formats(
                block_cursor,
                previous_heading_level,
                target_level,
            )
        edit_cursor.endEditBlock()

        self.editor.setTextCursor(original_cursor)
        current_char_format = QTextCharFormat()
        current_heading_level = self._detect_heading_level()
        if current_heading_level <= 0:
            current_char_format.setFontPointSize(PPOCR_RICH_EDITOR_BODY_PT)
            current_char_format.setFontWeight(QFont.Weight.Normal)
        else:
            heading_info = PPOCR_RICH_EDITOR_HEADING_LEVELS[
                current_heading_level
            ]
            current_char_format.setFontPointSize(heading_info["point_size"])
            current_char_format.setFontWeight(heading_info["weight"])
        current_char_format.setForeground(PPOCR_RICH_EDITOR_TEXT_COLOR)
        self.editor.setCurrentCharFormat(current_char_format)
        self.editor.setFocus()
        self._update_toolbar_state()
        self._update_editor_height()

    def _selected_blocks(self) -> list:
        cursor = self.editor.textCursor()
        if not cursor.hasSelection():
            return [cursor.block()]
        document = self.editor.document()
        start_position = min(cursor.selectionStart(), cursor.selectionEnd())
        end_position = max(cursor.selectionStart(), cursor.selectionEnd())
        if end_position > start_position:
            end_position -= 1
        start_block = document.findBlock(start_position)
        end_block = document.findBlock(end_position)
        blocks = []
        block = start_block
        while block.isValid():
            blocks.append(block)
            if block.position() >= end_block.position():
                break
            block = block.next()
        return blocks

    def _detect_heading_level(self) -> int:
        heading_levels = {
            block.blockFormat().headingLevel()
            for block in self._selected_blocks()
            if not _block_contains_only_image(block)
        }
        if len(heading_levels) != 1:
            return 0
        block_level = next(iter(heading_levels))
        return (
            block_level
            if block_level in PPOCR_RICH_EDITOR_HEADING_LEVELS
            else 0
        )

    def _update_heading_button_style(self) -> None:
        style = (
            PPOCR_RICH_EDITOR_HEADING_BUTTON_ACTIVE_STYLE
            if self._current_heading_level > 0
            else PPOCR_RICH_EDITOR_HEADING_BUTTON_STYLE
        )
        self.heading_button.setStyleSheet(style)

    def _update_toolbar_state(self) -> None:
        char_format = self.editor.currentCharFormat()
        self.bold_button.setChecked(
            char_format.fontWeight() >= QFont.Weight.Bold.value
        )
        self.italic_button.setChecked(char_format.fontItalic())
        self.strike_button.setChecked(char_format.fontStrikeOut())
        self._current_heading_level = self._detect_heading_level()
        self._update_heading_button_style()

    def content_height_valid(self) -> bool:
        self._update_editor_height()
        image_spacing = (
            12 if self._image_height and self.editor.height() > 0 else 0
        )
        content_height = (
            PPOCR_RICH_EDITOR_BODY_OVERHEAD_PX
            + self._image_height
            + image_spacing
            + self.editor.height()
        )
        return content_height <= self._max_content_height()

    def eventFilter(self, watched, event) -> bool:
        if (
            watched is self._tracked_scroll_viewport
            and event.type() == QEvent.Type.Resize
        ):
            QTimer.singleShot(0, self._update_editor_height)
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._bind_scroll_viewport()
        QTimer.singleShot(0, self._update_editor_height)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._update_editor_height)


class PPOCRLatexBlockEditor(QFrame):
    saveRequested = pyqtSignal(str)
    cancelRequested = pyqtSignal()

    def __init__(
        self,
        content: str,
        parent=None,
        image_path: str | Path | None = None,
    ) -> None:
        del image_path
        super().__init__(parent)
        self._preview_valid = False
        self._preview_pixmap = QPixmap()
        self._preview_message = ""
        self._initial_preview_pending = True
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(220)

        self._build_ui()
        self._load_content(content)
        self._connect_signals()
        self._set_preview_message(self.tr("Rendering preview..."))
        self.save_button.setEnabled(False)

    def _build_ui(self) -> None:
        self.setObjectName("PPOCRLatexBlockEditor")
        self.setStyleSheet(PPOCR_LATEX_EDITOR_CONTAINER_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QFrame(self)
        toolbar.setObjectName("PPOCRLatexBlockEditorToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(16, 8, 12, 8)
        toolbar_layout.setSpacing(8)

        title_label = QLabel("LaTeX", toolbar)
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setWeight(QFont.Weight.DemiBold)
        title_label.setFont(title_font)
        title_label.setStyleSheet(
            "QLabel { color: rgba(0, 0, 0, 0.88); background: transparent; }"
        )
        toolbar_layout.addWidget(title_label)
        toolbar_layout.addStretch()

        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(get_secondary_button_style())
        self.save_button = QPushButton(self.tr("Save"))
        self.save_button.setStyleSheet(get_primary_button_style())
        toolbar_layout.addWidget(self.cancel_button)
        toolbar_layout.addWidget(self.save_button)

        body = QFrame(self)
        body.setObjectName("PPOCRLatexBlockEditorBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 16)
        body_layout.setSpacing(0)

        source_section = QWidget(body)
        source_layout = QVBoxLayout(source_section)
        source_layout.setContentsMargins(8, 6, 8, 6)
        source_layout.setSpacing(0)

        self.source_editor = _PPOCRResizableLatexSourceEdit(body)
        self.editor = self.source_editor
        self.source_editor.setFrameShape(QFrame.Shape.NoFrame)
        self.source_editor.setStyleSheet(PPOCR_LATEX_SOURCE_STYLE)
        self.source_editor.setPlaceholderText(self.tr("Enter LaTeX source"))
        self.source_editor.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.source_editor.set_height_bounds(132, 360)
        source_layout.addWidget(self.source_editor)
        body_layout.addWidget(source_section)

        preview_section = QWidget(body)
        preview_layout = QVBoxLayout(preview_section)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)

        self.preview_divider = QFrame(preview_section)
        self.preview_divider.setObjectName("PPOCRLatexPreviewDivider")
        self.preview_divider.setFixedHeight(1)
        preview_layout.addWidget(self.preview_divider)

        self.preview_title = QLabel(self.tr("Preview"), preview_section)
        self.preview_title.setObjectName("PPOCRLatexPreviewTitle")
        preview_layout.addWidget(
            self.preview_title,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        preview_content_container = QWidget(preview_section)
        preview_content_layout = QVBoxLayout(preview_content_container)
        preview_content_layout.setContentsMargins(16, 6, 16, 0)
        preview_content_layout.setSpacing(0)

        self.preview_scroll = QScrollArea(preview_content_container)
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.preview_scroll.setStyleSheet(PPOCR_LATEX_PREVIEW_SCROLL_STYLE)
        self.preview_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.preview_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.preview_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.preview_content = QLabel()
        self.preview_content.setObjectName("PPOCRLatexPreviewContent")
        self.preview_content.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.preview_content.setWordWrap(True)
        self.preview_content.setMargin(0)
        self.preview_scroll.setWidget(self.preview_content)
        preview_content_layout.addWidget(self.preview_scroll)
        self.preview_scroll.setFixedHeight(PPOCR_LATEX_PREVIEW_MIN_HEIGHT_PX)
        preview_layout.addWidget(preview_content_container)
        body_layout.addWidget(preview_section, 1)

        layout.addWidget(toolbar)
        layout.addWidget(body)

    def _connect_signals(self) -> None:
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        self.save_button.clicked.connect(self._emit_save)
        self.source_editor.textChanged.connect(self._schedule_preview_update)
        self._preview_timer.timeout.connect(self._update_preview)
        shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        shortcut.activated.connect(self._emit_save)

    def _load_content(self, content: str) -> None:
        self.source_editor.setPlainText(content or "")
        cursor = self.source_editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.source_editor.setTextCursor(cursor)

    def _schedule_preview_update(self) -> None:
        self._preview_valid = False
        self.save_button.setEnabled(False)
        self._set_preview_message(self.tr("Rendering preview..."))
        self._preview_timer.start()

    def _set_preview_message(
        self,
        text: str,
        color: str = "rgba(0, 0, 0, 0.45)",
    ) -> None:
        self._preview_pixmap = QPixmap()
        self._preview_message = text or ""
        self.preview_content.clear()
        self.preview_content.setPixmap(QPixmap())
        self.preview_content.setText(self._preview_message)
        self.preview_content.setStyleSheet(
            "QLabel#PPOCRLatexPreviewContent {"
            "background: transparent;"
            f"color: {color};"
            "border: none;"
            "font-size: 13px;"
            "}"
        )
        self._update_preview_geometry()

    def _update_preview(self) -> None:
        source_text = self.source_editor.toPlainText()
        self._preview_valid = False
        self.save_button.setEnabled(False)
        try:
            preview_pixmap = render_latex_preview_pixmap(source_text)
        except Exception as exc:
            self._set_preview_message(str(exc), color="rgb(255, 77, 79)")
            return

        self._preview_valid = True
        self.save_button.setEnabled(True)
        self._preview_pixmap = preview_pixmap
        self._preview_message = ""
        self.preview_content.clear()
        self.preview_content.setStyleSheet(
            "QLabel#PPOCRLatexPreviewContent {"
            "background: transparent;"
            "color: rgba(0, 0, 0, 0.85);"
            "border: none;"
            "}"
        )
        self._update_preview_geometry()

    def _emit_save(self) -> None:
        if not self._preview_valid:
            return
        self.saveRequested.emit(self.source_editor.toPlainText())

    def content_height_valid(self) -> bool:
        return True

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._update_preview_geometry)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._update_preview_geometry)
        if self._initial_preview_pending:
            self._initial_preview_pending = False
            self._preview_timer.start()

    def _preview_available_width(self) -> int:
        width = self.preview_scroll.viewport().width()
        if width <= 0:
            width = self.preview_scroll.width()
        return max(48, width)

    def _update_preview_geometry(self) -> None:
        available_width = self._preview_available_width()
        if not self._preview_pixmap.isNull():
            pixmap = self._preview_pixmap
            if pixmap.width() > available_width:
                pixmap = pixmap.scaledToWidth(
                    available_width,
                    Qt.TransformationMode.SmoothTransformation,
                )
            self.preview_content.setText("")
            self.preview_content.setPixmap(pixmap)
            self.preview_content.setFixedSize(pixmap.size())
            self.preview_scroll.setFixedHeight(
                max(PPOCR_LATEX_PREVIEW_MIN_HEIGHT_PX, pixmap.height())
            )
            return

        message = self._preview_message or " "
        self.preview_content.setPixmap(QPixmap())
        self.preview_content.setText(message)
        self.preview_content.setFixedWidth(available_width)
        metrics = self.preview_content.fontMetrics()
        text_height = metrics.boundingRect(
            0,
            0,
            available_width,
            10000,
            int(Qt.TextFlag.TextWordWrap),
            message,
        ).height()
        content_height = max(
            PPOCR_LATEX_PREVIEW_MIN_HEIGHT_PX, text_height + 6
        )
        self.preview_content.setFixedHeight(content_height)
        self.preview_scroll.setFixedHeight(content_height)


class _PPOCRTableGridWidget(QTableWidget):
    copyShortcutTriggered = pyqtSignal()
    pasteShortcutTriggered = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._right_click_selection_snapshot: list[tuple[int, int]] = []

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copyShortcutTriggered.emit()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Paste):
            self.pasteShortcutTriggered.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            index = self.indexAt(event.position().toPoint())
            selected_indexes = self.selectedIndexes()
            if (
                len(selected_indexes) > 1
                and index.isValid()
                and self.selectionModel().isSelected(index)
            ):
                self._right_click_selection_snapshot = [
                    (selected.row(), selected.column())
                    for selected in selected_indexes
                ]
            else:
                self._right_click_selection_snapshot = []
        super().mousePressEvent(event)

    def restore_right_click_selection(self, index) -> None:
        snapshot = self._right_click_selection_snapshot
        self._right_click_selection_snapshot = []
        if not snapshot or not index.isValid():
            return
        if (index.row(), index.column()) not in snapshot:
            return
        if len(self.selectedIndexes()) > 1:
            return
        self.clearSelection()
        model = self.model()
        selection_model = self.selectionModel()
        for row_index, col_index in snapshot:
            selection_model.select(
                model.index(row_index, col_index),
                QItemSelectionModel.SelectionFlag.Select,
            )
        selection_model.setCurrentIndex(
            model.index(index.row(), index.column()),
            QItemSelectionModel.SelectionFlag.NoUpdate,
        )


class _PPOCRTableItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index) -> None:
        option_without_focus = QStyleOptionViewItem(option)
        option_without_focus.state &= ~QStyle.StateFlag.State_HasFocus
        if option_without_focus.state & QStyle.StateFlag.State_Editing:
            option_without_focus.text = ""
        super().paint(painter, option_without_focus, index)

    def createEditor(self, parent, option, index) -> QWidget:
        del option, index
        editor = QLineEdit(parent)
        editor.setFrame(False)
        editor.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        editor.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        editor.setStyleSheet(
            "QLineEdit {"
            f"background: {PPOCR_TABLE_SELECTION_COLOR};"
            "border: none;"
            f"padding: 0px {PPOCR_TABLE_CELL_HORIZONTAL_PADDING_PX}px;"
            "margin: 0px;"
            "}"
            "QLineEdit:focus {"
            "border: none;"
            "outline: none;"
            "}"
        )
        return editor

    def updateEditorGeometry(self, editor, option, index) -> None:
        del index
        editor.setGeometry(option.rect)


class PPOCRTableBlockEditor(QFrame):
    saveRequested = pyqtSignal(str)
    cancelRequested = pyqtSignal()

    def __init__(
        self,
        content: str,
        parent=None,
        image_path: str | Path | None = None,
    ) -> None:
        del image_path
        super().__init__(parent)
        self._span_owner: dict[tuple[int, int], tuple[int, int]] = {}
        self._span_size: dict[tuple[int, int], tuple[int, int]] = {}
        self._history_states: list[str] = []
        self._history_index = -1
        self._history_batch_depth = 0
        self._history_restoring = False
        self._history_max_depth = 120
        self._build_ui()
        self._load_content(content)
        self._connect_signals()
        self._reset_history()
        self._update_table_actions_state()

    def _build_ui(self) -> None:
        self.setObjectName("PPOCRTableBlockEditor")
        self.setStyleSheet(self._frame_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QFrame(self)
        toolbar.setObjectName("PPOCRTableBlockEditorToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 0, 12, 0)
        toolbar_layout.setSpacing(4)

        self.bold_button = self._build_format_button(
            _make_rich_editor_char_icon("B", bold=True)
        )
        self.italic_button = self._build_format_button(
            _make_rich_editor_char_icon("I", italic=True)
        )
        self.strike_button = self._build_format_button(
            _make_rich_editor_strike_icon()
        )
        self.bold_button.setToolTip(self.tr("Bold"))
        self.italic_button.setToolTip(self.tr("Italic"))
        self.strike_button.setToolTip(self.tr("Strikethrough"))

        toolbar_layout.addWidget(self.bold_button)
        toolbar_layout.addWidget(self.italic_button)
        toolbar_layout.addWidget(self.strike_button)
        toolbar_layout.addStretch()

        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(get_secondary_button_style())
        self.save_button = QPushButton(self.tr("Save"))
        self.save_button.setStyleSheet(get_primary_button_style())
        toolbar_layout.addWidget(self.cancel_button)
        toolbar_layout.addSpacing(12)
        toolbar_layout.addWidget(self.save_button)

        body = QFrame(self)
        body.setObjectName("PPOCRTableBlockEditorBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(16, 16, 16, 16)
        body_layout.setSpacing(0)

        self.table = _PPOCRTableGridWidget(body)
        self.editor = self.table
        self.table.setItemDelegate(_PPOCRTableItemDelegate(self.table))
        self.table.setStyleSheet(self._table_style())
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        self.table.setProperty("ppocrMultiSelect", False)
        self.table.setCornerButtonEnabled(False)
        self.table.setAlternatingRowColors(False)
        self.table.setShowGrid(True)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.table.horizontalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed
        )
        self.table.horizontalHeader().setMinimumSectionSize(1)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed
        )
        self.table.verticalHeader().setDefaultSectionSize(
            self._default_row_height()
        )
        self.table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        body_layout.addWidget(self.table)

        layout.addWidget(toolbar)
        layout.addWidget(body)

    def _build_format_button(
        self,
        icon: QIcon,
    ) -> QToolButton:
        button = QToolButton(self)
        button.setIcon(icon)
        button.setIconSize(
            QSize(
                PPOCR_RICH_EDITOR_ICON_SIZE,
                PPOCR_RICH_EDITOR_ICON_SIZE,
            )
        )
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setStyleSheet(PPOCR_RICH_EDITOR_FORMAT_BUTTON_STYLE)
        return button

    @staticmethod
    def _new_table_item(text: str = "") -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        )
        return item

    def _default_row_height(self) -> int:
        line_height = self.table.fontMetrics().height()
        return max(
            PPOCR_TABLE_MIN_ROW_HEIGHT_PX,
            int(round(line_height * PPOCR_TABLE_CELL_LINE_HEIGHT))
            + PPOCR_TABLE_CELL_VERTICAL_PADDING_PX * 2,
        )

    def _sync_column_widths(self) -> None:
        col_count = self.table.columnCount()
        if col_count <= 0:
            return
        viewport_width = self.table.viewport().width()
        if viewport_width <= 0:
            return
        usable_width = max(col_count, viewport_width)
        base_width = usable_width // col_count
        remainder = usable_width % col_count
        base_widths = [
            base_width + (1 if col_index < remainder else 0)
            for col_index in range(col_count)
        ]
        content_widths = [
            self._column_content_width_hint(col_index)
            for col_index in range(col_count)
        ]
        widths = [
            max(base_widths[col_index], content_widths[col_index])
            for col_index in range(col_count)
        ]
        total_width = sum(widths)
        if total_width < viewport_width:
            extra = viewport_width - total_width
            bonus = extra // col_count
            bonus_remainder = extra % col_count
            widths = [
                widths[col_index]
                + bonus
                + (1 if col_index < bonus_remainder else 0)
                for col_index in range(col_count)
            ]
        for col_index in range(col_count):
            self.table.setColumnWidth(col_index, max(1, widths[col_index]))

    def _column_content_width_hint(self, col_index: int) -> int:
        metrics = self.table.fontMetrics()
        max_text_width = 0
        for row_index in range(self.table.rowCount()):
            anchor_row, anchor_col = self._cell_anchor(row_index, col_index)
            if anchor_col != col_index or anchor_row != row_index:
                continue
            item = self.table.item(row_index, col_index)
            if item is None:
                continue
            text = item.text() or ""
            if not text:
                continue
            text_width = metrics.horizontalAdvance(text)
            row_span, col_span = self._anchor_span((row_index, col_index))
            span_factor = max(1, col_span)
            text_width = text_width // span_factor
            if text_width > max_text_width:
                max_text_width = text_width
        return max(
            PPOCR_TABLE_MIN_COLUMN_WIDTH_PX,
            min(
                PPOCR_TABLE_MAX_COLUMN_WIDTH_PX,
                max_text_width
                + PPOCR_TABLE_CELL_HORIZONTAL_PADDING_PX * 2
                + 8,
            ),
        )

    def _sync_row_heights(self) -> None:
        row_height = self._default_row_height()
        self.table.verticalHeader().setDefaultSectionSize(row_height)
        for row_index in range(self.table.rowCount()):
            if self.table.rowHeight(row_index) != row_height:
                self.table.setRowHeight(row_index, row_height)

    def _find_parent_scroll_area(self) -> QScrollArea | None:
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                return parent
            parent = parent.parentWidget()
        return None

    def _available_table_height(self) -> int:
        scroll_area = self._find_parent_scroll_area()
        if scroll_area is None:
            return 0
        viewport = scroll_area.viewport()
        viewport_height = viewport.height()
        if viewport_height <= 0:
            return 0
        table_top = self.table.mapTo(viewport, QPoint(0, 0)).y()
        table_top = max(0, min(viewport_height, table_top))
        return max(
            0,
            viewport_height
            - table_top
            - PPOCR_TABLE_VIEWPORT_BOTTOM_MARGIN_PX,
        )

    def _horizontal_scrollbar_needed(self) -> bool:
        col_count = self.table.columnCount()
        if col_count <= 0:
            return False
        viewport_width = self.table.viewport().width()
        if viewport_width <= 0:
            return False
        total_width = sum(
            self.table.columnWidth(col_index) for col_index in range(col_count)
        )
        return total_width > viewport_width

    def _table_content_height(self) -> int:
        row_height = self._default_row_height()
        row_count = max(1, self.table.rowCount())
        content_height = (
            row_count * row_height
            + self.table.frameWidth() * 2
            + (1 if self.table.showGrid() else 0)
        )
        if self._horizontal_scrollbar_needed():
            content_height += (
                self.table.horizontalScrollBar().sizeHint().height()
            )
        return content_height

    def _sync_table_height(self) -> None:
        min_height = self._default_row_height() + self.table.frameWidth() * 2
        target_height = self._table_content_height()
        available_height = self._available_table_height()
        if available_height > 0:
            target_height = min(target_height, available_height)
        target_height = max(min_height, target_height)
        if self.table.minimumHeight() != target_height:
            self.table.setMinimumHeight(target_height)
        if self.table.maximumHeight() != target_height:
            self.table.setMaximumHeight(target_height)

    def _sync_table_geometry(self) -> None:
        self._sync_row_heights()
        self._sync_column_widths()
        self._sync_table_height()

    def _connect_signals(self) -> None:
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        self.save_button.clicked.connect(self._emit_save)

        self.bold_button.clicked.connect(self._toggle_bold)
        self.italic_button.clicked.connect(self._toggle_italic)
        self.strike_button.clicked.connect(self._toggle_strike)

        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.itemSelectionChanged.connect(
            self._update_table_actions_state
        )
        self.table.currentCellChanged.connect(
            lambda *_args: self._update_table_actions_state()
        )
        self.table.copyShortcutTriggered.connect(self._copy_selection)
        self.table.pasteShortcutTriggered.connect(self._paste_selection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(
            self._show_table_context_menu
        )

        shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        shortcut.activated.connect(self._toggle_bold)
        shortcut = QShortcut(QKeySequence("Ctrl+I"), self)
        shortcut.activated.connect(self._toggle_italic)
        shortcut = QShortcut(QKeySequence("Ctrl+Shift+X"), self)
        shortcut.activated.connect(self._toggle_strike)
        shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        shortcut.activated.connect(self._emit_save)
        shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Undo), self)
        shortcut.activated.connect(self._undo_table_change)
        shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Redo), self)
        shortcut.activated.connect(self._redo_table_change)
        shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        shortcut.activated.connect(self._redo_table_change)

    def _history_recording_blocked(self) -> bool:
        return self._history_restoring or self._history_batch_depth > 0

    def _begin_history_batch(self) -> None:
        self._history_batch_depth += 1

    def _end_history_batch(self, record_state: bool = True) -> None:
        self._history_batch_depth = max(0, self._history_batch_depth - 1)
        if (
            record_state
            and self._history_batch_depth == 0
            and not self._history_restoring
        ):
            self._record_history_state()

    def _table_snapshot(self) -> str:
        return self._serialize_table_tokens()

    def _reset_history(self) -> None:
        snapshot = self._table_snapshot()
        self._history_states = [snapshot]
        self._history_index = 0

    def _record_history_state(self) -> None:
        if self._history_recording_blocked():
            return
        snapshot = self._table_snapshot()
        if (
            self._history_states
            and 0 <= self._history_index < len(self._history_states)
            and self._history_states[self._history_index] == snapshot
        ):
            return
        if self._history_index < len(self._history_states) - 1:
            self._history_states = self._history_states[
                : self._history_index + 1
            ]
        self._history_states.append(snapshot)
        if len(self._history_states) > self._history_max_depth:
            overflow = len(self._history_states) - self._history_max_depth
            self._history_states = self._history_states[overflow:]
            self._history_index = max(0, self._history_index - overflow)
        self._history_index = len(self._history_states) - 1

    def _restore_history_snapshot(self, snapshot: str) -> None:
        self._history_restoring = True
        try:
            self._load_content(snapshot)
        finally:
            self._history_restoring = False

    def _undo_table_change(self) -> None:
        if self._history_index <= 0 or not self._history_states:
            return
        self._history_index -= 1
        self._restore_history_snapshot(
            self._history_states[self._history_index]
        )

    def _redo_table_change(self) -> None:
        if (
            not self._history_states
            or self._history_index >= len(self._history_states) - 1
        ):
            return
        self._history_index += 1
        self._restore_history_snapshot(
            self._history_states[self._history_index]
        )

    def _on_table_item_changed(self, _item) -> None:
        self._record_history_state()

    def _load_content(self, content: str) -> None:
        parsed = self._parse_token_table_content(content)
        if parsed is not None:
            row_count, col_count, states, spans = parsed
            self._apply_parsed_table(row_count, col_count, states, spans)
            return
        parsed = self._parse_html_table_content(content)
        if parsed is not None:
            row_count, col_count, states, spans = parsed
            self._apply_parsed_table(row_count, col_count, states, spans)
            return

        raw = (content or "").strip()
        if not raw:
            self._apply_parsed_table(0, 0, {}, {})
            return

        lines = [
            line for line in raw.splitlines() if line.strip() or "\t" in line
        ]
        if lines and (len(lines) > 1 or "\t" in raw):
            grid = [line.split("\t") for line in lines]
            row_count = len(grid)
            col_count = max((len(row) for row in grid), default=0)
            states = {}
            for row_index, row in enumerate(grid):
                for col_index, value in enumerate(row):
                    text = value.strip()
                    if text:
                        states[(row_index, col_index)] = (
                            text,
                            False,
                            False,
                            False,
                        )
            self._apply_parsed_table(row_count, col_count, states, {})
            return

        self._apply_parsed_table(
            1, 1, {(0, 0): (raw, False, False, False)}, {}
        )

    def _parse_token_table_content(
        self,
        content: str,
    ) -> (
        tuple[
            int,
            int,
            dict[tuple[int, int], tuple[str, bool, bool, bool]],
            dict[tuple[int, int], tuple[int, int]],
        ]
        | None
    ):
        matches = list(_TABLE_TOKEN_PATTERN.finditer(content or ""))
        if not matches:
            return None

        rows: list[list[tuple[str, str]]] = []
        current_row: list[tuple[str, str]] = []
        for index, match in enumerate(matches):
            token = match.group(1).casefold()
            payload_start = match.end()
            payload_end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(content)
            )
            payload = (content[payload_start:payload_end] or "").strip()
            if token == "nl":
                rows.append(current_row)
                current_row = []
                continue
            current_row.append((token, payload))
        if current_row:
            rows.append(current_row)
        while rows and not rows[-1]:
            rows.pop()
        if not rows:
            return (0, 0, {}, {})

        row_count = len(rows)
        col_count = max((len(row) for row in rows), default=0)
        for row in rows:
            if len(row) < col_count:
                row.extend([("ecel", "")] * (col_count - len(row)))

        owner: dict[tuple[int, int], tuple[int, int]] = {}
        states: dict[tuple[int, int], tuple[str, bool, bool, bool]] = {}

        for row_index, row in enumerate(rows):
            for col_index, (token, payload) in enumerate(row):
                cell = (row_index, col_index)
                if token in {"fcel", "ecel"}:
                    owner[cell] = cell
                    text, bold, italic, strike = self._decode_cell_payload(
                        payload
                    )
                    states[cell] = (text, bold, italic, strike)
                    continue
                if token == "lcel":
                    owner[cell] = owner.get((row_index, col_index - 1), cell)
                    continue
                if token == "ucel":
                    owner[cell] = owner.get((row_index - 1, col_index), cell)
                    continue
                if token == "xcel":
                    left_anchor = owner.get((row_index, col_index - 1))
                    up_anchor = owner.get((row_index - 1, col_index))
                    owner[cell] = up_anchor or left_anchor or cell
                    continue
                owner[cell] = cell

        for row_index in range(row_count):
            for col_index in range(col_count):
                cell = (row_index, col_index)
                owner.setdefault(cell, cell)
                states.setdefault(cell, ("", False, False, False))

        grouped_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for cell, anchor in owner.items():
            grouped_cells.setdefault(anchor, []).append(cell)

        spans: dict[tuple[int, int], tuple[int, int]] = {}
        for cells in grouped_cells.values():
            if len(cells) <= 1:
                continue
            top = min(cell[0] for cell in cells)
            left = min(cell[1] for cell in cells)
            bottom = max(cell[0] for cell in cells)
            right = max(cell[1] for cell in cells)
            expected = {
                (row, col)
                for row in range(top, bottom + 1)
                for col in range(left, right + 1)
            }
            cell_set = set(cells)
            if expected != cell_set:
                continue
            top_left = (top, left)
            spans[top_left] = (bottom - top + 1, right - left + 1)
            if not states.get(top_left, ("", False, False, False))[0]:
                for cell in sorted(cell_set):
                    if states.get(cell, ("", False, False, False))[0]:
                        states[top_left] = states[cell]
                        break
            for cell in cell_set:
                if cell != top_left:
                    states.pop(cell, None)

        return (row_count, col_count, states, spans)

    def _parse_html_table_content(
        self,
        content: str,
    ) -> (
        tuple[
            int,
            int,
            dict[tuple[int, int], tuple[str, bool, bool, bool]],
            dict[tuple[int, int], tuple[int, int]],
        ]
        | None
    ):
        raw = (content or "").strip()
        if not raw:
            return None
        if not (
            _HTML_TABLE_OPEN_PATTERN.search(raw)
            and _HTML_TABLE_ROW_PATTERN.search(raw)
            and _HTML_TABLE_CELL_PATTERN.search(raw)
        ):
            return None

        parser = _PPOCRHTMLTableParser()
        try:
            parser.feed(raw)
        except Exception:
            return None
        rows = [row for row in parser.rows if row]
        if not rows:
            return None

        states: dict[tuple[int, int], tuple[str, bool, bool, bool]] = {}
        spans: dict[tuple[int, int], tuple[int, int]] = {}
        occupied: dict[tuple[int, int], tuple[int, int]] = {}
        touched_cells: set[tuple[int, int]] = set()
        max_col_index = -1

        for row_index, row in enumerate(rows):
            col_index = 0
            while (row_index, col_index) in occupied:
                col_index += 1
            for cell_data in row:
                while (row_index, col_index) in occupied:
                    col_index += 1
                row_span = _parse_span_int(cell_data.get("row_span"))
                col_span = _parse_span_int(cell_data.get("col_span"))
                anchor = (row_index, col_index)
                text = str(cell_data.get("text") or "")
                bold = bool(cell_data.get("bold"))
                italic = bool(cell_data.get("italic"))
                strike = bool(cell_data.get("strike"))
                states[anchor] = (text, bold, italic, strike)
                touched_cells.add(anchor)
                if row_span > 1 or col_span > 1:
                    spans[anchor] = (row_span, col_span)
                for span_row in range(row_index, row_index + row_span):
                    for span_col in range(col_index, col_index + col_span):
                        touched_cells.add((span_row, span_col))
                        max_col_index = max(max_col_index, span_col)
                        if (span_row, span_col) != anchor:
                            occupied[(span_row, span_col)] = anchor
                col_index += col_span
            max_col_index = max(max_col_index, col_index - 1)

        if not touched_cells:
            return None
        row_count = max((row for row, _col in touched_cells), default=-1) + 1
        col_count = max_col_index + 1
        if row_count <= 0 or col_count <= 0:
            return None
        return (row_count, col_count, states, spans)

    @staticmethod
    def _decode_cell_payload(payload: str) -> tuple[str, bool, bool, bool]:
        text_payload = payload or ""
        lowered = text_payload.casefold()
        bold = "<b>" in lowered or "<strong>" in lowered
        italic = "<i>" in lowered or "<em>" in lowered
        strike = (
            "<s>" in lowered or "<strike>" in lowered or "<del>" in lowered
        )
        plain = _TABLE_STYLE_TAG_PATTERN.sub("", text_payload).strip()
        return html_unescape(plain), bold, italic, strike

    def _apply_parsed_table(
        self,
        row_count: int,
        col_count: int,
        states: dict[tuple[int, int], tuple[str, bool, bool, bool]],
        spans: dict[tuple[int, int], tuple[int, int]],
    ) -> None:
        self.table.clearSpans()
        self.table.clearContents()
        self.table.setRowCount(max(0, row_count))
        self.table.setColumnCount(max(0, col_count))

        for row_index in range(self.table.rowCount()):
            for col_index in range(self.table.columnCount()):
                self.table.setItem(
                    row_index,
                    col_index,
                    self._new_table_item(""),
                )

        for (row_index, col_index), (
            text,
            bold,
            italic,
            strike,
        ) in states.items():
            if (
                row_index < 0
                or col_index < 0
                or row_index >= self.table.rowCount()
                or col_index >= self.table.columnCount()
            ):
                continue
            item = self._ensure_cell_item(row_index, col_index)
            item.setText(text)
            font = item.font()
            font.setBold(bold)
            font.setItalic(italic)
            font.setStrikeOut(strike)
            item.setFont(font)

        for (row_index, col_index), (row_span, col_span) in spans.items():
            if (
                row_index < 0
                or col_index < 0
                or row_span <= 1
                and col_span <= 1
                or row_index >= self.table.rowCount()
                or col_index >= self.table.columnCount()
            ):
                continue
            max_row_span = self.table.rowCount() - row_index
            max_col_span = self.table.columnCount() - col_index
            self.table.setSpan(
                row_index,
                col_index,
                max(1, min(row_span, max_row_span)),
                max(1, min(col_span, max_col_span)),
            )

        if self.table.rowCount() > 0 and self.table.columnCount() > 0:
            self.table.setCurrentCell(0, 0)
        self._rebuild_span_index()
        self._sync_table_geometry()
        self._update_table_actions_state()

    def _ensure_cell_item(
        self, row_index: int, col_index: int
    ) -> QTableWidgetItem:
        item = self.table.item(row_index, col_index)
        if item is None:
            item = self._new_table_item("")
            item.setFont(QFont(self.table.font()))
            self.table.setItem(row_index, col_index, item)
        return item

    def _rebuild_span_index(self) -> None:
        row_count = self.table.rowCount()
        col_count = self.table.columnCount()
        owner: dict[tuple[int, int], tuple[int, int]] = {}
        span_size: dict[tuple[int, int], tuple[int, int]] = {}

        for row_index in range(row_count):
            for col_index in range(col_count):
                owner[(row_index, col_index)] = (row_index, col_index)

        for row_index in range(row_count):
            for col_index in range(col_count):
                if owner[(row_index, col_index)] != (row_index, col_index):
                    continue
                row_span = max(1, self.table.rowSpan(row_index, col_index))
                col_span = max(1, self.table.columnSpan(row_index, col_index))
                if row_span <= 1 and col_span <= 1:
                    continue
                span_size[(row_index, col_index)] = (row_span, col_span)
                for span_row in range(
                    row_index, min(row_count, row_index + row_span)
                ):
                    for span_col in range(
                        col_index,
                        min(col_count, col_index + col_span),
                    ):
                        owner[(span_row, span_col)] = (row_index, col_index)

        self._span_owner = owner
        self._span_size = span_size

    def _cell_anchor(self, row_index: int, col_index: int) -> tuple[int, int]:
        return self._span_owner.get(
            (row_index, col_index), (row_index, col_index)
        )

    def _anchor_span(self, anchor: tuple[int, int]) -> tuple[int, int]:
        return self._span_size.get(anchor, (1, 1))

    def _selected_anchor_cells(self) -> list[tuple[int, int]]:
        if self.table.rowCount() <= 0 or self.table.columnCount() <= 0:
            return []
        anchors = []
        seen: set[tuple[int, int]] = set()
        indexes = sorted(
            self.table.selectedIndexes(),
            key=lambda index: (index.row(), index.column()),
        )
        if (
            not indexes
            and self.table.currentRow() >= 0
            and self.table.currentColumn() >= 0
        ):
            indexes = [
                self.table.model().index(
                    self.table.currentRow(),
                    self.table.currentColumn(),
                )
            ]
        for index in indexes:
            anchor = self._cell_anchor(index.row(), index.column())
            if anchor in seen:
                continue
            seen.add(anchor)
            anchors.append(anchor)
        return anchors

    def _selected_rect(self) -> tuple[int, int, int, int] | None:
        selected_cells = {
            (index.row(), index.column())
            for index in self.table.selectedIndexes()
        }
        if not selected_cells:
            return None
        top = min(cell[0] for cell in selected_cells)
        bottom = max(cell[0] for cell in selected_cells)
        left = min(cell[1] for cell in selected_cells)
        right = max(cell[1] for cell in selected_cells)
        if len(selected_cells) != (bottom - top + 1) * (right - left + 1):
            return None
        return top, left, bottom, right

    def _can_merge_selection(self) -> bool:
        selected_rect = self._selected_rect()
        if selected_rect is None:
            return False
        top, left, bottom, right = selected_rect
        if top == bottom and left == right:
            return False
        for row_index in range(top, bottom + 1):
            for col_index in range(left, right + 1):
                anchor = self._cell_anchor(row_index, col_index)
                row_span, col_span = self._anchor_span(anchor)
                if (
                    anchor[0] < top
                    or anchor[1] < left
                    or anchor[0] + row_span - 1 > bottom
                    or anchor[1] + col_span - 1 > right
                ):
                    return False
        return True

    def _merged_anchor_from_selection(self) -> tuple[int, int] | None:
        for anchor in self._selected_anchor_cells():
            row_span, col_span = self._anchor_span(anchor)
            if row_span > 1 or col_span > 1:
                return anchor
        current_row = self.table.currentRow()
        current_col = self.table.currentColumn()
        if current_row < 0 or current_col < 0:
            return None
        anchor = self._cell_anchor(current_row, current_col)
        row_span, col_span = self._anchor_span(anchor)
        if row_span > 1 or col_span > 1:
            return anchor
        return None

    def _toggle_selected_font_state(self, key: str) -> None:
        anchors = self._selected_anchor_cells()
        if not anchors:
            return
        current_states = []
        for row_index, col_index in anchors:
            item = self.table.item(row_index, col_index)
            if item is None:
                current_states.append(False)
                continue
            font = item.font()
            if key == "bold":
                current_states.append(font.bold())
            elif key == "italic":
                current_states.append(font.italic())
            else:
                current_states.append(font.strikeOut())
        next_state = not all(current_states)

        self._begin_history_batch()
        try:
            for row_index, col_index in anchors:
                item = self._ensure_cell_item(row_index, col_index)
                font = item.font()
                if key == "bold":
                    font.setBold(next_state)
                elif key == "italic":
                    font.setItalic(next_state)
                else:
                    font.setStrikeOut(next_state)
                item.setFont(font)
        finally:
            self._end_history_batch(record_state=False)
        self._record_history_state()
        self._update_table_actions_state()

    def _toggle_bold(self) -> None:
        self._toggle_selected_font_state("bold")

    def _toggle_italic(self) -> None:
        self._toggle_selected_font_state("italic")

    def _toggle_strike(self) -> None:
        self._toggle_selected_font_state("strike")

    def _edit_clicked_cell(self, index) -> None:
        if not index.isValid():
            return
        modifiers = QApplication.keyboardModifiers()
        if modifiers & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            return
        if len(self.table.selectedIndexes()) > 1:
            return
        anchor_row, anchor_col = self._cell_anchor(index.row(), index.column())
        anchor_index = self.table.model().index(anchor_row, anchor_col)
        if not anchor_index.isValid():
            return
        self.table.clearSelection()
        self.table.setCurrentIndex(anchor_index)
        self.table.edit(anchor_index)

    def _show_table_context_menu(self, pos: QPoint) -> None:
        index = self.table.indexAt(pos)
        self.table.restore_right_click_selection(index)
        selected_count = len(self.table.selectedIndexes())
        if index.isValid() and selected_count <= 1:
            if not self.table.selectionModel().isSelected(index):
                self.table.clearSelection()
                self.table.setCurrentCell(index.row(), index.column())
            else:
                self.table.setCurrentCell(index.row(), index.column())
        row_count = self.table.rowCount()
        col_count = self.table.columnCount()
        has_table = row_count > 0 and col_count > 0

        menu = QMenu(self)
        menu.setStyleSheet(self._table_context_menu_style())

        create_table_action = menu.addAction(self.tr("Create 3x3 Table"))
        create_table_action.setEnabled(not has_table)
        menu.addSeparator()

        insert_row_above_action = menu.addAction(self.tr("Insert Row Above"))
        insert_row_above_action.setEnabled(has_table)
        insert_row_below_action = menu.addAction(self.tr("Insert Row Below"))
        insert_row_below_action.setEnabled(has_table)
        delete_row_action = menu.addAction(self.tr("Delete Row"))
        delete_row_action.setEnabled(has_table)
        menu.addSeparator()

        insert_column_left_action = menu.addAction(
            self.tr("Insert Column Left")
        )
        insert_column_left_action.setEnabled(has_table)
        insert_column_right_action = menu.addAction(
            self.tr("Insert Column Right")
        )
        insert_column_right_action.setEnabled(has_table)
        delete_column_action = menu.addAction(self.tr("Delete Column"))
        delete_column_action.setEnabled(has_table)
        menu.addSeparator()

        merge_action = menu.addAction(self.tr("Merge Cells"))
        merge_action.setEnabled(has_table and self._can_merge_selection())
        split_action = menu.addAction(self.tr("Split Cells"))
        split_action.setEnabled(
            has_table and self._merged_anchor_from_selection() is not None
        )

        chosen_action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen_action is None:
            return
        if chosen_action == create_table_action:
            self._create_default_table()
            return
        if chosen_action == insert_row_above_action:
            self._insert_row_above()
            return
        if chosen_action == insert_row_below_action:
            self._insert_row_below()
            return
        if chosen_action == delete_row_action:
            self._delete_current_row()
            return
        if chosen_action == insert_column_left_action:
            self._insert_column_left()
            return
        if chosen_action == insert_column_right_action:
            self._insert_column_right()
            return
        if chosen_action == delete_column_action:
            self._delete_current_column()
            return
        if chosen_action == merge_action:
            self._merge_selection()
            return
        if chosen_action == split_action:
            self._split_selection()

    def _table_context_menu_style(self) -> str:
        theme = get_theme()
        return (
            "QMenu {"
            f"background: {theme['background']};"
            f"color: {theme['text']};"
            "border: 1px solid rgb(220, 228, 241);"
            "padding: 6px;"
            "}"
            "QMenu::item {"
            "padding: 6px 18px;"
            "border-radius: 6px;"
            "}"
            "QMenu::item:selected {"
            "background: rgb(241, 245, 255);"
            f"color: {PPOCR_COLOR_TEXT};"
            "}"
            "QMenu::separator {"
            "height: 1px;"
            "margin: 4px 8px;"
            "background: rgb(229, 234, 244);"
            "}"
        )

    def _insert_row_at(self, insert_at: int) -> None:
        if self.table.rowCount() <= 0 or self.table.columnCount() <= 0:
            return
        insert_at = max(0, min(insert_at, self.table.rowCount()))
        self._begin_history_batch()
        try:
            self.table.insertRow(insert_at)
            for col_index in range(self.table.columnCount()):
                self.table.setItem(
                    insert_at,
                    col_index,
                    self._new_table_item(""),
                )
            self._sync_row_heights()
            self._rebuild_span_index()
            self._update_table_actions_state()
        finally:
            self._end_history_batch()

    def _insert_row_above(self) -> None:
        row_index = self.table.currentRow()
        self._insert_row_at(row_index if row_index >= 0 else 0)

    def _insert_row_below(self) -> None:
        row_index = self.table.currentRow()
        self._insert_row_at(
            row_index + 1 if row_index >= 0 else self.table.rowCount()
        )

    def _delete_current_row(self) -> None:
        if self.table.rowCount() <= 0:
            return
        row_index = self.table.currentRow()
        remove_at = (
            row_index
            if 0 <= row_index < self.table.rowCount()
            else self.table.rowCount() - 1
        )
        self._begin_history_batch()
        try:
            self.table.removeRow(remove_at)
            if self.table.rowCount() == 0:
                self.table.setColumnCount(0)
            self._sync_table_geometry()
            self._rebuild_span_index()
            self._update_table_actions_state()
        finally:
            self._end_history_batch()

    def _insert_column_at(self, insert_at: int) -> None:
        if self.table.rowCount() <= 0 or self.table.columnCount() <= 0:
            return
        insert_at = max(0, min(insert_at, self.table.columnCount()))
        self._begin_history_batch()
        try:
            self.table.insertColumn(insert_at)
            for row_index in range(self.table.rowCount()):
                self.table.setItem(
                    row_index,
                    insert_at,
                    self._new_table_item(""),
                )
            self._sync_column_widths()
            self._rebuild_span_index()
            self._update_table_actions_state()
        finally:
            self._end_history_batch()

    def _insert_column_left(self) -> None:
        col_index = self.table.currentColumn()
        self._insert_column_at(col_index if col_index >= 0 else 0)

    def _insert_column_right(self) -> None:
        col_index = self.table.currentColumn()
        self._insert_column_at(
            col_index + 1 if col_index >= 0 else self.table.columnCount()
        )

    def _delete_current_column(self) -> None:
        if self.table.columnCount() <= 0:
            return
        col_index = self.table.currentColumn()
        remove_at = (
            col_index
            if 0 <= col_index < self.table.columnCount()
            else self.table.columnCount() - 1
        )
        self._begin_history_batch()
        try:
            self.table.removeColumn(remove_at)
            if self.table.columnCount() == 0:
                self.table.setRowCount(0)
            self._sync_table_geometry()
            self._rebuild_span_index()
            self._update_table_actions_state()
        finally:
            self._end_history_batch()

    def _merge_selection(self) -> None:
        if not self._can_merge_selection():
            return
        selected_rect = self._selected_rect()
        if selected_rect is None:
            return
        top, left, bottom, right = selected_rect
        self._begin_history_batch()
        try:
            self._ensure_cell_item(top, left)
            for row_index in range(top, bottom + 1):
                for col_index in range(left, right + 1):
                    if row_index == top and col_index == left:
                        continue
                    item = self._ensure_cell_item(row_index, col_index)
                    item.setText("")
                    item.setFont(QFont(self.table.font()))
            self.table.setSpan(top, left, bottom - top + 1, right - left + 1)
            self.table.clearSelection()
            self.table.setCurrentCell(top, left)
            self._rebuild_span_index()
            self._update_table_actions_state()
        finally:
            self._end_history_batch()

    def _split_selection(self) -> None:
        anchor = self._merged_anchor_from_selection()
        if anchor is None:
            return
        row_span, col_span = self._anchor_span(anchor)
        if row_span <= 1 and col_span <= 1:
            return
        row_index, col_index = anchor
        self._begin_history_batch()
        try:
            self.table.setSpan(row_index, col_index, 1, 1)
            for span_row in range(row_index, row_index + row_span):
                for span_col in range(col_index, col_index + col_span):
                    if span_row == row_index and span_col == col_index:
                        continue
                    item = self._ensure_cell_item(span_row, span_col)
                    item.setText("")
                    item.setFont(QFont(self.table.font()))
            self.table.clearSelection()
            self.table.setCurrentCell(row_index, col_index)
            self._rebuild_span_index()
            self._update_table_actions_state()
        finally:
            self._end_history_batch()

    def _create_default_table(self) -> None:
        if self.table.rowCount() > 0 and self.table.columnCount() > 0:
            return
        self._begin_history_batch()
        try:
            self.table.clearSpans()
            self.table.clearContents()
            self.table.setRowCount(3)
            self.table.setColumnCount(3)
            for row_index in range(3):
                for col_index in range(3):
                    self.table.setItem(
                        row_index,
                        col_index,
                        self._new_table_item(""),
                    )
            for col_index in range(3):
                item = self._ensure_cell_item(0, col_index)
                item.setText(f"Title {col_index + 1}")
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.table.setCurrentCell(0, 0)
            self._sync_table_geometry()
            self._rebuild_span_index()
            self._update_table_actions_state()
        finally:
            self._end_history_batch()

    def _copy_selection(self) -> None:
        ranges = self.table.selectedRanges()
        if not ranges:
            return
        selected_range = ranges[0]
        lines = []
        for row_index in range(
            selected_range.topRow(), selected_range.bottomRow() + 1
        ):
            cells = []
            for col_index in range(
                selected_range.leftColumn(),
                selected_range.rightColumn() + 1,
            ):
                anchor = self._cell_anchor(row_index, col_index)
                if anchor != (row_index, col_index):
                    cells.append("")
                    continue
                item = self.table.item(row_index, col_index)
                cells.append(item.text() if item is not None else "")
            lines.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(lines))

    def _paste_selection(self) -> None:
        clipboard_text = QApplication.clipboard().text()
        if not clipboard_text:
            return
        if self.table.rowCount() == 0 and self.table.columnCount() == 0:
            self._create_default_table()
        if self.table.rowCount() <= 0 or self.table.columnCount() <= 0:
            return
        start_row = (
            self.table.currentRow() if self.table.currentRow() >= 0 else 0
        )
        start_col = (
            self.table.currentColumn()
            if self.table.currentColumn() >= 0
            else 0
        )
        rows = clipboard_text.splitlines()
        touched_anchors: set[tuple[int, int]] = set()
        self._begin_history_batch()
        try:
            for row_offset, row_text in enumerate(rows):
                cells = row_text.split("\t")
                for col_offset, value in enumerate(cells):
                    target_row = start_row + row_offset
                    target_col = start_col + col_offset
                    if (
                        target_row >= self.table.rowCount()
                        or target_col >= self.table.columnCount()
                    ):
                        continue
                    anchor = self._cell_anchor(target_row, target_col)
                    if anchor in touched_anchors:
                        continue
                    item = self._ensure_cell_item(anchor[0], anchor[1])
                    item.setText(value)
                    touched_anchors.add(anchor)
            self._update_table_actions_state()
        finally:
            self._end_history_batch()

    def _encode_cell_payload(self, row_index: int, col_index: int) -> str:
        item = self.table.item(row_index, col_index)
        if item is None:
            return ""
        text = item.text() or ""
        if not text:
            return ""
        payload = html_escape(text)
        font = item.font()
        if font.strikeOut():
            payload = f"<s>{payload}</s>"
        if font.italic():
            payload = f"<i>{payload}</i>"
        if font.bold():
            payload = f"<b>{payload}</b>"
        return payload

    def _serialize_table_tokens(self) -> str:
        row_count = self.table.rowCount()
        col_count = self.table.columnCount()
        if row_count <= 0 or col_count <= 0:
            return ""
        self._rebuild_span_index()
        parts: list[str] = []
        for row_index in range(row_count):
            for col_index in range(col_count):
                anchor = self._cell_anchor(row_index, col_index)
                if anchor == (row_index, col_index):
                    payload = self._encode_cell_payload(row_index, col_index)
                    if payload:
                        parts.append("<fcel>")
                        parts.append(payload)
                    else:
                        parts.append("<ecel>")
                    continue
                anchor_row, anchor_col = anchor
                if row_index > anchor_row and col_index > anchor_col:
                    parts.append("<xcel>")
                elif row_index > anchor_row:
                    parts.append("<ucel>")
                else:
                    parts.append("<lcel>")
            parts.append("<nl>")
        return "".join(parts)

    def _update_table_actions_state(self) -> None:
        row_count = self.table.rowCount()
        col_count = self.table.columnCount()
        has_table = row_count > 0 and col_count > 0
        anchors = self._selected_anchor_cells() if has_table else []
        format_enabled = has_table and bool(anchors)
        selected_count = len(self.table.selectedIndexes())
        self._set_multi_select_visual_state(selected_count > 1)
        self.bold_button.setEnabled(format_enabled)
        self.italic_button.setEnabled(format_enabled)
        self.strike_button.setEnabled(format_enabled)

        bold_checked = False
        italic_checked = False
        strike_checked = False
        if format_enabled:
            bold_checked = True
            italic_checked = True
            strike_checked = True
            for row_index, col_index in anchors:
                item = self.table.item(row_index, col_index)
                if item is None:
                    bold_checked = False
                    italic_checked = False
                    strike_checked = False
                    break
                font = item.font()
                bold_checked = bold_checked and font.bold()
                italic_checked = italic_checked and font.italic()
                strike_checked = strike_checked and font.strikeOut()
        self.bold_button.blockSignals(True)
        self.italic_button.blockSignals(True)
        self.strike_button.blockSignals(True)
        self.bold_button.setChecked(bold_checked)
        self.italic_button.setChecked(italic_checked)
        self.strike_button.setChecked(strike_checked)
        self.bold_button.blockSignals(False)
        self.italic_button.blockSignals(False)
        self.strike_button.blockSignals(False)

    def _set_multi_select_visual_state(self, enabled: bool) -> None:
        current = bool(self.table.property("ppocrMultiSelect"))
        if current == enabled:
            return
        self.table.setProperty("ppocrMultiSelect", enabled)
        style = self.table.style()
        style.unpolish(self.table)
        style.polish(self.table)
        self.table.viewport().update()

    def _emit_save(self) -> None:
        self.saveRequested.emit(self._serialize_table_tokens())

    def content_height_valid(self) -> bool:
        return True

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._sync_table_geometry)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._sync_table_geometry)

    def _frame_style(self) -> str:
        theme = get_theme()
        return (
            "QFrame#PPOCRTableBlockEditor {"
            f"background: {theme['background']};"
            "border: 1px solid rgb(222, 228, 240);"
            "border-radius: 10px;"
            "}"
            "QFrame#PPOCRTableBlockEditorToolbar {"
            "background: rgb(245, 246, 249);"
            "border: none;"
            "border-top-left-radius: 10px;"
            "border-top-right-radius: 10px;"
            "border-bottom: 1px solid rgb(229, 234, 244);"
            "}"
            "QFrame#PPOCRTableBlockEditorBody {"
            f"background: {theme['background']};"
            "border: none;"
            "border-bottom-left-radius: 10px;"
            "border-bottom-right-radius: 10px;"
            "}"
        )

    def _table_style(self) -> str:
        theme = get_theme()
        return (
            "QTableWidget {"
            f"background: {theme['background']};"
            f"color: {theme['text']};"
            "border: 1px solid rgb(229, 234, 244);"
            "gridline-color: rgb(224, 230, 242);"
            f"selection-background-color: {PPOCR_TABLE_SELECTION_COLOR};"
            "selection-color: rgb(30, 33, 45);"
            "font-size: 13px;"
            "}"
            "QTableWidget::item {"
            f"padding: {PPOCR_TABLE_CELL_VERTICAL_PADDING_PX}px "
            f"{PPOCR_TABLE_CELL_HORIZONTAL_PADDING_PX}px;"
            "border: none;"
            "}"
            "QTableWidget::item:focus {"
            "outline: none;"
            "}"
            'QTableWidget[ppocrMultiSelect="true"]::item:selected {'
            f"background: {PPOCR_TABLE_SELECTION_COLOR};"
            "}"
            'QTableWidget[ppocrMultiSelect="false"]::item:selected {'
            f"background: {PPOCR_TABLE_SELECTION_COLOR};"
            "}"
            "QTableWidget::item:selected:active {"
            f"background: {PPOCR_TABLE_SELECTION_COLOR};"
            "}"
            "QTableWidget::item:selected:!active {"
            f"background: {PPOCR_TABLE_SELECTION_COLOR};"
            "}"
            "QTableWidget QLineEdit {"
            f"background: {PPOCR_TABLE_SELECTION_COLOR};"
            "border: none;"
            "border-radius: 0px;"
            f"color: {theme['text']};"
            f"padding: 0px {PPOCR_TABLE_CELL_HORIZONTAL_PADDING_PX}px;"
            "margin: 0px;"
            "selection-background-color: rgb(70, 88, 255);"
            "selection-color: #ffffff;"
            "}"
            "QTableWidget QLineEdit:focus {"
            "border-radius: 0px;"
            "border: none;"
            "}"
            "QTableWidget QLineEdit:hover {"
            "border-radius: 0px;"
            "}"
        )


def use_rich_text_block_editor(block_label: str) -> bool:
    block_label = normalize_block_label(block_label)
    return (
        block_label not in TABLE_BLOCK_LABELS
        and block_label not in FORMULA_BLOCK_LABELS
    )


def create_ppocr_block_editor(
    block_label: str,
    content: str,
    parent=None,
    image_path: str | Path | None = None,
) -> (
    PPOCRTextBlockEditor
    | PPOCRRichTextBlockEditor
    | PPOCRLatexBlockEditor
    | PPOCRTableBlockEditor
):
    block_label = normalize_block_label(block_label)
    if block_label in FORMULA_BLOCK_LABELS:
        editor_class = PPOCRLatexBlockEditor
    elif block_label in TABLE_BLOCK_LABELS or is_table_like_content(content):
        editor_class = PPOCRTableBlockEditor
    elif use_rich_text_block_editor(block_label):
        editor_class = PPOCRRichTextBlockEditor
    else:
        editor_class = PPOCRTextBlockEditor
    return editor_class(content, parent=parent, image_path=image_path)
