from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from html import escape as html_escape, unescape as html_unescape
from html.parser import HTMLParser
from pathlib import Path
import re

from PyQt6.QtCore import (
    QBuffer,
    QEvent,
    QIODevice,
    QPoint,
    QPointF,
    QRect,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QFontMetrics,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QScrollArea,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QGraphicsDropShadowEffect,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.utils.qt import new_icon, new_icon_path
from anylabeling.views.labeling.utils.theme import get_theme

from .config import (
    PPOCR_COLOR_EDITED,
    PPOCR_COLOR_OVERLAY,
    PPOCR_FILE_TYPE_IMAGE,
    PPOCR_STATUS_ERROR,
    PPOCR_STATUS_PARSED,
    PPOCR_STATUS_PENDING,
)
from .editors import (
    has_cached_latex_preview_pixmap,
    render_latex_preview_pixmap,
)
from .render import (
    FORMULA_BLOCK_LABELS,
    TABLE_BLOCK_LABELS,
    PPOCRBlockData,
    get_block_copy_text,
    is_rich_text_html,
    normalize_block_label,
)
from .style import (
    get_card_label_style,
    get_card_style,
    get_chip_button_style,
    get_floating_action_bar_style,
    get_floating_action_button_style,
    get_icon_button_style,
    get_overlay_label_style,
)

STATUS_COLORS = {
    PPOCR_STATUS_PENDING: "rgb(70, 88, 255)",
    PPOCR_STATUS_PARSED: "rgb(47, 189, 113)",
    PPOCR_STATUS_ERROR: "rgb(255, 69, 58)",
}

_TABLE_TOKEN_PATTERN = re.compile(
    r"<(fcel|ecel|lcel|ucel|xcel|nl)>",
    flags=re.IGNORECASE,
)
_TABLE_STYLE_TAG_PATTERN = re.compile(
    r"</?(?:b|strong|i|em|s|strike|del)>",
    flags=re.IGNORECASE,
)
_INLINE_FORMULA_PATTERN = re.compile(
    r"(?<!\\)(?P<delimiter>\${1,2})(?P<formula>.+?)(?<!\\)(?P=delimiter)",
    flags=re.DOTALL,
)
_HTML_TABLE_OPEN_PATTERN = re.compile(r"<\s*table\b", re.IGNORECASE)
_HTML_TABLE_ROW_PATTERN = re.compile(r"<\s*tr\b", re.IGNORECASE)
_HTML_TABLE_CELL_PATTERN = re.compile(r"<\s*t[dh]\b", re.IGNORECASE)


def _pixmap_to_data_uri(pixmap: QPixmap) -> str:
    if pixmap.isNull():
        return ""
    buffer = QBuffer()
    if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
        return ""
    try:
        if not pixmap.save(buffer, "PNG"):
            return ""
        encoded = bytes(buffer.data().toBase64()).decode("ascii")
    finally:
        buffer.close()
    if not encoded:
        return ""
    return f"data:image/png;base64,{encoded}"


@lru_cache(maxsize=384)
def _inline_formula_img_tag(formula_source: str) -> str:
    try:
        formula_pixmap = render_latex_preview_pixmap(formula_source)
    except Exception:
        return ""
    formula_data_uri = _pixmap_to_data_uri(formula_pixmap)
    if not formula_data_uri:
        return ""
    return (
        f'<img src="{formula_data_uri}" ' 'style="vertical-align: middle;"/>'
    )


def _text_with_inline_formulas_to_html(content: str) -> str:
    text = content or ""
    if "$" not in text:
        return ""

    chunks: list[str] = []
    last_end = 0
    rendered_formula = False

    for match in _INLINE_FORMULA_PATTERN.finditer(text):
        start, end = match.span()
        formula_source = match.group(0)
        formula_content = (match.group("formula") or "").strip()
        if not formula_content:
            continue

        if start > last_end:
            chunks.append(
                html_escape(text[last_end:start]).replace("\n", "<br/>")
            )

        formula_img_tag = _inline_formula_img_tag(formula_source)
        if not formula_img_tag:
            chunks.append(html_escape(formula_source).replace("\n", "<br/>"))
            last_end = end
            continue

        chunks.append(formula_img_tag)
        last_end = end
        rendered_formula = True

    if not rendered_formula:
        return ""

    if last_end < len(text):
        chunks.append(html_escape(text[last_end:]).replace("\n", "<br/>"))
    return "".join(chunks)


def create_floating_shadow(owner) -> QGraphicsDropShadowEffect:
    shadow = QGraphicsDropShadowEffect(owner)
    shadow.setBlurRadius(14)
    shadow.setColor(QColor(20, 14, 53, 36))
    shadow.setOffset(0, 2)
    return shadow


def resolve_qcolor(
    color_value: str,
    fallback: tuple[int, int, int] = (70, 88, 255),
) -> QColor:
    color = QColor(color_value)
    if color.isValid():
        return color
    if color_value.startswith("rgb(") and color_value.endswith(")"):
        parts = [part.strip() for part in color_value[4:-1].split(",")]
        if len(parts) == 3:
            try:
                channels = [max(0, min(255, int(part))) for part in parts]
            except ValueError:
                channels = []
            if len(channels) == 3:
                return QColor(*channels)
    return QColor(*fallback)


def _decode_table_cell_payload(payload: str) -> tuple[str, bool, bool, bool]:
    text_payload = payload or ""
    lowered = text_payload.casefold()
    bold = "<b>" in lowered or "<strong>" in lowered
    italic = "<i>" in lowered or "<em>" in lowered
    strike = "<s>" in lowered or "<strike>" in lowered or "<del>" in lowered
    plain = _TABLE_STYLE_TAG_PATTERN.sub("", text_payload).strip()
    return html_unescape(plain), bold, italic, strike


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


def _parse_table_token_content(
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
                states[cell] = _decode_table_cell_payload(payload)
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


def _table_tokens_to_html(content: str) -> str:
    parsed = _parse_table_token_content(content)
    if parsed is None:
        parsed = _parse_html_table_content(content)
    if parsed is None:
        return ""
    row_count, col_count, states, spans = parsed
    if row_count <= 0 or col_count <= 0:
        return ""

    covered_cells: set[tuple[int, int]] = set()
    row_html_chunks: list[str] = []
    for row_index in range(row_count):
        cell_chunks: list[str] = []
        for col_index in range(col_count):
            cell = (row_index, col_index)
            if cell in covered_cells:
                continue
            row_span, col_span = spans.get(cell, (1, 1))
            if row_span > 1 or col_span > 1:
                for span_row in range(row_index, row_index + row_span):
                    for span_col in range(col_index, col_index + col_span):
                        if (span_row, span_col) != cell:
                            covered_cells.add((span_row, span_col))
            text, bold, italic, strike = states.get(
                cell,
                ("", False, False, False),
            )
            text_html = html_escape(text).replace("\n", "<br/>")
            if not text_html:
                text_html = "&nbsp;"
            if strike:
                text_html = f"<s>{text_html}</s>"
            if italic:
                text_html = f"<em>{text_html}</em>"
            if bold:
                text_html = f"<strong>{text_html}</strong>"
            attrs: list[str] = []
            if row_span > 1:
                attrs.append(f' rowspan="{row_span}"')
            if col_span > 1:
                attrs.append(f' colspan="{col_span}"')
            cell_chunks.append(
                f"<td{''.join(attrs)} style=\""
                "border: 1px solid rgb(229, 234, 244);"
                "padding: 6px 10px;"
                "vertical-align: middle;"
                "text-align: left;"
                "font-size: 13px;"
                "line-height: 1.45;"
                f'">{text_html}</td>'
            )
        row_html_chunks.append("<tr>" + "".join(cell_chunks) + "</tr>")
    if not row_html_chunks:
        return ""
    return (
        '<table cellspacing="0" cellpadding="0" style="'
        "border-collapse: collapse;"
        "width: 100%;"
        "table-layout: fixed;"
        "background: transparent;"
        '">' + "".join(row_html_chunks) + "</table>"
    )


@dataclass
class PPOCRPreviewTransform:
    scale: float = 1.0
    offset_x: int = 16
    offset_y: int = 16


class PPOCRRecentListItemWidget(QWidget):
    deleteRequested = pyqtSignal(str)
    favoriteToggled = pyqtSignal(str, bool)

    def __init__(
        self, record, selected=False, favorite=False, parent=None
    ) -> None:
        super().__init__(parent)
        self.record = record
        self._selected = selected
        self._hovered = False
        self._favorite = favorite
        self.setFixedHeight(56)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setStyleSheet(
            "QWidget { background: transparent; border: none; }"
            "QLabel { background: transparent; border: none; }"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 8, 8)
        layout.setSpacing(10)

        icon_name = (
            "image" if record.file_type == PPOCR_FILE_TYPE_IMAGE else "pdf"
        )
        icon_ext = "svg"
        self.icon_label = QLabel()
        self.icon_label.setPixmap(
            QPixmap(new_icon_path(icon_name, icon_ext)).scaled(
                20,
                20,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        layout.addWidget(self.icon_label)

        text_widget = QWidget()
        text_widget.setStyleSheet("background: transparent;")
        text_layout = QVBoxLayout(text_widget)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        self.name_label = QLabel(record.filename)
        self.name_label.setStyleSheet(
            "QLabel { font-size: 13px; font-weight: 500; }"
        )
        self.name_label.setTextFormat(Qt.TextFormat.PlainText)

        status_row = QWidget()
        status_row.setStyleSheet("background: transparent;")
        status_layout = QHBoxLayout(status_row)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(6)

        self.status_dot = QLabel()
        self.status_dot.setFixedSize(8, 8)
        self.status_dot.setStyleSheet(
            "QLabel {"
            f"background: {STATUS_COLORS.get(record.status, STATUS_COLORS[PPOCR_STATUS_PENDING])};"
            "border-radius: 4px;"
            "}"
        )

        self.time_label = QLabel(record.timestamp)
        self.time_label.setStyleSheet(
            "QLabel { color: rgb(134, 134, 139); font-size: 11px; }"
        )

        status_layout.addWidget(self.status_dot)
        status_layout.addWidget(self.time_label, 1)

        text_layout.addWidget(self.name_label)
        text_layout.addWidget(status_row)
        layout.addWidget(text_widget, 1)

        self.delete_button = QPushButton()
        self.delete_button.setFixedSize(24, 24)
        self.delete_button.setStyleSheet(get_icon_button_style())
        self.delete_button.setVisible(False)
        self.delete_button.clicked.connect(
            lambda: self.deleteRequested.emit(self.record.filename)
        )
        self.delete_button.enterEvent = self._build_button_hover_handler(
            self.delete_button,
            "trash",
            True,
        )
        self.delete_button.leaveEvent = self._build_button_hover_handler(
            self.delete_button,
            "trash",
            False,
        )

        self.actions_widget = QWidget()
        self.actions_widget.setStyleSheet("background: transparent;")
        actions_layout = QHBoxLayout(self.actions_widget)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(2)
        actions_layout.addWidget(self.delete_button)

        self.favorite_button = QPushButton()
        self.favorite_button.setFixedSize(24, 24)
        self.favorite_button.setStyleSheet(get_icon_button_style())
        self.favorite_button.clicked.connect(self._toggle_favorite)
        self.favorite_button.enterEvent = self._build_button_hover_handler(
            self.favorite_button,
            self._favorite_icon_name,
            True,
        )
        self.favorite_button.leaveEvent = self._build_button_hover_handler(
            self.favorite_button,
            self._favorite_icon_name,
            False,
        )
        actions_layout.addWidget(self.favorite_button)
        layout.addWidget(self.actions_widget)
        self._refresh_action_buttons()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        self._refresh_action_buttons()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        self._refresh_action_buttons()
        super().leaveEvent(event)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.update()
        self._refresh_action_buttons()

    def _toggle_favorite(self) -> None:
        self._favorite = not self._favorite
        self._refresh_action_buttons()
        self.favoriteToggled.emit(self.record.filename, self._favorite)

    def _refresh_action_buttons(self) -> None:
        show_actions = self._hovered
        self.delete_button.setVisible(show_actions)
        self.favorite_button.setVisible(show_actions)
        self.actions_widget.setVisible(show_actions)
        self._set_icon(self.delete_button, "trash")
        self._set_icon(self.favorite_button, self._favorite_icon_name())

    def _favorite_icon_name(self) -> str:
        return "starred" if self._favorite else "star"

    def _set_icon(
        self,
        button: QPushButton,
        icon_name: str,
        hovered: bool = False,
    ) -> None:
        pixmap = QPixmap(new_icon_path(icon_name, "svg")).scaled(
            16,
            16,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if hovered and not pixmap.isNull():
            painter = QPainter(pixmap)
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceIn
            )
            painter.fillRect(pixmap.rect(), QColor(70, 88, 255))
            painter.end()
        button.setIcon(QIcon(pixmap))

    def _build_button_hover_handler(
        self,
        button: QPushButton,
        icon_name,
        hovered: bool,
    ):
        def handler(event):
            resolved_icon_name = (
                icon_name() if callable(icon_name) else icon_name
            )
            self._set_icon(button, resolved_icon_name, hovered=hovered)
            (
                QPushButton.enterEvent(button, event)
                if hovered
                else QPushButton.leaveEvent(button, event)
            )

        return handler

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        if self._selected:
            fill = QColor(241, 245, 255)
            border = QColor(220, 228, 255)
            painter.setPen(border)
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 10, 10)
        elif self._hovered:
            fill = QColor(249, 251, 255)
            border = QColor(229, 234, 244)
            painter.setPen(border)
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 10, 10)
        painter.end()
        super().paintEvent(event)


class PPOCRRecentsListWidget(QListWidget):
    fileSelected = pyqtSignal(str)
    deleteRequested = pyqtSignal(str)
    favoriteToggled = pyqtSignal(str, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records = []
        self._selected_name = ""
        self._item_widgets: list[PPOCRRecentListItemWidget] = []
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.itemClicked.connect(self._on_item_clicked)

    def render_records(self, records, selected_name: str) -> None:
        self._records = records
        self._item_widgets = []
        self.blockSignals(True)
        self.clear()
        for record in records:
            list_item = QListWidgetItem()
            list_item.setSizeHint(QSize(0, 64))
            list_item.setData(Qt.ItemDataRole.UserRole, record.filename)
            widget = PPOCRRecentListItemWidget(
                record,
                selected=False,
                favorite=bool(getattr(record, "favorite", False)),
            )
            widget.deleteRequested.connect(self.deleteRequested.emit)
            widget.favoriteToggled.connect(self.favoriteToggled.emit)
            self.addItem(list_item)
            self.setItemWidget(list_item, widget)
            self._item_widgets.append(widget)
        tail_item = QListWidgetItem()
        tail_item.setFlags(Qt.ItemFlag.NoItemFlags)
        if records:
            tail_label = QLabel(self.tr("No More Data"))
            tail_label.setStyleSheet(
                "QLabel { color: rgb(134, 134, 139); padding: 12px 0px; }"
            )
        else:
            tail_label = QLabel(self.tr("No Data"))
            tail_label.setStyleSheet(
                "QLabel { color: rgb(134, 134, 139); padding: 12px 0px; }"
            )
        tail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tail_item.setSizeHint(QSize(0, 48))
        self.addItem(tail_item)
        self.setItemWidget(tail_item, tail_label)
        self.set_selected_name(selected_name)
        self.blockSignals(False)

    def set_selected_name(self, selected_name: str) -> None:
        self._selected_name = selected_name
        for widget in self._item_widgets:
            widget.set_selected(widget.record.filename == selected_name)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        filename = item.data(Qt.ItemDataRole.UserRole)
        if filename:
            self.set_selected_name(filename)
            self.fileSelected.emit(filename)


class PPOCRPreviewCanvas(QWidget):
    blockHovered = pyqtSignal(str)
    blockSelected = pyqtSignal(str)
    blockCopyRequested = pyqtSignal(str)
    canvasCleared = pyqtSignal()
    scaleChanged = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._blocks: list[PPOCRBlockData] = []
        self._transform = PPOCRPreviewTransform()
        self._hovered_key = ""
        self._hovered_keys: set[str] = set()
        self._local_hovered_key = ""
        self._selected_key = ""
        self._selected_locally = False
        self._copy_button = QPushButton(self.tr("Copy"), self)
        self._copy_button.setIcon(QIcon(new_icon_path("copy", "svg")))
        self._copy_button.setIconSize(QSize(14, 14))
        self._copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._copy_button.setFixedHeight(34)
        self._copy_button.setStyleSheet("""
            QPushButton {
                background: rgb(255, 255, 255);
                border: 1px solid rgb(229, 234, 244);
                border-radius: 17px;
                padding: 0px 14px;
                color: rgb(20, 14, 53);
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgb(255, 255, 255);
                border: 1px solid rgb(229, 234, 244);
                color: rgb(70, 88, 255);
            }
            """)
        self._copy_button.setGraphicsEffect(
            create_floating_shadow(self._copy_button)
        )
        self._copy_button.clicked.connect(self._on_copy_button_clicked)
        self._copy_button.hide()
        self.setMouseTracking(True)
        self.setMinimumSize(320, 320)

    def set_page(self, pixmap: QPixmap, blocks: list[PPOCRBlockData]) -> None:
        self._pixmap = pixmap
        self._blocks = blocks
        self._hovered_key = ""
        self._hovered_keys = set()
        self._local_hovered_key = ""
        self._selected_key = ""
        self._selected_locally = False
        self._update_widget_size()
        self._update_copy_button_for_hovered_block()
        self.update()

    def set_fit_width(self, viewport_width: int) -> float:
        if self._pixmap.isNull():
            self.set_scale(1.0)
            return 1.0
        available_width = max(64, viewport_width - 32)
        scale = available_width / max(1, self._pixmap.width())
        self.set_scale(scale)
        return self._transform.scale

    def set_scale(self, scale: float) -> None:
        self._transform.scale = max(0.2, min(5.0, scale))
        self._update_widget_size()
        self.scaleChanged.emit(self._transform.scale)
        self._update_copy_button_for_hovered_block()
        self.update()

    def current_scale(self) -> float:
        return self._transform.scale

    def set_selected_block(
        self,
        block_key: str,
        selected_locally: bool = False,
    ) -> None:
        self._selected_key = block_key
        self._selected_locally = bool(block_key and selected_locally)
        self._update_copy_button_for_hovered_block()
        self.update()

    def set_hovered_block(self, block_key: str) -> None:
        self.set_hovered_blocks([block_key] if block_key else [])

    def set_hovered_blocks(self, block_keys: list[str]) -> None:
        normalized_keys = [key for key in block_keys if key]
        self._hovered_keys = set(normalized_keys)
        self._hovered_key = normalized_keys[0] if normalized_keys else ""
        self._update_copy_button_for_hovered_block()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        if self._pixmap.isNull():
            painter.end()
            return
        width = int(self._pixmap.width() * self._transform.scale)
        height = int(self._pixmap.height() * self._transform.scale)
        x, y = self._canvas_origin()
        painter.drawPixmap(x, y, width, height, self._pixmap)
        hovered_blocks = []
        if self._hovered_keys:
            for block in self._blocks:
                if block.block_key in self._hovered_keys:
                    hovered_blocks.append(block)
        elif self._hovered_key:
            block = self._block_for_key(self._hovered_key)
            if block is not None:
                hovered_blocks.append(block)
        first_polygon_rect = None
        first_label = ""
        first_color = QColor()
        for index, block in enumerate(hovered_blocks):
            polygon = self._to_polygon(block.points, x, y)
            if polygon.isEmpty():
                continue
            border_color = resolve_qcolor(block.category_color)
            fill_color = QColor(border_color)
            border_color.setAlpha(235)
            fill_color.setAlpha(64)
            painter.setPen(QPen(border_color, 1))
            painter.setBrush(fill_color)
            painter.drawPolygon(polygon)
            if index == 0:
                first_polygon_rect = polygon.boundingRect()
                first_label = block.display_label
                first_color = resolve_qcolor(block.category_color)
        if first_polygon_rect is not None:
            self._draw_hovered_label(
                painter,
                first_label,
                first_polygon_rect,
                first_color,
            )
        painter.end()

    def mouseMoveEvent(self, event):
        if self._should_preserve_hover(event.position()):
            super().mouseMoveEvent(event)
            return
        block_key = self._hit_test(event.position())
        if block_key != self._local_hovered_key:
            self._local_hovered_key = block_key
            self._hovered_key = block_key
            self._hovered_keys = {block_key} if block_key else set()
            self.blockHovered.emit(block_key)
            self._update_copy_button_for_hovered_block()
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hovered_key:
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            if self._should_preserve_hover(
                QPointF(float(cursor_pos.x()), float(cursor_pos.y()))
            ):
                super().leaveEvent(event)
                return
            self._local_hovered_key = ""
            self._hovered_key = ""
            self._hovered_keys = set()
            self.blockHovered.emit("")
            self._update_copy_button_for_hovered_block()
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if (
                self._copy_button.isVisible()
                and self._copy_button.geometry().contains(
                    event.position().toPoint()
                )
            ):
                event.accept()
                return
            block_key = self._hit_test(event.position())
            if block_key:
                self._selected_key = block_key
                self._selected_locally = True
                self.blockSelected.emit(block_key)
            else:
                self._selected_key = ""
                self._selected_locally = False
                self.canvasCleared.emit()
            self.update()
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta_y = event.angleDelta().y()
            if delta_y:
                self.set_scale(self.current_scale() + 0.1 * (delta_y / 120.0))
                event.accept()
                return
        event.ignore()

    def _update_widget_size(self) -> None:
        if self._pixmap.isNull():
            self.resize(max(320, self.width()), max(320, self.height()))
            self._update_copy_button_for_hovered_block()
            return
        width = int(self._pixmap.width() * self._transform.scale) + 32
        height = int(self._pixmap.height() * self._transform.scale) + 32
        self.setMinimumSize(width, height)
        self.resize(width, height)
        self._update_copy_button_for_hovered_block()

    def _to_polygon(
        self,
        points: list[tuple[float, float]],
        offset_x: int,
        offset_y: int,
    ) -> QPolygonF:
        polygon = QPolygonF()
        for x, y in points:
            polygon.append(
                QPointF(
                    offset_x + x * self._transform.scale,
                    offset_y + y * self._transform.scale,
                )
            )
        return polygon

    def _hit_test(self, position: QPointF) -> str:
        if self._pixmap.isNull():
            return ""
        x, y = self._canvas_origin()
        for block in reversed(self._blocks):
            polygon = self._to_polygon(block.points, x, y)
            if polygon.isEmpty():
                continue
            path = QPainterPath()
            path.addPolygon(polygon)
            if path.contains(position):
                return block.block_key
        return ""

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_copy_button_for_hovered_block()

    def _canvas_origin(self) -> tuple[int, int]:
        width = int(self._pixmap.width() * self._transform.scale)
        x = max(self._transform.offset_x, (self.width() - width) // 2)
        y = self._transform.offset_y
        return x, y

    def _block_for_key(self, block_key: str) -> PPOCRBlockData | None:
        for block in self._blocks:
            if block.block_key == block_key:
                return block
        return None

    @staticmethod
    def _distance_squared(point_a: QPointF, point_b: QPointF) -> float:
        delta_x = point_a.x() - point_b.x()
        delta_y = point_a.y() - point_b.y()
        return delta_x * delta_x + delta_y * delta_y

    def _nearest_point_on_segment(
        self,
        point: QPointF,
        start: QPointF,
        end: QPointF,
    ) -> QPointF:
        delta_x = end.x() - start.x()
        delta_y = end.y() - start.y()
        length_squared = delta_x * delta_x + delta_y * delta_y
        if length_squared <= 0:
            return QPointF(start)
        ratio = (
            ((point.x() - start.x()) * delta_x)
            + ((point.y() - start.y()) * delta_y)
        ) / length_squared
        ratio = max(0.0, min(1.0, ratio))
        return QPointF(
            start.x() + delta_x * ratio,
            start.y() + delta_y * ratio,
        )

    def _nearest_point_on_polygon(
        self,
        polygon: QPolygonF,
        point: QPointF,
    ) -> QPointF:
        if polygon.isEmpty():
            return QPointF()
        nearest_point = QPointF(polygon[0])
        min_distance = self._distance_squared(nearest_point, point)
        for index in range(polygon.count()):
            start = QPointF(polygon[index])
            end = QPointF(polygon[(index + 1) % polygon.count()])
            candidate = self._nearest_point_on_segment(point, start, end)
            distance = self._distance_squared(candidate, point)
            if distance < min_distance:
                min_distance = distance
                nearest_point = candidate
        return nearest_point

    @staticmethod
    def _nearest_point_on_rect(rect: QRect, point: QPointF) -> QPointF:
        return QPointF(
            float(max(rect.left(), min(int(round(point.x())), rect.right()))),
            float(max(rect.top(), min(int(round(point.y())), rect.bottom()))),
        )

    @staticmethod
    def _rect_intersection_area(rect_a: QRect, rect_b: QRect) -> int:
        intersection = rect_a.intersected(rect_b)
        if intersection.isNull():
            return 0
        return intersection.width() * intersection.height()

    def _resolve_copy_button_rect(
        self,
        polygon: QPolygonF,
        button_width: int,
        button_height: int,
    ) -> QRect:
        bounds = polygon.boundingRect().toRect()
        center = bounds.center()
        gap = 8
        margin = 4
        max_x = max(margin, self.width() - button_width - margin)
        max_y = max(margin, self.height() - button_height - margin)
        candidate_positions = [
            (bounds.right() + gap, center.y() - button_height // 2),
            (center.x() - button_width // 2, bounds.bottom() + gap),
            (
                center.x() - button_width // 2,
                bounds.top() - button_height - gap,
            ),
            (
                bounds.left() - button_width - gap,
                center.y() - button_height // 2,
            ),
            (bounds.right() + gap, bounds.bottom() + gap),
            (bounds.right() + gap, bounds.top() - button_height - gap),
            (bounds.left() - button_width - gap, bounds.bottom() + gap),
            (
                bounds.left() - button_width - gap,
                bounds.top() - button_height - gap,
            ),
        ]
        best_rect = QRect()
        best_score = None
        expanded_bounds = bounds.adjusted(-2, -2, 2, 2)
        for index, (candidate_x, candidate_y) in enumerate(
            candidate_positions
        ):
            resolved_rect = QRect(0, 0, button_width, button_height)
            resolved_rect.moveTo(
                max(margin, min(candidate_x, max_x)),
                max(margin, min(candidate_y, max_y)),
            )
            button_center = QPointF(
                float(resolved_rect.center().x()),
                float(resolved_rect.center().y()),
            )
            nearest_point = self._nearest_point_on_polygon(
                polygon, button_center
            )
            overlap_area = self._rect_intersection_area(
                resolved_rect,
                expanded_bounds,
            )
            distance = self._distance_squared(button_center, nearest_point)
            score = (overlap_area * 1000000) + distance + index
            if best_score is None or score < best_score:
                best_score = score
                best_rect = resolved_rect
        return best_rect

    def block_view_rect(self, block_key: str) -> QRect:
        block = self._block_for_key(block_key)
        if block is None:
            return QRect()
        x, y = self._canvas_origin()
        polygon = self._to_polygon(block.points, x, y)
        if polygon.isEmpty():
            return QRect()
        return polygon.boundingRect().toAlignedRect()

    def _draw_hovered_label(
        self,
        painter: QPainter,
        text: str,
        bounding_rect,
        background_color: QColor,
    ) -> None:
        if not text:
            return
        if not background_color.isValid():
            background_color = QColor(70, 88, 255)
        font = QFont(self.font())
        font.setPixelSize(10)
        font.setWeight(int(QFont.Weight.DemiBold))
        painter.setFont(font)
        fm = QFontMetrics(font)
        padding_x = 8
        padding_y = 2
        text_rect = fm.tightBoundingRect(text)
        rect_width = text_rect.width() + 2 * padding_x
        rect_height = fm.height() + 2 * padding_y
        bg_x = int(bounding_rect.left())
        bg_y = int(bounding_rect.top() - rect_height)
        if bg_y < 0:
            bg_y = int(bounding_rect.top())
        max_x = max(0, self.width() - rect_width - 1)
        bg_x = max(0, min(bg_x, max_x))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background_color)
        painter.drawRect(bg_x, bg_y, rect_width, rect_height)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            bg_x,
            bg_y,
            rect_width,
            rect_height,
            int(Qt.AlignmentFlag.AlignCenter),
            text,
        )

    def _on_copy_button_clicked(self) -> None:
        block_key = self._action_block_key()
        if block_key:
            self.blockCopyRequested.emit(block_key)

    def _action_block_key(self) -> str:
        return self._local_hovered_key

    def _should_preserve_hover(self, position: QPointF) -> bool:
        block_key = self._action_block_key()
        if not block_key or not self._copy_button.isVisible():
            return False
        pos = position.toPoint()
        button_rect = self._copy_button.geometry()
        if button_rect.adjusted(-6, -6, 6, 6).contains(pos):
            return True
        block = self._block_for_key(block_key)
        if block is None:
            return False
        x, y = self._canvas_origin()
        polygon = self._to_polygon(block.points, x, y)
        if polygon.isEmpty():
            return False
        anchor_point = self._nearest_point_on_polygon(
            polygon,
            QPointF(
                float(button_rect.center().x()),
                float(button_rect.center().y()),
            ),
        )
        contact_point = self._nearest_point_on_rect(button_rect, anchor_point)
        bridge_left = int(min(anchor_point.x(), contact_point.x())) - 8
        bridge_right = int(max(anchor_point.x(), contact_point.x())) + 8
        bridge_top = int(min(anchor_point.y(), contact_point.y())) - 8
        bridge_bottom = int(max(anchor_point.y(), contact_point.y())) + 8
        bridge_rect = QRect(
            bridge_left,
            bridge_top,
            max(1, bridge_right - bridge_left + 1),
            max(1, bridge_bottom - bridge_top + 1),
        )
        return bridge_rect.contains(pos)

    def _update_copy_button_for_hovered_block(self) -> None:
        block = self._block_for_key(self._action_block_key())
        if block is None:
            self._copy_button.hide()
            return
        x, y = self._canvas_origin()
        polygon = self._to_polygon(block.points, x, y)
        if polygon.isEmpty():
            self._copy_button.hide()
            return
        button_width = max(88, self._copy_button.sizeHint().width())
        button_height = 34
        self._copy_button.resize(button_width, button_height)
        button_rect = self._resolve_copy_button_rect(
            polygon,
            button_width,
            button_height,
        )
        self._copy_button.move(button_rect.topLeft())
        self._copy_button.show()
        self._copy_button.raise_()


class PPOCRBlockCard(QFrame):
    copyRequested = pyqtSignal(object)
    correctRequested = pyqtSignal(object)
    blockHovered = pyqtSignal(str)
    blockSelected = pyqtSignal(str)

    def __init__(
        self,
        block: PPOCRBlockData,
        root_dir: Path,
        parent=None,
        formula_render_delay_ms: int = 0,
    ) -> None:
        super().__init__(parent)
        self.block = block
        self.root_dir = root_dir
        self._hovered = False
        self._linked_hovered = False
        self._selected = False
        self._selected_locally = False
        self._has_any_selected = False
        self._actions_hovered = False
        self._formula_pixmap = QPixmap()
        self._scaled_formula_pixmap = QPixmap()
        self._scaled_formula_width = -1
        self._formula_render_pending = False
        self._formula_render_delay_ms = max(0, int(formula_render_delay_ms))
        self._formula_render_timer = QTimer(self)
        self._formula_render_timer.setSingleShot(True)
        self._formula_render_timer.timeout.connect(
            self._render_formula_preview_if_needed
        )
        self.setStyleSheet("QFrame { background: transparent; border: none; }")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.content_frame = QFrame()
        self.content_frame.setObjectName("PPOCRBlockCardContentFrame")
        self.content_frame.setStyleSheet(
            get_card_style(
                block.category_color,
                block.edited,
                active=False,
            )
        )
        content_layout = QVBoxLayout(self.content_frame)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(12)
        layout.addWidget(self.content_frame)

        self.label_chip = QLabel(block.display_label, self)
        self.label_chip.setStyleSheet(
            get_card_label_style(block.category_color)
        )
        self.label_chip.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.label_chip.hide()
        label_margin_top = max(1, self.label_chip.sizeHint().height())
        layout.setContentsMargins(
            0,
            label_margin_top,
            0,
            0,
        )

        self._has_image = False
        self.image_label = QLabel(self.content_frame)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setVisible(False)
        self.image_label.setStyleSheet(
            "QLabel { background: transparent; border: none; }"
        )
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.image_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        if block.image_path:
            image_path = root_dir / block.image_path
            if image_path.exists():
                pixmap = QPixmap(str(image_path))
                if not pixmap.isNull():
                    self.image_label.setPixmap(
                        pixmap.scaled(
                            280,
                            220,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                    self._has_image = True
                    self.image_label.setVisible(True)

        self._has_text = bool(block.content.strip())
        self._has_formula = (
            normalize_block_label(block.label) in FORMULA_BLOCK_LABELS
        )
        self._table_html = _table_tokens_to_html(block.content)
        self._has_table = normalize_block_label(
            block.label
        ) in TABLE_BLOCK_LABELS or bool(self._table_html)
        self.content_label = QTextEdit(self.content_frame)
        self.content_label.setReadOnly(True)
        self.content_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.content_label.setFrameShape(QFrame.Shape.NoFrame)
        self.content_label.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.content_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.content_label.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.content_label.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.content_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.content_label.viewport().setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.formula_label = QLabel(self.content_frame)
        self.formula_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formula_label.setVisible(False)
        self.formula_label.setStyleSheet(
            "QLabel { background: transparent; border: none; padding: 0px; }"
        )
        self.formula_label.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self.formula_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        if self._has_formula:
            self.formula_label.setFixedSize(1, 24)
            self.formula_label.show()
            render_delay_ms = self._formula_render_delay_ms
            if render_delay_ms > 0 and has_cached_latex_preview_pixmap(
                block.content
            ):
                render_delay_ms = 0
            self._schedule_formula_preview_render(render_delay_ms)
        elif self._has_table:
            if self._table_html:
                self.content_label.setHtml(self._table_html)
            else:
                self.content_label.setPlainText(block.content or "")
        elif is_rich_text_html(block.content):
            self.content_label.setHtml(block.content)
        else:
            inline_formula_html = _text_with_inline_formulas_to_html(
                block.content
            )
            if inline_formula_html:
                self.content_label.setHtml(inline_formula_html)
            else:
                self.content_label.setMarkdown(block.content)
        self.content_label.document().setDocumentMargin(0)
        self.content_label.setStyleSheet(
            "QTextEdit {"
            "background: transparent;"
            "border: none;"
            "padding: 0px;"
            "font-size: 13px;"
            "selection-background-color: rgb(70, 88, 255);"
            "}"
        )
        self.content_label.document().documentLayout().documentSizeChanged.connect(
            self._update_content_height
        )
        self.content_label.setVisible(self._has_text and not self._has_formula)

        if self._has_image:
            content_layout.addWidget(self.image_label)
        if self._has_formula:
            content_layout.addWidget(
                self.formula_label,
                0,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            )
        elif self._has_text:
            content_layout.addWidget(self.content_label)

        self.actions_widget = QFrame(self)
        self.actions_widget.setStyleSheet(get_floating_action_bar_style())
        self.actions_widget.setGraphicsEffect(
            create_floating_shadow(self.actions_widget)
        )
        self.actions_widget.installEventFilter(self)
        actions_row = QHBoxLayout(self.actions_widget)
        actions_row.setContentsMargins(8, 4, 8, 4)
        actions_row.setSpacing(2)

        self.copy_button = QPushButton(self.tr("Copy"))
        self.copy_button.setIcon(QIcon(new_icon_path("copy", "svg")))
        self.copy_button.setIconSize(QSize(13, 13))
        self.copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_button.setStyleSheet(get_floating_action_button_style())
        self.copy_button.clicked.connect(
            lambda: self.copyRequested.emit(self.block)
        )
        self.copy_button.installEventFilter(self)
        self.correct_button = QPushButton(self.tr("Correct"))
        self.correct_button.setIcon(QIcon(new_icon_path("edit", "svg")))
        self.correct_button.setIconSize(QSize(13, 13))
        self.correct_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.correct_button.setStyleSheet(get_floating_action_button_style())
        self.correct_button.clicked.connect(
            lambda: self.correctRequested.emit(self.block)
        )
        self.correct_button.installEventFilter(self)
        actions_row.addWidget(self.copy_button)
        actions_row.addWidget(self.correct_button)
        self.actions_widget.hide()
        self.destroyed.connect(self.actions_widget.deleteLater)
        self._update_content_height()
        self._update_card_state()

    def refresh_selection_state(
        self,
        selected_key: str,
        selected_locally: bool = False,
    ) -> None:
        self._has_any_selected = bool(selected_key)
        self._selected = bool(
            selected_key and self.block.block_key == selected_key
        )
        self._selected_locally = bool(self._selected and selected_locally)
        if selected_key and not self._selected:
            self.setWindowOpacity(0.35)
        else:
            self.setWindowOpacity(1.0)
        self._update_card_state()

    def refresh_hover_state(self, hovered_key: str) -> None:
        self._linked_hovered = bool(
            hovered_key and self.block.block_key == hovered_key
        )
        self._update_card_state()

    def enterEvent(self, event):
        self._hovered = True
        self._update_card_state()
        self.blockHovered.emit(self.block.block_key)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._actions_hovered = self._cursor_over_actions_widget()
        self._update_card_state()
        if not self._actions_hovered:
            self.blockHovered.emit("")
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.blockSelected.emit(self.block.block_key)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.correctRequested.emit(self.block)
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_formula_preview_pixmap()
        self._position_floating_widgets()
        self._update_content_height()

    def moveEvent(self, event):
        super().moveEvent(event)
        self._position_floating_widgets()

    def showEvent(self, event):
        super().showEvent(event)
        if self._has_formula and self._formula_pixmap.isNull():
            self._schedule_formula_preview_render(0)
        self._update_formula_preview_pixmap()
        self._position_floating_widgets()

    def eventFilter(self, watched, event):
        action_hover_widgets = {
            getattr(self, "actions_widget", None),
            getattr(self, "copy_button", None),
            getattr(self, "correct_button", None),
        }
        if watched in action_hover_widgets:
            if event.type() == QEvent.Type.Enter:
                self._actions_hovered = True
                self._update_card_state()
            elif event.type() == QEvent.Type.Leave:
                QTimer.singleShot(0, self._sync_actions_hover_state)
        return super().eventFilter(watched, event)

    def _update_content_height(self, *_args) -> None:
        if not self._has_text:
            return
        if self._has_formula and self.formula_label.isVisible():
            pixmap = self.formula_label.pixmap()
            if pixmap is not None and not pixmap.isNull():
                self.formula_label.setFixedSize(pixmap.size())
            return
        content_height = max(
            24, int(self.content_label.document().size().height()) + 6
        )
        if self.content_label.height() != content_height:
            self.content_label.setFixedHeight(content_height)

    def _set_formula_preview(self, content: str) -> None:
        try:
            self._formula_pixmap = render_latex_preview_pixmap(content)
        except Exception:
            self._formula_pixmap = QPixmap()
            self._scaled_formula_pixmap = QPixmap()
            self._scaled_formula_width = -1
            self.content_label.setPlainText(content or "")
            self.content_label.setVisible(self._has_text)
            self.formula_label.hide()
            self._has_formula = False
            return
        self._scaled_formula_pixmap = QPixmap()
        self._scaled_formula_width = -1
        self.content_label.hide()
        self.formula_label.show()
        self._update_formula_preview_pixmap()

    def _schedule_formula_preview_render(self, delay_ms: int = 0) -> None:
        if not self._has_formula or not self._formula_pixmap.isNull():
            return
        if delay_ms <= 0:
            self._render_formula_preview_if_needed()
            return
        if self._formula_render_pending:
            return
        self._formula_render_pending = True
        self._formula_render_timer.start(delay_ms)

    def _render_formula_preview_if_needed(self) -> None:
        self._formula_render_pending = False
        if not self._has_formula or not self._formula_pixmap.isNull():
            return
        self._set_formula_preview(self.block.content)

    def _update_formula_preview_pixmap(self) -> None:
        if not self._has_formula or self._formula_pixmap.isNull():
            return
        available_width = max(48, self.content_frame.width() - 32)
        parent = self.parentWidget()
        if parent is not None:
            visible_rect = self._visible_rect_in(parent)
            visible_width = visible_rect.width()
            layout = parent.layout()
            if layout is not None:
                margins = layout.contentsMargins()
                visible_width -= margins.left() + margins.right()
            if visible_width > 0:
                available_width = min(
                    available_width,
                    max(48, visible_width - 32),
                )
        pixmap = self._formula_pixmap
        if pixmap.width() > available_width:
            if (
                self._scaled_formula_width != available_width
                or self._scaled_formula_pixmap.isNull()
            ):
                self._scaled_formula_pixmap = pixmap.scaledToWidth(
                    available_width,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._scaled_formula_width = available_width
            pixmap = self._scaled_formula_pixmap
        elif self._scaled_formula_width >= 0:
            self._scaled_formula_pixmap = QPixmap()
            self._scaled_formula_width = -1
        self.formula_label.setPixmap(pixmap)
        self.formula_label.setFixedSize(pixmap.size())
        self.formula_label.updateGeometry()

    def _overlay_parent(self) -> QWidget:
        return self.parentWidget() or self

    def _cards_scroll_area(self) -> QScrollArea | None:
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                return parent
            parent = parent.parentWidget()
        return None

    def _visible_rect_in(self, target: QWidget) -> QRect:
        scroll_area = self._cards_scroll_area()
        if scroll_area is None:
            return target.rect()
        viewport = scroll_area.viewport()
        top_left_global = viewport.mapToGlobal(QPoint(0, 0))
        top_left = target.mapFromGlobal(top_left_global)
        return QRect(top_left, viewport.size())

    def _ensure_actions_overlay_parent(self) -> QWidget:
        overlay_parent = self._overlay_parent()
        if self.actions_widget.parentWidget() is overlay_parent:
            return overlay_parent
        self.actions_widget.setParent(overlay_parent)
        return overlay_parent

    def _cursor_over_actions_widget(self) -> bool:
        overlay_parent = self.actions_widget.parentWidget()
        if overlay_parent is None or not self.actions_widget.isVisible():
            return False
        cursor_pos = overlay_parent.mapFromGlobal(QCursor.pos())
        return self.actions_widget.geometry().contains(cursor_pos)

    def _sync_actions_hover_state(self) -> None:
        self._actions_hovered = self._cursor_over_actions_widget()
        self._update_card_state()
        if not self._hovered and not self._actions_hovered:
            self.blockHovered.emit("")

    def _position_floating_widgets(self) -> None:
        content_geometry = self.content_frame.geometry()
        if content_geometry.isNull():
            return
        self.label_chip.adjustSize()
        label_height = self.label_chip.height()
        container_layout = self.layout()
        if container_layout is not None:
            margins = container_layout.contentsMargins()
            if margins.top() != label_height:
                container_layout.setContentsMargins(
                    margins.left(),
                    label_height,
                    margins.right(),
                    margins.bottom(),
                )
                content_geometry = self.content_frame.geometry()
        label_y = content_geometry.y() - label_height
        self.label_chip.move(content_geometry.x(), label_y)
        self.label_chip.raise_()

        overlay_parent = self._ensure_actions_overlay_parent()
        pill_size = self.actions_widget.sizeHint()
        pill_width = max(140, pill_size.width())
        pill_height = max(36, pill_size.height())
        self.actions_widget.resize(pill_width, pill_height)
        content_top_left = self.content_frame.mapTo(
            overlay_parent, QPoint(0, 0)
        )
        pill_x = content_top_left.x() + self.content_frame.width() - pill_width
        pill_y = content_top_left.y() - pill_height
        visible_rect = self._visible_rect_in(overlay_parent)
        if not visible_rect.isNull():
            min_x = visible_rect.left()
            max_x = visible_rect.right() - pill_width + 1
            min_y = visible_rect.top()
            max_y = visible_rect.bottom() - pill_height + 1
            layout = overlay_parent.layout()
            if layout is not None:
                margins = layout.contentsMargins()
                min_x += margins.left()
                max_x -= margins.right()
                min_y += margins.top()
                max_y -= margins.bottom()
            if max_x < min_x:
                pill_x = min_x
            else:
                pill_x = max(min_x, min(pill_x, max_x))
            if max_y < min_y:
                pill_y = min_y
            else:
                pill_y = max(min_y, min(pill_y, max_y))
        self.actions_widget.move(max(0, pill_x), max(0, pill_y))
        self.actions_widget.raise_()

    def _update_card_state(self) -> None:
        active = self._hovered or self._linked_hovered or self._actions_hovered
        self.content_frame.setStyleSheet(
            get_card_style(
                self.block.category_color,
                self.block.edited,
                active=active,
            )
        )
        self.label_chip.setVisible(active)
        self.actions_widget.setVisible(self._hovered or self._actions_hovered)
        self._position_floating_widgets()


class PPOCRJsonViewer(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._text_view = QPlainTextEdit(self)
        self._text_view.setObjectName("PPOCRResultJsonViewer")
        self._text_view.setReadOnly(True)
        self._text_view.setFrameShape(QFrame.Shape.NoFrame)
        self._text_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._text_view.document().setDocumentMargin(12)
        layout.addWidget(self._text_view)
        self._apply_style()

    def setPlainText(self, text: str) -> None:
        self._text_view.setPlainText(text or "")
        self._apply_style()

    def toPlainText(self) -> str:
        return self._text_view.toPlainText()

    def _apply_style(self) -> None:
        theme = get_theme()
        self._text_view.setStyleSheet(
            "QPlainTextEdit {"
            f"background: {theme['background']};"
            f"color: {theme['text']};"
            "border: none;"
            "padding: 0px;"
            "font-family: 'SFMono-Regular', 'Menlo', 'Consolas', monospace;"
            "font-size: 12px;"
            "line-height: 1.5;"
            "selection-background-color: rgb(70, 88, 255);"
            "}"
        )


class PPOCRStatusBanner(QFrame):
    retryRequested = pyqtSignal()
    copyLogRequested = pyqtSignal(str)
    cancelRequested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("QFrame { border: none; background: transparent; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(get_overlay_label_style())

        self.detail_label = QLabel()
        self.detail_label.setWordWrap(True)
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        actions_layout = QHBoxLayout()
        actions_layout.addStretch()
        self.copy_log_button = QPushButton(self.tr("Copy Log"))
        self.copy_log_button.setStyleSheet(get_chip_button_style())
        self.copy_log_button.clicked.connect(self._emit_copy)
        self.retry_button = QPushButton(self.tr("Reparse"))
        self.retry_button.setStyleSheet(get_chip_button_style())
        self.retry_button.clicked.connect(self.retryRequested.emit)
        self.cancel_button = QPushButton(self.tr("Cancel Parsing"))
        self.cancel_button.setStyleSheet(get_chip_button_style())
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        actions_layout.addWidget(self.copy_log_button)
        actions_layout.addWidget(self.retry_button)
        actions_layout.addWidget(self.cancel_button)
        actions_layout.addStretch()

        layout.addStretch()
        layout.addWidget(self.title_label)
        layout.addWidget(self.detail_label)
        layout.addLayout(actions_layout)
        layout.addStretch()
        self._log_text = ""

    def set_pending_state(
        self,
        text: str,
        cancellable: bool = False,
        cancel_enabled: bool = True,
    ) -> None:
        self.title_label.setText(self.tr("Parsing"))
        self.detail_label.setText(text)
        self.copy_log_button.setVisible(False)
        self.retry_button.setVisible(False)
        self.cancel_button.setVisible(cancellable)
        self.cancel_button.setEnabled(cancel_enabled)

    def set_error_state(self, text: str) -> None:
        self._log_text = text
        self.title_label.setText(self.tr("Parsing Failed"))
        self.detail_label.setText(text)
        self.copy_log_button.setVisible(True)
        self.retry_button.setVisible(True)
        self.cancel_button.setVisible(False)

    def _emit_copy(self) -> None:
        self.copyLogRequested.emit(self._log_text)
