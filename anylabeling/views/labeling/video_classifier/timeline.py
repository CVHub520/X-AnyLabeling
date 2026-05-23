from PyQt6.QtCore import Qt, QRect, QSize, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFontMetrics,
    QImage,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import QWidget

from anylabeling.views.labeling.utils.theme import get_theme

from .config import (
    MIN_SEGMENT_MS,
    TIMELINE_HANDLE_WIDTH,
    TIMELINE_HEIGHT,
    TIMELINE_PAD_X,
    TIMELINE_TRACK_BOTTOM_PAD,
    TIMELINE_TRACK_GAP,
    TIMELINE_RULER_HEIGHT,
)
from .utils import color_for_label, ms_to_timecode


class TimelineWidget(QWidget):
    """Custom timeline:

    - Right drag on the ruler → emit ``segmentRequested(start_ms, end_ms)``
    - Click on a span → select it (emit ``segmentSelected``)
    - Drag span body → move (emit ``segmentEdited``)
    - Drag left/right handle → resize
    - Hover → emit ``hoverMs`` for tooltip / status
    """

    segmentRequested = pyqtSignal(int, int)  # (start_ms, end_ms)
    segmentCreationBlocked = pyqtSignal()
    segmentSelected = pyqtSignal(str)  # segment id, or "" for none
    segmentDoubleClicked = pyqtSignal(str)
    segmentEdited = pyqtSignal(str, int, int)  # (id, start_ms, end_ms)
    segmentEditFinished = pyqtSignal()
    seekRequested = pyqtSignal(int)  # ms
    hoverMs = pyqtSignal(int)  # ms
    viewChanged = pyqtSignal(
        int, int, int
    )  # start_ms, visible_ms, duration_ms

    HIT_NONE = 0
    HIT_BODY = 1
    HIT_LEFT = 2
    HIT_RIGHT = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFixedHeight(TIMELINE_HEIGHT)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._duration_ms = 0
        self._playhead_ms = 0
        self._segments = (
            []
        )  # list of dicts: {id, label, start_ms, end_ms, color}
        self._selected_id = ""
        self._hover_id = ""
        self._hover_pos = None
        self._hover_ms = 0
        self._current_label_color = None
        self._segment_creation_enabled = False
        self._zoom_factor = 1.0
        self._view_start_ms = 0
        self._thumbnails = []
        self._fps = 0.0

        # interaction state
        self._mode = (
            "idle"  # idle | create | move | resize_left | resize_right | seek
        )
        self._drag_start_ms = 0
        self._drag_current_ms = 0
        self._drag_offset_ms = 0
        self._active_seg_id = ""
        self._active_seg_initial = None  # (start_ms, end_ms)

    # public api
    def set_duration(self, duration_ms):
        self._duration_ms = max(0, int(duration_ms or 0))
        self._playhead_ms = min(self._playhead_ms, self._duration_ms)
        self._clamp_view()
        self.update()
        self._emit_view_changed()

    def duration_ms(self):
        return self._duration_ms

    def set_playhead(self, ms):
        old_view_start = self._view_start_ms
        self._playhead_ms = max(0, min(int(ms), self._duration_ms or int(ms)))
        self._ensure_playhead_visible()
        self.update()
        if self._view_start_ms != old_view_start:
            self._emit_view_changed()

    def set_segments(self, segments, label_colors=None):
        """Receives list of objects with .id, .label, .start_ms, .end_ms."""
        label_colors = label_colors or {}
        self._segments = [
            {
                "id": getattr(s, "id", ""),
                "label": getattr(s, "label", ""),
                "start_ms": int(getattr(s, "start_ms", 0)),
                "end_ms": int(getattr(s, "end_ms", 0)),
                "color": color_for_label(
                    getattr(s, "label", ""), label_colors
                ),
            }
            for s in segments
        ]
        if self._selected_id and not any(
            seg["id"] == self._selected_id for seg in self._segments
        ):
            self._selected_id = ""
        self.update()

    def set_selected(self, seg_id):
        self._selected_id = seg_id or ""
        self.update()

    def set_current_label_color(self, color):
        self._current_label_color = color
        self.update()

    def set_segment_creation_enabled(self, enabled):
        self._segment_creation_enabled = bool(enabled)

    def set_zoom_factor(self, factor):
        self._zoom_factor = max(1.0, float(factor or 1.0))
        self._center_view_on(self._playhead_ms)
        self.update()
        self._emit_view_changed()

    def zoom_factor(self):
        return self._zoom_factor

    def set_view_start(self, ms):
        visible = self._visible_duration_ms()
        max_start = max(0, self._duration_ms - visible)
        self._view_start_ms = max(0, min(int(ms or 0), max_start))
        self.update()
        self._emit_view_changed()

    def view_start_ms(self):
        return self._view_start_ms

    def visible_ms(self):
        return self._visible_duration_ms()

    def set_fps(self, fps):
        try:
            self._fps = max(0.0, float(fps or 0.0))
        except (TypeError, ValueError):
            self._fps = 0.0
        self.update()

    def set_thumbnails(self, thumbnails):
        self._thumbnails = [
            image.copy()
            for image in thumbnails or []
            if isinstance(image, QImage) and not image.isNull()
        ]
        self.update()

    def hover_hint(self, pos):
        if self._duration_ms <= 0:
            return ""
        hit, _seg = self._hit_test(pos)
        if hit != self.HIT_NONE:
            return "segment"
        if self._ruler_rect().contains(pos):
            return "ruler"
        return ""

    def sizeHint(self):
        return QSize(800, TIMELINE_HEIGHT)

    # painting
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        t = get_theme()

        painter.fillRect(rect, QColor(t["background_secondary"]))

        track_rect = self._track_rect()
        painter.fillRect(track_rect, QColor(t["surface"]))
        self._draw_thumbnail_track(painter, track_rect)

        # vertical secondary grid
        self._draw_ruler(painter)

        # segments
        for seg in self._segments:
            self._draw_segment(painter, seg)

        # selection drag preview
        if self._mode == "create":
            self._draw_drag_preview(painter)

        # playhead
        self._draw_playhead(painter)
        self._draw_hover_guide(painter)
        self._draw_hover_timecode(painter)

    def _track_rect(self):
        r = self.rect()
        top = r.top() + TIMELINE_RULER_HEIGHT + TIMELINE_TRACK_GAP
        height = max(
            0,
            r.height()
            - TIMELINE_RULER_HEIGHT
            - TIMELINE_TRACK_GAP
            - TIMELINE_TRACK_BOTTOM_PAD,
        )
        return QRect(
            r.left() + TIMELINE_PAD_X,
            top,
            max(0, r.width() - TIMELINE_PAD_X * 2),
            height,
        )

    def _ruler_rect(self):
        rect = self._track_rect()
        return QRect(rect.left(), 0, rect.width(), TIMELINE_RULER_HEIGHT)

    def _visible_range(self):
        if self._duration_ms <= 0:
            return 0, 0
        visible = self._visible_duration_ms()
        start = max(0, min(self._view_start_ms, self._duration_ms - visible))
        return start, start + visible

    def _visible_duration_ms(self):
        if self._duration_ms <= 0:
            return 0
        visible = max(1, int(self._duration_ms / self._zoom_factor))
        return min(visible, self._duration_ms)

    def _emit_view_changed(self):
        self.viewChanged.emit(
            self._view_start_ms,
            self._visible_duration_ms(),
            self._duration_ms,
        )

    def _clamp_view(self):
        start, _ = self._visible_range()
        self._view_start_ms = start

    def _center_view_on(self, ms):
        if self._duration_ms <= 0:
            self._view_start_ms = 0
            return
        visible = self._visible_duration_ms()
        self._view_start_ms = max(
            0, min(self._duration_ms - visible, int(ms) - visible // 2)
        )

    def _ensure_playhead_visible(self):
        start, end = self._visible_range()
        if self._playhead_ms < start:
            self._view_start_ms = self._playhead_ms
        elif self._playhead_ms > end:
            visible = max(1, end - start)
            self._view_start_ms = min(
                max(0, self._duration_ms - visible),
                self._playhead_ms - visible,
            )

    def _draw_ruler(self, painter):
        theme = get_theme()
        rect = self._track_rect()
        ruler_rect = self._ruler_rect()
        painter.fillRect(ruler_rect, QColor(theme["background"]))

        if self._duration_ms <= 0 or rect.width() <= 0:
            return

        start_ms, end_ms = self._visible_range()
        visible_ms = max(1, end_ms - start_ms)
        fps = self._fps
        if fps > 0:
            frame_ms = 1000.0 / fps
            if rect.width() / visible_ms * frame_ms >= 10:
                self._draw_frame_ruler(
                    painter, rect, ruler_rect, start_ms, end_ms, fps
                )
                return

        self._draw_time_ruler(painter, rect, ruler_rect, start_ms, end_ms)

    def _draw_time_ruler(self, painter, rect, ruler_rect, start_ms, end_ms):
        theme = get_theme()
        major_step = self._choose_time_step(end_ms - start_ms, rect.width())
        minor_step = max(1, major_step // 10)

        pen = QPen(QColor(theme["border_light"]))
        painter.setPen(pen)
        font = painter.font()
        fm = QFontMetrics(font)
        guide_color = QColor(theme["border_light"])
        guide_color.setAlpha(70)

        tick = self._ceil_to_step(start_ms, minor_step)
        last_label_right = rect.left() - 999
        while tick <= end_ms:
            x = self._ms_to_x(tick)
            is_major = tick % major_step == 0
            if is_major:
                painter.setPen(QColor(theme["border_light"]))
                painter.drawLine(x, ruler_rect.top(), x, ruler_rect.bottom())
                painter.setPen(guide_color)
                painter.drawLine(x, ruler_rect.bottom() + 1, x, rect.bottom())
                painter.setPen(QColor(theme["text_secondary"]))
                label = ms_to_timecode(tick, with_ms=False)
                last_label_right = self._draw_ruler_label(
                    painter, fm, label, x, rect, ruler_rect, last_label_right
                )
            else:
                painter.setPen(QColor(theme["border_light"]))
                painter.drawLine(
                    x, ruler_rect.bottom() - 7, x, ruler_rect.bottom()
                )
            tick += minor_step

    def _draw_frame_ruler(
        self, painter, rect, ruler_rect, start_ms, end_ms, fps
    ):
        theme = get_theme()
        frame_ms = 1000.0 / fps
        px_per_frame = rect.width() / max(1, end_ms - start_ms) * frame_ms
        major_frames = self._choose_frame_step(px_per_frame)
        minor_frames = max(1, major_frames // 4)
        fps_int = max(1, int(round(fps)))

        font = painter.font()
        fm = QFontMetrics(font)
        guide_color = QColor(theme["border_light"])
        guide_color.setAlpha(70)
        start_frame = int(start_ms / frame_ms)
        if start_frame * frame_ms < start_ms:
            start_frame += 1
        start_frame = self._ceil_to_step(start_frame, minor_frames)
        end_frame = int(end_ms / frame_ms) + 1
        frame = start_frame
        last_label_right = rect.left() - 999
        while frame <= end_frame:
            ms = int(round(frame * frame_ms))
            x = self._ms_to_x(ms)
            is_major = frame % major_frames == 0
            if is_major:
                painter.setPen(QColor(theme["border_light"]))
                painter.drawLine(x, ruler_rect.top(), x, ruler_rect.bottom())
                painter.setPen(guide_color)
                painter.drawLine(x, ruler_rect.bottom() + 1, x, rect.bottom())
                painter.setPen(QColor(theme["text_secondary"]))
                label = self._format_frame_tick(frame, fps_int, fps)
                last_label_right = self._draw_ruler_label(
                    painter, fm, label, x, rect, ruler_rect, last_label_right
                )
            else:
                painter.setPen(QColor(theme["border_light"]))
                painter.drawLine(
                    x, ruler_rect.bottom() - 7, x, ruler_rect.bottom()
                )
            frame += minor_frames

    @staticmethod
    def _choose_time_step(visible_ms, width):
        target_px = 260.0
        ms_per_px = max(1.0, float(visible_ms or 1)) / max(1, width)
        target_ms = target_px * ms_per_px
        candidates = [
            1000,
            2000,
            3000,
            5000,
            10000,
            15000,
            30000,
            60000,
            120000,
            180000,
            300000,
            600000,
            900000,
            1200000,
            1800000,
            3600000,
        ]
        for step in candidates:
            if step >= target_ms:
                return step
        return candidates[-1]

    @staticmethod
    def _choose_frame_step(px_per_frame):
        target_px = 96.0
        candidates = [2, 4, 5, 10, 15, 20, 30, 60, 120, 300]
        for frames in candidates:
            if frames * px_per_frame >= target_px:
                return frames
        return candidates[-1]

    @staticmethod
    def _format_frame_tick(frame, fps_int, fps):
        if frame % fps_int == 0:
            return ms_to_timecode(
                int(round(frame / fps * 1000)), with_ms=False
            )
        return f"{frame % fps_int}f"

    @staticmethod
    def _ceil_to_step(value, step):
        step = max(1, int(step))
        return ((int(value) + step - 1) // step) * step

    @staticmethod
    def _draw_ruler_label(
        painter, font_metrics, label, x, rect, ruler_rect, last_label_right
    ):
        tw = font_metrics.horizontalAdvance(label)
        tx = max(rect.left(), min(rect.right() - tw, x - tw // 2))
        if tx <= last_label_right + 14:
            return last_label_right
        painter.drawText(tx, ruler_rect.top() + font_metrics.ascent(), label)
        return tx + tw

    def _draw_thumbnail_track(self, painter, rect):
        theme = get_theme()
        painter.save()
        painter.setClipRect(rect)
        painter.fillRect(rect, QColor(theme["surface"]))
        if self._thumbnails:
            count = len(self._thumbnails)
            tile_w = self._thumbnail_tile_width(rect.height())
            x = rect.left()
            while x <= rect.right():
                center_ms = self._x_to_ms(x + tile_w // 2)
                idx = self._thumbnail_index_for_ms(center_ms, count)
                target = QRect(x, rect.top(), tile_w + 1, rect.height())
                self._draw_aspect_fill_image(
                    painter, self._thumbnails[idx], target
                )
                x += tile_w
        else:
            stripe = QColor(theme["background_secondary"])
            for idx in range(16):
                if idx % 2 == 0:
                    x = rect.left() + int(rect.width() * idx / 16)
                    w = int(rect.width() / 16) + 1
                    painter.fillRect(
                        QRect(x, rect.top(), w, rect.height()), stripe
                    )

        overlay = QColor(theme["background"])
        overlay.setAlpha(120)
        painter.fillRect(rect, overlay)
        painter.restore()

    @staticmethod
    def _draw_aspect_fill_image(painter, image, target):
        if image.isNull() or target.width() <= 0 or target.height() <= 0:
            return
        iw = max(1, image.width())
        ih = max(1, image.height())
        target_ratio = target.width() / max(1, target.height())
        image_ratio = iw / ih
        if image_ratio > target_ratio:
            sw = int(round(ih * target_ratio))
            sx = max(0, (iw - sw) // 2)
            source = QRect(sx, 0, max(1, sw), ih)
        else:
            sh = int(round(iw / target_ratio))
            sy = max(0, (ih - sh) // 2)
            source = QRect(0, sy, iw, max(1, sh))
        painter.drawImage(target, image, source)

    def _thumbnail_tile_width(self, height):
        image = self._thumbnails[0]
        ratio = image.width() / max(1, image.height())
        return max(24, int(round(max(1, height) * ratio)))

    def _thumbnail_index_for_ms(self, ms, count):
        if self._duration_ms <= 0 or count <= 1:
            return 0
        frac = max(0.0, min(1.0, float(ms) / float(self._duration_ms)))
        return min(count - 1, int(round(frac * (count - 1))))

    def _draw_segment(self, painter, seg):
        t = get_theme()
        rect = self._track_rect()
        start_ms, end_ms = self._visible_range()
        if seg["end_ms"] < start_ms or seg["start_ms"] > end_ms:
            return
        x1 = self._ms_to_x(seg["start_ms"])
        x2 = self._ms_to_x(seg["end_ms"])
        if x2 - x1 < 1:
            x2 = x1 + 1
        body = QRect(x1, rect.top(), x2 - x1, rect.height())
        color = QColor(seg["color"])

        fill = QColor(color)
        fill.setAlpha(170 if seg["id"] == self._selected_id else 125)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(body, 5, 5)

        # border
        is_selected = seg["id"] == self._selected_id
        border_color = (
            QColor(t["primary"]) if is_selected else QColor(color).darker(130)
        )
        painter.setPen(QPen(border_color, 2.0 if is_selected else 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(body, 5, 5)

    def _draw_drag_preview(self, painter):
        s = min(self._drag_start_ms, self._drag_current_ms)
        e = max(self._drag_start_ms, self._drag_current_ms)
        s, e = self._clamp_created_range(s, e)
        if e - s < 1:
            return
        rect = self._track_rect()
        x1 = self._ms_to_x(s)
        x2 = self._ms_to_x(e)
        body = QRect(x1, rect.top(), x2 - x1, rect.height())
        color = QColor(self._current_label_color or "#A0A0A0")
        fill = QColor(color)
        fill.setAlpha(90)
        painter.setBrush(QBrush(fill))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(body, 5, 5)
        painter.setPen(QPen(color.darker(150), 1.5, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(body, 5, 5)

    def _draw_playhead(self, painter):
        t = get_theme()
        rect = self._track_rect()
        x = self._ms_to_x(self._playhead_ms)
        pen = QPen(QColor(t["error"]), 2)
        painter.setPen(pen)
        painter.drawLine(x, 0, x, rect.bottom())
        # small triangle on top
        painter.setBrush(QBrush(QColor(t["error"])))
        painter.setPen(Qt.PenStyle.NoPen)
        tri = [
            (x, TIMELINE_RULER_HEIGHT),
            (x - 5, TIMELINE_RULER_HEIGHT - 7),
            (x + 5, TIMELINE_RULER_HEIGHT - 7),
        ]
        from PyQt6.QtGui import QPolygon
        from PyQt6.QtCore import QPoint

        painter.drawPolygon(
            QPolygon([QPoint(int(px), int(py)) for px, py in tri])
        )

    def _draw_hover_timecode(self, painter):
        if self._hover_pos is None or self._duration_ms <= 0:
            return
        theme = get_theme()
        ruler_rect = self._ruler_rect()
        if not ruler_rect.isValid():
            return
        label = self._format_hover_timecode(self._hover_ms)
        font = painter.font()
        fm = QFontMetrics(font)
        pad_x = 6
        pad_y = 2
        width = fm.horizontalAdvance(label) + pad_x * 2
        height = fm.height() + pad_y * 2
        line_x = self._ms_to_x(self._hover_ms)
        x = line_x + 8
        if x + width > ruler_rect.right() + 1:
            x = line_x - width - 8
        y = ruler_rect.top() + max(1, (ruler_rect.height() - height) // 2)
        x = max(ruler_rect.left(), min(ruler_rect.right() - width + 1, x))
        bubble = QRect(x, y, width, height)
        bg = QColor(theme["tooltip_bg"])
        bg.setAlpha(230)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(bubble, 4, 4)
        painter.setPen(QColor(theme["tooltip_text"]))
        painter.drawText(
            bubble,
            int(Qt.AlignmentFlag.AlignCenter),
            label,
        )

    def _draw_hover_guide(self, painter):
        if self._hover_pos is None or self._duration_ms <= 0:
            return
        theme = get_theme()
        rect = self._track_rect()
        x = self._ms_to_x(self._hover_ms)
        color = QColor(theme["primary"])
        color.setAlpha(180)
        painter.save()
        painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
        painter.drawLine(x, TIMELINE_RULER_HEIGHT, x, rect.bottom())
        painter.restore()

    def _format_hover_timecode(self, ms):
        fps = float(self._fps or 0.0)
        if fps > 0:
            fps_int = max(1, int(round(fps)))
            total_frames = int(round(float(ms or 0) / 1000.0 * fps))
            total_seconds, frame = divmod(total_frames, fps_int)
        else:
            total_seconds = int((ms or 0) // 1000)
            frame = 0
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame:02d}"

    # coordinate helpers
    def _ms_to_x(self, ms):
        rect = self._track_rect()
        start_ms, end_ms = self._visible_range()
        visible = end_ms - start_ms
        if self._duration_ms <= 0 or visible <= 0 or rect.width() <= 0:
            return rect.left()
        frac = max(0.0, min(1.0, float(ms - start_ms) / float(visible)))
        return int(rect.left() + frac * rect.width())

    def _x_to_ms(self, x):
        rect = self._track_rect()
        if rect.width() <= 0:
            return 0
        start_ms, end_ms = self._visible_range()
        frac = (x - rect.left()) / rect.width()
        frac = max(0.0, min(1.0, frac))
        return self._snap_ms_to_frame(
            int(round(start_ms + frac * max(0, end_ms - start_ms)))
        )

    def _snap_ms_to_frame(self, ms):
        if self._fps <= 0:
            return max(0, min(int(ms), self._duration_ms or int(ms)))
        frame_ms = 1000.0 / float(self._fps)
        snapped = int(round(round(float(ms) / frame_ms) * frame_ms))
        return max(0, min(snapped, self._duration_ms or snapped))

    def _min_segment_ms(self):
        if self._fps <= 0:
            return MIN_SEGMENT_MS
        return max(1, int(round(1000.0 / float(self._fps))))

    def _adjacent_bounds(self, seg_id):
        seg = self._find_seg(seg_id)
        if not seg:
            return 0, self._duration_ms
        prev_end = 0
        next_start = self._duration_ms
        for other in self._segments:
            if other["id"] == seg_id:
                continue
            if other["end_ms"] <= seg["start_ms"]:
                prev_end = max(prev_end, int(other["end_ms"]))
            if other["start_ms"] >= seg["end_ms"]:
                next_start = min(next_start, int(other["start_ms"]))
        return prev_end, next_start

    def _move_bounds(self, seg_id, length):
        initial_start, initial_end = self._active_seg_initial
        prev_end = 0
        next_start = self._duration_ms
        for other in self._segments:
            if other["id"] == seg_id:
                continue
            if other["end_ms"] <= initial_start:
                prev_end = max(prev_end, int(other["end_ms"]))
            if other["start_ms"] >= initial_end:
                next_start = min(next_start, int(other["start_ms"]))
        return prev_end, max(prev_end, next_start - int(length))

    def _clamp_created_range(self, start_ms, end_ms):
        if end_ms <= start_ms:
            return start_ms, end_ms
        anchor = self._drag_start_ms
        if self._has_segment_at_ms(anchor):
            return anchor, anchor
        if self._drag_current_ms >= self._drag_start_ms:
            limit = self._duration_ms
            for seg in self._segments:
                if seg["start_ms"] >= anchor:
                    limit = min(limit, int(seg["start_ms"]))
            return start_ms, min(end_ms, limit)
        limit = 0
        for seg in self._segments:
            if seg["end_ms"] <= anchor:
                limit = max(limit, int(seg["end_ms"]))
        return max(start_ms, limit), end_ms

    def _create_ms_for_x(self, x):
        ms = self._x_to_ms(x)
        boundary_ms = self._nearest_segment_boundary_ms(x)
        return ms if boundary_ms is None else boundary_ms

    def _nearest_segment_boundary_ms(self, x):
        tolerance = max(4, TIMELINE_HANDLE_WIDTH)
        start_ms, end_ms = self._visible_range()
        closest_ms = None
        closest_distance = tolerance + 1
        for seg in self._segments:
            for ms in (seg["start_ms"], seg["end_ms"]):
                if ms < start_ms or ms > end_ms:
                    continue
                distance = abs(x - self._ms_to_x(ms))
                if distance < closest_distance:
                    closest_ms = int(ms)
                    closest_distance = distance
        return closest_ms if closest_distance <= tolerance else None

    def _has_segment_at_ms(self, ms):
        for seg in self._segments:
            if seg["start_ms"] < ms < seg["end_ms"]:
                return True
        return False

    # hit testing
    def _hit_test(self, pos):
        rect = self._track_rect()
        if not rect.contains(pos):
            return self.HIT_NONE, None
        x = pos.x()
        start_ms, end_ms = self._visible_range()
        for seg in reversed(self._segments):
            if seg["end_ms"] < start_ms or seg["start_ms"] > end_ms:
                continue
            x1 = self._ms_to_x(seg["start_ms"])
            x2 = self._ms_to_x(seg["end_ms"])
            if seg["id"] == self._selected_id:
                # check handles first
                hw = TIMELINE_HANDLE_WIDTH
                if abs(x - x1) <= hw // 2 + 1:
                    return self.HIT_LEFT, seg
                if abs(x - x2) <= hw // 2 + 1:
                    return self.HIT_RIGHT, seg
            if x1 <= x <= x2:
                return self.HIT_BODY, seg
        return self.HIT_NONE, None

    # events
    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        rect = self._track_rect()
        if event.button() == Qt.MouseButton.RightButton:
            if self._duration_ms > 0 and self._ruler_rect().contains(pos):
                if not self._segment_creation_enabled:
                    self.segmentCreationBlocked.emit()
                    return
                ms = self._create_ms_for_x(pos.x())
                if self._has_segment_at_ms(ms):
                    return
                self._mode = "create"
                self._drag_start_ms = ms
                self._drag_current_ms = ms
                self.set_selected("")
                self.segmentSelected.emit("")
                self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        # ruler click → seek
        if (
            pos.y() < TIMELINE_RULER_HEIGHT
            and rect.left() <= pos.x() <= rect.right()
        ):
            self._mode = "seek"
            ms = self._x_to_ms(pos.x())
            self.seekRequested.emit(ms)
            return

        hit, seg = self._hit_test(pos)
        if hit == self.HIT_NONE:
            self.set_selected("")
            self.segmentSelected.emit("")
            self.update()
            return

        self._active_seg_id = seg["id"]
        self._active_seg_initial = (seg["start_ms"], seg["end_ms"])
        self.set_selected(seg["id"])
        self.segmentSelected.emit(seg["id"])
        if hit == self.HIT_LEFT:
            self._mode = "resize_left"
        elif hit == self.HIT_RIGHT:
            self._mode = "resize_right"
        else:
            self._mode = "move"
            self._drag_offset_ms = self._x_to_ms(pos.x()) - seg["start_ms"]

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        hit, seg = self._hit_test(event.position().toPoint())
        if hit != self.HIT_NONE and seg:
            self.segmentDoubleClicked.emit(seg["id"])
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        ms = self._x_to_ms(pos.x())
        timeline_rect = self._track_rect().united(self._ruler_rect())
        if timeline_rect.contains(pos):
            self._hover_pos = pos
            self._hover_ms = self._create_ms_for_x(pos.x())
        else:
            self._hover_pos = None
        self.hoverMs.emit(ms)
        self.update()

        if self._mode == "idle":
            self._update_cursor(pos)
            return

        if self._mode == "seek":
            self.seekRequested.emit(ms)
            return

        if self._mode == "create":
            self._drag_current_ms = self._create_ms_for_x(pos.x())
            self.update()
            return

        seg = self._find_seg(self._active_seg_id)
        if not seg:
            return

        if self._mode == "move":
            initial_start, initial_end = self._active_seg_initial
            length = initial_end - initial_start
            lower, upper = self._move_bounds(seg["id"], length)
            new_start = ms - self._drag_offset_ms
            new_start = self._snap_ms_to_frame(new_start)
            new_start = max(lower, min(upper, new_start))
            seg["start_ms"] = int(new_start)
            seg["end_ms"] = int(new_start + length)
            self.segmentEdited.emit(seg["id"], seg["start_ms"], seg["end_ms"])
            self.update()
        elif self._mode == "resize_left":
            prev_end, _next_start = self._adjacent_bounds(seg["id"])
            new_start = max(
                prev_end,
                min(seg["end_ms"] - self._min_segment_ms(), ms),
            )
            seg["start_ms"] = int(new_start)
            self.segmentEdited.emit(seg["id"], seg["start_ms"], seg["end_ms"])
            self.update()
        elif self._mode == "resize_right":
            _prev_end, next_start = self._adjacent_bounds(seg["id"])
            new_end = min(
                next_start,
                max(seg["start_ms"] + self._min_segment_ms(), ms),
            )
            seg["end_ms"] = int(new_end)
            self.segmentEdited.emit(seg["id"], seg["start_ms"], seg["end_ms"])
            self.update()

    def mouseReleaseEvent(self, event):
        if self._mode == "create":
            if event.button() != Qt.MouseButton.RightButton:
                return
            s = min(self._drag_start_ms, self._drag_current_ms)
            e = max(self._drag_start_ms, self._drag_current_ms)
            self._mode = "idle"
            self.update()
            s, e = self._clamp_created_range(s, e)
            if e - s >= self._min_segment_ms():
                self.segmentRequested.emit(int(s), int(e))
            self._update_cursor(event.position().toPoint())
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        was_editing = self._mode in ("move", "resize_left", "resize_right")
        self._mode = "idle"
        self._active_seg_id = ""
        self._active_seg_initial = None
        if was_editing:
            self.segmentEditFinished.emit()
        self._update_cursor(event.position().toPoint())

    def leaveEvent(self, event):
        self._hover_pos = None
        self.update()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        return super().leaveEvent(event)

    def _update_cursor(self, pos):
        hit, _ = self._hit_test(pos)
        if hit in (self.HIT_LEFT, self.HIT_RIGHT):
            self.setCursor(Qt.CursorShape.SplitHCursor)
        elif hit == self.HIT_BODY:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif self._segment_creation_enabled and self._ruler_rect().contains(
            pos
        ):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _find_seg(self, seg_id):
        for s in self._segments:
            if s["id"] == seg_id:
                return s
        return None
