from PyQt6.QtCore import QPointF, Qt, QRect, QSize, QUrl, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPalette
from PyQt6.QtMultimedia import (
    QAudioOutput,
    QMediaPlayer,
    QVideoFrame,
    QVideoSink,
)
from PyQt6.QtWidgets import (
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class VideoFrameView(QWidget):
    """A plain ``QWidget`` that paints frames from a ``QVideoSink``.

    ``QVideoWidget`` renders to a native subsurface that ignores parent
    stylesheets / QStackedWidget visibility transitions and, on Linux builds
    without PipeWire / VA-API, frequently produces a blank window.  Instead we
    grab each :class:`QVideoFrame` from a :class:`QVideoSink`, convert it to a
    :class:`QImage` and paint it ourselves — works everywhere QtMultimedia can
    decode the file.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setMinimumSize(QSize(320, 240))
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#000000"))
        self.setPalette(pal)

        self._sink = QVideoSink(self)
        self._sink.videoFrameChanged.connect(self._on_frame)
        self._image: QImage | None = None
        self._frame_size: QSize | None = None
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._drag_pos = None
        self._drag_pan = QPointF(0.0, 0.0)

    # Public
    def video_sink(self):
        return self._sink

    def clear(self):
        self._image = None
        self._frame_size = None
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def current_image(self):
        if self._image is None or self._image.isNull():
            return QImage()
        return self._image.copy()

    def set_zoom(self, zoom, anchor=None):
        old_target = self._target_rect()
        old_zoom = self._zoom
        self._zoom = max(0.25, min(4.0, float(zoom or 1.0)))
        if anchor is not None and old_target.width() and old_target.height():
            anchor = QPointF(float(anchor.x()), float(anchor.y()))
            new_target = self._target_rect(centered=True)
            rx = (anchor.x() - old_target.left()) / old_target.width()
            ry = (anchor.y() - old_target.top()) / old_target.height()
            rx = max(0.0, min(1.0, rx))
            ry = max(0.0, min(1.0, ry))
            self._pan = QPointF(
                anchor.x() - new_target.left() - rx * new_target.width(),
                anchor.y() - new_target.top() - ry * new_target.height(),
            )
        elif self._zoom <= 1.0 and old_zoom > 1.0:
            self._pan = QPointF(0.0, 0.0)
        self._clamp_pan()
        self.update()

    def zoom(self):
        return self._zoom

    def reset_view(self):
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    # Internal
    def _on_frame(self, frame: QVideoFrame):
        if not frame.isValid():
            return
        image = frame.toImage()
        if image.isNull():
            return
        # Convert to a format that paints cheaply across platforms.
        if image.format() != QImage.Format.Format_RGB32:
            image = image.convertToFormat(QImage.Format.Format_RGB32)
        self._image = image
        self._frame_size = image.size()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        if self._image is None or self._image.isNull():
            return
        target = self._target_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(target, self._image)

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._image is not None
            and not self._image.isNull()
        ):
            self._drag_pos = event.position()
            self._drag_pan = QPointF(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            delta = event.position() - self._drag_pos
            self._pan = self._drag_pan + delta
            self._clamp_pan()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._drag_pos is not None
        ):
            self._drag_pos = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        self._clamp_pan()
        return super().resizeEvent(event)

    def _target_rect(self, centered=False):
        if self._image is None or self._image.isNull():
            return QRect()
        rect = self._aspect_fit(self._image.size(), self.size(), self._zoom)
        if centered:
            return rect
        return rect.translated(
            int(round(self._pan.x())), int(round(self._pan.y()))
        )

    def _clamp_pan(self):
        if self._image is None or self._image.isNull():
            self._pan = QPointF(0.0, 0.0)
            return
        centered = self._aspect_fit(
            self._image.size(), self.size(), self._zoom
        )
        viewport = self.rect()
        pan_x = self._pan.x()
        pan_y = self._pan.y()
        if centered.width() <= viewport.width():
            pan_x = 0.0
        else:
            min_x = viewport.right() - centered.right()
            max_x = viewport.left() - centered.left()
            pan_x = max(min_x, min(max_x, pan_x))
        if centered.height() <= viewport.height():
            pan_y = 0.0
        else:
            min_y = viewport.bottom() - centered.bottom()
            max_y = viewport.top() - centered.top()
            pan_y = max(min_y, min(max_y, pan_y))
        self._pan = QPointF(pan_x, pan_y)

    @staticmethod
    def _aspect_fit(frame_size: QSize, widget_size: QSize, zoom=1.0) -> QRect:
        fw = max(1, frame_size.width())
        fh = max(1, frame_size.height())
        ww = max(1, widget_size.width())
        wh = max(1, widget_size.height())
        scale = min(ww / fw, wh / fh) * max(0.25, float(zoom or 1.0))
        tw = int(round(fw * scale))
        th = int(round(fh * scale))
        x = (ww - tw) // 2
        y = (wh - th) // 2
        return QRect(x, y, tw, th)


class VideoPlayer(QWidget):
    """QMediaPlayer wrapper with toolbar (play / step / rate / volume)."""

    positionChanged = pyqtSignal(int)
    durationChanged = pyqtSignal(int)
    playbackStateChanged = pyqtSignal(int)
    errorOccurred = pyqtSignal(str)
    markInRequested = pyqtSignal()
    markOutRequested = pyqtSignal()
    commitSegmentRequested = pyqtSignal()

    def __init__(self, fps_provider=None, parent=None):
        super().__init__(parent)
        self._fps_provider = fps_provider or (lambda: 0.0)
        self._duration_ms = 0

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(8)

        # Video surface — software-painted via QVideoSink (see VideoFrameView).
        self.video_widget = VideoFrameView()
        self.video_widget.setMinimumHeight(280)
        root_layout.addWidget(self.video_widget, 1)

        # Player
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self.video_widget.video_sink())
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(
            self._on_playback_state_changed
        )
        self._player.errorOccurred.connect(self._on_error)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)
        self._audio.setVolume(0.7)
        self._first_frame_pending = False

        # Keyboard shortcuts and controls are owned by the parent dialog.

    # Public API
    def load(self, video_path):
        # Reset before swapping sources so QtMultimedia frees the previous
        # decoder cleanly and the new file's first frame is rendered.
        self._player.stop()
        self.video_widget.clear()
        self._duration_ms = 0
        if not video_path:
            self._player.setSource(QUrl())
            self._first_frame_pending = False
            return
        self._first_frame_pending = True
        self._player.setSource(QUrl.fromLocalFile(video_path))

    def _on_media_status_changed(self, status):
        # When the media reaches Buffered/Loaded the decoder is ready; play and
        # immediately pause to flush the first frame to QVideoWidget so the
        # preview is not blank before the user hits "Play".
        try:
            loaded_states = (
                QMediaPlayer.MediaStatus.LoadedMedia,
                QMediaPlayer.MediaStatus.BufferedMedia,
            )
        except AttributeError:
            loaded_states = ()
        if self._first_frame_pending and status in loaded_states:
            self._first_frame_pending = False
            self._player.play()
            QTimer.singleShot(120, self._pause_first_frame_preview)

    def _pause_first_frame_preview(self):
        self._player.pause()
        self._player.setPosition(0)

    def play(self):
        self._player.play()

    def pause(self):
        self._player.pause()

    def toggle_play(self):
        if (
            self._player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        ):
            self.pause()
        else:
            self.play()

    def stop(self):
        self._player.stop()

    def release(self):
        self._first_frame_pending = False
        self._player.stop()
        try:
            self._player.setVideoOutput(None)
        except TypeError:
            pass
        try:
            self._player.setAudioOutput(None)
        except TypeError:
            pass
        self._player.setSource(QUrl())
        self.video_widget.clear()

    def is_playing(self):
        return (
            self._player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        )

    def seek(self, ms):
        ms = max(0, min(int(ms), self._duration_ms or int(ms)))
        self._player.setPosition(ms)

    def step(self, delta_ms):
        self.seek(self._player.position() + int(delta_ms))

    def step_frames(self, frames=1):
        fps = float(self._fps_provider() or 0.0)
        if fps <= 0:
            self.step(33 * frames)
            return
        self.step(int(round(frames * 1000.0 / fps)))

    def position(self):
        return int(self._player.position())

    def duration(self):
        return int(self._duration_ms or self._player.duration() or 0)

    def set_volume(self, vol_0_to_100):
        self._audio.setVolume(max(0, min(100, int(vol_0_to_100))) / 100.0)

    def volume(self):
        return int(round(self._audio.volume() * 100))

    def set_playback_rate(self, rate):
        try:
            self._player.setPlaybackRate(float(rate))
        except Exception:
            pass

    def current_image(self):
        return self.video_widget.current_image()

    def set_zoom(self, zoom, anchor=None):
        self.video_widget.set_zoom(zoom, anchor)

    def zoom(self):
        return self.video_widget.zoom()

    def reset_view(self):
        self.video_widget.reset_view()

    # Slots
    def _on_playback_state_changed(self, state):
        self.playbackStateChanged.emit(
            int(state.value if hasattr(state, "value") else state)
        )

    def _on_position_changed(self, pos_ms):
        self.positionChanged.emit(int(pos_ms))

    def _on_duration_changed(self, dur_ms):
        self._duration_ms = int(dur_ms)
        self.durationChanged.emit(int(dur_ms))

    def _on_error(self, err, msg=""):
        text = msg or str(err)
        self.errorOccurred.emit(text)
