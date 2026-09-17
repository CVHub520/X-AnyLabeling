"""Native OpenGL point-cloud display and transactional selection gestures."""

import math

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFunctions_2_0,
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLShader,
    QOpenGLShaderProgram,
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from .selection import (
    SelectionSnapshot,
    project_points,
    select_brush,
    select_polygon,
)

VERTEX_SHADER = """
#version 120
attribute vec3 position;
attribute vec4 color;
attribute float visible;
uniform mat4 matrix;
uniform vec2 viewport;
uniform float pointSize;
varying vec4 pointColor;
void main() {
    vec4 clip = matrix * vec4(position, 1.0);
    if (visible < 0.5 || any(greaterThan(abs(clip.xyz), vec3(clip.w)))
        || clip.x >= clip.w || clip.y <= -clip.w) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    } else {
        vec2 screen = (clip.xy / clip.w + 1.0) * viewport * 0.5;
        screen.y = viewport.y - screen.y;
        screen = floor(screen) + vec2(mod(pointSize, 2.0) * 0.5);
        screen.y = viewport.y - screen.y;
        clip.xy = (screen / viewport * 2.0 - 1.0) * clip.w;
        gl_Position = clip;
    }
    gl_PointSize = pointSize;
    pointColor = color;
}
"""

FRAGMENT_SHADER = """
#version 120
varying vec4 pointColor;
void main() {
    gl_FragColor = pointColor;
}
"""


class _PointCloudOverlay(QtWidgets.QWidget):
    def __init__(self, owner, parent):
        super().__init__(parent)
        self.owner = owner
        self.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.owner._paint_overlay(painter)
        painter.end()


class _PointCloudGLWidget(QOpenGLWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        surface = QtGui.QSurfaceFormat()
        surface.setVersion(2, 1)
        surface.setDepthBufferSize(24)
        surface.setSamples(0)
        self.setFormat(surface)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.functions = None
        self.program = None
        self.buffer = None
        self.color_buffer = None
        self.visibility_buffer = None
        self.preview_buffer = None
        self.picking_buffer = None
        self.picking_framebuffer = None
        self.scene_framebuffer = None
        self.scene_key = None
        self.scene_cache_supported = False
        self.data_dirty = True
        self.color_dirty = True
        self.color_indices = None
        self.visibility_dirty = True
        self.preview_dirty = True
        self.picking_dirty = True
        self.preview_count = 0
        self.overlay = _PointCloudOverlay(owner, self)

    def initializeGL(self):
        try:
            self.functions = QOpenGLFunctions_2_0()
            if not self.functions.initializeOpenGLFunctions():
                raise RuntimeError("OpenGL 2.0 functions are unavailable.")
            self.program = QOpenGLShaderProgram(self)
            for shader_type, source in (
                (QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SHADER),
                (QOpenGLShader.ShaderTypeBit.Fragment, FRAGMENT_SHADER),
            ):
                if not self.program.addShaderFromSourceCode(
                    shader_type, source
                ):
                    raise RuntimeError(self.program.log())
            if not self.program.link():
                raise RuntimeError(self.program.log())
            self.buffer = QOpenGLBuffer()
            self.color_buffer = QOpenGLBuffer()
            self.visibility_buffer = QOpenGLBuffer()
            self.preview_buffer = QOpenGLBuffer()
            self.picking_buffer = QOpenGLBuffer()
            for buffer in (
                self.buffer,
                self.color_buffer,
                self.visibility_buffer,
                self.preview_buffer,
                self.picking_buffer,
            ):
                if not buffer.create():
                    raise RuntimeError(
                        "Could not allocate the point-cloud buffer."
                    )
                buffer.setUsagePattern(QOpenGLBuffer.UsagePattern.DynamicDraw)
            self.context().aboutToBeDestroyed.connect(self.cleanup)
            self.scene_cache_supported = (
                QOpenGLFramebufferObject.hasOpenGLFramebufferBlit()
            )
        except (RuntimeError, MemoryError) as error:
            self.owner._renderer_failed(str(error))

    def cleanup(self):
        self.makeCurrent()
        for buffer in (
            self.buffer,
            self.color_buffer,
            self.visibility_buffer,
            self.preview_buffer,
            self.picking_buffer,
        ):
            if buffer is not None and buffer.isCreated():
                buffer.destroy()
        self.program = None
        self.functions = None
        self.picking_framebuffer = None
        self.scene_framebuffer = None
        self.scene_key = None
        self.scene_cache_supported = False
        self.data_dirty = True
        self.color_dirty = True
        self.color_indices = None
        self.visibility_dirty = True
        self.preview_dirty = True
        self.picking_dirty = True
        self.owner._selection_cache = None
        self.doneCurrent()

    def showEvent(self, event):
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._check_context)

    def _check_context(self):
        if self.isVisible() and not self.isValid():
            self.owner._renderer_failed(
                self.tr("Could not create an OpenGL display context.")
            )

    def paintGL(self):
        if self.program is None or self.owner._error:
            return
        try:
            self._draw_scene()
        except (RuntimeError, MemoryError) as error:
            self.owner._renderer_failed(str(error))
            return

    def _draw_scene(self):
        if not self.scene_cache_supported:
            self._draw_points()
            return
        size = QtCore.QSize(*self.owner._physical_size())
        key = (self.owner._matrix().tobytes(), size, self.owner._pixel_size())
        if (
            self.scene_framebuffer is None
            or self.scene_framebuffer.size() != size
        ):
            surface_format = QOpenGLFramebufferObjectFormat()
            surface_format.setAttachment(
                QOpenGLFramebufferObject.Attachment.CombinedDepthStencil
            )
            surface_format.setInternalTextureFormat(
                self.textureFormat() or 0x8058
            )
            self.scene_framebuffer = QOpenGLFramebufferObject(
                size, surface_format
            )
            self.scene_key = None
        framebuffer = self.scene_framebuffer
        if not framebuffer.isValid() or not framebuffer.bind():
            self._draw_scene_fallback()
            return
        try:
            if self.scene_key != key:
                self._draw_points(preview=False)
                self.scene_key = key
        finally:
            framebuffer.release()
        rectangle = QtCore.QRect(QtCore.QPoint(), size)
        QOpenGLFramebufferObject.blitFramebuffer(
            None, rectangle, framebuffer, rectangle, 0x00004000 | 0x00000100
        )
        if self.functions.glGetError() != 0:
            self._draw_scene_fallback()
            return
        self._draw_preview()

    def _draw_scene_fallback(self):
        self.scene_cache_supported = False
        self.scene_framebuffer = None
        self.scene_key = None
        QOpenGLFramebufferObject.bindDefault()
        for _ in range(8):
            if self.functions.glGetError() == 0:
                break
        self._draw_points()

    def _prepare_program(self):
        gl = self.functions
        gl.glViewport(0, 0, *self.owner._physical_size())
        gl.glEnable(0x0B71)
        gl.glDepthFunc(0x0203)
        gl.glDepthMask(True)
        gl.glDisable(0x0BE2)
        gl.glDisable(0x0BD0)
        gl.glEnable(0x8642)
        self.program.bind()
        matrix = QtGui.QMatrix4x4(self.owner._matrix().ravel().tolist())
        self.program.setUniformValue("matrix", matrix)
        self.program.setUniformValue(
            "viewport", QtGui.QVector2D(*self.owner._physical_size())
        )
        self.program.setUniformValue(
            "pointSize", float(self.owner._pixel_size())
        )

    def _draw_points(self, picking=False, preview=True):
        gl = self.functions
        self._prepare_program()
        gl.glClearColor(
            *((0, 0, 0, 0) if picking else (0.075, 0.09, 0.12, 1.0))
        )
        gl.glClear(0x00004000 | 0x00000100)
        if not len(self.owner._points):
            self.program.release()
            return
        if self.data_dirty:
            self._upload_buffer(
                self.buffer,
                np.ascontiguousarray(self.owner._points[:, :3]),
            )
            self.data_dirty = False
        if self.visibility_dirty:
            self._upload_buffer(
                self.visibility_buffer,
                self.owner._visible.astype(np.float32),
            )
            self.visibility_dirty = False
        if picking and self.picking_dirty:
            self._upload_buffer(
                self.picking_buffer,
                np.arange(1, len(self.owner._points) + 1, dtype="<u4"),
            )
            self.picking_dirty = False
        elif not picking and self.color_dirty:
            self._upload_colors()
        for name, buffer, value_type, components, stride in (
            ("position", self.buffer, 0x1406, 3, 12),
            (
                "color",
                self.picking_buffer if picking else self.color_buffer,
                0x1401 if picking else 0x1406,
                4,
                4 if picking else 16,
            ),
            ("visible", self.visibility_buffer, 0x1406, 1, 4),
        ):
            buffer.bind()
            self.program.enableAttributeArray(name)
            self.program.setAttributeBuffer(
                name, value_type, 0, components, stride
            )
        gl.glDrawArrays(0x0000, 0, len(self.owner._points))
        for name in ("position", "color", "visible"):
            self.program.disableAttributeArray(name)
        buffer.release()
        self.program.release()
        if preview and not picking:
            self._draw_preview()

    def _draw_preview(self):
        if not len(self.owner._preview):
            return
        self._prepare_program()
        self.preview_buffer.bind()
        if self.preview_dirty:
            data = np.ascontiguousarray(
                self.owner._points[self.owner._preview, :3],
                dtype=np.float32,
            )
            self._upload_buffer(self.preview_buffer, data)
            self.preview_count = len(data)
            self.preview_dirty = False
        self.program.disableAttributeArray("color")
        self.program.disableAttributeArray("visible")
        self.program.setAttributeValue(
            "color", QtGui.QVector4D(1, 0.85, 0.15, 1)
        )
        self.program.setAttributeValue("visible", 1.0)
        self.program.enableAttributeArray("position")
        self.program.setAttributeBuffer("position", 0x1406, 0, 3, 12)
        self.functions.glDrawArrays(0x0000, 0, self.preview_count)
        self.program.disableAttributeArray("position")
        self.preview_buffer.release()
        self.program.release()

    def _upload_buffer(self, buffer, data):
        buffer.bind()
        buffer.allocate(memoryview(data), data.nbytes)
        if self.functions.glGetError() != 0:
            raise RuntimeError("Could not upload the point-cloud buffer.")

    def _upload_colors(self):
        colors = self.owner._colors
        indices = self.color_indices
        blocks = np.unique(indices // 1024) if indices is not None else None
        if blocks is None or len(blocks) * 1024 >= len(colors) // 2:
            self._upload_buffer(self.color_buffer, colors)
        else:
            self.color_buffer.bind()
            boundaries = np.flatnonzero(np.diff(blocks) > 1) + 1
            for group in np.split(blocks, boundaries):
                if not len(group):
                    continue
                start = int(group[0]) * 1024
                end = min((int(group[-1]) + 1) * 1024, len(colors))
                data = colors[start:end]
                self.color_buffer.write(
                    start * 16, memoryview(data), data.nbytes
                )
            if self.functions.glGetError() != 0:
                raise RuntimeError("Could not upload the point-cloud buffer.")
        self.color_dirty = False
        self.color_indices = None

    def capture_surface(self):
        if not self.isValid() or self.program is None:
            raise RuntimeError(
                "The OpenGL viewport is not ready for selection."
            )
        self.makeCurrent()
        size = QtCore.QSize(*self.owner._physical_size())
        if (
            self.picking_framebuffer is None
            or self.picking_framebuffer.size() != size
        ):
            surface_format = QOpenGLFramebufferObjectFormat()
            surface_format.setAttachment(
                QOpenGLFramebufferObject.Attachment.CombinedDepthStencil
            )
            surface_format.setInternalTextureFormat(0x8058)
            self.picking_framebuffer = QOpenGLFramebufferObject(
                size, surface_format
            )
        framebuffer = self.picking_framebuffer
        try:
            if not framebuffer.isValid() or not framebuffer.bind():
                raise RuntimeError(
                    "Could not allocate the selection framebuffer."
                )
            self._draw_points(picking=True)
            image = framebuffer.toImage()
            if (
                image.format()
                == QtGui.QImage.Format.Format_ARGB32_Premultiplied
            ):
                image.reinterpretAsFormat(QtGui.QImage.Format.Format_ARGB32)
            elif (
                image.format()
                == QtGui.QImage.Format.Format_RGBA8888_Premultiplied
            ):
                image.reinterpretAsFormat(QtGui.QImage.Format.Format_RGBA8888)
            image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
            data = image.constBits()
            data.setsize(image.sizeInBytes())
            owners = (
                np.frombuffer(data, dtype="<u4")
                .reshape(image.height(), image.width())
                .copy()
            )
            if owners.max(initial=0) > len(self.owner._points):
                raise RuntimeError(
                    "Invalid point IDs from the selection framebuffer."
                )
            return owners
        finally:
            framebuffer.release()
            self.makeCurrent()
            self.doneCurrent()

    def mousePressEvent(self, event):
        self.owner._mouse_press(event)

    def mouseMoveEvent(self, event):
        self.owner._mouse_move(event)

    def mouseReleaseEvent(self, event):
        self.owner._mouse_release(event)

    def mouseDoubleClickEvent(self, event):
        self.owner._mouse_double_click(event)

    def wheelEvent(self, event):
        self.owner._wheel(event)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.owner.cancel_selection()
        elif event.key() in (
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.Key.Key_Enter,
        ):
            self.owner.finish_polygon()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        self.owner.cancel_selection()
        super().resizeEvent(event)
        self.overlay.setGeometry(self.rect())


class PointCloudViewport(QtWidgets.QWidget):
    camera_view_changed = QtCore.pyqtSignal(str)
    selection_completed = QtCore.pyqtSignal(object)
    selection_started = QtCore.pyqtSignal()
    selection_cancelled = QtCore.pyqtSignal()
    status_message = QtCore.pyqtSignal(str)
    renderer_error = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points = np.empty((0, 4), dtype=np.float32)
        self._colors = np.empty((0, 4), dtype=np.float32)
        self._visible = np.empty(0, dtype=bool)
        self._visible_count = 0
        self._preview = np.empty(0, dtype=np.int64)
        self._snapshot = None
        self._selection_cache = None
        self._selection_cache_key = None
        self._stroke = []
        self._polygon = []
        self._cursor = None
        self._last_mouse = None
        self._drag_button = None
        self._tool = "browse"
        self._depth_mode = "surface"
        self._point_size = 2.0
        self._brush_radius = 20.0
        self._center = np.zeros(3, dtype=np.float32)
        self._scale = 10.0
        self._extent = 10.0
        self._yaw = -45.0
        self._pitch = 35.0
        self._error = None
        self.setMinimumSize(240, 180)
        layout = QtWidgets.QStackedLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._error_label = QtWidgets.QLabel(self)
        self._error_label.setWordWrap(True)
        self._error_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._error_label.setMargin(24)
        self._gl = None
        if QtGui.QGuiApplication.platformName() not in (
            "offscreen",
            "minimal",
        ):
            self._gl = _PointCloudGLWidget(self)
            layout.addWidget(self._gl)
            self.setFocusProxy(self._gl)
        layout.addWidget(self._error_label)
        if self._gl is None:
            QtCore.QTimer.singleShot(
                0,
                lambda: self._renderer_failed(
                    self.tr("The current Qt platform does not support OpenGL.")
                ),
            )

    @property
    def selection_active(self):
        return self._snapshot is not None

    def set_cloud(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] not in (3, 4):
            raise ValueError(
                "Point coordinates must have shape (N, 3) or (N, 4)."
            )
        if not np.isfinite(points[:, :3]).all():
            raise ValueError("Point coordinates must be finite.")
        self.cancel_selection()
        self._points = points
        self._colors = np.full((len(points), 4), 0.7, dtype=np.float32)
        self._colors[:, 3] = 1
        self._visible = np.ones(len(points), dtype=bool)
        self._visible_count = len(points)
        self.reset_view()
        self._update(data=True)

    def validate_rendering(self):
        if self._gl is not None and self.isVisible():
            self._gl.grabFramebuffer()
            if self._error:
                raise RuntimeError(self._error)

    def _point_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(
            (indices < 0) | (indices >= len(self._points))
        ):
            raise ValueError("Point indices must be within the point cloud.")
        return indices

    def set_colors(self, colors, indices=None):
        if indices is not None:
            indices = self._point_indices(indices)
        colors = np.asarray(colors)
        count = len(self._points) if indices is None else len(indices)
        if colors.shape not in (
            (count, 3),
            (count, 4),
        ):
            raise ValueError("Colors must have shape (N, 3) or (N, 4).")
        normalized = colors.astype(np.float32)
        if np.issubdtype(colors.dtype, np.integer):
            normalized /= 255.0
        if not np.isfinite(normalized).all():
            raise ValueError("Colors must be finite.")
        target = slice(None) if indices is None else indices
        self._colors[target, :3] = np.clip(normalized[:, :3], 0, 1)
        self._colors[target, 3] = (
            np.clip(normalized[:, 3], 0, 1) if normalized.shape[1] == 4 else 1
        )
        if count:
            self._update(colors=True, indices=indices)

    def set_visible_mask(self, mask, indices=None):
        if indices is not None:
            indices = self._point_indices(indices)
        mask = np.asarray(mask, dtype=bool)
        count = len(self._points) if indices is None else len(indices)
        if mask.shape != (count,):
            raise ValueError("Visibility must have one entry per point.")
        self.cancel_selection()
        target = slice(None) if indices is None else indices
        if np.array_equal(self._visible[target], mask):
            return
        self._visible[target] = mask
        self._visible_count = np.count_nonzero(self._visible)
        self._update(visible=True)

    def set_tool(self, tool):
        if tool not in ("browse", "brush", "polygon"):
            raise ValueError("Unknown point-cloud tool")
        self.cancel_selection()
        self._tool = tool
        if self._gl:
            self._gl.setCursor(
                QtCore.Qt.CursorShape.ArrowCursor
                if tool == "browse"
                else QtCore.Qt.CursorShape.CrossCursor
            )
        self._update()

    def set_depth_mode(self, mode):
        if mode not in ("surface", "through"):
            raise ValueError("Unknown depth selection mode")
        self.cancel_selection()
        self._depth_mode = mode

    def set_point_size(self, size):
        self.cancel_selection()
        self._point_size = min(10.0, max(1.0, float(size)))
        self._update()

    def set_brush_radius(self, radius):
        self.cancel_selection()
        self._brush_radius = min(100.0, max(2.0, float(radius)))
        self._update()

    def fit_all(self):
        self.cancel_selection()
        self._fit_indices(None)

    def reset_view(self):
        self._yaw, self._pitch = -45.0, 35.0
        self.fit_all()
        self.camera_view_changed.emit("")

    def set_view(self, view):
        directions = {
            "top": (0.0, 90.0),
            "front": (0.0, 0.0),
            "side": (90.0, 0.0),
        }
        if view not in directions:
            raise ValueError("Unknown point-cloud view")
        self.cancel_selection()
        self._yaw, self._pitch = directions[view]
        self._update()
        self.camera_view_changed.emit(view)

    def focus_indices(self, indices):
        self.cancel_selection()
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            self.status_message.emit(self.tr("There are no points to focus."))
            return
        self._fit_indices(indices)
        self._preview = indices[self._visible[indices]]
        self._update(preview=True)

    def _fit_indices(self, indices):
        if len(self._points) and (indices is None or len(indices)):
            points = (
                self._points[:, :3]
                if indices is None
                else self._points[indices, :3]
            )
            low, high = points.min(axis=0), points.max(axis=0)
            self._center = (low.astype(np.float64) + high) / 2
            radius = max(
                float(np.linalg.norm(high.astype(np.float64) - low) / 2), 0.1
            )
            aspect = max(self.width(), 1) / max(self.height(), 1)
            self._scale = radius * 1.1 / min(1, aspect)
            self._extent = (
                max(
                    float(np.max(np.abs(self._points[:, :3] - self._center))),
                    radius,
                )
                * 4
            )
        self._update()

    def _basis(self):
        yaw, pitch = math.radians(self._yaw), math.radians(self._pitch)
        right = np.array((math.cos(yaw), math.sin(yaw), 0))
        forward = np.array(
            (
                -math.sin(yaw) * math.cos(pitch),
                math.cos(yaw) * math.cos(pitch),
                -math.sin(pitch),
            )
        )
        up = np.cross(right, forward)
        return right, up, forward

    def _matrix(self):
        right, up, forward = self._basis()
        width, height = self._physical_size()
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, :3] = right / (self._scale * width / height)
        matrix[1, :3] = up / self._scale
        matrix[2, :3] = forward / max(self._extent, 0.1)
        matrix[:3, 3] = -matrix[:3, :3] @ self._center
        return matrix

    def _physical_size(self):
        ratio = self.devicePixelRatioF()
        return max(1, round(self.width() * ratio)), max(
            1, round(self.height() * ratio)
        )

    def _pixel_size(self):
        return max(1, round(self._point_size * self.devicePixelRatioF()))

    def _begin_selection(self):
        if not len(self._points) or self._error:
            return False
        width, height = self._physical_size()
        try:
            matrix = self._matrix()
            key = (
                matrix.tobytes(),
                width,
                height,
                self._pixel_size(),
                self._depth_mode,
            )
            if (
                self._selection_cache is None
                or self._selection_cache_key != key
            ):
                screen, depths, in_view = project_points(
                    self._points, matrix, width, height
                )
                owners = (
                    self._gl.capture_surface()
                    if self._gl is not None and self._depth_mode == "surface"
                    else None
                )
                self._selection_cache = SelectionSnapshot.create(
                    screen,
                    depths,
                    self._visible & in_view,
                    width,
                    height,
                    self._pixel_size(),
                    self._depth_mode,
                    surface_owners=owners,
                )
                self._selection_cache_key = key
            self._snapshot = self._selection_cache
        except (MemoryError, RuntimeError) as error:
            self.status_message.emit(
                self.tr("Could not start selection: {error}").format(
                    error=error
                )
            )
            return False
        had_preview = bool(len(self._preview))
        self._preview = np.empty(0, dtype=np.int64)
        self._update(preview=had_preview, scene=had_preview)
        self.selection_started.emit()
        return True

    def _brush_segment(self, start, end):
        ratio = self.devicePixelRatioF()
        indices = select_brush(
            self._snapshot,
            np.array(start) * ratio,
            np.array(end) * ratio,
            self._brush_radius * ratio,
        )
        preview = np.union1d(self._preview, indices)
        changed = len(preview) != len(self._preview)
        self._preview = preview
        self.status_message.emit(
            self.tr("Selected points: {count}").format(
                count=len(self._preview)
            )
        )
        self._update(preview=changed, scene=changed)

    def cancel_selection(self):
        active = self.selection_active
        had_preview = bool(len(self._preview))
        self._snapshot = None
        self._stroke.clear()
        self._polygon.clear()
        self._preview = np.empty(0, dtype=np.int64)
        self._drag_button = None
        if active:
            self.status_message.emit(
                self.tr("Unfinished selection cancelled.")
            )
            self.selection_cancelled.emit()
        self._update(preview=had_preview, scene=had_preview)

    def _complete_selection(self):
        selected = self._preview.copy()
        self._snapshot = None
        self._stroke.clear()
        self._polygon.clear()
        self._preview = np.empty(0, dtype=np.int64)
        self._drag_button = None
        self._update(preview=True)
        self.selection_completed.emit(selected)

    def finish_polygon(self):
        if self._tool != "polygon" or self._snapshot is None:
            return
        if not self._begin_selection():
            return
        try:
            self._preview = select_polygon(
                self._snapshot,
                np.asarray(self._polygon) * self.devicePixelRatioF(),
            )
        except ValueError as error:
            self.status_message.emit(str(error))
            return
        self._complete_selection()

    def _mouse_press(self, event):
        point = (event.position().x(), event.position().y())
        self._last_mouse = point
        self._cursor = point
        button = event.button()
        if self._gl:
            self._gl.setFocus()
        if (
            self._tool == "polygon"
            and button == QtCore.Qt.MouseButton.LeftButton
            and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self._drag_button = button
            return
        if (
            button != QtCore.Qt.MouseButton.LeftButton
            or self._tool == "browse"
        ):
            self.cancel_selection()
            self._drag_button = button
            return
        if self._snapshot is None and not self._begin_selection():
            return
        if self._tool == "brush":
            self._stroke = [point]
            self._brush_segment(point, point)
        else:
            if self._near_polygon_start(point):
                self.finish_polygon()
                return
            if not self._polygon or point != self._polygon[-1]:
                self._polygon.append(point)
            self._update()

    def _near_polygon_start(self, point):
        return (
            len(self._polygon) >= 3
            and math.hypot(
                point[0] - self._polygon[0][0], point[1] - self._polygon[0][1]
            )
            <= 10
        )

    def _mouse_double_click(self, event):
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self._tool == "polygon"
            and not event.modifiers()
            & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self.finish_polygon()
            event.accept()

    def _mouse_move(self, event):
        point = (event.position().x(), event.position().y())
        self._cursor = point
        if self._drag_button is not None and self._last_mouse is not None:
            dx, dy = np.array(point) - self._last_mouse
            if self._drag_button in (
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.MiddleButton,
            ):
                right, up, _ = self._basis()
                self._center += (
                    (up * dy - right * dx)
                    * 2
                    * self._scale
                    / max(self.height(), 1)
                )
                if self._tool == "polygon" and self._polygon:
                    self._polygon = [
                        (x + dx, y + dy) for x, y in self._polygon
                    ]
            elif dx or dy:
                self._yaw -= dx * 0.4
                self._pitch = min(90, max(-90, self._pitch + dy * 0.4))
                self.camera_view_changed.emit("")
        elif (
            self._tool == "brush"
            and self._snapshot is not None
            and self._stroke
            and point != self._stroke[-1]
        ):
            self._brush_segment(self._stroke[-1], point)
            self._stroke.append(point)
        self._last_mouse = point
        self._update(scene=self._drag_button is not None)

    def _mouse_release(self, event):
        self._drag_button = None
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self._tool == "brush"
            and self._snapshot is not None
        ):
            point = (event.position().x(), event.position().y())
            if point != self._stroke[-1]:
                self._brush_segment(self._stroke[-1], point)
            self._complete_selection()

    def _wheel(self, event):
        if (
            self._tool == "brush"
            and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self.set_brush_radius(
                min(
                    100,
                    max(
                        2,
                        self._brush_radius + event.angleDelta().y() / 120 * 2,
                    ),
                )
            )
            self.status_message.emit(
                self.tr("Brush radius: {radius} px").format(
                    radius=int(self._brush_radius)
                )
            )
            event.accept()
            return
        preserve_polygon = self._tool == "polygon" and bool(self._polygon)
        if not preserve_polygon:
            self.cancel_selection()
        old_scale = self._scale
        self._scale = min(
            1e12,
            max(1e-5, self._scale * math.exp(-event.angleDelta().y() / 900)),
        )
        if preserve_polygon:
            factor = old_scale / self._scale
            cx, cy = self.width() / 2, self.height() / 2
            self._polygon = [
                (cx + (x - cx) * factor, cy + (y - cy) * factor)
                for x, y in self._polygon
            ]
        self._update()
        event.accept()

    def _paint_overlay(self, painter):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor(245, 211, 75), 1.5))
        painter.setBrush(QtGui.QColor(245, 211, 75, 25))
        if self._tool == "brush" and self._cursor is not None:
            painter.drawEllipse(
                QtCore.QPointF(*self._cursor),
                self._brush_radius,
                self._brush_radius,
            )
        if self._polygon:
            vertices = [QtCore.QPointF(*point) for point in self._polygon]
            closing = self._cursor is not None and self._near_polygon_start(
                self._cursor
            )
            if self._cursor:
                vertices.append(
                    QtCore.QPointF(
                        *(self._polygon[0] if closing else self._cursor)
                    )
                )
            painter.drawPolygon(QtGui.QPolygonF(vertices))
            for point in self._polygon:
                painter.drawEllipse(QtCore.QPointF(*point), 3, 3)
            if closing:
                painter.setBrush(QtGui.QColor(245, 211, 75))
                for point in (self._polygon[0], self._polygon[-1]):
                    painter.drawEllipse(QtCore.QPointF(*point), 5, 5)
        origin = QtCore.QPointF(48, self.height() - 48)
        right, up, _ = self._basis()
        for i, (label, color) in enumerate(
            (("X", "#f07178"), ("Y", "#a5d674"), ("Z", "#79b8ff"))
        ):
            end = origin + QtCore.QPointF(right[i] * 30, -up[i] * 30)
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 2))
            painter.drawLine(origin, end)
            painter.drawText(end + QtCore.QPointF(3, -3), label)
        painter.setPen(QtGui.QColor("#cbd2df"))
        if not len(self._points):
            painter.drawText(
                self.rect(),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                self.tr("Open a BIN or PLY point cloud to begin."),
            )
        elif not self._visible_count:
            painter.drawText(
                self.rect(),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                self.tr(
                    "All points are hidden. Use Restore All to show them."
                ),
            )

    def _renderer_failed(self, message):
        if self._error:
            return
        self._error = message
        self.cancel_selection()
        self._error_label.setText(
            self.tr(
                "Point-cloud rendering is unavailable.\n{message}\nThe main application remains available."
            ).format(message=message)
        )
        self.layout().setCurrentWidget(self._error_label)
        self.renderer_error.emit(message)

    def _update(
        self,
        data=False,
        preview=False,
        colors=False,
        visible=False,
        indices=None,
        scene=True,
    ):
        if data or visible:
            self._selection_cache = None
        if self._gl is not None:
            if data or colors or visible:
                self._gl.scene_key = None
            self._gl.data_dirty |= data
            self._gl.picking_dirty |= data
            self._gl.visibility_dirty |= data or visible
            if data:
                self._gl.color_dirty = True
                self._gl.color_indices = None
            elif colors:
                if not self._gl.color_dirty:
                    self._gl.color_indices = (
                        indices.copy() if indices is not None else None
                    )
                elif self._gl.color_indices is not None:
                    self._gl.color_indices = (
                        np.union1d(self._gl.color_indices, indices)
                        if indices is not None
                        else None
                    )
                self._gl.color_dirty = True
            self._gl.preview_dirty |= preview
            if scene:
                self._gl.update()
            self._gl.overlay.update()
