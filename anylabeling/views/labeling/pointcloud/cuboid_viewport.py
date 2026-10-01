from dataclasses import replace
import math

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .cuboid import Cuboid, EDGES, FACES, MIN_SIZE, VIEW_AXES
from .icons import draw_rotation_handle, get_icon
from .viewport import PointCloudViewport

FOCUS_DECAY = -60 * math.log(0.95)


class CuboidViewport(PointCloudViewport):
    cuboid_selected = QtCore.pyqtSignal(object)
    cuboid_preview = QtCore.pyqtSignal(object)
    cuboid_edited = QtCore.pyqtSignal(object)
    cuboid_created = QtCore.pyqtSignal(object, object)
    cuboids_changed = QtCore.pyqtSignal()
    cancel_requested = QtCore.pyqtSignal()
    expand_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None, view=None):
        self.orthographic_view = view
        self.detection_enabled = False
        self.creating = False
        self.cuboids = ()
        self.selected_id = None
        self.class_colors = {}
        self.class_names = {}
        self._orientation = np.eye(3)
        self._gesture = None
        self._gesture_button = None
        self._hover_handle = None
        self._navigation_button = None
        self._navigation_start = None
        self._navigation_dragged = False
        self._navigation_select = False
        self._creation_start = None
        self._creation_end = None
        self._creation_preview = None
        self._creation_ray_cache = None
        self.expand_button = None
        self.camera_actions = {}
        self.camera_buttons = {}
        self._camera_panels = []
        self._expanded = False
        self._hovered = False
        super().__init__(parent)
        self._perspective = view is None
        self._show_axes = view is None
        self._show_hidden_message = view is None
        self._focus_animation = QtCore.QVariantAnimation(self)
        self._focus_animation.setDuration(250)
        self._focus_animation.setStartValue(0.0)
        self._focus_animation.setEndValue(1.0)
        self._focus_animation.setEasingCurve(
            QtCore.QEasingCurve.Type.Linear
            if view is None
            else QtCore.QEasingCurve.Type.InOutCubic
        )
        self._focus_animation.valueChanged.connect(self._advance_focus)
        self._scale_limits = (
            (0.5 / 20, 0.5 / 0.01)
            if view is not None
            else (
                0.3 * math.tan(math.radians(25)),
                100 * math.tan(math.radians(25)),
            )
        )
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.installEventFilter(self)
        if self._gl is not None:
            self._gl.installEventFilter(self)
        if view is not None:
            self.expand_button = QtWidgets.QToolButton(self)
            self.expand_button.setObjectName("pointcloudExpandButton")
            self.expand_button.setFixedSize(24, 24)
            self.expand_button.setIconSize(QtCore.QSize(18, 18))
            self.expand_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            self.expand_button.clicked.connect(
                lambda checked=False: self.expand_requested.emit()
            )
            self.set_expanded(False)
        else:
            self._build_camera_controls()

    def _build_camera_controls(self):
        controls = (
            (
                ("U", self.tr("Move camera up"), 0, 0),
                ("I", self.tr("Zoom camera in"), 0, 1),
                ("O", self.tr("Move camera down"), 0, 2),
                ("J", self.tr("Move camera left"), 1, 0),
                ("K", self.tr("Zoom camera out"), 1, 1),
                ("L", self.tr("Move camera right"), 1, 2),
            ),
            (
                ("Up", self.tr("Tilt camera up"), 0, 1),
                ("Left", self.tr("Rotate camera left"), 1, 0),
                ("Down", self.tr("Tilt camera down"), 1, 1),
                ("Right", self.tr("Rotate camera right"), 1, 2),
            ),
        )
        for group in controls:
            panel = QtWidgets.QWidget(self)
            panel.setObjectName("pointcloudCameraControls")
            layout = QtWidgets.QGridLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            for key, title, row, column in group:
                action = QtGui.QAction(title, self)
                action.setShortcut(
                    ("Alt+" if len(key) == 1 else "Shift+") + key
                )
                action.setShortcutContext(
                    QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
                )
                action.setToolTip(f"{title} ({action.shortcut().toString()})")
                action.triggered.connect(
                    lambda checked=False, key=key: self.move_camera(key)
                )
                self.addAction(action)
                self.camera_actions[key] = action
                button = QtWidgets.QToolButton(panel)
                button.setObjectName("pointcloudCameraButton")
                button.setDefaultAction(action)
                button.setAccessibleName(title)
                button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
                button.setFixedSize(32, 24)
                if len(key) == 1:
                    button.setText(key)
                else:
                    button.setIcon(
                        get_icon(f"camera-{key.lower()}", "#1d1d1f", "#1d1d1f")
                    )
                    button.setIconSize(QtCore.QSize(18, 18))
                layout.addWidget(button, row, column)
                self.camera_buttons[key] = button
            panel.setFixedSize(layout.sizeHint())
            panel.hide()
            self._camera_panels.append(panel)
        self._position_camera_controls()

    def set_camera_controls_visible(self, visible):
        for panel in self._camera_panels:
            panel.setVisible(visible)
        self._position_camera_controls()

    def _position_camera_controls(self):
        for index, panel in enumerate(self._camera_panels):
            x = 12 if index == 0 else self.width() - panel.width() - 12
            panel.move(x, self.height() - panel.height() - 12)
            panel.raise_()

    def move_camera(self, key):
        if not len(self._points) or self._error or self.selection_active:
            return
        start = (self._center.copy(), self._scale, (self._yaw, self._pitch))
        target = (
            self._focus_target
            if self._focus_animation.state()
            == QtCore.QAbstractAnimation.State.Running
            else start
        )
        right, up, _ = self._basis()
        self.cancel_selection()
        center, scale, angles = target
        self._center, self._scale = center.copy(), scale
        self._yaw, self._pitch = angles
        if key in ("U", "O", "J", "L"):
            direction = {"U": -up, "O": up, "J": -right, "L": right}[key]
            self._center += direction * 2
        elif key in ("I", "K"):
            distance = -5 if key == "I" else 5
            self._set_scale(
                self._scale + distance * math.tan(math.radians(25))
            )
        elif key in ("Left", "Right"):
            self._yaw += -20 if key == "Left" else 20
        else:
            limit = 90 - math.degrees(1e-6)
            self._pitch = min(
                limit,
                max(-limit, self._pitch + (10 if key == "Up" else -10)),
            )
        self._animate_camera(start, shortest_rotation=False)
        self.camera_view_changed.emit("")
        self.setFocus()

    def set_expanded(self, expanded):
        self._expanded = expanded
        if self.expand_button is None:
            return
        title = self.tr("Restore view") if expanded else self.tr("Expand view")
        self.expand_button.setToolTip(title)
        self.expand_button.setAccessibleName(title)
        self.expand_button.setIcon(
            get_icon(
                "restore-view" if expanded else "expand-view",
                "#cbd2df",
                "#cbd2df",
            )
        )
        self.expand_button.setVisible(expanded or self._hovered)
        self._position_expand_button()

    def _position_expand_button(self):
        if self.expand_button is not None:
            self.expand_button.move(self.width() - 36, 6)
            self.expand_button.raise_()

    def enterEvent(self, event):
        super().enterEvent(event)
        self._hovered = True
        if self.expand_button is not None:
            self.expand_button.show()
            self.expand_button.raise_()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._hovered = False
        if self.expand_button is not None:
            self.expand_button.setVisible(self._expanded)

    @property
    def selection_active(self):
        return (
            super().selection_active
            or self._gesture is not None
            or self._creation_start is not None
        )

    @property
    def selected_cuboid(self):
        return next(
            (box for box in self.cuboids if box.id == self.selected_id), None
        )

    def set_cuboids(self, cuboids, selected_id):
        previous = self.selected_cuboid
        self.cuboids = tuple(
            box
            for box in cuboids
            if self.orthographic_view is None or box.id == selected_id
        )
        self.selected_id = selected_id
        selected = self.selected_cuboid
        if self._gesture is None:
            self._update_handle_hover(None)
        if (
            self.orthographic_view is not None
            and previous is not None
            and selected is not None
            and previous.id == selected.id
            and previous.rotation != selected.rotation
        ):
            self._focus_animation.stop()
            orientation = selected.matrix
            self._center = np.asarray(selected.center) + orientation @ (
                self._orientation.T @ (self._center - selected.center)
            )
            self._orientation = orientation
        self._update_cuboid_mesh()

        self.cuboids_changed.emit()

    def _update_cuboid_mesh(self):
        faces, edges = [], []
        boxes = self.cuboids
        if self._creation_preview is not None:
            boxes += (self._creation_preview,)
        for box in boxes:
            preview = box is self._creation_preview
            corners = box.corners()
            color = QtGui.QColor(
                "#ffd54f"
                if preview
                else self.class_colors.get(box.class_id, "#64b5f6")
            )
            rgb = (color.redF(), color.greenF(), color.blueF())
            if preview or box.id == self.selected_id:
                for face in FACES:
                    vertices = corners[list(face)]
                    normal = np.cross(
                        vertices[1] - vertices[0], vertices[2] - vertices[0]
                    )
                    if normal @ (vertices.mean(axis=0) - box.center) < 0:
                        vertices = vertices[::-1]
                    faces.extend(
                        [
                            (*vertices[i], *rgb, 0.4 if preview else 0.3)
                            for i in (0, 1, 2, 0, 2, 3)
                        ]
                    )
            outline = (
                (1, 0.93, 0.65)
                if preview
                else (1, 1, 1) if self.orthographic_view else rgb
            )
            for a, b in EDGES:
                vertices = corners[[a, b]]
                if box.occluded and self.orthographic_view is None:
                    direction = vertices[1] - vertices[0]
                    length = np.linalg.norm(direction)
                    starts = np.arange(0, length, 0.1)
                    steps = np.column_stack(
                        (starts, np.minimum(starts + 0.05, length))
                    ).ravel()
                    vertices = (
                        vertices[0] + steps[:, None] * direction / length
                    )
                edges.extend([(*point, *outline, 1) for point in vertices])
        self._cuboid_mesh = (
            np.asarray(faces + edges, dtype=np.float32).reshape(-1, 7),
            len(faces),
        )
        if self._gl is not None:
            self._gl.cuboid_dirty = True
        self._update()

    def _set_creation_preview(self, box):
        if box != self._creation_preview:
            self._creation_preview = box
            self._update_cuboid_mesh()
            self.cuboids_changed.emit()

    def _update_creation_preview(self, point):
        if (
            not self.creating
            or self.orthographic_view is not None
            or not len(self._points)
            or self._error
            or not QtCore.QRectF(self.rect()).contains(QtCore.QPointF(*point))
        ):
            self._set_creation_preview(None)
            return
        distance = self._scale / math.tan(math.radians(25))
        origin = self._center - self._basis()[2] * distance
        direction = self.unproject(point) - origin
        direction /= np.linalg.norm(direction)
        key = self._matrix().tobytes()
        if (
            self._creation_ray_cache is None
            or self._creation_ray_cache[0] != key
        ):
            offsets = self._points[:, :3] - origin
            squared = np.einsum("ij,ij->i", offsets, offsets)
            self._creation_ray_cache = (key, offsets, squared)
        _, offsets, squared = self._creation_ray_cache
        depths = np.einsum("ij,j->i", offsets, direction)
        valid = self._visible & (depths >= 0.1) & (squared - depths**2 < 1.0)
        if not valid.any():
            return
        depth = np.min(depths, where=valid, initial=np.inf)
        self._set_creation_preview(
            Cuboid(1, 1, tuple(origin + direction * depth), (1, 1, 1))
        )

    def align_cuboid(self, box, fit=False, animate=False):
        self._focus_animation.stop()
        start = (self._center.copy(), self._scale, (self._yaw, self._pitch))
        if self.orthographic_view is None:
            if fit and box is not None:
                self._center = np.array(box.center)
                right, up, _ = self._basis()
                offsets = box.corners() - box.center
                aspect = max(self.width(), 1) / max(self.height(), 1)
                self._set_scale(
                    max(np.ptp(offsets @ up), np.ptp(offsets @ right) / aspect)
                    * 0.75
                )
        else:
            self._orientation = np.eye(3) if box is None else box.matrix
        if box is not None and self.orthographic_view is not None:
            self._center = np.array(box.center)
            self._extent = max(self._extent, max(box.size) * 2)
            if fit:
                horizontal, vertical = VIEW_AXES[self.orthographic_view]
                width, height = max(self.width(), 1), max(self.height(), 1)
                padding = min(50, min(width, height) / 4)
                self._set_scale(
                    max(
                        box.size[vertical] * height / (height - 2 * padding),
                        box.size[horizontal] * height / (width - 2 * padding),
                    )
                    / 2
                )
        if animate and box is not None:
            self._animate_camera(start)
        self._update()

    def _animate_camera(self, start, shortest_rotation=True):
        self._focus_start = start
        self._focus_target = (
            self._center.copy(),
            self._scale,
            (self._yaw, self._pitch),
        )
        self._focus_yaw_delta = self._yaw - start[2][0]
        if shortest_rotation:
            self._focus_yaw_delta = (self._focus_yaw_delta + 180) % 360 - 180
        if self.orthographic_view is None:
            center, scale, angles = start
            delta = max(
                np.max(np.abs(self._center - center)),
                abs(self._scale - scale) / math.tan(math.radians(25)),
                abs(math.radians(self._focus_yaw_delta)),
                abs(math.radians(self._pitch - angles[1])),
                1e-5,
            )
            self._focus_animation.setDuration(
                max(1, math.ceil(math.log(delta / 1e-5) / FOCUS_DECAY * 1000))
            )
        self._center, self._scale, angles = start
        self._yaw, self._pitch = angles
        self._focus_animation.start()

    def _advance_focus(self, progress):
        if self.orthographic_view is None and progress < 1:
            progress = -math.expm1(
                -FOCUS_DECAY * self._focus_animation.currentTime() / 1000
            )
        center, scale, angles = self._focus_start
        target_center, target_scale, target_angles = self._focus_target
        self._center = center + (target_center - center) * progress
        self._set_scale(scale + (target_scale - scale) * progress)
        self._yaw = angles[0] + self._focus_yaw_delta * progress
        self._pitch = angles[1] + (target_angles[1] - angles[1]) * progress
        if progress == 1:
            self._yaw, self._pitch = target_angles
        self._update()
        if self.creating and self._cursor is not None:
            self._update_creation_preview(self._cursor)

    def reset_view(self, animate=False):
        start = (self._center.copy(), self._scale, (self._yaw, self._pitch))
        super().reset_view()
        if animate:
            self._animate_camera(start)

    def hideEvent(self, event):
        self._focus_animation.stop()
        self._hovered = False
        if self.expand_button is not None:
            self.expand_button.setVisible(self._expanded)
        super().hideEvent(event)

    def _basis(self):
        if self.orthographic_view is None:
            return super()._basis()
        horizontal, vertical = VIEW_AXES[self.orthographic_view]
        right, up = (
            self._orientation[:, horizontal],
            self._orientation[:, vertical],
        )
        return right, up, np.cross(up, right)

    def resizeEvent(self, event):
        self.cancel_selection()
        super().resizeEvent(event)
        self._position_expand_button()
        self._position_camera_controls()
        if self.orthographic_view and self.selected_cuboid:
            self.align_cuboid(self.selected_cuboid, fit=True)

    def project(self, points):
        matrix = self._matrix().astype(np.float64)
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        clip = points @ matrix[:3, :3].T + matrix[:3, 3]
        depth = points @ matrix[3, :3] + matrix[3, 3]
        ndc = np.full((len(points), 2), np.nan)
        np.divide(
            clip[:, :2], depth[:, None], out=ndc, where=(depth >= 0.1)[:, None]
        )
        return np.column_stack(
            (
                (ndc[:, 0] + 1) * self.width() / 2,
                (1 - ndc[:, 1]) * self.height() / 2,
            )
        )

    def _pixel_size(self):
        if self.orthographic_view is not None:
            return max(
                1,
                round(0.5 * (self._point_size / 2) * self.devicePixelRatioF()),
            )
        return super()._pixel_size()

    def unproject(self, point):
        right, up, _ = self._basis()
        factor = 2 * self._scale / max(self.height(), 1)
        return self._center + factor * (
            (point[0] - self.width() / 2) * right
            - (point[1] - self.height() / 2) * up
        )

    def _handles(self, box):
        if self.orthographic_view is None or box.locked:
            return [], None
        axes = VIEW_AXES[self.orthographic_view]
        handles = []
        for signs in (
            (-1, -1),
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
        ):
            local = np.zeros(3)
            local[list(axes)] = (
                np.array(signs) * np.array(box.size)[list(axes)] / 2
            )
            point = self.project([box.center + box.matrix @ local])[0]
            handles.append((signs, point))
        top = handles[5][1]
        center = self.project([box.center])[0]
        direction = top - center
        length = np.linalg.norm(direction)
        rotation = top + direction / max(length, 1e-6) * 28
        return handles, rotation

    def _hit_handle(self, point):
        box = self.selected_cuboid
        if (
            not self.detection_enabled
            or self.creating
            or box is None
            or point is None
            or not len(self._points)
            or self._error
        ):
            return None
        handles, rotation = self._handles(box)
        if not handles:
            return None
        signs, position = min(
            handles, key=lambda handle: np.linalg.norm(point - handle[1])
        )
        center_distance = np.linalg.norm(point - self.project([box.center])[0])
        if np.linalg.norm(point - position) < min(9, center_distance):
            return "resize", signs
        if rotation is not None and np.linalg.norm(point - rotation) < 10:
            return "rotate", None
        return None

    def _hit_box(self, point):
        forward = self._basis()[2]
        if self._perspective:
            distance = self._scale / math.tan(math.radians(25))
            origin = self._center - forward * distance
            direction = (self.unproject(point) - origin) / distance
            near = 0.1
        else:
            origin = self.unproject(point) - forward * self._extent
            direction = forward
            near = 0.0
        candidates = []
        for box in self.cuboids:
            local = (origin - box.center) @ box.matrix
            ray = direction @ box.matrix
            half = np.asarray(box.size) / 2
            parallel = np.abs(ray) < 1e-10
            if np.any(parallel & (np.abs(local) > half)):
                continue
            lower = np.full(3, -np.inf)
            upper = np.full(3, np.inf)
            np.divide(-half - local, ray, out=lower, where=~parallel)
            np.divide(half - local, ray, out=upper, where=~parallel)
            entry = max(float(np.minimum(lower, upper).max()), near)
            exit = float(np.maximum(lower, upper).min())
            if entry <= exit:
                candidates.append((entry, box.id))
        return min(candidates)[1] if candidates else None

    def cancel_selection(self):
        if hasattr(self, "_focus_animation"):
            self._focus_animation.stop()
        active = self._gesture is not None or self._creation_start is not None
        self._gesture = None
        self._gesture_button = None
        self._update_handle_hover(None)
        self._navigation_button = None
        self._navigation_start = None
        self._navigation_dragged = False
        self._navigation_select = False
        self._creation_start = None
        self._creation_end = None
        self._creation_ray_cache = None
        self._set_creation_preview(None)
        super().cancel_selection()
        if active:
            self.cuboid_preview.emit(None)
            self.selection_cancelled.emit()

    def _start_navigation(self, point, button, rotate=False, select=False):
        self.cancel_selection()
        self._navigation_button = button
        self._navigation_start = point
        self._navigation_select = select
        self._last_mouse = point
        self._cursor = point
        self._drag_button = (
            QtCore.Qt.MouseButton.LeftButton
            if rotate
            else QtCore.Qt.MouseButton.RightButton
        )

    def _mouse_press(self, event):
        if not self.detection_enabled or (
            self.orthographic_view is None
            and self._tool != "browse"
            and not self.creating
        ):
            return super()._mouse_press(event)
        if (
            self._gesture is not None
            or self._creation_start is not None
            or self._navigation_button is not None
        ):
            return
        button = event.button()
        if button not in (
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.RightButton,
            QtCore.Qt.MouseButton.MiddleButton,
        ):
            return
        if self._gl is not None:
            self._gl.setFocus()
        point = np.array([event.position().x(), event.position().y()])
        if self.creating and self.orthographic_view is None:
            self._update_creation_preview(point)
            return
        if button == QtCore.Qt.MouseButton.MiddleButton or (
            button == QtCore.Qt.MouseButton.LeftButton
            and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self._start_navigation(point, button)
            return
        if self.orthographic_view is None:
            self._start_navigation(
                point,
                button,
                rotate=button == QtCore.Qt.MouseButton.LeftButton,
                select=button == QtCore.Qt.MouseButton.LeftButton,
            )
            return
        if not len(self._points) or self._error:
            return
        if self.creating:
            if button != QtCore.Qt.MouseButton.LeftButton:
                return
            self._creation_start = point
            self._creation_end = point
            self.selection_started.emit()
            return
        box = self.selected_cuboid
        handle = self._hit_handle(point)
        if handle is not None:
            kind, signs = handle
            self._gesture = (kind, box, point, signs)
        if self._gesture is None:
            selected_id = self._hit_box(point)
            if selected_id is None:
                return
            self.cuboid_selected.emit(selected_id)
            box = self.selected_cuboid
            if box is None or box.locked:
                return
            self._gesture = ("move", box, point, None)
        self._gesture_button = button
        self._update_handle_hover(point)
        if self._gesture[0] == "rotate" and self._gl is not None:
            self._gl.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        self._focus_animation.stop()
        self.selection_started.emit()

    def _mouse_move(self, event):
        point = np.array([event.position().x(), event.position().y()])
        if self.creating and self.orthographic_view is None:
            self._cursor = tuple(point)
            self._update_creation_preview(point)
            return
        if self._navigation_button is not None:
            self._navigation_dragged |= (
                np.linalg.norm(point - self._navigation_start)
                >= QtWidgets.QApplication.startDragDistance()
            )
            if not self._navigation_dragged:
                return
            return super()._mouse_move(event)
        if self._creation_start is not None:
            self._creation_end = point
            self._update(scene=False)
            return
        if self._gesture is None:
            super()._mouse_move(event)
            self._update_handle_hover(point)
            return
        if not QtCore.QRectF(self.rect()).contains(event.position()) or (
            event.type() == QtCore.QEvent.Type.MouseMove
            and not event.buttons() & self._gesture_button
        ):
            self._finish_edit()
            return
        kind, box, start, signs = self._gesture
        delta = self.unproject(point) - self.unproject(start)
        if kind == "move":
            preview = replace(
                box, center=tuple(np.asarray(box.center) + delta)
            )
        elif kind == "resize":
            preview = box.resized(
                VIEW_AXES[self.orthographic_view], signs, delta @ box.matrix
            )
        else:
            center = self.project([box.center])[0]
            before, after = start - center, point - center
            angle = math.atan2(-after[1], after[0]) - math.atan2(
                -before[1], before[0]
            )
            axes = VIEW_AXES[self.orthographic_view]
            axis = next(i for i in range(3) if i not in axes)
            direction = np.cross(
                box.matrix[:, axes[0]], box.matrix[:, axes[1]]
            )
            angle *= np.dot(direction, box.matrix[:, axis])
            if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                angle = round(angle / math.radians(15)) * math.radians(15)
            preview = box.rotated(axis, angle)
        self.cuboid_preview.emit(preview)

    def _mouse_release(self, event):
        if self.creating and self.orthographic_view is None:
            return
        if self._navigation_button is not None:
            if event.button() != self._navigation_button:
                return
            self._mouse_move(event)
            select = self._navigation_select and not self._navigation_dragged
            self._navigation_button = None
            self._navigation_start = None
            self._navigation_dragged = False
            self._navigation_select = False
            super()._mouse_release(event)
            if select:
                self.cuboid_selected.emit(
                    self._hit_box((event.position().x(), event.position().y()))
                )
            return
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self._creation_start is not None
        ):
            self._creation_end = np.array(
                [event.position().x(), event.position().y()]
            )
            self._finish_creation()
            return
        if (
            event.button() == self._gesture_button
            and self._gesture is not None
        ):
            self._mouse_move(event)
            self._finish_edit()
            return
        super()._mouse_release(event)

    def _finish_edit(self):
        if self._gesture is None:
            return
        box = self.selected_cuboid
        self._gesture = None
        self._gesture_button = None
        self._update_handle_hover(None)
        if box is not None:
            self.cuboid_edited.emit(box)

    def _update_handle_hover(self, point):
        if self._gl is None or self.orthographic_view is None:
            return
        handle = self._hit_handle(point)
        if handle != self._hover_handle:
            self._hover_handle = handle
            self._update(scene=False)
        cursor = QtCore.Qt.CursorShape.ArrowCursor
        if handle is not None:
            kind, signs = handle
            if kind == "rotate":
                cursor = QtCore.Qt.CursorShape.OpenHandCursor
            elif signs[0] == 0:
                cursor = QtCore.Qt.CursorShape.SizeVerCursor
            elif signs[1] == 0:
                cursor = QtCore.Qt.CursorShape.SizeHorCursor
            else:
                cursor = (
                    QtCore.Qt.CursorShape.SizeFDiagCursor
                    if signs[0] * signs[1] < 0
                    else QtCore.Qt.CursorShape.SizeBDiagCursor
                )
        self._gl.setCursor(cursor)

    def _mouse_double_click(self, event):
        if (
            not self.detection_enabled
            or self.orthographic_view is not None
            or self._tool != "browse"
            or event.button() != QtCore.Qt.MouseButton.LeftButton
            or event.modifiers()
        ):
            return super()._mouse_double_click(event)
        if self.creating:
            self._update_creation_preview(
                (event.position().x(), event.position().y())
            )
            box = self._creation_preview
            if box is not None:
                self.cuboid_created.emit(box.center, box.size)
            event.accept()
            return
        selected_id = self._hit_box(
            (event.position().x(), event.position().y())
        )
        self.cancel_selection()
        self.cuboid_selected.emit(selected_id)
        if selected_id is None:
            self.reset_view(animate=True)
        else:
            self.align_cuboid(self.selected_cuboid, fit=True, animate=True)
        event.accept()

    def _finish_creation(self):
        start, end = self._creation_start, self._creation_end
        self._creation_start = self._creation_end = None
        screen = self.project(self._points[:, :3])
        if np.linalg.norm(end - start) < 5:
            self.selection_cancelled.emit()
            self._update(scene=False)
            return
        low, high = np.minimum(start, end), np.maximum(start, end)
        inside = (
            np.all((screen >= low) & (screen <= high), axis=1) & self._visible
        )
        local_start = self.unproject(start) @ self._orientation
        local_end = self.unproject(end) @ self._orientation
        center = (local_start + local_end) / 2
        size = np.maximum(np.abs(local_end - local_start), MIN_SIZE)
        axes = VIEW_AXES[self.orthographic_view]
        depth = next(i for i in range(3) if i not in axes)
        if inside.any():
            values = self._points[inside, :3] @ self._orientation[:, depth]
            center[depth] = (values.min() + values.max()) / 2
            size[depth] = max(float(np.ptp(values)), MIN_SIZE)
        else:
            size[depth] = 1.6
        center = self._orientation @ center
        self.cuboid_created.emit(center, size)
        self._update(scene=False)

    def _wheel(self, event):
        if self._gesture is not None or self._creation_start is not None:
            event.accept()
            return
        if self.orthographic_view is not None and self.selected_cuboid:
            self.align_cuboid(self.selected_cuboid)
        super()._wheel(event)
        self._update_handle_hover(
            np.array([event.position().x(), event.position().y()])
        )
        if self.creating and self.orthographic_view is None:
            self._update_creation_preview(
                (event.position().x(), event.position().y())
            )

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Type.Leave:
            self._finish_edit()
            self._update_handle_hover(None)
            self._set_creation_preview(None)
        if (
            self.detection_enabled
            and (
                self.orthographic_view is not None
                or self._tool == "browse"
                or self.creating
            )
            and event.type() == QtCore.QEvent.Type.KeyPress
        ):
            if event.key() == QtCore.Qt.Key.Key_Escape:
                self.cancel_requested.emit()
                return True
            box = self.selected_cuboid
            directions = {
                QtCore.Qt.Key.Key_Left: (-1, 0),
                QtCore.Qt.Key.Key_Right: (1, 0),
                QtCore.Qt.Key.Key_Up: (0, 1),
                QtCore.Qt.Key.Key_Down: (0, -1),
            }
            if (
                box is not None
                and not box.locked
                and event.key() in directions
                and not event.modifiers()
            ):
                self.cancel_selection()
                x, y = directions[event.key()]
                right, up, _ = self._basis()
                self.cuboid_edited.emit(
                    replace(
                        box,
                        center=tuple(
                            np.asarray(box.center) + (right * x + up * y) * 0.1
                        ),
                    )
                )
                return True
        return super().eventFilter(watched, event)

    def _paint_overlay(self, painter):
        super()._paint_overlay(painter)
        if not self.detection_enabled:
            return
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        for box in self.cuboids:
            corners = box.corners()
            points = self.project(corners)
            visible = np.isfinite(points).all(axis=1)
            if not visible.any():
                continue
            selected = box.id == self.selected_id
            color = QtGui.QColor(
                self.class_colors.get(box.class_id, "#64b5f6")
            )
            pen = QtGui.QPen(color, 2.5 if selected else 1.3)
            if box.occluded:
                pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            painter.setPen(pen)
            if self.orthographic_view is None:
                label = f"#{box.id} {self.class_names.get(box.class_id, box.class_id)}"
                if box.locked:
                    label += self.tr(" [locked]")
                painter.drawText(
                    QtCore.QPointF(
                        float(points[visible, 0].min()),
                        float(points[visible, 1].min()) - 7,
                    ),
                    label,
                )
            if selected:
                handles, rotation = self._handles(box)
                for signs, point in handles:
                    active = self._hover_handle == ("resize", signs)
                    radius = 4 if active else 3
                    painter.setPen(QtGui.QPen(color, 1.5) if active else pen)
                    painter.setBrush(
                        QtGui.QColor("#ffffff") if active else color
                    )
                    painter.drawRect(
                        QtCore.QRectF(
                            point[0] - radius,
                            point[1] - radius,
                            radius * 2,
                            radius * 2,
                        )
                    )
                if rotation is not None:
                    painter.setPen(
                        QtGui.QColor("#ffffff")
                        if self._hover_handle == ("rotate", None)
                        else color
                    )
                    draw_rotation_handle(painter, QtCore.QPointF(*rotation))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        if (
            self._creation_start is not None
            and self.orthographic_view is not None
        ):
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffd54f"), 1.5))
            painter.drawRect(
                QtCore.QRectF(
                    QtCore.QPointF(*self._creation_start),
                    QtCore.QPointF(*self._creation_end),
                ).normalized()
            )
        painter.setPen(QtGui.QColor("#cbd2df"))
        title = (
            {
                "top": self.tr("Top"),
                "side": self.tr("Side"),
                "front": self.tr("Front"),
            }[self.orthographic_view]
            if self.orthographic_view
            else self.tr("3D")
        )
        painter.drawText(QtCore.QPointF(12, 22), title)
