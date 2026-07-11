"""This module defines Canvas widget - the core component for drawing image labels"""

import copy
import math

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QWheelEvent

from anylabeling.services.auto_labeling.types import AutoLabelingMode
from anylabeling.views.labeling.utils.colormap import label_colormap
from anylabeling.views.labeling.utils.theme import get_theme

from .. import utils
from ..shape import Shape

CURSOR_DEFAULT = QtCore.Qt.CursorShape.ArrowCursor
CURSOR_POINT = QtCore.Qt.CursorShape.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CursorShape.CrossCursor
CURSOR_MOVE = QtCore.Qt.CursorShape.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.CursorShape.OpenHandCursor

AUTO_DECODE_DELAY_MS = 100
MAX_AUTO_DECODE_MARKS = 42
AUTO_DECODE_MOVE_THRESHOLD = 5.0
MOVE_SPEED = 5.0
LARGE_ROTATION_INCREMENT = math.radians(1.0)
SMALL_ROTATION_INCREMENT = math.radians(0.1)
ROTATION_HANDLE_DISTANCE = 32.0
ROTATION_HANDLE_HIT_RADIUS = 10.0
ROTATION_HANDLE_SNAP_DEGREES = 15.0
CUBOID_FRONT_EDGE_CENTER_INDICES = {
    Shape.CUBOID_FRONT_LEFT_EDGE_CENTER,
    Shape.CUBOID_FRONT_RIGHT_EDGE_CENTER,
    Shape.CUBOID_FRONT_TOP_EDGE_CENTER,
    Shape.CUBOID_FRONT_BOTTOM_EDGE_CENTER,
}
CUBOID_BACK_EDGE_CENTER_INDICES = {
    Shape.CUBOID_BACK_LEFT_EDGE_CENTER,
    Shape.CUBOID_BACK_RIGHT_EDGE_CENTER,
}
CUBOID_FACE_FRONT = "front"
CUBOID_FACE_RIGHT = "right"
CUBOID_FACE_LEFT = "left"
CUBOID_FACE_TOP = "top"
CUBOID_FACE_BOTTOM = "bottom"
CUBOID_FACE_BACK = "back"

LABEL_COLORMAP = label_colormap()


class Canvas(
    QtWidgets.QWidget
):  # pylint: disable=too-many-public-methods, too-many-instance-attributes
    """Canvas widget to handle label drawing"""

    zoom_request = QtCore.pyqtSignal(int, QtCore.QPoint)
    scroll_request = QtCore.pyqtSignal(float, object, int)
    # [Feature] support for automatically switching to editing mode
    # when the cursor moves over an object
    mode_changed = QtCore.pyqtSignal()
    new_shape = QtCore.pyqtSignal()
    show_shape = QtCore.pyqtSignal(int, int, QtCore.QPointF)
    selection_changed = QtCore.pyqtSignal(list)
    shape_moved = QtCore.pyqtSignal()
    shape_rotated = QtCore.pyqtSignal()
    shapes_deleted = QtCore.pyqtSignal(list)
    drawing_polygon = QtCore.pyqtSignal(bool)
    vertex_selected = QtCore.pyqtSignal(bool)
    auto_labeling_marks_updated = QtCore.pyqtSignal(list)
    auto_decode_requested = QtCore.pyqtSignal(list)
    auto_decode_finish_requested = QtCore.pyqtSignal()
    shape_hover_changed = QtCore.pyqtSignal()
    split_position_changed = QtCore.pyqtSignal(float)
    edit_label_requested = QtCore.pyqtSignal()
    # Emitted when brush-edit mode is toggled on/off (keeps the UI in sync).
    brush_mode_changed = QtCore.pyqtSignal(bool)
    brush_history_changed = QtCore.pyqtSignal(bool)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, rotation, line, or point
    _create_mode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                f"Unexpected value for double_click event: {self.double_click}"
            )
        self.double_click_edit_label = kwargs.pop(
            "double_click_edit_label", True
        )
        self.num_backups = kwargs.pop("num_backups", 10)
        self.wheel_rectangle_editing = kwargs.pop(
            "wheel_rectangle_editing", {}
        )
        self.enable_wheel_rectangle_editing = self.wheel_rectangle_editing.get(
            "enable", False
        )
        self.rect_adjust_step = self.wheel_rectangle_editing.get(
            "adjust_step", 2.0
        )
        self.rect_scale_step = self.wheel_rectangle_editing.get(
            "scale_step", 0.05
        )
        self.auto_highlight_shape = kwargs.pop("auto_highlight_shape", False)
        self.attributes_config = kwargs.pop("attributes", {})
        self.rotation_config = kwargs.pop("rotation", {})
        self.mask_config = kwargs.pop("mask", {})
        self.brush_config = kwargs.pop("brush", {})
        self.cuboid_config = kwargs.pop("cuboid", {})
        self.parent = kwargs.pop("parent")
        super().__init__(*args, **kwargs)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(
            QtGui.QPalette.ColorRole.Window,
            QtGui.QColor(get_theme()["background"]),
        )
        self.setPalette(palette)
        # Initialise local state.
        self.mode = self.EDIT
        self.is_auto_labeling = False
        self.is_move_editing = False
        self.auto_labeling_mode: AutoLabelingMode = None
        self.shapes = []
        self.shapes_backups = []
        self.current = None
        self.selected_shapes = []  # save the selected shapes here
        self.selected_shapes_copy = []
        # self.line represents:
        #   - create_mode == 'polygon': edge from last point to current
        #   - create_mode == 'rectangle': diagonal line of the rectangle
        #   - create_mode == 'line': the line
        #   - create_mode == 'point': the point
        self.line = Shape()
        self.prev_point = QtCore.QPointF()
        self.prev_pan_point = QtCore.QPointF()
        self.prev_move_point = QtCore.QPointF()
        self._space_pressed = False
        self._space_panning = False
        self._space_pan_prev_point = None
        self._space_pan_suppress_until_release = False
        self._vertex_erasing = False
        self._vertex_eraser_cursor_cache = None
        self.offsets = QtCore.QPointF(), QtCore.QPointF()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hide_backround = False
        self.hide_backround = False
        self.h_shape = None
        self.prev_h_shape = None
        self.h_vertex = None
        self.prev_h_vertex = None
        self.h_edge = None
        self.prev_h_edge = None
        self.h_cuboid_face = None
        self.prev_h_cuboid_face = None
        self.h_rotation_shape = None
        self.prev_h_rotation_shape = None
        self.moving_shape = False
        self._pending_edge_point = None
        self.rotating_shape = False
        self._rotation_drag_shape = None
        self._rotation_drag_prev_angle = None
        self.snapping = True
        self.h_shape_is_selected = False
        self.h_shape_is_hovered = None
        self._selected_group_id = None
        self._hovered_group_id = None
        self.allowed_oop_shape_types = ["rotation", "quadrilateral", "cuboid"]
        default_cuboid_depth_vector = self.cuboid_config.get(
            "default_depth_vector", [24.0, -24.0]
        )
        if (
            not isinstance(default_cuboid_depth_vector, (list, tuple))
            or len(default_cuboid_depth_vector) != 2
        ):
            default_cuboid_depth_vector = [24.0, -24.0]
        self.cuboid_default_depth_vector = [
            float(default_cuboid_depth_vector[0]),
            float(default_cuboid_depth_vector[1]),
        ]
        self.cuboid_min_depth = float(self.cuboid_config.get("min_depth", 5.0))
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.WheelFocus)
        self.show_groups = False
        self.show_masks = True
        self.show_texts = True
        self.show_labels = True
        self.show_scores = True
        self.show_degrees = False
        self.show_attributes = True
        self.show_linking = True

        # Set cross line options.
        self.cross_line_show = True
        self.cross_line_width = 2.0
        self.cross_line_color = "#00FF00"
        self.cross_line_opacity = 0.5

        # Set attributes color options.
        self.attr_background_color = self.attributes_config.get(
            "background_color", [33, 33, 33, 255]
        )
        self.attr_border_color = self.attributes_config.get(
            "border_color", [66, 66, 66, 255]
        )
        self.attr_text_color = self.attributes_config.get(
            "text_color", [33, 150, 243, 255]
        )

        # Set rotation increment options.
        self.large_rotation_increment = math.radians(
            self.rotation_config.get("large_increment", 1.0)
        )
        self.small_rotation_increment = math.radians(
            self.rotation_config.get("small_increment", 0.1)
        )

        # Set mask opacity options.
        self.mask_opacity = self.mask_config.get("opacity", 80)

        # Global opacity multiplier for labels/shapes (1.0 = fully opaque).
        # Controlled by the canvas adjustment panel's opacity slider.
        self.shape_opacity = 1.0

        self.is_loading = False
        self.loading_text = self.tr("Loading...")
        self.loading_angle = 0

        # Auto mask decode mode
        self.auto_decode_mode = False
        self.auto_decode_timer = QTimer()
        self.auto_decode_timer.timeout.connect(self.on_auto_decode_timeout)
        self.auto_decode_timer.setSingleShot(True)
        self.auto_decode_tracklet = []
        self.last_mouse_pos = None

        # Brush drawing mode for polygon
        self._brush_drawing = False
        self.brush_point_distance = self.brush_config.get(
            "point_distance", 25.0
        )

        # Brush edit mode (refine a selected shape by painting/erasing).
        self.is_brush_mode = False
        self.brush_radius = 12  # in image pixels
        self.eraser_mode = False  # True while Ctrl is held
        self._brush_target_shape = None
        self._brush_original_shape = None
        self._prev_brush_pos = None
        # shape -> (mask_version, outline_path)
        self._brush_overlay_cache = {}
        self._brush_modified = False
        # Mask -> polygon simplification tolerance, in image pixels.
        self.brush_simplify_epsilon_px = float(
            self.brush_config.get("simplify_epsilon", 2.0)
        )
        # Per-stroke mask snapshots powering brush undo/redo.
        self._brush_undo_stack = []
        self._brush_redo_stack = []
        self._brush_baseline_mask = None
        self._brush_stroke_dirty = False
        self._brush_max_undo_steps = max(
            1, int(self.brush_config.get("max_undo_steps", 30))
        )
        self._brush_max_undo_bytes = (
            max(1, int(self.brush_config.get("max_undo_memory_mb", 128)))
            * 1024
            * 1024
        )

        # Compare view support
        self.compare_pixmap = None
        self.split_position = 0.5

    def set_loading(self, is_loading: bool, loading_text: str = None):
        """Set loading state"""
        self.is_loading = is_loading
        if loading_text:
            self.loading_text = loading_text
        self.update()

    def set_auto_labeling_mode(self, mode: AutoLabelingMode):
        """Set auto labeling mode"""
        if mode == AutoLabelingMode.NONE:
            self.is_auto_labeling = False
            self.auto_labeling_mode = mode
        else:
            self.is_auto_labeling = True
            self.auto_labeling_mode = mode
            self.create_mode = mode.shape_type
            self.parent.toggle_draw_mode(
                False, mode.shape_type, disable_auto_labeling=False
            )

    def set_auto_decode_mode(self, enabled: bool):
        """Set auto decode mode"""
        if self.auto_decode_mode and not enabled:
            self.reset_auto_decode_state()
        self.auto_decode_mode = enabled

    def reset_auto_decode_state(self):
        """Reset auto decode state"""
        if self.auto_decode_timer.isActive():
            self.auto_decode_timer.stop()
        self.auto_decode_tracklet.clear()
        self.last_mouse_pos = None

    def fill_drawing(self):
        """Get option to fill shapes by color"""
        return self._fill_drawing

    def set_fill_drawing(self, value):
        """Set shape filling option"""
        self._fill_drawing = value
        self.update()

    @property
    def create_mode(self):
        """Create mode for canvas - Modes: polygon, rectangle, rotation, circle,..."""
        return self._create_mode

    @create_mode.setter
    def create_mode(self, value):
        """Set create mode for canvas"""
        if value not in Shape.get_supported_shape():
            raise ValueError(f"Unsupported create_mode: {value}")
        self._create_mode = value

    def store_shapes(self):
        """Store shapes for restoring later (Undo feature)"""
        shapes_backup = []
        for shape in self.shapes:
            shapes_backup.append(shape.copy())
        if len(self.shapes_backups) > self.num_backups:
            self.shapes_backups = self.shapes_backups[-self.num_backups - 1 :]
        self.shapes_backups.append(shapes_backup)

    def store_moving_shape(self):
        """Store a moving shape"""
        if self.moving_shape:
            moving_shapes = (
                [self.h_shape] + self.selected_shapes
                if self.h_shape and self.h_shape not in self.selected_shapes
                else self.selected_shapes.copy()
            )
            for shape in moving_shapes:
                if shape in self.shapes:
                    index = self.shapes.index(shape)
                    if (
                        len(self.shapes_backups) > 0
                        and index < len(self.shapes_backups[-1])
                        and self.shapes_backups[-1][index].points
                        != self.shapes[index].points
                    ):
                        self.store_shapes()
                        self.shape_moved.emit()
                        break

            self.moving_shape = False

    def clip_rectangle_to_pixmap(self, shape):
        """Clip rectangle shape to pixmap boundaries"""
        if self.pixmap is None or shape.shape_type != "rectangle":
            return True

        w, h = self.pixmap.width(), self.pixmap.height()
        points = shape.points

        if len(points) != 4:
            return True

        x_coords = [p.x() for p in points]
        y_coords = [p.y() for p in points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        clipped_min_x = max(0, min_x)
        clipped_min_y = max(0, min_y)
        clipped_max_x = min(w - 1, max_x)
        clipped_max_y = min(h - 1, max_y)

        if clipped_max_x <= clipped_min_x or clipped_max_y <= clipped_min_y:
            return False

        shape.points = [
            QtCore.QPointF(clipped_min_x, clipped_min_y),
            QtCore.QPointF(clipped_max_x, clipped_min_y),
            QtCore.QPointF(clipped_max_x, clipped_max_y),
            QtCore.QPointF(clipped_min_x, clipped_max_y),
        ]
        return True

    def clip_rotation_to_pixmap(self, shape):
        """Clip an axis-aligned rotation shape's bounding box to pixmap boundaries.

        Only clamps shapes whose direction is zero, i.e. freshly drawn in
        manual mode before any rotation has been applied.

        Args:
            shape (Shape): The rotation shape to clip.

        Returns:
            bool: True if the resulting shape is valid, False if it degenerates
                to zero area and should be discarded.
        """
        if self.pixmap is None or shape.shape_type != "rotation":
            return True
        if shape.direction != 0:
            return True
        if len(shape.points) != 4:
            return True

        w, h = self.pixmap.width(), self.pixmap.height()
        x_coords = [p.x() for p in shape.points]
        y_coords = [p.y() for p in shape.points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        clipped_min_x = max(0, min_x)
        clipped_min_y = max(0, min_y)
        clipped_max_x = min(w - 1, max_x)
        clipped_max_y = min(h - 1, max_y)

        if clipped_max_x <= clipped_min_x or clipped_max_y <= clipped_min_y:
            return False

        shape.points = [
            QtCore.QPointF(clipped_min_x, clipped_min_y),
            QtCore.QPointF(clipped_max_x, clipped_min_y),
            QtCore.QPointF(clipped_max_x, clipped_max_y),
            QtCore.QPointF(clipped_min_x, clipped_max_y),
        ]
        shape.center = QtCore.QPointF(
            (clipped_min_x + clipped_max_x) / 2,
            (clipped_min_y + clipped_max_y) / 2,
        )
        return True

    @property
    def is_shape_restorable(self):
        """Check if shape can be restored from backup"""
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapes_backups) < 2:
            return False
        return True

    def restore_shape(self):
        """Restore/Undo a shape"""
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::load_shapes and our own Canvas::load_shapes function.
        if not self.is_shape_restorable:
            return
        self.shapes_backups.pop()  # latest

        # The application will eventually call Canvas.load_shapes which will
        # push this right back onto the stack.
        shapes_backup = self.shapes_backups.pop()
        self.shapes = shapes_backup
        self.selected_shapes = []
        self._selected_group_id = None
        self._hovered_group_id = None
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, _):
        """Mouse enter event"""
        self.override_cursor(self._cursor)

    def leaveEvent(self, _):
        """Mouse leave event"""
        self._clear_space_pan_state()
        self.store_moving_shape()
        self.un_highlight()
        self._hovered_group_id = None
        self.restore_cursor()
        self.shape_hover_changed.emit()

    def focusOutEvent(self, _):
        """Window out of focus event"""
        self._clear_space_pan_state()
        self.restore_cursor()

    def is_visible(self, shape):
        """Check if a shape is visible"""
        return self.visible.get(shape, True)

    def _shape_hit_candidates(self, point):
        """Return shapes under a point in interaction priority order."""
        candidates = []
        epsilon = self.epsilon / self.scale
        for stack_index, shape in enumerate(self.shapes):
            if not self.is_visible(shape):
                continue

            rect = shape.bounding_rect()
            area = max(0.0, rect.width()) * max(0.0, rect.height())
            vertex_distance = None
            if not shape.locked:
                if shape.shape_type == "cuboid" and len(shape.points) == 8:
                    vertex_index = self.nearest_cuboid_control(
                        shape, point, epsilon
                    )
                    vertex = (
                        self.cuboid_control_point(shape, vertex_index)
                        if vertex_index is not None
                        else None
                    )
                else:
                    vertex_index = shape.nearest_vertex(point, epsilon)
                    vertex = (
                        shape.points[vertex_index]
                        if vertex_index is not None
                        else None
                    )
                if vertex is not None:
                    vertex_distance = utils.distance(vertex - point)

            if vertex_distance is not None:
                priority = (0, vertex_distance, area, -stack_index)
                candidates.append((priority, shape))
                continue

            if (
                not shape.locked
                and len(shape.points) > 1
                and shape.can_add_point()
                and shape.shape_type != "quadrilateral"
            ):
                edge_index = shape.nearest_edge(point, epsilon)
                if edge_index is not None:
                    line = [
                        shape.points[edge_index - 1],
                        shape.points[edge_index],
                    ]
                    edge_distance = utils.distance_to_line(point, line)
                    priority = (1, edge_distance, area, -stack_index)
                    candidates.append((priority, shape))
                    continue

            if shape.shape_type in ["point", "line", "linestrip"]:
                vertex_index = shape.nearest_vertex(point, epsilon * 3)
                if vertex_index is None:
                    continue
                distance = utils.distance(shape.points[vertex_index] - point)
                priority = (1, distance, area, -stack_index)
                candidates.append((priority, shape))
                continue

            if shape.shape_type == "cuboid" and len(shape.points) == 8:
                front_path = self.cuboid_face_path(shape, CUBOID_FACE_FRONT)
                hit = (
                    front_path is not None and front_path.contains(point)
                ) or self.cuboid_face_hit_test(shape, point) is not None
            else:
                hit = len(shape.points) > 1 and shape.contains_point(point)
            if hit:
                priority = (2, area, 0.0, -stack_index)
                candidates.append((priority, shape))

        candidates.sort(key=lambda item: item[0])
        return [shape for _, shape in candidates]

    def drawing(self):
        """Check if user is drawing (mode==CREATE)"""
        return self.mode == self.CREATE

    def editing(self):
        """Check if user is editing (mode==EDIT)"""
        return self.mode == self.EDIT

    # ------------------------------------------------------------------ #
    # Brush edit mode
    #
    # Refine a single selected shape by painting (add) or Ctrl+painting
    # (erase) onto a rasterized mask, resize the brush with the mouse
    # wheel, then convert the mask back into a simplified polygon on exit.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _polygon_to_mask(points_xy: list, shape_hw: tuple) -> np.ndarray:
        """Rasterize polygon vertices into a binary ``uint8`` mask.

        Args:
            points_xy: Polygon vertices as ``(x, y)`` integer pairs.
            shape_hw: Target mask shape as ``(height, width)``.

        Returns:
            A ``(height, width)`` ``uint8`` array with filled pixels set
            to ``255`` and the background set to ``0``.
        """
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        if not points_xy:
            return mask
        pts = np.array(points_xy, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        return mask

    @staticmethod
    def _mask_to_polylines(mask: np.ndarray) -> list:
        """Extract a mask's external contours as polylines.

        Args:
            mask: A 2D ``uint8`` mask whose non-zero pixels are foreground.

        Returns:
            A list of polylines, each a list of ``(x, y)`` integer tuples.
            Contours with fewer than three points are dropped.
        """
        if mask is None:
            return []
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        if mask.ndim != 2:
            mask = mask.squeeze()
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polylines = []
        for cnt in contours:
            if cnt is None or len(cnt) < 3:
                continue
            pts = cnt.reshape(-1, 2)
            polylines.append([(int(x), int(y)) for x, y in pts])
        return polylines

    @staticmethod
    def _apply_brush_to_mask(
        mask: np.ndarray,
        x: float,
        y: float,
        radius: int,
        add: bool = True,
    ) -> np.ndarray:
        """Stamp a filled circle onto a mask in place.

        Args:
            mask: The 2D ``uint8`` mask to modify.
            x: Circle center x coordinate, in image pixels.
            y: Circle center y coordinate, in image pixels.
            radius: Brush radius in image pixels (clamped to ``>= 1``).
            add: If ``True`` paint foreground (``255``); otherwise erase
                to background (``0``).

        Returns:
            The same mask instance, modified in place.
        """
        if mask is None:
            return mask
        cv2.circle(
            mask,
            (int(round(x)), int(round(y))),
            max(1, int(round(radius))),
            255 if add else 0,
            thickness=-1,
            lineType=cv2.LINE_8,
        )
        return mask

    @staticmethod
    def _simplify_contour(cnt: np.ndarray, epsilon_px: float) -> list:
        """Simplify a contour with the Ramer-Douglas-Peucker algorithm.

        Args:
            cnt: An OpenCV contour of shape ``(N, 1, 2)``.
            epsilon_px: Approximation tolerance in image pixels. A value of
                zero preserves the extracted contour.

        Returns:
            A list of simplified ``(x, y)`` integer vertices, or an empty
            list when the result would have fewer than three points.
        """
        if cnt is None or len(cnt) < 3:
            return []
        eps = max(0.0, float(epsilon_px))
        if eps == 0:
            pts = cnt.reshape(-1, 2)
            return [(int(x), int(y)) for x, y in pts]
        approx = cv2.approxPolyDP(cnt, eps, True)
        if approx is None or len(approx) < 3:
            return []
        pts = approx.reshape(-1, 2)
        return [(int(x), int(y)) for x, y in pts]

    def _ensure_brush_mask(self, shape: Shape) -> None:
        """Ensure ``shape`` owns an editable mask buffer.

        If the shape has no mask yet, one is rasterized from its current
        polygon points at the image resolution. The ``_brush_using_mask``
        flag is always set so the canvas renders the mask in place of the
        vector polygon.

        Args:
            shape: The shape being brush-edited.
        """
        if shape is None or self.pixmap is None:
            return
        if getattr(shape, "mask", None) is None:
            h, w = int(self.pixmap.height()), int(self.pixmap.width())
            points = [
                (int(round(point.x())), int(round(point.y())))
                for point in shape.points
            ]
            shape.mask = self._polygon_to_mask(points, (h, w))
            shape._brush_mask_version = 0
        shape._brush_using_mask = True

    def _update_shape_points_from_mask(self, shape: Shape) -> bool:
        """Rewrite ``shape.points`` from its mask's largest component.

        The largest external contour is simplified and stored as the new
        polygon. Donut holes are not supported and are discarded.

        Args:
            shape: The brush-edited shape whose mask is converted back
                into polygon vertices.

        Returns:
            ``True`` when a valid polygon was produced.
        """
        if shape is None or getattr(shape, "mask", None) is None:
            return False
        polylines = self._mask_to_polylines(shape.mask)
        if not polylines:
            shape.points = []
            return False
        best = None
        best_area = -1.0
        for poly in polylines:
            if len(poly) < 3:
                continue
            area = float(cv2.contourArea(np.array(poly, dtype=np.int32)))
            if area > best_area:
                best_area = area
                best = poly
        if best is None or len(best) < 3:
            shape.points = []
            return False
        cnt = np.array(best, dtype=np.int32).reshape((-1, 1, 2))
        outer = self._simplify_contour(cnt, self.brush_simplify_epsilon_px)
        if len(outer) < 3:
            outer = best
        shape.mask.fill(0)
        cv2.fillPoly(shape.mask, [cnt], 255)
        self._bump_brush_version(shape)
        shape.shape_type = "polygon"
        shape.points = [QtCore.QPointF(float(x), float(y)) for x, y in outer]
        # Donut holes are not representable as a single polygon.
        shape.other_data.pop("holes", None)
        shape.close()
        return True

    def _invalidate_brush_cache(self, shape: Shape) -> None:
        """Drop the cached overlay image for ``shape``.

        Args:
            shape: The shape whose render cache is invalidated so the next
                paint regenerates it.
        """
        if shape is None:
            return
        self._brush_overlay_cache.pop(shape, None)

    def _bump_brush_version(self, shape: Shape) -> None:
        """Advance a shape's mask version and invalidate its cache.

        Args:
            shape: The shape whose rendered overlay must be refreshed.
        """
        shape._brush_mask_version = (
            int(getattr(shape, "_brush_mask_version", 0)) + 1
        )
        self._invalidate_brush_cache(shape)

    def _get_brush_render_data(
        self, shape: Shape
    ) -> QtGui.QPainterPath | None:
        """Build and cache the outline path for a mask.

        Args:
            shape: A shape currently rendered from its brush mask.

        Returns:
            A ``QPainterPath``, or ``None`` when the shape has no mask.
            Results are cached by mask version so repeated repaints between
            mask updates stay cheap.
        """
        if shape is None or getattr(shape, "mask", None) is None:
            return None
        version = int(getattr(shape, "_brush_mask_version", 0))
        cached = self._brush_overlay_cache.get(shape)
        if cached and cached[0] == version:
            return cached[1]

        mask = shape.mask
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        if mask.ndim != 2:
            mask = mask.squeeze()

        outline_path = QtGui.QPainterPath()
        for poly in self._mask_to_polylines(mask):
            if len(poly) < 3:
                continue
            outline_path.moveTo(float(poly[0][0]), float(poly[0][1]))
            for x, y in poly[1:]:
                outline_path.lineTo(float(x), float(y))
            outline_path.closeSubpath()

        self._brush_overlay_cache[shape] = (version, outline_path)
        return outline_path

    def _restore_brush_original_geometry(self, shape: Shape) -> None:
        """Restore geometry captured when brush editing started."""
        original = self._brush_original_shape
        if shape is None or original is None:
            return
        shape.shape_type = original.shape_type
        shape.points = [QtCore.QPointF(point) for point in original.points]
        shape.direction = original.direction
        shape.center = original.center
        shape.other_data = copy.deepcopy(original.other_data)

    def _leave_brush_mode(self, cancel: bool) -> None:
        """Leave brush mode by committing or discarding mask changes."""
        target = self._brush_target_shape
        self.is_brush_mode = False
        self.override_cursor(CURSOR_DEFAULT)
        self._prev_brush_pos = None
        self._brush_target_shape = None
        self.eraser_mode = False

        if target is not None and getattr(target, "mask", None) is not None:
            if self._brush_baseline_mask is not None and np.array_equal(
                target.mask, self._brush_baseline_mask
            ):
                self._brush_modified = False
            has_geometry = True
            if cancel or not self._brush_modified:
                self._restore_brush_original_geometry(target)
            else:
                has_geometry = self._update_shape_points_from_mask(target)
            target._brush_using_mask = False
            target.mask = None
            self._invalidate_brush_cache(target)
            if self._brush_modified and not cancel:
                if not has_geometry and target in self.shapes:
                    self.shapes.remove(target)
                    self.selected_shapes = [
                        shape
                        for shape in self.selected_shapes
                        if shape is not target
                    ]
                    target.selected = False
                self.store_shapes()
                if has_geometry:
                    self.shape_moved.emit()
                else:
                    self.shapes_deleted.emit([target])

        self._brush_modified = False
        self._brush_undo_stack = []
        self._brush_redo_stack = []
        self._brush_baseline_mask = None
        self._brush_original_shape = None
        self._brush_stroke_dirty = False
        self.brush_mode_changed.emit(False)
        self.brush_history_changed.emit(self.is_shape_restorable)
        self.update()

    def cancel_brush_mode(self) -> None:
        """Discard brush changes and restore the original polygon."""
        if self.is_brush_mode:
            self._leave_brush_mode(cancel=True)
        else:
            self.brush_mode_changed.emit(False)

    def set_brush_mode(self, enabled: bool) -> None:
        """Enter or leave brush-edit mode for the selected shape.

        On enter the single selected shape (brush editing requires exactly
        one) is converted to a mask, a blank cursor is shown so the preview
        circle reads as the brush, and an undo baseline is captured. On
        exit the mask is converted back into a simplified polygon, the
        shape is stored/saved when modified, and brush state is cleared.

        Args:
            enabled: ``True`` to enter brush mode, ``False`` to leave it.
        """
        if enabled:
            if (
                len(self.selected_shapes) != 1
                or self.selected_shapes[0].shape_type != "polygon"
                or self.selected_shapes[0].locked
            ):
                self.brush_mode_changed.emit(False)
                return
            self.set_editing(True)
            self.is_brush_mode = True
            self._brush_modified = False
            self.override_cursor(QtCore.Qt.CursorShape.BlankCursor)
            self._brush_target_shape = (
                self.selected_shapes[0]
                if len(self.selected_shapes) == 1
                else None
            )
            self._brush_original_shape = (
                self._brush_target_shape.copy()
                if self._brush_target_shape is not None
                else None
            )
            self._prev_brush_pos = None
            if self._brush_target_shape is not None:
                self._ensure_brush_mask(self._brush_target_shape)
                self._invalidate_brush_cache(self._brush_target_shape)
                mask = getattr(self._brush_target_shape, "mask", None)
                if mask is not None:
                    self._brush_baseline_mask = mask.copy()
                    self._brush_undo_stack = [self._brush_baseline_mask.copy()]
                    self._brush_redo_stack = []
                else:
                    self._brush_baseline_mask = None
                    self._brush_undo_stack = []
                    self._brush_redo_stack = []
            else:
                self._brush_baseline_mask = None
                self._brush_undo_stack = []
                self._brush_redo_stack = []
            self.brush_mode_changed.emit(True)
            self.brush_history_changed.emit(False)
            self.update()
            return

        self._leave_brush_mode(cancel=False)

    def _push_brush_undo_state(self) -> None:
        """Snapshot the current mask onto the brush undo stack.

        Called once per completed stroke. Consecutive duplicate states are
        skipped, the redo stack is cleared, and the stack is trimmed to
        ``_brush_max_undo_steps`` while preserving the baseline at index 0.
        """
        shape = self._brush_target_shape
        if shape is None or getattr(shape, "mask", None) is None:
            return
        mask = shape.mask
        if not self._brush_undo_stack:
            self._brush_undo_stack = [mask.copy()]
            self._brush_redo_stack = []
            return
        last = self._brush_undo_stack[-1]
        if last.shape == mask.shape and np.array_equal(last, mask):
            return
        self._brush_undo_stack.append(mask.copy())
        self._brush_redo_stack.clear()
        while len(self._brush_undo_stack) > self._brush_max_undo_steps + 1:
            del self._brush_undo_stack[1]
        while (
            len(self._brush_undo_stack) > 2
            and sum(state.nbytes for state in self._brush_undo_stack)
            > self._brush_max_undo_bytes
        ):
            del self._brush_undo_stack[1]
        self.brush_history_changed.emit(self.brush_can_undo())

    def brush_can_undo(self) -> bool:
        """Return whether a brush stroke is available to undo."""
        return self.is_brush_mode and len(self._brush_undo_stack) > 1

    def brush_can_redo(self) -> bool:
        """Return whether a brush stroke is available to redo."""
        return self.is_brush_mode and len(self._brush_redo_stack) > 0

    def _restore_brush_mask(self, mask: np.ndarray) -> None:
        """Apply a mask snapshot to the target shape and refresh it.

        Args:
            mask: The mask snapshot to restore onto the target shape.
        """
        shape = self._brush_target_shape
        shape.mask = mask.copy()
        self._bump_brush_version(shape)
        self._update_shape_points_from_mask(shape)
        if self._brush_baseline_mask is not None:
            self._brush_modified = not np.array_equal(
                shape.mask, self._brush_baseline_mask
            )
        self.update()

    def brush_undo(self) -> None:
        """Revert the target shape's mask to the previous stroke."""
        if not self.brush_can_undo():
            return
        shape = self._brush_target_shape
        if shape is None or getattr(shape, "mask", None) is None:
            return
        current = self._brush_undo_stack.pop()
        self._brush_redo_stack.append(current)
        self._restore_brush_mask(self._brush_undo_stack[-1])
        self.brush_history_changed.emit(self.brush_can_undo())

    def brush_redo(self) -> None:
        """Re-apply the next stroke on the brush redo stack."""
        if not self.brush_can_redo():
            return
        shape = self._brush_target_shape
        if shape is None or getattr(shape, "mask", None) is None:
            return
        nxt = self._brush_redo_stack.pop()
        self._brush_undo_stack.append(nxt.copy())
        self._restore_brush_mask(nxt)
        self.brush_history_changed.emit(self.brush_can_undo())

    def _brush_mouse_press(self, ev, pos: QtCore.QPointF) -> bool:
        """Handle a mouse press while brush mode is active.

        A left press stamps the brush at ``pos`` (erasing while Ctrl is
        held); a right press is consumed so its release can leave brush
        mode without opening the context menu.

        Args:
            ev: The Qt mouse event.
            pos: Cursor position in image coordinates.

        Returns:
            ``True`` if the event was consumed by brush mode.
        """
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            return True
        if (
            ev.button() == QtCore.Qt.MouseButton.LeftButton
            and self.editing()
            and self._brush_target_shape is not None
        ):
            self._ensure_brush_mask(self._brush_target_shape)
            self.eraser_mode = bool(
                ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
            )
            self._apply_brush_to_mask(
                self._brush_target_shape.mask,
                pos.x(),
                pos.y(),
                radius=max(1, int(round(self.brush_radius))),
                add=not self.eraser_mode,
            )
            self._prev_brush_pos = QtCore.QPointF(pos)
            self._brush_stroke_dirty = True
            self._brush_modified = True
            self._bump_brush_version(self._brush_target_shape)
            self.update()
            return True
        return False

    def _brush_mouse_move(self, ev, pos: QtCore.QPointF) -> bool:
        """Handle mouse movement while brush mode is active.

        With the left button held, brush stamps are interpolated between
        the previous and current positions so fast drags leave no gaps.
        Otherwise the canvas just repaints to refresh the preview circle.

        Args:
            ev: The Qt mouse event.
            pos: Cursor position in image coordinates.

        Returns:
            ``True`` (brush mode always consumes movement events).
        """
        self.prev_move_point = pos
        self.eraser_mode = bool(
            ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
        )
        target = self._brush_target_shape
        if (
            QtCore.Qt.MouseButton.LeftButton & ev.buttons()
        ) and target is not None:
            self._ensure_brush_mask(target)
            add = not self.eraser_mode
            radius = max(1, int(round(self.brush_radius)))
            prev = self._prev_brush_pos
            if prev is None:
                self._apply_brush_to_mask(
                    target.mask, pos.x(), pos.y(), radius=radius, add=add
                )
            else:
                dx = pos.x() - prev.x()
                dy = pos.y() - prev.y()
                dist = float((dx * dx + dy * dy) ** 0.5)
                step = max(1.0, radius * 0.5)
                steps = int(dist // step) if dist > 0 else 0
                for i in range(steps + 1):
                    t = (i / steps) if steps > 0 else 1.0
                    x = prev.x() * (1 - t) + pos.x() * t
                    y = prev.y() * (1 - t) + pos.y() * t
                    self._apply_brush_to_mask(
                        target.mask, x, y, radius=radius, add=add
                    )
            self._prev_brush_pos = QtCore.QPointF(pos)
            self._brush_stroke_dirty = True
            self._brush_modified = True
            self._bump_brush_version(target)
        self.update()
        return True

    def _brush_mouse_release(self, ev) -> bool:
        """Finish a brush stroke or exit brush mode on mouse release.

        A right release leaves brush mode; a left release converts the
        mask to polygon points and snapshots an undo step.

        Args:
            ev: The Qt mouse event.

        Returns:
            ``True`` if the event was consumed by brush mode.
        """
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.set_brush_mode(False)
            return True
        if (
            ev.button() == QtCore.Qt.MouseButton.LeftButton
            and self._brush_target_shape is not None
            and getattr(self._brush_target_shape, "mask", None) is not None
        ):
            self._update_shape_points_from_mask(self._brush_target_shape)
            if self._brush_stroke_dirty:
                self._push_brush_undo_state()
            self._brush_stroke_dirty = False
            self._prev_brush_pos = None
            self.update()
            return True
        return False

    def _brush_key_press(self, ev) -> bool:
        """Handle brush-specific undo/redo shortcuts.

        ``Ctrl+Z`` undoes a stroke; ``Ctrl+Shift+Z`` and ``Ctrl+Y`` redo.

        Args:
            ev: The Qt key event.

        Returns:
            ``True`` if the shortcut was consumed by brush mode.
        """
        modifiers = ev.modifiers()
        key = ev.key()
        if key == QtCore.Qt.Key.Key_Escape:
            self.cancel_brush_mode()
            ev.accept()
            return True
        ctrl = bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)
        if ctrl and key == QtCore.Qt.Key.Key_Z:
            if shift:
                self.brush_redo()
            else:
                self.brush_undo()
            ev.accept()
            return True
        if ctrl and key == QtCore.Qt.Key.Key_Y:
            self.brush_redo()
            ev.accept()
            return True
        return False

    def _brush_drawing_can_close(self, pos: QtCore.QPointF) -> bool:
        """Return whether a freehand polygon stroke reaches its start."""
        if self.current is None or len(self.current) < 3:
            return False
        start = self.current[0]
        threshold = self.epsilon / self.scale
        return self.close_enough(pos, start) or (
            utils.distance_to_line(start, [self.current[-1], pos]) < threshold
        )

    def _paint_brush_overlays(self, p: QtGui.QPainter) -> None:
        """Paint live mask overlays for shapes being brush-edited.

        Args:
            p: The active painter, already translated/scaled to image space.
        """
        for shape in self.shapes:
            if not getattr(shape, "_brush_using_mask", False):
                continue
            if getattr(shape, "mask", None) is None or not shape.visible:
                continue
            outline_path = self._get_brush_render_data(shape)
            if outline_path is not None:
                outline_color = (
                    shape.select_line_color
                    if shape.selected
                    else shape.line_color
                )
                pen = QtGui.QPen(outline_color)
                pen.setWidth(
                    max(1, int(round(shape.line_width / Shape.scale)))
                )
                if getattr(shape, "difficult", False):
                    pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                fill_color = QtGui.QColor(outline_color)
                fill_color.setAlpha(int(self.mask_opacity))
                p.setBrush(fill_color)
                p.drawPath(outline_path)

    def _paint_brush_cursor(self, p: QtGui.QPainter) -> None:
        """Draw the circular brush-size preview at the cursor.

        The circle is white while adding and red while erasing so the
        active modifier is obvious at a glance.

        Args:
            p: The active painter, already translated/scaled to image space.
        """
        if not self.is_brush_mode:
            return
        r = max(1.0, float(self.brush_radius))
        p.setOpacity(1.0)
        if self.eraser_mode:
            pen_color = QtGui.QColor(255, 100, 100, 255)
            fill_color = QtGui.QColor(255, 100, 100, 50)
        else:
            pen_color = QtGui.QColor(255, 255, 255, 255)
            fill_color = QtGui.QColor(255, 255, 255, 50)
        p.setPen(QtGui.QPen(pen_color, 2))
        p.setBrush(fill_color)
        p.drawEllipse(QtCore.QPointF(self.prev_move_point), r, r)

    def _paint_rotation_handles(self, p):
        for shape in self._rotation_handle_shapes():
            self._paint_rotation_handle(p, shape)

    def _paint_rotation_handle(self, p, shape):
        geometry = self._rotation_handle_geometry(shape)
        if geometry is None:
            return
        _, handle, _ = geometry
        scale = max(self.scale, 1e-6)
        vertex_radius = Shape.point_size / (2.0 * scale)
        vertex_pen_width = max(1.0 / scale, float(shape.line_width) / scale)
        radius = vertex_radius + vertex_pen_width / 2.0
        hovered = shape in (self.h_rotation_shape, self._rotation_drag_shape)
        ring_width = (vertex_pen_width / 2.0) * (2.2 if hovered else 1.0)
        inner_radius = max(0.5 / scale, radius - ring_width)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(0, 0, 0, 255))
        p.drawEllipse(handle, radius, radius)
        p.setBrush(QtGui.QColor(255, 255, 255, 255))
        p.drawEllipse(handle, inner_radius, inner_radius)

    def set_auto_labeling(self, value=True):
        """Set auto labeling mode"""
        self.is_auto_labeling = value
        if self.auto_labeling_mode is None:
            self.auto_labeling_mode = AutoLabelingMode.NONE
            self.parent.toggle_draw_mode(
                True, "rectangle", disable_auto_labeling=True
            )

    def get_mode(self):
        """Get current mode"""
        if (
            self.is_auto_labeling
            and self.auto_labeling_mode != AutoLabelingMode.NONE
        ):
            return self.tr("Auto Labeling")
        if self.mode == self.CREATE:
            return self.tr("Drawing")
        elif self.mode == self.EDIT:
            return self.tr("Editing")
        else:
            return self.tr("Unknown")

    def set_editing(self, value=True):
        """Set editing mode. Editing is set to False, user is drawing"""
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.un_highlight()
            self.deselect_shape()
            self.is_move_editing = False
            self.shape_hover_changed.emit()

    def un_highlight(self):
        """Unhighlight shape/vertex/edge"""
        if self.h_shape:
            self.h_shape.highlight_clear()
            self.update()
        self.prev_h_shape = self.h_shape
        self.prev_h_vertex = self.h_vertex
        self.prev_h_edge = self.h_edge
        self.prev_h_cuboid_face = self.h_cuboid_face
        self.prev_h_rotation_shape = self.h_rotation_shape
        self.h_shape = self.h_vertex = self.h_edge = self.h_cuboid_face = None
        self.h_rotation_shape = None

    def selected_vertex(self):
        """Check if selected a vertex"""
        return self.h_vertex is not None

    def selected_edge(self):
        """Check if selected an edge"""
        return self.h_edge is not None

    def selected_cuboid_face(self):
        return self.h_cuboid_face is not None

    def can_erase_selected_vertices(self):
        if not self.editing() or len(self.selected_shapes) != 1:
            return False
        shape = self.selected_shapes[0]
        return not shape.locked and shape.shape_type in [
            "polygon",
            "linestrip",
        ]

    def _vertex_eraser_cursor(self):
        if self._vertex_eraser_cursor_cache is None:
            pixmap = QtGui.QPixmap(":/images/images/eraser.svg")
            if pixmap.isNull():
                return CURSOR_POINT
            self._vertex_eraser_cursor_cache = QtGui.QCursor(
                pixmap.scaled(
                    24,
                    24,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                ),
                3,
                20,
            )
        return self._vertex_eraser_cursor_cache

    def _set_vertex_eraser_tooltip(self):
        if not self.can_erase_selected_vertices():
            return
        tooltip = self.tr("Click & drag to erase points of shape '%s'") % (
            self.selected_shapes[0].label
        )
        self.setToolTip(tooltip)
        self.setStatusTip(tooltip)

    @staticmethod
    def _minimum_points_for_shape(shape):
        if shape.shape_type == "polygon":
            return 3
        if shape.shape_type == "linestrip":
            return 2
        return 0

    def erase_selected_vertex_at(self, pos):
        if not self.can_erase_selected_vertices():
            return False
        shape = self.selected_shapes[0]
        index = shape.nearest_vertex(pos, self.epsilon / self.scale)
        if index is None:
            return False

        shape.remove_point(index)
        shape.highlight_clear()
        self.h_shape = shape
        self.h_vertex = None
        self.prev_h_vertex = None
        if len(shape.points) < self._minimum_points_for_shape(shape):
            if shape in self.shapes:
                self.shapes.remove(shape)
            self.selected_shapes = []
            self.h_shape = None
            self.h_edge = None
            self.moving_shape = False
            self.store_shapes()
            self.shapes_deleted.emit([shape])
        else:
            self.moving_shape = True
        self.update()
        return True

    @staticmethod
    def _snap_line_pos(anchor, pos):
        """Snap line endpoint to horizontal or vertical direction."""
        dx = abs(pos.x() - anchor.x())
        dy = abs(pos.y() - anchor.y())
        if dx >= dy:
            return QtCore.QPointF(pos.x(), anchor.y())
        return QtCore.QPointF(anchor.x(), pos.y())

    def _should_trigger_auto_decode(self, pos):
        """Check if mouse movement exceeds threshold to trigger auto decode"""
        if not self.auto_decode_tracklet:
            return True

        last_point = self.auto_decode_tracklet[-1]["data"]
        distance = (
            (pos.x() - last_point[0]) ** 2 + (pos.y() - last_point[1]) ** 2
        ) ** 0.5
        return distance >= AUTO_DECODE_MOVE_THRESHOLD

    def _has_valid_pixmap(self):
        return (
            self.pixmap
            and not self.pixmap.isNull()
            and self.pixmap.width()
            and self.pixmap.height()
        )

    def _clear_space_pan_state(self):
        self._space_pressed = False
        self._space_panning = False
        self._space_pan_prev_point = None
        self._space_pan_suppress_until_release = False

    def _restore_space_pan_cursor(self):
        if self._space_pressed:
            self.override_cursor(CURSOR_GRAB)
        elif self.drawing():
            self.override_cursor(CURSOR_DRAW)
        else:
            self.override_cursor(CURSOR_DEFAULT)

    def _start_space_pan(self, point):
        if not self._has_valid_pixmap():
            return False
        self._space_panning = True
        self._space_pan_suppress_until_release = True
        self._space_pan_prev_point = point
        self._clear_space_pan_hover_state()
        self.override_cursor(CURSOR_MOVE)
        return True

    def _left_button_pressed(self, ev):
        return bool(QtCore.Qt.MouseButton.LeftButton & ev.buttons())

    def _clear_space_pan_hover_state(self):
        had_hover = any(
            item is not None
            for item in (
                self.h_shape,
                self.h_vertex,
                self.h_edge,
                self.h_cuboid_face,
                self.h_rotation_shape,
            )
        )
        self.un_highlight()
        self.h_shape_is_selected = False
        self.vertex_selected.emit(False)
        if had_hover:
            self.shape_hover_changed.emit()

    @staticmethod
    def _rotation_shape_center(shape):
        return QtCore.QPointF(
            (shape.points[0].x() + shape.points[2].x()) / 2.0,
            (shape.points[0].y() + shape.points[2].y()) / 2.0,
        )

    def _rotation_handle_geometry(self, shape):
        if (
            shape is None
            or shape.shape_type != "rotation"
            or len(shape.points) != 4
        ):
            return None
        p0, p1 = shape.points[0], shape.points[1]
        dx = p1.x() - p0.x()
        dy = p1.y() - p0.y()
        edge_length = math.hypot(dx, dy)
        if edge_length < 1e-6:
            return None
        edge_mid = QtCore.QPointF(
            (p0.x() + p1.x()) / 2.0,
            (p0.y() + p1.y()) / 2.0,
        )
        normal_x = dy / edge_length
        normal_y = -dx / edge_length
        distance = ROTATION_HANDLE_DISTANCE / max(self.scale, 1e-6)
        handle = QtCore.QPointF(
            edge_mid.x() + normal_x * distance,
            edge_mid.y() + normal_y * distance,
        )
        return edge_mid, handle, self._rotation_shape_center(shape)

    def _rotation_handle_shapes(self):
        candidates = []
        for shape in self.selected_shapes:
            if shape not in candidates:
                candidates.append(shape)
        for shape in (self.h_shape, self.h_rotation_shape):
            if shape is not None and shape not in candidates:
                candidates.append(shape)
        return sorted(
            [
                shape
                for shape in candidates
                if shape in self.shapes
                and not shape.locked
                and shape.visible
                and self.is_visible(shape)
                and shape.shape_type == "rotation"
                and len(shape.points) == 4
            ],
            key=lambda shape: self.shapes.index(shape),
            reverse=True,
        )

    def _rotation_handle_shape_at(self, pos):
        hit_radius = ROTATION_HANDLE_HIT_RADIUS / max(self.scale, 1e-6)
        for shape in self._rotation_handle_shapes():
            geometry = self._rotation_handle_geometry(shape)
            if geometry is None:
                continue
            edge_mid, handle, _ = geometry
            if utils.distance(handle - pos) <= hit_radius:
                return shape
            if utils.distance_to_line(pos, [edge_mid, handle]) <= hit_radius:
                return shape
        return None

    def _set_rotation_handle_hover(self, shape):
        if self.h_shape is not None:
            self.h_shape.highlight_clear()
        self.prev_h_vertex = self.h_vertex
        self.h_vertex = None
        self.prev_h_shape = self.h_shape = shape
        self.prev_h_edge = self.h_edge
        self.h_edge = None
        self.prev_h_cuboid_face = self.h_cuboid_face
        self.h_cuboid_face = None
        self.prev_h_rotation_shape = self.h_rotation_shape
        self.h_rotation_shape = shape
        self.override_cursor(CURSOR_POINT)
        self.setToolTip(
            self.tr("Click & drag to rotate shape '%s'") % shape.label
        )
        self.setStatusTip(self.toolTip())
        self.update()

    def _rotation_mouse_angle(self, shape, pos):
        if shape is None or len(shape.points) != 4:
            return None
        center = self._rotation_shape_center(shape)
        return math.atan2(pos.y() - center.y(), pos.x() - center.x())

    @staticmethod
    def _snap_rotation_angle(angle):
        step = math.radians(ROTATION_HANDLE_SNAP_DEGREES)
        return round(angle / step) * step

    def _start_rotation_handle_drag(
        self, shape, pos, multiple_selection_mode, modifiers
    ):
        self.set_hiding()
        if shape not in self.selected_shapes:
            if multiple_selection_mode:
                self.selection_changed.emit(self.selected_shapes + [shape])
            else:
                self.selection_changed.emit([shape])
            self.h_shape_is_selected = False
        else:
            self.h_shape_is_selected = True
        self.h_shape = shape
        self.h_rotation_shape = shape
        self.h_vertex = None
        self.h_edge = None
        self.h_cuboid_face = None
        angle = self._rotation_mouse_angle(shape, pos)
        if angle is None:
            return
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            angle = self._snap_rotation_angle(angle)
        self._rotation_drag_shape = shape
        self._rotation_drag_prev_angle = angle
        self.prev_point = pos
        self.calculate_offsets(pos)
        self.override_cursor(CURSOR_MOVE)

    def _update_rotation_handle_drag(self, pos, modifiers):
        shape = self._rotation_drag_shape
        if shape is None or shape.locked:
            return
        angle = self._rotation_mouse_angle(shape, pos)
        if angle is None or self._rotation_drag_prev_angle is None:
            return
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            angle = self._snap_rotation_angle(angle)
        theta = self._rotation_drag_prev_angle - angle
        if abs(theta) < 1e-9:
            return
        if self.bounded_rotate_shapes(0, shape, theta):
            self._rotation_drag_prev_angle = angle
            self.rotating_shape = True
            self.h_shape = shape
            self.h_rotation_shape = shape
            self.repaint()

    def _store_rotated_shape(self, shape):
        if shape is None or shape not in self.shapes:
            return
        index = self.shapes.index(shape)
        if (
            self.shapes_backups
            and index < len(self.shapes_backups[-1])
            and self.shapes_backups[-1][index].points
            != self.shapes[index].points
        ):
            self.store_shapes()
            self.shape_rotated.emit()

    def _finish_rotation_handle_drag(self):
        shape = self._rotation_drag_shape
        self._rotation_drag_shape = None
        self._rotation_drag_prev_angle = None
        if self.rotating_shape:
            self._store_rotated_shape(shape)
            self.rotating_shape = False
        self.override_cursor(CURSOR_POINT)
        self.update()

    def _sync_drawing_line(self, pos, modifiers):
        if not self.drawing() or not self.current:
            return
        if self.out_off_pixmap(pos) and self.create_mode not in [
            "rectangle",
            "rotation",
            "quadrilateral",
            "cuboid",
        ]:
            pos = self.intersection_point(self.current[-1], pos)
        elif (
            self.snapping
            and len(self.current) > 1
            and self.create_mode == "polygon"
            and self.close_enough(pos, self.current[0])
        ):
            pos = self.current[0]
        elif (
            self.create_mode == "rotation"
            and len(self.current) > 0
            and self.close_enough(pos, self.current[0])
        ):
            pos = self.current[0]
        elif (
            self.create_mode == "quadrilateral"
            and len(self.current) >= 3
            and self.close_enough(pos, self.current[0])
        ):
            pos = self.current[0]
        if (
            self.create_mode in ["line", "linestrip"]
            and modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier
        ):
            pos = self._snap_line_pos(self.current[-1], pos)
        if self.create_mode in ["polygon", "linestrip", "quadrilateral"]:
            self.line[0] = self.current[-1]
            self.line[1] = pos
        elif self.create_mode == "rectangle":
            self.line.points = [self.current[0], pos]
            self.line.close()
        elif self.create_mode == "rotation":
            self.line[1] = pos
        elif self.create_mode == "circle":
            self.line.points = [self.current[0], pos]
            self.line.shape_type = "circle"
        elif self.create_mode == "line":
            self.line.points = [self.current[0], pos]
            self.line.close()
        elif self.create_mode == "cuboid":
            self.line.points = [self.current[0], pos]
            self.line.close()

    def _update_space_pan(self, point):
        if not self._space_panning:
            return False
        if not self._has_valid_pixmap():
            self._space_panning = False
            self._space_pan_prev_point = None
            self._space_pan_suppress_until_release = False
            return False
        if self._space_pan_prev_point is None:
            self._space_pan_prev_point = point
            return True

        delta = point - self._space_pan_prev_point
        self._space_pan_prev_point = point
        self.override_cursor(CURSOR_MOVE)
        self.scroll_request.emit(
            delta.x() / (self.pixmap.width() * self.scale),
            Qt.Orientation.Horizontal,
            1,
        )
        self.scroll_request.emit(
            delta.y() / (self.pixmap.height() * self.scale),
            Qt.Orientation.Vertical,
            1,
        )
        self.repaint()
        return True

    # QT Overload
    def mouseMoveEvent(self, ev):  # noqa: C901
        """Update line with last point and current coordinates"""
        if self.is_loading:
            return
        try:
            pos = self.transform_pos(ev.position())
        except AttributeError:
            return

        if self._space_panning:
            if self._left_button_pressed(ev):
                self._update_space_pan(ev.position())
                ev.accept()
                return
            self._space_panning = False
            self._space_pan_prev_point = None
            self._space_pan_suppress_until_release = False
        if self._space_pan_suppress_until_release:
            if self._left_button_pressed(ev):
                ev.accept()
                return
            self._space_pan_suppress_until_release = False
        if self._space_pressed:
            self.override_cursor(CURSOR_GRAB)
            ev.accept()
            return

        if self.is_brush_mode and self.editing():
            self._brush_mouse_move(ev, pos)
            return

        if (
            self.editing()
            and ev.modifiers() == QtCore.Qt.KeyboardModifier.AltModifier
            and self.can_erase_selected_vertices()
        ):
            self.override_cursor(self._vertex_eraser_cursor())
            self._set_vertex_eraser_tooltip()
            if QtCore.Qt.MouseButton.LeftButton & ev.buttons():
                self._vertex_erasing = True
                self.erase_selected_vertex_at(pos)
                self.repaint()
            ev.accept()
            return

        prev_hover_shape = self.h_shape
        self.prev_move_point = pos
        self.repaint()

        # Handle auto decode mode
        if (
            self.auto_decode_mode
            and self.is_auto_labeling
            and self.auto_decode_tracklet
        ):
            if self._should_trigger_auto_decode(pos):
                self.last_mouse_pos = pos
                if not self.auto_decode_timer.isActive():
                    self.auto_decode_timer.start(AUTO_DECODE_DELAY_MS)

        # Polygon drawing.
        if self.drawing():
            line_color = utils.hex_to_rgb(self.cross_line_color)
            self.line.line_color = QtGui.QColor(*line_color)
            self.line.shape_type = self.create_mode
            if self.create_mode == "cuboid":
                self.line.shape_type = "rectangle"

            if not self.current:
                self.override_cursor(CURSOR_DRAW)
                return

            if self.create_mode in ["rectangle", "cuboid"]:
                shape_width = int(abs(self.current[0].x() - pos.x()))
                shape_height = int(abs(self.current[0].y() - pos.y()))
                self.show_shape.emit(shape_height, shape_width, pos)

            color = QtGui.QColor(0, 0, 255)
            if self.out_off_pixmap(pos) and self.create_mode not in [
                "rectangle",
                "rotation",
                "quadrilateral",
                "cuboid",
            ]:
                pos = self.intersection_point(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.create_mode == "polygon"
                and self.close_enough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.override_cursor(CURSOR_POINT)
                self.current.highlight_vertex(0, Shape.NEAR_VERTEX)
            elif (
                self.create_mode == "rotation"
                and len(self.current) > 0
                and self.close_enough(pos, self.current[0])
            ):
                pos = self.current[0]
                color = self.current.line_color
                self.override_cursor(CURSOR_POINT)
                self.current.highlight_vertex(0, Shape.NEAR_VERTEX)
            elif (
                self.create_mode == "quadrilateral"
                and len(self.current) >= 3
                and self.close_enough(pos, self.current[0])
            ):
                pos = self.current[0]
                self.override_cursor(CURSOR_POINT)
                self.current.highlight_vertex(0, Shape.NEAR_VERTEX)
            else:
                self.override_cursor(CURSOR_DRAW)
            if (
                self.create_mode in ["line", "linestrip"]
                and ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
            ):
                pos = self._snap_line_pos(self.current[-1], pos)
            if self.create_mode in ["polygon", "linestrip", "quadrilateral"]:
                self.line[0] = self.current[-1]
                self.line[1] = pos
            elif self.create_mode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.create_mode == "rotation":
                self.line[1] = pos
                self.line.line_color = color
            elif self.create_mode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"
            elif self.create_mode == "line":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.create_mode == "point":
                self.line.points = [self.current[0]]
                self.line.close()
            elif self.create_mode == "cuboid":
                self.line.points = [self.current[0], pos]
                self.line.close()
            if self._brush_drawing and self.create_mode == "polygon":
                if self.snapping and self._brush_drawing_can_close(pos):
                    self.current.highlight_clear()
                    self.finalise()
                    return
                point_dist = utils.distance(pos - self.current[-1])
                if point_dist * self.scale >= self.brush_point_distance:
                    self.current.add_point(pos)
                    self.line[0] = self.current[-1]
            self.repaint()
            self.current.highlight_clear()
            return

        # Polygon copy moving.
        if QtCore.Qt.MouseButton.RightButton & ev.buttons():
            if self.selected_shapes_copy and self.prev_point:
                self.override_cursor(CURSOR_MOVE)
                self.bounded_move_shapes(self.selected_shapes_copy, pos)
                self.repaint()
            elif self.selected_shapes:
                self.selected_shapes_copy = [
                    s.copy() for s in self.selected_shapes
                ]
                self.repaint()
            return

        if self._rotation_drag_shape is not None:
            if QtCore.Qt.MouseButton.LeftButton & ev.buttons():
                self.is_move_editing = False
                self._update_rotation_handle_drag(pos, ev.modifiers())
            else:
                self._finish_rotation_handle_drag()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.MouseButton.LeftButton & ev.buttons():
            if self.selected_vertex():
                self.h_cuboid_face = None
                self.is_move_editing = False
                try:
                    self.bounded_move_vertex(pos)
                    self.repaint()
                    self.moving_shape = True
                except IndexError:
                    return
                if self.h_shape.shape_type == "rectangle":
                    p1 = self.h_shape[0]
                    p2 = self.h_shape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_height, shape_width, pos)
                elif (
                    self.h_shape.shape_type == "cuboid"
                    and len(self.h_shape) >= 4
                ):
                    p1 = self.h_shape[0]
                    p2 = self.h_shape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_height, shape_width, pos)
            elif (
                self.selected_cuboid_face()
                and self.h_shape is not None
                and self.h_shape.shape_type == "cuboid"
                and self.prev_point is not None
            ):
                self.is_move_editing = False
                offset = pos - self.prev_point
                self.move_cuboid_face_by(
                    self.h_shape, self.h_cuboid_face, offset
                )
                self.prev_point = pos
                self.repaint()
                self.moving_shape = True
                p1 = self.h_shape[0]
                p2 = self.h_shape[2]
                shape_width = int(abs(p2.x() - p1.x()))
                shape_height = int(abs(p2.y() - p1.y()))
                self.show_shape.emit(shape_height, shape_width, pos)
            elif self.selected_shapes and self.prev_point:
                self.h_cuboid_face = None
                group_shapes = self._active_group_shapes()
                if group_shapes and any(
                    shape.locked for shape in group_shapes
                ):
                    return
                self.override_cursor(CURSOR_MOVE)
                self.bounded_move_shapes(self.selected_shapes, pos)
                self.repaint()
                self.moving_shape = True
                if self.selected_shapes[-1].shape_type == "rectangle":
                    p1 = self.selected_shapes[-1][0]
                    p2 = self.selected_shapes[-1][2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_height, shape_width, pos)
                elif (
                    self.selected_shapes[-1].shape_type == "cuboid"
                    and len(self.selected_shapes[-1]) >= 4
                ):
                    p1 = self.selected_shapes[-1][0]
                    p2 = self.selected_shapes[-1][2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_height, shape_width, pos)
            else:
                if (
                    self.pixmap
                    and self.pixmap.width()
                    and self.pixmap.height()
                ):
                    self.override_cursor(CURSOR_MOVE)
                    delta = ev.position() - self.prev_pan_point
                    self.scroll_request.emit(
                        delta.x() / (self.pixmap.width() * self.scale),
                        Qt.Orientation.Horizontal,
                        1,
                    )
                    self.scroll_request.emit(
                        delta.y() / (self.pixmap.height() * self.scale),
                        Qt.Orientation.Vertical,
                        1,
                    )
                    self.repaint()
            return

        if self.editing() and self.is_move_editing:
            self.override_cursor(CURSOR_MOVE)
            if self.selected_vertex():
                self.h_cuboid_face = None
                try:
                    self.bounded_move_vertex(pos)
                    self.repaint()
                    self.moving_shape = True
                except IndexError:
                    return
                if self.h_shape.shape_type == "rectangle":
                    p1 = self.h_shape[0]
                    p2 = self.h_shape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_height, shape_width, pos)
                elif (
                    self.h_shape.shape_type == "cuboid"
                    and len(self.h_shape) >= 4
                ):
                    p1 = self.h_shape[0]
                    p2 = self.h_shape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_height, shape_width, pos)
            elif (
                self.selected_cuboid_face()
                and self.h_shape is not None
                and self.h_shape.shape_type == "cuboid"
                and self.prev_point is not None
            ):
                offset = pos - self.prev_point
                self.move_cuboid_face_by(
                    self.h_shape, self.h_cuboid_face, offset
                )
                self.prev_point = pos
                self.repaint()
                self.moving_shape = True
                p1 = self.h_shape[0]
                p2 = self.h_shape[2]
                shape_width = int(abs(p2.x() - p1.x()))
                shape_height = int(abs(p2.y() - p1.y()))
                self.show_shape.emit(shape_height, shape_width, pos)
            else:
                self.is_move_editing = False

            return

        self.show_shape.emit(-1, -1, pos)

        self._hovered_group_id = None

        rotation_handle_shape = self._rotation_handle_shape_at(pos)
        if rotation_handle_shape is not None:
            self._set_rotation_handle_hover(rotation_handle_shape)
            self.vertex_selected.emit(False)
            if prev_hover_shape != self.h_shape:
                self.shape_hover_changed.emit()
            return
        if self.h_rotation_shape is not None:
            self.prev_h_rotation_shape = self.h_rotation_shape
            self.h_rotation_shape = None

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        # self.setToolTip(self.tr("Image"))
        for shape in self._shape_hit_candidates(pos):
            if (
                not shape.locked
                and shape.shape_type == "cuboid"
                and len(shape.points) == 8
            ):
                index = self.nearest_cuboid_control(
                    shape, pos, self.epsilon / self.scale
                )
                if index is not None:
                    if self.selected_vertex():
                        self.h_shape.highlight_clear()
                    self.prev_h_vertex = self.h_vertex
                    self.h_vertex = index
                    self.prev_h_shape = self.h_shape = shape
                    self.prev_h_edge = self.h_edge
                    self.h_edge = None
                    self.prev_h_cuboid_face = self.h_cuboid_face
                    self.h_cuboid_face = None
                    shape.highlight_vertex(index, shape.MOVE_VERTEX)
                    self.override_cursor(CURSOR_POINT)
                    if index in CUBOID_BACK_EDGE_CENTER_INDICES:
                        self.setToolTip(
                            self.tr(
                                "Click & drag to adjust cuboid depth of shape '%s'"
                            )
                            % shape.label
                        )
                    elif index in [4, 5, 6, 7]:
                        self.setToolTip(
                            self.tr(
                                "Click & drag to adjust rear edge of cuboid shape '%s'"
                            )
                            % shape.label
                        )
                    else:
                        self.setToolTip(
                            self.tr("Click & drag to move point of shape '%s'")
                            % shape.label
                        )
                    self.setStatusTip(self.toolTip())
                    self.update()
                    break
                front_path = self.cuboid_face_path(shape, CUBOID_FACE_FRONT)
                if front_path is not None and front_path.contains(pos):
                    if self.selected_vertex():
                        self.h_shape.highlight_clear()
                    self.prev_h_vertex = self.h_vertex
                    self.h_vertex = None
                    self.prev_h_shape = self.h_shape = shape
                    self.prev_h_edge = self.h_edge
                    self.h_edge = None
                    self.prev_h_cuboid_face = self.h_cuboid_face
                    self.h_cuboid_face = None
                    self.setToolTip(
                        self.tr("Click & drag to move shape '%s'")
                        % shape.label
                    )
                    self.setStatusTip(self.toolTip())
                    self.override_cursor(CURSOR_GRAB)
                    self.update()
                    break
                face_name = self.cuboid_face_hit_test(shape, pos)
                if face_name and face_name != CUBOID_FACE_FRONT:
                    if self.selected_vertex():
                        self.h_shape.highlight_clear()
                    self.prev_h_vertex = self.h_vertex
                    self.h_vertex = None
                    self.prev_h_shape = self.h_shape = shape
                    self.prev_h_edge = self.h_edge
                    self.h_edge = None
                    self.prev_h_cuboid_face = self.h_cuboid_face
                    self.h_cuboid_face = face_name
                    self.override_cursor(CURSOR_POINT)
                    self.setToolTip(
                        self.tr(
                            "Click & drag to adjust cuboid %s face of shape '%s'"
                        )
                        % (face_name, shape.label)
                    )
                    self.setStatusTip(self.toolTip())
                    self.update()
                    break
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = (
                None
                if shape.locked
                else shape.nearest_vertex(pos, self.epsilon / self.scale)
            )
            index_edge = (
                None
                if shape.locked
                else shape.nearest_edge(pos, self.epsilon / self.scale)
            )
            if index is not None:
                if self.selected_vertex():
                    self.h_shape.highlight_clear()
                self.prev_h_vertex = self.h_vertex = index
                self.prev_h_shape = self.h_shape = shape
                self.prev_h_edge = self.h_edge
                self.h_edge = None
                self.prev_h_cuboid_face = self.h_cuboid_face
                self.h_cuboid_face = None
                shape.highlight_vertex(index, shape.MOVE_VERTEX)
                self.override_cursor(CURSOR_POINT)
                self.setToolTip(
                    self.tr("Click & drag to move point of shape '%s'")
                    % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.update()
                break
            if (
                index_edge is not None
                and shape.can_add_point()
                and shape.shape_type != "quadrilateral"
            ):
                if self.selected_vertex():
                    self.h_shape.highlight_clear()
                self.prev_h_vertex = self.h_vertex
                self.h_vertex = None
                self.prev_h_shape = self.h_shape = shape
                self.prev_h_edge = self.h_edge = index_edge
                self.prev_h_cuboid_face = self.h_cuboid_face
                self.h_cuboid_face = None
                self.override_cursor(CURSOR_POINT)
                self.setToolTip(
                    self.tr("Click to create point of shape '%s'")
                    % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.update()
                break
            shape_hit = False
            if shape.shape_type in ["point", "line", "linestrip"]:
                nearest_index = shape.nearest_vertex(
                    pos, self.epsilon * 3 / self.scale
                )
                if nearest_index is not None:
                    shape_hit = True
            elif shape.shape_type == "cuboid" and len(shape.points) == 8:
                front_path = self.cuboid_face_path(shape, CUBOID_FACE_FRONT)
                shape_hit = front_path is not None and front_path.contains(pos)
            elif len(shape.points) > 1 and shape.contains_point(pos):
                shape_hit = True

            if shape_hit:
                if self.selected_vertex():
                    self.h_shape.highlight_clear()
                self.prev_h_vertex = self.h_vertex
                self.h_vertex = None
                self.prev_h_shape = self.h_shape = shape
                self.prev_h_edge = self.h_edge
                self.h_edge = None
                self.prev_h_cuboid_face = self.h_cuboid_face
                self.h_cuboid_face = None
                if shape.locked:
                    self.setToolTip(self.tr("Locked shape '%s'") % shape.label)
                elif shape.group_id and shape.shape_type == "rectangle":
                    tooltip_text = "Click & drag to move shape '{label} {group_id}'".format(
                        label=shape.label, group_id=shape.group_id
                    )
                    self.setToolTip(self.tr(tooltip_text))
                else:
                    self.setToolTip(
                        self.tr("Click & drag to move shape '%s'")
                        % shape.label
                    )
                self.setStatusTip(self.toolTip())
                self.override_cursor(
                    CURSOR_DEFAULT if shape.locked else CURSOR_GRAB
                )
                # [Feature] Automatically highlight shape when the mouse is moved inside it
                if self.h_shape_is_hovered:
                    group_mode = (
                        ev.modifiers()
                        == QtCore.Qt.KeyboardModifier.ControlModifier
                    )
                    self.select_shape_point(
                        pos, multiple_selection_mode=group_mode
                    )
                self.update()

                if shape.shape_type == "rectangle":
                    p1 = self.h_shape[0]
                    p2 = self.h_shape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_height, shape_width, pos)
                elif shape.shape_type == "cuboid" and len(self.h_shape) >= 4:
                    p1 = self.h_shape[0]
                    p2 = self.h_shape[2]
                    shape_width = int(abs(p2.x() - p1.x()))
                    shape_height = int(abs(p2.y() - p1.y()))
                    self.show_shape.emit(shape_height, shape_width, pos)
                break
        else:  # Nothing found, clear highlights, reset state.
            group_id = self._group_at_point(pos)
            if group_id is not None:
                self.un_highlight()
                self._hover_group(group_id)
                return
            self.un_highlight()
            self.override_cursor(CURSOR_DEFAULT)
            self.setToolTip("")
            self.setStatusTip("")
        self.vertex_selected.emit(self.h_vertex is not None)

        if prev_hover_shape != self.h_shape:
            self.shape_hover_changed.emit()

    def add_point_to_edge(self):
        """Add a point to current shape"""
        shape = self.prev_h_shape
        index = self.prev_h_edge
        point = self.prev_move_point
        if shape is None or shape.locked or index is None or point is None:
            return
        shape.insert_point(index, point)
        shape.highlight_vertex(index, shape.MOVE_VERTEX)
        self.h_shape = shape
        self.h_vertex = index
        self.h_edge = None
        self.moving_shape = True
        self._pending_edge_point = (shape, index)

    def _undo_pending_edge_point(self):
        """Undo the edge point inserted by the preceding mousePressEvent"""
        if self._pending_edge_point is None:
            return
        shape, index = self._pending_edge_point
        self._pending_edge_point = None
        shape.remove_point(index)
        shape.highlight_clear()
        if len(self.shapes_backups) >= 2 and shape in self.shapes:
            self.shapes_backups.pop()

    def remove_selected_point(self):
        """Remove a point from current shape"""
        shape = self.prev_h_shape
        index = self.prev_h_vertex
        if shape is None or shape.locked or index is None:
            return
        shape.remove_point(index)
        shape.highlight_clear()
        self.h_shape = shape
        self.prev_h_vertex = None
        self.moving_shape = True  # Save changes

    def on_auto_decode_timeout(self):
        """Handle auto decode timeout"""
        if (
            not self.auto_decode_mode
            or self.auto_labeling_mode.shape_type != AutoLabelingMode.POINT
        ):
            return

        flag = -1
        if self.auto_labeling_mode.edit_mode == AutoLabelingMode.ADD:
            flag = 1
        elif self.auto_labeling_mode.edit_mode == AutoLabelingMode.REMOVE:
            flag = 0
        if flag == -1:
            return

        if self.auto_decode_mode and self.last_mouse_pos:
            if len(self.auto_decode_tracklet) >= MAX_AUTO_DECODE_MARKS:
                self.auto_decode_tracklet.pop(0)

            marks = {
                "type": "point",
                "data": [
                    int(self.last_mouse_pos.x()),
                    int(self.last_mouse_pos.y()),
                ],
                "label": flag,
            }
            self.auto_decode_tracklet.append(marks)
            self.auto_decode_requested.emit(self.auto_decode_tracklet)

    # QT Overload
    def mousePressEvent(self, ev):  # noqa: C901
        """Mouse press event"""
        if self.is_loading:
            return
        self._pending_edge_point = None
        pos = self.transform_pos(ev.position())

        if self.is_brush_mode and self._brush_mouse_press(ev, pos):
            return

        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if (
                self._space_pan_suppress_until_release
                and not self._space_pressed
            ):
                self._space_pan_suppress_until_release = False
            if self._space_pressed and self._start_space_pan(ev.position()):
                ev.accept()
                return
            if self.drawing():
                if self.current:
                    self._sync_drawing_line(pos, ev.modifiers())
                    # Add point to existing shape.
                    if self.create_mode == "polygon":
                        self.current.add_point(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.is_closed():
                            self.finalise()
                    elif self.create_mode in ["circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.create_mode == "rectangle":
                        if self.current.reach_max_points() is False:
                            init_pos = self.current[0]
                            min_x = init_pos.x()
                            min_y = init_pos.y()
                            target_pos = self.line[1]
                            max_x = target_pos.x()
                            max_y = target_pos.y()
                            self.current.add_point(
                                QtCore.QPointF(max_x, min_y)
                            )
                            self.current.add_point(target_pos)
                            self.current.add_point(
                                QtCore.QPointF(min_x, max_y)
                            )
                            self.finalise()
                    elif self.create_mode == "cuboid":
                        if len(self.current.points) == 1:
                            init_pos = self.current[0]
                            target_pos = self.line[1]
                            front_points = self.make_rectangle_points(
                                init_pos, target_pos
                            )
                            if (
                                abs(front_points[2].x() - front_points[0].x())
                                < 1
                                or abs(
                                    front_points[2].y() - front_points[0].y()
                                )
                                < 1
                            ):
                                return
                            depth_vector = QtCore.QPointF(
                                self.cuboid_default_depth_vector[0],
                                self.cuboid_default_depth_vector[1],
                            )
                            self.set_cuboid_points(
                                self.current, front_points, depth_vector
                            )
                            self.finalise()
                    elif self.create_mode == "rotation":
                        initPos = self.current[0]
                        minX = initPos.x()
                        minY = initPos.y()
                        targetPos = self.line[1]
                        maxX = targetPos.x()
                        maxY = targetPos.y()
                        self.current.add_point(QtCore.QPointF(maxX, minY))
                        self.current.add_point(targetPos)
                        self.current.add_point(QtCore.QPointF(minX, maxY))
                        self.current.add_point(initPos)
                        self.line[0] = self.current[-1]
                        if self.current.is_closed():
                            self.finalise()
                    elif self.create_mode == "quadrilateral":
                        self.current.add_point(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.is_closed():
                            self.finalise()
                    elif self.create_mode == "linestrip":
                        self.current.add_point(self.line[1])
                        self.line[0] = self.current[-1]
                        if (
                            ev.modifiers()
                            == QtCore.Qt.KeyboardModifier.ControlModifier
                        ):
                            self.finalise()
                    # [Feature] support for automatically switching to editing mode
                    # when the cursor moves over an object
                    if (
                        self.create_mode
                        in [
                            "rectangle",
                            "rotation",
                            "quadrilateral",
                            "cuboid",
                            "circle",
                            "line",
                            "point",
                        ]
                        and not self.is_auto_labeling
                        and not self.current
                    ):
                        self.prev_pan_point = ev.position()
                        self.mode_changed.emit()
                elif not self.out_off_pixmap(pos):
                    # Handle auto decode mode first click
                    if self.auto_decode_mode and self.is_auto_labeling:
                        if (
                            self.auto_labeling_mode.shape_type
                            == AutoLabelingMode.POINT
                        ):
                            self.last_mouse_pos = pos
                            self.on_auto_decode_timeout()
                            return

                    # Create new shape.
                    self.current = Shape(shape_type=self.create_mode)
                    self.current.add_point(pos)
                    if self.create_mode == "point":
                        self.finalise()
                    else:
                        if self.create_mode == "circle":
                            self.current.shape_type = "circle"
                        self.line.points = [pos, pos]
                        self.set_hiding()
                        self.drawing_polygon.emit(True)
                        self.update()
                elif (
                    self.out_off_pixmap(pos)
                    and self.create_mode == "linestrip"
                ):
                    w = self.pixmap.width()
                    h = self.pixmap.height()
                    if w > 0 and h > 0:
                        pos = QtCore.QPointF(
                            min(max(pos.x(), 0), w - 1),
                            min(max(pos.y(), 0), h - 1),
                        )
                        self.current = Shape(shape_type=self.create_mode)
                        self.current.add_point(pos)
                        self.line.points = [pos, pos]
                        self.set_hiding()
                        self.drawing_polygon.emit(True)
                        self.update()
                elif self.out_off_pixmap(pos) and self.create_mode in [
                    "rectangle",
                    "rotation",
                    "quadrilateral",
                    "cuboid",
                ]:
                    # Create new shape.
                    self.current = Shape(shape_type=self.create_mode)
                    self.current.add_point(pos)
                    self.line.points = [pos, pos]
                    self.set_hiding()
                    self.drawing_polygon.emit(True)
                    self.update()
            elif self.editing():
                if (
                    ev.modifiers() == QtCore.Qt.KeyboardModifier.AltModifier
                    and self.can_erase_selected_vertices()
                ):
                    self._vertex_erasing = True
                    self.override_cursor(self._vertex_eraser_cursor())
                    self._set_vertex_eraser_tooltip()
                    self.erase_selected_vertex_at(pos)
                    self.prev_point = pos
                    self.prev_pan_point = ev.position()
                    self.repaint()
                    ev.accept()
                    return
                rotation_handle_shape = self._rotation_handle_shape_at(pos)
                if rotation_handle_shape is not None:
                    group_mode = (
                        ev.modifiers()
                        == QtCore.Qt.KeyboardModifier.ControlModifier
                    )
                    self._start_rotation_handle_drag(
                        rotation_handle_shape,
                        pos,
                        group_mode,
                        ev.modifiers(),
                    )
                    self.prev_point = pos
                    self.prev_pan_point = ev.position()
                    self.repaint()
                    ev.accept()
                    return
                if self.selected_edge():
                    self.add_point_to_edge()
                elif (
                    self.selected_vertex()
                    and ev.modifiers()
                    == QtCore.Qt.KeyboardModifier.ShiftModifier
                    and self.h_shape.shape_type
                    not in [
                        "rectangle",
                        "rotation",
                        "quadrilateral",
                        "line",
                        "cuboid",
                    ]
                ):
                    # Delete point if: left-click + SHIFT on a point
                    # (quadrilateral must keep exactly 4 points)
                    self.remove_selected_point()

                if (
                    self.selected_vertex()
                    and ev.modifiers()
                    != QtCore.Qt.KeyboardModifier.ShiftModifier
                ):
                    self.is_move_editing = not self.is_move_editing
                    if self.is_move_editing:
                        self.override_cursor(CURSOR_MOVE)
                    else:
                        self.override_cursor(CURSOR_POINT)

                group_mode = (
                    ev.modifiers()
                    == QtCore.Qt.KeyboardModifier.ControlModifier
                )
                self.select_shape_point(
                    pos, multiple_selection_mode=group_mode
                )
                self.prev_point = pos
                self.prev_pan_point = ev.position()
                self.repaint()
        elif (
            ev.button() == QtCore.Qt.MouseButton.RightButton and self.editing()
        ):
            group_mode = (
                ev.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier
            )
            if not self.selected_shapes or (
                self.h_shape is not None
                and self.h_shape not in self.selected_shapes
            ):
                self.select_shape_point(
                    pos, multiple_selection_mode=group_mode
                )
                self.repaint()
            self.prev_point = pos

    # QT Overload
    def mouseReleaseEvent(self, ev):
        """Mouse release event"""
        if self.is_loading:
            return

        if ev.button() == QtCore.Qt.MouseButton.LeftButton and (
            self._space_panning or self._space_pan_suppress_until_release
        ):
            self._space_panning = False
            self._space_pan_prev_point = None
            self._space_pan_suppress_until_release = False
            self._restore_space_pan_cursor()
            ev.accept()
            return

        if self.is_brush_mode and self._brush_mouse_release(ev):
            return

        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            menu = self.menus[len(self.selected_shapes_copy) > 0]
            self.restore_cursor()
            if (
                not menu.exec(self.mapToGlobal(ev.position().toPoint()))
                and self.selected_shapes_copy
            ):
                # Cancel the move by deleting the shadow copy.
                self.selected_shapes_copy = []
                self.repaint()
        elif ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if self._rotation_drag_shape is not None:
                self._finish_rotation_handle_drag()
                ev.accept()
                return
            if self._vertex_erasing:
                self._vertex_erasing = False
                self.store_moving_shape()
                ev.accept()
                return
            if self.editing():
                if (
                    self.h_shape is not None
                    and self.h_shape_is_selected
                    and not self.moving_shape
                ):
                    self.selection_changed.emit(
                        [x for x in self.selected_shapes if x != self.h_shape]
                    )

        self.store_moving_shape()

    def end_move(self, copy):
        """End of move"""
        assert self.selected_shapes and self.selected_shapes_copy
        assert len(self.selected_shapes_copy) == len(self.selected_shapes)
        if copy:
            for i, shape in enumerate(self.selected_shapes_copy):
                self.shapes.append(shape)
                self.selected_shapes[i].selected = False
                self.selected_shapes[i] = shape
        else:
            for i, shape in enumerate(self.selected_shapes_copy):
                self.selected_shapes[i].points = shape.points
        self.selected_shapes_copy = []
        self.repaint()
        self.store_shapes()
        return True

    def hide_background_shapes(self, value):
        """Set hide background - hide other shapes when some shapes are selected"""
        self.hide_backround = value
        if self.selected_shapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.set_hiding(True)
            self.update()

    def set_hiding(self, enable=True):
        """Set background hiding"""
        self._hide_backround = self.hide_backround if enable else False

    def can_close_shape(self):
        """Check if a shape can be closed (number of points > 2)"""
        return self.drawing() and self.current and len(self.current) > 2

    # QT Overload
    def mouseDoubleClickEvent(self, ev):
        """Mouse double click event"""
        if self.is_loading:
            return
        if (
            self._space_pressed
            or self._space_panning
            or self._space_pan_suppress_until_release
        ):
            ev.accept()
            return

        # Handle auto decode mode double click to finish
        if (
            self.auto_decode_mode
            and self.is_auto_labeling
            and self.auto_decode_tracklet
        ):
            self.auto_decode_finish_requested.emit()
            return

        if self.editing() and self.double_click_edit_label:
            pos = self.transform_pos(ev.position())
            for shape in self._shape_hit_candidates(pos):
                self._undo_pending_edge_point()
                if shape not in self.selected_shapes:
                    self.selection_changed.emit([shape])
                self.h_shape_is_selected = False
                self.edit_label_requested.emit()
                return

        # For polygon/quadrilateral the mousePress handler adds a spurious
        # duplicate point before this handler fires, so we pop it first.
        # For linestrip the press-added point IS the intended final point,
        # so we keep it and finalize directly.
        if self.double_click == "close" and self.can_close_shape():
            if self.create_mode == "linestrip":
                self.finalise()
            elif len(self.current) > 3:
                self.current.pop_point()
                self.finalise()

    def select_shapes(self, shapes):
        """Select some shapes"""
        self._selected_group_id = None
        self.set_hiding()
        self.selection_changed.emit(shapes)
        self.update()

    def select_shape_point(self, point, multiple_selection_mode):  # noqa: C901
        """Select the first shape created which contains this point."""
        self._selected_group_id = None
        if self.selected_vertex():  # A vertex is marked for selection.
            index, shape = self.h_vertex, self.h_shape
            if shape.shape_type == "cuboid":
                self.set_hiding()
                if shape not in self.selected_shapes:
                    if multiple_selection_mode:
                        self.selection_changed.emit(
                            self.selected_shapes + [shape]
                        )
                    else:
                        self.selection_changed.emit([shape])
                    self.h_shape_is_selected = False
                else:
                    self.h_shape_is_selected = True
                self.calculate_offsets(point)
                return
            shape.highlight_vertex(index, shape.MOVE_VERTEX)
            if shape.shape_type == "rotation":
                self.set_hiding()
                if shape not in self.selected_shapes:
                    if multiple_selection_mode:
                        self.selection_changed.emit(
                            self.selected_shapes + [shape]
                        )
                    else:
                        self.selection_changed.emit([shape])
                    self.h_shape_is_selected = False
                else:
                    self.h_shape_is_selected = True
                self.calculate_offsets(point)
                return
            self.set_hiding()
            if shape not in self.selected_shapes:
                if multiple_selection_mode:
                    self.selection_changed.emit(self.selected_shapes + [shape])
                else:
                    self.selection_changed.emit([shape])
            self.h_shape_is_selected = False
            self.calculate_offsets(point)
            return
        elif (
            self.selected_cuboid_face()
            and self.h_shape is not None
            and self.h_shape.shape_type == "cuboid"
        ):
            shape = self.h_shape
            self.set_hiding()
            if shape not in self.selected_shapes:
                if multiple_selection_mode:
                    self.selection_changed.emit(self.selected_shapes + [shape])
                else:
                    self.selection_changed.emit([shape])
                self.h_shape_is_selected = False
            else:
                self.h_shape_is_selected = True
            self.calculate_offsets(point)
            return

        else:
            for shape in self._shape_hit_candidates(point):
                self._selected_group_id = None
                self.set_hiding()
                if shape not in self.selected_shapes:
                    if multiple_selection_mode:
                        self.selection_changed.emit(
                            self.selected_shapes + [shape]
                        )
                    else:
                        self.selection_changed.emit([shape])
                    self.h_shape_is_selected = False
                else:
                    self.h_shape_is_selected = True
                self.calculate_offsets(point)
                return
            self._select_group_at_point_or_deselect(point)

    def _select_group_at_point_or_deselect(self, point):
        group_id = self._group_at_point(point)
        if group_id is None:
            self.deselect_shape()
            return
        self._select_group(group_id, point)

    def _grouped_shapes(self):
        grouped_shapes = {}
        if not self.show_groups:
            return grouped_shapes
        for shape in self.shapes:
            if shape.group_id is None:
                continue
            grouped_shapes.setdefault(shape.group_id, []).append(shape)
        return grouped_shapes

    def _group_rect(self, shapes):
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        for shape in shapes:
            rect = shape.bounding_rect()
            if shape.shape_type == "point":
                point = shape.points[0]
                min_x = min(min_x, point.x())
                min_y = min(min_y, point.y())
                max_x = max(max_x, point.x())
                max_y = max(max_y, point.y())
            else:
                min_x = min(min_x, rect.left())
                min_y = min(min_y, rect.top())
                max_x = max(max_x, rect.right())
                max_y = max(max_y, rect.bottom())
        return QtCore.QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def _group_label(self, group_id, shape_count):
        return f"G{group_id} · S{shape_count}"

    def _group_label_font(self):
        return QtGui.QFont(
            "Arial", int(max(6.0, int(round(8.0 / self.scale))))
        )

    def _group_label_rect(self, group_id, shape_count, group_rect):
        font = self._group_label_font()
        metrics = QtGui.QFontMetricsF(font)
        padding_x = 5.0 / self.scale
        padding_y = 2.0 / self.scale
        text = self._group_label(group_id, shape_count)
        width = metrics.horizontalAdvance(text) + 2 * padding_x
        height = metrics.height() + 2 * padding_y
        top = group_rect.top() - height
        return QtCore.QRectF(group_rect.left(), top, width, height)

    def _group_at_point(self, point):
        candidates = []
        tolerance = max(0.5, 6.0 / self.scale)
        for group_id, group_shapes in self._grouped_shapes().items():
            visible_shapes = [shape for shape in group_shapes if shape.visible]
            if not visible_shapes:
                continue
            rect = self._group_rect(visible_shapes)
            shape_count = len(group_shapes)
            label_rect = self._group_label_rect(
                group_id, shape_count, rect
            ).adjusted(-tolerance, -tolerance, tolerance, tolerance)
            outer = rect.adjusted(-tolerance, -tolerance, tolerance, tolerance)
            inner = rect.adjusted(tolerance, tolerance, -tolerance, -tolerance)
            on_boundary = outer.contains(point) and not inner.contains(point)
            if label_rect.contains(point) or on_boundary:
                candidates.append((rect.width() * rect.height(), group_id))
            elif rect.contains(point):
                candidates.append((rect.width() * rect.height(), group_id))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _group_shapes(self, group_id):
        return [shape for shape in self.shapes if shape.group_id == group_id]

    def _active_group_shapes(self):
        if not self.show_groups or self._selected_group_id is None:
            return []
        shapes = self._group_shapes(self._selected_group_id)
        if len(shapes) != len(self.selected_shapes):
            return []
        if any(shape not in self.selected_shapes for shape in shapes):
            return []
        return shapes

    def _select_group(self, group_id, point):
        shapes = self._group_shapes(group_id)
        if not shapes:
            return
        self._selected_group_id = group_id
        self._hovered_group_id = group_id
        self.h_shape = None
        self.h_vertex = None
        self.h_edge = None
        self.h_cuboid_face = None
        self.h_shape_is_selected = False
        self.set_hiding()
        self.selection_changed.emit(shapes)
        self.calculate_offsets(point)
        self.prev_point = point
        self.override_cursor(
            CURSOR_DEFAULT
            if any(shape.locked for shape in shapes)
            else CURSOR_GRAB
        )
        self.update()

    def _hover_group(self, group_id):
        shapes = self._group_shapes(group_id)
        self._hovered_group_id = group_id
        locked = any(shape.locked for shape in shapes)
        tooltip = self.tr("Group %s · %d shapes") % (group_id, len(shapes))
        if locked:
            tooltip = self.tr("Locked %s") % tooltip
        else:
            tooltip = self.tr("Click & drag to move %s") % tooltip
        self.setToolTip(tooltip)
        self.setStatusTip(tooltip)
        self.override_cursor(CURSOR_DEFAULT if locked else CURSOR_GRAB)
        self.update()

    def calculate_offsets(self, point):
        """Calculate offsets of a point to pixmap borders"""
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selected_shapes:
            rect = s.bounding_rect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def get_adjoint_points(self, theta, p3, p1, index):
        a1 = math.tan(theta)
        if a1 == 0:
            if index % 2 == 0:
                p2 = QtCore.QPointF(p3.x(), p1.y())
                p4 = QtCore.QPointF(p1.x(), p3.y())
            else:
                p4 = QtCore.QPointF(p3.x(), p1.y())
                p2 = QtCore.QPointF(p1.x(), p3.y())
        else:
            a3 = a1
            a2 = -1 / a1
            a4 = -1 / a1
            b1 = p1.y() - a1 * p1.x()
            b2 = p1.y() - a2 * p1.x()
            b3 = p3.y() - a1 * p3.x()
            b4 = p3.y() - a2 * p3.x()

            if index % 2 == 0:
                p2 = self.get_cross_point(a1, b1, a4, b4)
                p4 = self.get_cross_point(a2, b2, a3, b3)
            else:
                p4 = self.get_cross_point(a1, b1, a4, b4)
                p2 = self.get_cross_point(a2, b2, a3, b3)

        return p2, p3, p4

    @staticmethod
    def get_cross_point(a1, b1, a2, b2):
        x = (b2 - b1) / (a1 - a2)
        y = (a1 * b2 - a2 * b1) / (a1 - a2)
        return QtCore.QPointF(x, y)

    @staticmethod
    def make_rectangle_points(pt1, pt2):
        min_x = min(pt1.x(), pt2.x())
        min_y = min(pt1.y(), pt2.y())
        max_x = max(pt1.x(), pt2.x())
        max_y = max(pt1.y(), pt2.y())
        return [
            QtCore.QPointF(min_x, min_y),
            QtCore.QPointF(max_x, min_y),
            QtCore.QPointF(max_x, max_y),
            QtCore.QPointF(min_x, max_y),
        ]

    def get_cuboid_depth_vector(self, shape):
        depth_vector = shape.get_cuboid_depth_vector()
        return QtCore.QPointF(depth_vector[0], depth_vector[1])

    def cuboid_constraint_margin(self):
        return max(2.0, self.cuboid_min_depth * 0.2)

    def normalize_cuboid_depth(self, depth_vector):
        depth = math.hypot(depth_vector.x(), depth_vector.y())
        if depth >= self.cuboid_min_depth:
            return depth_vector
        if depth <= 1e-6:
            default_depth = QtCore.QPointF(
                self.cuboid_default_depth_vector[0],
                self.cuboid_default_depth_vector[1],
            )
            default_len = math.hypot(default_depth.x(), default_depth.y())
            if default_len <= 1e-6:
                return QtCore.QPointF(self.cuboid_min_depth, 0.0)
            scale = self.cuboid_min_depth / default_len
            return QtCore.QPointF(
                default_depth.x() * scale,
                default_depth.y() * scale,
            )
        scale = self.cuboid_min_depth / depth
        return QtCore.QPointF(
            depth_vector.x() * scale, depth_vector.y() * scale
        )

    def make_cuboid_points(self, front_points, depth_vector):
        depth_vector = self.normalize_cuboid_depth(depth_vector)
        back_points = [p + depth_vector for p in front_points]
        return list(front_points) + back_points

    def set_cuboid_points(
        self, shape, front_points, depth_vector, source="manual"
    ):
        shape.points = self.make_cuboid_points(front_points, depth_vector)
        shape.set_cuboid_depth_vector(
            [depth_vector.x(), depth_vector.y()],
            mode="from_rectangle",
            source=source,
        )

    def set_cuboid_raw_points(self, shape, points):
        shape.points = [QtCore.QPointF(p) for p in points]
        shape.sync_cuboid_depth_vector()

    @staticmethod
    def get_cuboid_back_offsets(shape):
        if shape.shape_type != "cuboid" or len(shape.points) != 8:
            return []
        return [shape.points[i + 4] - shape.points[i] for i in range(4)]

    def set_cuboid_front_with_offsets(self, shape, front_points, offsets):
        if len(front_points) != 4 or len(offsets) != 4:
            return
        points = [QtCore.QPointF(p) for p in front_points]
        points.extend([front_points[i] + offsets[i] for i in range(4)])
        self.set_cuboid_raw_points(shape, points)

    @staticmethod
    def _vector_dot(v1, v2):
        return v1.x() * v2.x() + v1.y() * v2.y()

    @staticmethod
    def _vector_length(v):
        return math.hypot(v.x(), v.y())

    @staticmethod
    def _vector_scale(v, s):
        return QtCore.QPointF(v.x() * s, v.y() * s)

    @staticmethod
    def _solve_vector_basis(target, basis_u, basis_v):
        det = basis_u.x() * basis_v.y() - basis_u.y() * basis_v.x()
        if abs(det) <= 1e-6:
            return None
        coeff_u = (target.x() * basis_v.y() - target.y() * basis_v.x()) / det
        coeff_v = (basis_u.x() * target.y() - basis_u.y() * target.x()) / det
        return coeff_u, coeff_v

    @staticmethod
    def get_mid_point(p1, p2):
        return QtCore.QPointF(
            (p1.x() + p2.x()) / 2.0,
            (p1.y() + p2.y()) / 2.0,
        )

    def cuboid_control_point(self, shape, index):
        if shape.shape_type != "cuboid" or len(shape.points) != 8:
            return None
        return shape.get_cuboid_control_point(index)

    def cuboid_visible_control_indices(self, shape):
        if shape.shape_type != "cuboid" or len(shape.points) != 8:
            return []
        return shape.get_cuboid_visible_control_indices()

    def nearest_cuboid_control(self, shape, pos, epsilon):
        min_distance = float("inf")
        nearest_index = None
        for index in self.cuboid_visible_control_indices(shape):
            control_point = self.cuboid_control_point(shape, index)
            if control_point is None:
                continue
            dist = utils.distance(control_point - pos)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                nearest_index = index
        return nearest_index

    @staticmethod
    def cuboid_face_vertex_indices(face_name):
        mapping = {
            CUBOID_FACE_FRONT: [0, 1, 2, 3],
            CUBOID_FACE_RIGHT: [1, 2, 6, 5],
            CUBOID_FACE_LEFT: [0, 4, 7, 3],
            CUBOID_FACE_TOP: [0, 1, 5, 4],
            CUBOID_FACE_BOTTOM: [3, 2, 6, 7],
            CUBOID_FACE_BACK: [4, 5, 6, 7],
        }
        return mapping.get(face_name, [])

    def cuboid_face_path(self, shape, face_name):
        face_indices = self.cuboid_face_vertex_indices(face_name)
        if shape.shape_type != "cuboid" or len(shape.points) != 8:
            return None
        if len(face_indices) != 4:
            return None
        path = QtGui.QPainterPath()
        points = [shape.points[i] for i in face_indices]
        path.moveTo(points[0])
        for point in points[1:]:
            path.lineTo(point)
        path.closeSubpath()
        return path

    def cuboid_face_hit_test(self, shape, pos):
        if shape.shape_type != "cuboid" or len(shape.points) != 8:
            return None
        depth_vector = self.get_cuboid_depth_vector(shape)
        horizontal_faces = [CUBOID_FACE_RIGHT, CUBOID_FACE_LEFT]
        if depth_vector.x() < 0:
            horizontal_faces = [CUBOID_FACE_LEFT, CUBOID_FACE_RIGHT]
        face_order = horizontal_faces + [CUBOID_FACE_BACK]
        for face_name in face_order:
            face_path = self.cuboid_face_path(shape, face_name)
            if face_path is not None and face_path.contains(pos):
                return face_name
        return None

    def adjust_cuboid_visible_back_vertex(self, shape, index, pos):
        if shape.shape_type != "cuboid" or len(shape.points) != 8:
            return
        visible_rear = shape.get_cuboid_visible_rear_edge_indices()
        if len(visible_rear) != 2 or index not in visible_rear:
            return
        top_index, bottom_index = visible_rear
        points = [QtCore.QPointF(p) for p in shape.points]
        margin = self.cuboid_constraint_margin()
        top_indices = [0, 1, 4, 5]
        bottom_indices = [2, 3, 6, 7]
        if index == top_index:
            dy = pos.y() - points[top_index].y()
            max_dy = min(
                points[bottom_i].y() - margin - points[top_i].y()
                for top_i, bottom_i in zip(top_indices, bottom_indices)
            )
            dy = min(dy, max_dy)
            for top_i in top_indices:
                points[top_i].setY(points[top_i].y() + dy)
        else:
            dy = pos.y() - points[bottom_index].y()
            min_dy = max(
                points[top_i].y() + margin - points[bottom_i].y()
                for top_i, bottom_i in zip(top_indices, bottom_indices)
            )
            dy = max(dy, min_dy)
            for bottom_i in bottom_indices:
                points[bottom_i].setY(points[bottom_i].y() + dy)
        self.set_cuboid_raw_points(shape, points)

    def adjust_cuboid_front_vertex(self, shape, index, pos):
        if shape.shape_type != "cuboid" or len(shape.points) != 8:
            return
        if index not in [0, 1, 2, 3]:
            return
        front_points = [QtCore.QPointF(p) for p in shape.points[:4]]
        offsets = self.get_cuboid_back_offsets(shape)
        min_size = self.cuboid_constraint_margin()
        order = [index, (index + 1) % 4, (index + 2) % 4, (index + 3) % 4]
        p0 = QtCore.QPointF(front_points[order[0]])
        p1 = QtCore.QPointF(front_points[order[1]])
        p2 = QtCore.QPointF(front_points[order[2]])
        p3 = QtCore.QPointF(front_points[order[3]])
        basis_u = p1 - p0
        basis_v = p3 - p0
        len_u = self._vector_length(basis_u)
        len_v = self._vector_length(basis_v)
        if len_u <= 1e-6 or len_v <= 1e-6:
            return
        unit_u = self._vector_scale(basis_u, 1.0 / len_u)
        unit_v = self._vector_scale(basis_v, 1.0 / len_v)
        target = p2 - pos
        solved = self._solve_vector_basis(target, unit_u, unit_v)
        if solved is None:
            return
        coeff_u, coeff_v = solved
        coeff_u = max(coeff_u, min_size)
        coeff_v = max(coeff_v, min_size)
        new_p0 = (
            p2
            - self._vector_scale(unit_u, coeff_u)
            - self._vector_scale(unit_v, coeff_v)
        )
        new_p1 = p2 - self._vector_scale(unit_v, coeff_v)
        new_p3 = p2 - self._vector_scale(unit_u, coeff_u)
        front_points[order[0]] = new_p0
        front_points[order[1]] = new_p1
        front_points[order[3]] = new_p3
        front_points[order[2]] = p2
        self.set_cuboid_front_with_offsets(shape, front_points, offsets)

    def adjust_cuboid_front_edge(self, shape, index, pos):
        if shape.shape_type != "cuboid" or len(shape.points) != 8:
            return
        front_points = [QtCore.QPointF(p) for p in shape.points[:4]]
        offsets = self.get_cuboid_back_offsets(shape)
        min_size = self.cuboid_constraint_margin()
        edge_map = {
            Shape.CUBOID_FRONT_LEFT_EDGE_CENTER: (0, 3, 1, 2),
            Shape.CUBOID_FRONT_RIGHT_EDGE_CENTER: (1, 2, 0, 3),
            Shape.CUBOID_FRONT_TOP_EDGE_CENTER: (0, 1, 3, 2),
            Shape.CUBOID_FRONT_BOTTOM_EDGE_CENTER: (3, 2, 0, 1),
        }
        if index not in edge_map:
            return
        edge_a, edge_b, opposite_a, opposite_b = edge_map[index]
        dragged_center = self.get_mid_point(
            front_points[edge_a], front_points[edge_b]
        )
        edge_vector = front_points[edge_b] - front_points[edge_a]
        edge_length = self._vector_length(edge_vector)
        if edge_length <= 1e-6:
            return
        normal = QtCore.QPointF(
            -edge_vector.y() / edge_length,
            edge_vector.x() / edge_length,
        )
        edge_distance = self._vector_dot(
            front_points[opposite_a] - front_points[edge_a], normal
        )
        if edge_distance < 0:
            normal = self._vector_scale(normal, -1.0)
            edge_distance = -edge_distance
        shift = self._vector_dot(pos - dragged_center, normal)
        max_shift = edge_distance - min_size
        if shift > max_shift:
            shift = max_shift
        shift_vector = self._vector_scale(normal, shift)
        front_points[edge_a] = front_points[edge_a] + shift_vector
        front_points[edge_b] = front_points[edge_b] + shift_vector
        self.set_cuboid_front_with_offsets(shape, front_points, offsets)

    def adjust_cuboid_back_edge_center(self, shape, index, pos):
        if shape.shape_type != "cuboid" or len(shape.points) != 8:
            return
        visible_center_index = shape.get_cuboid_visible_rear_center_index()
        if index != visible_center_index:
            return
        points = [QtCore.QPointF(p) for p in shape.points]
        center = shape.get_cuboid_control_point(index)
        if center is None:
            return
        margin = self.cuboid_constraint_margin()
        front_right = max(points[1].x(), points[2].x())
        front_left = min(points[0].x(), points[3].x())
        target_x = pos.x()
        front_center_index = Shape.CUBOID_FRONT_RIGHT_EDGE_CENTER
        if index == Shape.CUBOID_BACK_RIGHT_EDGE_CENTER:
            target_x = max(target_x, front_right + margin)
        else:
            target_x = min(target_x, front_left - margin)
            front_center_index = Shape.CUBOID_FRONT_LEFT_EDGE_CENTER
        dx = target_x - center.x()
        front_center = shape.get_cuboid_control_point(front_center_index)
        dy = 0.0
        if front_center is not None:
            dir_x = center.x() - front_center.x()
            if abs(dir_x) > 1e-6:
                dir_y = center.y() - front_center.y()
                dy = dx * dir_y / dir_x
        for i in range(4, 8):
            points[i].setX(points[i].x() + dx)
            points[i].setY(points[i].y() + dy)
        self.set_cuboid_raw_points(shape, points)

    def move_cuboid_control(self, shape, index, pos):
        if (
            shape.locked
            or shape.shape_type != "cuboid"
            or len(shape.points) != 8
        ):
            return
        if index in [0, 1, 2, 3]:
            self.adjust_cuboid_front_vertex(shape, index, pos)
            return
        if index in [4, 5, 6, 7]:
            self.adjust_cuboid_visible_back_vertex(shape, index, pos)
            return
        if index in CUBOID_FRONT_EDGE_CENTER_INDICES:
            self.adjust_cuboid_front_edge(shape, index, pos)
            return
        if index in CUBOID_BACK_EDGE_CENTER_INDICES:
            self.adjust_cuboid_back_edge_center(shape, index, pos)

    def move_cuboid_face_by(self, shape, face_name, offset):
        if (
            shape.locked
            or shape.shape_type != "cuboid"
            or len(shape.points) != 8
        ):
            return
        min_size = self.cuboid_constraint_margin()
        if face_name == CUBOID_FACE_LEFT:
            points = [QtCore.QPointF(p) for p in shape.points]
            front_right_mid = (points[1].x() + points[2].x()) / 2.0
            back_right_mid = (points[5].x() + points[6].x()) / 2.0
            right_top, right_bottom = (1, 2)
            if back_right_mid > front_right_mid:
                right_top, right_bottom = (5, 6)
            right_candidates = [
                points[right_top].x(),
                (points[right_top].x() + points[right_bottom].x()) / 2.0,
                points[right_bottom].x(),
            ]
            right_limit_x = min(right_candidates) - min_size
            left_indices = [0, 3, 4, 7]
            left_max_x = max(points[i].x() for i in left_indices)
            dx = offset.x()
            if dx > 0:
                dx = min(dx, right_limit_x - left_max_x)
            dy = offset.y()
            for i in left_indices:
                points[i].setX(points[i].x() + dx)
                points[i].setY(points[i].y() + dy)
            self.set_cuboid_raw_points(shape, points)
        elif face_name == CUBOID_FACE_RIGHT:
            points = [QtCore.QPointF(p) for p in shape.points]
            front_left_mid = (points[0].x() + points[3].x()) / 2.0
            back_left_mid = (points[4].x() + points[7].x()) / 2.0
            left_top, left_bottom = (0, 3)
            if back_left_mid < front_left_mid:
                left_top, left_bottom = (4, 7)
            left_candidates = [
                points[left_top].x(),
                (points[left_top].x() + points[left_bottom].x()) / 2.0,
                points[left_bottom].x(),
            ]
            left_limit_x = max(left_candidates) + min_size
            right_indices = [1, 2, 5, 6]
            right_min_x = min(points[i].x() for i in right_indices)
            dx = offset.x()
            if dx < 0:
                dx = max(dx, left_limit_x - right_min_x)
            dy = offset.y()
            for i in right_indices:
                points[i].setX(points[i].x() + dx)
                points[i].setY(points[i].y() + dy)
            self.set_cuboid_raw_points(shape, points)
        elif face_name == CUBOID_FACE_BACK:
            points = [QtCore.QPointF(p) for p in shape.points]
            next_back_points = [QtCore.QPointF(points[i]) for i in range(4, 8)]
            for p in next_back_points:
                p.setX(p.x() + offset.x())
                p.setY(p.y() + offset.y())
            for i, p in enumerate(next_back_points, start=4):
                points[i] = p
            self.set_cuboid_raw_points(shape, points)
        else:
            return

    def bounded_move_vertex(self, pos):
        """Move a vertex. Adjust position to be bounded by pixmap border"""
        index, shape = self.h_vertex, self.h_shape
        if shape.locked:
            return
        if shape.shape_type == "cuboid":
            self.move_cuboid_control(shape, index, pos)
            return
        point = shape[index]
        if (
            self.out_off_pixmap(pos)
            and shape.shape_type not in self.allowed_oop_shape_types
        ):
            pos = self.intersection_point(point, pos)

        if shape.shape_type == "rotation":
            sindex = (index + 2) % 4
            # Get the other 3 points after transformed
            p2, p3, p4 = self.get_adjoint_points(
                shape.direction, shape[sindex], pos, index
            )
            # if (
            #     self.out_off_pixmap(p2)
            #     or self.out_off_pixmap(p3)
            #     or self.out_off_pixmap(p4)
            # ):
            #     # No need to move if one pixal out of map
            #     return
            # Move 4 pixal one by one
            shape.move_vertex_by(index, pos - point)
            lindex = (index + 1) % 4
            rindex = (index + 3) % 4
            shape[lindex] = p2
            shape[rindex] = p4
            shape.close()
        elif shape.shape_type == "rectangle":
            shift_pos = pos - point
            shape.move_vertex_by(index, shift_pos)
            left_index = (index + 1) % 4
            right_index = (index + 3) % 4
            left_shift = None
            right_shift = None
            if index % 2 == 0:
                right_shift = QtCore.QPointF(shift_pos.x(), 0)
                left_shift = QtCore.QPointF(0, shift_pos.y())
            else:
                left_shift = QtCore.QPointF(shift_pos.x(), 0)
                right_shift = QtCore.QPointF(0, shift_pos.y())
            shape.move_vertex_by(right_index, right_shift)
            shape.move_vertex_by(left_index, left_shift)
        else:
            shape.move_vertex_by(index, pos - point)

    def bounded_move_shapes(self, shapes, pos):
        """Move shapes. Adjust position to be bounded by pixmap border"""
        shapes = [shape for shape in shapes if not shape.locked]
        if not shapes:
            return False
        shape_types = []
        for shape in shapes:
            if shape.shape_type in self.allowed_oop_shape_types:
                shape_types.append(shape.shape_type)

        if self.out_off_pixmap(pos) and len(shape_types) == 0:
            return False  # No need to move
        if len(shape_types) > 0 and len(shapes) != len(shape_types):
            return False

        if len(shape_types) == 0:
            o1 = pos + self.offsets[0]
            if self.out_off_pixmap(o1):
                pos -= QtCore.QPointF(min(0, int(o1.x())), min(0, int(o1.y())))
            o2 = pos + self.offsets[1]
            if self.out_off_pixmap(o2):
                pos += QtCore.QPointF(
                    min(0, int(self.pixmap.width() - o2.x())),
                    min(0, int(self.pixmap.height() - o2.y())),
                )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prev_point
        if dp:
            for shape in shapes:
                shape.move_by(dp)
            self.prev_point = pos
            return True
        return False

    def rotate_point(self, p, center, theta):
        order = p - center
        cosTheta = math.cos(theta)
        sinTheta = math.sin(theta)
        pResx = cosTheta * order.x() + sinTheta * order.y()
        pResy = -sinTheta * order.x() + cosTheta * order.y()
        pRes = QtCore.QPointF(center.x() + pResx, center.y() + pResy)
        return pRes

    def bounded_rotate_shapes(self, i, shape, theta):
        """Rotate shapes. Adjust position to be bounded by pixmap border"""
        if shape.locked:
            return False
        if len(shape.points) == 2:
            p0 = shape.points[0]
            p1 = shape.points[1]
            shape.points = [
                p0,
                QtCore.QPointF(
                    (p0.x() + p1.x()) / 2,
                    p0.y(),
                ),
                p1,
                QtCore.QPointF(p1.x(), (p0.y() + p1.y()) / 2),
            ]
        center = QtCore.QPointF(
            (shape.points[0].x() + shape.points[2].x()) / 2,
            (shape.points[0].y() + shape.points[2].y()) / 2,
        )
        for j, p in enumerate(shape.points):
            pos = self.rotate_point(p, center, theta)
            # TODO: Reserved for now
            # if self.out_off_pixmap(pos):
            #     return False  # No need to rotate
            shape.points[j] = pos
        shape.direction = (shape.direction - theta) % (2 * math.pi)
        return True

    def deselect_shape(self):
        """Deselect all shapes"""
        self._selected_group_id = None
        if self.selected_shapes:
            self.set_hiding(False)
            self.selection_changed.emit([])
            self.h_shape_is_selected = False
            self.h_cuboid_face = None
            self.update()

    def delete_selected(self):
        """Remove selected shapes"""
        deleted_shapes = []
        if self.selected_shapes:
            for shape in self.selected_shapes:
                if not shape.locked and shape in self.shapes:
                    self.shapes.remove(shape)
                    deleted_shapes.append(shape)
            if deleted_shapes:
                self.store_shapes()
            self.selected_shapes = [
                shape for shape in self.selected_shapes if shape.locked
            ]
            self.update()
        return deleted_shapes

    def delete_shape(self, shape):
        """Remove a specific shape"""
        if shape.locked:
            return
        if shape in self.selected_shapes:
            self.selected_shapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.store_shapes()
        self.update()

    def duplicate_selected_shapes(self):
        """Duplicate selected shapes"""
        if self.selected_shapes:
            self.selected_shapes_copy = [
                s.copy() for s in self.selected_shapes
            ]
            for shape in self.selected_shapes_copy:
                shape.locked = False
            self.bounded_shift_shapes(self.selected_shapes_copy)
            self.end_move(copy=True)
        return self.selected_shapes

    def bounded_shift_shapes(self, shapes):
        """
        Shift shapes by an offset. Adjust positions to be bounded
        by pixmap borders
        """
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPointF(2.0, 2.0)
        self.offsets = QtCore.QPointF(), QtCore.QPointF()
        self.prev_point = point
        if not self.bounded_move_shapes(shapes, point - offset):
            self.bounded_move_shapes(shapes, point + offset)

    def prepare_pasted_shapes(self, shapes, copied_group_id=None):
        pasted_shapes = [shape.copy() for shape in shapes]
        if not pasted_shapes:
            return pasted_shapes
        for shape in pasted_shapes:
            shape.locked = False
        if copied_group_id is not None and all(
            shape.group_id == copied_group_id for shape in pasted_shapes
        ):
            group_id = self.gen_new_group_id()
            for shape in pasted_shapes:
                shape.group_id = group_id
        return pasted_shapes

    def _paint_groups(self, painter):
        if not self.show_groups:
            return

        painter.save()
        theme = get_theme()
        active_shapes = self._active_group_shapes()
        for group_id, group_shapes in self._grouped_shapes().items():
            visible_shapes = [shape for shape in group_shapes if shape.visible]
            if not visible_shapes:
                continue
            group_rect = self._group_rect(visible_shapes)
            group_color = QtGui.QColor(
                *LABEL_COLORMAP[int(group_id) % len(LABEL_COLORMAP)]
            )
            selected = bool(
                active_shapes and group_id == self._selected_group_id
            )
            hovered = group_id == self._hovered_group_id

            for shape in visible_shapes:
                rect = shape.bounding_rect()
                center = rect.center()
                radius = max(1.0, 3.0 / self.scale)
                triangle = [
                    QtCore.QPointF(center.x(), center.y() - radius),
                    QtCore.QPointF(center.x() - radius, center.y() + radius),
                    QtCore.QPointF(center.x() + radius, center.y() + radius),
                ]
                painter.setPen(
                    QtGui.QPen(
                        group_color,
                        max(1.0, 4.0 / self.scale),
                        Qt.PenStyle.SolidLine,
                    )
                )
                painter.drawPolygon(triangle)

            if selected or hovered:
                color = QtGui.QColor(
                    theme["selection" if selected else "primary_hover"]
                )
                glow = QtGui.QColor(color)
                glow.setAlpha(80 if selected else 50)
                painter.setPen(
                    QtGui.QPen(
                        glow,
                        (6.0 if selected else 4.0) / self.scale,
                        Qt.PenStyle.SolidLine,
                    )
                )
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(group_rect)
                width = (2.5 if selected else 2.0) / self.scale
                style = Qt.PenStyle.SolidLine
            else:
                color = QtGui.QColor("#EEEEEE")
                width = 1.0 / self.scale
                style = Qt.PenStyle.DashLine

            painter.setPen(QtGui.QPen(color, width, style))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(group_rect)

            label_rect = self._group_label_rect(
                group_id, len(group_shapes), group_rect
            )
            label_color = color if selected or hovered else group_color
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(label_color)
            painter.drawRoundedRect(
                label_rect, 3.0 / self.scale, 3.0 / self.scale
            )
            painter.setFont(self._group_label_font())
            painter.setPen(QtGui.QColor(theme["selection_text"]))
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignCenter,
                self._group_label(group_id, len(group_shapes)),
            )
        painter.restore()

    # QT Overload
    def paintEvent(self, event):  # noqa: C901
        """Paint event for canvas"""
        if (
            self.pixmap is None
            or self.pixmap.width() == 0
            or self.pixmap.height() == 0
        ):
            super().paintEvent(event)
            return

        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offset_to_center())

        p.drawPixmap(0, 0, self.pixmap)

        # Draw compare view: left side shows compare image, right side shows original
        # split_position: 0 = all original, 1 = all compare
        if (
            self.compare_pixmap is not None
            and not self.compare_pixmap.isNull()
        ):
            split_x = int(self.split_position * self.pixmap.width())
            if split_x > 0:
                p.drawPixmap(
                    0,
                    0,
                    self.compare_pixmap,
                    0,
                    0,
                    split_x,
                    self.pixmap.height(),
                )

        Shape.scale = self.scale

        # Draw loading/waiting screen
        if self.is_loading:
            # Draw a semi-transparent rectangle
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(0, 0, 0, 20))
            p.drawRect(self.pixmap.rect())

            # Draw a spinning wheel
            p.setPen(QtGui.QColor(255, 255, 255))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.save()
            p.translate(self.pixmap.width() / 2, self.pixmap.height() / 2 - 50)
            p.rotate(self.loading_angle)
            p.drawEllipse(-20, -20, 40, 40)
            p.drawLine(0, 0, 0, -20)
            p.restore()
            self.loading_angle += 30
            if self.loading_angle >= 360:
                self.loading_angle = 0

            # Draw the loading text
            p.setPen(QtGui.QColor(255, 255, 255))
            p.setFont(QtGui.QFont("Arial", 20))
            p.drawText(
                self.pixmap.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self.loading_text,
            )
            p.end()
            self.update()
            return

        # Apply the global label/shape opacity to every annotation drawn
        # below (masks, shapes, degrees, groups, brush overlays). Image text
        # labels are restored to full opacity before being painted.
        p.setOpacity(self.shape_opacity)

        # Draw KIE linking
        if self.show_linking:
            pen = QtGui.QPen(QtGui.QColor("#AAAAAA"), 2, Qt.PenStyle.SolidLine)
            p.setPen(pen)
            gid2point = {}
            linking_pairs = []
            group_color = (255, 128, 0)
            for shape in self.shapes:
                if not shape.visible:
                    continue

                try:
                    linking_pairs += shape.kie_linking
                except Exception:
                    pass

                if shape.group_id is None or shape.shape_type not in [
                    "rectangle",
                    "polygon",
                    "rotation",
                    "quadrilateral",
                    "cuboid",
                ]:
                    continue
                rect = shape.bounding_rect()
                cx = rect.x() + (rect.width() / 2.0)
                cy = rect.y() + (rect.height() / 2.0)
                gid2point[shape.group_id] = (cx, cy)

            for linking in linking_pairs:
                pen.setStyle(Qt.PenStyle.SolidLine)
                pen.setWidth(max(1, int(round(4.0 / Shape.scale))))
                pen.setColor(QtGui.QColor(*group_color))
                p.setPen(pen)
                key, value = linking
                # Adapt to the 'ungroup_selected_shapes' operation
                if key not in gid2point or value not in gid2point:
                    continue
                kp, vp = gid2point[key], gid2point[value]
                # Draw a link from key point to value point
                p.drawLine(QtCore.QPointF(*kp), QtCore.QPointF(*vp))
                # Draw the triangle arrowhead
                arrow_size = max(
                    1, int(round(10.0 / Shape.scale))
                )  # Size of the arrowhead
                angle = math.atan2(
                    vp[1] - kp[1], vp[0] - kp[0]
                )  # Angle towards the value point
                arrow_points = [
                    QtCore.QPointF(vp[0], vp[1]),
                    QtCore.QPointF(
                        vp[0] - arrow_size * math.cos(angle - math.pi / 6),
                        vp[1] - arrow_size * math.sin(angle - math.pi / 6),
                    ),
                    QtCore.QPointF(
                        vp[0] - arrow_size * math.cos(angle + math.pi / 6),
                        vp[1] - arrow_size * math.sin(angle + math.pi / 6),
                    ),
                ]
                p.drawPolygon(arrow_points)

        # Draw shape masks
        if self.show_masks:
            for shape in self.shapes:
                if not shape.visible:
                    continue
                # Shapes under live brush editing render their own overlay.
                if getattr(shape, "_brush_using_mask", False):
                    continue
                if shape.shape_type not in [
                    "polygon",
                    "rectangle",
                    "rotation",
                    "quadrilateral",
                    "circle",
                ]:
                    continue
                if shape.shape_type == "polygon" and len(shape.points) < 3:
                    continue
                if shape.shape_type == "rectangle" and len(shape.points) < 2:
                    continue
                if shape.shape_type == "rotation" and len(shape.points) < 2:
                    continue
                if (
                    shape.shape_type == "quadrilateral"
                    and len(shape.points) < 4
                ):
                    continue
                if shape.shape_type == "circle" and len(shape.points) < 2:
                    continue
                if not (
                    (shape.selected or not self._hide_backround)
                    and self.is_visible(shape)
                ):
                    continue

                mask_path = QtGui.QPainterPath()
                if shape.shape_type == "polygon":
                    mask_path.moveTo(shape.points[0])
                    for point in shape.points[1:]:
                        mask_path.lineTo(point)
                    if shape.is_closed() or len(shape.points) >= 3:
                        mask_path.closeSubpath()
                elif shape.shape_type == "rectangle":
                    if len(shape.points) == 2:
                        rectangle = shape.get_rect_from_line(*shape.points)
                        mask_path.addRect(rectangle)
                    elif len(shape.points) == 4:
                        mask_path.moveTo(shape.points[0])
                        for point in shape.points[1:]:
                            mask_path.lineTo(point)
                        mask_path.closeSubpath()
                elif shape.shape_type == "rotation":
                    if len(shape.points) == 2:
                        rectangle = shape.get_rect_from_line(*shape.points)
                        mask_path.addRect(rectangle)
                    elif len(shape.points) == 4:
                        mask_path.moveTo(shape.points[0])
                        for point in shape.points[1:]:
                            mask_path.lineTo(point)
                        mask_path.closeSubpath()
                elif shape.shape_type == "quadrilateral":
                    if len(shape.points) == 4:
                        mask_path.moveTo(shape.points[0])
                        for point in shape.points[1:]:
                            mask_path.lineTo(point)
                        mask_path.closeSubpath()
                elif shape.shape_type == "circle":
                    if len(shape.points) == 2:
                        rectangle = shape.get_circle_rect_from_line(
                            shape.points
                        )
                        mask_path.addEllipse(rectangle)

                fill_color = (
                    shape.select_line_color
                    if shape.selected
                    else shape.line_color
                )
                fill_color_alpha = QtGui.QColor(
                    fill_color.red(),
                    fill_color.green(),
                    fill_color.blue(),
                    self.mask_opacity,
                )
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(fill_color_alpha)
                p.drawPath(mask_path)

                outline_color = (
                    shape.select_line_color
                    if shape.selected
                    else shape.line_color
                )
                pen = QtGui.QPen(outline_color)
                pen.setWidth(
                    max(1, int(round(shape.line_width / Shape.scale)))
                )
                if shape.difficult:
                    pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(mask_path)

        # Draw degrees
        for shape in self.shapes:
            if (
                shape.selected or not self._hide_backround
            ) and self.is_visible(shape):
                shape.hovered = shape == self.h_shape
                shape.fill = (
                    self._fill_drawing
                    and (shape.selected or shape == self.h_shape)
                    and not (self.selected_vertex() and self.moving_shape)
                )
                # Brush-edited shapes are drawn from their mask instead.
                if not getattr(shape, "_brush_using_mask", False):
                    shape.paint(p)

            if (
                shape.shape_type == "rotation"
                and len(shape.points) == 4
                and self.is_visible(shape)
            ):
                d = shape.point_size / shape.scale
                center = QtCore.QPointF(
                    (shape.points[0].x() + shape.points[2].x()) / 2,
                    (shape.points[0].y() + shape.points[2].y()) / 2,
                )
                if self.show_degrees:
                    degrees = math.degrees(shape.direction)
                    if abs(degrees - 360.0) < 0.1:
                        degrees = 0.0
                    degrees = f"{degrees:.2f}°"
                    p.setFont(
                        QtGui.QFont(
                            "Arial",
                            int(max(6.0, int(round(8.0 / Shape.scale)))),
                        )
                    )
                    pen = QtGui.QPen(
                        QtGui.QColor("#FF9900"),
                        8,
                        QtCore.Qt.PenStyle.SolidLine,
                    )
                    p.setPen(pen)
                    fm = QtGui.QFontMetrics(p.font())
                    rect = fm.boundingRect(degrees)
                    p.fillRect(
                        int(rect.x() + center.x() - d),
                        int(rect.y() + center.y() + d),
                        int(rect.width()),
                        int(rect.height()),
                        QtGui.QColor("#FF9900"),
                    )
                    pen = QtGui.QPen(
                        QtGui.QColor("#FFFFFF"),
                        7,
                        QtCore.Qt.PenStyle.SolidLine,
                    )
                    p.setPen(pen)
                    p.drawText(
                        int(center.x() - d),
                        int(center.y() + d),
                        degrees,
                    )
                else:
                    cp = QtGui.QPainterPath()
                    cp.addRect(
                        int(center.x() - d / 2),
                        int(center.y() - d / 2),
                        int(d),
                        int(d),
                    )
                    p.drawPath(cp)
                    p.fillPath(cp, QtGui.QColor(255, 153, 0, 255))

        self._paint_rotation_handles(p)

        self._paint_groups(p)

        # Draw live brush-edit overlays on top of the regular shapes.
        self._paint_brush_overlays(p)

        if self.current:
            self.current.paint(p)
            self.line.paint(p)

            if (
                self.create_mode == "quadrilateral"
                and len(self.current.points) == 3
                and len(self.line.points) >= 2
            ):
                color = (
                    self.current.select_line_color
                    if self.current.selected
                    else self.current.line_color
                )
                pen = QtGui.QPen(color)
                pen.setWidth(
                    max(1, int(round(self.current.line_width / Shape.scale)))
                )
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawLine(QtCore.QLineF(self.line[1], self.current.points[0]))
        if self.selected_shapes_copy:
            for s in self.selected_shapes_copy:
                s.paint(p)

        if (
            self.fill_drawing()
            and self.create_mode == "polygon"
            and self.current is not None
            and len(self.current.points) >= 2
        ):
            drawing_shape = copy.copy(self.current)
            drawing_shape.points = self.current.points + [self.line[1]]
            drawing_shape.fill = True
            drawing_shape.paint(p)
        if (
            self.fill_drawing()
            and self.create_mode == "quadrilateral"
            and self.current is not None
            and len(self.current.points) == 3
            and len(self.line.points) >= 2
        ):
            drawing_shape = copy.copy(self.current)
            drawing_shape.points = list(self.current.points) + [
                QtCore.QPointF(self.line[1].x(), self.line[1].y())
            ]
            drawing_shape.fill = True
            drawing_shape._closed = True
            drawing_shape.paint(p)

        # Restore full opacity so labels/scores/attributes stay readable.
        p.setOpacity(1.0)

        # Draw texts
        if self.show_texts:
            text_color = "#FFFFFF"
            background_color = "#007BFF"
            p.setFont(
                QtGui.QFont(
                    "Arial", int(max(6.0, int(round(8.0 / Shape.scale))))
                )
            )
            pen = QtGui.QPen(
                QtGui.QColor(background_color), 8, Qt.PenStyle.SolidLine
            )
            p.setPen(pen)
            for shape in self.shapes:
                if not shape.visible:
                    continue
                description = shape.description
                if description:
                    bbox = shape.bounding_rect()
                    fm = QtGui.QFontMetrics(p.font())
                    text_rect = fm.tightBoundingRect(description)

                    padding_x = 4
                    padding_y = 2
                    rect_width = text_rect.width() + 2 * padding_x
                    rect_height = fm.height() + 2 * padding_y

                    bg_x = int(bbox.x())
                    bg_y = int(bbox.y() - rect_height)

                    p.fillRect(
                        bg_x,
                        bg_y,
                        rect_width,
                        rect_height,
                        QtGui.QColor(background_color),
                    )

            pen = QtGui.QPen(
                QtGui.QColor(text_color), 8, Qt.PenStyle.SolidLine
            )
            p.setPen(pen)
            for shape in self.shapes:
                if not shape.visible:
                    continue
                description = shape.description
                if description:
                    bbox = shape.bounding_rect()
                    fm = QtGui.QFontMetrics(p.font())

                    padding_x = 4
                    padding_y = 2

                    text_x = int(bbox.x() + padding_x)
                    text_y = int(bbox.y() - padding_y - fm.descent())

                    p.drawText(
                        text_x,
                        text_y,
                        description,
                    )

        # Draw labels
        if self.show_labels:
            p.setFont(
                QtGui.QFont(
                    "Arial", int(max(6.0, int(round(8.0 / Shape.scale))))
                )
            )
            labels = []
            for shape in self.shapes:
                if not shape.visible:
                    continue
                d_react = shape.point_size / shape.scale
                if shape.label in [
                    "AUTOLABEL_OBJECT",
                    "AUTOLABEL_ADD",
                    "AUTOLABEL_REMOVE",
                ]:
                    continue
                label_text = (
                    (
                        f"id:{shape.group_id} "
                        if shape.group_id is not None
                        else ""
                    )
                    + (f"{shape.label}")
                    + (
                        f" {float(shape.score):.2f}"
                        if (shape.score is not None and self.show_scores)
                        else ""
                    )
                )
                if not label_text:
                    continue
                fm = QtGui.QFontMetrics(p.font())
                text_rect = fm.tightBoundingRect(label_text)
                padding_x = 4
                padding_y = 2
                rect_width = text_rect.width() + 2 * padding_x
                rect_height = fm.height() + 2 * padding_y

                if shape.shape_type in [
                    "rectangle",
                    "polygon",
                    "rotation",
                    "quadrilateral",
                    "cuboid",
                ]:
                    try:
                        bbox = shape.bounding_rect()
                    except IndexError:
                        continue
                    rect = QtCore.QRect(
                        int(bbox.x()),
                        int(bbox.y()),
                        rect_width,
                        rect_height,
                    )
                    text_pos = QtCore.QPoint(
                        int(bbox.x() + padding_x),
                        int(bbox.y() + rect_height - padding_y - fm.descent()),
                    )
                elif shape.shape_type == "circle":
                    points = shape.points
                    if not points:
                        continue
                    point = points[0]
                    rect = QtCore.QRect(
                        int(point.x() - rect_width / 2),
                        int(point.y() - rect_height / 2),
                        rect_width,
                        rect_height,
                    )
                    text_pos = QtCore.QPoint(
                        int(point.x() - rect_width / 2 + padding_x),
                        int(
                            point.y()
                            + rect_height / 2
                            - padding_y
                            - fm.descent()
                        ),
                    )
                elif shape.shape_type in [
                    "line",
                    "linestrip",
                    "point",
                ]:
                    points = shape.points
                    if not points:
                        continue
                    point = points[0]
                    rect = QtCore.QRect(
                        int(point.x() + d_react),
                        int(point.y() - 15),
                        rect_width,
                        rect_height,
                    )
                    text_pos = QtCore.QPoint(
                        int(point.x() + d_react + padding_x),
                        int(
                            point.y()
                            - 15
                            + rect_height
                            - padding_y
                            - fm.descent()
                        ),
                    )
                else:
                    continue
                labels.append((shape, rect, text_pos, label_text))

            pen = QtGui.QPen(QtGui.QColor("#FFA500"), 8, Qt.PenStyle.SolidLine)
            p.setPen(pen)
            for shape, rect, _, _ in labels:
                if not shape.visible:
                    continue
                p.fillRect(rect, shape.line_color)

            pen = QtGui.QPen(QtGui.QColor("#000000"), 8, Qt.PenStyle.SolidLine)
            p.setPen(pen)
            for shape, _, text_pos, label_text in labels:
                if not shape.visible:
                    continue
                p.drawText(text_pos, label_text)

        # Draw mouse coordinates
        if self.cross_line_show:
            pen = QtGui.QPen(
                QtGui.QColor(self.cross_line_color),
                max(1, int(round(self.cross_line_width / Shape.scale))),
                Qt.PenStyle.DashLine,
            )
            p.setPen(pen)
            p.setOpacity(self.cross_line_opacity)
            p.drawLine(
                QtCore.QPointF(self.prev_move_point.x(), 0),
                QtCore.QPointF(self.prev_move_point.x(), self.pixmap.height()),
            )
            p.drawLine(
                QtCore.QPointF(0, self.prev_move_point.y()),
                QtCore.QPointF(self.pixmap.width(), self.prev_move_point.y()),
            )

        # Draw attributes
        if self.show_attributes:
            font_size = int(max(8.0, int(round(10.0 / Shape.scale))))
            font = QtGui.QFont("Arial", font_size, QtGui.QFont.Weight.Bold)
            p.setFont(font)
            attributes_list = []

            for shape in self.shapes:
                if not shape.visible:
                    continue
                if not hasattr(shape, "attributes") or not shape.attributes:
                    continue
                if shape.label in [
                    "AUTOLABEL_OBJECT",
                    "AUTOLABEL_ADD",
                    "AUTOLABEL_REMOVE",
                ]:
                    continue

                attrs_text = []
                for key, value in shape.attributes.items():
                    attrs_text.append(f"{key}: {value}")
                if not attrs_text:
                    continue

                max_attrs_per_line = 1
                attribute_lines = []
                for i in range(0, len(attrs_text), max_attrs_per_line):
                    line_attrs = attrs_text[i : i + max_attrs_per_line]
                    attribute_lines.append(" | ".join(line_attrs))

                fm = QtGui.QFontMetrics(font)
                max_width = 0
                line_heights = []
                for line in attribute_lines:
                    line_rect = fm.tightBoundingRect(line)
                    max_width = max(max_width, line_rect.width())
                    line_heights.append(fm.height())
                total_height = sum(line_heights)

                padding_x = 8
                padding_y = 2
                rect_width = max_width + 2 * padding_x
                rect_height = total_height + 2 * padding_y
                d_react = shape.point_size / shape.scale

                if shape.shape_type in [
                    "rectangle",
                    "polygon",
                    "rotation",
                    "quadrilateral",
                    "cuboid",
                ]:
                    try:
                        bbox = shape.bounding_rect()
                    except IndexError:
                        continue

                    rect = QtCore.QRect(
                        int(bbox.x()),
                        int(bbox.y() + bbox.height() + 1),
                        rect_width,
                        rect_height,
                    )

                    text_positions = []
                    y_offset = 0
                    for i, line_height in enumerate(line_heights):
                        text_pos = QtCore.QPoint(
                            int(bbox.x() + padding_x),
                            int(
                                bbox.y()
                                + bbox.height()
                                + 1
                                + padding_y
                                + y_offset
                                + fm.ascent()
                            ),
                        )
                        text_positions.append(text_pos)
                        y_offset += line_height

                elif shape.shape_type in [
                    "circle",
                    "line",
                    "linestrip",
                    "point",
                ]:
                    points = shape.points
                    if not points:
                        continue
                    point = points[0]

                    rect = QtCore.QRect(
                        int(point.x() + d_react),
                        int(point.y() + 1),
                        rect_width,
                        rect_height,
                    )

                    text_positions = []
                    y_offset = 0
                    for i, line_height in enumerate(line_heights):
                        text_pos = QtCore.QPoint(
                            int(point.x() + d_react + padding_x),
                            int(
                                point.y()
                                + 1
                                + padding_y
                                + y_offset
                                + fm.ascent()
                            ),
                        )
                        text_positions.append(text_pos)
                        y_offset += line_height
                else:
                    continue

                attributes_list.append(
                    (shape, rect, text_positions, attribute_lines)
                )

            for shape, rect, _, _ in attributes_list:
                if not shape.visible:
                    continue

                background_color = QtGui.QColor(*self.attr_background_color)
                p.fillRect(rect, background_color)

                pen = QtGui.QPen(
                    QtGui.QColor(*self.attr_border_color),
                    1,
                    Qt.PenStyle.SolidLine,
                )
                p.setPen(pen)
                p.drawRect(rect)

            pen = QtGui.QPen(
                QtGui.QColor(*self.attr_text_color), 1, Qt.PenStyle.SolidLine
            )
            p.setPen(pen)
            p.setFont(font)

            for _, _, text_positions, attribute_lines in attributes_list:
                for i, (text_pos, line_text) in enumerate(
                    zip(text_positions, attribute_lines)
                ):
                    p.drawText(text_pos, line_text)

        # Draw compare view split line
        if (
            self.compare_pixmap is not None
            and not self.compare_pixmap.isNull()
        ):
            split_x = int(self.split_position * self.pixmap.width())
            img_h = self.pixmap.height()
            if 0 < split_x < self.pixmap.width():
                p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                line_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 180), 1.5)
                line_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(line_pen)
                p.drawLine(split_x, 0, split_x, img_h)
                handle_radius = 16
                handle_y = img_h // 2
                gradient = QtGui.QRadialGradient(
                    split_x, handle_y, handle_radius
                )
                gradient.setColorAt(0, QtGui.QColor(255, 255, 255, 200))
                gradient.setColorAt(1, QtGui.QColor(255, 255, 255, 0))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(gradient)
                p.drawEllipse(
                    split_x - handle_radius,
                    handle_y - handle_radius,
                    handle_radius * 2,
                    handle_radius * 2,
                )
                p.setBrush(QtGui.QColor(255, 255, 255, 200))
                p.drawEllipse(split_x - 8, handle_y - 8, 16, 16)
                arrow_pen = QtGui.QPen(QtGui.QColor(100, 100, 100, 180), 1.5)
                arrow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                arrow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.setPen(arrow_pen)
                arrow_size = 4
                p.drawLine(
                    split_x - arrow_size, handle_y, split_x - 1, handle_y
                )
                p.drawLine(
                    split_x - arrow_size,
                    handle_y,
                    split_x - arrow_size + 2,
                    handle_y - 2,
                )
                p.drawLine(
                    split_x - arrow_size,
                    handle_y,
                    split_x - arrow_size + 2,
                    handle_y + 2,
                )
                p.drawLine(
                    split_x + 1, handle_y, split_x + arrow_size, handle_y
                )
                p.drawLine(
                    split_x + arrow_size,
                    handle_y,
                    split_x + arrow_size - 2,
                    handle_y - 2,
                )
                p.drawLine(
                    split_x + arrow_size,
                    handle_y,
                    split_x + arrow_size - 2,
                    handle_y + 2,
                )
                p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)

        # Brush-size preview circle follows the cursor in brush mode.
        self._paint_brush_cursor(p)

        p.end()

    def render_visualization(
        self,
        pixmap,
        shapes,
        show_labels=True,
        show_scores=True,
        show_groups=False,
        show_texts=True,
        show_masks=True,
    ):
        old_shape_scale = Shape.scale
        scratch = type(self)(parent=self.parent)
        scratch.resize(pixmap.size())
        scratch.pixmap = pixmap
        scratch.shapes = list(shapes)
        scratch.scale = 1.0
        scratch.current = None
        scratch.selected_shapes = []
        scratch.selected_shapes_copy = []
        scratch.h_shape = None
        scratch.h_vertex = None
        scratch.h_edge = None
        scratch.h_cuboid_face = None
        scratch.compare_pixmap = None
        scratch.cross_line_show = False
        scratch.show_labels = show_labels
        scratch.show_scores = show_scores
        scratch.show_groups = show_groups
        scratch.show_texts = show_texts
        scratch.show_masks = show_masks
        scratch.show_degrees = self.show_degrees
        scratch.show_attributes = self.show_attributes
        scratch.show_linking = self.show_linking
        scratch.mask_opacity = self.mask_opacity
        scratch.attr_background_color = self.attr_background_color
        scratch.attr_border_color = self.attr_border_color
        scratch.attr_text_color = self.attr_text_color
        scratch.visible = {shape: shape.visible for shape in scratch.shapes}

        image = QtGui.QImage(pixmap.size(), QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(image)
        try:
            scratch.render(painter)
        finally:
            painter.end()
            Shape.scale = old_shape_scale
            scratch.deleteLater()

        return image

    def transform_pos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offset_to_center()

    def offset_to_center(self):
        """Calculate offset to the center"""
        if self.pixmap is None:
            return QtCore.QPointF()
        s = self.scale
        area = super().size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        area_width, area_height = area.width(), area.height()
        x = (area_width - w) / (2 * s) if area_width > w else 0
        y = (area_height - h) / (2 * s) if area_height > h else 0
        return QtCore.QPointF(x, y)

    def out_off_pixmap(self, p):
        """Check if a position is out of pixmap"""
        if self.pixmap is None:
            return True
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        """Finish drawing for a shape"""
        assert self.current
        self._brush_drawing = False
        if (
            self.is_auto_labeling
            and self.auto_labeling_mode != AutoLabelingMode.NONE
        ):
            self.current.label = self.auto_labeling_mode.edit_mode
        if self.current.label is None:
            self.current.label = ""
        self.current.close()
        if self.current.shape_type == "rectangle":
            if not self.clip_rectangle_to_pixmap(self.current):
                self.current = None
                self.set_hiding(False)
                self.drawing_polygon.emit(False)
                self.update()
                return
        elif self.current.shape_type == "rotation":
            if not self.clip_rotation_to_pixmap(self.current):
                self.current = None
                self.set_hiding(False)
                self.drawing_polygon.emit(False)
                self.update()
                return
        elif self.current.shape_type == "cuboid":
            self.current.sync_cuboid_depth_vector()

        self.shapes.append(self.current)
        self.store_shapes()
        self.current = None
        self.set_hiding(False)
        self.new_shape.emit()
        self.update()
        if self.is_auto_labeling:
            self.update_auto_labeling_marks()

    def update_auto_labeling_marks(self):
        """Update the auto labeling marks"""
        marks = []
        for shape in self.shapes:
            if shape.label == AutoLabelingMode.ADD:
                if shape.shape_type == AutoLabelingMode.POINT:
                    marks.append(
                        {
                            "type": "point",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                            ],
                            "label": 1,
                        }
                    )
                elif shape.shape_type == AutoLabelingMode.RECTANGLE:
                    marks.append(
                        {
                            "type": "rectangle",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                                int(shape.points[2].x()),
                                int(shape.points[2].y()),
                            ],
                            "label": 1,
                        }
                    )
            elif shape.label == AutoLabelingMode.REMOVE:
                if shape.shape_type == AutoLabelingMode.POINT:
                    marks.append(
                        {
                            "type": "point",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                            ],
                            "label": 0,
                        }
                    )
                elif shape.shape_type == AutoLabelingMode.RECTANGLE:
                    marks.append(
                        {
                            "type": "rectangle",
                            "data": [
                                int(shape.points[0].x()),
                                int(shape.points[0].y()),
                                int(shape.points[2].x()),
                                int(shape.points[2].y()),
                            ],
                            "label": 0,
                        }
                    )

        self.auto_labeling_marks_updated.emit(marks)

    def close_enough(self, p1, p2):
        """Check if 2 points are close enough (by an threshold epsilon)"""
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersection_point(self, p1, p2):
        """Cycle through each image edge in clockwise fashion,
        and find the one intersecting the current line segment.
        """
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        _, i, (x, y) = min(self.intersecting_edges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        x3, y3 = int(x3), int(y3)
        x4, y4 = int(x4), int(y4)
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            # y3 == y4
            return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(int(x), int(y))

    def intersecting_edges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        x1, y1 = point1
        x2, y2 = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    # QT Overload
    def sizeHint(self):
        """Get size hint"""
        return self.minimumSizeHint()

    # QT Overload
    def minimumSizeHint(self):
        """Get minimum size hint"""
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super().minimumSizeHint()

    # QT Overload
    def wheelEvent(self, ev: QWheelEvent):
        """Mouse wheel event"""
        mods = ev.modifiers()
        delta = ev.angleDelta()

        if self.is_brush_mode:
            # Resize the brush instead of zooming/scrolling.
            if delta.y() == 0:
                ev.accept()
                return
            step = 1 if delta.y() > 0 else -1
            self.brush_radius = int(max(1, min(200, self.brush_radius + step)))
            self.update()
            ev.accept()
            return

        if (
            self.editing()
            and self.enable_wheel_rectangle_editing
            and not self.auto_highlight_shape
            and len(self.selected_shapes) == 1
            and self.selected_shapes[0].shape_type == "rectangle"
            and not self.selected_shapes[0].locked
            and not (mods & QtCore.Qt.KeyboardModifier.ControlModifier)
        ):
            try:
                pos = self.transform_pos(ev.position())
            except AttributeError:
                pos = self.transform_pos(ev.position())

            shape = self.selected_shapes[0]
            wheel_up = delta.y() > 0

            if shape.contains_point(pos):
                self._scale_rectangle(shape, wheel_up)
            else:
                self._adjust_rectangle_edge(shape, pos, wheel_up)

            self.store_shapes()
            self.shape_moved.emit()
            self.update()
            ev.accept()
            return

        # Shift+wheel: adjust compare view split position
        if (
            mods == QtCore.Qt.KeyboardModifier.ShiftModifier
            and self.compare_pixmap is not None
            and not self.compare_pixmap.isNull()
        ):
            step = 0.02 if delta.y() > 0 else -0.02
            self.split_position = max(
                0.0, min(1.0, self.split_position + step)
            )
            self.split_position_changed.emit(self.split_position)
            self.update()
            ev.accept()
            return

        if mods & QtCore.Qt.KeyboardModifier.ControlModifier:
            # with Ctrl/Command key
            # zoom
            self.zoom_request.emit(delta.y(), ev.position().toPoint())
        else:
            # scroll
            self.scroll_request.emit(
                delta.x(), QtCore.Qt.Orientation.Horizontal, 0
            )
            self.scroll_request.emit(
                delta.y(), QtCore.Qt.Orientation.Vertical, 0
            )
        ev.accept()

    def _scale_rectangle(self, shape, scale_up):
        """Scale rectangle from center while keeping within image boundaries"""
        if len(shape.points) < 4:
            return

        if self.pixmap is None:
            return
        img_width = self.pixmap.width()
        img_height = self.pixmap.height()

        x_coords = [p.x() for p in shape.points]
        y_coords = [p.y() for p in shape.points]
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        center = QtCore.QPointF(center_x, center_y)

        scale_factor = (
            1.0 + self.rect_scale_step
            if scale_up
            else 1.0 - self.rect_scale_step
        )
        scale_factor = max(0.1, scale_factor)

        new_points = []
        for i in range(len(shape.points)):
            point = shape.points[i]
            offset = point - center
            scaled_offset = offset * scale_factor
            new_point = center + scaled_offset

            if (
                new_point.x() < 0
                or new_point.x() >= img_width
                or new_point.y() < 0
                or new_point.y() >= img_height
            ):
                return

            new_points.append(new_point)

        for i, new_point in enumerate(new_points):
            shape.points[i] = new_point

    def _adjust_rectangle_edge(self, shape, cursor_pos, move_outward):
        """Adjust the rectangle edge closest to cursor position within image boundaries"""
        if len(shape.points) < 4:
            return

        rect = shape.bounding_rect()
        min_x, max_x = rect.left(), rect.right()
        min_y, max_y = rect.top(), rect.bottom()

        distances = {}

        if cursor_pos.x() < min_x:
            distances["left"] = min_x - cursor_pos.x()
        elif cursor_pos.x() > max_x:
            distances["right"] = cursor_pos.x() - max_x
        else:
            distances["left"] = abs(cursor_pos.x() - min_x)
            distances["right"] = abs(cursor_pos.x() - max_x)

        if cursor_pos.y() < min_y:
            distances["top"] = min_y - cursor_pos.y()
        elif cursor_pos.y() > max_y:
            distances["bottom"] = cursor_pos.y() - max_y
        else:
            distances["top"] = abs(cursor_pos.y() - min_y)
            distances["bottom"] = abs(cursor_pos.y() - max_y)

        if (
            cursor_pos.x() < min_x
            and cursor_pos.y() >= min_y
            and cursor_pos.y() <= max_y
        ):
            closest_edge = "left"
        elif (
            cursor_pos.x() > max_x
            and cursor_pos.y() >= min_y
            and cursor_pos.y() <= max_y
        ):
            closest_edge = "right"
        elif (
            cursor_pos.y() < min_y
            and cursor_pos.x() >= min_x
            and cursor_pos.x() <= max_x
        ):
            closest_edge = "top"
        elif (
            cursor_pos.y() > max_y
            and cursor_pos.x() >= min_x
            and cursor_pos.x() <= max_x
        ):
            closest_edge = "bottom"
        else:
            closest_edge = min(distances, key=distances.get)

        step = (
            self.rect_adjust_step if move_outward else -self.rect_adjust_step
        )

        if self.pixmap is None:
            return
        img_width = self.pixmap.width()
        img_height = self.pixmap.height()

        for i, point in enumerate(shape.points):
            new_point = None

            if closest_edge == "left" and abs(point.x() - min_x) < 1e-6:
                new_x = max(0, point.x() - step)
                new_point = QtCore.QPointF(new_x, point.y())
            elif closest_edge == "right" and abs(point.x() - max_x) < 1e-6:
                new_x = min(img_width - 1, point.x() + step)
                new_point = QtCore.QPointF(new_x, point.y())
            elif closest_edge == "top" and abs(point.y() - min_y) < 1e-6:
                new_y = max(0, point.y() - step)
                new_point = QtCore.QPointF(point.x(), new_y)
            elif closest_edge == "bottom" and abs(point.y() - max_y) < 1e-6:
                new_y = min(img_height - 1, point.y() + step)
                new_point = QtCore.QPointF(point.x(), new_y)

            if new_point is not None:
                shape.points[i] = new_point

    def move_by_keyboard(self, offset):
        """Move selected shapes by an offset (using keyboard)"""
        if self.selected_shapes:
            group_shapes = self._active_group_shapes()
            if group_shapes and any(shape.locked for shape in group_shapes):
                return
            self.bounded_move_shapes(
                self.selected_shapes, self.prev_point + offset
            )
            self.repaint()
            self.moving_shape = True

    def rotate_by_keyboard(self, theta):
        """Rotate selected shapes by an theta (using keyboard)"""
        if self.selected_shapes:
            rotating_shape = False
            for i, shape in enumerate(self.selected_shapes):
                if shape._shape_type == "rotation":
                    self.bounded_rotate_shapes(i, shape, theta)
                    rotating_shape = True
            if rotating_shape:
                self.repaint()
                self.rotating_shape = True

    # QT Overload
    def keyPressEvent(self, ev):  # noqa: C901
        """Key press event"""
        modifiers = ev.modifiers()
        key = ev.key()
        if key == QtCore.Qt.Key.Key_Space and not ev.isAutoRepeat():
            self._space_pressed = True
            self._clear_space_pan_hover_state()
            if self._space_panning:
                self.override_cursor(CURSOR_MOVE)
            else:
                self.override_cursor(CURSOR_GRAB)
            ev.accept()
            return
        if self.is_brush_mode and self.editing():
            if self._brush_key_press(ev):
                return
        if self.drawing():
            if key == QtCore.Qt.Key.Key_Escape and self.current:
                self.current = None
                self._brush_drawing = False
                self.drawing_polygon.emit(False)
                self.update()
            elif key == QtCore.Qt.Key.Key_Backspace and self.current:
                if self.create_mode in ["polygon", "linestrip"]:
                    if len(self.current.points) > 1:
                        self.current.points.pop()
                        self.line[0] = self.current[-1]
                        self.update()
                    elif len(self.current.points) == 1:
                        self.current = None
                        self._brush_drawing = False
                        self.drawing_polygon.emit(False)
                        self.update()
            elif key == QtCore.Qt.Key.Key_Return and self.can_close_shape():
                self.finalise()
            elif modifiers == QtCore.Qt.KeyboardModifier.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == QtCore.Qt.Key.Key_Escape:
                self.deselect_shape()
                return
            if (
                key == QtCore.Qt.Key.Key_Alt
                and self.can_erase_selected_vertices()
            ):
                self.override_cursor(self._vertex_eraser_cursor())
                self._set_vertex_eraser_tooltip()
                ev.accept()
                return
            move_speed = MOVE_SPEED
            if self._active_group_shapes():
                move_speed = (
                    10.0
                    if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier
                    else 1.0
                )
            if key == QtCore.Qt.Key.Key_Up:
                self.move_by_keyboard(QtCore.QPointF(0.0, -move_speed))
            elif key == QtCore.Qt.Key.Key_Down:
                self.move_by_keyboard(QtCore.QPointF(0.0, move_speed))
            elif key == QtCore.Qt.Key.Key_Left:
                self.move_by_keyboard(QtCore.QPointF(-move_speed, 0.0))
            elif key == QtCore.Qt.Key.Key_Right:
                self.move_by_keyboard(QtCore.QPointF(move_speed, 0.0))
            elif key == QtCore.Qt.Key.Key_Z:
                self.rotate_by_keyboard(self.large_rotation_increment)
            elif key == QtCore.Qt.Key.Key_X:
                self.rotate_by_keyboard(self.small_rotation_increment)
            elif key == QtCore.Qt.Key.Key_C:
                self.rotate_by_keyboard(-self.small_rotation_increment)
            elif key == QtCore.Qt.Key.Key_V:
                self.rotate_by_keyboard(-self.large_rotation_increment)

    # QT Overload
    def keyReleaseEvent(self, ev):
        """Key release event"""
        modifiers = ev.modifiers()
        key = ev.key()
        if key == QtCore.Qt.Key.Key_Space and not ev.isAutoRepeat():
            self._space_pressed = False
            self._space_panning = False
            self._space_pan_prev_point = None
            self._restore_space_pan_cursor()
            ev.accept()
            return
        if (
            key == QtCore.Qt.Key.Key_Alt
            and self.editing()
            and not ev.isAutoRepeat()
        ):
            self._vertex_erasing = False
            self.override_cursor(CURSOR_DEFAULT)
            ev.accept()
            return
        if self.drawing():
            if modifiers == QtCore.Qt.KeyboardModifier.NoModifier:
                self.snapping = True
        elif self.editing():
            # NOTE: Temporary fix to avoid ValueError
            # when the selected shape is not in the shapes list
            if (
                (self.moving_shape or self.rotating_shape)
                and self.selected_shapes
                and self.selected_shapes[0] in self.shapes
            ):
                index = self.shapes.index(self.selected_shapes[0])
                if (
                    self.shapes_backups
                    and index < len(self.shapes_backups[-1])
                    and self.shapes_backups[-1][index].points
                    != self.shapes[index].points
                ):
                    self.store_shapes()
                    if self.moving_shape:
                        self.shape_moved.emit()
                    if self.rotating_shape:
                        self.shape_rotated.emit()

                if self.moving_shape:
                    self.moving_shape = False
                if self.rotating_shape:
                    self.rotating_shape = False

    def set_last_label(self, text, flags, group_id):
        """Set label and flags for last shape"""
        assert text
        if self.is_auto_labeling:
            self.shapes[-1].label = self.auto_labeling_mode.edit_mode
        else:
            self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        self.shapes[-1].group_id = group_id
        self.shapes_backups.pop()
        self.store_shapes()
        return self.shapes[-1]

    def undo_last_line(self):
        """Undo last line"""
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.set_open()
        if self.create_mode in ["polygon", "linestrip", "quadrilateral"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.create_mode in [
            "rectangle",
            "line",
            "circle",
            "rotation",
            "cuboid",
        ]:
            self.current.points = self.current.points[0:1]
        elif self.create_mode == "point":
            self.current = None
        self.drawing_polygon.emit(True)

    def undo_last_point(self):
        """Undo last point"""
        if not self.current or self.current.is_closed():
            return
        self.current.pop_point()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self._brush_drawing = False
            self.drawing_polygon.emit(False)
        self.update()

    def load_pixmap(self, pixmap, clear_shapes=True):
        """Load pixmap"""
        self.cancel_brush_mode()
        self.pixmap = pixmap
        if clear_shapes:
            self.shapes = []
        self.update()

    def load_shapes(self, shapes, replace=True):
        """Load shapes"""
        self.cancel_brush_mode()
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.store_shapes()
        self.current = None
        self._brush_drawing = False
        self.h_shape = None
        self.h_vertex = None
        self.h_edge = None
        self.h_cuboid_face = None
        self._selected_group_id = None
        self._hovered_group_id = None
        self.update()

    def set_shape_visible(self, shape, value):
        """Set visibility for a shape"""
        self.visible[shape] = value
        self.update()

    def current_cursor(self):
        """Current cursor"""
        cursor = QtWidgets.QApplication.overrideCursor()
        cursor = cursor.shape() if cursor else None

        return cursor

    def override_cursor(self, cursor):
        """Override cursor"""
        current_cursor = self.current_cursor()
        cursor_shape = (
            cursor.shape() if isinstance(cursor, QtGui.QCursor) else cursor
        )
        if current_cursor != cursor_shape:
            self._cursor = cursor
            if current_cursor is None:
                QtWidgets.QApplication.setOverrideCursor(cursor)
            else:
                QtWidgets.QApplication.changeOverrideCursor(cursor)

    def restore_cursor(self):
        """Restore override cursor"""
        QtWidgets.QApplication.restoreOverrideCursor()

    def reset_state(self):
        """Clear shapes and pixmap"""
        self._clear_space_pan_state()
        self.restore_cursor()
        self.pixmap = None
        self.shapes_backups = []
        self.is_move_editing = False
        self.compare_pixmap = None
        self._selected_group_id = None
        self._hovered_group_id = None
        self.update()

    def set_cross_line(self, show, width, color, opacity):
        """Set cross line options"""
        self.cross_line_show = show
        self.cross_line_width = width
        self.cross_line_color = color
        self.cross_line_opacity = opacity
        self.update()

    def gen_new_group_id(self):
        """Generate new shape's group_id based on current shapes"""
        max_group_id = 0
        for shape in self.shapes:
            if shape.group_id is not None:
                max_group_id = max(max_group_id, shape.group_id)
        return max_group_id + 1

    def merge_group_ids(self, group_ids, new_group_id):
        """Merge multiple shapes' group_id into a new one"""
        for shape in self.shapes:
            if shape.group_id in group_ids:
                shape.group_id = new_group_id

    def group_selected_shapes(self):
        """Group selected shapes"""
        if len(self.selected_shapes) == 0:
            return

        # List all group ids for selected shapes
        group_ids = set()
        has_non_group_shape = False
        for shape in self.selected_shapes:
            if shape.group_id is not None:
                group_ids.add(shape.group_id)
            else:
                has_non_group_shape = True

        # If there is at least 1 shape having a group id,
        # use that id as the new group id. Otherwise, generate a new group_id
        new_group_id = None
        if len(group_ids) > 0:
            new_group_id = min(group_ids)
        else:
            new_group_id = self.gen_new_group_id()

        # Merge group ids
        if len(group_ids) > 1:
            self.merge_group_ids(
                group_ids=group_ids, new_group_id=new_group_id
            )
        # Assign new_group_id to non-group shapes
        if has_non_group_shape:
            for shape in self.selected_shapes:
                if shape.group_id is None:
                    shape.group_id = new_group_id

        self.update()

    def ungroup_selected_shapes(self):
        """Ungroup selected shapes"""
        if len(self.selected_shapes) == 0:
            return

        # List all group ids for selected shapes
        group_ids = set()
        for shape in self.selected_shapes:
            if shape.group_id is not None:
                group_ids.add(shape.group_id)

        for group_id in group_ids:
            for shape in self.shapes:
                if shape.group_id == group_id:
                    shape.group_id = None

        self.update()
