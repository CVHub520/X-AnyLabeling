"""A Photoshop-style widget for image navigation"""

from typing import List, Optional, Any

from PyQt5 import QtWidgets
from PyQt5.QtCore import (
    QPoint,
    QRect,
    QSize,
    Qt,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.chatbot import ChatbotDialogStyle


class ClickableSlider(QSlider):
    """Custom slider that supports clicking anywhere on the track to jump to position"""

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for click-to-jump functionality"""
        if event.button() == Qt.LeftButton:
            if self.orientation() == Qt.Horizontal:
                handle_width = self.style().pixelMetric(
                    self.style().PM_SliderThickness
                )
                slider_min = self.minimum()
                slider_max = self.maximum()
                current_value = self.value()
                slider_width = self.width() - handle_width

                if slider_max > slider_min:
                    handle_ratio = (current_value - slider_min) / (
                        slider_max - slider_min
                    )
                    handle_pos = (
                        handle_width // 2 + handle_ratio * slider_width
                    )

                    click_x = event.x()
                    if abs(click_x - handle_pos) <= handle_width // 2 + 5:
                        super().mousePressEvent(event)
                        return

                # Click is on track, jump to position
                click_x = event.x()
                effective_x = max(
                    handle_width // 2,
                    min(slider_width + handle_width // 2, click_x),
                )
                ratio = (effective_x - handle_width // 2) / slider_width
                new_value = slider_min + ratio * (slider_max - slider_min)
                new_value = max(slider_min, min(slider_max, int(new_value)))
                self.setValue(new_value)
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)


class NavigatorWidget(QWidget):
    """Navigator widget showing thumbnail with viewport rectangle"""

    navigation_requested = pyqtSignal(float, float)  # x_ratio, y_ratio
    viewport_update_needed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Widget properties
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setWindowTitle(self.tr("Navigator"))

        # Image and viewport data
        self.original_image = None
        self.thumbnail = None
        self.viewport_rect = QRect()
        self.image_rect = QRect()

        # Shapes data for overlay
        self.shapes = []  # List of shapes to draw on thumbnail
        self.visible_shapes = {}

        # Interaction state
        self.dragging = False
        self.last_drag_pos = QPoint()

        # Styling
        self.viewport_pen = QPen(QColor(255, 0, 0, 255), 2)  # Red pen
        self.background_brush = QBrush(QColor(64, 64, 64))  # Dark background
        self.shape_pen = QPen(
            QColor(0, 255, 0, 180), 1
        )  # Green pen for shapes

        self.setMouseTracking(True)

    def set_image(self, image_data: Any) -> None:
        """Set the image to display in the navigator widget."""
        if image_data is None:
            self.original_image = None
            self.thumbnail = None
            self.update()
            return

        if isinstance(image_data, bytes):
            pixmap = QPixmap()
            pixmap.loadFromData(image_data)
            self.original_image = pixmap
        elif isinstance(image_data, QPixmap):
            self.original_image = image_data
        else:
            try:
                self.original_image = QPixmap(str(image_data))
            except:
                return

        self._update_thumbnail()
        self.update()

    def _update_thumbnail(self):
        """Update thumbnail to fit widget size"""
        if not self.original_image or self.original_image.isNull():
            return

        widget_size = self.size()
        available_size = QSize(
            widget_size.width() - 20, widget_size.height() - 20
        )

        # Scale image to fit available space while keeping aspect ratio
        self.thumbnail = self.original_image.scaled(
            available_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Calculate thumbnail position (centered)
        thumb_size = self.thumbnail.size()
        x = (widget_size.width() - thumb_size.width()) // 2
        y = (widget_size.height() - thumb_size.height()) // 2
        self.image_rect = QRect(x, y, thumb_size.width(), thumb_size.height())

    def set_viewport(
        self,
        x_ratio: float,
        y_ratio: float,
        width_ratio: float,
        height_ratio: float,
    ) -> None:
        """Set the viewport rectangle that shows the visible area of the main canvas."""
        if not self.thumbnail or self.image_rect.isEmpty():
            return

        # Convert ratios to pixel coordinates within thumbnail
        thumb_width = self.image_rect.width()
        thumb_height = self.image_rect.height()
        x = int(self.image_rect.x() + x_ratio * thumb_width)
        y = int(self.image_rect.y() + y_ratio * thumb_height)
        width = max(1, int(width_ratio * thumb_width))
        height = max(1, int(height_ratio * thumb_height))

        self.viewport_rect = QRect(x, y, width, height)
        self.update()

    def set_shapes(
        self,
        shapes: Optional[List[Any]],
        visible_shapes: Optional[dict] = None,
    ) -> None:
        """Set the shapes to display on the thumbnail overlay."""
        self.shapes = shapes if shapes else []
        self.visible_shapes = visible_shapes or {}
        self.update()

    def resizeEvent(self, event) -> None:
        """Handle widget resize events to maintain proper thumbnail display."""
        super().resizeEvent(event)
        self._update_thumbnail()
        self.update()
        self.viewport_update_needed.emit()

    def paintEvent(self, event) -> None:
        """Paint the navigator widget with thumbnail, shapes, and viewport overlay."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.background_brush)

        if self.thumbnail and not self.thumbnail.isNull():
            painter.drawPixmap(self.image_rect, self.thumbnail)
            self._draw_shapes_overlay(painter)
            if not self.viewport_rect.isEmpty():
                painter.setPen(self.viewport_pen)
                painter.setBrush(QBrush(Qt.NoBrush))  # No fill
                painter.drawRect(self.viewport_rect)

    def _draw_shapes_overlay(self, painter):
        """Draw shapes overlay on thumbnail"""
        if not self.shapes or self.image_rect.isEmpty():
            return

        if not self.original_image or self.original_image.isNull():
            return

        original_width = self.original_image.width()
        original_height = self.original_image.height()
        if original_width <= 0 or original_height <= 0:
            return

        # Draw each shape
        for shape in self.shapes:
            if not hasattr(shape, "points") or not shape.points:
                continue

            # Skip hidden shapes
            if shape in self.visible_shapes and not self.visible_shapes[shape]:
                continue

            # Get shape color and brush
            shape_color = self._get_shape_color(shape)
            shape_brush = self._get_shape_brush(shape)

            # Use thicker lines for better visibility in thumbnail
            line_width = 2
            painter.setPen(QPen(shape_color, line_width))
            painter.setBrush(shape_brush)

            # Convert shape points to thumbnail coordinates
            thumbnail_points = []
            for point in shape.points:
                thumb_x = (
                    self.image_rect.x()
                    + (point.x() / original_width) * self.image_rect.width()
                )
                thumb_y = (
                    self.image_rect.y()
                    + (point.y() / original_height) * self.image_rect.height()
                )
                thumbnail_points.append(QPoint(int(thumb_x), int(thumb_y)))

            if self._points_in_bounds(thumbnail_points):
                if hasattr(shape, "shape_type"):
                    if shape.shape_type in ["rectangle", "rotation"]:
                        self._draw_rectangle_on_thumbnail(
                            painter, thumbnail_points
                        )
                    elif shape.shape_type == "polygon":
                        self._draw_polygon_on_thumbnail(
                            painter, thumbnail_points
                        )
                    elif shape.shape_type == "circle":
                        self._draw_circle_on_thumbnail(
                            painter, thumbnail_points
                        )
                    elif shape.shape_type == "line":
                        self._draw_line_on_thumbnail(painter, thumbnail_points)
                    elif shape.shape_type == "linestrip":
                        self._draw_linestrip_on_thumbnail(
                            painter, thumbnail_points
                        )
                    elif shape.shape_type == "point":
                        self._draw_point_on_thumbnail(
                            painter, thumbnail_points
                        )
                else:
                    self._draw_polygon_on_thumbnail(painter, thumbnail_points)

    def _get_shape_color(self, shape: Any) -> QColor:
        """Get the display color for a shape following main interface logic."""
        # Hover highlighting takes precedence - use bright yellow
        if hasattr(shape, "_is_highlighted") and shape._is_highlighted:
            return QColor(255, 255, 0)  # Bright yellow for hover state

        # Handle selection state
        if hasattr(shape, "selected") and shape.selected:
            if hasattr(shape, "select_line_color") and shape.select_line_color:
                color = shape.select_line_color
                if color and color.isValid():
                    return color

        # Use normal line color
        if hasattr(shape, "line_color") and shape.line_color:
            color = shape.line_color
            if color and color.isValid():
                return color

        # Fallback to visible green
        return QColor(0, 255, 0)

    def _get_shape_brush(self, shape: Any) -> QBrush:
        """Get the fill brush for a shape following main interface logic."""
        should_fill: bool = getattr(shape, "fill", False)

        if not should_fill:
            return QBrush(Qt.NoBrush)

        fill_color: Optional[QColor] = None

        if hasattr(shape, "selected") and shape.selected:
            if hasattr(shape, "select_fill_color") and shape.select_fill_color:
                fill_color = shape.select_fill_color

        if not fill_color:
            if hasattr(shape, "fill_color") and shape.fill_color:
                fill_color = shape.fill_color

        if fill_color and fill_color.isValid():
            return QBrush(fill_color)

        return QBrush(Qt.NoBrush)

    def _points_in_bounds(self, points: List[QPoint]) -> bool:
        """Check if shape points are within reasonable bounds for rendering."""
        if not points:
            return False

        # Use generous margin to avoid culling partially visible shapes
        margin = 100
        bounds = self.image_rect.adjusted(-margin, -margin, margin, margin)

        # First check: any point within bounds
        has_point_in_bounds = False
        for point in points:
            if bounds.contains(point):
                has_point_in_bounds = True
                break

        # Second check: shape bounding box intersects with image area
        if not has_point_in_bounds and len(points) >= 2:
            min_x = min(p.x() for p in points)
            max_x = max(p.x() for p in points)
            min_y = min(p.y() for p in points)
            max_y = max(p.y() for p in points)
            shape_rect = QRect(min_x, min_y, max_x - min_x, max_y - min_y)
            return self.image_rect.intersects(shape_rect)

        return has_point_in_bounds

    def _draw_rectangle_on_thumbnail(self, painter, points):
        """Draw rectangle on thumbnail"""
        if len(points) >= 2:
            if len(points) == 2:
                # Standard rectangle from two points
                rect = QRect(points[0], points[1])
                painter.drawRect(rect)
            else:
                # Rotated rectangle - draw as polygon
                self._draw_polygon_on_thumbnail(painter, points)

    def _draw_polygon_on_thumbnail(self, painter, points):
        """Draw polygon on thumbnail"""
        if len(points) >= 2:
            from PyQt5.QtGui import QPolygon

            polygon = QPolygon(points)
            painter.drawPolygon(polygon)

    def _draw_circle_on_thumbnail(self, painter, points):
        """Draw circle on thumbnail"""
        if len(points) >= 2:
            center = points[0]
            radius_point = points[1]
            radius = int(
                (
                    (radius_point.x() - center.x()) ** 2
                    + (radius_point.y() - center.y()) ** 2
                )
                ** 0.5
            )
            painter.drawEllipse(center, radius, radius)

    def _draw_line_on_thumbnail(self, painter, points):
        """Draw line on thumbnail"""
        if len(points) >= 2:
            painter.drawLine(points[0], points[1])

    def _draw_linestrip_on_thumbnail(
        self, painter: QPainter, points: List[QPoint]
    ) -> None:
        """Draw a linestrip (connected line segments) on the thumbnail."""
        if len(points) >= 2:
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])

    def _draw_point_on_thumbnail(self, painter, points):
        """Draw point on thumbnail"""
        if len(points) >= 1:
            point_size = 2
            painter.fillRect(
                points[0].x() - point_size,
                points[0].y() - point_size,
                point_size * 2,
                point_size * 2,
                self.shape_pen.color(),
            )

    def mousePressEvent(self, event) -> None:
        """Handle mouse press events for navigation interaction."""
        if event.button() == Qt.LeftButton and self.image_rect.contains(
            event.pos()
        ):
            self.dragging = True
            self.last_drag_pos = event.pos()
            self._emit_navigation_signal(event.pos())

    def mouseMoveEvent(self, event) -> None:
        """Handle mouse move events during navigation dragging."""
        if self.dragging and self.image_rect.contains(event.pos()):
            self._emit_navigation_signal(event.pos())
            self.last_drag_pos = event.pos()

    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release events to end navigation interaction."""
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel events for navigator zoom functionality."""
        # Forward wheel events to parent dialog if it exists
        if hasattr(self.parent(), "handle_wheel_zoom"):
            self.parent().handle_wheel_zoom(event)
        else:
            event.accept()

    def _emit_navigation_signal(self, pos):
        """Emit navigation signal with position ratios"""
        if self.image_rect.isEmpty():
            return

        # Convert widget coordinates to ratios
        relative_x = pos.x() - self.image_rect.x()
        relative_y = pos.y() - self.image_rect.y()

        x_ratio = max(0.0, min(1.0, relative_x / self.image_rect.width()))
        y_ratio = max(0.0, min(1.0, relative_y / self.image_rect.height()))

        self.navigation_requested.emit(x_ratio, y_ratio)


class NavigatorDialog(QtWidgets.QDialog):
    """Standalone navigator window with zoom controls"""

    zoom_changed = pyqtSignal([int], [int, QPoint])
    viewport_update_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle(self.tr("Navigator"))
        self.setWindowFlags(
            Qt.Window | Qt.WindowStaysOnTopHint | Qt.WindowCloseButtonHint
        )

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        self.navigator = NavigatorWidget(self)
        main_layout.addWidget(self.navigator, 1)

        self.navigator.viewport_update_needed.connect(
            self.viewport_update_requested.emit
        )

        zoom_container = QWidget()
        zoom_container.setFixedHeight(35)
        zoom_layout = QHBoxLayout(zoom_container)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        zoom_layout.setSpacing(8)

        zoom_input_container = QWidget()
        zoom_input_container.setFixedSize(60, 24)
        zoom_input_container.setStyleSheet(
            """
            QWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #ffffff;
            }
            QWidget:hover {
                background-color: #f5f5f5;
            }
        """
        )

        zoom_input_layout = QHBoxLayout(zoom_input_container)
        zoom_input_layout.setContentsMargins(6, 2, 6, 2)
        zoom_input_layout.setSpacing(0)

        self.zoom_input = QLineEdit()
        self.zoom_input.setFixedWidth(35)
        self.zoom_input.setAlignment(Qt.AlignRight)
        self.zoom_input.setText("100")
        self.zoom_input.setStyleSheet(
            """
            QLineEdit {
                border: none;
                background: transparent;
                color: #495057;
                font-size: 10px;
                font-weight: 500;
                padding: 0px;
            }
            QLineEdit:focus {
                border: none;
                background: transparent;
            }
        """
        )
        self.zoom_input.returnPressed.connect(self.on_zoom_input_changed)
        self.zoom_input.editingFinished.connect(self.on_zoom_input_changed)

        percentage_label = QLabel("%")
        percentage_label.setStyleSheet(
            """
            QLabel { 
                color: #6c757d; 
                font-size: 10px; 
                font-weight: 500;
                background: transparent;
                border: none;
                padding: 0px;
            }
        """
        )

        zoom_input_layout.addWidget(self.zoom_input)
        zoom_input_layout.addWidget(percentage_label)

        zoom_out_icon = QLabel("−")
        zoom_out_icon.setFixedSize(16, 16)
        zoom_out_icon.setAlignment(Qt.AlignCenter)
        zoom_out_icon.setStyleSheet(
            """
            QLabel { 
                color: #6c757d; 
                font-size: 14px; 
                font-weight: bold;
                background: transparent;
            }
        """
        )

        self.zoom_slider = ClickableSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 1000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setStyleSheet(ChatbotDialogStyle.get_slider_style())
        self.zoom_slider.valueChanged.connect(self.on_slider_changed)

        zoom_in_icon = QLabel("+")
        zoom_in_icon.setFixedSize(16, 16)
        zoom_in_icon.setAlignment(Qt.AlignCenter)
        zoom_in_icon.setStyleSheet(
            """
            QLabel { 
                color: #6c757d; 
                font-size: 14px; 
                font-weight: bold;
                background: transparent;
            }
        """
        )

        zoom_layout.addWidget(zoom_input_container)
        zoom_layout.addWidget(zoom_out_icon)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(zoom_in_icon)

        main_layout.addWidget(zoom_container, 0)

        self.setLayout(main_layout)
        self.resize(220, 280)
        self.setMinimumSize(180, 220)

        self.current_zoom = 100

    def resizeEvent(self, event):
        """Handle dialog resize"""
        super().resizeEvent(event)
        self.viewport_update_requested.emit()

    def set_image(self, image_data):
        """Set image in navigator"""
        self.navigator.set_image(image_data)

    def set_viewport(self, x_ratio, y_ratio, width_ratio, height_ratio):
        """Set viewport in navigator"""
        self.navigator.set_viewport(
            x_ratio, y_ratio, width_ratio, height_ratio
        )

    def set_shapes(self, shapes, visible_shapes=None):
        """Set shapes to display in navigator"""
        self.navigator.set_shapes(shapes, visible_shapes)

    def set_zoom_value(self, zoom_percentage: int) -> None:
        """Set zoom value and update UI elements"""
        self.current_zoom = zoom_percentage

        self.zoom_slider.blockSignals(True)
        self.zoom_input.blockSignals(True)

        self.zoom_slider.setValue(zoom_percentage)
        self.zoom_input.setText(str(zoom_percentage))

        self.zoom_slider.blockSignals(False)
        self.zoom_input.blockSignals(False)

    def on_slider_changed(self, value):
        """Handle slider value change event"""
        self.current_zoom = value
        self.zoom_input.setText(str(value))
        self.zoom_changed[int].emit(value)

    def on_zoom_input_changed(self):
        """Handle zoom input text change event"""
        try:
            value = int(self.zoom_input.text())
            value = max(1, min(1000, value))

            self.current_zoom = value
            self.zoom_slider.setValue(value)
            self.zoom_input.setText(str(value))
            self.zoom_changed[int].emit(value)
        except ValueError:
            self.zoom_input.setText(str(self.current_zoom))

    def zoom_in(self):
        """Increase zoom level by 1"""
        new_zoom = min(1000, self.current_zoom + 1)
        self.set_zoom_value(new_zoom)
        self.zoom_changed[int].emit(new_zoom)

    def zoom_out(self):
        """Decrease zoom level by 1"""
        new_zoom = max(1, self.current_zoom - 1)
        self.set_zoom_value(new_zoom)
        self.zoom_changed[int].emit(new_zoom)

    def handle_wheel_zoom(self, event) -> None:
        """Handle mouse wheel zoom event"""
        delta = event.angleDelta().y()

        if delta > 0:
            zoom_increment = 1
        elif delta < 0:
            zoom_increment = -1
        else:
            zoom_increment = 0

        new_zoom = self.current_zoom + zoom_increment
        new_zoom = max(1, min(1000, new_zoom))

        self.set_zoom_value(new_zoom)
        self.zoom_changed[int, QPoint].emit(new_zoom, event.pos())

        event.accept()
