from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt5.QtGui import QPainter, QColor, QBrush


class PulsatingDot(QWidget):
    def __init__(
        self,
        parent=None,
        size_range=(10, 30),
        color_range=((0, 0, 0), (120, 120, 120)),
        duration=1000,
    ):
        """
        Create a pulsating dot animation widget

        Args:
            parent: Parent widget
            size_range: Tuple of (min_size, max_size) in pixels
            color_range: Tuple of ((min_r, min_g, min_b), (max_r, max_g, max_b))
            duration: Animation cycle duration in milliseconds
        """
        super().__init__(parent)

        # Animation parameters
        self._min_size, self._max_size = size_range
        self._min_color, self._max_color = color_range
        self._current_size = self._max_size
        self._current_color = QColor(*self._min_color)
        self._color_progress = 0  # Initialize color progress value

        # Set fixed size to max_size to ensure proper layout
        self.setFixedSize(self._max_size + 4, self._max_size + 4)

        # Initialize animation objects first before any getter/setter is called
        self._size_animation = QPropertyAnimation(self, b"dot_size")
        self._color_animation = QPropertyAnimation(self, b"dot_color")

        # Set up size animation
        self._size_animation.setDuration(duration)
        self._size_animation.setStartValue(self._max_size)
        self._size_animation.setEndValue(self._min_size)
        self._size_animation.setEasingCurve(QEasingCurve.InOutQuad)

        # Set up color animation
        self._color_animation.setDuration(duration)
        self._color_animation.setStartValue(0)
        self._color_animation.setEndValue(100)
        self._color_animation.setEasingCurve(QEasingCurve.InOutQuad)

        # Connect animations to loop
        self._size_animation.finished.connect(self._toggle_size_animation)
        self._color_animation.finished.connect(self._toggle_color_animation)

        # Start animations
        self._size_animation.start()
        self._color_animation.start()

    def _toggle_size_animation(self):
        """Toggle the direction of the size animation"""
        start_val, end_val = (
            self._size_animation.endValue(),
            self._size_animation.startValue(),
        )
        self._size_animation.setStartValue(start_val)
        self._size_animation.setEndValue(end_val)
        self._size_animation.start()

    def _toggle_color_animation(self):
        """Toggle the direction of the color animation"""
        start_val, end_val = (
            self._color_animation.endValue(),
            self._color_animation.startValue(),
        )
        self._color_animation.setStartValue(start_val)
        self._color_animation.setEndValue(end_val)
        self._color_animation.start()

    def get_dot_size(self):
        """Property getter for dot size"""
        return self._current_size

    def set_dot_size(self, size):
        """Property setter for dot size"""
        self._current_size = size
        self.update()

    # Define the dot_size property
    dot_size = pyqtProperty(float, get_dot_size, set_dot_size)

    def get_dot_color(self):
        """Property getter for color animation progress"""
        return self._color_progress

    def set_dot_color(self, progress):
        """Property setter for color animation progress (0-100)"""
        self._color_progress = progress

        # Interpolate between min and max colors
        r = (
            self._min_color[0]
            + (self._max_color[0] - self._min_color[0]) * progress / 100
        )
        g = (
            self._min_color[1]
            + (self._max_color[1] - self._min_color[1]) * progress / 100
        )
        b = (
            self._min_color[2]
            + (self._max_color[2] - self._min_color[2]) * progress / 100
        )

        self._current_color = QColor(int(r), int(g), int(b))
        self.update()

    # Define the dot_color property
    dot_color = pyqtProperty(float, get_dot_color, set_dot_color)

    def paintEvent(self, event):
        """Paint the pulsating dot"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate center position
        width, height = self.width(), self.height()
        center_x, center_y = width // 2, height // 2

        # Draw dot
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self._current_color))
        painter.drawEllipse(
            int(center_x - self._current_size // 2),
            int(center_y - self._current_size // 2),
            int(self._current_size),
            int(self._current_size),
        )

    def stop_animation(self):
        """Stop all animations"""
        if hasattr(self, "_size_animation"):
            self._size_animation.stop()
        if hasattr(self, "_color_animation"):
            self._color_animation.stop()
