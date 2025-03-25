from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPainter, QPainterPath, QColor


class Popup(QWidget):
    def __init__(self, text, parent=None, msec=5000):
        super().__init__(parent, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        self.setStyleSheet("""
            QWidget {
                background-color: #f2edec;
                border-radius: 16px;
            }
        """)

        layout = QVBoxLayout()

        self.label = QLabel(text)
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignLeft)
        layout.addWidget(self.label)

        self.setLayout(layout)

        # Set window properties
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # Add drop shadow effect
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(16)
        self.shadow.setColor(QColor(0, 0, 0, 80))
        self.shadow.setOffset(0, 3)
        self.setGraphicsEffect(self.shadow)

        # Create auto-close timer
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.close)
        self.timer.start(msec)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        path = QPainterPath()
        rect = QRectF(self.rect())
        path.addRoundedRect(rect, 10, 10)

        painter.fillPath(path, QColor("#f2edec"))

    def show_popup(self, parent_widget, popup_width=350, popup_height=50):
        # Calculate the position to place the popup at the top center
        parent_geo = parent_widget.geometry()

        # Position popup at the top center with a small margin from the top
        x = parent_geo.x() + (parent_geo.width() - popup_width) // 2
        y = parent_geo.y() + 20

        self.setGeometry(x, y, popup_width, popup_height)
        self.show()
