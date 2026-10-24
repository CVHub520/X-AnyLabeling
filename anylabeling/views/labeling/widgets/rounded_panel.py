from PyQt6 import QtCore, QtGui, QtWidgets

from ..utils.style import get_panel_style
from ..utils.theme import get_theme


class _PanelCorner(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: none;")

    def paintEvent(self, event):
        panel = self.parentWidget()
        rect = QtCore.QRectF(panel.rect())
        rect.translate(-self.x(), -self.y())
        rounded = QtGui.QPainterPath()
        rounded.addRoundedRect(
            rect.adjusted(0.5, 0.5, -0.5, -0.5), panel.radius, panel.radius
        )
        outside = QtGui.QPainterPath()
        outside.addRect(rect)
        outside.addPath(rounded)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillPath(
            outside,
            panel.parentWidget()
            .palette()
            .color(QtGui.QPalette.ColorRole.Window),
        )
        painter.setPen(QtGui.QPen(QtGui.QColor(get_theme()["border"]), 1))
        painter.drawPath(rounded)


class RoundedPanel(QtWidgets.QFrame):
    def __init__(self, parent=None, radius=12):
        super().__init__(parent)
        self.radius = radius
        self.setObjectName("sidebarPanel")
        self.setStyleSheet(get_panel_style(radius))
        self._corners = [_PanelCorner(self) for _ in range(4)]

    def resizeEvent(self, event):
        super().resizeEvent(event)
        size = self.radius + 2
        for corner, (x, y) in zip(
            self._corners,
            (
                (0, 0),
                (self.width() - size, 0),
                (0, self.height() - size),
                (self.width() - size, self.height() - size),
            ),
        ):
            corner.setGeometry(x, y, size, size)
            corner.raise_()
            corner.update()

    def showEvent(self, event):
        super().showEvent(event)
        for corner in self._corners:
            corner.raise_()
