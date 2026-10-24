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


class PanelHeader(QtWidgets.QWidget):
    def __init__(self, title, controls):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(0)
        actions = QtWidgets.QWidget()
        actions_layout = QtWidgets.QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(2)
        theme = get_theme()
        for control in controls:
            control.setFixedSize(18, 20)
            if isinstance(control, QtWidgets.QToolButton):
                control.setFixedSize(22, 22)
                control.setIconSize(QtCore.QSize(14, 14))
                control.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
                control.setStyleSheet(f"""
                    QToolButton {{
                        background: transparent;
                        border: none;
                        border-radius: 6px;
                        padding: 0;
                    }}
                    QToolButton:checked {{ background: transparent; }}
                    QToolButton:hover {{ background: {theme['surface_hover']}; }}
                    QToolButton:pressed {{ background: {theme['surface_pressed']}; }}
                """)
            actions_layout.addWidget(control)
        width = (
            sum(control.width() for control in controls)
            + (len(controls) - 1) * 2
        )
        layout.addSpacing(width)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title, 1)
        actions.setFixedWidth(width)
        layout.addWidget(actions)


class HoverControl(QtCore.QObject):
    def __init__(self, panel, control):
        super().__init__(panel)
        self._control = control
        self._hovered = False
        self._effect = QtWidgets.QGraphicsOpacityEffect(control)
        self._effect.setOpacity(0.0)
        control.setGraphicsEffect(self._effect)
        panel.installEventFilter(self)
        control.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.parent():
            if event.type() == QtCore.QEvent.Type.Enter:
                self._hovered = True
            elif event.type() in (
                QtCore.QEvent.Type.Leave,
                QtCore.QEvent.Type.Hide,
            ):
                self._hovered = False
            else:
                return super().eventFilter(obj, event)
        elif event.type() not in (
            QtCore.QEvent.Type.FocusIn,
            QtCore.QEvent.Type.FocusOut,
        ):
            return super().eventFilter(obj, event)
        visible = self._hovered or self._control.hasFocus()
        self._effect.setOpacity(1.0 if visible else 0.0)
        return super().eventFilter(obj, event)


class HoverScrollBars(QtCore.QObject):
    def __init__(self, area):
        super().__init__(area)
        self._area = area
        self._hovered = False
        self._sources = (area.verticalScrollBar(), area.horizontalScrollBar())
        self._bars = tuple(
            QtWidgets.QScrollBar(source.orientation(), area.viewport())
            for source in self._sources
        )
        for source, bar in zip(self._sources, self._bars):
            source.valueChanged.connect(bar.setValue)
            source.valueChanged.connect(self._update_geometry)
            bar.valueChanged.connect(source.setValue)
            source.rangeChanged.connect(self._update_geometry)
            bar.sliderReleased.connect(self._update_visibility)
            bar.hide()
        area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        area.installEventFilter(self)
        area.viewport().installEventFilter(self)
        self._update_geometry()

    def _update_geometry(self):
        viewport = self._area.viewport()
        vertical, horizontal = self._bars
        vertical_source, horizontal_source = self._sources
        width = vertical.sizeHint().width()
        height = horizontal.sizeHint().height()
        has_vertical = vertical_source.maximum() > vertical_source.minimum()
        has_horizontal = (
            horizontal_source.maximum() > horizontal_source.minimum()
        )
        vertical.setGeometry(
            viewport.width() - width,
            0,
            width,
            viewport.height() - (height if has_horizontal else 0),
        )
        horizontal.setGeometry(
            0,
            viewport.height() - height,
            viewport.width() - (width if has_vertical else 0),
            height,
        )
        for source, bar in zip(self._sources, self._bars):
            with QtCore.QSignalBlocker(bar):
                bar.setRange(source.minimum(), source.maximum())
                bar.setPageStep(source.pageStep())
                bar.setSingleStep(source.singleStep())
                bar.setValue(source.value())
            bar.raise_()
        self._update_visibility()

    def _update_visibility(self):
        visible = self._hovered or any(
            bar.isSliderDown() for bar in self._bars
        )
        for bar in self._bars:
            bar.setVisible(visible and bar.maximum() > bar.minimum())

    def eventFilter(self, obj, event):
        if event.type() in (
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Show,
        ):
            self._update_geometry()
        elif obj is self._area and event.type() == QtCore.QEvent.Type.Enter:
            self._hovered = True
            self._update_geometry()
        elif obj is self._area and event.type() in (
            QtCore.QEvent.Type.Leave,
            QtCore.QEvent.Type.Hide,
        ):
            self._hovered = False
            self._update_visibility()
        return super().eventFilter(obj, event)
