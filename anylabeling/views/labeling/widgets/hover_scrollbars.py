from PyQt6 import QtCore, QtWidgets


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
