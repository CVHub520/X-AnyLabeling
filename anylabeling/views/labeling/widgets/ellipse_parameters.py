"""Small numeric editor for the selected rotatable ellipse."""

from PyQt6 import QtCore, QtWidgets


class EllipseParameterBar(QtWidgets.QWidget):
    geometry_committed = QtCore.pyqtSignal(float, float, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Create the parameter controls.

        Args:
            parent: Widget owning the parameter bar.
        """
        super().__init__(parent)
        self.setObjectName("ellipseParameterBar")
        self._updating = False
        self._displayed = None
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)
        layout.addWidget(QtWidgets.QLabel(self.tr("Ellipse")))
        self.width_spin = self._add_spin(
            layout, self.tr("Width (px)"), 2, 1.0, 1000000.0, 1.0
        )
        self.height_spin = self._add_spin(
            layout, self.tr("Height (px)"), 2, 1.0, 1000000.0, 1.0
        )
        self.angle_spin = self._add_spin(
            layout, self.tr("Angle (°)"), 1, 0.0, 360.0, 0.1
        )
        layout.addStretch()
        for spin in (self.width_spin, self.height_spin, self.angle_spin):
            spin.editingFinished.connect(self._commit)
            spin.installEventFilter(self)
        self.hide()

    def _add_spin(
        self,
        layout: QtWidgets.QHBoxLayout,
        label: str,
        decimals: int,
        minimum: float,
        maximum: float,
        step: float,
    ) -> QtWidgets.QDoubleSpinBox:
        """Add one numeric control.

        Args:
            layout: Destination layout.
            label: Translated field label.
            decimals: Display precision.
            minimum: Minimum allowed value.
            maximum: Maximum allowed value.
            step: Arrow-key increment.

        Returns:
            The configured numeric widget.
        """
        layout.addWidget(QtWidgets.QLabel(label))
        spin = QtWidgets.QDoubleSpinBox(self)
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setKeyboardTracking(False)
        spin.setFixedWidth(100)
        layout.addWidget(spin)
        return spin

    def values(self) -> tuple[float, float, float]:
        """Return the displayed full width, height and clockwise angle."""
        return (
            self.width_spin.value(),
            self.height_spin.value(),
            self.angle_spin.value(),
        )

    def show_geometry(
        self, width: float, height: float, angle: float, editable: bool = True
    ) -> None:
        """Refresh controls without emitting a geometry commit.

        Args:
            width: Full width in pixels.
            height: Full height in pixels.
            angle: Clockwise angle in degrees.
            editable: Whether the selected ellipse is unlocked.
        """
        self._updating = True
        try:
            for spin, value in zip(
                (self.width_spin, self.height_spin, self.angle_spin),
                (width, height, angle),
            ):
                spin.setValue(value)
            self._displayed = self.values()
            self.setEnabled(editable)
            self.show()
        finally:
            self._updating = False

    def _commit(self) -> None:
        """Emit one commit when a numeric editing session finishes."""
        if self._updating or not self.isEnabled():
            return
        values = self.values()
        if values != self._displayed:
            self._displayed = values
            self.geometry_committed.emit(*values)

    def eventFilter(
        self, watched: QtCore.QObject, event: QtCore.QEvent
    ) -> bool:
        """Cancel uncommitted numeric input with Escape.

        Args:
            watched: Numeric control receiving the event.
            event: Incoming Qt event.

        Returns:
            Whether the event was handled.
        """
        if (
            watched in (self.width_spin, self.height_spin, self.angle_spin)
            and event.type() == QtCore.QEvent.Type.KeyPress
            and event.key() == QtCore.Qt.Key.Key_Escape
        ):
            if self._displayed is not None:
                self.show_geometry(*self._displayed, editable=self.isEnabled())
            return True
        return super().eventFilter(watched, event)
