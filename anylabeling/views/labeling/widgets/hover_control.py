from PyQt6 import QtCore, QtWidgets


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
