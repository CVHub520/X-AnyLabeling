from PyQt5 import QtCore, QtGui, QtWidgets


class ZoomWidget(QtWidgets.QSpinBox):
    def __init__(self, value=100):
        super().__init__()
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.setRange(1, 1000)
        self.setSuffix("%")
        self.setValue(value)
        self.setToolTip(self.tr("Zoom Level"))
        self.setStatusTip(self.toolTip())
        self.setAlignment(QtCore.Qt.AlignCenter)
        font = self.font()
        font.setPointSize(9)
        self.setFont(font)

    # QT Overload
    def minimumSizeHint(self):
        height = super().minimumSizeHint().height()
        font_metric = QtGui.QFontMetrics(self.font())
        width = font_metric.horizontalAdvance(str(self.maximum()))
        return QtCore.QSize(width, height)
