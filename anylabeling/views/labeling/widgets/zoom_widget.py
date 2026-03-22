from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.utils.theme import get_theme


class ZoomWidget(QtWidgets.QSpinBox):
    def __init__(self, value=100):
        super().__init__()
        self.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.setRange(1, 1000)
        self.setSuffix("%")
        self.setValue(value)
        self.setToolTip(self.tr("Zoom Level"))
        self.setStatusTip(self.toolTip())
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = self.font()
        font.setPointSize(9)
        self.setFont(font)

        t = get_theme()
        self.setStyleSheet(f"""
            QSpinBox {{
                background-color: transparent;
                color: {t["text"]};
                border: none;
                padding: 0;
                min-height: 0;
            }}
            """)
