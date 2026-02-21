from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import (
    get_dialog_style,
    get_ok_btn_style,
    get_cancel_btn_style,
    get_spinbox_style,
)


class PolygonSidesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Set Polygon Sides"))
        self.setFixedSize(350, 180)
        self.setModal(True)

        self.init_ui()

    def init_ui(self):
        self.setStyleSheet(get_dialog_style())

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        label = QLabel(self.tr("Enter number of polygon sides:"))
        label.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
                font-weight: 500;
            }
        """
        )
        layout.addWidget(label)

        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(3)
        self.spinbox.setMaximum(100)
        self.spinbox.setValue(32)
        self.spinbox.setFixedHeight(40)
        self.spinbox.setStyleSheet(get_spinbox_style())
        layout.addWidget(self.spinbox)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        ok_button = QPushButton(self.tr("OK"))
        ok_button.clicked.connect(self.accept)
        ok_button.setStyleSheet(get_ok_btn_style())

        cancel_button = QPushButton(self.tr("Cancel"))
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet(get_cancel_btn_style())

        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_value(self):
        return self.spinbox.value()
