from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from anylabeling.views.labeling.utils.qt import new_icon_path


class PolygonSidesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Set Polygon Sides"))
        self.setFixedSize(350, 180)
        self.setModal(True)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        label = QLabel(self.tr("Enter number of polygon sides:"))
        label.setStyleSheet(
            """
            QLabel {
                color: #1d1d1f;
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
        self.spinbox.setStyleSheet(
            f"""
            QSpinBox {{
                border: 1px solid #d2d2d7;
                border-radius: 8px;
                padding: 2px 30px 2px 8px;
                background-color: #ffffff;
                color: #1d1d1f;
                font-size: 14px;
                min-height: 36px;
            }}
            QSpinBox:hover {{
                background-color: #f5f5f7;
            }}
            QSpinBox:focus {{
                border: 2px solid #0071e3;
                background-color: #ffffff;
            }}
            QSpinBox::up-button {{
                subcontrol-origin: border;
                subcontrol-position: top right;
                top: 5px;
                width: 22px;
                height: 18px;
                background-color: transparent;
                border: none;
                margin: 0px;
                margin-right: 10px;
            }}
            QSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                bottom: 5px;
                width: 22px;
                height: 18px;
                background-color: transparent;
                border: none;
                margin: 0px;
                margin-right: 10px;
            }}
            QSpinBox::up-arrow {{
                width: 22px;
                height: 16px;
                image: url({new_icon_path("caret-up", "svg")});
            }}
            QSpinBox::down-arrow {{
                width: 22px;
                height: 16px;
                image: url({new_icon_path("caret-down", "svg")});
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background-color: rgba(0, 0, 0, 0.1);
                border-radius: 3px;
            }}
        """
        )
        layout.addWidget(self.spinbox)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        ok_button = QPushButton(self.tr("OK"))
        ok_button.setFixedSize(100, 32)
        ok_button.clicked.connect(self.accept)
        ok_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0071e3;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0077ED;
            }
            QPushButton:pressed {
                background-color: #0068D0;
            }
        """
        )

        cancel_button = QPushButton(self.tr("Cancel"))
        cancel_button.setFixedSize(100, 32)
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f5f5f7;
                color: #1d1d1f;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #e5e5e5;
            }
            QPushButton:pressed {
                background-color: #d5d5d5;
            }
        """
        )

        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_value(self):
        return self.spinbox.value()
