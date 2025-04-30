from PyQt5 import QtWidgets, QtCore

from anylabeling.views.labeling.utils.qt import new_icon_path


class CrosshairSettingsDialog(QtWidgets.QDialog):
    def __init__(
        self, show=True, width=2.0, color="#00FF00", opacity=0.5, parent=None
    ):
        super().__init__(parent)

        self._show = show
        self._width = width
        self._color = color
        self._opacity = opacity

        self.setWindowTitle(self.tr("Crosshair Settings"))
        self.setModal(True)
        self.setFixedSize(380, 280)
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint
        )

        # Apply macOS style
        self.setStyleSheet(
            f"""
                QDialog {{
                    background-color: #f5f5f7;
                    border-radius: 10px;
                }}
                QLabel {{
                    color: #1d1d1f;
                    font-size: 13px;
                }}
                QCheckBox {{
                    spacing: 8px;
                }}
                QCheckBox::indicator {{
                    width: 18px;
                    height: 18px;
                    border-radius: 3px;
                    border: 1px solid #d2d2d7;
                    background-color: white;
                }}
                QCheckBox::indicator:checked {{
                    background-color: white;
                    border: 1px solid #d2d2d7;
                    image: url({new_icon_path("checkmark", "svg")});
                }}
                QSlider {{
                    height: 28px;
                }}
                QSlider::groove:horizontal {{
                    height: 4px;
                    background: #d2d2d7;
                    border-radius: 2px;
                }}
                QSlider::handle:horizontal {{
                    background: #0071e3;
                    border: none;
                    width: 16px;
                    height: 16px;
                    margin: -6px 0;
                    border-radius: 8px;
                }}
                QSlider::sub-page:horizontal {{
                    background: #0071e3;
                    border-radius: 2px;
                }}
                QDoubleSpinBox {{
                    padding: 5px 8px;
                    background: white;
                    border: 1px solid #d2d2d7;
                    border-radius: 6px;
                    min-height: 24px;
                    selection-background-color: #0071e3;
                }}
                QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                    width: 20px;
                    border: none;
                    background: #f0f0f0;
                }}

                QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
                    background: #e0e0e0;
                }}

                QDoubleSpinBox::up-arrow {{
                    image: url({new_icon_path("caret-up", "svg")});
                    width: 12px;
                    height: 12px;
                }}

                QDoubleSpinBox::down-arrow {{
                    image: url({new_icon_path("caret-down", "svg")});
                    width: 12px;
                    height: 12px;
                }}

                QLineEdit {{
                    padding: 5px 8px;
                    background: white;
                    border: 1px solid #d2d2d7;
                    border-radius: 6px;
                    min-height: 24px;
                    selection-background-color: #0071e3;
                }}
        """
        )

        # Create layout with proper spacing
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Show Crosshair option
        show_layout = QtWidgets.QHBoxLayout()
        self.show_label = QtWidgets.QLabel(self.tr("Show Crosshair:"))
        self.show_label.setMinimumWidth(100)
        self.show_checkbox = QtWidgets.QCheckBox()
        self.show_checkbox.setChecked(self._show)
        show_layout.addWidget(self.show_label)
        show_layout.addWidget(self.show_checkbox)
        show_layout.addStretch()

        # Line width controls
        width_layout = QtWidgets.QHBoxLayout()
        width_layout.setSpacing(10)
        self.width_label = QtWidgets.QLabel(self.tr("Line width:"))
        self.width_label.setMinimumWidth(100)
        width_layout.addWidget(self.width_label)

        self.width_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.width_slider.setMinimum(10)
        self.width_slider.setMaximum(100)
        self.width_slider.setValue(int(self._width * 10))
        self.width_slider.setTickInterval(1)
        width_layout.addWidget(self.width_slider)

        self.width_spinbox = QtWidgets.QDoubleSpinBox()
        self.width_spinbox.setRange(1.0, 10.0)
        self.width_spinbox.setSingleStep(0.1)
        self.width_spinbox.setValue(self._width)
        self.width_spinbox.setFixedWidth(68)
        self.width_spinbox.setAlignment(QtCore.Qt.AlignRight)
        width_layout.addWidget(self.width_spinbox)

        # Line opacity controls
        opacity_layout = QtWidgets.QHBoxLayout()
        opacity_layout.setSpacing(10)
        self.opacity_label = QtWidgets.QLabel(self.tr("Line Opacity:"))
        self.opacity_label.setMinimumWidth(100)
        opacity_layout.addWidget(self.opacity_label)

        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(int(self._opacity * 100))
        self.opacity_slider.setTickInterval(1)
        opacity_layout.addWidget(self.opacity_slider)

        self.opacity_spinbox = QtWidgets.QDoubleSpinBox()
        self.opacity_spinbox.setRange(0.0, 1.0)
        self.opacity_spinbox.setSingleStep(0.01)
        self.opacity_spinbox.setValue(self._opacity)
        self.opacity_spinbox.setFixedWidth(68)
        self.opacity_spinbox.setAlignment(QtCore.Qt.AlignRight)
        opacity_layout.addWidget(self.opacity_spinbox)

        # Color controls
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.setSpacing(8)

        self.color_label = QtWidgets.QLabel(self.tr("Line Color:"))
        self.color_label.setMinimumWidth(100)

        self.color_lineedit = QtWidgets.QLineEdit()
        self.color_lineedit.setText(self._color)
        self.color_lineedit.setFixedSize(100, 32)
        self.color_lineedit.setStyleSheet(
            """
            QLineEdit {
                padding: 0px 8px;
                background-color: white;
                color: #1d1d1f;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
            }
        """
        )

        self.color_button = QtWidgets.QPushButton(self.tr("Choose Color"))
        self.color_button.clicked.connect(self.choose_color)
        self.color_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f5f5f7;
                color: #1d1d1f;
                border: 1px solid #d2d2d7;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #e5e5e5;
            }
            QPushButton:pressed {
                background-color: #d5d5d5;
            }
        """
        )
        self.color_button.setFixedSize(100, 32)

        color_layout.addWidget(self.color_label)
        color_layout.addStretch()
        color_layout.addWidget(self.color_lineedit)
        color_layout.addWidget(self.color_button)

        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(8)

        self.reset_button = QtWidgets.QPushButton(self.tr("Reset"))
        self.reset_button.setFixedSize(100, 32)
        self.reset_button.clicked.connect(self.reset_settings)
        self.reset_button.setStyleSheet(
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

        ok_button = QtWidgets.QPushButton(self.tr("OK"))
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

        cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
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

        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)

        # Add all layouts to the main layout
        layout.addLayout(show_layout)
        layout.addLayout(width_layout)
        layout.addLayout(opacity_layout)
        layout.addLayout(color_layout)
        layout.addStretch(1)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect signals for slider and spinbox synchronization
        self.width_slider.valueChanged.connect(self.update_width_spinbox)
        self.width_spinbox.valueChanged.connect(self.update_width_slider)
        self.opacity_slider.valueChanged.connect(self.update_opacity_spinbox)
        self.opacity_spinbox.valueChanged.connect(self.update_opacity_slider)

        self.move_to_center()

    def move_to_center(self):
        """Move dialog to center of the screen"""
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def update_width_slider(self, value):
        self.width_slider.setValue(int(value * 10))

    def update_width_spinbox(self, value):
        self.width_spinbox.setValue(value / 10.0)

    def update_opacity_slider(self, value):
        self.opacity_slider.setValue(int(value * 100))

    def update_opacity_spinbox(self, value):
        self.opacity_spinbox.setValue(value / 100.0)

    def choose_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.color_lineedit.setText(color.name())

    def reset_settings(self):
        self.show_checkbox.setChecked(self._show)
        self.width_slider.setValue(int(self._width * 100))
        self.width_spinbox.setValue(self._width)
        self.color_lineedit.setText(self._color)
        self.opacity_slider.setValue(int(self._opacity * 100))
        self.opacity_spinbox.setValue(self._opacity)

    def get_settings(self):
        return {
            "show": self.show_checkbox.isChecked(),
            "width": self.width_spinbox.value(),
            "color": self.color_lineedit.text(),
            "opacity": self.opacity_spinbox.value(),
        }
