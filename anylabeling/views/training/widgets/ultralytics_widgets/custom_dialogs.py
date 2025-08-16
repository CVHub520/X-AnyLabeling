from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout
from .custom_widgets import CustomComboBox, PrimaryButton, SecondaryButton


class ExportFormatDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export Settings"))
        self.setFixedSize(400, 220)
        self.setModal(True)
        self.selected_format = "onnx"

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(24, 24, 24, 24)

        desc_label = QLabel(
            self.tr("Select the format for exporting your trained model:")
        )
        desc_label.setStyleSheet("color: #718096; margin-bottom: 8px;")
        layout.addWidget(desc_label)

        self.format_combo = CustomComboBox()
        formats = [
            ("ONNX", "onnx"),
            ("TorchScript", "torchscript"),
            ("OpenVINO", "openvino"),
            ("TensorRT", "engine"),
            ("CoreML", "coreml"),
            ("TensorFlow SavedModel", "saved_model"),
            ("TensorFlow Lite", "tflite"),
            ("TensorFlow Edge TPU", "edgetpu"),
            ("TensorFlow.js", "tfjs"),
            ("PaddlePaddle", "paddle"),
            ("MNN", "mnn"),
            ("NCNN", "ncnn"),
            ("IMX500", "imx"),
            ("RKNN", "rknn"),
        ]

        for display_name, format_code in formats:
            self.format_combo.addItem(display_name, format_code)

        self.format_combo.setCurrentIndex(0)
        layout.addWidget(self.format_combo)

        info_label = QLabel(
            self.tr(
                "Note: Some formats may require additional dependencies to be installed."
            )
        )
        info_label.setStyleSheet(
            """
            color: #e69500;
            font-size: 12px;
            margin-top: 8px;
            padding: 4px;
            min-height: 20px;
        """
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = SecondaryButton(self.tr("Cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.ok_btn = PrimaryButton(self.tr("Export"))
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_selected_format(self):
        return self.format_combo.currentData()
