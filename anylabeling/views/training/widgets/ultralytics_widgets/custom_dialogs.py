from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout
from .custom_widgets import CustomComboBox, PrimaryButton, SecondaryButton
from anylabeling.services.auto_training.ultralytics.style import (
    get_ultralytics_dialog_style,
)
from anylabeling.views.labeling.utils.theme import get_theme


class ExportFormatDialog(QDialog):
    export_requested = pyqtSignal(str)
    stop_requested = pyqtSignal()
    apply_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export Settings"))
        self.setFixedSize(400, 220)
        self.setModal(True)
        self.selected_format = "onnx"
        self.action_state = "export"
        self.setStyleSheet(get_ultralytics_dialog_style())

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(24, 24, 24, 24)

        desc_label = QLabel(
            self.tr("Select the format for exporting your trained model:")
        )
        t = get_theme()
        desc_label.setStyleSheet(
            f"color: {t['text_secondary']}; margin-bottom: 8px;"
        )
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
        info_label.setStyleSheet(f"""
            color: {t['warning']};
            font-size: 12px;
            margin-top: 8px;
            padding: 4px;
            min-height: 20px;
        """)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = SecondaryButton(self.tr("Cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.ok_btn = PrimaryButton(self.tr("Export"))
        self.ok_btn.clicked.connect(self.on_primary_action)
        button_layout.addWidget(self.ok_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_selected_format(self):
        return self.format_combo.currentData()

    def on_primary_action(self):
        if self.action_state == "export":
            self.export_requested.emit(self.get_selected_format())
        elif self.action_state == "stop":
            self.stop_requested.emit()
        elif self.action_state == "apply":
            self.apply_requested.emit()

    def set_exporting(self, can_stop=True):
        self.action_state = "stop" if can_stop else "exporting"
        self.ok_btn.setText(
            self.tr("Stop") if can_stop else self.tr("Exporting...")
        )
        self.ok_btn.setToolTip(
            self.tr("Stop the current export") if can_stop else ""
        )
        self.ok_btn.setEnabled(can_stop)
        self.format_combo.setEnabled(False)
        self.cancel_btn.setEnabled(False)

    def set_apply_ready(self):
        self.action_state = "apply"
        self.ok_btn.setText(self.tr("Apply"))
        self.ok_btn.setToolTip(
            self.tr(
                "Generate the model configuration and load it in X-AnyLabeling"
            )
        )
        self.ok_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)

    def reset_export(self):
        self.action_state = "export"
        self.ok_btn.setText(self.tr("Export"))
        self.ok_btn.setToolTip("")
        self.ok_btn.setEnabled(True)
        self.format_combo.setEnabled(True)
        self.cancel_btn.setEnabled(True)

    def reject(self):
        if self.action_state in {"stop", "exporting"}:
            return
        super().reject()
