import os

from PyQt6.QtCore import (
    Qt,
    pyqtSignal,
)
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QTextEdit,
    QSpinBox,
)

from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_ok_btn_style,
)
from anylabeling.views.labeling.utils.theme import get_theme


class BatchProcessDialog(QDialog):
    """Batch processing dialog class"""

    promptReady = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle(self.tr("Batch Process All Images"))
        self.setMinimumWidth(450)

        self.cpu_count = os.cpu_count() or 1
        self.max_concurrency = max(1, int(self.cpu_count * 0.95))
        self.default_concurrency = max(1, int(self.cpu_count * 0.8))

        self.setup_ui()

    def setup_ui(self):
        """Set up the UI interface"""
        t = get_theme()
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {t["background"]};
                border-radius: 8px;
            }}
        """)

        # Main layout
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(24, 24, 24, 24)
        dialog_layout.setSpacing(20)

        # Instruction label
        instruction_label = QLabel(
            self.tr("Enter the prompt to apply to all images:")
        )
        instruction_label.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                color: {t["text"]};
                font-weight: 500;
            }}
        """)
        dialog_layout.addWidget(instruction_label)

        # Input box design
        self.batch_message_input = QTextEdit()
        self.batch_message_input.setPlaceholderText(
            self.tr(
                "Type your prompt here and use `@image` to reference the image."
            )
        )
        self.batch_message_input.setStyleSheet(f"""
            QTextEdit {{
                border: 1px solid {t["border"]};
                border-radius: 8px;
                background-color: {t["background_secondary"]};
                color: {t["text"]};
                font-size: 14px;
                line-height: 1.5;
                padding: 12px;
            }}
            QTextEdit:focus {{
                border: 1px solid {t["primary"]};
            }}
            QScrollBar:vertical {{
                width: 8px;
                background: transparent;
            }}
            QScrollBar::handle:vertical {{
                background: {t["scrollbar"]};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        self.batch_message_input.setAcceptRichText(False)
        self.batch_message_input.setMinimumHeight(160)
        self.batch_message_input.setMaximumHeight(200)
        self.batch_message_input.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        dialog_layout.addWidget(self.batch_message_input)

        # Concurrency setting
        settings_container = QHBoxLayout()
        settings_container.setContentsMargins(0, 0, 0, 0)
        settings_container.setSpacing(8)
        settings_container.addStretch()

        concurrency_label = QLabel(self.tr("Concurrency:"))
        concurrency_label.setStyleSheet(f"""
            QLabel {{
                font-size: 12px;
                color: {t["text_secondary"]};
                font-weight: 400;
            }}
        """)
        settings_container.addWidget(concurrency_label)

        self.concurrency_spinbox = QSpinBox()
        self.concurrency_spinbox.setMinimum(1)
        self.concurrency_spinbox.setMaximum(self.max_concurrency)
        self.concurrency_spinbox.setValue(self.default_concurrency)
        tooltip_text = self.tr("Max: {}").format(self.max_concurrency)
        self.concurrency_spinbox.setToolTip(tooltip_text)
        self.concurrency_spinbox.setSuffix(f" / {self.max_concurrency}")
        self.concurrency_spinbox.setStyleSheet(f"""
            QSpinBox {{
                border: 1px solid {t["border"]};
                border-radius: 4px;
                background-color: {t["background_secondary"]};
                color: {t["text"]};
                font-size: 12px;
                padding: 4px 8px;
                min-width: 80px;
                max-width: 80px;
            }}
            QSpinBox:focus {{
                border: 1px solid {t["primary"]};
                background-color: {t["background_secondary"]};
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                width: 16px;
                border: none;
                background: transparent;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background-color: {t["surface_hover"]};
            }}
        """)
        settings_container.addWidget(self.concurrency_spinbox)

        dialog_layout.addLayout(settings_container)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 8, 0, 0)
        button_layout.setSpacing(12)
        button_layout.addStretch()

        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.setStyleSheet(get_cancel_btn_style())
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)

        confirm_btn = QPushButton(self.tr("Confirm"))
        confirm_btn.setStyleSheet(get_ok_btn_style())
        confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        confirm_btn.clicked.connect(self.accept)

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(confirm_btn)
        dialog_layout.addLayout(button_layout)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.FramelessWindowHint
        )

    def center_on_parent(self):
        """Center the dialog on the parent window"""
        if self.parent:
            center_point = self.parent.mapToGlobal(self.parent.rect().center())
            dialog_rect = self.rect()
            self.move(
                center_point.x() - dialog_rect.width() // 2,
                center_point.y() - dialog_rect.height() // 2,
            )

    def get_prompt(self):
        """Get the user input prompt"""
        return self.batch_message_input.toPlainText().strip()

    def get_concurrency(self):
        """Get the concurrency setting"""
        return self.concurrency_spinbox.value()

    def exec(self):
        """Override exec method to adjust position before showing the dialog"""
        self.adjustSize()
        self.center_on_parent()
        result = super().exec()

        if result == QDialog.DialogCode.Accepted:
            prompt = self.get_prompt()
            concurrency = self.get_concurrency()
            if prompt:
                self.promptReady.emit(prompt)
                return (prompt, concurrency)
        return None
