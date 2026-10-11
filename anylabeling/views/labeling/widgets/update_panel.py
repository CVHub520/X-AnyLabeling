from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
)

from anylabeling.views.labeling.utils.theme import get_theme


class UpdatePanel(QDialog):
    update_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    def __init__(self, parent):
        super().__init__(
            parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle(self.tr("Software Update"))
        self.setFixedWidth(396)
        self.mode = "available"
        theme = get_theme()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        card = QFrame()
        card.setObjectName("updateCard")
        layout.addWidget(card)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 45))
        card.setGraphicsEffect(shadow)
        content = QVBoxLayout(card)
        content.setContentsMargins(22, 20, 22, 20)
        content.setSpacing(12)
        header = QHBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("updateTitle")
        header.addWidget(self.title, 1)
        self.close_button = QToolButton()
        self.close_button.setText("×")
        self.close_button.setFixedSize(24, 24)
        self.close_button.setAccessibleName(self.tr("Close update panel"))
        self.close_button.clicked.connect(self.hide)
        header.addWidget(self.close_button)
        content.addLayout(header)
        self.description = QLabel()
        self.description.setTextFormat(Qt.TextFormat.PlainText)
        self.description.setWordWrap(True)
        self.description.setObjectName("updateDescription")
        content.addWidget(self.description)
        progress_layout = QHBoxLayout()
        progress_layout.setSpacing(14)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        progress_layout.addWidget(self.progress, 1)
        self.percentage = QLabel("0%")
        self.percentage.setObjectName("updateDescription")
        self.percentage.setFixedWidth(34)
        self.percentage.setAlignment(Qt.AlignmentFlag.AlignRight)
        progress_layout.addWidget(self.percentage)
        content.addLayout(progress_layout)
        footer = QHBoxLayout()
        footer.addStretch()
        self.secondary = QPushButton(self.tr("Later"))
        self.secondary.clicked.connect(self._secondary_clicked)
        footer.addWidget(self.secondary)
        self.primary = QPushButton(self.tr("Update and Restart"))
        self.primary.setObjectName("updatePrimary")
        self.primary.clicked.connect(self.update_requested)
        footer.addWidget(self.primary)
        content.addLayout(footer)
        self.setStyleSheet(f"""
            QFrame#updateCard {{ background: {theme['background_secondary']};
                border: 1px solid {theme['border']}; border-radius: 14px; }}
            QLabel {{ background: transparent; border: none; color: {theme['text']}; }}
            QLabel#updateTitle {{ font-size: 17px; font-weight: 600; }}
            QLabel#updateDescription {{ color: {theme['text_secondary']}; font-size: 12px; }}
            QToolButton {{ border: none; border-radius: 6px; background: transparent;
                color: {theme['text_secondary']}; font-size: 18px; }}
            QToolButton:hover {{ background: {theme['surface_hover']}; }}
            QPushButton {{ padding: 8px 12px; border: 1px solid {theme['border']};
                border-radius: 7px; background: transparent; color: {theme['text']}; }}
            QPushButton:hover {{ background: {theme['surface_hover']}; }}
            QPushButton#updatePrimary {{ background: {theme['primary']}; color: white;
                border-color: {theme['primary']}; }}
            QPushButton#updatePrimary:hover {{ background: {theme['primary_hover']}; }}
            QProgressBar {{ border: none; border-radius: 3px; background: {theme['surface_pressed']}; }}
            QProgressBar::chunk {{ border-radius: 3px; background: {theme['primary']}; }}
        """)
        self.fade = QPropertyAnimation(self, b"windowOpacity", self)
        self.fade.setDuration(140)
        self.fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.progress_animation = QPropertyAnimation(
            self.progress, b"value", self
        )
        self.progress_animation.setDuration(160)
        self.progress_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _secondary_clicked(self):
        if self.mode == "downloading":
            self.cancel_requested.emit()
        else:
            self.hide()

    def set_state(self, mode, version="", message=""):
        self.mode = mode
        titles = {
            "available": self.tr("Update available"),
            "downloading": self.tr("Downloading update"),
            "installing": self.tr("Preparing restart"),
            "error": self.tr("Update failed"),
            "current": self.tr("You're up to date"),
        }
        descriptions = {
            "available": self.tr(
                "Version {version} is ready. The app will restart after updating."
            ).format(version=version),
            "downloading": self.tr(
                "You can keep working while the update downloads."
            ),
            "installing": self.tr(
                "The app will restart when installation finishes."
            ),
            "error": message,
            "current": self.tr("No compatible update is available."),
        }
        self.title.setText(titles[mode])
        self.description.setText(descriptions[mode])
        running = mode in ("downloading", "installing")
        self.progress.setVisible(running)
        self.percentage.setVisible(mode == "downloading")
        self.primary.setVisible(mode in ("available", "error"))
        self.primary.setText(
            self.tr("Retry")
            if mode == "error"
            else self.tr("Update and Restart")
        )
        self.secondary.setVisible(mode != "installing")
        self.secondary.setText(
            self.tr("Cancel download")
            if mode == "downloading"
            else self.tr("Later")
        )
        self.close_button.setVisible(mode != "installing")
        self.progress_animation.stop()
        self.progress.setRange(0, 0 if mode == "installing" else 100)
        if mode == "downloading":
            self.progress.setValue(0)
            self.percentage.setText("0%")
        self.adjustSize()

    def set_progress(self, value):
        if self.mode != "downloading":
            return
        self.percentage.setText(f"{value}%")
        self.progress_animation.stop()
        self.progress_animation.setStartValue(self.progress.value())
        self.progress_animation.setEndValue(value)
        self.progress_animation.start()

    def reveal(self, anchor=None):
        self.adjustSize()
        if anchor is not None and anchor.isVisible():
            position = anchor.mapToGlobal(
                QPoint(anchor.width(), anchor.height() + 6)
            )
            screen = anchor.screen().availableGeometry()
            x = max(
                screen.left(),
                min(
                    position.x() - self.width(), screen.right() - self.width()
                ),
            )
            y = max(
                screen.top(),
                min(position.y(), screen.bottom() - self.height()),
            )
            self.move(x, y)
        else:
            center = self.parentWidget().mapToGlobal(
                self.parentWidget().rect().center()
            )
            self.move(center - self.rect().center())
        if not self.isVisible():
            self.setWindowOpacity(0)
            self.show()
            self.fade.setStartValue(0)
            self.fade.setEndValue(1)
            self.fade.start()
        self.raise_()
        self.activateWindow()
