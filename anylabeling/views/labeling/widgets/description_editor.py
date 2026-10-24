from PyQt6 import QtCore, QtGui, QtSvg, QtWidgets

from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_ok_btn_style,
    get_plain_text_edit_style,
    get_scrollbar_style,
)
from anylabeling.views.labeling.utils.theme import get_theme


class DescriptionEditor(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.expand_button = QtWidgets.QToolButton(self)
        self.expand_button.setFixedSize(22, 22)
        self.expand_button.setIconSize(QtCore.QSize(14, 14))
        self.expand_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.expand_button.setToolTip(self.tr("Expand description"))
        self.expand_button.setAccessibleName(self.tr("Expand description"))
        self.expand_button.hide()
        self.viewport().installEventFilter(self)
        self.expand_button.clicked.connect(self.edit_description)
        self.setStyleSheet(get_plain_text_edit_style())

    def setStyleSheet(self, style):
        super().setStyleSheet(
            style
            + get_scrollbar_style()
            + "QPlainTextEdit { padding: 4px; }"
            + "QPlainTextEdit:focus { border: none; }"
        )
        theme = get_theme()
        self.expand_button.setStyleSheet(f"""
            QToolButton {{
                background: {theme['background']};
                border: none;
                border-radius: 6px;
            }}
            QToolButton:hover {{ background: {theme['surface_hover']}; }}
            QToolButton:pressed {{ background: {theme['surface_pressed']}; }}
        """)
        source = QtCore.QFile(new_icon_path("maximize-2", "svg"))
        if not source.open(QtCore.QIODevice.OpenModeFlag.ReadOnly):
            return
        svg = bytes(source.readAll())
        source.close()
        svg = svg.replace(b"currentColor", theme["text_secondary"].encode())
        renderer = QtSvg.QSvgRenderer(svg)
        ratio = self.devicePixelRatioF()
        pixmap = QtGui.QPixmap(round(14 * ratio), round(14 * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        self.expand_button.setIcon(QtGui.QIcon(pixmap))

    def enterEvent(self, event):
        super().enterEvent(event)
        self._position_expand_button()
        self.expand_button.show()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.expand_button.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_expand_button()

    def eventFilter(self, watched, event):
        if (
            watched is self.viewport()
            and event.type() == QtCore.QEvent.Type.Resize
        ):
            self._position_expand_button()
        return super().eventFilter(watched, event)

    def _position_expand_button(self):
        bounds = self.rect().adjusted(3, 3, -3, -3)
        if self.verticalScrollBar().isVisible():
            bounds.setRight(self.viewport().geometry().right() - 3)
        if self.horizontalScrollBar().isVisible():
            bounds.setBottom(self.viewport().geometry().bottom() - 3)
        self.expand_button.move(
            bounds.right() - self.expand_button.width() + 1,
            bounds.bottom() - self.expand_button.height() + 1,
        )
        self.expand_button.raise_()

    def edit_description(self):
        dialog = QtWidgets.QDialog(self.window())
        dialog.setWindowTitle(self.tr("Edit description"))
        dialog.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        theme = get_theme()
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        panel = QtWidgets.QFrame()
        panel.setObjectName("descriptionPanel")
        panel.setStyleSheet(f"""
            QFrame#descriptionPanel {{
                background: {theme['background']};
                border: 1px solid {theme['border_light']};
                border-radius: 16px;
            }}
            QLabel {{ color: {theme['text']}; font-size: 16px; }}
        """)
        layout.addWidget(panel)
        content = QtWidgets.QVBoxLayout(panel)
        content.setContentsMargins(24, 24, 24, 24)
        content.setSpacing(18)
        content.addWidget(QtWidgets.QLabel(dialog.windowTitle()))
        editor = QtWidgets.QPlainTextEdit()
        editor.setPlainText(self.toPlainText())
        editor.setReadOnly(self.isReadOnly())
        editor.setStyleSheet(
            get_plain_text_edit_style() + get_scrollbar_style() + f"""
            QPlainTextEdit {{
                border: 1px solid {theme['border_light']};
                border-radius: 10px;
                padding: 12px;
                font-size: 14px;
            }}
            QPlainTextEdit:focus {{ border: 1px solid {theme['primary']}; }}
        """
        )
        content.addWidget(editor, 1)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(10)
        buttons.addStretch()
        cancel = QtWidgets.QPushButton(self.tr("Cancel"))
        cancel.setStyleSheet(get_cancel_btn_style())
        cancel.clicked.connect(dialog.reject)
        confirm = QtWidgets.QPushButton(self.tr("Confirm"))
        confirm.setStyleSheet(get_ok_btn_style())
        confirm.clicked.connect(dialog.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        content.addLayout(buttons)
        available = self.screen().availableGeometry()
        dialog.resize(
            min(680, available.width() - 48), min(460, available.height() - 48)
        )
        dialog.move(available.center() - dialog.rect().center())
        editor.setFocus()
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            text = editor.toPlainText()
            if text != self.toPlainText():
                self.setPlainText(text)
        dialog.deleteLater()
