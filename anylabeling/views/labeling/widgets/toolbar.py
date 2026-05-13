"""Defines toolbar for anylabeling, including"""

from PyQt6 import QtCore, QtGui, QtWidgets
from anylabeling.views.labeling.utils.theme import get_mode, get_theme


class ToolBar(QtWidgets.QFrame):
    """Toolbar widget for labeling tool"""

    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        self._orientation = QtCore.Qt.Orientation.Vertical
        self._tool_button_style = QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        self._icon_size = QtCore.QSize(24, 24)
        self._owned_widgets = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(2, 2, 2, 2)
        self._content_widget = QtWidgets.QWidget(self)
        self._content_layout = QtWidgets.QVBoxLayout(self._content_widget)
        self._content_layout.setSpacing(0)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            self._content_widget, 0, QtCore.Qt.AlignmentFlag.AlignTop
        )
        layout.addStretch(1)
        self.setContentsMargins(0, 0, 0, 0)
        self.setWindowFlags(
            self.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint
        )

        self._is_dark = get_mode() == "dark"
        t = get_theme()
        separator_qss = ""
        if self._is_dark:
            separator_qss = f"""
            QFrame#ToolBarSeparator {{
                background: {t["border"]};
            }}
            """
        self.setStyleSheet(f"""
            ToolBar {{
                background: {t["background"]};
                padding: 0px;
                border: 2px solid {t["border"]};
                border-radius: 5px;
            }}
            ToolBar QToolButton {{
                min-width: 28px;
                min-height: 28px;
                max-width: 28px;
                max-height: 28px;
                border: none;
                background: transparent;
                padding: 0px;
                margin: 0px;
            }}
            {separator_qss}
            """)

    def sizeHint(self):
        hint = self._content_widget.sizeHint()
        margins = self.layout().contentsMargins()
        frame = self.frameWidth() * 2
        return QtCore.QSize(
            hint.width() + margins.left() + margins.right() + frame,
            hint.height() + margins.top() + margins.bottom() + frame,
        )

    def minimumSizeHint(self):
        return self.sizeHint()

    def setOrientation(self, orientation):
        self._orientation = orientation

    def setToolButtonStyle(self, style):
        self._tool_button_style = style
        for button in self.findChildren(QtWidgets.QToolButton):
            button.setToolButtonStyle(style)

    def toolButtonStyle(self):
        return self._tool_button_style

    def setIconSize(self, size):
        self._icon_size = size
        for button in self.findChildren(QtWidgets.QToolButton):
            button.setIconSize(size)

    def clear(self):
        for action in self.actions():
            self.removeAction(action)
        self.setMinimumHeight(0)
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            if widget in self._owned_widgets:
                widget.deleteLater()
            else:
                widget.setParent(None)
        self._owned_widgets = []

    def addAction(self, action):
        if isinstance(action, QtWidgets.QWidgetAction):
            super().addAction(action)
            widget = action.defaultWidget()
            if widget is not None:
                self._content_layout.addWidget(
                    widget, 0, QtCore.Qt.AlignmentFlag.AlignCenter
                )
                widget.show()
            return action

        super().addAction(action)
        btn = QtWidgets.QToolButton(self)
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        btn.setIconSize(self._icon_size)
        btn.setFixedSize(28, 28)
        self._owned_widgets.append(btn)
        self._content_layout.addWidget(
            btn, 0, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        return action

    def addSeparator(self):
        action = QtGui.QAction(self)
        action.setSeparator(True)
        super().addAction(action)
        separator = QtWidgets.QFrame(self)
        separator.setObjectName("ToolBarSeparator")
        if self._orientation == QtCore.Qt.Orientation.Vertical:
            separator.setFixedSize(24, 1)
            separator.setContentsMargins(6, 4, 6, 4)
        else:
            separator.setFixedSize(1, 24)
        self._owned_widgets.append(separator)
        self._content_layout.addWidget(
            separator, 0, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        return action

    def add_action(self, action):
        """Add an action (button) to the toolbar"""
        return self.addAction(action)
