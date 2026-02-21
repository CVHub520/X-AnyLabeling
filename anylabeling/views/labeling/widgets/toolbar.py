"""Defines toolbar for anylabeling, including"""

from PyQt5 import QtCore, QtWidgets
from anylabeling.views.labeling.utils.theme import get_mode, get_theme


class ToolBar(QtWidgets.QToolBar):
    """Toolbar widget for labeling tool"""

    def __init__(self, title):
        super().__init__(title)
        layout = self.layout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)

        self._is_dark = get_mode() == "dark"
        t = get_theme()
        separator_qss = ""
        if self._is_dark:
            separator_qss = f"""
            QToolBar::separator {{
                background: {t["border"]};
                height: 1px;
                margin: 4px 6px;
            }}
            """
        self.setStyleSheet(
            f"""
            QToolBar {{
                background: {t["background"]};
                padding: 0px;
                border: 2px solid {t["border"]};
                border-radius: 5px;
            }}
            {separator_qss}
            """
        )
        if self._is_dark:
            layout.setContentsMargins(0, 4, 0, 4)

    def clear(self):
        super().clear()
        layout = self.layout()
        layout.setSpacing(0)
        if self._is_dark:
            layout.setContentsMargins(0, 4, 0, 4)
        else:
            layout.setContentsMargins(0, 0, 0, 0)

    def add_action(self, action):
        """Add an action (button) to the toolbar"""
        if isinstance(action, QtWidgets.QWidgetAction):
            return super().addAction(action)
        btn = QtWidgets.QToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        self.addWidget(btn)

        # Center alignment
        for i in range(self.layout().count()):
            if isinstance(
                self.layout().itemAt(i).widget(), QtWidgets.QToolButton
            ):
                self.layout().itemAt(i).setAlignment(QtCore.Qt.AlignCenter)

        return True
