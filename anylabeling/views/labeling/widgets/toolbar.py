"""Defines toolbar for anylabeling, including """

import darkdetect
from PyQt5 import QtCore, QtWidgets


class ToolBar(QtWidgets.QToolBar):
    """Toolbar widget for labeling tool"""

    def __init__(self, title):
        super().__init__(title)
        layout = self.layout()
        margin = (0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(*margin)
        self.setContentsMargins(*margin)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)

        if darkdetect.isDark():
            self.setStyleSheet(
                """
                QToolBar {
                    background: #555;
                    padding: 0px;
                    border: 0px;
                    border-radius: 5px;
                    border: 2px solid #222;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QToolBar {
                    background: #fff;
                    padding: 0px;
                    border: 0px;
                    border-radius: 5px;
                    border: 2px solid #aaa;
                }
                """
            )

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
