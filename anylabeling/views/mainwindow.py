"""This module defines the main application window"""

import os

from PyQt6 import QtCore
from PyQt6.QtWidgets import QMainWindow, QStatusBar, QVBoxLayout, QWidget

from ..app_info import __appdescription__, __appname__, __version__
from .labeling.label_wrapper import LabelingWrapper


def _is_wsl_environment() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", encoding="utf-8") as file_obj:
            return "microsoft" in file_obj.read().lower()
    except OSError:
        return False


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(
        self,
        app,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        super().__init__()
        self.app = app
        self.config = config

        self.setContentsMargins(0, 0, 0, 0)
        self.setWindowTitle(__appname__)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.labeling_widget = LabelingWrapper(
            self,
            config=config,
            filename=filename,
            output=output,
            output_file=output_file,
            output_dir=output_dir,
        )
        main_layout.addWidget(self.labeling_widget)
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        self.settings = self.labeling_widget.view.settings
        self._is_wsl_environment = _is_wsl_environment()
        if self._is_wsl_environment:
            self.settings.remove("window/size")
            self.settings.remove("window/position")
            self.settings.remove("window/state")
            self.settings.sync()

        status_bar = QStatusBar()
        status_bar.showMessage(
            f"{__appname__} v{__version__} - {__appdescription__}"
        )
        self.setStatusBar(status_bar)
        self._restore_window_geometry()

    def closeEvent(self, event):
        self.labeling_widget.closeEvent(event)
        if not event.isAccepted():
            return
        if self._is_wsl_environment:
            self.settings.remove("window/size")
            self.settings.remove("window/position")
            self.settings.remove("window/state")
        else:
            self.settings.setValue("window/size", self.size())
            self.settings.setValue("window/position", self.pos())
            self.settings.setValue("window/state", self.saveState())
        super().closeEvent(event)

    def _restore_window_geometry(self):
        if self._is_wsl_environment:
            return
        if self.settings.contains("window/size"):
            size = self.settings.value("window/size", type=QtCore.QSize)
            if isinstance(size, QtCore.QSize) and size.isValid():
                self.resize(size)
        if self.settings.contains("window/position"):
            position = self.settings.value(
                "window/position", type=QtCore.QPoint
            )
            if isinstance(position, QtCore.QPoint):
                self.move(position)
        if self.settings.contains("window/state"):
            state = self.settings.value("window/state", type=QtCore.QByteArray)
            if state:
                self.restoreState(state)
