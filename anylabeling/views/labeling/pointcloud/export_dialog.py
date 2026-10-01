from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from .export import EXPORT_FORMATS, ExportCancelled, export_dataset
from .icons import get_icon


class ExportWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int)

    def __init__(self, options, parent):
        super().__init__(parent)
        self.options = options
        self.export_result = None
        self.error = None

    def run(self):
        try:
            self.export_result = export_dataset(
                **self.options,
                progress=self.progress.emit,
                cancelled=self.isInterruptionRequested,
            )
        except ExportCancelled:
            pass
        except Exception as error:
            self.error = str(error)


class PointCloudExportDialog(QtWidgets.QDialog):
    def __init__(self, frames, classes, parent):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export 3D objects"))
        self.setMinimumWidth(700)
        self.frames = tuple(frames)
        self.classes = tuple(classes)
        self.calibration = parent.camera_panel.calibration
        self.worker = None
        self.export_result = None
        self.output_path = None
        self._cancel_requested = False
        directory = frames[0].path.parent.parent
        self.path_input = QtWidgets.QLineEdit(
            str(directory / "dataset_export.zip")
        )
        self.path_input.setPlaceholderText(self.tr("Export ZIP path"))
        self.path_input.setAccessibleName(self.tr("Export ZIP path"))
        self.browse_button = QtWidgets.QPushButton(self.tr("Browse…"))
        self.browse_button.setIcon(parent._icon("folder-chatbot"))
        self.browse_button.setAutoDefault(False)
        self.browse_button.clicked.connect(self._browse)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(16)
        layout.addWidget(self.path_input, 0, 0, 1, 2)
        layout.addWidget(self.browse_button, 0, 2)
        self.formats = QtWidgets.QButtonGroup(self)
        self.format_buttons = {}
        for column, (key, title) in enumerate(EXPORT_FORMATS.items()):
            button = QtWidgets.QRadioButton(title)
            button.setToolTip(
                self.tr(
                    "Export 3D cuboids from all frames. Point segmentation is not included."
                )
            )
            self.formats.addButton(button)
            self.format_buttons[key] = button
            layout.addWidget(button, 1, column)
            layout.setColumnStretch(column, 1)
        self.format_buttons["datumaro"].setChecked(True)
        self.save_images = QtWidgets.QCheckBox(self.tr("Save images"))
        self.save_images.setToolTip(
            self.tr("Include point clouds and associated camera images.")
        )
        layout.addWidget(self.save_images, 2, 0)
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button = QtWidgets.QPushButton(self.tr("OK"))
        self.ok_button.setObjectName("pointcloudConfirmButton")
        self.ok_button.setIcon(get_icon("confirm", "#ffffff", "#ffffff"))
        self.ok_button.setIconSize(QtCore.QSize(16, 16))
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.cancel_button, 2, 1)
        layout.addWidget(self.ok_button, 2, 2)
        for widget in (
            self.path_input,
            self.browse_button,
            self.cancel_button,
            self.ok_button,
        ):
            widget.setMinimumHeight(32)
        self.path_input.textChanged.connect(self._update_ok)
        self._update_ok()

    def _update_ok(self):
        self.ok_button.setEnabled(
            bool(self.path_input.text().strip()) and self.worker is None
        )

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Export 3D objects"),
            self.path_input.text(),
            self.tr("ZIP archives (*.zip)"),
            options=(
                QtWidgets.QFileDialog.Option.DontUseNativeDialog
                | QtWidgets.QFileDialog.Option.DontConfirmOverwrite
            ),
        )
        if path:
            self.path_input.setText(path)

    def accept(self):
        if self.worker is not None or not self.path_input.text().strip():
            return
        path = Path(self.path_input.text().strip()).expanduser()
        if not path.suffix:
            path = path.with_suffix(".zip")
        if (
            path.suffix.lower() != ".zip"
            or not path.parent.is_dir()
            or path.is_dir()
        ):
            QtWidgets.QMessageBox.warning(
                self,
                self.windowTitle(),
                self.tr("Choose a ZIP file in an existing directory."),
            )
            return
        if (
            path.exists()
            and QtWidgets.QMessageBox.question(
                self,
                self.windowTitle(),
                self.tr("Replace the existing file?\n{path}").format(
                    path=path
                ),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            != QtWidgets.QMessageBox.StandardButton.Yes
        ):
            return
        self.output_path = path
        self._cancel_requested = False
        format_name = next(
            key
            for key, button in self.format_buttons.items()
            if button.isChecked()
        )
        self.worker = ExportWorker(
            dict(
                path=path,
                format_name=format_name,
                frames=self.frames,
                classes=self.classes,
                save_images=self.save_images.isChecked(),
                calibration=self.calibration,
            ),
            self,
        )
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._finished)
        self._set_busy(True)
        self._progress(0, len(self.frames))
        self.worker.start()

    def _set_busy(self, busy):
        for widget in (
            self.path_input,
            self.browse_button,
            self.save_images,
            *self.format_buttons.values(),
        ):
            widget.setEnabled(not busy)
        self._update_ok()

    def _progress(self, completed, total):
        if not self._cancel_requested:
            self.setWindowTitle(
                self.tr("Exporting 3D objects ({completed}/{total})…").format(
                    completed=completed, total=total
                )
            )

    def _finished(self):
        worker = self.worker
        self.worker = None
        self.export_result = worker.export_result
        error = worker.error
        worker.deleteLater()
        self.setWindowTitle(self.tr("Export 3D objects"))
        self._set_busy(False)
        self.cancel_button.setEnabled(True)
        if self.export_result is not None:
            super().accept()
        elif self._cancel_requested:
            super().reject()
        elif error is not None:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), error)

    def reject(self):
        if self.worker is not None:
            self._cancel_requested = True
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.setWindowTitle(self.tr("Cancelling export…"))
            return
        super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self.reject()
            event.ignore()
            return
        super().closeEvent(event)
