from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from .export import ExportCancelled
from .export_dialog import PointCloudExportDialog
from .import_dataset import apply_import, prepare_import


class ImportWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int)

    def __init__(self, options, parent, plan=None):
        super().__init__(parent)
        self.options = options
        self.plan = plan
        self.import_result = None
        self.error = None

    def run(self):
        try:
            if self.plan is None:
                self.import_result = prepare_import(
                    **self.options, cancelled=self.isInterruptionRequested
                )
            else:
                apply_import(
                    self.plan,
                    cancelled=self.isInterruptionRequested,
                    progress=self.progress.emit,
                )
                self.import_result = self.plan
        except ExportCancelled:
            pass
        except Exception as error:
            self.error = str(error)


class PointCloudImportDialog(PointCloudExportDialog):
    def __init__(self, targets, classes, parent):
        super().__init__(targets, classes, parent)
        self.setWindowTitle(self.tr("Import 3D objects"))
        self.path_input.clear()
        self.path_input.setPlaceholderText(self.tr("Annotation ZIP path"))
        self.path_input.setAccessibleName(self.tr("Annotation ZIP path"))
        self.save_images.hide()
        self.scope_label = QtWidgets.QLabel(self.tr("Replace matched frames"))
        self.layout().addWidget(self.scope_label, 2, 0)
        for button in self.format_buttons.values():
            button.setToolTip(
                self.tr(
                    "Import cuboids into the current point cloud sequence."
                )
            )
        self.import_plan = None
        self.config_path = parent.config_path or parent._default_class_path()
        if parent.task_type is not None:
            self.config_path = None
        self.segmentation_classes = tuple(
            parent.class_definitions["segmentation"]
        )

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Import 3D objects"),
            self.path_input.text()
            or self.parent().settings.value(
                "import_directory", str(self.frames[0].path.parent)
            ),
            self.tr("ZIP archives (*.zip)"),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self.path_input.setText(path)

    def accept(self):
        if self.worker is not None:
            return
        path = Path(self.path_input.text().strip()).expanduser()
        if not path.is_file() or path.suffix.lower() != ".zip":
            QtWidgets.QMessageBox.warning(
                self,
                self.windowTitle(),
                self.tr("Choose an existing annotation ZIP file."),
            )
            return
        self.output_path = path
        self.import_plan = None
        self.options = dict(
            path=path,
            format_name=next(
                key
                for key, button in self.format_buttons.items()
                if button.isChecked()
            ),
            targets=self.frames,
            classes=self.classes,
            config_path=self.config_path,
            segmentation_classes=self.segmentation_classes,
        )
        self._start_worker()

    def _start_worker(self, plan=None):
        self._cancel_requested = False
        self.worker = ImportWorker(self.options, self, plan)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._finished)
        self._set_busy(True)
        self.setWindowTitle(
            self.tr("Validating import…")
            if plan is None
            else self.tr("Importing 3D objects…")
        )
        self.worker.start()

    def _progress(self, completed, total):
        if not self._cancel_requested:
            self.setWindowTitle(
                self.tr("Importing 3D objects ({completed}/{total})…").format(
                    completed=completed, total=total
                )
            )

    def _finished(self):
        worker = self.worker
        self.worker = None
        plan, error = worker.import_result, worker.error
        applied = worker.plan is not None and plan is not None
        worker.deleteLater()
        self.setWindowTitle(self.tr("Import 3D objects"))
        self._set_busy(False)
        self.cancel_button.setEnabled(True)
        if applied:
            self.import_plan = plan
            self.parent().settings.setValue(
                "import_directory", str(self.output_path.parent)
            )
            QtWidgets.QDialog.accept(self)
            return
        if error is not None:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), error)
        elif self._cancel_requested:
            QtWidgets.QDialog.reject(self)
        elif plan is not None:
            if self.parent().task_type is not None and plan.new_classes:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.windowTitle(),
                    self.tr(
                        "The import contains classes outside this task. Create a new task with these classes before importing."
                    ),
                )
                return
            message = self.tr(
                "Import {objects} objects into {frames} frames?\n"
                "This replaces {previous} existing objects, including locked objects, "
                "and adds {classes} classes.\n"
                "Point segmentation and unmatched frames are preserved."
            ).format(
                objects=plan.object_count,
                frames=len(plan.targets),
                previous=plan.previous_objects,
                classes=plan.new_classes,
            )
            if (
                QtWidgets.QMessageBox.question(
                    self,
                    self.windowTitle(),
                    message,
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                == QtWidgets.QMessageBox.StandardButton.Yes
            ):
                self._start_worker(plan)

    def reject(self):
        super().reject()
        if self.worker is not None:
            self.setWindowTitle(self.tr("Cancelling import…"))
