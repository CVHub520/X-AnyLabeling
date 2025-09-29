import os
import os.path as osp
import shutil
from typing import List

import PIL.Image
import PIL.ImageOps
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QObject, pyqtSignal

from ...labeling.logger import logger


class ExifScannerWorker(QObject):
    exif_files_found = pyqtSignal(list)
    scan_finished = pyqtSignal()

    def __init__(self, image_files: List[str], parent=None):
        super().__init__(parent)
        self.image_files = image_files
        self._should_stop = False

    def stop(self):
        self._should_stop = True

    def scan_files(self):
        exif_files = []
        for filename in self.image_files:
            if self._should_stop:
                break
            try:
                with PIL.Image.open(filename) as img:
                    exif = img.getexif()
                    orientation = exif.get(0x0112, 1)
                    if orientation not in (1, None):
                        exif_files.append(filename)
            except Exception:
                continue

        if not self._should_stop and exif_files:
            self.exif_files_found.emit(exif_files)

        self.scan_finished.emit()


class AsyncExifScanner(QObject):
    exif_detected = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.thread = None

    def start_scan(self, image_files: List[str]):
        if self.thread and self.thread.isRunning():
            self.stop_scan()

        self.thread = QtCore.QThread(self)
        self.worker = ExifScannerWorker(image_files)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.scan_files)
        self.worker.exif_files_found.connect(self.exif_detected)
        self.worker.scan_finished.connect(self._cleanup_thread)

        self.thread.start()

    def _cleanup_thread(self):
        if self.thread:
            try:
                if self.thread.isRunning():
                    self.thread.quit()
                    if not self.thread.wait(3000):
                        self.thread.terminate()
                        self.thread.wait()
                self.thread.deleteLater()
                self.thread = None
            except RuntimeError:
                self.thread = None
        if self.worker:
            try:
                self.worker.deleteLater()
            except RuntimeError:
                pass
            self.worker = None

    def stop_scan(self):
        if self.worker:
            self.worker.stop()
        self._cleanup_thread()


class ExifProcessingDialog:

    @staticmethod
    def show_detection_dialog(parent, exif_count: int) -> bool:
        reply = QtWidgets.QMessageBox.question(
            parent,
            parent.tr("EXIF Orientation Detected"),
            parent.tr(
                "Detected %s images with EXIF orientation data. "
                "Direct annotation without correction may cause training anomalies.\n\n"
                "We will process these images in background and create backups in "
                "'x-anylabeling-exif-backup' folder under current directory. "
                "This may take some time.\n\n"
                "Continue processing or ignore?"
            )
            % exif_count,
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Ok,
        )
        return reply == QtWidgets.QMessageBox.Ok

    @staticmethod
    def process_exif_files_with_progress(parent, exif_files: List[str]):
        progress = QtWidgets.QProgressDialog(
            parent.tr("Processing EXIF orientation..."),
            parent.tr("Cancel"),
            0,
            len(exif_files),
            parent,
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)

        template = parent.tr("Processing: %s")
        for i, filename in enumerate(exif_files):
            if progress.wasCanceled():
                break
            progress.setLabelText(template % osp.basename(filename))
            progress.setValue(i)
            QtWidgets.QApplication.processEvents()
            ExifProcessingDialog._process_single_file(filename)

        progress.setValue(len(exif_files))

        if exif_files:
            backup_dir = osp.join(
                osp.dirname(osp.dirname(exif_files[0])),
                "x-anylabeling-exif-backup",
            )
            template = parent.tr(
                "Successfully processed %s images.\n\n"
                "Original images backed up to:\n"
                "%s"
            )
            QtWidgets.QMessageBox.information(
                parent,
                parent.tr("EXIF Processing Complete"),
                template % (len(exif_files), backup_dir),
            )

    @staticmethod
    def _process_single_file(filename: str):
        try:
            with PIL.Image.open(filename) as img:
                exif = img.getexif()
                orientation = exif.get(0x0112, 1)
                if orientation in (1, None):
                    return

                corrected_img = PIL.ImageOps.exif_transpose(img)

                backup_dir = osp.join(
                    osp.dirname(osp.dirname(filename)),
                    "x-anylabeling-exif-backup",
                )
                os.makedirs(backup_dir, exist_ok=True)
                backup_filename = osp.join(backup_dir, osp.basename(filename))
                shutil.copy2(filename, backup_filename)
                corrected_img.save(filename)

        except Exception as e:
            logger.error(
                f"Error processing EXIF orientation for {filename}: {e}"
            )
