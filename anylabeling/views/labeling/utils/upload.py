import json
import os
import os.path as osp
import time
import yaml

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QProgressDialog,
)

from anylabeling.views.labeling.label_converter import LabelConverter
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.widgets import Popup
from anylabeling.views.labeling.utils.style import *


class UploadPPOCRThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, converter, input_file, output_path, image_path, mode):
        super().__init__()
        self.converter = converter
        self.input_file = input_file
        self.output_path = output_path
        self.image_path = image_path
        self.mode = mode

    def run(self):
        try:
            time.sleep(1)

            self.converter.ppocr_to_custom(
                input_file=self.input_file,
                output_path=self.output_path,
                image_path=self.image_path,
                mode=self.mode,
            )

            self.finished.emit(True, "")

        except Exception as e:
            self.finished.emit(False, str(e))


class UploadOdvgThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, converter, input_file, output_path):
        super().__init__()
        self.converter = converter
        self.input_file = input_file
        self.output_path = output_path

    def run(self):
        try:
            time.sleep(1)

            self.converter.odvg_to_custom(
                input_file=self.input_file,
                output_path=self.output_path,
            )

            self.finished.emit(True, "")

        except Exception as e:
            self.finished.emit(False, str(e))


class UploadMotThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, converter, gt_file, output_path, image_path):
        super().__init__()
        self.converter = converter
        self.gt_file = gt_file
        self.output_path = output_path
        self.image_path = image_path

    def run(self):
        try:
            time.sleep(1)

            self.converter.mot_to_custom(
                input_file=self.gt_file,
                output_path=self.output_path,
                image_path=self.image_path,
            )

            self.finished.emit(True, "")

        except Exception as e:
            self.finished.emit(False, str(e))


class UploadCocoThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, converter, input_file, output_dir_path, mode):
        super().__init__()
        self.converter = converter
        self.input_file = input_file
        self.output_dir_path = output_dir_path
        self.mode = mode

    def run(self):
        try:
            time.sleep(1)

            self.converter.coco_to_custom(
                input_file=self.input_file,
                output_dir_path=self.output_dir_path,
                mode=self.mode,
            )

            self.finished.emit(True, "")

        except Exception as e:
            self.finished.emit(False, str(e))


def upload_ppocr_annotation(self, mode):
    if not self.may_continue():
        return

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    if mode == "rec":
        filter = "Attribute Files (*.txt);;All Files (*)"
        input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a custom annotation file (Label.txt)"),
            "",
            filter,
        )

        if not input_file:
            popup = Popup(
                self.tr("Please select a specific rec file!"),
                self,
                icon="anylabeling/resources/icons/warning.svg",
            )
            popup.show_popup(self, position="center")
            return

    elif mode == "kie":
        filter = "Attribute Files (*.json);;All Files (*)"
        input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a custom annotation file (ppocr_kie.json)"),
            "",
            filter,
        )

        if not input_file:
            popup = Popup(
                self.tr("Please select a specific kie file!"),
                self,
                icon="anylabeling/resources/icons/warning.svg",
            )
            popup.show_popup(self, position="center")
            return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    converter = LabelConverter()
    image_path = osp.dirname(self.filename)
    output_path = osp.dirname(self.filename)
    if self.output_dir:
        output_path = self.output_dir
    self.upload_thread = UploadPPOCRThread(
        converter, input_file, output_path, image_path, mode
    )

    def on_upload_finished(success, error_msg):
        progress_dialog.close()
        if success:
            # update and refresh the current canvas
            self.load_file(self.filename)

            popup = Popup(
                self.tr(f"Uploading annotations successfully!"),
                self,
                icon="anylabeling/resources/icons/copy-green.svg",
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            logger.error(
                f"Error occurred while uploading annotations: {error_msg}"
            )
            popup = Popup(
                self.tr(f"Error occurred while uploading annotations!"),
                self,
                icon="anylabeling/resources/icons/error.svg",
            )
            popup.show_popup(self, position="center")

    self.upload_thread.finished.connect(on_upload_finished)

    progress_dialog.show()
    self.upload_thread.start()

    progress_dialog.canceled.connect(self.upload_thread.terminate)


def upload_odvg_annotation(self):
    if not self.may_continue():
        return

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    filter = "OD Files (*.json *.jsonl);;All Files (*)"
    input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific OD file"),
        "",
        filter,
    )

    if not input_file:
        popup = Popup(
            self.tr("Please select a specific OD file!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    converter = LabelConverter()
    output_path = osp.dirname(self.filename)
    if self.output_dir:
        output_path = self.output_dir
    self.upload_thread = UploadOdvgThread(converter, input_file, output_path)

    def on_upload_finished(success, error_msg):
        progress_dialog.close()
        if success:
            # update and refresh the current canvas
            self.load_file(self.filename)

            popup = Popup(
                self.tr(f"Uploading annotations successfully!"),
                self,
                icon="anylabeling/resources/icons/copy-green.svg",
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            logger.error(
                f"Error occurred while uploading annotations: {error_msg}"
            )
            popup = Popup(
                self.tr(f"Error occurred while uploading annotations!"),
                self,
                icon="anylabeling/resources/icons/error.svg",
            )
            popup.show_popup(self, position="center")

    self.upload_thread.finished.connect(on_upload_finished)

    progress_dialog.show()
    self.upload_thread.start()

    progress_dialog.canceled.connect(self.upload_thread.terminate)


def upload_mot_annotation(self, LABEL_OPACITY):
    if not self.may_continue():
        return

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    filter = "Classes Files (*.txt);;All Files (*)"
    classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific classes file"),
        "",
        filter,
    )

    if not classes_file:
        popup = Popup(
            self.tr("Please select a specific classes file!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    filter = "Gt Files (*.txt);;All Files (*)"
    gt_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific gt file"),
        "",
        filter,
    )

    if not gt_file:
        popup = Popup(
            self.tr("Please select a specific gt file!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    # Initialize unique labels
    with open(classes_file, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
        for label in labels:
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    converter = LabelConverter(classes_file=classes_file)
    image_path = osp.dirname(self.filename)
    output_path = image_path
    if self.output_dir:
        output_path = self.output_dir
    self.upload_thread = UploadMotThread(
        converter, gt_file, output_path, image_path
    )

    def on_upload_finished(success, error_msg):
        progress_dialog.close()
        if success:
            # update and refresh the current canvas
            self.load_file(self.filename)

            popup = Popup(
                self.tr(f"Uploading annotations successfully!"),
                self,
                icon="anylabeling/resources/icons/copy-green.svg",
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            logger.error(
                f"Error occurred while uploading annotations: {error_msg}"
            )
            popup = Popup(
                self.tr(f"Error occurred while uploading annotations!"),
                self,
                icon="anylabeling/resources/icons/error.svg",
            )
            popup.show_popup(self, position="center")

    self.upload_thread.finished.connect(on_upload_finished)

    progress_dialog.show()
    self.upload_thread.start()

    progress_dialog.canceled.connect(self.upload_thread.terminate)


def upload_mask_annotation(self, LABEL_OPACITY):
    if not self.may_continue():
        return

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    filter = "Attribute Files (*.json);;All Files (*)"
    color_map_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific color_map file"),
        "",
        filter,
    )

    if not color_map_file:
        popup = Popup(
            self.tr("Please select a specific color_map file!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.dirname(osp.dirname(self.filename)))

    def browse_upload_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            path_edit.text(),
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks
            | QtWidgets.QFileDialog.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_upload_folder)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec_()

    if not result:
        return

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    # Initialize unique labels
    with open(color_map_file, "r", encoding="utf-8") as f:
        mapping_table = json.load(f)
        classes = list(mapping_table["colors"].keys())
        for label in classes:
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )

    label_dir_path = path_edit.text()
    image_dir_path = osp.dirname(self.filename)
    image_file_list = os.listdir(image_dir_path)
    label_file_list = os.listdir(label_dir_path)
    output_dir_path = self.output_dir if self.output_dir else image_dir_path
    converter = LabelConverter()

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(get_progress_dialog_style(
        color="#1d1d1f", height=20
    ))

    try:
        for i, image_filename in enumerate(image_file_list):
            if image_filename.endswith(".json"):
                continue
            label_filename = osp.splitext(image_filename)[0] + ".png"
            data_filename = osp.splitext(image_filename)[0] + ".json"
            if label_filename not in label_file_list:
                continue
            input_file = osp.join(label_dir_path, label_filename)
            output_file = osp.join(output_dir_path, data_filename)
            image_file = osp.join(image_dir_path, image_filename)
            converter.mask_to_custom(
                input_file=input_file,
                output_file=output_file,
                image_file=image_file,
                mapping_table=mapping_table,
            )

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        template = self.tr(
            "Uploading annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % output_dir_path
        popup = Popup(
            message_text,
            self,
            icon="anylabeling/resources/icons/copy-green.svg",
        )
        popup.show_popup(self, popup_height=65, position="center")

        # update and refresh the current canvas
        self.load_file(self.filename)

    except Exception as e:
        logger.error(f"Error occurred while uploading annotations: {e}")
        popup = Popup(
            self.tr(f"Error occurred while uploading annotations!"),
            self,
            icon="anylabeling/resources/icons/error.svg",
        )
        popup.show_popup(self, position="center")


def upload_dota_annotation(self):
    if not self.may_continue():
        return

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.dirname(osp.dirname(self.filename)))

    def browse_upload_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            path_edit.text(),
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks
            | QtWidgets.QFileDialog.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_upload_folder)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    # Button section
    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec_()

    if not result:
        return

    label_dir_path = path_edit.text()
    image_dir_path = osp.dirname(self.filename)
    label_file_list = os.listdir(label_dir_path)
    output_dir_path = self.output_dir if self.output_dir else image_dir_path
    converter = LabelConverter()

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(get_progress_dialog_style(
        color="#1d1d1f", height=20
    ))

    try:
        for i, image_path in enumerate(image_list):
            image_filename = osp.basename(image_path)
            label_filename = osp.splitext(image_filename)[0] + ".txt"
            if label_filename not in label_file_list:
                continue

            input_file = osp.join(label_dir_path, label_filename)
            output_file = osp.join(
                output_dir_path, osp.splitext(image_filename)[0] + ".json"
            )
            image_file = osp.join(image_dir_path, image_filename)

            converter.dota_to_custom(
                input_file=input_file,
                output_file=output_file,
                image_file=image_file,
            )

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Uploading annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % output_dir_path
        popup = Popup(
            message_text,
            self,
            icon="anylabeling/resources/icons/copy-green.svg",
        )
        popup.show_popup(self, popup_height=65, position="center")

        # update and refresh the current canvas
        self.load_file(self.filename)

    except Exception as e:
        logger.error(f"Error occurred while uploading annotations: {e}")
        popup = Popup(
            self.tr(f"Error occurred while uploading annotations!"),
            self,
            icon="anylabeling/resources/icons/error.svg",
        )
        popup.show_popup(self, position="center")


def upload_coco_annotation(self, mode):
    if not self.may_continue():
        return

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    filter = "Attribute Files (*.json);;All Files (*)"
    input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a custom coco annotation file"),
        "",
        filter,
    )

    if not input_file:
        popup = Popup(
            self.tr("Please select a specific coco annotation file!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    output_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        output_dir_path = self.output_dir

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    self.upload_thread = UploadCocoThread(
        LabelConverter(), input_file, output_dir_path, mode
    )

    def on_upload_finished(success, error_msg):
        progress_dialog.close()
        if success:
            # update and refresh the current canvas
            self.load_file(self.filename)

            popup = Popup(
                self.tr(f"Uploading annotations successfully!"),
                self,
                icon="anylabeling/resources/icons/copy-green.svg",
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            logger.error(
                f"Error occurred while uploading annotations: {error_msg}"
            )
            popup = Popup(
                self.tr(f"Error occurred while uploading annotations!"),
                self,
                icon="anylabeling/resources/icons/error.svg",
            )
            popup.show_popup(self, position="center")

    self.upload_thread.finished.connect(on_upload_finished)

    progress_dialog.show()
    self.upload_thread.start()

    progress_dialog.canceled.connect(self.upload_thread.terminate)


def upload_voc_annotation(self, mode):
    if not self.may_continue():
        return

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.dirname(osp.dirname(self.filename)))

    def browse_upload_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            path_edit.text(),
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks
            | QtWidgets.QFileDialog.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_upload_folder)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec_()

    if not result:
        return

    label_dir_path = path_edit.text()
    image_dir_path = osp.dirname(self.filename)
    label_file_list = os.listdir(label_dir_path)
    output_dir_path = self.output_dir if self.output_dir else image_dir_path
    converter = LabelConverter()

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Uploading..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(get_progress_dialog_style(
        color="#1d1d1f", height=20
    ))

    try:
        for i, image_path in enumerate(image_list):
            image_filename = osp.basename(image_path)
            label_filename = osp.splitext(image_filename)[0] + ".xml"
            if label_filename not in label_file_list:
                continue

            input_file = osp.join(label_dir_path, label_filename)
            output_file = osp.join(
                output_dir_path, osp.splitext(image_filename)[0] + ".json"
            )
            converter.voc_to_custom(
                input_file=input_file,
                output_file=output_file,
                image_filename=image_filename,
                mode=mode,
            )

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Uploading annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % output_dir_path
        popup = Popup(
            message_text,
            self,
            icon="anylabeling/resources/icons/copy-green.svg",
        )
        popup.show_popup(self, popup_height=65, position="center")

        # update and refresh the current canvas
        self.load_file(self.filename)

    except Exception as e:
        logger.error(f"Error occurred while uploading annotations: {e}")
        popup = Popup(
            self.tr(f"Error occurred while uploading annotations!"),
            self,
            icon="anylabeling/resources/icons/error.svg",
        )
        popup.show_popup(self, position="center")


def upload_yolo_annotation(self, mode, LABEL_OPACITY):
    if not self.may_continue():
        return

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    if mode == "pose":
        filter = "Classes Files (*.yaml);;All Files (*)"
        self.yaml_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific yolo-pose config file"),
            "",
            filter,
        )
        if not self.yaml_file:
            popup = Popup(
                self.tr("Please select a specific config file!"),
                self,
                icon="anylabeling/resources/icons/warning.svg",
            )
            popup.show_popup(self, position="center")
            return

        labels = []
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            for class_name, keypoint_name in data["classes"].items():
                labels.append(class_name)
                labels.extend(keypoint_name)
        converter = LabelConverter(pose_cfg_file=self.yaml_file)

    elif mode in ["hbb", "obb", "seg"]:
        filter = "Classes Files (*.txt);;All Files (*)"
        self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific classes file"),
            "",
            filter,
        )
        if not self.classes_file:
            popup = Popup(
                self.tr("Please select a specific config file!"),
                self,
                icon="anylabeling/resources/icons/warning.svg",
            )
            popup.show_popup(self, position="center")
            return

        with open(self.classes_file, "r", encoding="utf-8") as f:
            labels = f.read().splitlines()
        converter = LabelConverter(classes_file=self.classes_file)

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Upload Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Select Upload Folder"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.dirname(osp.dirname(self.filename)))

    def browse_upload_folder():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Upload Folder"),
            path_edit.text(),
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks
            | QtWidgets.QFileDialog.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_upload_folder)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec_()

    if not result:
        return

    label_dir_path = path_edit.text()
    image_dir_path = osp.dirname(self.filename)
    image_file_list = os.listdir(image_dir_path)
    label_file_list = os.listdir(label_dir_path)
    output_dir_path = self.output_dir if self.output_dir else image_dir_path

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Warning"))
    response.setText(self.tr("Current annotation will be lost"))
    response.setInformativeText(
        self.tr(
            "You are going to upload new annotations to this task. Continue?"
        )
    )
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())

    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    progress_dialog = QProgressDialog(
        self.tr("Uploading..."),
        self.tr("Cancel"),
        0,
        len(image_file_list),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(get_progress_dialog_style(
        color="#1d1d1f", height=20
    ))

    try:
        for i, image_filename in enumerate(image_file_list):
            if image_filename.endswith(".json"):
                continue
            label_filename = osp.splitext(image_filename)[0] + ".txt"
            data_filename = osp.splitext(image_filename)[0] + ".json"
            if label_filename not in label_file_list:
                continue
            input_file = osp.join(label_dir_path, label_filename)
            output_file = osp.join(output_dir_path, data_filename)
            image_file = osp.join(image_dir_path, image_filename)

            if mode in ["hbb", "seg"]:
                converter.yolo_to_custom(
                    input_file=input_file,
                    output_file=output_file,
                    image_file=image_file,
                    mode=mode,
                )
            elif mode == "obb":
                converter.yolo_obb_to_custom(
                    input_file=input_file,
                    output_file=output_file,
                    image_file=image_file,
                )
            elif mode == "pose":
                converter.yolo_pose_to_custom(
                    input_file=input_file,
                    output_file=output_file,
                    image_file=image_file,
                )

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        self.load_file(self.filename)
        popup = Popup(
            self.tr("Upload completed successfully!"),
            self,
            icon="anylabeling/resources/icons/copy-green.svg",
        )
        popup.show_popup(self, position="center")

        # Initialize unique labels
        for label in labels:
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )

    except Exception as e:
        logger.error(f"Error occurred while uploading annotations: {e}")
        progress_dialog.close()
        popup = Popup(
            self.tr("Error occurred while uploading annotations!"),
            self,
            icon="anylabeling/resources/icons/error.svg",
        )
        popup.show_popup(self, position="center")


def upload_shape_attrs_file(self, LABEL_OPACITY):
    filter = "Shape Attributes Files (*.json);;All Files (*)"
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific shape attributes file"),
        "",
        filter,
    )
    if not file_path:
        popup = Popup(
            self.tr("Please select a specific shape attributes file!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            self.attributes = json.load(f)
            for label in list(self.attributes.keys()):
                if not self.unique_label_list.find_items_by_label(label):
                    item = self.unique_label_list.create_item_from_label(label)
                    self.unique_label_list.addItem(item)
                    rgb = self._get_rgb_by_label(label)
                    self.unique_label_list.set_item_label(
                        item, label, rgb, LABEL_OPACITY
                    )

        # update the shape attributes dialog
        self.shape_attributes.show()
        self.scroll_area.show()
        self.canvas.h_shape_is_hovered = False
        self.canvas.mode_changed.disconnect(self.set_edit_mode)

        popup = Popup(
            self.tr(f"Uploading shape attributes file successfully!"),
            self,
            icon="anylabeling/resources/icons/copy-green.svg",
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        logger.error(
            f"Error occurred while uploading shape attributes file: {e}"
        )
        popup = Popup(
            self.tr(f"Error occurred while uploading shape attributes file!"),
            self,
            icon="anylabeling/resources/icons/error.svg",
        )
        popup.show_popup(self, position="center")


def upload_label_flags_file(self, LABEL_OPACITY):
    filter = "Label Flags Files (*.yaml);;All Files (*)"
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific flags file"),
        "",
        filter,
    )
    if not file_path:
        popup = Popup(
            self.tr("Please select a specific flags file!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Each line in the file is an flag-level flag
            self.label_flags = yaml.safe_load(f)
            for label in list(self.label_flags.keys()):
                if not self.unique_label_list.find_items_by_label(label):
                    item = self.unique_label_list.create_item_from_label(label)
                    self.unique_label_list.addItem(item)
                    rgb = self._get_rgb_by_label(label)
                    self.unique_label_list.set_item_label(
                        item, label, rgb, LABEL_OPACITY
                    )

        # update the label dialog
        self.label_dialog.upload_flags(self.label_flags)

        popup = Popup(
            self.tr(f"Uploading flags file successfully!"),
            self,
            icon="anylabeling/resources/icons/copy-green.svg",
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        logger.error(f"Error occurred while uploading flags file: {e}")
        popup = Popup(
            self.tr(f"Error occurred while uploading flags file!"),
            self,
            icon="anylabeling/resources/icons/error.svg",
        )
        popup.show_popup(self, position="center")


def upload_image_flags_file(self):
    filter = "Image Flags Files (*.txt);;All Files (*)"
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific flags file"),
        "",
        filter,
    )
    if not file_path:
        popup = Popup(
            self.tr("Please select a specific flags file!"),
            self,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Each line in the file is an image-level flag
            self.image_flags = f.read().splitlines()
            self.load_flags({k: False for k in self.image_flags})
        self.flag_dock.show()

        # update and refresh the current canvas
        self.load_file(self.filename)

        popup = Popup(
            self.tr(f"Uploading flags file successfully!"),
            self,
            icon="anylabeling/resources/icons/copy-green.svg",
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        logger.error(f"Error occurred while uploading flags file: {e}")
        popup = Popup(
            self.tr(f"Error occurred while uploading flags file!"),
            self,
            icon="anylabeling/resources/icons/error.svg",
        )
        popup.show_popup(self, position="center")
