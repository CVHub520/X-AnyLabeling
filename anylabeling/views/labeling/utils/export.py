import json
import os
import os.path as osp
import pathlib
import shutil
import time

from PyQt6 import QtWidgets
from PyQt6.QtCore import QCoreApplication, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QProgressDialog,
)

from anylabeling.views.labeling.label_converter import (
    LabelConverter,
    PoseClassError,
    PoseGroupError,
)
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.widgets import Popup
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.ellipse_io import (
    atomic_bytes,
    export_report,
)
from anylabeling.views.labeling.utils.style import *


def _configured_converter(
    widget: QtWidgets.QWidget, **kwargs: object
) -> LabelConverter:
    """Build a converter from the current effective application settings.

    Args:
        widget: Labeling widget with its loaded application configuration.
        kwargs: Existing converter construction arguments.

    Returns:
        Converter using the same sampling count as the canvas and settings.
    """
    config = getattr(widget, "_config", {})
    count = (
        config.get("canvas", {}).get("ellipse", {}).get("polygon_points", 128)
    )
    return LabelConverter(ellipse_polygon_points=count, **kwargs)


def _prepare_export_directory(
    widget: QtWidgets.QWidget,
    converter: LabelConverter,
    save_path: str,
    mode: str,
    use_classes: bool = False,
    mapping: dict | None = None,
    checked_only: bool = False,
    ignored_labels: tuple[str, ...] = (),
) -> bool:
    """Preflight a selected GUI batch before merging into an output folder.

    Args:
        widget: Labeling widget owning the selected images.
        converter: Configured converter retaining input hashes and outcomes.
        save_path: Output directory; it will never be emptied.
        mode: Supported segmentation or descriptive unsupported mode.
        use_classes: Whether the format requires class lookup.
        mapping: Optional mask mapping.
        checked_only: Exclude unchecked images using the existing mask rule.
        ignored_labels: Existing format-specific ignored class names.

    Returns:
        Whether preflight and output-directory confirmation succeeded.
    """
    image_list = widget.image_list if widget.image_list else [widget.filename]
    labels = []
    try:
        for image_file in image_list:
            candidate = osp.join(
                widget.output_dir or osp.dirname(image_file),
                osp.splitext(osp.basename(image_file))[0] + ".json",
            )
            if (
                checked_only
                and osp.isfile(candidate)
                and converter.read_json(candidate).get("checked", False)
                is not True
            ):
                converter.export_results[str(candidate)] = (
                    "skipped",
                    "unchecked image",
                )
                continue
            labels.append(candidate)
        converter.prepare_export(
            labels,
            mode,
            use_classes=use_classes,
            mapping=mapping,
            ignored_labels=ignored_labels,
        )
        for source in labels + list(image_list):
            source_directory = osp.dirname(osp.abspath(source))
            output = osp.abspath(save_path)
            if (
                osp.splitdrive(source_directory)[0].lower()
                == osp.splitdrive(output)[0].lower()
                and osp.commonpath([source_directory, output]) == output
            ):
                raise ValueError(
                    "Export output must not contain the source annotation directory"
                )
        if osp.exists(save_path):
            if not osp.isdir(save_path):
                raise ValueError("Export destination is not a directory")
            response = QtWidgets.QMessageBox.question(
                widget,
                widget.tr("Output Directory Exists!"),
                widget.tr(
                    "Merge exported files into this directory? Unrelated files will be retained."
                ),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if response != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        os.makedirs(save_path, exist_ok=True)
        return True
    except (ValueError, OSError) as error:
        QtWidgets.QMessageBox.warning(
            widget, widget.tr("Export preflight failed"), str(error)
        )
        return False


class ExportThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        converter,
        image_list,
        label_dir_path,
        save_path,
        mode,
        prefix=None,
    ):
        super().__init__()
        self.converter = converter
        self.image_list = image_list
        self.label_dir_path = label_dir_path
        self.save_path = save_path
        self.mode = mode
        self.prefix = prefix

    def run(self):
        try:
            time.sleep(1)

            if self.mode == "vlm_r1_ovd":
                self.converter.custom_to_vlm_r1_ovd(
                    self.image_list,
                    self.label_dir_path,
                    self.save_path,
                    self.prefix,
                )
            elif self.mode == "mot":
                self.converter.custom_to_mot(
                    self.label_dir_path, self.save_path
                )
            elif self.mode == "mots":
                self.converter.custom_to_mots(
                    self.label_dir_path, self.save_path
                )
            elif self.mode == "odvg":
                self.converter.custom_to_odvg(
                    self.image_list, self.label_dir_path, self.save_path
                )
            else:
                self.converter.custom_to_coco(
                    self.image_list,
                    self.label_dir_path,
                    self.save_path,
                    self.mode,
                )
            report = export_report(self.converter.export_results)
            logger.info(report)
            if self.isInterruptionRequested():
                raise InterruptedError(
                    "Export canceled; completed outputs were retained"
                )
            self.finished.emit(True, report)
        except Exception as e:
            report = export_report(self.converter.export_results)
            logger.error(report)
            self.finished.emit(False, str(e) + "\n" + report)


def _check_filename_exist(self):
    if not self.may_continue():
        return False

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            icon=new_icon_path("warning", "svg"),
        )
        popup.show_popup(self, position="center")
        return False

    return True


def _show_yolo_export_error(parent, image_file, error):
    image_path = osp.abspath(image_file) if image_file else None
    message = (
        QCoreApplication.translate("LabelingWidget", "Failed on image: %s")
        % image_path
        if image_path
        else QCoreApplication.translate("LabelingWidget", "Export failed.")
    )
    if isinstance(error, PoseGroupError):
        message += "\n\n" + QCoreApplication.translate(
            "LabelingWidget",
            "Reason: Pose instance grouping is incomplete or mismatched.\n"
            "Please ensure that each instance has one bounding box and that "
            "its bounding box and keypoints use the same numeric group ID.",
        )
    elif isinstance(error, PoseClassError):
        message += "\n\n" + QCoreApplication.translate(
            "LabelingWidget",
            "Reason: The bounding box label is not defined in the pose "
            "configuration.\nPlease ensure that the bounding box label is "
            "listed under classes in the pose YAML file.",
        )
    elif str(error):
        message += "\n\n" + QCoreApplication.translate(
            "LabelingWidget", "Reason: %s"
        ) % str(error)

    msg_box = QtWidgets.QMessageBox(parent)
    msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
    msg_box.setWindowTitle(
        QCoreApplication.translate("LabelingWidget", "Export Failed")
    )
    msg_box.setText(message)
    msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Ok)
    msg_box.setStyleSheet(get_msg_box_style())
    msg_box.exec()

    loaded_image_path = (
        osp.abspath(parent.filename) if parent.filename else None
    )
    if image_path and image_path != loaded_image_path:
        parent.load_file(image_file)


def _get_yolo_source_root(filename, last_open_dir):
    source_root = osp.dirname(osp.abspath(filename))
    if not last_open_dir:
        return source_root

    last_open_dir = osp.abspath(last_open_dir)
    try:
        if osp.commonpath((last_open_dir, source_root)) == last_open_dir:
            return last_open_dir
    except ValueError:
        pass
    return source_root


def _validate_yolo_export_path(source_root, save_path):
    if not save_path:
        raise ValueError("Please select an export root directory.")

    source_root = osp.realpath(source_root)
    save_path = osp.realpath(save_path)
    if osp.exists(save_path) and not osp.isdir(save_path):
        raise ValueError("The export root path must be a directory.")

    try:
        common_path = osp.commonpath((source_root, save_path))
    except ValueError:
        return
    if common_path == source_root:
        raise ValueError(
            "The export root directory cannot be the loaded image directory "
            "or one of its subdirectories."
        )
    if common_path == save_path:
        raise ValueError(
            "The export root directory cannot contain the loaded image "
            "directory."
        )


def _get_yolo_export_files(image_list, source_root, save_path):
    export_files = []
    label_destinations = {}
    for image_file in image_list:
        try:
            relative_image_path = osp.relpath(image_file, source_root)
        except ValueError:
            relative_image_path = osp.basename(image_file)
        if (
            relative_image_path == osp.pardir
            or relative_image_path.startswith(osp.pardir + osp.sep)
        ):
            relative_image_path = osp.basename(image_file)
        relative_label_path = osp.splitext(relative_image_path)[0] + ".txt"
        destination_key = osp.normcase(osp.normpath(relative_label_path))
        if destination_key in label_destinations:
            raise ValueError(
                "Multiple images map to the same YOLO label file "
                f"'{relative_label_path}':\n"
                f"{label_destinations[destination_key]}\n{image_file}"
            )
        label_destinations[destination_key] = image_file
        export_files.append(
            (
                image_file,
                osp.join(save_path, relative_label_path),
                osp.join(save_path, relative_image_path),
            )
        )
    return export_files


def export_yolo_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    # Handle config/classes file selection based on mode
    if mode == "pose":
        filter = QCoreApplication.translate(
            "LabelingWidget", "Classes Files (*.yaml);;All Files (*)"
        )
        self.yaml_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            QCoreApplication.translate(
                "LabelingWidget",
                "Select a specific yolo-pose config file",
            ),
            "",
            filter,
        )
        if not self.yaml_file:
            return
        try:
            converter = _configured_converter(
                self, pose_cfg_file=self.yaml_file
            )
        except Exception as e:
            logger.error(f"Failed to load pose config: {self.yaml_file}: {e}")
            popup = Popup(
                QCoreApplication.translate(
                    "LabelingWidget", "Invalid pose config file:\n%s"
                )
                % str(e),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
            return

    elif mode in ["hbb", "obb", "seg"]:
        filter = QCoreApplication.translate(
            "LabelingWidget", "Classes Files (*.txt);;All Files (*)"
        )
        self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            QCoreApplication.translate(
                "LabelingWidget", "Select a specific classes file"
            ),
            "",
            filter,
        )
        if not self.classes_file:
            return
        converter = _configured_converter(self, classes_file=self.classes_file)

    source_root = _get_yolo_source_root(
        self.filename, getattr(self, "last_open_dir", None)
    )

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(
        QCoreApplication.translate("LabelingWidget", "Export options")
    )
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(
        QCoreApplication.translate("LabelingWidget", "Export path")
    )
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.realpath(osp.join(source_root, "..", "labels")))
    path_edit.setPlaceholderText(
        QCoreApplication.translate("LabelingWidget", "Select Export Directory")
    )

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            QCoreApplication.translate(
                "LabelingWidget", "Select Export Directory"
            ),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(
        QCoreApplication.translate("LabelingWidget", "Browse")
    )
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    options_label = QtWidgets.QLabel(
        QCoreApplication.translate("LabelingWidget", "Export Options")
    )
    layout.addWidget(options_label)

    save_images_checkbox = QtWidgets.QCheckBox(
        QCoreApplication.translate("LabelingWidget", "Save with images?")
    )
    save_images_checkbox.setChecked(False)
    layout.addWidget(save_images_checkbox)

    skip_empty_files_checkbox = QtWidgets.QCheckBox(
        QCoreApplication.translate("LabelingWidget", "Skip empty labels?")
    )
    skip_empty_files_checkbox.setChecked(False)
    layout.addWidget(skip_empty_files_checkbox)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(
        QCoreApplication.translate("LabelingWidget", "Cancel")
    )
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(
        QCoreApplication.translate("LabelingWidget", "OK")
    )
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()

    if not result:
        return

    save_images = save_images_checkbox.isChecked()
    skip_empty_files = skip_empty_files_checkbox.isChecked()
    save_path = path_edit.text()
    image_list = self.image_list if self.image_list else [self.filename]

    try:
        _validate_yolo_export_path(source_root, save_path)
        export_files = _get_yolo_export_files(
            image_list, source_root, save_path
        )
    except ValueError as error:
        _show_yolo_export_error(self, None, error)
        return

    def get_label_file(image_file):
        label_file_name = osp.splitext(osp.basename(image_file))[0] + ".json"
        label_dir = self.output_dir or osp.dirname(image_file)
        return osp.join(label_dir, label_file_name)

    obb_boundary_policy = "keep"
    if mode == "obb":
        out_of_bounds_count = 0
        for image_file in image_list:
            label_file = get_label_file(image_file)
            if not osp.exists(label_file):
                continue
            data = converter.read_json(label_file)
            image_width = data["imageWidth"]
            image_height = data["imageHeight"]
            for shape in data["shapes"]:
                points = shape["points"]
                if shape["shape_type"] != "rotation" or len(points) != 4:
                    continue
                if any(
                    point[0] < 0
                    or point[0] > image_width
                    or point[1] < 0
                    or point[1] > image_height
                    for point in points
                ):
                    out_of_bounds_count += 1

        if out_of_bounds_count:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg_box.setWindowTitle(
                QCoreApplication.translate(
                    "LabelingWidget", "Out-of-bounds OBBs"
                )
            )
            msg_box.setText(
                QCoreApplication.translate(
                    "LabelingWidget",
                    "Detected %d oriented bounding boxes with points outside "
                    "the image boundaries. Keep them?",
                )
                % out_of_bounds_count
            )
            msg_box.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No
                | QtWidgets.QMessageBox.StandardButton.Cancel
            )
            msg_box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)
            msg_box.setStyleSheet(get_msg_box_style())
            response = QtWidgets.QMessageBox.StandardButton(msg_box.exec())
            if response == QtWidgets.QMessageBox.StandardButton.Cancel:
                return
            if response == QtWidgets.QMessageBox.StandardButton.No:
                obb_boundary_policy = "skip"

    if not _prepare_export_directory(
        self,
        converter,
        save_path,
        "seg" if mode == "seg" else f"YOLO {mode}",
        use_classes=(mode == "seg"),
    ):
        return

    progress_dialog = QProgressDialog(
        QCoreApplication.translate("LabelingWidget", "Exporting..."),
        QCoreApplication.translate("LabelingWidget", "Cancel"),
        0,
        len(image_list),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(
        QCoreApplication.translate("LabelingWidget", "Progress")
    )
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    current_image_file = None
    try:
        for i, (image_file, dst_file, image_dst) in enumerate(export_files):
            current_image_file = image_file
            src_file = get_label_file(image_file)
            os.makedirs(osp.dirname(dst_file), exist_ok=True)

            is_empty_file = converter.custom_to_yolo(
                src_file,
                dst_file,
                mode,
                skip_empty_files=skip_empty_files,
                obb_boundary_policy=obb_boundary_policy,
            )

            if save_images and not (skip_empty_files and is_empty_file):
                os.makedirs(osp.dirname(image_dst), exist_ok=True)
                atomic_bytes(
                    image_dst,
                    pathlib.Path(image_file).read_bytes(),
                    converter._input_fingerprints(src_file),
                )

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        current_image_file = None
        progress_dialog.close()
        template = QCoreApplication.translate(
            "LabelingWidget",
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s",
        )
        message_text = template % save_path
        if progress_dialog.wasCanceled():
            message_text = self.tr(
                "Export canceled; completed outputs were retained."
            )
        report = export_report(converter.export_results)
        logger.info(report)
        message_text += "\n" + report.splitlines()[0]
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        progress_dialog.close()
        failed_image_path = (
            osp.abspath(current_image_file) if current_image_file else None
        )
        if failed_image_path:
            logger.error(
                "Error occurred while exporting annotations for image:\n"
                f"{failed_image_path}\n{e}"
            )
        else:
            logger.error(f"Error occurred while exporting annotations: {e}")

        logger.error(export_report(converter.export_results))
        _show_yolo_export_error(self, current_image_file, e)


def export_voc_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(osp.dirname(self.filename), "..", "Annotations"))
    )
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    options_label = QtWidgets.QLabel(self.tr("Export Options"))
    layout.addWidget(options_label)

    save_images_checkbox = QtWidgets.QCheckBox(self.tr("Save with images?"))
    save_images_checkbox.setChecked(False)
    layout.addWidget(save_images_checkbox)

    skip_empty_files_checkbox = QtWidgets.QCheckBox(
        self.tr("Skip empty labels?")
    )
    skip_empty_files_checkbox.setChecked(False)
    layout.addWidget(skip_empty_files_checkbox)

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
    result = dialog.exec()

    if not result:
        return

    save_images = save_images_checkbox.isChecked()
    skip_empty_files = skip_empty_files_checkbox.isChecked()
    save_path = path_edit.text()

    converter = _configured_converter(self)
    if not _prepare_export_directory(
        self,
        converter,
        save_path,
        "seg" if mode == "polygon" else f"VOC {mode}",
    ):
        return

    image_list = self.image_list if self.image_list else [self.filename]

    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        for i, image_file in enumerate(image_list):
            image_file_name = osp.basename(image_file)
            label_file_name = osp.splitext(image_file_name)[0] + ".json"
            dst_file_name = osp.splitext(image_file_name)[0] + ".xml"

            if self.output_dir:
                src_file = osp.join(self.output_dir, label_file_name)
            else:
                src_file = osp.join(osp.dirname(image_file), label_file_name)
            dst_file = osp.join(save_path, dst_file_name)

            is_empty_file = converter.custom_to_voc(
                image_file, src_file, dst_file, mode, skip_empty_files
            )

            if save_images and not (skip_empty_files and is_empty_file):
                image_dst = osp.join(save_path, image_file_name)
                shutil.copy(image_file, image_dst)

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % save_path
        report = export_report(converter.export_results)
        logger.info(report)
        message_text += "\n" + report.splitlines()[0]
        if any(
            status == "unprocessed"
            for status, _ in converter.export_results.values()
        ):
            message_text = (
                self.tr("Export canceled; completed outputs were retained.")
                + "\n"
                + report.splitlines()[0]
            )
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = (
            f"Error occurred while exporting annotations: {str(e)}\n"
            + export_report(converter.export_results)
        )
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def export_coco_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    if mode == "pose":
        filter = "Classes Files (*.yaml);;All Files (*)"
        self.yaml_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific coco-pose config file"),
            "",
            filter,
        )
        if not self.yaml_file:
            return
        try:
            converter = _configured_converter(
                self, pose_cfg_file=self.yaml_file
            )
        except Exception as e:
            logger.error(f"Failed to load pose config: {self.yaml_file}: {e}")
            popup = Popup(
                self.tr("Invalid pose config file:\n%s") % str(e),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
            return
    elif mode in ["rectangle", "polygon"]:
        filter = "Classes Files (*.txt);;All Files (*)"
        self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific classes file"),
            "",
            filter,
        )
        if not self.classes_file:
            return
        converter = _configured_converter(self, classes_file=self.classes_file)

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(label_dir_path, "..", "annotations"))
    )
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
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
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    if not _prepare_export_directory(
        self,
        converter,
        save_path,
        "seg" if mode == "polygon" else f"COCO {mode}",
        use_classes=(mode == "polygon"),
        ignored_labels=("__ignore__",) if mode == "polygon" else (),
    ):
        return

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    self.export_thread = ExportThread(
        converter, image_list, label_dir_path, save_path, mode
    )

    def on_export_finished(success, error_msg):
        progress_dialog.close()
        if success:
            template = self.tr(
                "Exporting annotations successfully!\n"
                "Results have been saved to:\n"
                "%s"
            )
            message_text = template % save_path
            report = export_report(converter.export_results)
            logger.info(report)
            message_text += "\n" + report.splitlines()[0]
            if any(
                status == "unprocessed"
                for status, _ in converter.export_results.values()
            ):
                message_text = (
                    self.tr(
                        "Export canceled; completed outputs were retained."
                    )
                    + "\n"
                    + report.splitlines()[0]
                )
            popup = Popup(
                message_text,
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while exporting annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.export_thread.finished.connect(on_export_finished)

    progress_dialog.show()
    self.export_thread.start()

    progress_dialog.canceled.connect(self.export_thread.requestInterruption)


def export_dota_annotation(self):
    if not _check_filename_exist(self):
        return

    filter = "Classes Files (*.txt);;All Files (*)"
    self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific classes file"),
        "",
        filter,
    )
    if not self.classes_file:
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(osp.dirname(self.filename), "..", "labelTxt"))
    )
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
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
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()

    converter = _configured_converter(self, classes_file=self.classes_file)
    if not _prepare_export_directory(
        self, converter, save_path, "DOTA detection"
    ):
        return

    image_list = self.image_list if self.image_list else [self.filename]

    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        for i, image_file in enumerate(image_list):
            image_file_name = osp.basename(image_file)
            label_file_name = osp.splitext(image_file_name)[0] + ".json"
            dst_file_name = osp.splitext(image_file_name)[0] + ".txt"

            if self.output_dir:
                src_file = osp.join(self.output_dir, label_file_name)
            else:
                src_file = osp.join(osp.dirname(image_file), label_file_name)
            dst_file = osp.join(save_path, dst_file_name)

            if not osp.exists(src_file):
                pathlib.Path(dst_file).touch()
            else:
                converter.custom_to_dota(src_file, dst_file)

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        template = self.tr(
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % save_path
        report = export_report(converter.export_results)
        logger.info(report)
        message_text += "\n" + report.splitlines()[0]
        if any(
            status == "unprocessed"
            for status, _ in converter.export_results.values()
        ):
            message_text = (
                self.tr("Export canceled; completed outputs were retained.")
                + "\n"
                + report.splitlines()[0]
            )
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = (
            f"Error occurred while exporting annotations: {str(e)}\n"
            + export_report(converter.export_results)
        )
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def _export_mask_files(
    converter,
    image_list,
    output_dir,
    save_path,
    mapping_table,
    include_null_images,
    only_checked_images,
    progress_dialog,
):
    for i, image_file in enumerate(image_list):
        image_file_name = osp.basename(image_file)
        label_file_name = osp.splitext(image_file_name)[0] + ".json"
        dst_file_name = osp.splitext(image_file_name)[0] + ".png"

        if output_dir:
            src_file = osp.join(output_dir, label_file_name)
        else:
            src_file = osp.join(osp.dirname(image_file), label_file_name)
        dst_file = osp.join(save_path, dst_file_name)

        if osp.exists(src_file):
            if (
                not only_checked_images
                or converter.read_json(src_file).get("checked", False) is True
            ):
                converter.custom_to_mask(src_file, dst_file, mapping_table)
        elif include_null_images and not only_checked_images:
            converter.custom_image_to_empty_mask(
                image_file, dst_file, mapping_table
            )

        progress_dialog.setValue(i + 1)
        if progress_dialog.wasCanceled():
            break


def export_mask_annotation(self):
    if not _check_filename_exist(self):
        return

    filter = "JSON Files (*.json);;All Files (*)"
    color_map_file, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        self.tr("Select a specific color_map file"),
        "",
        filter,
    )
    if not color_map_file:
        return

    with open(color_map_file, "r", encoding="utf-8") as f:
        mapping_table = json.load(f)

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.realpath(osp.join(label_dir_path, "..", "masks")))
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    options_label = QtWidgets.QLabel(self.tr("Export Options"))
    layout.addWidget(options_label)

    include_null_images_checkbox = QtWidgets.QCheckBox(
        self.tr("Include images without labels?")
    )
    include_null_images_checkbox.setChecked(False)
    layout.addWidget(include_null_images_checkbox)

    only_checked_images_checkbox = QtWidgets.QCheckBox(
        self.tr("Only export checked images?")
    )
    only_checked_images_checkbox.setChecked(False)
    layout.addWidget(only_checked_images_checkbox)

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
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    include_null_images = include_null_images_checkbox.isChecked()
    only_checked_images = only_checked_images_checkbox.isChecked()
    converter = _configured_converter(self)
    if not _prepare_export_directory(
        self,
        converter,
        save_path,
        "seg",
        mapping=mapping_table,
        checked_only=only_checked_images,
    ):
        return

    image_list = self.image_list if self.image_list else [self.filename]

    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        _export_mask_files(
            converter,
            image_list,
            self.output_dir,
            save_path,
            mapping_table,
            include_null_images,
            only_checked_images,
            progress_dialog,
        )

        progress_dialog.close()
        template = self.tr(
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % save_path
        report = export_report(converter.export_results)
        logger.info(report)
        message_text += "\n" + report.splitlines()[0]
        if any(
            status == "unprocessed"
            for status, _ in converter.export_results.values()
        ):
            message_text = (
                self.tr("Export canceled; completed outputs were retained.")
                + "\n"
                + report.splitlines()[0]
            )
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = (
            f"Error occurred while exporting annotations: {str(e)}\n"
            + export_report(converter.export_results)
        )
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def export_mot_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    converter = _configured_converter(
        self,
    )
    if mode in ["mot", "mots", "odvg"]:
        filter = "Classes Files (*.txt);;All Files (*)"
        self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific classes file"),
            "",
            filter,
        )
        if not self.classes_file:
            return
        converter = _configured_converter(self, classes_file=self.classes_file)

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(osp.realpath(osp.join(label_dir_path, "..", mode)))
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
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
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    if not _prepare_export_directory(
        self,
        converter,
        save_path,
        "seg" if mode == "mots" else f"{mode.upper()} detection",
        use_classes=(mode == "mots"),
    ):
        return

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    self.export_thread = ExportThread(
        converter, image_list, label_dir_path, save_path, mode
    )

    def on_export_finished(success, error_msg):
        progress_dialog.close()
        if success:
            template = self.tr(
                "Exporting annotations successfully!\n"
                "Results have been saved to:\n"
                "%s"
            )
            message_text = template % save_path
            report = export_report(converter.export_results)
            logger.info(report)
            message_text += "\n" + report.splitlines()[0]
            if any(
                status == "unprocessed"
                for status, _ in converter.export_results.values()
            ):
                message_text = (
                    self.tr(
                        "Export canceled; completed outputs were retained."
                    )
                    + "\n"
                    + report.splitlines()[0]
                )
            popup = Popup(
                message_text,
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")
        else:
            message = (
                f"Error occurred while exporting annotations: {str(error_msg)}"
            )
            logger.error(message)
            popup = Popup(
                message,
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

    self.export_thread.finished.connect(on_export_finished)

    progress_dialog.show()
    self.export_thread.start()

    progress_dialog.canceled.connect(self.export_thread.requestInterruption)


def export_odvg_annotation(self):
    export_mot_annotation(self, "odvg")


def export_pporc_annotation(self, mode):
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(label_dir_path, "..", f"ppocr_{mode}"))
    )
    path_edit.setPlaceholderText(self.tr("Select Export Directory"))

    def browse_export_path():
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Export Directory"),
            path_edit.text(),
            QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
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
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    converter = _configured_converter(self)
    if not _prepare_export_directory(
        self, converter, save_path, f"PPOCR {mode}"
    ):
        return

    if mode == "rec":
        save_crop_img_path = osp.join(save_path, "crop_img")
        os.makedirs(save_crop_img_path, exist_ok=True)
        for fname in ("Label.txt", "rec_gt.txt"):
            fpath = osp.join(save_path, fname)
            if osp.exists(fpath):
                os.remove(fpath)
    elif mode == "kie":
        total_class_set = set()
        class_list_file = osp.join(save_path, "class_list.txt")
        ppocr_kie_file = osp.join(save_path, "ppocr_kie.json")
        if osp.exists(ppocr_kie_file):
            os.remove(ppocr_kie_file)

    image_list = self.image_list if self.image_list else [self.filename]
    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, len(image_list), self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )

    try:
        for i, image_file in enumerate(image_list):
            image_file_name = osp.basename(image_file)
            label_file_name = osp.splitext(image_file_name)[0] + ".json"
            label_file = osp.join(osp.dirname(image_file), label_file_name)
            if mode == "rec":
                converter.custom_to_ppocr(
                    image_file, label_file, save_path, mode
                )
            elif mode == "kie":
                class_set = converter.custom_to_ppocr(
                    image_file, label_file, save_path, mode
                )
                total_class_set = total_class_set.union(class_set)

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        if mode == "kie":
            with open(class_list_file, "w") as f:
                for c in total_class_set:
                    f.writelines(f"{c.upper()}\n")

        progress_dialog.close()

        template = self.tr(
            "Exporting annotations successfully!\n"
            "Results have been saved to:\n"
            "%s"
        )
        message_text = template % save_path
        report = export_report(converter.export_results)
        logger.info(report)
        message_text += "\n" + report.splitlines()[0]
        if any(
            status == "unprocessed"
            for status, _ in converter.export_results.values()
        ):
            message_text = (
                self.tr("Export canceled; completed outputs were retained.")
                + "\n"
                + report.splitlines()[0]
            )
        popup = Popup(
            message_text,
            self,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        message = (
            f"Error occurred while exporting annotations: {str(e)}\n"
            + export_report(converter.export_results)
        )
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def export_vlm_r1_ovd_annotation(self):
    if not _check_filename_exist(self):
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle(self.tr("Export VLM-R1 OVD Annotation"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    main_layout = QVBoxLayout()
    main_layout.setContentsMargins(24, 24, 24, 24)
    main_layout.setSpacing(16)

    # --- File path selection ---
    path_layout = QVBoxLayout()
    path_label = QtWidgets.QLabel(self.tr("Export to"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    # Default export path and filename
    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir
    default_export_path = osp.realpath(
        osp.join(label_dir_path, "..", "vlm_r1_ovd.jsonl")
    )

    path_edit = QtWidgets.QLineEdit()
    path_edit.setText(default_export_path)
    path_edit.setPlaceholderText(self.tr("Select Export File"))

    def browse_export_file():
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            dialog,
            self.tr("Select Export File"),
            path_edit.text(),
            "JSONL Files (*.jsonl)",
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            if not path.endswith(".jsonl"):
                path += ".jsonl"
            path_edit.setText(path)

    path_button = QtWidgets.QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_file)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    main_layout.addLayout(path_layout)

    # --- Prefix input ---
    prefix_layout = QVBoxLayout()
    prefix_layout.setSpacing(8)

    prefix_label = QHBoxLayout()
    prefix_label.setSpacing(2)

    prefix_title_label = QtWidgets.QLabel(self.tr("Prefix:"))
    prefix_preview_label = QtWidgets.QLabel("")
    prefix_preview_label.setStyleSheet(
        "color: gray; font-style: italic; padding-left: 5px;"
    )

    prefix_label.addWidget(prefix_title_label)
    prefix_label.addWidget(prefix_preview_label)
    prefix_label.addStretch()

    prefix_edit = QtWidgets.QLineEdit()
    prefix_edit_placeholder = self.tr(
        "Optional prefix for image filenames (e.g., 'path/to/images/')"
    )
    prefix_edit.setPlaceholderText(prefix_edit_placeholder)

    prefix_layout.addLayout(prefix_label)
    prefix_layout.addWidget(prefix_edit)
    main_layout.addLayout(prefix_layout)

    def _update_preview():
        prefix = prefix_edit.text()
        if not prefix:
            prefix = "demo.jpg"
        else:
            prefix += "demo.jpg"
        preview_text = self.tr("{}").format(prefix)
        prefix_preview_label.setText(preview_text)

    prefix_edit.textChanged.connect(_update_preview)
    _update_preview()

    # --- Class Filtering ---
    self.classes_file = None

    # --- Class Label ---
    class_label = QtWidgets.QLabel(self.tr("Use specific classes? (Optional)"))
    main_layout.addWidget(class_label)

    # --- Class Path Layout ---
    class_path_layout = QHBoxLayout()
    class_path_layout.setSpacing(8)

    class_path_edit = QtWidgets.QLineEdit()
    class_path_edit.setPlaceholderText(
        self.tr("Select a specific classes file")
    )

    def _handle_class_file_upload():
        filter = "Classes Files (*.txt);;All Files (*)"
        classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific classes file"),
            "",
            filter,
        )
        class_path_edit.setText(classes_file)

    class_path_button = QtWidgets.QPushButton(self.tr("Upload"))
    class_path_edit.textChanged.connect(
        lambda text: setattr(self, "classes_file", text)
    )
    class_path_button.clicked.connect(_handle_class_file_upload)
    class_path_button.setStyleSheet(get_cancel_btn_style())

    class_path_layout.addWidget(class_path_edit)
    class_path_layout.addWidget(class_path_button)
    main_layout.addLayout(class_path_layout)

    # --- Hint Label ---
    class_hint_label = QtWidgets.QLabel(
        self.tr(
            "Hint: If you don't upload a specific classes file, all unique labels found in one of the annotations will be used for the export."
        )
    )
    class_hint_label.setStyleSheet(
        "color: gray; font-style: italic; padding-left: 5px;"
    )
    class_hint_label.setWordWrap(True)
    main_layout.addWidget(class_hint_label)

    # --- Buttons layout ---
    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QtWidgets.QPushButton(self.tr("OK"))
    ok_button.clicked.connect(dialog.accept)
    ok_button.setStyleSheet(get_ok_btn_style())

    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    main_layout.addLayout(button_layout)

    dialog.setLayout(main_layout)
    result = dialog.exec()

    if not result:
        return

    save_path = path_edit.text()
    prefix = prefix_edit.text().strip()

    # --- File Exists Check ---
    if osp.exists(save_path):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        msg_box.setWindowTitle(self.tr("File Exists!"))
        msg_box.setText(self.tr("File already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Overwrite - Replace existing file\n"  # Escaped newline for informative text
                "• Cancel - Abort export"
            )
        )
        _ = msg_box.addButton(
            self.tr("Overwrite"), QtWidgets.QMessageBox.ButtonRole.YesRole
        )
        cancel_msg_button = msg_box.addButton(
            self.tr("Cancel"), QtWidgets.QMessageBox.ButtonRole.RejectRole
        )
        msg_box.setDefaultButton(cancel_msg_button)

        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button == cancel_msg_button:
            return

    image_list = self.image_list if self.image_list else [self.filename]
    label_dir_path = osp.dirname(self.filename)
    if self.output_dir:
        label_dir_path = self.output_dir

    # --- Attempt to create LabelConverter first ---
    try:
        converter = _configured_converter(self, classes_file=self.classes_file)
    except Exception as e:
        logger.error(f"Failed to initialize LabelConverter: {e}")
        template = self.tr("Error initializing export: %s")
        popup = Popup(
            template % e,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    # --- Progress Dialog ---
    progress_dialog = QProgressDialog(
        self.tr("Exporting..."), self.tr("Cancel"), 0, 0, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setRange(0, 0)
    progress_dialog.setStyleSheet(get_progress_dialog_style())

    try:
        self.export_thread = ExportThread(
            converter,
            image_list,
            label_dir_path,
            save_path,
            "vlm_r1_ovd",
            prefix=prefix,
        )

        def on_export_finished(success, error_msg):
            progress_dialog.close()
            if success:
                template = self.tr(
                    "Exporting annotations successfully!\n"
                    "Results have been saved to:\n"
                    "%s"
                )
                message_text = template % save_path
                report = export_report(converter.export_results)
                logger.info(report)
                message_text += "\n" + report.splitlines()[0]
                if any(
                    status == "unprocessed"
                    for status, _ in converter.export_results.values()
                ):
                    message_text = (
                        self.tr(
                            "Export canceled; completed outputs were retained."
                        )
                        + "\n"
                        + report.splitlines()[0]
                    )
                popup = Popup(
                    message_text,
                    self,
                    icon=new_icon_path("copy-green", "svg"),
                )
                popup.show_popup(self, popup_height=65, position="center")
            else:
                message = f"Error occurred while exporting annotations: {str(error_msg)}"
                logger.error(message)
                popup = Popup(
                    message,
                    self,
                    icon=new_icon_path("error", "svg"),
                )
                popup.show_popup(self, position="center")

        self.export_thread.finished.connect(on_export_finished)

        progress_dialog.show()
        self.export_thread.start()

        progress_dialog.canceled.connect(
            self.export_thread.requestInterruption
        )

    except Exception as e:
        message = (
            f"Error occurred while exporting annotations: {str(e)}\n"
            + export_report(converter.export_results)
        )
        progress_dialog.close()
        logger.error(message)
        popup = Popup(
            message,
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
