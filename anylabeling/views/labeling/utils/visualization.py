import json
import os
import os.path as osp
import zipfile
from typing import Literal

import cv2
import numpy as np
from PIL import Image
from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from anylabeling.views.labeling.label_file import LabelFile
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets import Popup
from anylabeling.views.labeling.utils._io import io_open
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.shape import rectangle_from_diagonal
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_export_option_style,
    get_msg_box_style,
    get_ok_btn_style,
    get_progress_dialog_style,
    get_spinbox_style,
)

__all__ = ["save_visualization"]


def save_visualization(self, export_type: Literal["image", "video"]):
    if export_type not in ("image", "video"):
        raise ValueError(f"Unexpected export_type: {export_type}")
    if not _check_filename_exist(self):
        return

    if (
        export_type == "video"
        and _check_video_frame_sizes(self, _image_files(self)) is None
    ):
        return

    options = _show_visualization_dialog(self, export_type)
    if options is None:
        return

    if export_type == "image":
        _save_visualization_image(self, options)
    else:
        _save_visualization_video(self, options)


def _check_filename_exist(self):
    if not self.may_continue():
        return False

    if not self.filename:
        _show_popup(
            self,
            self.tr("Please load an image folder before proceeding!"),
            "warning",
        )
        return False

    return True


def _show_visualization_dialog(self, export_type):
    dialog = QDialog(self)
    title = (
        self.tr("Visualization Image Options")
        if export_type == "image"
        else self.tr("Visualization Video Options")
    )
    dialog.setWindowTitle(title)
    dialog.setMinimumWidth(520)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    current_frame_radio = None
    all_files_radio = None
    path_edit = None
    fps_spin = None

    if export_type == "image":
        layout.addWidget(QLabel(self.tr("Export Range")))
        range_layout = QHBoxLayout()
        range_layout.setSpacing(16)
        current_frame_radio = QRadioButton(self.tr("Current Frame"))
        all_files_radio = QRadioButton(self.tr("All Files"))
        current_frame_radio.setChecked(True)
        range_group = QButtonGroup(dialog)
        range_group.addButton(current_frame_radio)
        range_group.addButton(all_files_radio)
        range_layout.addWidget(current_frame_radio)
        range_layout.addWidget(all_files_radio)
        range_layout.addStretch()
        layout.addLayout(range_layout)
    else:
        path_label = QLabel(self.tr("Save Path"))
        layout.addWidget(path_label)

        path_input_layout = QHBoxLayout()
        path_input_layout.setSpacing(8)

        path_edit = QLineEdit()
        path_edit.setText(_default_video_path(self))
        path_edit.setPlaceholderText(self.tr("Select Save File"))

        def browse_video_path():
            path, _ = QFileDialog.getSaveFileName(
                self,
                self.tr("Save Visualization Video"),
                path_edit.text(),
                self.tr("MP4 Video (*.mp4);;All Files (*)"),
                options=QFileDialog.Option.DontUseNativeDialog,
            )
            if path:
                path_edit.setText(_ensure_extension(path, ".mp4"))

        path_button = QPushButton(self.tr("Browse"))
        path_button.clicked.connect(browse_video_path)
        path_button.setStyleSheet(get_cancel_btn_style())

        path_input_layout.addWidget(path_edit)
        path_input_layout.addWidget(path_button)
        layout.addLayout(path_input_layout)

        video_grid = QGridLayout()
        video_grid.setHorizontalSpacing(12)
        video_grid.setVerticalSpacing(12)
        video_grid.setColumnStretch(0, 1)
        fps_label = QLabel(self.tr("FPS:"))
        fps_spin = QSpinBox()
        fps_spin.setRange(1, 120)
        fps_spin.setValue(25)
        fps_spin.setStyleSheet(get_spinbox_style())
        video_grid.addWidget(fps_label, 0, 0)
        video_grid.addWidget(fps_spin, 0, 1)
        layout.addLayout(video_grid)

    layout.addWidget(QLabel(self.tr("Shape Types")))
    shape_grid = QGridLayout()
    shape_grid.setHorizontalSpacing(12)
    shape_grid.setVerticalSpacing(8)
    shape_checkboxes = []
    for index, shape_type in enumerate(Shape.get_supported_shape()):
        checkbox = QCheckBox(shape_type)
        checkbox.setChecked(False)
        shape_checkboxes.append(checkbox)
        shape_grid.addWidget(checkbox, index // 3, index % 3)
    layout.addLayout(shape_grid)

    layout.addWidget(QLabel(self.tr("Visualization Options")))
    options_grid = QGridLayout()
    options_grid.setHorizontalSpacing(12)
    options_grid.setVerticalSpacing(8)

    save_labels_checkbox = QCheckBox(self.tr("Save Labels"))
    save_labels_checkbox.setChecked(self.canvas.show_labels)
    save_scores_checkbox = QCheckBox(self.tr("Save Scores"))
    save_scores_checkbox.setChecked(self.canvas.show_scores)
    save_groups_checkbox = QCheckBox(self.tr("Save Group IDs"))
    save_groups_checkbox.setChecked(self.canvas.show_groups)
    save_texts_checkbox = QCheckBox(self.tr("Save Texts"))
    save_texts_checkbox.setChecked(self.canvas.show_texts)
    save_masks_checkbox = QCheckBox(self.tr("Save Semi-transparent Masks"))
    save_masks_checkbox.setChecked(self.canvas.show_masks)
    skip_empty_files_checkbox = QCheckBox(self.tr("Skip Empty Files"))
    skip_empty_files_checkbox.setChecked(False)

    option_widgets = [
        save_labels_checkbox,
        save_scores_checkbox,
        save_groups_checkbox,
        save_texts_checkbox,
        save_masks_checkbox,
        skip_empty_files_checkbox,
    ]
    for index, widget in enumerate(option_widgets):
        options_grid.addWidget(widget, index // 2, index % 2)
    layout.addLayout(options_grid)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 8, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QPushButton(self.tr("OK"))
    ok_button.setStyleSheet(get_ok_btn_style())

    def accept_dialog():
        if not any(checkbox.isChecked() for checkbox in shape_checkboxes):
            msg_box = QMessageBox(dialog)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle(self.tr("Shape Types Required"))
            msg_box.setText(
                self.tr(
                    "No shape types selected. Please select at least one shape type before exporting."
                )
            )
            msg_box.setStyleSheet(get_msg_box_style())
            msg_box.exec()
            return

        dialog.accept()

    ok_button.clicked.connect(accept_dialog)

    button_layout.addStretch()
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(ok_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)
    result = dialog.exec()
    if not result:
        return None

    selected_shape_types = {
        checkbox.text()
        for checkbox in shape_checkboxes
        if checkbox.isChecked()
    }
    image_range = "current"
    if export_type == "image" and all_files_radio.isChecked():
        image_range = "all"

    options = {
        "shape_types": selected_shape_types,
        "show_labels": save_labels_checkbox.isChecked(),
        "show_scores": save_scores_checkbox.isChecked(),
        "show_groups": save_groups_checkbox.isChecked(),
        "show_texts": save_texts_checkbox.isChecked(),
        "show_masks": save_masks_checkbox.isChecked(),
        "skip_empty": skip_empty_files_checkbox.isChecked(),
        "image_range": image_range,
    }
    if export_type == "video":
        options["save_path"] = _ensure_extension(path_edit.text(), ".mp4")
        options["fps"] = fps_spin.value()
    return options


def _save_visualization_image(self, options):
    image_range = options["image_range"]
    image_files = _image_files(self)

    if image_range == "all" and not _has_single_directory(image_files):
        _show_popup(
            self,
            self.tr(
                "Batch zip export does not support nested or mixed directories yet."
            ),
            "warning",
            popup_height=65,
        )
        return

    default_path = _default_image_path(self, image_range)
    if image_range == "current":
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Visualization Image"),
            default_path,
            self.tr("PNG Image (*.png);;All Files (*)"),
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        save_path = _ensure_extension(save_path, ".png") if save_path else ""
    else:
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Visualization Images"),
            default_path,
            self.tr("Zip Archive (*.zip);;All Files (*)"),
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        save_path = _ensure_extension(save_path, ".zip") if save_path else ""

    if not save_path:
        return
    if not _confirm_overwrite(self, save_path):
        return

    _ensure_parent_dir(save_path)
    if image_range == "current":
        _export_current_image(self, save_path, options)
    else:
        _export_image_zip(self, save_path, image_files, options)


def _save_visualization_video(self, options):
    save_path = options["save_path"]
    if not save_path:
        return

    image_files = _image_files(self)
    if options["skip_empty"]:
        image_files = _filter_non_empty_files(self, image_files, options)
        if image_files is None:
            return

    if not image_files:
        _show_popup(self, self.tr("No files to export."), "warning")
        return

    frame_size = _check_video_frame_sizes(self, image_files)
    if frame_size is None:
        return
    if not _confirm_overwrite(self, save_path):
        return

    _ensure_parent_dir(save_path)
    width, height = frame_size
    writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        options["fps"],
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        _remove_file(save_path)
        _show_popup(
            self,
            self.tr("Current environment cannot write mp4 files."),
            "error",
            popup_height=65,
        )
        return

    progress_dialog = _create_progress_dialog(
        self, self.tr("Exporting..."), len(image_files)
    )
    canceled = False
    try:
        for index, image_file in enumerate(image_files):
            progress_dialog.setValue(index)
            QApplication.processEvents()
            if progress_dialog.wasCanceled():
                canceled = True
                break

            image = _render_image_file(self, image_file, options)
            writer.write(_qimage_to_bgr_array(image))

        progress_dialog.setValue(len(image_files))
    except Exception as e:
        logger.error(
            f"Error occurred while exporting visualization video: {e}"
        )
        _show_popup(
            self,
            self.tr("Error occurred while exporting visualization video!"),
            "error",
            popup_height=65,
        )
        canceled = True
    finally:
        writer.release()
        progress_dialog.close()

    if canceled:
        _remove_file(save_path)
        return

    _show_popup(
        self,
        self.tr("Visualization video saved to:\n%s") % save_path,
        "copy-green",
        popup_height=65,
    )


def _export_current_image(self, save_path, options):
    try:
        image = _render_current_image(self, options)
        if not image.save(save_path, "PNG"):
            raise RuntimeError(f"Failed to save image: {save_path}")
    except Exception as e:
        logger.error(
            f"Error occurred while exporting visualization image: {e}"
        )
        _show_popup(
            self,
            self.tr("Error occurred while exporting visualization image!"),
            "error",
            popup_height=65,
        )
        return

    _show_popup(
        self,
        self.tr("Visualization image saved to:\n%s") % save_path,
        "copy-green",
        popup_height=65,
    )


def _export_image_zip(self, save_path, image_files, options):
    progress_dialog = _create_progress_dialog(
        self, self.tr("Exporting..."), len(image_files)
    )
    canceled = False
    exported_count = 0

    try:
        with zipfile.ZipFile(
            save_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as zip_file:
            for index, image_file in enumerate(image_files):
                progress_dialog.setValue(index)
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    canceled = True
                    break

                shapes = _load_shapes_for_image(self, image_file, options)
                if options["skip_empty"] and not shapes:
                    continue

                image = _render_image_file(
                    self, image_file, options, shapes=shapes
                )
                archive_name = f"{osp.splitext(osp.basename(image_file))[0]}_visualization.png"
                zip_file.writestr(archive_name, _qimage_to_png_bytes(image))
                exported_count += 1

        progress_dialog.setValue(len(image_files))
    except Exception as e:
        logger.error(
            f"Error occurred while exporting visualization images: {e}"
        )
        _show_popup(
            self,
            self.tr("Error occurred while exporting visualization images!"),
            "error",
            popup_height=65,
        )
        canceled = True
    finally:
        progress_dialog.close()

    if canceled or exported_count == 0:
        _remove_file(save_path)
        if exported_count == 0 and not canceled:
            _show_popup(self, self.tr("No files to export."), "warning")
        return

    _show_popup(
        self,
        self.tr("Visualization images saved to:\n%s") % save_path,
        "copy-green",
        popup_height=65,
    )


def _render_current_image(self, options):
    pixmap = self.canvas.pixmap
    if pixmap is None or pixmap.isNull():
        raise RuntimeError("Current image is empty")
    shapes = _prepare_current_shapes(self, options)
    return self.canvas.render_visualization(
        pixmap,
        shapes,
        show_labels=options["show_labels"],
        show_scores=options["show_scores"],
        show_groups=options["show_groups"],
        show_texts=options["show_texts"],
        show_masks=options["show_masks"],
    )


def _render_image_file(self, image_file, options, shapes=None):
    image = _load_qimage(image_file)
    if image.isNull():
        raise RuntimeError(f"Failed to read image: {image_file}")

    if shapes is None:
        shapes = _load_shapes_for_image(self, image_file, options)

    pixmap = QtGui.QPixmap.fromImage(image)
    return self.canvas.render_visualization(
        pixmap,
        shapes,
        show_labels=options["show_labels"],
        show_scores=options["show_scores"],
        show_groups=options["show_groups"],
        show_texts=options["show_texts"],
        show_masks=options["show_masks"],
    )


def _prepare_current_shapes(self, options):
    shapes = []
    for shape in self.canvas.shapes:
        if shape.shape_type not in options["shape_types"]:
            continue
        if not shape.visible or not self.canvas.is_visible(shape):
            continue
        shape_copy = shape.copy()
        _prepare_shape(shape_copy, options["show_groups"])
        if _is_drawable_shape(shape_copy):
            shapes.append(shape_copy)
    return shapes


def _load_shapes_for_image(self, image_file, options):
    label_file = _label_file_for_image(self, image_file)
    if not osp.exists(label_file):
        return []

    with io_open(label_file, "r") as f:
        data = json.load(f)

    shapes = []
    for shape_data in data.get("shapes", []):
        shape_type = shape_data.get("shape_type", "polygon")
        if shape_type not in options["shape_types"]:
            continue
        shape_dict = dict(shape_data)
        if (
            shape_type == "rectangle"
            and len(shape_dict.get("points", [])) == 2
        ):
            shape_dict["points"] = rectangle_from_diagonal(
                shape_dict["points"]
            )
        shape = Shape().load_from_dict(shape_dict)
        _apply_shape_color(self, shape)
        _prepare_shape(shape, options["show_groups"])
        if _is_drawable_shape(shape):
            shapes.append(shape)
    return shapes


def _filter_non_empty_files(self, image_files, options):
    progress_dialog = _create_progress_dialog(
        self, self.tr("Preparing..."), len(image_files)
    )
    filtered_files = []
    try:
        for index, image_file in enumerate(image_files):
            progress_dialog.setValue(index)
            QApplication.processEvents()
            if progress_dialog.wasCanceled():
                return None
            if _load_shapes_for_image(self, image_file, options):
                filtered_files.append(image_file)
        progress_dialog.setValue(len(image_files))
    except Exception as e:
        logger.error(
            f"Error occurred while preparing visualization video: {e}"
        )
        _show_popup(
            self,
            self.tr("Error occurred while preparing visualization video!"),
            "error",
            popup_height=65,
        )
        return None
    finally:
        progress_dialog.close()
    return filtered_files


def _check_video_frame_sizes(self, image_files):
    progress_dialog = _create_progress_dialog(
        self, self.tr("Checking frame sizes..."), len(image_files)
    )
    base_size = None
    try:
        for index, image_file in enumerate(image_files):
            progress_dialog.setValue(index)
            QApplication.processEvents()
            if progress_dialog.wasCanceled():
                return None

            image_size = _get_image_size(image_file)
            if base_size is None:
                base_size = image_size
            elif image_size != base_size:
                _show_popup(
                    self,
                    self.tr(
                        "Video export requires all frames to have the same size."
                    ),
                    "warning",
                    popup_height=65,
                )
                return None

        progress_dialog.setValue(len(image_files))
    except Exception as e:
        logger.error(f"Error occurred while checking frame sizes: {e}")
        _show_popup(
            self,
            self.tr("Error occurred while checking frame sizes!"),
            "error",
            popup_height=65,
        )
        return None
    finally:
        progress_dialog.close()
    return base_size


def _prepare_shape(shape, show_groups):
    shape.selected = False
    shape.hovered = False
    shape.fill = False
    shape.visible = True
    shape.highlight_clear()
    if not show_groups:
        shape.group_id = None


def _is_drawable_shape(shape):
    point_count = len(shape.points)
    if shape.shape_type == "polygon":
        return point_count >= 3
    if shape.shape_type in ["rectangle", "rotation", "circle"]:
        return point_count >= 2
    if shape.shape_type == "quadrilateral":
        return point_count >= 4
    if shape.shape_type == "point":
        return point_count >= 1
    if shape.shape_type in ["line", "linestrip"]:
        return point_count >= 2
    if shape.shape_type == "cuboid":
        return point_count >= 8
    return False


def _apply_shape_color(self, shape):
    rgb = _get_label_rgb(self, shape.label)
    shape.line_color = QtGui.QColor(*rgb)
    shape.vertex_fill_color = QtGui.QColor(*rgb)
    shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
    shape.fill_color = QtGui.QColor(*rgb, 128)
    shape.select_line_color = QtGui.QColor(255, 255, 255)
    shape.select_fill_color = QtGui.QColor(*rgb, 155)


def _get_label_rgb(self, label):
    if label == "AUTOLABEL_ADD":
        return (144, 238, 144)
    if label == "AUTOLABEL_REMOVE":
        return (255, 182, 193)
    label_info = getattr(self, "label_info", {})
    if label in label_info:
        return tuple(label_info[label]["color"])
    config = getattr(self, "_config", {})
    label_colors = config.get("label_colors") or {}
    if config.get("shape_color") == "manual" and label in label_colors:
        return tuple(label_colors[label])
    if config.get("default_shape_color"):
        return tuple(config["default_shape_color"])
    return (0, 255, 0)


def _label_file_for_image(self, image_file):
    label_dir = self.output_dir or osp.dirname(image_file)
    return osp.join(
        label_dir, osp.splitext(osp.basename(image_file))[0] + ".json"
    )


def _load_qimage(image_file):
    image_data = LabelFile.load_image_file(image_file)
    if not image_data:
        return QtGui.QImage()
    return QtGui.QImage.fromData(image_data)


def _get_image_size(image_file):
    with Image.open(image_file) as image:
        return image.size


def _qimage_to_png_bytes(image):
    buffer = QtCore.QBuffer()
    if not buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly):
        raise RuntimeError("Failed to open PNG buffer")
    try:
        if not image.save(buffer, "PNG"):
            raise RuntimeError("Failed to encode PNG")
        return bytes(buffer.data())
    finally:
        buffer.close()


def _qimage_to_bgr_array(image):
    image = image.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    array = np.frombuffer(ptr, dtype=np.uint8).reshape(
        height, image.bytesPerLine()
    )
    array = array[:, : width * 4].reshape(height, width, 4)
    return array[:, :, :3].copy()


def _image_files(self):
    return list(self.image_list) if self.image_list else [self.filename]


def _has_single_directory(image_files):
    if not image_files:
        return True
    first_dir = osp.normcase(osp.abspath(osp.dirname(image_files[0])))
    return all(
        osp.normcase(osp.abspath(osp.dirname(image_file))) == first_dir
        for image_file in image_files
    )


def _default_image_path(self, image_range):
    if image_range == "current":
        base_name = (
            f"{osp.splitext(osp.basename(self.filename))[0]}_visualization.png"
        )
    else:
        base_name = f"{_default_export_base(self)}_visualizations.zip"
    return osp.realpath(osp.join(_default_export_dir(self), base_name))


def _default_video_path(self):
    base_name = f"{_default_export_base(self)}_visualization.mp4"
    return osp.realpath(osp.join(_default_export_dir(self), base_name))


def _default_export_base(self):
    if self.image_list:
        folder_name = osp.basename(
            osp.normpath(osp.dirname(self.image_list[0]))
        )
        if folder_name:
            return folder_name
    return osp.splitext(osp.basename(self.filename))[0]


def _default_export_dir(self):
    return osp.realpath(
        osp.join(osp.dirname(_image_files(self)[0]), "..", "visualizations")
    )


def _ensure_extension(path, extension):
    if not path:
        return ""
    if osp.splitext(path)[1].lower() == extension:
        return path
    return path + extension


def _ensure_parent_dir(path):
    parent_dir = osp.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _confirm_overwrite(self, save_path):
    if not osp.exists(save_path):
        return True

    msg_box = QMessageBox(self)
    msg_box.setIcon(QMessageBox.Icon.Warning)
    msg_box.setWindowTitle(self.tr("Output File Exists!"))
    msg_box.setText(self.tr("File already exists. Choose an action:"))
    msg_box.setInformativeText(
        self.tr(
            "• Overwrite - Overwrite existing file\n" "• Cancel - Abort export"
        )
    )
    msg_box.addButton(self.tr("Overwrite"), QMessageBox.ButtonRole.YesRole)
    cancel_button = msg_box.addButton(
        self.tr("Cancel"), QMessageBox.ButtonRole.RejectRole
    )
    msg_box.setStyleSheet(get_msg_box_style())
    msg_box.exec()
    return msg_box.clickedButton() != cancel_button


def _create_progress_dialog(self, label_text, maximum):
    progress_dialog = QProgressDialog(
        label_text, self.tr("Cancel"), 0, maximum, self
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(500)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )
    progress_dialog.show()
    QApplication.processEvents()
    return progress_dialog


def _show_popup(self, message, icon, popup_height=None):
    popup = Popup(
        message,
        self,
        msec=3000,
        icon=new_icon_path(icon, "svg"),
    )
    popup.show_popup(self, popup_height=popup_height, position="center")


def _remove_file(path):
    try:
        if path and osp.exists(path):
            os.remove(path)
    except OSError as e:
        logger.warning(f"Failed to remove incomplete export file: {e}")
