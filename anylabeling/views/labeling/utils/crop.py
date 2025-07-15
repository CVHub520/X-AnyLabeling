import json
import multiprocessing
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QLineEdit,
    QHBoxLayout,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QMessageBox,
)

from anylabeling.views.labeling.chatbot.style import ChatbotDialogStyle
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.widgets import Popup
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_export_option_style,
    get_ok_btn_style,
    get_msg_box_style,
    get_progress_dialog_style,
)


__all__ = ["save_crop"]


def crop_and_save(
    image_file,
    label,
    points,
    save_path,
    label_to_count,
    shape_type,
    min_width,
    min_height,
):
    """Crops and saves a region from an image.

    Args:
        image_file (str): Path to the source image file
        label (str): Label for the cropped region
        points (np.ndarray): Points defining the region to crop
        save_path (str): Base directory to save cropped images
        label_to_count (dict): Counter for each label type
        shape_type (str): Type of shape used for cropping
        min_width (int): Minimum width of the cropped region
        min_height (int): Minimum height of the cropped region

    The cropped image is saved using the original filename as a prefix.
    """
    image_path = Path(image_file)
    orig_filename = image_path.stem

    # Calculate crop coordinates
    x, y, w, h = cv2.boundingRect(points)
    if w < min_width or h < min_height:
        return
    xmin, ymin, xmax, ymax = x, y, x + w, y + h

    # Read image safely handling non-ASCII paths
    try:
        image = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError(f"Failed to read image: {image_file}")
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        return

    # Crop image with bounds checking
    height, width = image.shape[:2]
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(width, xmax), min(height, ymax)

    if xmin >= xmax or ymin >= ymax:
        logger.warning(
            f"Invalid crop region: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}"
        )
        return

    cropped_image = image[ymin:ymax, xmin:xmax]
    if cropped_image.size == 0:
        logger.warning(f"Empty cropped image, skipping save")
        return

    # Create output directory
    dst_path = Path(save_path) / label
    dst_path.mkdir(parents=True, exist_ok=True)

    # Update counter and create output filename
    label_to_count[label] = label_to_count.get(label, 0) + 1
    dst_file = (
        dst_path / f"{orig_filename}_{label_to_count[label]}-{shape_type}.jpg"
    )

    # Save image safely handling non-ASCII paths
    try:
        is_success, buf = cv2.imencode(".jpg", cropped_image)
        if is_success and buf is not None:
            with open(str(dst_file), "wb") as f:
                f.write(buf.tobytes())
        else:
            raise ValueError(f"Failed to save image: {dst_file}")
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")


def process_single_image(args):
    """Process a single image with cropping parameters

    Args:
        args: Tuple containing
        (image_file, label_dir_path, save_path, min_width, min_height, label_start_indices)
    """
    (
        image_file,
        label_dir_path,
        save_path,
        min_width,
        min_height,
        label_start_indices,
    ) = args
    try:
        image_name = osp.basename(image_file)
        label_file = osp.join(
            label_dir_path, osp.splitext(image_name)[0] + ".json"
        )

        if not osp.exists(label_file):
            return True

        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        image_path = Path(image_file)
        orig_filename = image_path.stem

        try:
            image = cv2.imdecode(
                np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise ValueError(f"Failed to read image: {image_file}")
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            return False

        for shape in shapes:
            label = shape.get("label", "")
            points = np.array(shape.get("points", [])).astype(np.int32)
            shape_type = shape.get("shape_type", "")

            if (
                shape_type not in ["rectangle", "polygon", "rotation"]
                or len(points) < 3
            ):
                continue

            current_index = label_start_indices[label]
            label_start_indices[label] += 1

            x, y, w, h = cv2.boundingRect(points)
            if w < min_width or h < min_height:
                continue

            height, width = image.shape[:2]
            xmin, ymin = max(0, x), max(0, y)
            xmax, ymax = min(width, x + w), min(height, y + h)

            if xmin >= xmax or ymin >= ymax:
                logger.warning(
                    f"Invalid crop region: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}"
                )
                continue

            cropped_image = image[ymin:ymax, xmin:xmax]
            if cropped_image.size == 0:
                logger.warning(f"Empty cropped image for {dst_file}")
                continue

            dst_path = Path(save_path) / label
            dst_path.mkdir(parents=True, exist_ok=True)

            dst_file = (
                dst_path / f"{orig_filename}_{current_index}-{shape_type}.jpg"
            )

            try:
                is_success, buf = cv2.imencode(".jpg", cropped_image)
                if is_success and buf is not None:
                    with open(str(dst_file), "wb") as f:
                        f.write(buf.tobytes())
                else:
                    raise ValueError(f"Failed to save image: {dst_file}")
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")

        return True
    except Exception as e:
        logger.error(f"Error processing {image_file}: {str(e)}")
        return False


def save_crop(self):
    """Save the cropped image with multiprocessing optimization"""

    if not self.filename:
        popup = Popup(
            self.tr("Please load an image folder before proceeding!"),
            self,
            msec=1000,
            icon=new_icon_path("warning", "svg"),
        )
        popup.show_popup(self, position="center")
        return

    dialog = QDialog(self)
    dialog.setWindowTitle(self.tr("Cropped Image Options"))
    dialog.setMinimumWidth(500)
    dialog.setStyleSheet(get_export_option_style())

    layout = QVBoxLayout()
    layout.setContentsMargins(24, 24, 24, 24)
    layout.setSpacing(16)

    path_layout = QVBoxLayout()
    path_label = QLabel(self.tr("Save Path"))
    path_layout.addWidget(path_label)

    path_input_layout = QHBoxLayout()
    path_input_layout.setSpacing(8)

    path_edit = QLineEdit()
    path_edit.setText(
        osp.realpath(osp.join(osp.dirname(self.filename), "..", "crops"))
    )
    path_edit.setPlaceholderText(self.tr("Select Save Directory"))

    def browse_export_path():
        path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Save Directory"),
            path_edit.text(),
            QFileDialog.DontUseNativeDialog,
        )
        if path:
            path_edit.setText(path)

    path_button = QPushButton(self.tr("Browse"))
    path_button.clicked.connect(browse_export_path)
    path_button.setStyleSheet(get_cancel_btn_style())

    path_input_layout.addWidget(path_edit)
    path_input_layout.addWidget(path_button)
    path_layout.addLayout(path_input_layout)
    layout.addLayout(path_layout)

    min_width_layout = QHBoxLayout()
    min_width_label = QLabel(self.tr("Minimum width:"))
    min_width_spin = QSpinBox()
    min_width_spin.setRange(0, 10000)
    min_width_spin.setValue(0)
    min_width_spin.setMinimumWidth(100)
    min_width_spin.setStyleSheet(
        ChatbotDialogStyle.get_spinbox_style(
            up_arrow_url=new_icon_path("caret-up", "svg"),
            down_arrow_url=new_icon_path("caret-down", "svg"),
        )
    )
    min_width_layout.addWidget(min_width_label)
    min_width_layout.addWidget(min_width_spin)
    layout.addLayout(min_width_layout)

    min_height_layout = QHBoxLayout()
    min_height_label = QLabel(self.tr("Minimum height:"))
    min_height_spin = QSpinBox()
    min_height_spin.setRange(0, 10000)
    min_height_spin.setValue(0)
    min_height_spin.setMinimumWidth(100)
    min_height_spin.setStyleSheet(
        ChatbotDialogStyle.get_spinbox_style(
            up_arrow_url=new_icon_path("caret-up", "svg"),
            down_arrow_url=new_icon_path("caret-down", "svg"),
        )
    )
    min_height_layout.addWidget(min_height_label)
    min_height_layout.addWidget(min_height_spin)
    layout.addLayout(min_height_layout)

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 16, 0, 0)
    button_layout.setSpacing(8)

    cancel_button = QPushButton(self.tr("Cancel"))
    cancel_button.clicked.connect(dialog.reject)
    cancel_button.setStyleSheet(get_cancel_btn_style())

    ok_button = QPushButton(self.tr("OK"))
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

    save_path = path_edit.text()

    if osp.exists(save_path):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle(self.tr("Output Directory Exists!"))
        msg_box.setText(self.tr("Directory already exists. Choose an action:"))
        msg_box.setInformativeText(
            self.tr(
                "• Overwrite - Overwrite existing directory\n"
                "• Cancel - Abort export"
            )
        )

        msg_box.addButton(self.tr("Overwrite"), QMessageBox.YesRole)
        cancel_button = msg_box.addButton(
            self.tr("Cancel"), QMessageBox.RejectRole
        )
        msg_box.setStyleSheet(get_msg_box_style())
        msg_box.exec_()

        clicked_button = msg_box.clickedButton()
        if clicked_button == cancel_button:
            return
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
    else:
        os.makedirs(save_path)

    image_file_list = (
        [self.filename] if not self.image_list else self.image_list
    )
    label_dir_path = self.output_dir or osp.dirname(self.filename)

    progress_dialog = QProgressDialog(
        self.tr("Processing..."),
        self.tr("Cancel"),
        0,
        len(image_file_list),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(
        get_progress_dialog_style(color="#1d1d1f", height=20)
    )
    progress_dialog.show()

    QApplication.processEvents()

    try:
        image_file_list = (
            [self.filename] if not self.image_list else self.image_list
        )
        label_dir_path = self.output_dir or osp.dirname(self.filename)

        label_counts = {}
        for image_file in image_file_list:
            label_file = osp.join(
                label_dir_path,
                osp.splitext(osp.basename(image_file))[0] + ".json",
            )
            if osp.exists(label_file):
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for shape in data.get("shapes", []):
                        label = shape.get("label", "")
                        if label:
                            label_counts[label] = (
                                label_counts.get(label, 0) + 1
                            )

        current_indices = {label: 1 for label in label_counts}

        process_args = [
            (
                image_file,
                label_dir_path,
                save_path,
                min_width_spin.value(),
                min_height_spin.value(),
                current_indices.copy(),
            )
            for image_file in image_file_list
        ]

        is_frozen = getattr(sys, "frozen", False)

        if is_frozen:
            logger.info(
                "Running in PyInstaller environment, using single-thread processing"
            )
            for i, args in enumerate(process_args):
                process_single_image(args)
                progress_dialog.setValue(i + 1)
                QApplication.processEvents()

                if progress_dialog.wasCanceled():
                    return
        else:
            # Use multiprocessing to process images in parallel in the dev environment only.
            num_cores = max(1, int(multiprocessing.cpu_count() * 0.9))
            with multiprocessing.Pool(processes=num_cores) as pool:
                for i, _ in enumerate(
                    pool.imap(process_single_image, process_args)
                ):
                    progress_dialog.setValue(i + 1)
                    QApplication.processEvents()

                    if progress_dialog.wasCanceled():
                        pool.terminate()
                        pool.join()
                        return

        progress_dialog.close()
        popup = Popup(
            self.tr(
                f"Cropped images successfully!\nResults have been saved to:\n{save_path}"
            ),
            self,
            msec=3000,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self, popup_height=65, position="center")

    except Exception as e:
        logger.error(f"Error occurred while exporting cropped images: {e}")
        popup = Popup(
            self.tr(f"Error occurred while exporting cropped images!"),
            self,
            msec=3000,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
