import json
import os
import os.path as osp
import shutil
from pathlib import Path

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
)
from PyQt5.QtCore import Qt

from anylabeling.views.labeling.logger import logger


class ConfigurationDialog(QDialog):
    """Dialog for configuring save directory and crop size limits."""

    def __init__(self, default_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Cropped Image Settings"))
        self.setModal(True)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #ffffff;
                border-radius: 10px;
            }
            QLabel {
                color: #2c3e50;
                font-size: 13px;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background-color: #f8f9fa;
            }
            QSpinBox {
                padding: 6px;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                background-color: #ffffff;
                min-width: 80px;
                color: #2c3e50;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
                background-color: #f8f9fa;
                border: none;
                border-radius: 3px;
                margin: 1px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #e9ecef;
            }
            QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {
                background-color: #dee2e6;
            }
            QSpinBox::up-arrow {
                image: url(up_arrow.png);
                width: 10px;
                height: 10px;
            }
            QSpinBox::down-arrow {
                image: url(down_arrow.png);
                width: 10px;
                height: 10px;
            }
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                border: none;
                color: white;
                font-weight: bold;
            }
            QPushButton#browse {
                background-color: #5c7cfa;
            }
            QPushButton#ok {
                background-color: #40c057;
            }
            QPushButton#cancel {
                background-color: #868e96;
            }
        """
        )

        # Save directory setting
        self.save_dir = default_dir
        self.dir_label = QLabel(self.tr("Save Directory:"))
        self.dir_line_edit = QLineEdit(self.save_dir)
        self.browse_button = QPushButton(self.tr("Browse"))
        self.browse_button.clicked.connect(self.select_directory)
        self.browse_button.setObjectName("browse")

        # Min size settings
        self.min_width_label = QLabel(self.tr("Minimum Width:"))
        self.min_width_spin = QSpinBox()
        self.min_width_spin.setRange(0, 9999)
        self.min_width_spin.setValue(0)

        self.min_height_label = QLabel(self.tr("Minimum Height:"))
        self.min_height_spin = QSpinBox()
        self.min_height_spin.setRange(0, 9999)
        self.min_height_spin.setValue(0)

        # Max size settings
        self.max_width_label = QLabel(self.tr("Maximum Width:"))
        self.max_width_spin = QSpinBox()
        self.max_width_spin.setRange(1, 9999)
        self.max_width_spin.setValue(9999)

        self.max_height_label = QLabel(self.tr("Maximum Height:"))
        self.max_height_spin = QSpinBox()
        self.max_height_spin.setRange(1, 9999)
        self.max_height_spin.setValue(9999)

        # Ok and Cancel buttons
        self.ok_button = QPushButton(self.tr("OK"))
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button.setObjectName("ok")
        self.cancel_button.setObjectName("cancel")

        # Layouts
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Directory section
        dir_group = QVBoxLayout()
        dir_group.addWidget(QLabel(self.tr("Save Directory")))
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.dir_line_edit)
        dir_layout.addWidget(self.browse_button)
        dir_group.addLayout(dir_layout)
        layout.addLayout(dir_group)

        # Size limits section
        size_group = QVBoxLayout()

        # Minimum size layout
        min_size_layout = QHBoxLayout()
        min_width_group = QVBoxLayout()
        min_width_group.addWidget(self.min_width_label)
        min_width_group.addWidget(self.min_width_spin)
        min_size_layout.addLayout(min_width_group)

        min_size_layout.addSpacing(20)  # Add spacing between width and height

        min_height_group = QVBoxLayout()
        min_height_group.addWidget(self.min_height_label)
        min_height_group.addWidget(self.min_height_spin)
        min_size_layout.addLayout(min_height_group)
        min_size_layout.addStretch()
        size_group.addLayout(min_size_layout)

        # Maximum size layout
        max_size_layout = QHBoxLayout()
        max_width_group = QVBoxLayout()
        max_width_group.addWidget(self.max_width_label)
        max_width_group.addWidget(self.max_width_spin)
        max_size_layout.addLayout(max_width_group)

        max_size_layout.addSpacing(20)  # Add spacing between width and height

        max_height_group = QVBoxLayout()
        max_height_group.addWidget(self.max_height_label)
        max_height_group.addWidget(self.max_height_spin)
        max_size_layout.addLayout(max_height_group)
        max_size_layout.addStretch()
        size_group.addLayout(max_size_layout)

        layout.addLayout(size_group)

        # Buttons section
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def select_directory(self):
        """Open directory selection dialog."""
        selected_dir = QFileDialog.getExistingDirectory(
            self, self.tr("Select Save Directory"), self.save_dir
        )
        if selected_dir:
            self.dir_line_edit.setText(selected_dir)

    def get_values(self):
        """Return the directory and size limits set by the user."""
        return (
            self.dir_line_edit.text(),
            (self.min_width_spin.value(), self.min_height_spin.value()),
            (self.max_width_spin.value(), self.max_height_spin.value()),
        )


class ImageCropperDialog:
    """A class to crop labeled regions from images based on shape labels and save them to a specified directory."""

    def __init__(
        self, filename, image_list=None, output_dir=None, parent=None
    ):
        """Initializes the ImageCropper class with the main image file and options."""
        self.filename = filename
        self.image_list = image_list or []
        self.output_dir = output_dir
        self.parent = parent
        self.save_dir = None
        self.size_limits = None

        self.process_images()

    def configure_settings(self):
        """Opens a unified dialog to set the save directory and crop size limits."""
        default_save_dir = osp.realpath(
            osp.join(osp.dirname(self.filename), "..", "x-anylabeling-crops")
        )
        config_dialog = ConfigurationDialog(default_save_dir)
        if config_dialog.exec_() == QDialog.Accepted:
            self.save_dir, min_size, max_size = config_dialog.get_values()
            self.size_limits = (min_size, max_size)

    def process_images(self):
        """Initiates the image processing to crop and save labeled regions."""
        if not self.filename:
            return

        # Set save directory and size limits if not already configured
        if not self.save_dir or not self.size_limits:
            self.configure_settings()
            if not self.save_dir or not self.size_limits:
                return

        # Initialize processing paths and file lists
        image_file_list = (
            [self.filename] if not self.image_list else self.image_list
        )
        label_dir_path = self.output_dir or osp.dirname(self.filename)

        # Prepare save directory
        save_path = self.save_dir
        if osp.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        # Initialize progress dialog
        progress_dialog = QProgressDialog(
            self.parent.tr("Processing..."),
            self.parent.tr("Cancel"),
            0,
            len(image_file_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.parent.tr("Progress"))
        progress_dialog.setStyleSheet(self._progress_dialog_stylesheet())

        # Perform cropping operation on images
        self._save_cropped_images(
            image_file_list, label_dir_path, save_path, progress_dialog
        )

    def _save_cropped_images(
        self, image_file_list, label_dir_path, save_path, progress_dialog
    ):
        """Saves cropped images from labeled shapes in the images."""
        label_to_count = {}
        (min_width, min_height), (max_width, max_height) = (
            self.size_limits
            if self.size_limits
            else ((None, None), (None, None))
        )

        try:
            for i, image_file in enumerate(image_file_list):
                image_name = osp.basename(image_file)
                label_file = osp.join(
                    label_dir_path, osp.splitext(image_name)[0] + ".json"
                )

                if not osp.exists(label_file):
                    continue

                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                shapes = data.get("shapes", [])

                for shape in shapes:
                    label = shape.get("label", "")
                    points = np.array(shape.get("points", [])).astype(np.int32)
                    shape_type = shape.get("shape_type", "")

                    if (
                        shape_type not in ["rectangle", "polygon", "rotation"]
                        or len(points) < 3
                    ):
                        continue

                    # Calculate and validate bounding rectangle
                    x, y, w, h = cv2.boundingRect(points)
                    if min_width and (w < min_width):
                        continue
                    if min_height and (h < min_height):
                        continue
                    if max_width and (w > max_width):
                        continue
                    if max_height and (h > max_height):
                        continue

                    # Crop and save image
                    self._crop_and_save(
                        image_file,
                        label,
                        points,
                        save_path,
                        label_to_count,
                        shape_type,
                    )

                # Update the progress dialog
                progress_dialog.setValue(i + 1)
                if progress_dialog.wasCanceled():
                    break

            progress_dialog.close()
            self._show_completion_message(save_path)

        except Exception as e:
            progress_dialog.close()
            self._show_error_message(str(e))

    def _crop_and_save(
        self, image_file, label, points, save_path, label_to_count, shape_type
    ):
        """Crops and saves a region from an image.
        
        Args:
            image_file (str): Path to the source image file
            label (str): Label for the cropped region
            points (np.ndarray): Points defining the region to crop
            save_path (str): Base directory to save cropped images
            label_to_count (dict): Counter for each label type
            shape_type (str): Type of shape used for cropping
            
        The cropped image is saved using the original filename as a prefix.
        """
        image_path = Path(image_file)
        orig_filename = image_path.stem

        # Calculate crop coordinates
        x, y, w, h = cv2.boundingRect(points)
        xmin, ymin, xmax, ymax = x, y, x + w, y + h

        # Read image safely handling non-ASCII paths
        try:
            image = cv2.imdecode(
                np.fromfile(str(image_path), dtype=np.uint8), 
                cv2.IMREAD_COLOR
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
        crop_image = image[ymin:ymax, xmin:xmax]

        # Create output directory
        dst_path = Path(save_path) / label
        dst_path.mkdir(parents=True, exist_ok=True)

        # Update counter and create output filename
        label_to_count[label] = label_to_count.get(label, 0) + 1
        dst_file = dst_path / f"{orig_filename}_{label_to_count[label]}-{shape_type}.jpg"

        # Save image safely handling non-ASCII paths
        try:
            is_success, buf = cv2.imencode(".jpg", crop_image)
            if is_success:
                buf.tofile(str(dst_file))
            else:
                raise ValueError(f"Failed to save image: {dst_file}")
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")

    def _show_completion_message(self, save_path):
        """Displays a message box upon successful completion of the cropping operation."""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(self.parent.tr("Cropping completed successfully!"))
        message = self.parent.tr("Cropped images have been saved to:")
        msg_box.setInformativeText(f"{message}\n{osp.realpath(save_path)}")
        msg_box.setWindowTitle(self.parent.tr("Success"))
        msg_box.exec_()

    def _show_error_message(self, error_message):
        """Displays an error message if the cropping operation fails."""
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setText(
            self.parent.tr("Error occurred while saving cropped image.")
        )
        error_dialog.setInformativeText(error_message)
        error_dialog.setWindowTitle(self.parent.tr("Error"))
        error_dialog.exec_()

    def _progress_dialog_stylesheet(self):
        """Returns the stylesheet for the progress dialog."""
        return """
            QProgressDialog QProgressBar {
                border: 1px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressDialog QProgressBar::chunk {
                background-color: orange;
            }
        """
