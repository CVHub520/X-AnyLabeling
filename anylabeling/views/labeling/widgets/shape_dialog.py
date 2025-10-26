import json
import os
import shutil

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QPushButton,
    QMessageBox,
    QProgressDialog,
)
from PyQt5.QtGui import QImage

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.widgets.popup import Popup


def get_ok_btn_style():
    return """
        QPushButton {
            background-color: #0071e3;
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            min-width: 100px;
            height: 36px;
        }
        QPushButton:hover {
            background-color: #0077ED;
        }
        QPushButton:pressed {
            background-color: #0068D0;
        }
    """


def get_spinbox_style():
    return f"""
        QSpinBox {{
            padding: 5px 8px;
            background: white;
            border: 1px solid #d2d2d7;
            border-radius: 6px;
            min-height: 24px;
            selection-background-color: #0071e3;
        }}
        QSpinBox::up-button, QSpinBox::down-button {{
            width: 20px;
            border: none;
            background: #f0f0f0;
        }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
            background: #e0e0e0;
        }}
        QSpinBox::up-arrow {{
            image: url({new_icon_path("caret-up", "svg")});
            width: 12px;
            height: 12px;
        }}
        QSpinBox::down-arrow {{
            image: url({new_icon_path("caret-down", "svg")});
            width: 12px;
            height: 12px;
        }}
    """


class ShapeModifyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle(self.tr("Shape Manager"))
        self.setModal(True)
        self.setFixedSize(480, 240)

        self.image_file_list = self.get_image_file_list()
        self.need_reload = True

        current_index = 1
        if (
            self.parent.filename
            and str(self.parent.filename) in self.parent.fn_to_index
        ):
            current_index = (
                self.parent.fn_to_index[str(self.parent.filename)] + 1
            )

        self.start_index = current_index
        self.end_index = 0

        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: #f5f5f7;
                border-radius: 10px;
            }}
            QLabel {{
                color: #1d1d1f;
                font-size: 13px;
            }}
            QCheckBox {{
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 1px solid #d2d2d7;
                background-color: white;
            }}
            QCheckBox::indicator:checked {{
                background-color: white;
                border: 1px solid #d2d2d7;
                image: url({new_icon_path("checkmark", "svg")});
            }}
            """
        )

        self.init_ui()

    def get_image_file_list(self):
        """
        Retrieves the list of image files from the parent widget.

        Returns:
            list: List of image file paths.
        """
        image_file_list = []
        count = self.parent.file_list_widget.count()
        for c in range(count):
            image_file = self.parent.file_list_widget.item(c).text()
            image_file_list.append(image_file)
        return image_file_list

    def init_ui(self):
        """
        Initializes the user interface components.
        """
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        self.delete_annotations_cb = QCheckBox(
            self.tr("Delete All Annotations")
        )
        self.delete_annotations_cb.setToolTip(
            self.tr("Delete annotation files in the selected frame range")
        )
        self.delete_annotations_cb.toggled.connect(self.on_checkbox_toggled)

        self.delete_images_cb = QCheckBox(
            self.tr("Delete All Images with Annotations")
        )
        self.delete_images_cb.setToolTip(
            self.tr(
                "Delete both image and annotation files in the selected frame range"
            )
        )
        self.delete_images_cb.toggled.connect(self.on_checkbox_toggled)

        self.remove_selected_cb = QCheckBox(self.tr("Remove Selected Shapes"))
        self.remove_selected_cb.setToolTip(
            self.tr(
                "Remove shapes matching selected objects from frames in the range"
            )
        )
        self.remove_selected_cb.toggled.connect(self.on_checkbox_toggled)

        self.add_selected_cb = QCheckBox(self.tr("Add Selected Shapes"))
        self.add_selected_cb.setToolTip(
            self.tr("Add selected shapes to frames in the selected range")
        )
        self.add_selected_cb.toggled.connect(self.on_checkbox_toggled)

        layout.addWidget(self.delete_annotations_cb)
        layout.addWidget(self.delete_images_cb)
        layout.addWidget(self.remove_selected_cb)
        layout.addWidget(self.add_selected_cb)

        self.update_checkbox_state()

        layout.addStretch()

        bottom_layout = QHBoxLayout()
        from_label = QLabel("From:")
        self.from_input = QtWidgets.QSpinBox()
        self.from_input.setMinimum(1)
        self.from_input.setMaximum(len(self.image_file_list))
        self.from_input.setValue(self.start_index)
        self.from_input.setStyleSheet(get_spinbox_style())

        to_label = QLabel("To:")
        self.to_input = QtWidgets.QSpinBox()
        self.to_input.setMinimum(0)
        self.to_input.setMaximum(len(self.image_file_list))
        self.to_input.setSpecialValueText(" ")
        self.to_input.setValue(self.end_index)
        self.to_input.setStyleSheet(get_spinbox_style())

        self.ok_button = QPushButton("Go")
        self.ok_button.setStyleSheet(get_ok_btn_style())
        self.ok_button.clicked.connect(self.accept)

        bottom_layout.addWidget(from_label)
        bottom_layout.addWidget(self.from_input)
        bottom_layout.addSpacing(10)
        bottom_layout.addWidget(to_label)
        bottom_layout.addWidget(self.to_input)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.ok_button)

        layout.addLayout(bottom_layout)
        self.setLayout(layout)

    def update_checkbox_state(self):
        """
        Updates the enabled state of operation checkboxes based on selection.
        """
        selected_shapes = self.parent.canvas.selected_shapes
        has_selection = len(selected_shapes) > 0

        if not has_selection:
            self.remove_selected_cb.setEnabled(False)
            self.add_selected_cb.setEnabled(False)
        else:
            self.remove_selected_cb.setEnabled(True)
            self.add_selected_cb.setEnabled(True)

    def on_checkbox_toggled(self):
        """
        Handles checkbox toggle events to ensure mutual exclusivity.
        """
        sender = self.sender()
        if sender.isChecked():
            checkboxes = [
                self.delete_annotations_cb,
                self.delete_images_cb,
                self.remove_selected_cb,
                self.add_selected_cb,
            ]
            for cb in checkboxes:
                if cb != sender:
                    cb.setChecked(False)

    def accept(self):
        """
        Validates inputs and executes the selected operation.
        """
        start_idx = self.from_input.value()
        end_idx = self.to_input.value()

        if end_idx == 0:
            QMessageBox.warning(
                self,
                self.tr("Invalid Range"),
                self.tr("Please specify the end frame index"),
            )
            return

        if start_idx > end_idx:
            QMessageBox.warning(
                self,
                self.tr("Invalid Range"),
                self.tr("Start index cannot be greater than end index"),
            )
            return

        operation_name = ""
        if self.delete_annotations_cb.isChecked():
            operation_name = self.tr("Delete All Annotations")
        elif self.delete_images_cb.isChecked():
            operation_name = self.tr("Delete All Images with Annotations")
        elif self.remove_selected_cb.isChecked():
            operation_name = self.tr("Remove Selected Shapes")
        elif self.add_selected_cb.isChecked():
            operation_name = self.tr("Add Selected Shapes")
        else:
            QMessageBox.information(
                self,
                self.tr("No Operation"),
                self.tr("Please select an operation to perform"),
            )
            return

        template = self.tr(
            "Are you sure you want to perform '%s' on frames %s to %s?"
        )
        confirm_msg = template % (operation_name, start_idx, end_idx)

        reply = QMessageBox.question(
            self,
            self.tr("Confirm Operation"),
            confirm_msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        if self.delete_annotations_cb.isChecked():
            self.delete_annotations_in_range(start_idx, end_idx)
        elif self.delete_images_cb.isChecked():
            self.delete_images_in_range(start_idx, end_idx)
            self.need_reload = False
        elif self.remove_selected_cb.isChecked():
            self.remove_selected_shapes_in_range(start_idx, end_idx)
        elif self.add_selected_cb.isChecked():
            self.add_selected_shapes_to_range(start_idx, end_idx)

        super().accept()

    def delete_annotations_in_range(self, start_idx, end_idx):
        """
        Deletes annotation files in the specified frame range.

        Args:
            start_idx (int): Starting frame index (inclusive).
            end_idx (int): Ending frame index (inclusive).
        """
        deleted_count = 0

        for i in range(start_idx - 1, end_idx):
            if i >= len(self.image_file_list):
                break

            image_file = self.image_file_list[i]
            label_dir = os.path.dirname(image_file)
            if self.parent.output_dir:
                label_dir = self.parent.output_dir

            filename = os.path.basename(image_file)
            label_file = os.path.join(
                label_dir, os.path.splitext(filename)[0] + ".json"
            )

            if os.path.exists(label_file):
                try:
                    os.remove(label_file)
                    deleted_count += 1

                    item = self.parent.file_list_widget.item(i)
                    if item:
                        item.setCheckState(Qt.Unchecked)
                except Exception as e:
                    logger.error(f"Error deleting {label_file}: {e}")

        logger.info(
            f"Deleted {deleted_count} annotation files in range {start_idx}-{end_idx}"
        )

        template = self.tr("Deleted %s annotation files")
        popup = Popup(
            template % deleted_count,
            self.parent,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self.parent, position="center")

    def delete_images_in_range(self, start_idx, end_idx):
        """
        Deletes image files and their annotations in the specified frame range.

        Args:
            start_idx (int): Starting frame index (inclusive).
            end_idx (int): Ending frame index (inclusive).
        """
        deleted_count = 0
        current_dir = os.path.dirname(self.image_file_list[0])
        save_path = os.path.join(current_dir, "..", "_delete_")
        os.makedirs(save_path, exist_ok=True)

        for i in range(start_idx - 1, end_idx):
            if i >= len(self.image_file_list):
                break

            image_file = self.image_file_list[i]

            if os.path.exists(image_file):
                try:
                    image_name = os.path.basename(image_file)
                    save_file = os.path.join(save_path, image_name)
                    shutil.move(image_file, save_file)

                    label_dir = os.path.dirname(image_file)
                    if self.parent.output_dir:
                        label_dir = self.parent.output_dir
                    label_file = os.path.join(
                        label_dir, os.path.splitext(image_name)[0] + ".json"
                    )

                    if os.path.exists(label_file):
                        os.remove(label_file)

                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting {image_file}: {e}")

        if deleted_count > 0:
            self.parent.reset_state()
            self.parent.import_image_folder(current_dir)

            if len(self.parent.image_list) > 0:
                filename = self.parent.image_list[0]
                self.parent.filename = filename
                if filename:
                    self.parent.load_file(filename)

        logger.info(
            f"Deleted {deleted_count} image files in range {start_idx}-{end_idx}"
        )

        template = self.tr("Deleted %s image files")
        popup = Popup(
            template % deleted_count,
            self.parent,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self.parent, position="center")

    def shapes_are_identical(self, shape1, shape2, tolerance=1.0):
        """
        Checks if two shapes are identical.

        Args:
            shape1: The first shape to compare.
            shape2: The second shape to compare.
            tolerance (float): Tolerance for coordinate comparison, defaults to 1.0.

        Returns:
            bool: True if shapes are identical, False otherwise.
        """
        if shape1.label != shape2.label:
            return False
        if shape1.shape_type != shape2.shape_type:
            return False
        if len(shape1.points) != len(shape2.points):
            return False
        for p1, p2 in zip(shape1.points, shape2.points):
            if (
                abs(p1.x() - p2.x()) > tolerance
                or abs(p1.y() - p2.y()) > tolerance
            ):
                return False
        return True

    def is_shape_within_bounds(self, shape, image_width, image_height):
        """
        Checks if a shape is within image boundaries.

        Args:
            shape: The shape to check.
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            bool: True if shape is within bounds, False otherwise.
        """
        for point in shape.points:
            if point.x() < 0 or point.x() > image_width:
                return False
            if point.y() < 0 or point.y() > image_height:
                return False
        return True

    def remove_selected_shapes_in_range(self, start_idx, end_idx):
        """
        Removes selected shapes from frames in the specified range.

        Args:
            start_idx (int): Starting frame index (inclusive).
            end_idx (int): Ending frame index (inclusive).
        """
        selected_shapes = self.parent.canvas.selected_shapes
        if not selected_shapes:
            return

        progress_dialog = QProgressDialog(
            self.tr("Processing..."),
            self.tr("Cancel"),
            0,
            end_idx - start_idx + 1,
            self,
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Removing Shapes"))
        progress_dialog.setMinimumWidth(400)

        removed_count = 0
        deleted_files_count = 0
        processed = 0

        for i in range(start_idx - 1, end_idx):
            if progress_dialog.wasCanceled():
                break
            if i >= len(self.image_file_list):
                break

            image_file = self.image_file_list[i]
            label_dir = os.path.dirname(image_file)
            if self.parent.output_dir:
                label_dir = self.parent.output_dir

            filename = os.path.basename(image_file)
            label_file = os.path.join(
                label_dir, os.path.splitext(filename)[0] + ".json"
            )

            if os.path.exists(label_file):
                try:
                    with open(label_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    all_shapes = data.get("shapes", [])
                    if not all_shapes:
                        processed += 1
                        progress_dialog.setValue(processed)
                        continue

                    shapes_to_remove = []
                    for shape_data in all_shapes:
                        shape = Shape().load_from_dict(shape_data, close=False)
                        for selected_shape in selected_shapes:
                            if self.shapes_are_identical(
                                shape, selected_shape
                            ):
                                shapes_to_remove.append(shape_data)
                                break

                    if len(shapes_to_remove) == len(all_shapes):
                        os.remove(label_file)
                        removed_count += len(shapes_to_remove)
                        deleted_files_count += 1
                    elif shapes_to_remove:
                        filtered_shapes = [
                            shape
                            for shape in all_shapes
                            if shape not in shapes_to_remove
                        ]
                        removed_count += len(shapes_to_remove)
                        data["shapes"] = filtered_shapes
                        with open(label_file, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)

                except Exception as e:
                    logger.error(f"Error processing {label_file}: {e}")

            processed += 1
            progress_dialog.setValue(processed)

        progress_dialog.close()

        logger.info(
            f"Removed {removed_count} shapes in range {start_idx}-{end_idx}"
        )

        template = self.tr("Removed %s shapes")
        message = template % removed_count

        popup = Popup(
            message,
            self.parent,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self.parent, position="center")

    def add_selected_shapes_to_range(self, start_idx, end_idx):
        """
        Adds selected shapes to frames in the specified range.

        Args:
            start_idx (int): Starting frame index (inclusive).
            end_idx (int): Ending frame index (inclusive).
        """
        selected_shapes = self.parent.canvas.selected_shapes
        if not selected_shapes:
            return

        progress_dialog = QProgressDialog(
            self.tr("Processing..."),
            self.tr("Cancel"),
            0,
            end_idx - start_idx + 1,
            self,
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Adding Shapes"))
        progress_dialog.setMinimumWidth(400)

        added_count = 0
        created_files_count = 0
        skipped_count = 0
        processed = 0

        for i in range(start_idx - 1, end_idx):
            if progress_dialog.wasCanceled():
                break
            if i >= len(self.image_file_list):
                break

            image_file = self.image_file_list[i]
            label_dir = os.path.dirname(image_file)
            if self.parent.output_dir:
                label_dir = self.parent.output_dir

            filename = os.path.basename(image_file)
            label_file = os.path.join(
                label_dir, os.path.splitext(filename)[0] + ".json"
            )

            if not os.path.exists(image_file):
                processed += 1
                progress_dialog.setValue(processed)
                continue

            image = QImage(image_file)
            if image.isNull():
                processed += 1
                progress_dialog.setValue(processed)
                continue

            image_width = image.width()
            image_height = image.height()

            if os.path.exists(label_file):
                try:
                    with open(label_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    logger.error(f"Error reading {label_file}: {e}")
                    processed += 1
                    progress_dialog.setValue(processed)
                    continue
            else:
                data = {
                    "version": "5.5.0",
                    "flags": {},
                    "shapes": [],
                    "imagePath": filename,
                    "imageData": None,
                    "imageHeight": image_height,
                    "imageWidth": image_width,
                }
                created_files_count += 1

            try:
                if "shapes" not in data:
                    data["shapes"] = []

                for selected_shape in selected_shapes:
                    if not self.is_shape_within_bounds(
                        selected_shape, image_width, image_height
                    ):
                        skipped_count += 1
                        continue

                    already_exists = False
                    for shape_data in data["shapes"]:
                        shape = Shape().load_from_dict(shape_data, close=False)
                        if self.shapes_are_identical(shape, selected_shape):
                            already_exists = True
                            break

                    if not already_exists:
                        data["shapes"].append(selected_shape.to_dict())
                        added_count += 1

                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                item = self.parent.file_list_widget.item(i)
                if item:
                    item.setCheckState(Qt.Checked)

            except Exception as e:
                logger.error(f"Error processing {label_file}: {e}")

            processed += 1
            progress_dialog.setValue(processed)

        progress_dialog.close()

        logger.info(
            f"Added {added_count} shapes in range {start_idx}-{end_idx} and skipped {skipped_count} out-of-bounds shapes"
        )

        template = self.tr("Added %s shapes")
        message = template % added_count
        if skipped_count > 0:
            template_skipped = self.tr(" and skipped %s out-of-bounds shapes")
            message += template_skipped % skipped_count

        popup = Popup(
            message,
            self.parent,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self.parent, position="center")
