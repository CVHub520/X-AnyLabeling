import os
import csv
import json
import zipfile

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressDialog,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import get_progress_dialog_style
from anylabeling.views.labeling.widgets.popup import Popup


overview_dialog_styles = f"""
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

    .secondary-button {{
        background-color: #f5f5f7;
        color: #1d1d1f;
        border: 1px solid #d2d2d7;
        border-radius: 8px;
        font-weight: 500;
        min-width: 100px;
        height: 36px;
    }}
    .secondary-button:hover {{
        background-color: #e5e5e5;
    }}
    .secondary-button:pressed {{
        background-color: #d5d5d5;
    }}

    .primary-button {{
        background-color: #0071e3;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        min-width: 100px;
        height: 36px;
    }}
    .primary-button:hover {{
        background-color: #0077ED;
    }}
    .primary-button:pressed {{
        background-color: #0068D0;
    }}
"""


class OverviewDialog(QtWidgets.QDialog):
    """
    This dialog displays an overview of the label information and shape information for the images in the current project.
    It allows the user to select a range of images to display and export the data as a CSV file.
    """

    DIFFICULT_HIGHLIGHT_COLOR = QColor(255, 153, 0, 128)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.supported_shape = parent.supported_shape
        self.image_file_list = self.get_image_file_list()
        self.start_index = 1
        self.end_index = len(self.image_file_list)
        self.showing_label_infos = True
        self.current_shape_infos = []
        self.shape_to_image_file = []
        if self.image_file_list:
            self.init_ui()

    def init_ui(self):
        """
        Initialize the UI components for the overview dialog.
        """
        self.setWindowTitle(self.tr("Overview"))
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        self.resize(960, 480)
        self.move_to_center()

        layout = QVBoxLayout(self)
        self.table = QTableWidget(self)

        self.populate_table()

        layout.addWidget(self.table)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)

        range_layout = QHBoxLayout()
        range_layout.addStretch(1)

        from_label = QLabel("From:")
        self.from_input = QSpinBox()
        self.from_input.setMinimum(1)
        self.from_input.setMaximum(len(self.image_file_list))
        self.from_input.setSingleStep(1)
        self.from_input.setValue(self.start_index)
        self.from_input.setProperty("class", "")
        range_layout.addWidget(from_label)
        range_layout.addWidget(self.from_input)

        to_label = QLabel("To:")
        self.to_input = QSpinBox()
        self.to_input.setMinimum(1)
        self.to_input.setMaximum(len(self.image_file_list))
        self.to_input.setSingleStep(1)
        self.to_input.setValue(len(self.image_file_list))
        self.to_input.setProperty("class", "")
        range_layout.addWidget(to_label)
        range_layout.addWidget(self.to_input)

        self.range_button = QPushButton("Go")
        self.range_button.setProperty("class", "primary-button")
        range_layout.addWidget(self.range_button)
        self.range_button.clicked.connect(self.update_range)

        range_layout.addStretch(1)

        # Add export button for exporting data
        self.export_button = QPushButton(self.tr("Export"))
        self.export_button.setProperty("class", "secondary-button")

        # Add toggle button to switch between label_infos and shape_infos
        self.toggle_button = QPushButton(self.tr("Shape"))
        self.toggle_button.setProperty("class", "secondary-button")
        self.toggle_button.clicked.connect(self.toggle_info)

        range_and_export_layout = QHBoxLayout()
        range_and_export_layout.addWidget(self.toggle_button, 0, Qt.AlignLeft)
        range_and_export_layout.addStretch(1)
        range_and_export_layout.addLayout(range_layout)
        range_and_export_layout.addStretch(1)
        range_and_export_layout.addWidget(self.export_button, 0, Qt.AlignRight)

        layout.addLayout(range_and_export_layout)

        self.export_button.clicked.connect(self.export_to_csv)

        self.setStyleSheet(overview_dialog_styles)

        self.exec_()

    def move_to_center(self):
        """
        Move the dialog to the center of the screen.
        """
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def get_image_file_list(self):
        """
        Get the list of image files in the current project.
        """
        image_file_list = []
        count = self.parent.file_list_widget.count()
        for c in range(count):
            image_file = self.parent.file_list_widget.item(c).text()
            image_file_list.append(image_file)
        return image_file_list

    def get_label_infos(self, start_index: int = -1, end_index: int = -1):
        """
        Get the label information for the images in the current project.
        """
        initial_nums = [0 for _ in range(len(self.supported_shape))]
        label_infos = {}
        shape_infos = []
        self.shape_to_image_file = []

        progress_dialog = QProgressDialog(
            self.tr("Loading..."),
            self.tr("Cancel"),
            0,
            len(self.image_file_list),
            self,
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setMinimumWidth(400)
        progress_dialog.setMinimumHeight(150)
        progress_dialog.setStyleSheet(
            get_progress_dialog_style(color="#1d1d1f", height=20)
        )

        if start_index == -1:
            start_index = self.start_index
        if end_index == -1:
            end_index = self.end_index
        for i, image_file in enumerate(self.image_file_list):
            if i < start_index - 1 or i > end_index - 1:
                continue
            label_dir, filename = os.path.split(image_file)
            if self.parent.output_dir:
                label_dir = self.parent.output_dir
            label_file = os.path.join(
                label_dir, os.path.splitext(filename)[0] + ".json"
            )
            if not os.path.exists(label_file):
                continue
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            filename = data["imagePath"]
            shapes = data.get("shapes", [])
            for shape in shapes:
                if "label" not in shape or "shape_type" not in shape:
                    continue
                shape_type = shape["shape_type"]
                if shape_type not in self.supported_shape:
                    logger.warning(
                        f"Invalid shape_type {shape_type} of {label_file}!"
                    )
                    continue
                label = shape["label"]
                score = shape.get("score", 0.0)
                flags = shape.get("flags", {})
                points = shape.get("points", [])
                group_id = shape.get("group_id", -1)
                difficult = shape.get("difficult", False)
                description = shape.get("description", "")
                kie_linking = shape.get("kie_linking", [])
                if label not in label_infos:
                    label_infos[label] = dict(
                        zip(self.supported_shape, initial_nums)
                    )
                label_infos[label][shape_type] += 1
                current_shape = dict(
                    filename=filename,
                    label=label,
                    score=score,
                    flags=flags,
                    points=points,
                    group_id=group_id,
                    difficult=difficult,
                    shape_type=shape_type,
                    description=description,
                    kie_linking=kie_linking,
                )
                shape_infos.append(current_shape)
                self.shape_to_image_file.append(image_file)

            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break
        progress_dialog.close()

        label_infos = {k: label_infos[k] for k in sorted(label_infos)}
        return label_infos, shape_infos

    def get_total_infos(self, start_index: int = -1, end_index: int = -1):
        """
        Get the total information for the images in the current project.
        """
        label_infos, shape_infos = self.get_label_infos(start_index, end_index)
        total_infos = [["Label"] + self.supported_shape + ["Total"]]
        shape_counter = [0 for _ in range(len(self.supported_shape) + 1)]

        for label, infos in label_infos.items():
            counter = [
                infos[shape_type] for shape_type in self.supported_shape
            ]
            counter.append(sum(counter))
            row = [label] + counter
            total_infos.append(row)
            shape_counter = [x + y for x, y in zip(counter, shape_counter)]

        total_infos.append(["Total"] + shape_counter)
        return total_infos, shape_infos

    def get_shape_infos_table(self, shape_infos):
        """
        Get the shape information table for the images in the current project.
        """
        headers = [
            "Filename",
            "Label",
            "Type",
            "Linking",
            "Group ID",
            "Difficult",
            "Description",
            "Flags",
            "Points",
        ]
        table_data = []
        for shape in shape_infos:
            row = [
                shape["filename"],
                shape["label"],
                shape["shape_type"],
                str(shape["kie_linking"]),
                str(shape["group_id"]),
                str(shape["difficult"]),
                shape["description"],
                str(shape["flags"]),
                str(shape["points"]),
            ]
            table_data.append(row)
        return headers, table_data

    def populate_table(self, start_index: int = -1, end_index: int = -1):
        """
        Populate the table with the label or shape information.
        """
        if self.showing_label_infos:
            self.current_shape_infos = []
            self.shape_to_image_file = []
            total_infos, _ = self.get_total_infos(start_index, end_index)
            rows = len(total_infos) - 1
            cols = len(total_infos[0])
            self.table.setRowCount(rows)
            self.table.setColumnCount(cols)
            self.table.setHorizontalHeaderLabels(total_infos[0])

            data = [list(map(str, info)) for info in total_infos[1:]]

            for row, info in enumerate(data):
                for col, value in enumerate(info):
                    item = QTableWidgetItem(value)
                    self.table.setItem(row, col, item)
        else:
            _, shape_infos = self.get_label_infos(start_index, end_index)
            self.current_shape_infos = shape_infos
            headers, table_data = self.get_shape_infos_table(shape_infos)
            self.table.setRowCount(len(table_data))
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels(headers)

            difficult_col_index = headers.index("Difficult")
            difficult_header_item = self.table.horizontalHeaderItem(
                difficult_col_index
            )
            if difficult_header_item:
                difficult_header_item.setBackground(
                    QBrush(self.DIFFICULT_HIGHLIGHT_COLOR)
                )

            highlight_brush = QBrush(self.DIFFICULT_HIGHLIGHT_COLOR)
            for row, data in enumerate(table_data):
                is_difficult = row < len(shape_infos) and shape_infos[row].get(
                    "difficult", False
                )
                for col, value in enumerate(data):
                    item = QTableWidgetItem(value)
                    item.setToolTip(value)
                    if is_difficult:
                        item.setBackground(highlight_brush)
                    self.table.setItem(row, col, item)
            self.table.horizontalHeader().setSectionResizeMode(
                QtWidgets.QHeaderView.Stretch
            )

    def update_range(self):
        """
        Update the range of images to display in the table.
        """
        from_value = (
            int(self.from_input.text())
            if self.from_input.text()
            else self.start_index
        )
        to_value = (
            int(self.to_input.text())
            if self.to_input.text()
            else self.end_index
        )
        if (
            (from_value > to_value)
            or (from_value < 1)
            or (to_value > len(self.image_file_list))
        ):
            self.from_input.setValue(1)
            self.to_input.setValue(len(self.image_file_list))
            self.populate_table(1, len(self.image_file_list))
        else:
            self.start_index = from_value
            self.end_index = to_value
            self.populate_table()

    def export_to_csv(self):
        """
        Export the label and shape information to a CSV file.
        """
        directory = QFileDialog.getExistingDirectory(
            self, self.tr("Select Directory"), ""
        )
        if not directory:
            return

        self.accept()

        try:
            label_infos, shape_infos = self.get_total_infos(
                1, len(self.image_file_list)
            )
            headers, shape_infos_data = self.get_shape_infos_table(shape_infos)

            label_infos_path = os.path.join(directory, "label_infos.csv")
            shape_infos_path = os.path.join(directory, "shape_infos.csv")
            classes_path = os.path.join(directory, "classes.txt")
            zip_path = os.path.join(directory, "export_data.zip")

            # Write label_infos.csv
            with open(
                label_infos_path, "w", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.writer(csvfile)
                for row in label_infos:
                    writer.writerow(row)

            # Write shape_infos.csv
            with open(
                shape_infos_path, "w", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                for row in shape_infos_data:
                    writer.writerow(row)

            # Write classes.txt
            classes = [
                row[0] for row in label_infos[1:-1]
            ]  # Exclude header and total
            with open(classes_path, "w", encoding="utf-8") as f:
                f.write("\n".join(classes))

            # Create zip file
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(label_infos_path, os.path.basename(label_infos_path))
                zf.write(shape_infos_path, os.path.basename(shape_infos_path))
                zf.write(classes_path, os.path.basename(classes_path))

            # Clean up temporary files
            os.remove(label_infos_path)
            os.remove(shape_infos_path)
            os.remove(classes_path)

            template = self.tr(
                "Exporting annotations successfully!\n"
                "Results have been saved to:\n"
                "%s"
            )
            message_text = template % zip_path
            popup = Popup(
                message_text,
                self,
                msec=5000,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")

        except Exception as e:
            logger.error(f"Error occurred while exporting file: {e}")

            popup = Popup(
                self.tr(
                    f"Error occurred while exporting annotations statistics file."
                ),
                self.parent,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self.parent)

    def toggle_info(self):
        """
        Toggle the display of label or shape information.
        """
        self.showing_label_infos = not self.showing_label_infos
        if self.showing_label_infos:
            self.toggle_button.setText(self.tr("Shape"))
        else:
            self.toggle_button.setText(self.tr("Label"))
        self.populate_table(self.start_index, self.end_index)

    def on_cell_double_clicked(self, row: int, col: int) -> None:
        """
        Handle double-click on table cell to navigate to corresponding image.

        Args:
            row (int): The row index of the clicked cell.
            col (int): The column index of the clicked cell.
        """
        if not self.showing_label_infos and self.current_shape_infos:
            if row < len(self.current_shape_infos):
                shape_info = self.current_shape_infos[row]
                filename = shape_info.get("filename")
                if filename and self.parent:
                    if not os.path.isabs(filename) and row < len(
                        self.shape_to_image_file
                    ):
                        image_file = self.shape_to_image_file[row]
                        image_dir = os.path.dirname(image_file)
                        filename = os.path.join(image_dir, filename)
                    self.parent.load_file(filename)
