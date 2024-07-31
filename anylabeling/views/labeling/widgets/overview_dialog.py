import os
import csv
import json

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class OverviewDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.supported_shape = parent.supported_shape
        self.image_file_list = self.get_image_file_list()
        self.start_index = 1
        self.end_index = len(self.image_file_list)
        self.showing_label_infos = True
        if self.image_file_list:
            self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Overview"))
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        self.resize(600, 400)
        self.move_to_center()

        layout = QVBoxLayout(self)

        # Table widget
        self.table = QTableWidget(self)

        self.populate_table()

        layout.addWidget(self.table)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )

        # Add input fields for range selection
        range_layout = QHBoxLayout()
        # Add stretch to center the widgets
        range_layout.addStretch(1)

        from_label = QLabel("From:")
        self.from_input = QSpinBox()
        self.from_input.setMinimum(1)
        self.from_input.setMaximum(len(self.image_file_list))
        self.from_input.setSingleStep(1)
        self.from_input.setValue(self.start_index)
        range_layout.addWidget(from_label)
        range_layout.addWidget(self.from_input)

        to_label = QLabel("To:")
        self.to_input = QSpinBox()
        self.to_input.setMinimum(1)
        self.to_input.setMaximum(len(self.image_file_list))
        self.to_input.setSingleStep(1)
        self.to_input.setValue(len(self.image_file_list))
        range_layout.addWidget(to_label)
        range_layout.addWidget(self.to_input)

        self.range_button = QPushButton("Go")
        range_layout.addWidget(self.range_button)
        self.range_button.clicked.connect(self.update_range)

        # Add stretch to center the widgets
        range_layout.addStretch(1)

        # Add export button for exporting data
        self.export_button = QPushButton(self.tr("Export"))

        # Add toggle button to switch between label_infos and shape_infos
        self.toggle_button = QPushButton(self.tr("Show Shape Infos"))
        self.toggle_button.clicked.connect(self.toggle_info)

        range_and_export_layout = QHBoxLayout()
        range_and_export_layout.addWidget(self.toggle_button, 0, Qt.AlignLeft)
        range_and_export_layout.addStretch(1)
        range_and_export_layout.addLayout(range_layout)
        range_and_export_layout.addStretch(1)
        range_and_export_layout.addWidget(self.export_button, 0, Qt.AlignRight)

        layout.addLayout(range_and_export_layout)

        self.export_button.clicked.connect(self.export_to_csv)

        self.exec_()

    def move_to_center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def get_image_file_list(self):
        image_file_list = []
        count = self.parent.file_list_widget.count()
        for c in range(count):
            image_file = self.parent.file_list_widget.item(c).text()
            image_file_list.append(image_file)
        return image_file_list

    def get_label_infos(self, start_index: int = -1, end_index: int = -1):
        initial_nums = [0 for _ in range(len(self.supported_shape))]
        label_infos = {}
        shape_infos = []

        progress_dialog = QProgressDialog(
            self.tr("Loading..."),
            self.tr("Cancel"),
            0,
            len(self.image_file_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet(
            """
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """
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
                    print(f"Invalid shape_type {shape_type} of {label_file}!")
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
            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break
        progress_dialog.close()
        label_infos = {k: label_infos[k] for k in sorted(label_infos)}
        return label_infos, shape_infos

    def get_total_infos(self, start_index: int = -1, end_index: int = -1):
        label_infos, _ = self.get_label_infos(start_index, end_index)
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
        return total_infos

    def get_shape_infos_table(self, shape_infos):
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
        if self.showing_label_infos:
            total_infos = self.get_total_infos(start_index, end_index)
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
            headers, table_data = self.get_shape_infos_table(shape_infos)
            self.table.setRowCount(len(table_data))
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels(headers)

            for row, data in enumerate(table_data):
                for col, value in enumerate(data):
                    item = QTableWidgetItem(value)
                    item.setToolTip(value)
                    self.table.setItem(row, col, item)
            self.table.horizontalHeader().setSectionResizeMode(
                QtWidgets.QHeaderView.Stretch
            )

    def update_range(self):
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
        directory = QFileDialog.getExistingDirectory(
            self, self.tr("Select Directory"), ""
        )
        if not directory:
            return

        try:
            # Export label_infos
            label_infos = self.get_total_infos(1, len(self.image_file_list))
            label_infos_path = os.path.join(directory, "label_infos.csv")
            with open(
                label_infos_path, "w", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.writer(csvfile)
                for row in label_infos:
                    writer.writerow(row)

            # Export shape_infos
            _, shape_infos = self.get_label_infos(1, len(self.image_file_list))
            headers, shape_infos_data = self.get_shape_infos_table(shape_infos)
            shape_infos_path = os.path.join(directory, "shape_infos.csv")
            with open(
                shape_infos_path, "w", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                for row in shape_infos_data:
                    writer.writerow(row)

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(self.tr("Exporting successfully!"))
            msg_box.setInformativeText(
                self.tr(
                    f"Results have been saved to:\n{label_infos_path}\nand\n{shape_infos_path}"
                )
            )
            msg_box.setWindowTitle(self.tr("Success"))
            msg_box.exec_()
        except Exception as e:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr(
                    "Error occurred while exporting annotations statistics file."
                )
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def toggle_info(self):
        self.showing_label_infos = not self.showing_label_infos
        if self.showing_label_infos:
            self.toggle_button.setText(self.tr("Show Shape Infos"))
        else:
            self.toggle_button.setText(self.tr("Show Label Infos"))
        self.populate_table(self.start_index, self.end_index)
