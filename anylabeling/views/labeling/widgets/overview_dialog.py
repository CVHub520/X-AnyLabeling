import os
import json

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QTableWidget,
    QProgressDialog,
    QTableWidgetItem,
)


class OverviewDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.label_file_list = parent.get_label_file_list()
        self.supported_shape = parent.supported_shape
        self.current_file = self.get_current_file()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Overview"))
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.resize(520, 350)
        self.move_to_center()

        layout = QVBoxLayout(self)
        table = QTableWidget(self)

        label_infos = self.load_label_infos()
        total_infos = self.calculate_total_infos(label_infos)
        self.populate_table(table, total_infos)

        layout.addWidget(table)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        layout.addWidget(table)
        self.exec_()

    def move_to_center(self):
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def get_current_file(self):
        try:
            dir_path, filename = os.path.split(self.parent.filename)
            filename = os.path.splitext(filename)[0] + ".json"
            current_file = os.path.join(dir_path, filename)
            if self.parent.output_dir:
                current_file = os.path.join(self.parent.output_dir, filename)
        except:
            return ""
        if not os.path.exists(current_file):
            QtWidgets.QMessageBox.warning(
                self,
                self.parent.tr("Warning"),
                self.parent.tr("No file selected.")
            )
            return ""
        return current_file

    def load_label_infos(self):
        label_infos = {}
        initial_nums = [0 for _ in range(len(self.supported_shape))]

        progress_dialog = QProgressDialog(
            self.tr("Loading..."),
            self.tr("Cancel"),
            0,
            len(self.label_file_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet("""
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """)

        for i, label_file in enumerate(self.label_file_list):
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            for shape in shapes:
                label = shape.get("label", "_empty")
                if label not in label_infos:
                    label_infos[label] = dict(
                        zip(self.supported_shape, initial_nums)
                    )
                shape_type = shape.get("shape_type", "")
                label_infos[label][shape_type] += 1
            progress_dialog.setValue(i)
            if progress_dialog.wasCanceled():
                break

        progress_dialog.close()
        return label_infos

    def calculate_total_infos(self, label_infos):
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

    def populate_table(self, table, total_infos):
        rows = len(total_infos) - 1
        cols = len(total_infos[0])
        table.setRowCount(rows)
        table.setColumnCount(cols)
        table.setHorizontalHeaderLabels(total_infos[0])

        data = [list(map(str, info)) for info in total_infos[1:]]

        for row, info in enumerate(data):
            for col, value in enumerate(info):
                item = QTableWidgetItem(value)
                table.setItem(row, col, item)
