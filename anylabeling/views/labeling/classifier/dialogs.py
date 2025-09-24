import os
from collections import Counter

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.classifier.style import (
    get_dialog_button_style,
    get_filename_label_style,
)
from anylabeling.views.labeling.classifier.utils import (
    get_label_file_path,
    load_flags_from_json,
)


class ExportPathDialog(QDialog):
    def __init__(self, default_path="classified", parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export Images"))
        self.setModal(True)
        self.setFixedSize(480, 180)
        self.default_path = default_path
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        label = QLabel(self.tr("Export Directory:"))
        label.setStyleSheet(get_filename_label_style())
        layout.addWidget(label)

        path_layout = QHBoxLayout()
        path_layout.setSpacing(8)

        if self.parent() and self.parent().image_files:
            base_dir = os.path.dirname(
                os.path.dirname(self.parent().image_files[0])
            )
            full_path = os.path.join(base_dir, self.default_path)
        else:
            full_path = self.default_path

        self.path_edit = QLineEdit(full_path)
        self.path_edit.setStyleSheet(self._get_input_style())
        self.path_edit.setToolTip(full_path)
        path_layout.addWidget(self.path_edit, 1)

        browse_button = QPushButton(self.tr("Browse"))
        browse_button.setStyleSheet(
            get_dialog_button_style("outline", "medium")
        )
        browse_button.clicked.connect(self.browse_path)
        path_layout.addWidget(browse_button)

        layout.addLayout(path_layout)
        layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.ok_button = QPushButton(self.tr("Export"))
        self.ok_button.setStyleSheet(
            get_dialog_button_style("primary", "medium")
        )
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

    def _get_input_style(self):
        return """
            QLineEdit {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 0 16px;
                background-color: #f8fafc;
                color: #374151;
                font-size: 13px;
                font-weight: 500;
                height: 32px;
            }
            QLineEdit:hover {
                background-color: #f1f5f9;
                border-color: #cbd5e1;
            }
            QLineEdit:focus {
                border: 2px solid #0077ed;
                background-color: #ffffff;
                outline: none;
            }
        """

    def browse_path(self):
        path = QFileDialog.getExistingDirectory(
            self, self.tr("Select Export Directory")
        )
        if path:
            self.path_edit.setText(path)
            self.path_edit.setToolTip(path)

    def get_path(self):
        return self.path_edit.text().strip()


class NewLabelDialog(QDialog):
    def __init__(self, existing_labels, parent=None):
        super().__init__(parent)
        self.existing_labels = existing_labels
        self.setWindowTitle(self.tr("Add Labels"))
        self.setModal(True)
        self.setFixedSize(500, 350)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        if self.existing_labels:
            existing_label = QLabel(self.tr("Existing Labels:"))
            existing_label.setStyleSheet(get_filename_label_style())
            layout.addWidget(existing_label)

            self.existing_text = QTextEdit()
            self.existing_text.setPlainText("\n".join(self.existing_labels))
            self.existing_text.setReadOnly(True)
            self.existing_text.setMaximumHeight(120)
            self.existing_text.setStyleSheet(
                """
                QTextEdit {
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                    padding: 8px;
                    background-color: #f8fafc;
                    color: #6b7280;
                    font-size: 13px;
                }
            """
            )
            layout.addWidget(self.existing_text)

        new_label = QLabel(self.tr("Enter new labels (one per line):"))
        new_label.setStyleSheet(get_filename_label_style())
        layout.addWidget(new_label)

        self.text_edit = QTextEdit()
        self.text_edit.setStyleSheet(
            """
            QTextEdit {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 8px;
                background-color: #ffffff;
                color: #374151;
                font-size: 13px;
            }
            QTextEdit:focus {
                border: 2px solid #0077ed;
                outline: none;
            }
        """
        )
        layout.addWidget(self.text_edit)

        button_layout = QHBoxLayout()

        upload_button = QPushButton(self.tr("Upload"))
        upload_button.setStyleSheet(
            get_dialog_button_style("primary", "medium")
        )
        upload_button.clicked.connect(self.upload_labels_file)
        button_layout.addWidget(upload_button)
        button_layout.addStretch()

        ok_button = QPushButton(self.tr("Add"))
        ok_button.setStyleSheet(get_dialog_button_style("success", "medium"))
        ok_button.clicked.connect(self.accept_with_validation)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton(self.tr("Cancel"))
        cancel_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def upload_labels_file(self):
        """Upload and parse labels from txt file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Labels File"),
            "",
            "Text Files (*.txt);;All Files (*)",
        )

        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()

            if content:
                labels = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip()
                ]
                if labels:
                    self.text_edit.setPlainText("\n".join(labels))
                    QMessageBox.information(
                        self,
                        self.tr("Success"),
                        self.tr("Loaded %d labels from file.") % len(labels),
                    )
                else:
                    QMessageBox.warning(
                        self,
                        self.tr("Warning"),
                        self.tr("No valid labels found in the file."),
                    )
            else:
                QMessageBox.warning(
                    self,
                    self.tr("Warning"),
                    self.tr("The selected file is empty."),
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Error"),
                self.tr("Failed to read file: %s") % str(e),
            )

    def accept_with_validation(self):
        new_labels_text = self.text_edit.toPlainText().strip()
        if not new_labels_text:
            QMessageBox.warning(
                self,
                self.tr("Invalid Input"),
                self.tr("Please enter at least one label!"),
            )
            return

        new_labels = [
            label.strip()
            for label in new_labels_text.split("\n")
            if label.strip()
        ]
        if not new_labels:
            QMessageBox.warning(
                self,
                self.tr("Invalid Input"),
                self.tr("Please enter valid label names!"),
            )
            return

        duplicates = [
            label for label in new_labels if label in self.existing_labels
        ]
        if duplicates:
            QMessageBox.warning(
                self,
                self.tr("Duplicate Labels"),
                self.tr("These labels already exist: {}").format(
                    ", ".join(duplicates)
                ),
            )
            return

        internal_duplicates = [
            label for label in new_labels if new_labels.count(label) > 1
        ]
        if internal_duplicates:
            QMessageBox.warning(
                self,
                self.tr("Duplicate Labels"),
                self.tr("Duplicate labels found in input: {}").format(
                    ", ".join(set(internal_duplicates))
                ),
            )
            return

        if self.existing_labels:
            reply = QMessageBox.question(
                self,
                self.tr("Confirm Add Labels"),
                self.tr(
                    "This will:\n"
                    "1. Skip files that don't exist\n"
                    "2. Reset files with mismatched labels to current template\n"
                    "3. Add new labels to matching files\n\n"
                    "Continue?"
                ),
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.accept()
        else:
            self.accept()

    def get_labels(self):
        new_labels_text = self.text_edit.toPlainText().strip()
        return [
            label.strip()
            for label in new_labels_text.split("\n")
            if label.strip()
        ]


class DeleteLabelDialog(QDialog):
    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.labels = labels
        self.setWindowTitle(self.tr("Delete Labels"))
        self.setModal(True)
        self.setFixedSize(500, 350)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        label = QLabel(self.tr("Select labels to delete:"))
        label.setStyleSheet(get_filename_label_style())
        layout.addWidget(label)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        for label_name in self.labels:
            self.list_widget.addItem(label_name)
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        delete_button = QPushButton(self.tr("Delete"))
        delete_button.setStyleSheet(
            get_dialog_button_style("danger", "medium")
        )
        delete_button.clicked.connect(self.delete_with_confirmation)
        button_layout.addWidget(delete_button)

        cancel_button = QPushButton(self.tr("Cancel"))
        cancel_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def delete_with_confirmation(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return

        selected_labels = [item.text() for item in selected_items]
        template = self.tr(
            "This will remove %s and all related data. Continue?"
        )
        reply = QMessageBox.question(
            self,
            self.tr("Confirm Delete"),
            template % ", ".join(selected_labels),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.accept()

    def get_selected_labels(self):
        return [item.text() for item in self.list_widget.selectedItems()]


class EditLabelDialog(QDialog):
    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.labels = labels
        self.setWindowTitle(self.tr("Edit Labels"))
        self.setModal(True)
        self.setFixedSize(500, 350)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        label = QLabel(self.tr("Edit labels:"))
        label.setStyleSheet(get_filename_label_style())
        layout.addWidget(label)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(
            [self.tr("Current"), self.tr("New")]
        )
        self.table_widget.setRowCount(len(self.labels))
        self.table_widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        for i, label_text in enumerate(self.labels):
            current_label = QTableWidgetItem(label_text)
            current_label.setFlags(current_label.flags() & ~Qt.ItemIsEditable)
            self.table_widget.setItem(i, 0, current_label)
            new_label = QTableWidgetItem(label_text)
            self.table_widget.setItem(i, 1, new_label)

        layout.addWidget(self.table_widget)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        save_button = QPushButton(self.tr("Save"))
        save_button.setStyleSheet(get_dialog_button_style("success", "medium"))
        save_button.clicked.connect(self.accept_with_validation)
        button_layout.addWidget(save_button)

        cancel_button = QPushButton(self.tr("Cancel"))
        cancel_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def accept_with_validation(self):
        new_labels = {}
        seen_labels = set()

        for i in range(self.table_widget.rowCount()):
            old_label = self.table_widget.item(i, 0).text()
            new_label = self.table_widget.item(i, 1).text().strip()

            if not new_label:
                QMessageBox.warning(
                    self,
                    self.tr("Invalid Input"),
                    self.tr("Label name cannot be empty!"),
                )
                return

            if new_label in seen_labels:
                QMessageBox.warning(
                    self,
                    self.tr("Duplicate Labels"),
                    self.tr("Label %s is used multiple times!") % new_label,
                )
                return

            if new_label != old_label:
                new_labels[i] = new_label

            seen_labels.add(new_label)

        if not new_labels:
            QMessageBox.information(
                self,
                self.tr("No Changes"),
                self.tr("No labels were modified."),
            )
            return

        reply = QMessageBox.question(
            self,
            self.tr("Confirm Changes"),
            self.tr("Save changes to %d labels?") % len(new_labels),
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.accept()

    def get_edit_info(self):
        return {
            self.table_widget.item(i, 0)
            .text(): self.table_widget.item(i, 1)
            .text()
            .strip()
            for i in range(self.table_widget.rowCount())
        }


class StatisticsViewDialog(QDialog):
    def __init__(self, labels, image_files, output_dir, parent=None):
        super().__init__(parent)
        self.labels = labels
        self.image_files = image_files
        self.output_dir = output_dir
        self.color_palette = [
            "#3b82f6",
            "#10b981",
            "#f59e0b",
            "#ef4444",
            "#8b5cf6",
            "#06b6d4",
            "#84cc16",
            "#f97316",
        ]
        self.setWindowTitle(self.tr("Dataset Statistics"))
        self.setModal(True)
        self.setFixedSize(520, 480)
        self.init_ui()
        self.load_statistics()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 20, 24, 20)

        self.kpi_layout = QHBoxLayout()
        self.kpi_layout.setSpacing(12)
        layout.addLayout(self.kpi_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet(
            """
            QFrame {
                color: #e5e7eb;
                background-color: #e5e7eb;
                border: none;
                height: 1px;
                margin: 8px 0;
            }
        """
        )
        layout.addWidget(separator)

        self.distribution_widget = QWidget()
        layout.addWidget(self.distribution_widget)
        layout.addStretch()

    def load_statistics(self):
        labeled_count = 0
        label_counts = Counter()

        for image_file in self.image_files:
            label_path = get_label_file_path(image_file, self.output_dir)
            flags = load_flags_from_json(label_path)

            if flags and any(flags.values()):
                labeled_count += 1
                for label, is_selected in flags.items():
                    if is_selected:
                        label_counts[label] += 1

        for label in self.labels:
            if label not in label_counts:
                label_counts[label] = 0

        unlabeled_count = len(self.image_files) - labeled_count
        total_count = len(self.image_files)

        self.create_kpi_chips(total_count, labeled_count, unlabeled_count)
        self.create_distribution_chart(label_counts, labeled_count)
        self.validate_statistics(total_count, labeled_count, label_counts)

    def create_kpi_chips(self, total_count, labeled_count, unlabeled_count):
        kpi_data = [
            ("Total", total_count, "#6b7280"),
            ("Labeled", labeled_count, "#10b981"),
            ("Unlabeled", unlabeled_count, "#f59e0b"),
        ]

        for title, value, color in kpi_data:
            chip = self.create_kpi_chip(title, value, color)
            self.kpi_layout.addWidget(chip)

    def create_kpi_chip(self, title, value, color):
        widget = QWidget()
        widget.setStyleSheet(
            """
            QWidget {
                background-color: #f9fafb;
                border-radius: 8px;
                padding: 12px 16px;
            }
        """
        )

        layout = QVBoxLayout(widget)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        value_label = QLabel(str(value))
        value_label.setStyleSheet(
            f"""
            QLabel {{
                font-size: 12px;
                font-weight: 700;
                color: {color};
                margin: 0;
            }}
        """
        )
        value_label.setAlignment(Qt.AlignCenter)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            """
            QLabel {
                font-size: 11px;
                color: #6b7280;
                font-weight: 500;
                margin: 0;
            }
        """
        )
        title_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(value_label)
        layout.addWidget(title_label)

        return widget

    def get_category_color(self, category_name, index):
        return self.color_palette[index % len(self.color_palette)]

    def create_distribution_chart(self, label_counts, total_labeled):
        chart_layout = QVBoxLayout(self.distribution_widget)
        chart_layout.setSpacing(8)
        chart_layout.setContentsMargins(0, 0, 0, 0)

        if not label_counts:
            no_data_label = QLabel(self.tr("No labeled data available"))
            no_data_label.setStyleSheet(
                """
                QLabel {
                    color: #9ca3af;
                    font-size: 13px;
                    font-style: italic;
                    padding: 20px;
                }
            """
            )
            no_data_label.setAlignment(Qt.AlignCenter)
            chart_layout.addWidget(no_data_label)
            return

        max_count = max(label_counts.values()) if label_counts else 1
        max_label_width = min(
            120, max(len(label) * 8 for label in label_counts.keys()) + 20
        )

        sorted_labels = sorted(
            label_counts.items(), key=lambda x: x[1], reverse=True
        )
        for i, (label, count) in enumerate(sorted_labels):
            percentage = (
                (count / total_labeled * 100) if total_labeled > 0 else 0
            )
            bar_widget = self.create_distribution_bar(
                label, count, percentage, max_count, i, max_label_width
            )
            chart_layout.addWidget(bar_widget)

    def create_distribution_bar(
        self, label_name, count, percentage, max_count, index, label_width
    ):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        display_name = (
            label_name if len(label_name) <= 15 else label_name[:12] + "..."
        )
        label_widget = QLabel(display_name)
        label_widget.setFixedWidth(label_width)
        if len(label_name) > 15:
            label_widget.setToolTip(label_name)
        label_widget.setStyleSheet(
            """
            QLabel {
                font-size: 12px;
                color: #374151;
                font-weight: 500;
            }
        """
        )
        layout.addWidget(label_widget)

        progress_bar = QProgressBar()
        progress_bar.setMaximum(max_count)
        progress_bar.setValue(count)
        progress_bar.setFixedHeight(20)
        progress_bar.setFormat(f"{count} ({percentage:.0f}%)")
        progress_bar.setTextVisible(True)

        color = self.get_category_color(label_name, index)
        fill_ratio = count / max_count if max_count > 0 else 0

        if fill_ratio >= 0.6:
            text_color = "#ffffff"
        elif fill_ratio >= 0.3:
            text_color = "#1f2937"
        else:
            text_color = "#374151"

        progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: none;
                border-radius: 10px;
                background-color: #f1f5f9;
                text-align: center;
                font-size: 10px;
                font-weight: 600;
                color: {text_color};
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 10px;
            }}
        """
        )

        layout.addWidget(progress_bar, 1)
        return widget

    def validate_statistics(self, total_count, labeled_count, label_counts):
        label_sum = sum(label_counts.values())
        if label_sum != labeled_count:
            print(
                f"Warning: Label count mismatch. Labeled: {labeled_count}, Sum: {label_sum}"
            )

        if total_count == 0:
            print("Warning: No images found in dataset")
