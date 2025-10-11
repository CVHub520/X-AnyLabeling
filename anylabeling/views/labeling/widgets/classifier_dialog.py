import os
import re

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import (
    QIcon,
    QIntValidator,
    QKeySequence,
    QPixmap,
)

from anylabeling.views.labeling.classifier import *
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.qt import new_icon, new_icon_path
from anylabeling.views.labeling.utils.style import get_progress_dialog_style
from anylabeling.views.labeling.widgets.popup import Popup
from anylabeling.views.labeling.vqa.dialogs import (
    AILoadingDialog,
    AIPromptDialog,
)
from anylabeling.views.labeling.vqa.style import get_page_input_style
from anylabeling.views.labeling.vqa.utils import AIWorkerThread


class ClassifierDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr(DEFAULT_WINDOW_TITLE))
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.resize(*DEFAULT_WINDOW_SIZE)

        self.image_files = []
        self.current_image_index = 0
        self.labels = []
        self.is_multiclass = True
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.switching_image = False
        self.top_window = None

        if parent:
            self.top_window = parent
            current = self.top_window
            while True:
                parent_widget = (
                    current.parent()
                    if hasattr(current, "parent") and callable(current.parent)
                    else getattr(current, "parent", None)
                )
                if parent_widget is None:
                    break
                current = parent_widget
            self.top_window = current

        self.init_ui()
        self.load_initial_data()
        self.setup_shortcuts()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(20)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(3)
        self.main_splitter.setStyleSheet(get_main_splitter_style())
        self.main_splitter.setChildrenCollapsible(False)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.setSpacing(10)

        self.filename_label = QLabel(self.tr("No image loaded"))
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setFixedHeight(DEFAULT_COMPONENT_HEIGHT)
        self.filename_label.setStyleSheet(get_filename_label_style())
        left_layout.addWidget(self.filename_label)

        self.image_container = QWidget()
        self.image_container.setStyleSheet(get_image_container_style())
        container_layout = QVBoxLayout(self.image_container)
        container_layout.setContentsMargins(8, 8, 8, 8)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(get_image_label_style())
        self.image_label.setFixedSize(PANEL_SIZE - 16, PANEL_SIZE - 16)
        self.image_label.setScaledContents(False)
        container_layout.addWidget(self.image_label, 1)

        self.overlay_label = ClassificationOverlay(self.image_label)

        left_layout.addWidget(self.image_container, 1)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        action_widget = QWidget()
        action_widget.setFixedHeight(DEFAULT_COMPONENT_HEIGHT)
        action_layout = QHBoxLayout(action_widget)
        action_layout.setContentsMargins(10, 0, 10, 0)
        action_layout.setSpacing(8)

        self.export_button = QPushButton(self.tr("Export"))
        self.export_button.setStyleSheet(
            get_dialog_button_style("primary", "medium")
        )
        self.export_button.setToolTip(
            self.tr("Export classified images to folders by category")
        )
        self.export_button.clicked.connect(self.export_images)

        self.multiclass_button = QPushButton(self.tr("MultiClass"))
        self.multiclass_button.setStyleSheet(
            get_dialog_button_style("light_green", "medium")
        )
        self.multiclass_button.setToolTip(
            self.tr("Single-label classification mode")
        )
        self.multiclass_button.clicked.connect(self.switch_to_multiclass)

        self.multilabel_button = QPushButton(self.tr("MultiLabel"))
        self.multilabel_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        self.multilabel_button.setToolTip(
            self.tr("Multi-label classification mode")
        )
        self.multilabel_button.clicked.connect(self.switch_to_multilabel)

        self.auto_run_button = QPushButton(self.tr("AutoRun"))
        self.auto_run_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        self.auto_run_button.setToolTip(
            self.tr("Use AI to automatically classify all images in batch")
        )
        self.auto_run_button.clicked.connect(self.auto_run_batch)

        action_layout.addWidget(self.export_button, 1)
        action_layout.addWidget(self.multiclass_button, 1)
        action_layout.addWidget(self.multilabel_button, 1)
        action_layout.addWidget(self.auto_run_button, 1)

        right_layout.addWidget(action_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameStyle(QFrame.NoFrame)

        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll_layout.setSpacing(15)

        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel(self.tr("Category"))
        title_label.setStyleSheet(get_filename_label_style())
        title_label.setToolTip(
            self.tr("Use number keys (0-9) to quickly select categories")
        )
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        self.ai_button = QPushButton()
        self.ai_button.setIcon(QIcon(new_icon("wand", "svg")))
        self.ai_button.setFixedSize(*ICON_SIZE_SMALL)
        self.ai_button.setStyleSheet(get_button_style())
        self.ai_button.setToolTip(self.tr("AI Assistant"))
        self.ai_button.clicked.connect(self.ai_classify_current)
        title_layout.addWidget(self.ai_button)

        self.new_button = QPushButton()
        self.new_button.setIcon(QIcon(new_icon("new", "svg")))
        self.new_button.setFixedSize(*ICON_SIZE_SMALL)
        self.new_button.setStyleSheet(get_button_style())
        self.new_button.setToolTip(self.tr("Add Label"))
        self.new_button.clicked.connect(self.add_new_label)
        title_layout.addWidget(self.new_button)

        self.delete_button = QPushButton()
        self.delete_button.setIcon(QIcon(new_icon("minus", "svg")))
        self.delete_button.setFixedSize(*ICON_SIZE_SMALL)
        self.delete_button.setStyleSheet(get_button_style())
        self.delete_button.setToolTip(self.tr("Delete Label"))
        self.delete_button.clicked.connect(self.delete_label)
        title_layout.addWidget(self.delete_button)

        self.edit_button = QPushButton()
        self.edit_button.setIcon(QIcon(new_icon("edit", "svg")))
        self.edit_button.setFixedSize(*ICON_SIZE_SMALL)
        self.edit_button.setStyleSheet(get_button_style())
        self.edit_button.setToolTip(self.tr("Edit Label"))
        self.edit_button.clicked.connect(self.edit_label)
        title_layout.addWidget(self.edit_button)

        self.view_button = QPushButton()
        self.view_button.setIcon(QIcon(new_icon("view", "svg")))
        self.view_button.setFixedSize(*ICON_SIZE_SMALL)
        self.view_button.setStyleSheet(get_button_style())
        self.view_button.setToolTip(self.tr("View Statistics"))
        self.view_button.clicked.connect(self.view_statistics)
        title_layout.addWidget(self.view_button)

        self.scroll_layout.addWidget(title_widget)

        self.checkbox_group = None
        self.scroll_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_widget)
        right_layout.addWidget(self.scroll_area, 1)

        nav_widget = QWidget()
        nav_widget.setFixedHeight(DEFAULT_COMPONENT_HEIGHT)
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(10, 0, 10, 0)
        nav_layout.setSpacing(8)

        self.prev_button = QPushButton()
        self.prev_button.setIcon(QIcon(new_icon("arrow-left", "svg")))
        self.prev_button.setFixedSize(*ICON_SIZE_NORMAL)
        self.prev_button.setStyleSheet(get_button_style())
        self.prev_button.setToolTip(
            self.tr("Previous image (A) | Previous unlabeled image (Ctrl+A)")
        )
        self.prev_button.clicked.connect(self.prev_image)

        page_widget = QWidget()
        page_widget.setFixedWidth(82)
        page_layout = QHBoxLayout(page_widget)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(2)

        self.page_input = PageInputLineEdit()
        self.page_input.classifier_dialog = self
        self.page_input.setFixedSize(68, DEFAULT_COMPONENT_HEIGHT)
        self.page_input.setAlignment(Qt.AlignCenter)
        self.page_input.setStyleSheet(get_page_input_style())
        self.page_input.textChanged.connect(self.validate_page_input)
        self.page_input.editingFinished.connect(self.on_page_input_finished)

        page_layout.addWidget(self.page_input)

        self.next_button = QPushButton()
        self.next_button.setIcon(QIcon(new_icon("arrow-right", "svg")))
        self.next_button.setFixedSize(*ICON_SIZE_NORMAL)
        self.next_button.setStyleSheet(get_button_style())
        self.next_button.setToolTip(
            self.tr("Next image (D) | Next unlabeled image (Ctrl+D)")
        )
        self.next_button.clicked.connect(self.next_image)

        nav_layout.addWidget(self.prev_button)
        nav_layout.addStretch()
        nav_layout.addWidget(page_widget)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_button)

        right_layout.addWidget(nav_widget)

        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(right_widget)

        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.target_left_width = PANEL_SIZE

        main_layout.addWidget(self.main_splitter)

        self.update_navigation_state()

    def setup_shortcuts(self):
        prev_shortcut = QShortcut(QKeySequence("A"), self)
        prev_shortcut.activated.connect(self.prev_image)

        next_shortcut = QShortcut(QKeySequence("D"), self)
        next_shortcut.activated.connect(self.next_image)

        prev_unlabeled_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)
        prev_unlabeled_shortcut.activated.connect(self.prev_unlabeled_image)

        next_unlabeled_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        next_unlabeled_shortcut.activated.connect(self.next_unlabeled_image)

        for i in range(10):
            shortcut = QShortcut(QKeySequence(str(i)), self)
            shortcut.activated.connect(
                lambda idx=i: self.select_by_number(idx)
            )

    def select_by_number(self, number):
        if self.checkbox_group and number < len(self.labels):
            label = self.labels[number]
            if label in self.checkbox_group.checkboxes:
                checkbox = self.checkbox_group.checkboxes[label]
                if self.is_multiclass:
                    if checkbox.isChecked():
                        checkbox.setChecked(False)
                    else:
                        checkbox.setChecked(True)
                else:
                    checkbox.setChecked(not checkbox.isChecked())
                self.save_current_flags()

    def load_initial_data(self):
        if self.parent().image_list:
            self.image_files = self.parent().image_list
            self.update_image_display()
            self.update_navigation_state()
            self.page_input.setValidator(
                QIntValidator(1, len(self.image_files))
            )

        # Reset flags from current data
        self.labels = []
        if self.parent().filename:
            label_path = get_label_file_path(
                self.parent().filename,
                getattr(self.parent(), "output_dir", None),
            )
            flags = load_flags_from_json(label_path)
            if flags:
                self.labels = list(flags.keys())

        if self.labels:
            self.parent().image_flags = self.labels[:]
            self.create_checkbox_group()

        self.load_current_flags()

    def create_checkbox_group(self):
        if self.checkbox_group:
            try:
                self.scroll_layout.removeWidget(self.checkbox_group)
                self.checkbox_group.deleteLater()
            except RuntimeError:
                pass
            self.checkbox_group = None

        if not self.labels:
            return

        self.checkbox_group = ClassificationCheckBoxGroup(
            self.labels, self.is_multiclass, self
        )

        for checkbox in self.checkbox_group.checkboxes.values():
            checkbox.toggled.connect(self.on_checkbox_changed)

        stretch_item = None
        if self.scroll_layout.count() > 0:
            stretch_item = self.scroll_layout.itemAt(
                self.scroll_layout.count() - 1
            )
            if stretch_item.spacerItem():
                self.scroll_layout.removeItem(stretch_item)

        self.scroll_layout.addWidget(self.checkbox_group)

        if stretch_item:
            self.scroll_layout.addStretch()

    def reconnect_save_signals(self):
        if self.checkbox_group:
            for checkbox in self.checkbox_group.checkboxes.values():
                checkbox.toggled.connect(self.on_checkbox_changed)

    def on_checkbox_changed(self):
        if not self.switching_image and self.checkbox_group:
            if self.is_multiclass:
                sender = self.sender()
                if sender and sender.isChecked():
                    for (
                        label,
                        checkbox,
                    ) in self.checkbox_group.checkboxes.items():
                        if checkbox != sender and checkbox.isChecked():
                            checkbox.blockSignals(True)
                            checkbox.setChecked(False)
                            checkbox.blockSignals(False)
            self.save_current_flags()
        self.update_filename_label()

    def update_filename_label(self):
        if self.parent().filename:
            filename = os.path.basename(self.parent().filename)
            status_icon = (
                "✅" if self.is_image_labeled(self.parent().filename) else "❌"
            )

            if self.image_files and self.parent().filename in self.image_files:
                current_index = self.image_files.index(self.parent().filename)
                total_count = len(self.image_files)
                self.filename_label.setText(
                    f"{filename} ({current_index + 1}/{total_count}) {status_icon}"
                )
            else:
                self.filename_label.setText(f"{filename} {status_icon}")

    def add_new_label(self):
        dialog = NewLabelDialog(self.labels, self)
        if dialog.exec_() == QDialog.Accepted:
            new_labels = dialog.get_labels()
            self.labels.extend(new_labels)
            self.parent().image_flags.extend(new_labels)

            if hasattr(self.parent(), "flag_widget"):
                self.ensure_labels_in_flag_widget(new_labels)

            self.create_checkbox_group()
            if self.parent().filename:
                self.load_current_flags()
            self.update_label_files()
            self.update_filename_label()

    def delete_label(self):
        if not self.labels:
            return

        dialog = DeleteLabelDialog(self.labels, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_labels = dialog.get_selected_labels()
            for label in selected_labels:
                self.labels.remove(label)
                self.parent().image_flags.remove(label)
            self.update_label_files()
            self.update_filename_label()

    def edit_label(self):
        if not self.labels:
            return

        dialog = EditLabelDialog(self.labels, self)
        if dialog.exec_() == QDialog.Accepted:
            labels_map = dialog.get_edit_info()

            new_labels = []
            for old_label in self.labels:
                new_labels.append(labels_map[old_label])

            self.labels = new_labels
            self.parent().image_flags = new_labels[:]

            reversed_labels_map = {v: k for k, v in labels_map.items()}
            self.update_label_files(reversed_labels_map)
            if hasattr(self.parent(), "flag_widget"):
                self.update_main_flag_widget_labels(labels_map)
            self.create_checkbox_group()
            if self.parent().filename:
                self.load_current_flags()
            self.update_filename_label()

    def view_statistics(self):
        if not self.labels:
            QMessageBox.information(
                self, self.tr("Info"), self.tr("Please set labels first.")
            )
            return

        if not self.image_files:
            QMessageBox.information(
                self,
                self.tr("Info"),
                self.tr("No images loaded for statistics."),
            )
            return

        dialog = StatisticsViewDialog(
            self.labels, self.image_files, self.parent().output_dir, self
        )
        dialog.exec_()

    def switch_to_multiclass(self):
        if self.is_multiclass:
            return

        reply = QMessageBox.question(
            self,
            self.tr("Switch Mode"),
            self.tr(
                "Switching to MultiClass mode will only keep the first label for each image. "
                "Other labels will be discarded. Continue?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.is_multiclass = True
            self.multiclass_button.setStyleSheet(
                get_dialog_button_style("light_green", "medium")
            )
            self.multilabel_button.setStyleSheet(
                get_dialog_button_style("secondary", "medium")
            )
            self.create_checkbox_group()
            self.load_current_flags()

    def switch_to_multilabel(self):
        if not self.is_multiclass:
            return

        self.is_multiclass = False
        self.multilabel_button.setStyleSheet(
            get_dialog_button_style("light_green", "medium")
        )
        self.multiclass_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        self.create_checkbox_group()
        self.load_current_flags()

    def export_images(self):
        if not self.labels:
            QMessageBox.warning(
                self, self.tr("Warning"), self.tr("No labels configured!")
            )
            return

        if not self.is_multiclass:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Export only supports multi-class tasks!"),
            )
            return

        dialog = ExportPathDialog(self.output_dir, self)
        if dialog.exec_() != QDialog.Accepted:
            return

        self.output_dir = dialog.get_path()

        if not self.image_files:
            QMessageBox.warning(
                self, self.tr("Warning"), self.tr("No images loaded!")
            )
            return

        exported_count = 0
        base_dir = (
            os.path.dirname(self.image_files[0]) if self.image_files else ""
        )
        output_path = os.path.join(base_dir, self.output_dir)

        for image_path in self.image_files:
            label_path = get_label_file_path(
                image_path, getattr(self.parent(), "output_dir", None)
            )
            flags = load_flags_from_json(label_path)

            if flags:
                first_true = get_first_true_flag(flags)
                if first_true:
                    export_image_to_category(
                        image_path, first_true, output_path
                    )
                    exported_count += 1

        template = self.tr("Exported %d images to %s")
        message_text = template % (exported_count, output_path)
        QMessageBox.information(self, self.tr("Success"), message_text)

    def ai_classify_current(self):
        if not self.labels:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please configure labels first!"),
            )
            return

        if not self.parent().filename:
            QMessageBox.warning(
                self, self.tr("Warning"), self.tr("No image loaded!")
            )
            return

        prompt = create_ai_prompt_template(self.labels, self.is_multiclass)
        dialog = AIPromptDialog(self)
        dialog.prompt_input.setPlainText(prompt)
        if dialog.exec_() == QDialog.Accepted:
            final_prompt = dialog.get_prompt()
            if final_prompt:
                self.loading_msg = AILoadingDialog(self)

                self.ai_worker = AIWorkerThread(
                    final_prompt,
                    "",
                    {},
                    self.parent().filename,
                    [],
                    self.parent(),
                )
                self.ai_worker.finished.connect(self.handle_ai_result)
                self.loading_msg.cancel_button.clicked.connect(
                    self.cancel_ai_processing
                )

                self.ai_worker.start()
                if self.loading_msg.exec_() == QDialog.Rejected:
                    self.cancel_ai_processing()

    def handle_ai_result(self, result, success, error_message):
        if hasattr(self, "loading_msg"):
            self.loading_msg.close()

        if success and result:
            try:
                json_match = re.search(
                    r"```json\s*(.*?)\s*```", result, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = result

                flags = json.loads(json_str)
                if self.checkbox_group:
                    self.checkbox_group.set_flags(flags)
                    self.save_current_flags()
                    QMessageBox.information(
                        self,
                        self.tr("Success"),
                        self.tr("AI classification completed!"),
                    )
            except (json.JSONDecodeError, Exception):
                QMessageBox.warning(
                    self,
                    self.tr("Error"),
                    self.tr("Failed to parse AI result"),
                )
        else:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                error_message or self.tr("AI classification failed"),
            )

    def cancel_ai_processing(self):
        if hasattr(self, "ai_worker") and self.ai_worker.isRunning():
            self.ai_worker.terminate()
            self.ai_worker.wait(1000)
        if hasattr(self, "loading_msg"):
            self.loading_msg.close()

    def auto_run_batch(self):
        if not self.labels:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please configure labels first!"),
            )
            return

        if not self.image_files:
            QMessageBox.warning(
                self, self.tr("Warning"), self.tr("No images loaded!")
            )
            return

        prompt = create_ai_prompt_template(self.labels, self.is_multiclass)
        dialog = AIPromptDialog(self)
        dialog.prompt_input.setPlainText(prompt)
        if dialog.exec_() != QDialog.Accepted:
            return

        final_prompt = dialog.get_prompt()
        if not final_prompt:
            return

        reply = QMessageBox.question(
            self,
            self.tr("Confirmation"),
            self.tr(f"Process {len(self.image_files)} images with AI?"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        self.batch_progress = QProgressDialog(
            self.tr("Processing images..."),
            self.tr("Cancel"),
            0,
            len(self.image_files),
            self,
        )
        self.batch_progress.setWindowModality(Qt.WindowModal)
        self.batch_progress.setWindowTitle(self.tr("Progress"))
        self.batch_progress.setMinimumWidth(500)
        self.batch_progress.setMinimumHeight(150)
        self.batch_progress.setStyleSheet(
            get_progress_dialog_style(color="#1d1d1f", height=20)
        )
        self.batch_progress.show()
        self.batch_index = 0
        self.batch_prompt = final_prompt
        self.process_next_image()

    def process_next_image(self):
        if self.batch_index >= len(self.image_files):
            self.batch_progress.close()
            QMessageBox.information(
                self,
                self.tr("Success"),
                self.tr("Batch processing completed!"),
            )
            self.load_current_flags()
            return

        if self.batch_progress.wasCanceled():
            return

        image_path = self.image_files[self.batch_index]
        logger.debug(f"Processing {image_path}")
        self.batch_progress.setLabelText(
            f"Processing: {os.path.basename(image_path)}"
        )
        self.batch_progress.setValue(self.batch_index)
        self.ai_worker = AIWorkerThread(
            self.batch_prompt,
            "",
            {},
            image_path,
            [],
            self.parent(),
        )
        self.ai_worker.finished.connect(self.handle_batch_result)
        self.ai_worker.start()

    def handle_batch_result(self, result, success, error_message):
        image_path = self.image_files[self.batch_index]

        if success and result:
            try:
                json_match = re.search(
                    r"```json\s*(.*?)\s*```", result, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = result

                flags = json.loads(json_str)

                current_image = self.parent().filename
                self.parent().load_file(image_path)

                if hasattr(self.parent(), "flags"):
                    self.parent().flags = flags

                if hasattr(self.parent(), "flag_widget"):
                    for i in range(self.parent().flag_widget.count()):
                        item = self.parent().flag_widget.item(i)
                        key = item.text()
                        if key in flags:
                            item.setCheckState(
                                Qt.Checked if flags[key] else Qt.Unchecked
                            )

                if hasattr(self.parent(), "set_dirty"):
                    self.parent().set_dirty()

                if current_image:
                    self.parent().load_file(current_image)

            except Exception as e:
                logger.error(
                    f"Error processing batch result for {image_path}: {e}"
                )

        self.batch_index += 1
        QTimer.singleShot(100, self.process_next_image)

    def prev_image(self):
        if not self.image_files:
            return

        current_file = self.parent().filename
        if not current_file or current_file not in self.image_files:
            return

        current_index = self.image_files.index(current_file)
        if current_index > 0:
            self.switch_to_image(current_index - 1)

    def next_image(self):
        if not self.image_files:
            return

        current_file = self.parent().filename
        if not current_file or current_file not in self.image_files:
            return

        current_index = self.image_files.index(current_file)
        if current_index < len(self.image_files) - 1:
            self.switch_to_image(current_index + 1)

    def is_image_labeled(self, image_path):
        if not self.labels:
            return False

        label_path = get_label_file_path(image_path, self.parent().output_dir)
        flags = load_flags_from_json(label_path)
        if not flags:
            return False

        return any(flags.get(label, False) for label in self.labels)

    def prev_unlabeled_image(self):
        if not self.image_files or not self.labels:
            return

        current_file = self.parent().filename
        if not current_file or current_file not in self.image_files:
            return

        current_index = self.image_files.index(current_file)
        for i in range(current_index - 1, -1, -1):
            if not self.is_image_labeled(self.image_files[i]):
                self.switch_to_image(i)
                return

        for i in range(len(self.image_files) - 1, current_index, -1):
            if not self.is_image_labeled(self.image_files[i]):
                self.switch_to_image(i)
                return

        QMessageBox.information(
            self, self.tr("Info"), self.tr("No unlabeled images found.")
        )

    def next_unlabeled_image(self):
        if not self.image_files or not self.labels:
            return

        current_file = self.parent().filename
        if not current_file or current_file not in self.image_files:
            return

        current_index = self.image_files.index(current_file)
        for i in range(current_index + 1, len(self.image_files)):
            if not self.is_image_labeled(self.image_files[i]):
                self.switch_to_image(i)
                return

        for i in range(0, current_index):
            if not self.is_image_labeled(self.image_files[i]):
                self.switch_to_image(i)
                return

        QMessageBox.information(
            self, self.tr("Info"), self.tr("No unlabeled images found.")
        )

    def switch_to_image(self, index):
        if index < 0 or index >= len(self.image_files):
            return

        new_file = self.image_files[index]

        self.switching_image = True
        self.save_current_flags()
        self.parent().load_file(new_file)

        self.update_image_display()
        self.update_navigation_state()
        self.load_current_flags()

        self.switching_image = False

    def jump_to_page(self, page_num):
        if not self.image_files:
            return

        if page_num < 1 or page_num > len(self.image_files):
            return

        self.switch_to_image(page_num - 1)

    def validate_page_input(self, text):
        if not text:
            return

        try:
            if not text.isdigit():
                cursor_pos = self.page_input.cursorPosition()
                clean_text = "".join(c for c in text if c.isdigit())
                self.page_input.setText(clean_text)
                self.page_input.setCursorPosition(
                    min(cursor_pos, len(clean_text))
                )
                return

            page_num = int(text)
            max_pages = len(self.image_files) if self.image_files else 1
            if page_num > max_pages:
                self.page_input.setText(str(max_pages))
                self.page_input.setCursorPosition(len(str(max_pages)))

        except ValueError:
            pass

    def on_page_input_finished(self):
        text = self.page_input.text().strip()

        if not text:
            self.restore_current_page_number()
            return

        try:
            page_num = int(text)
            max_pages = len(self.image_files) if self.image_files else 1

            if page_num < 1:
                self.page_input.setText("1")
            elif page_num > max_pages:
                self.page_input.setText(str(max_pages))

        except ValueError:
            self.restore_current_page_number()

    def restore_current_page_number(self):
        if self.parent().filename and self.image_files:
            try:
                current_index = self.image_files.index(self.parent().filename)
                self.page_input.setText(str(current_index + 1))
            except (ValueError, AttributeError):
                self.page_input.setText("1")
        else:
            self.page_input.setText("1")

    def update_image_display(self):
        if self.parent().filename:
            self.update_filename_label()

            if self.image_files and self.parent().filename in self.image_files:
                current_index = self.image_files.index(self.parent().filename)
                self.page_input.setText(str(current_index + 1))
            else:
                self.page_input.setText("1")

            pixmap = QPixmap(self.parent().filename)
            if not pixmap.isNull():
                label_size = self.image_label.size()
                scaled_pixmap = pixmap.scaled(
                    label_size.width(),
                    label_size.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.image_label.setPixmap(scaled_pixmap)

            self.update_overlay_text()
        else:
            self.filename_label.setText(self.tr("No image loaded"))
            self.image_label.clear()
            self.page_input.setText("0")
            self.overlay_label.hide()

    def update_overlay_text(self):
        if self.checkbox_group:
            try:
                flags = self.checkbox_group.get_selected_flags()
                text = get_display_text_for_flags(flags, self.labels)
                self.overlay_label.update_text(text)
                self.overlay_label.position_overlay(self.image_label)
            except RuntimeError:
                pass

    def update_navigation_state(self):
        if self.image_files and self.parent().filename:
            try:
                current_index = self.image_files.index(self.parent().filename)
                self.prev_button.setEnabled(current_index > 0)
                self.next_button.setEnabled(
                    current_index < len(self.image_files) - 1
                )
            except (ValueError, AttributeError):
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(False)
        else:
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)

    def load_current_flags(self):
        if not self.checkbox_group:
            return

        if self.parent().filename:
            flags = {}
            if hasattr(self.parent(), "flag_widget") and self.labels:
                flag_widget = self.parent().flag_widget

                for label in self.labels:
                    found = False
                    for i in range(flag_widget.count()):
                        item = flag_widget.item(i)
                        if item.text() == label:
                            flags[label] = item.checkState() == Qt.Checked
                            found = True
                            break
                    if not found:
                        flags[label] = False
            else:
                flags = {label: False for label in self.labels}

            self.switching_image = True
            try:
                self.checkbox_group.set_flags(flags)
                self.update_overlay_text()
            except RuntimeError:
                pass
            finally:
                self.switching_image = False

    def save_current_flags(self):
        if self.switching_image or not self.checkbox_group:
            return

        try:
            flags = self.checkbox_group.get_selected_flags()
        except RuntimeError:
            return

        if self.parent().filename:
            if hasattr(self.parent(), "flags"):
                self.parent().flags = flags

            if hasattr(self.parent(), "flag_widget"):
                self.sync_to_main_flag_widget(flags)

            if hasattr(self.parent(), "set_dirty"):
                self.parent().set_dirty()

        self.update_overlay_text()

    def ensure_labels_in_flag_widget(self, new_labels):
        flag_widget = self.parent().flag_widget

        for label in new_labels:
            found = False
            for i in range(flag_widget.count()):
                item = flag_widget.item(i)
                if item.text() == label:
                    found = True
                    break

            if not found:
                item = QListWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                flag_widget.addItem(item)

    def sync_to_main_flag_widget(self, flags):
        try:
            flag_widget = self.parent().flag_widget

            for i in range(flag_widget.count()):
                item = flag_widget.item(i)
                key = item.text()
                if key in flags:
                    item.setCheckState(
                        Qt.Checked if flags[key] else Qt.Unchecked
                    )
        except Exception as e:
            logger.error(f"Error syncing to flag_widget: {e}")

    def update_main_flag_widget_labels(self, labels_map):
        """更新主flag_widget中的标签名称"""
        try:
            flag_widget = self.parent().flag_widget

            for i in range(flag_widget.count()):
                item = flag_widget.item(i)
                current_label = item.text()
                if current_label in labels_map:
                    new_label = labels_map[current_label]
                    if new_label != current_label:
                        # 保存当前的勾选状态
                        check_state = item.checkState()
                        item.setText(new_label)
                        item.setCheckState(check_state)
        except Exception as e:
            logger.error(f"Error updating flag_widget labels: {e}")

    def _delayed_update_ui(self):
        self.create_checkbox_group()
        self.load_current_flags()
        self.update_filename_label()

    def update_label_files(self, labels_map={}):
        progress_dialog = QProgressDialog(
            self.tr("Updating label files..."),
            self.tr("Cancel"),
            0,
            len(self.image_files),
            self,
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setMinimumWidth(500)
        progress_dialog.setMinimumHeight(150)
        progress_dialog.setStyleSheet(
            get_progress_dialog_style(color="#1d1d1f", height=20)
        )

        try:
            for i, image_file in enumerate(self.image_files):
                label_path = get_label_file_path(
                    image_file, self.parent().output_dir
                )
                if not os.path.exists(label_path):
                    continue

                flags = load_flags_from_json(label_path)
                save_flags = {}
                for label in self.labels:
                    check_label = labels_map[label] if labels_map else label
                    save_flags[label] = flags.get(check_label, False) is True
                save_flags_to_json(label_path, save_flags)

                progress_dialog.setValue(i + 1)
                if progress_dialog.wasCanceled():
                    break

            progress_dialog.close()
            message = self.tr("Labels update successfully!")
            popup = Popup(
                message,
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")

        except Exception as e:
            progress_dialog.close()
            template = self.tr("Error occurred while updating label files: %s")
            logger.error(template % str(e))
            popup = Popup(
                template % str(e),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

        finally:
            QTimer.singleShot(100, self._delayed_update_ui)
            if self.image_files:
                self.page_input.setValidator(
                    QIntValidator(1, len(self.image_files))
                )

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(10, self.set_initial_splitter_size)
        QTimer.singleShot(100, self.update_image_display)

    def set_initial_splitter_size(self):
        total_width = self.main_splitter.width()
        right_width = total_width - self.target_left_width
        if right_width > 0:
            self.main_splitter.setSizes([self.target_left_width, right_width])

    def closeEvent(self, event):
        if self.top_window:
            self.top_window.show()
            self.top_window.raise_()
            self.top_window.activateWindow()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
