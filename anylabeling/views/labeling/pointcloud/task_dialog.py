import os
from dataclasses import dataclass
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.utils.colormap import label_colormap
from anylabeling.views.labeling.utils.style import get_dialog_style
from anylabeling.views.labeling.utils.theme import get_theme

from .camera import CameraConfigurationDialog, CameraSource
from .io import class_config_data, discover_frames, load_classes, save_classes
from .model import DEFAULT_CLASSES, ClassDefinition


@dataclass(frozen=True)
class TaskConfiguration:
    task: str
    classes: tuple[ClassDefinition, ...]
    files: tuple[Path, ...]
    output_directory: Path
    cameras: tuple[CameraSource, ...]


class _TaskProgress(QtWidgets.QWidget):
    def __init__(self, titles, parent):
        super().__init__(parent)
        self.titles = titles
        self.step = 0
        self.setFixedHeight(40)

    def set_step(self, step):
        self.step = step
        self.setAccessibleName(self.titles[step])
        self.update()

    def paintEvent(self, event):
        theme = get_theme()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        centers = [self.width() * (index + 0.5) / 3 for index in range(3)]
        for index in range(2):
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor(
                        theme["primary"]
                        if index < self.step
                        else theme["border"]
                    ),
                    2,
                )
            )
            painter.drawLine(
                QtCore.QPointF(centers[index] + 24, 20),
                QtCore.QPointF(centers[index + 1] - 24, 20),
            )
        for index, center in enumerate(centers):
            active = index <= self.step
            color = QtGui.QColor(
                theme["primary"] if active else theme["border"]
            )
            painter.setPen(QtGui.QPen(color, 1.5))
            painter.setBrush(
                color if active else QtGui.QColor(theme["background"])
            )
            circle = QtCore.QRectF(center - 16, 4, 32, 32)
            painter.drawEllipse(circle)
            painter.setPen(
                QtGui.QColor("#ffffff" if active else theme["text_secondary"])
            )
            font = self.font()
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                circle, QtCore.Qt.AlignmentFlag.AlignCenter, str(index + 1)
            )


class CreateTaskDialog(QtWidgets.QDialog):
    def __init__(self, workspace):
        super().__init__(workspace)
        self.setWindowTitle(self.tr("Create task"))
        self.setStyleSheet(get_dialog_style())
        self.workspace = workspace
        self.configuration = None
        self.task = "detection"
        self.rows = []
        self.drafts = {"detection": [], "segmentation": []}
        self.unlabeled = DEFAULT_CLASSES[0]
        self.files = []
        self.output_directory = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetNoConstraint
        )
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(16)
        self.progress = _TaskProgress(
            (
                self.tr("Task and classes"),
                self.tr("Point clouds"),
                self.tr("Camera images"),
            ),
            self,
        )
        layout.addWidget(self.progress)
        self.pages = QtWidgets.QStackedWidget()
        layout.addWidget(self.pages, 1)
        self._build_classes_page()
        self._build_data_page()
        self.camera_page = CameraConfigurationDialog(
            workspace.camera_panel, self, sources=[], embedded=True
        )
        self.camera_page.layout().setContentsMargins(0, 0, 0, 0)
        camera_hint = QtWidgets.QLabel(
            self.tr("This step is optional. You can create the task now.")
        )
        camera_hint.setStyleSheet(f"color: {get_theme()['primary']};")
        self.camera_page.layout().addWidget(camera_hint, 5, 0)
        camera_scroll = QtWidgets.QScrollArea()
        camera_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        camera_scroll.setWidgetResizable(True)
        camera_scroll.setWidget(self.camera_page)
        self.pages.addWidget(camera_scroll)
        self.error_label = QtWidgets.QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.error_label.setStyleSheet(f"color: {get_theme()['error']};")
        self.error_label.hide()
        error_content = QtWidgets.QWidget()
        error_layout = QtWidgets.QVBoxLayout(error_content)
        error_layout.setContentsMargins(0, 0, 0, 0)
        error_layout.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize
        )
        error_layout.addWidget(self.error_label)
        self.error_scroll = error_scroll = QtWidgets.QScrollArea()
        error_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        error_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        error_scroll.setWidgetResizable(True)
        error_scroll.setWidget(error_content)
        error_content.setAutoFillBackground(False)
        error_scroll.setFixedHeight(40)
        error_scroll.hide()
        layout.addWidget(error_scroll)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, self.camera_page._scrollbar_gutter, 0)
        buttons.setSpacing(10)
        self.save_classes_button = QtWidgets.QPushButton(
            self.tr("Save labels")
        )
        self.save_classes_button.setIcon(
            self.workspace._icon("pointcloud-download")
        )
        self.save_classes_button.clicked.connect(self._save_classes)
        buttons.addWidget(self.save_classes_button)
        buttons.addWidget(self.camera_page.add_button)
        buttons.addStretch()
        self.back_button = QtWidgets.QPushButton(self.tr("Back"))
        self.back_button.clicked.connect(self._back)
        buttons.addWidget(self.back_button)
        self.next_button = QtWidgets.QPushButton(self.tr("Next"))
        self.next_button.setObjectName("pointcloudConfirmButton")
        self.next_button.clicked.connect(self.accept)
        buttons.addWidget(self.next_button)
        layout.addLayout(buttons)
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setAutoDefault(False)
        self.next_button.setDefault(True)
        self.pages.widget(1).layout().setContentsMargins(
            0, 0, self.camera_page._scrollbar_gutter, 0
        )
        self._update_step()
        self.ensurePolished()
        margins = layout.contentsMargins()
        height = (
            margins.top()
            + self.progress.height()
            + self.camera_page.sizeHint().height()
            + buttons.sizeHint().height()
            + layout.spacing() * 2
            + margins.bottom()
        )
        self.setFixedSize(
            680, min(height, self.screen().availableGeometry().height() - 60)
        )

    def _build_classes_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        header = QtWidgets.QGridLayout()
        header.setHorizontalSpacing(10)
        header.setVerticalSpacing(12)
        header.setColumnStretch(3, 1)
        header.addWidget(QtWidgets.QLabel(self.tr("Task:")), 0, 0)
        tasks = QtWidgets.QHBoxLayout()
        tasks.setSpacing(10)
        self.task_buttons = {}
        for task, title in (("detection", "Det"), ("segmentation", "Seg")):
            button = QtWidgets.QRadioButton(title)
            button.setChecked(task == self.task)
            button.toggled.connect(
                lambda checked, task=task: (
                    self._change_task(task) if checked else None
                )
            )
            self.task_buttons[task] = button
            tasks.addWidget(button)
        header.addLayout(tasks, 0, 1, QtCore.Qt.AlignmentFlag.AlignCenter)
        header.addWidget(QtWidgets.QLabel(self.tr("Classes:")), 1, 0)
        self.add_class_button = QtWidgets.QPushButton(self.tr("New class"))
        self.add_class_button.setIcon(self.workspace._icon("new"))
        self.add_class_button.clicked.connect(lambda: self.add_class())
        header.addWidget(self.add_class_button, 1, 1)
        self.upload_button = QtWidgets.QPushButton(self.tr("Load classes"))
        self.upload_button.setIcon(self.workspace._icon("pointcloud-upload"))
        self.upload_button.clicked.connect(self._upload_classes)
        header.addWidget(self.upload_button, 1, 2)
        layout.addLayout(header)
        self.classes_scroll = scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        content = QtWidgets.QWidget()
        self.rows_layout = QtWidgets.QVBoxLayout(content)
        self.rows_layout.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetMinimumSize
        )
        self.rows_layout.setContentsMargins(0, 0, 8, 0)
        self.rows_layout.setSpacing(8)
        self.rows_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(content)
        content.setAutoFillBackground(False)
        layout.addWidget(scroll, 1)
        self.pages.addWidget(page)

    def _build_data_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(12)
        layout.setColumnStretch(0, 1)
        self.directory_input = QtWidgets.QLineEdit()
        self.directory_input.setPlaceholderText(
            self.tr("Point cloud directory")
        )
        self.output_input = QtWidgets.QLineEdit()
        self.output_input.setPlaceholderText(
            self.tr("Optional — defaults to the point cloud directory")
        )
        for row, (title, field) in enumerate(
            (
                (self.tr("Point cloud directory"), self.directory_input),
                (self.tr("Save directory"), self.output_input),
            )
        ):
            label = QtWidgets.QLabel(title)
            label.setBuddy(field)
            layout.addWidget(label, row * 2, 0, 1, 2)
            field.setAccessibleName(title)
            layout.addWidget(field, row * 2 + 1, 0)
            browse = QtWidgets.QPushButton(self.tr("Browse…"))
            browse.clicked.connect(
                lambda checked=False, field=field, title=title: self._browse(
                    field, title
                )
            )
            layout.addWidget(browse, row * 2 + 1, 1)
        layout.setRowStretch(4, 1)
        self.pages.addWidget(page)

    def add_class(self, definition=None):
        if definition is None:
            used = {row.id_input.value() for row in self.rows}
            identifier = next(
                (value for value in range(1, 65536) if value not in used), None
            )
            if identifier is None:
                return
            rgb = label_colormap(256)[identifier % 255 + 1]
            definition = ClassDefinition(
                identifier, "", QtGui.QColor(*(int(v) for v in rgb)).name()
            )
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        row.id_input = QtWidgets.QSpinBox()
        row.id_input.setRange(1, 65535)
        row.id_input.setValue(definition.id)
        row.id_input.setFixedWidth(86)
        row.id_input.setAccessibleName(self.tr("Class ID"))
        row.name_input = QtWidgets.QLineEdit(definition.name)
        row.name_input.setPlaceholderText(self.tr("Class name"))
        row.color = definition.color
        row.color_button = QtWidgets.QToolButton()
        row.color_button.setFixedSize(32, 32)
        row.color_button.setIconSize(QtCore.QSize(20, 20))
        row.color_button.setToolTip(self.tr("Choose color"))
        row.color_button.clicked.connect(lambda: self._choose_color(row))
        self._update_color(row)
        delete = QtWidgets.QToolButton()
        delete.setFixedSize(32, 32)
        delete.setIcon(self.workspace._icon("trash"))
        delete.setToolTip(self.tr("Delete class"))
        delete.clicked.connect(lambda: self._remove_class(row))
        layout.addWidget(row.id_input)
        layout.addWidget(row.name_input, 1)
        layout.addWidget(row.color_button)
        layout.addWidget(delete)
        self.rows.append(row)
        self.rows_layout.addWidget(row)
        row.name_input.setFocus()

    def _remove_class(self, row):
        self.rows.remove(row)
        self.rows_layout.removeWidget(row)
        row.deleteLater()

    def _choose_color(self, row):
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(row.color), self, self.tr("Choose color")
        )
        if color.isValid():
            row.color = color.name()
            self._update_color(row)

    def _update_color(self, row):
        pixmap = QtGui.QPixmap(20, 20)
        pixmap.fill(QtGui.QColor(row.color))
        row.color_button.setIcon(QtGui.QIcon(pixmap))
        row.color_button.setAccessibleName(
            self.tr("Choose color") + ": " + row.color
        )

    def _classes(self):
        return [
            ClassDefinition(
                row.id_input.value(), row.name_input.text().strip(), row.color
            )
            for row in self.rows
        ]

    def _set_classes(self, classes):
        for row in self.rows[:]:
            self._remove_class(row)
        for definition in classes:
            if definition.id:
                self.add_class(definition)

    def _validated_classes(self):
        classes = self._classes()
        if not classes:
            raise ValueError(self.tr("Add at least one class."))
        if self.task == "segmentation":
            classes = [self.unlabeled] + classes
        class_config_data({self.task: classes})
        return tuple(classes)

    def _save_classes(self):
        try:
            classes = self._validated_classes()
        except ValueError as error:
            self._error(str(error))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Save labels"),
            str(Path(self.workspace._recent()) / "pointcloud_classes.json"),
            self.tr("Class definitions (*.json)"),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog
            | QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons,
        )
        if not path:
            return
        try:
            save_classes(path, {self.task: classes})
        except (OSError, ValueError) as error:
            self._error(str(error))
            return
        self.error_label.hide()
        self.error_scroll.hide()

    def _change_task(self, task):
        self.drafts[self.task] = self._classes()
        self.task = task
        self._set_classes(self.drafts[task])
        self.error_label.hide()
        self.error_scroll.hide()

    def _upload_classes(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Load classes"),
            self.workspace._recent(),
            self.tr("Class definitions (*.json)"),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog
            | QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons,
        )
        if not path:
            return
        try:
            definitions = load_classes(path)
            if self.task not in definitions:
                raise ValueError(
                    self.tr(
                        "This file does not contain classes for the selected task."
                    )
                )
        except (OSError, ValueError) as error:
            self._error(str(error))
            return
        if (
            QtWidgets.QMessageBox.question(
                self,
                self.tr("Load classes"),
                self.tr(
                    "Replace all {task} classes in this setup with this file?"
                ).format(task=self.task_buttons[self.task].text()),
                QtWidgets.QMessageBox.StandardButton.Ok
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            != QtWidgets.QMessageBox.StandardButton.Ok
        ):
            return
        self._set_classes(definitions[self.task])
        if self.task == "segmentation":
            self.unlabeled = next(
                item for item in definitions[self.task] if item.id == 0
            )
        self.error_label.hide()
        self.error_scroll.hide()

    def _browse(self, field, title):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            title,
            field.text()
            or self.directory_input.text()
            or self.workspace._recent(),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog
            | QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons,
        )
        if directory:
            field.setText(directory)

    def _error(self, text):
        self.error_label.setText(text)
        self.error_label.show()
        self.error_scroll.show()

    def _update_step(self):
        step = self.pages.currentIndex()
        for index in range(self.pages.count()):
            self.pages.widget(index).setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                (
                    QtWidgets.QSizePolicy.Policy.Preferred
                    if index == step
                    else QtWidgets.QSizePolicy.Policy.Ignored
                ),
            )
        self.progress.set_step(step)
        self.save_classes_button.setVisible(step == 0)
        self.camera_page.add_button.setVisible(step == 2)
        self.back_button.setVisible(step > 0)
        self.next_button.setText(
            self.tr("Create task") if step == 2 else self.tr("Next")
        )
        self.error_label.hide()
        self.error_scroll.hide()

    def _back(self):
        self.pages.setCurrentIndex(self.pages.currentIndex() - 1)
        self._update_step()

    def accept(self):
        step = self.pages.currentIndex()
        if step == 0:
            try:
                self.classes = self._validated_classes()
            except ValueError as error:
                self._error(str(error))
                return
        elif step == 1:
            directory = self.directory_input.text().strip()
            if not directory:
                self._error(self.tr("Choose a point cloud directory."))
                return
            try:
                self.files = discover_frames(
                    Path(directory).expanduser().resolve()
                )
                output = self.output_input.text().strip()
                self.output_directory = (
                    Path(output).expanduser().resolve()
                    if output
                    else self.files[0].parent
                )
                parent = self.output_directory
                while not parent.exists():
                    parent = parent.parent
                if not parent.is_dir() or not os.access(parent, os.W_OK):
                    raise ValueError(
                        self.tr("Choose a writable save directory.")
                    )
            except (OSError, ValueError) as error:
                self._error(str(error))
                return
        else:
            cameras = self.camera_page.read_configuration()
            if cameras is None:
                self._error(self.camera_page.error_label.text())
                self.camera_page.error_label.hide()
                return
            self.configuration = TaskConfiguration(
                self.task,
                self.classes,
                tuple(self.files),
                self.output_directory,
                tuple(cameras),
            )
            super().accept()
            return
        self.pages.setCurrentIndex(step + 1)
        self._update_step()
