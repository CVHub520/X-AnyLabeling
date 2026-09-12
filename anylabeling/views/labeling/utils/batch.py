import base64
import json
import os.path as osp
from PIL import Image

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QGridLayout,
    QHBoxLayout,
    QProgressDialog,
    QDialog,
    QLabel,
    QLineEdit,
    QSpinBox,
    QPushButton,
    QDialogButtonBox,
    QApplication,
)

from anylabeling.app_info import __version__
from anylabeling.views.labeling.utils.theme import get_theme
from anylabeling.services.auto_labeling import (
    _BATCH_PROCESSING_AUTO_GRID_MODELS,
    _BATCH_PROCESSING_INVALID_MODELS,
    _BATCH_PROCESSING_TEXT_PROMPT_MODELS,
    _BATCH_PROCESSING_VIDEO_MODELS,
    _SKIP_DET_MODELS,
)
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.schema import IMAGE_TAGS_FIELD
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils._io import io_open
from anylabeling.views.labeling.utils.image_tags import normalize_image_tags
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import get_dialog_style
from anylabeling.views.labeling.widgets.popup import Popup

__all__ = ["run_all_images"]


class BatchRangeDialog(QDialog):
    def __init__(self, image_count, start_index=1, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Auto Run"))
        self.setMinimumWidth(440)
        theme = get_theme()
        self.setStyleSheet(get_dialog_style() + f"""
            QLabel#rangeTitle {{ font-size: 18px; font-weight: 600; }}
            QLabel#rangeHint, QLabel#rangeSummary {{
                color: {theme["text_secondary"]};
                font-size: 12px;
            }}
            QSpinBox {{
                padding: 6px 12px;
                font-size: 14px;
            }}
            QPushButton {{ min-width: 60px; }}
            QPushButton#runButton {{
                background-color: {theme["primary"]};
                color: white;
                border-color: {theme["primary"]};
            }}
            QPushButton#runButton:hover {{
                background-color: {theme["primary_hover"]};
            }}
            QPushButton#runButton:pressed {{
                background-color: {theme["primary_pressed"]};
            }}
            """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)
        title = QLabel(self.tr("Image range"))
        title.setObjectName("rangeTitle")
        layout.addWidget(title)
        layout.addSpacing(6)
        hint = QLabel(self.tr("Select the first and last images to process."))
        hint.setObjectName("rangeHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addSpacing(20)

        range_layout = QGridLayout()
        range_layout.setHorizontalSpacing(16)
        range_layout.setVerticalSpacing(8)
        range_layout.setColumnStretch(0, 1)
        range_layout.setColumnStretch(1, 1)
        self.from_input = QSpinBox()
        self.from_input.setRange(1, image_count)
        self.from_input.setValue(start_index)
        self.to_input = QSpinBox()
        self.to_input.setRange(start_index, image_count)
        self.to_input.setValue(image_count)
        self.from_input.valueChanged.connect(self.to_input.setMinimum)
        for column, (text, field) in enumerate(
            (
                (self.tr("From"), self.from_input),
                (self.tr("To"), self.to_input),
            )
        ):
            label = QLabel(text)
            label.setBuddy(field)
            field.setMinimumHeight(40)
            field.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            range_layout.addWidget(label, 0, column)
            range_layout.addWidget(field, 1, column)
        range_layout.setRowMinimumHeight(2, 8)
        summary = QLabel()
        summary.setObjectName("rangeSummary")

        def update_summary():
            selected = self.to_input.value() - self.from_input.value() + 1
            summary.setText(
                self.tr("%s of %s images selected") % (selected, image_count)
            )

        self.from_input.valueChanged.connect(update_summary)
        self.to_input.valueChanged.connect(update_summary)
        update_summary()
        range_layout.addWidget(
            summary,
            3,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        cancel_button = QPushButton(self.tr("Cancel"))
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(cancel_button, 1)
        run_button = QPushButton(self.tr("Run"))
        run_button.setObjectName("runButton")
        run_button.setDefault(True)
        run_button.clicked.connect(self.accept)
        buttons.addWidget(run_button, 1)
        range_layout.addLayout(buttons, 3, 1)
        layout.addLayout(range_layout)


class TextInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Enter Text Prompt"))
        self.setFixedSize(400, 180)
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.MSWindowsFixedSizeDialogHint
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        prompt_label = QLabel(self.tr("Please enter your text prompt:"))
        prompt_label.setStyleSheet(
            f"font-size: 13px; color: {get_theme()['text']}; font-weight: 500;"
        )
        layout.addWidget(prompt_label)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText(self.tr("Enter prompt here..."))
        layout.addWidget(self.text_input)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        t = get_theme()
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {t["background"]};
                border-radius: 10px;
            }}

            QLineEdit {{
                border: 1px solid {t["border"]};
                border-radius: 8px;
                background-color: {t["background_secondary"]};
                font-size: 13px;
                height: 36px;
                padding: 0 12px;
                color: {t["text"]};
            }}

            QLineEdit:hover {{
                background-color: {t["background_hover"]};
            }}

            QLineEdit:focus {{
                border: 2px solid {t["highlight"]};
                background-color: {t["background_secondary"]};
            }}

            QPushButton {{
                min-width: 100px;
                height: 36px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 13px;
            }}

            QPushButton[text="OK"] {{
                background-color: {t["primary"]};
                color: white;
                border: none;
            }}

            QPushButton[text="OK"]:hover {{
                background-color: {t["primary_hover"]};
            }}

            QPushButton[text="OK"]:pressed {{
                background-color: {t["primary"]};
            }}

            QPushButton[text="Cancel"] {{
                background-color: {t["surface"]};
                color: {t["text"]};
                border: 1px solid {t["border"]};
            }}

            QPushButton[text="Cancel"]:hover {{
                background-color: {t["background_hover"]};
            }}

            QPushButton[text="Cancel"]:pressed {{
                background-color: {t["surface"]};
            }}
        """)

    def get_input_text(self):
        if self.exec() == QDialog.DialogCode.Accepted:
            return self.text_input.text().strip()
        return ""


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def load_existing_shapes(image_file):
    """
    Loads existing shapes from the JSON file for skip detection.

    Args:
        image_file (str): The path to the image file.

    Returns:
        list: A list of Shape objects loaded from the JSON file, or None if
              the file does not exist or contains no shapes.
    """
    label_file = osp.splitext(image_file)[0] + ".json"
    if not osp.exists(label_file):
        return None

    try:
        with io_open(label_file, "r") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        if not shapes:
            return None

        existing_shapes = []
        for shape_data in shapes:
            shape = Shape()
            shape.load_from_dict(shape_data, close=False)
            if shape.shape_type in ["rectangle", "rotation", "polygon"]:
                shape.selected = True
                existing_shapes.append(shape)

        return existing_shapes if existing_shapes else None

    except Exception as e:
        logger.warning(f"Failed to load existing shapes: {e}")
        return None


def finish_processing(self, progress_dialog):
    if not getattr(self, "_batch_processing_active", False):
        progress_dialog.close()
        return

    try:
        target_file = self.image_list[self.current_index]
        self.import_image_folder(osp.dirname(target_file), load=False)
        target_index = self.fn_to_index[str(target_file)]
        signals_blocked = self.file_list_widget.blockSignals(True)
        try:
            self.file_list_widget.setCurrentRow(target_index)
        finally:
            self.file_list_widget.blockSignals(signals_blocked)
        self.load_file(target_file)
        QApplication.processEvents()
    finally:
        _reset_batch_processing_state(self)
        progress_dialog.close()

    popup = Popup(
        self.tr("Processing completed successfully!"),
        self,
        icon=new_icon_path("copy-green", "svg"),
    )
    popup.show_popup(self, position="center")


def cancel_operation(self):
    self.cancel_processing = True


def _start_batch_processing(self):
    self._batch_processing_active = True
    show_progress_dialog_and_process(self)


def _reset_batch_processing_state(self):
    self._batch_processing_active = False
    for attribute in (
        "text_prompt",
        "run_tracker",
        "image_index",
        "current_index",
        "_batch_start_index",
        "_batch_end_index",
    ):
        if hasattr(self, attribute):
            delattr(self, attribute)


def _reset_auto_labeling_tracker(self):
    model_manager = self.auto_labeling_widget.model_manager
    if model_manager.loaded_model_config is None:
        return
    model_manager.set_auto_labeling_reset_tracker()


def save_auto_labeling_result(self, image_file, auto_labeling_result):
    try:
        label_file = osp.splitext(image_file)[0] + ".json"
        if self.output_dir:
            label_file = osp.join(self.output_dir, osp.basename(label_file))

        if auto_labeling_result is None:
            new_shapes = []
            new_description = ""
            new_tags = None
            replace = True
        else:
            new_shapes = [
                shape.to_dict() for shape in auto_labeling_result.shapes
            ]
            new_description = auto_labeling_result.description
            new_tags = getattr(auto_labeling_result, "tags", None)
            replace = auto_labeling_result.replace

        if osp.exists(label_file):
            with io_open(label_file, "r") as f:
                data = json.load(f)

            if replace:
                if (
                    data["shapes"] != new_shapes
                    or data.get("description", "") != new_description
                ):
                    data["checked"] = False
                data["shapes"] = new_shapes
                data["description"] = new_description
            else:
                if new_shapes or new_description:
                    data["checked"] = False
                data["shapes"].extend(new_shapes)
                if "description" in data:
                    data["description"] += new_description
                else:
                    data["description"] = new_description
            if new_tags is not None:
                tags = normalize_image_tags(
                    new_tags, f"auto labeling result for {image_file}"
                )
                if data.get(IMAGE_TAGS_FIELD) != tags:
                    data["checked"] = False
                data[IMAGE_TAGS_FIELD] = tags
        else:
            if self._config["store_data"]:
                with open(image_file, "rb") as f:
                    image_data = f.read()
                image_data = base64.b64encode(image_data).decode("utf-8")
            else:
                image_data = None

            image_path = osp.basename(image_file)
            image_width, image_height = get_image_size(image_file)

            data = {
                "version": __version__,
                "flags": {},
                "checked": False,
                "shapes": new_shapes,
                "imagePath": image_path,
                "imageData": image_data,
                "imageHeight": image_height,
                "imageWidth": image_width,
                "description": new_description,
            }
            if new_tags is not None:
                data[IMAGE_TAGS_FIELD] = normalize_image_tags(
                    new_tags, f"auto labeling result for {image_file}"
                )

        with io_open(label_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(
            f"Failed to save auto labeling result for image file '{image_file}': {str(e)}"
        )


class BatchProcessingThread(QThread):
    progress_updated = pyqtSignal(int, str)
    processing_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        app,
        image_list,
        image_index,
        model_type,
        text_prompt,
        run_tracker,
        skip_detection,
    ):
        super().__init__()
        self.app = app
        self.image_list = image_list
        self.image_index = image_index
        self.model_type = model_type
        self.text_prompt = text_prompt
        self.run_tracker = run_tracker
        self.skip_detection = skip_detection

    def run(self):
        total_images = len(self.image_list)
        start_index = self.image_index
        image_count = total_images - start_index
        try:
            while (
                self.image_index < total_images
                and not self.app.cancel_processing
            ):
                image_file = self.image_list[self.image_index]

                if self.text_prompt:
                    result = self.app.auto_labeling_widget.model_manager.predict_shapes(
                        self.app.image,
                        image_file,
                        text_prompt=self.text_prompt,
                        batch=True,
                    )
                elif self.run_tracker:
                    result = self.app.auto_labeling_widget.model_manager.predict_shapes(
                        self.app.image,
                        image_file,
                        run_tracker=self.run_tracker,
                        batch=True,
                    )
                else:
                    existing_shapes = None
                    if (
                        self.model_type in _SKIP_DET_MODELS
                        and self.skip_detection
                    ):
                        existing_shapes = load_existing_shapes(image_file)
                    result = self.app.auto_labeling_widget.model_manager.predict_shapes(
                        self.app.image,
                        image_file,
                        batch=True,
                        existing_shapes=existing_shapes,
                    )

                save_auto_labeling_result(self.app, image_file, result)
                self.image_index += 1
                completed = self.image_index - start_index
                self.progress_updated.emit(
                    completed,
                    f"Progress: {completed}/{image_count}",
                )

            self.app.image_index = self.image_index
            self.processing_finished.emit()
        except Exception as e:
            self.app.image_index = self.image_index
            self.error_occurred.emit(str(e))
        finally:
            _reset_auto_labeling_tracker(self.app)


def process_next_image(self, progress_dialog, batch=True):
    """Process images in batch mode.

    Args:
        progress_dialog: Progress dialog widget for displaying progress.
        batch: If True, results are saved directly without updating canvas.
               If False, results trigger UI updates and canvas refresh.
               Defaults to True for batch processing mode.
    """
    model_type = self.auto_labeling_widget.model_manager.loaded_model_config[
        "type"
    ]
    model = self.auto_labeling_widget.model_manager.loaded_model_config[
        "model"
    ]
    total_images = self._batch_end_index
    image_count = total_images - self._batch_start_index
    self._progress_dialog = progress_dialog

    batch_processing_mode = "default"
    if model_type == "remote_server":
        batch_processing_mode = model.get_batch_processing_mode()

    if (
        model_type not in _BATCH_PROCESSING_VIDEO_MODELS
        and batch_processing_mode != "video"
    ):
        skip_detection = (
            self.auto_labeling_widget.button_skip_detection.isChecked()
        )
        self._batch_thread = BatchProcessingThread(
            self,
            self.image_list[:total_images],
            self.image_index,
            model_type,
            self.text_prompt,
            self.run_tracker,
            skip_detection,
        )

        def _on_progress(value, label):
            progress_dialog.setValue(value)
            progress_dialog.setLabelText(label)

        def _on_error(msg):
            _reset_batch_processing_state(self)
            progress_dialog.close()
            logger.error(f"Error occurred while processing images: {msg}")
            popup = Popup(
                self.tr("Error occurred while processing images!"),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

        self._batch_thread.progress_updated.connect(_on_progress)
        self._batch_thread.processing_finished.connect(
            lambda: finish_processing(self, progress_dialog)
        )
        self._batch_thread.error_occurred.connect(_on_error)
        self._batch_thread.start()
        return

    try:
        while (self.image_index < total_images) and (
            not self.cancel_processing
        ):
            image_file = self.image_list[self.image_index]

            batch_processing_mode = "default"
            if model_type == "remote_server":
                batch_processing_mode = model.get_batch_processing_mode()
                if batch_processing_mode == "video":
                    model._widget = self
                    self.filename = image_file
                    self.load_file(self.filename)
                    batch = False
            elif model_type in _BATCH_PROCESSING_VIDEO_MODELS:
                self.filename = image_file
                self.load_file(self.filename)
                batch = False

            if self.text_prompt:
                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        text_prompt=self.text_prompt,
                        batch=batch,
                    )
                )
            elif self.run_tracker:
                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        run_tracker=self.run_tracker,
                        batch=batch,
                    )
                )
                if batch_processing_mode == "video":
                    logger.info("Video propagation completed, breaking loop")
                    self.image_index = total_images
                    break
            else:
                existing_shapes = None
                if (
                    model_type in _SKIP_DET_MODELS
                    and self.auto_labeling_widget.button_skip_detection.isChecked()
                ):
                    existing_shapes = load_existing_shapes(image_file)

                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        batch=batch,
                        existing_shapes=existing_shapes,
                    )
                )

            if batch:
                save_auto_labeling_result(
                    self, image_file, auto_labeling_result
                )

            self.image_index += 1
            completed = self.image_index - self._batch_start_index
            progress_dialog.setValue(completed)
            progress_dialog.setLabelText(
                f"Progress: {completed}/{image_count}"
            )
            QApplication.processEvents()

        finish_processing(self, progress_dialog)

    except Exception as e:
        _reset_batch_processing_state(self)
        progress_dialog.close()

        logger.error(f"Error occurred while processing images: {e}")
        popup = Popup(
            self.tr("Error occurred while processing images!"),
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
    finally:
        _reset_auto_labeling_tracker(self)


def show_progress_dialog_and_process(self):
    self.cancel_processing = False
    image_count = self._batch_end_index - self._batch_start_index

    progress_dialog = QProgressDialog(
        self.tr("Processing..."),
        self.tr("Cancel"),
        0,
        image_count,
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Batch Processing"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setAutoClose(False)
    progress_dialog.setAutoReset(False)

    progress_dialog.setValue(0)
    progress_dialog.setLabelText(f"Progress: 0/{image_count}")
    progress_bar = progress_dialog.findChild(QtWidgets.QProgressBar)

    if progress_bar:
        model_type = (
            self.auto_labeling_widget.model_manager.loaded_model_config.get(
                "type", ""
            )
        )
        batch_processing_mode = "default"
        if model_type == "remote_server":
            model = self.auto_labeling_widget.model_manager.loaded_model_config.get(
                "model"
            )
            batch_processing_mode = model.get_batch_processing_mode()

        def update_progress(value):
            if batch_processing_mode != "video":
                progress_dialog.setLabelText(f"{value}/{image_count}")

        progress_bar.valueChanged.connect(update_progress)

    t = get_theme()
    progress_dialog.setStyleSheet(f"""
        QProgressDialog {{
            background-color: {t["background"]};
            border-radius: 12px;
            min-width: 280px;
            min-height: 120px;
            padding: 20px;
        }}
        QProgressBar {{
            border: none;
            border-radius: 4px;
            background-color: {t["surface"]};
            text-align: center;
            color: {t["text"]};
            font-size: 13px;
            min-height: 20px;
            max-height: 20px;
            margin: 16px 0;
        }}
        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {t["primary"]},
                stop:0.5 {t["highlight"]},
                stop:1 {t["primary"]});
            border-radius: 3px;
        }}
        QLabel {{
            color: {t["text"]};
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 8px;
        }}
        QPushButton {{
            background-color: {t["surface"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
            color: {t["primary"]};
            min-width: 82px;
            height: 36px;
            padding: 0 16px;
            margin-top: 16px;
        }}
        QPushButton:hover {{
            background-color: {t["background_hover"]};
        }}
        QPushButton:pressed {{
            background-color: {t["surface"]};
        }}
    """)
    progress_dialog.canceled.connect(lambda: cancel_operation(self))
    progress_dialog.show()

    QTimer.singleShot(200, lambda: process_next_image(self, progress_dialog))


def run_all_images(self):
    if getattr(self, "_batch_processing_active", False):
        logger.warning("Batch processing is already running.")
        return

    if len(self.image_list) < 1:
        return

    if self.auto_labeling_widget.model_manager.loaded_model_config is None:
        self.auto_labeling_widget.model_manager.new_model_status.emit(
            self.tr("Model is not loaded. Choose a mode to continue.")
        )
        return

    if (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in _BATCH_PROCESSING_INVALID_MODELS
    ):
        logger.warning(
            f"The model `{self.auto_labeling_widget.model_manager.loaded_model_config['type']}`"
            f" is not supported for this action."
            f" Please choose a valid model to execute."
        )
        self.auto_labeling_widget.model_manager.new_model_status.emit(
            self.tr(
                "Invalid model type, please choose a valid model_type to run."
            )
        )
        return

    current_index = self.fn_to_index[str(self.filename)]
    response = BatchRangeDialog(
        len(self.image_list), current_index + 1, parent=self
    )
    if response.exec() != QDialog.DialogCode.Accepted:
        return

    logger.info("Start running all images...")

    self.current_index = current_index
    self._batch_start_index = response.from_input.value() - 1
    self._batch_end_index = response.to_input.value()
    self.image_index = self._batch_start_index
    self.text_prompt = ""
    self.run_tracker = False

    model_type = self.auto_labeling_widget.model_manager.loaded_model_config[
        "type"
    ]

    if model_type == "remote_server":
        batch_processing_mode = "default"
        model = self.auto_labeling_widget.model_manager.loaded_model_config[
            "model"
        ]
        if hasattr(model, "get_batch_processing_mode"):
            batch_processing_mode = model.get_batch_processing_mode()
        else:
            batch_processing_mode = "default"
        if batch_processing_mode is None:
            self.auto_labeling_widget.model_manager.new_model_status.emit(
                self.tr(
                    "Batch processing is not supported for the current task."
                )
            )
            return
        if batch_processing_mode == "video":
            self.run_tracker = True
            _start_batch_processing(self)
        elif batch_processing_mode == "text_prompt":
            text_input_dialog = TextInputDialog(parent=self)
            self.text_prompt = text_input_dialog.get_input_text()
            if self.text_prompt:
                _start_batch_processing(self)
        else:
            _start_batch_processing(self)
    elif model_type in _BATCH_PROCESSING_AUTO_GRID_MODELS:
        self.auto_labeling_widget.model_manager.set_auto_labeling_marks(
            [{"type": "auto_grid"}]
        )
        _start_batch_processing(self)
    elif model_type in _BATCH_PROCESSING_TEXT_PROMPT_MODELS:
        text_input_dialog = TextInputDialog(parent=self)
        self.text_prompt = text_input_dialog.get_input_text()
        if self.text_prompt or model_type == "yoloe":
            _start_batch_processing(self)
    elif (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        == "florence2"
    ):
        self.text_prompt = self.auto_labeling_widget.edit_text.text()
        _start_batch_processing(self)
    elif (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in _BATCH_PROCESSING_VIDEO_MODELS
    ):
        self.run_tracker = True
        _start_batch_processing(self)
    else:
        _start_batch_processing(self)
