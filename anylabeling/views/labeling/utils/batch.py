import base64
import json
import os.path as osp
from PIL import Image

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QProgressDialog,
    QDialog,
    QLabel,
    QLineEdit,
    QDialogButtonBox,
    QApplication,
)

from anylabeling.app_info import __version__
from anylabeling.views.labeling.utils.theme import get_theme
from anylabeling.services.auto_labeling import (
    _BATCH_PROCESSING_INVALID_MODELS,
    _BATCH_PROCESSING_TEXT_PROMPT_MODELS,
    _BATCH_PROCESSING_VIDEO_MODELS,
    _SKIP_DET_MODELS,
)
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils._io import io_open
from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import get_msg_box_style
from anylabeling.views.labeling.widgets.popup import Popup


__all__ = ["run_all_images"]


class TextInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Enter Text Prompt"))
        self.setFixedSize(400, 180)
        self.setWindowFlags(Qt.Dialog | Qt.MSWindowsFixedSizeDialogHint)

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
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        t = get_theme()
        self.setStyleSheet(
            f"""
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
        """
        )

    def get_input_text(self):
        if self.exec_() == QDialog.Accepted:
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
    target_index = self.current_index
    target_file = self.image_list[self.current_index]
    self.import_image_folder(osp.dirname(target_file), load=False)
    self.file_list_widget.setCurrentRow(target_index)

    del self.text_prompt
    del self.run_tracker
    del self.image_index
    del self.current_index

    progress_dialog.close()

    popup = Popup(
        self.tr("Processing completed successfully!"),
        self,
        icon=new_icon_path("copy-green", "svg"),
    )
    popup.show_popup(self, position="center")


def cancel_operation(self):
    self.cancel_processing = True


def save_auto_labeling_result(self, image_file, auto_labeling_result):
    try:
        label_file = osp.splitext(image_file)[0] + ".json"
        if self.output_dir:
            label_file = osp.join(self.output_dir, osp.basename(label_file))

        if auto_labeling_result is None:
            new_shapes = []
            new_description = ""
            replace = True
        else:
            new_shapes = [
                shape.to_dict() for shape in auto_labeling_result.shapes
            ]
            new_description = auto_labeling_result.description
            replace = auto_labeling_result.replace

        if osp.exists(label_file):
            with io_open(label_file, "r") as f:
                data = json.load(f)

            if replace:
                data["shapes"] = new_shapes
                data["description"] = new_description
            else:
                data["shapes"].extend(new_shapes)
                if "description" in data:
                    data["description"] += new_description
                else:
                    data["description"] = new_description
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
                "shapes": new_shapes,
                "imagePath": image_path,
                "imageData": image_data,
                "imageHeight": image_height,
                "imageWidth": image_width,
                "description": new_description,
            }

        with io_open(label_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(
            f"Failed to save auto labeling result for image file '{image_file}': {str(e)}"
        )


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
    total_images = len(self.image_list)
    self._progress_dialog = progress_dialog

    try:
        while (self.image_index < total_images) and (
            not self.cancel_processing
        ):
            image_file = self.image_list[self.image_index]
            current_progress = self.image_index + 1
            progress_dialog.setValue(current_progress)
            progress_dialog.setLabelText(
                f"Progress: {current_progress}/{total_images}"
            )
            QApplication.processEvents()

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

        finish_processing(self, progress_dialog)

    except Exception as e:
        progress_dialog.close()

        logger.error(f"Error occurred while processing images: {e}")
        popup = Popup(
            self.tr("Error occurred while processing images!"),
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")


def show_progress_dialog_and_process(self):
    self.cancel_processing = False

    progress_dialog = QProgressDialog(
        self.tr("Processing..."),
        self.tr("Cancel"),
        0,
        len(self.image_list),
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Batch Processing"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)

    initial_progress = (
        self.image_index + 1
        if self.image_index < len(self.image_list)
        else len(self.image_list)
    )
    progress_dialog.setValue(initial_progress)
    progress_dialog.setLabelText(
        f"Progress: {initial_progress}/{len(self.image_list)}"
    )
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
                progress_dialog.setLabelText(f"{value}/{len(self.image_list)}")

        progress_bar.valueChanged.connect(update_progress)

    t = get_theme()
    progress_dialog.setStyleSheet(
        f"""
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
    """
    )
    progress_dialog.canceled.connect(lambda: cancel_operation(self))
    progress_dialog.show()

    QTimer.singleShot(200, lambda: process_next_image(self, progress_dialog))


def run_all_images(self):
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

    response = QtWidgets.QMessageBox()
    response.setIcon(QtWidgets.QMessageBox.Warning)
    response.setWindowTitle(self.tr("Confirmation"))
    response.setText(self.tr("Do you want to process all images?"))
    response.setStandardButtons(
        QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok
    )
    response.setStyleSheet(get_msg_box_style())
    if response.exec_() != QtWidgets.QMessageBox.Ok:
        return

    logger.info("Start running all images...")

    self.current_index = self.fn_to_index[str(self.filename)]
    self.image_index = self.current_index
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
            show_progress_dialog_and_process(self)
        elif batch_processing_mode == "text_prompt":
            text_input_dialog = TextInputDialog(parent=self)
            self.text_prompt = text_input_dialog.get_input_text()
            if self.text_prompt:
                show_progress_dialog_and_process(self)
        else:
            show_progress_dialog_and_process(self)
    elif model_type in _BATCH_PROCESSING_TEXT_PROMPT_MODELS:
        text_input_dialog = TextInputDialog(parent=self)
        self.text_prompt = text_input_dialog.get_input_text()
        if self.text_prompt or model_type == "yoloe":
            show_progress_dialog_and_process(self)
    elif (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        == "florence2"
    ):
        self.text_prompt = self.auto_labeling_widget.edit_text.text()
        show_progress_dialog_and_process(self)
    elif (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in _BATCH_PROCESSING_VIDEO_MODELS
    ):
        self.run_tracker = True
        show_progress_dialog_and_process(self)
    else:
        show_progress_dialog_and_process(self)
