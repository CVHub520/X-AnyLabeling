import cv2
import os
import os.path as osp
import shutil

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
)

from anylabeling.views.labeling.chatbot.style import ChatbotDialogStyle
from anylabeling.views.labeling.chatbot.utils import set_icon_path
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.general import is_chinese
from anylabeling.views.labeling.utils.style import (
    get_msg_box_style,
    get_progress_dialog_style,
    get_ok_btn_style,
    get_cancel_btn_style,
)
from anylabeling.views.labeling.widgets import Popup


class FrameExtractionDialog(QDialog):
    def __init__(self, parent=None, total_frames=0, fps=0):
        super().__init__(parent)
        self.total_frames = total_frames
        self.fps = fps
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle(self.tr("Frame Extraction Settings"))
        self.setStyleSheet(get_msg_box_style())
        layout = QVBoxLayout()

        # Interval input
        interval_layout = QHBoxLayout()
        interval_label = QLabel(self.tr(f"Extract every N frames (fps: {self.fps}):"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, self.total_frames)
        self.interval_spin.setValue(1)
        self.interval_spin.setStyleSheet(ChatbotDialogStyle.get_spinbox_style(
            up_arrow_url=set_icon_path("caret-up"),
            down_arrow_url=set_icon_path("caret-down"),
        ))
        self.interval_spin.setMinimumWidth(100)
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_spin)

        # Prefix input
        prefix_layout = QHBoxLayout()
        prefix_label = QLabel(self.tr("Filename prefix:"))
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setStyleSheet(ChatbotDialogStyle.get_settings_edit_style())
        self.prefix_edit.setText("frame_")
        prefix_layout.addWidget(prefix_label)
        prefix_layout.addWidget(self.prefix_edit)

        # Sequence length input
        seq_layout = QHBoxLayout()
        seq_label = QLabel(self.tr("Number sequence length:"))
        self.seq_spin = QSpinBox()
        self.seq_spin.setRange(3, 10)
        self.seq_spin.setValue(5)
        self.seq_spin.setMinimumWidth(100)
        self.seq_spin.setStyleSheet(ChatbotDialogStyle.get_spinbox_style(
            up_arrow_url=set_icon_path("caret-up"),
            down_arrow_url=set_icon_path("caret-down"),
        ))
        seq_layout.addWidget(seq_label)
        seq_layout.addWidget(self.seq_spin)

        layout.addLayout(prefix_layout)
        layout.addLayout(seq_layout)
        layout.addLayout(interval_layout)

        # Example output
        self.example_label = QLabel()
        self.update_example()
        layout.addWidget(self.example_label)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton(self.tr("OK"))
        ok_button.setStyleSheet(get_ok_btn_style())
        cancel_button = QPushButton(self.tr("Cancel"))
        cancel_button.setStyleSheet(get_cancel_btn_style())
        
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # Connect signals
        self.prefix_edit.textChanged.connect(self.update_example)
        self.seq_spin.valueChanged.connect(self.update_example)

        self.setLayout(layout)
        self.resize(480, 240)

    def update_example(self):
        example = f"{self.prefix_edit.text()}{str(1).zfill(self.seq_spin.value())}.jpg"
        self.example_label.setText(self.tr(f"Example output: {example}"))

    def get_values(self):
        return (
            self.interval_spin.value(),
            self.prefix_edit.text(),
            self.seq_spin.value()
        )

def extract_frames_from_video(self, input_file, out_dir):
    video_capture = cv2.VideoCapture(input_file)
    if not video_capture.isOpened():
        QMessageBox.critical(self, "Error", "Failed to open video file.")
        return None

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Show settings dialog
    dialog = FrameExtractionDialog(self, total_frames, fps)
    if not dialog.exec_():
        video_capture.release()
        return None

    interval, prefix, seq_len = dialog.get_values()
    os.makedirs(out_dir, exist_ok=True)

    progress_dialog = QProgressDialog(
        self.tr("Extracting frames. Please wait..."), self.tr("Cancel"), 
        0, total_frames // interval, self
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Progress"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setStyleSheet(get_progress_dialog_style(
        color="#1d1d1f", height=20
    ))

    frame_count = 0
    saved_frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_filename = osp.join(
                out_dir,
                f"{prefix}{str(saved_frame_count).zfill(seq_len)}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        progress_dialog.setValue(saved_frame_count)
        if progress_dialog.wasCanceled():
            break

        frame_count += 1

    video_capture.release()
    progress_dialog.close()


def open_video_file(self):
    if not self.may_continue():
        return

    filter = (
        "*.asf *.avi *.m4v *.mkv *.mov *.mp4 *.mpeg *.mpg *.ts *.wmv"
    )
    input_file, _ = QFileDialog.getOpenFileName(
        self,
        self.tr("Open Video file"),
        "",
        filter,
    )

    if not osp.exists(input_file):
        return

    # Check if the path contains Chinese characters
    if is_chinese(input_file):
        popup = Popup(
            self.tr("File path contains Chinese characters, invalid path!"),
            self, msec=3000,
            icon="anylabeling/resources/icons/warning.svg",
        )
        popup.show_popup(self, position="center")
        return

    out_dir = osp.join(
        osp.dirname(input_file),
        osp.splitext(osp.basename(input_file))[0]
    )

    if osp.exists(out_dir):
        response = QMessageBox()
        response.setIcon(QMessageBox.Warning)
        response.setWindowTitle(self.tr("Warning"))
        response.setText(self.tr("Directory Already Exists"))
        response.setInformativeText(
            self.tr(
                "Do you want to overwrite the existing directory?"
            )
        )
        response.setStandardButtons(
            QMessageBox.Cancel | QMessageBox.Ok
        )
        response.setStyleSheet(get_msg_box_style())

        if response.exec_() != QMessageBox.Ok:
            return

        shutil.rmtree(out_dir)

    # Extract frames from video
    extract_frames_from_video(self, input_file, out_dir)
    logger.info(
        f"✅ Frames successfully extracted to: {out_dir}"
    )

    # Update the canvas
    self.import_image_folder(out_dir)
