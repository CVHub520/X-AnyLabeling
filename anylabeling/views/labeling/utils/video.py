import cv2
import os
import os.path as osp
import shutil
import tempfile
import time
import subprocess

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
    QApplication,
)

from anylabeling.views.labeling.chatbot.style import ChatbotDialogStyle
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.qt import new_icon_path
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
        layout.setSpacing(8)
        layout.setContentsMargins(12, 8, 12, 12)

        # Interval input
        interval_layout = QHBoxLayout()
        template = self.tr("Frame interval (fps: %.2f):")
        interval_label = QLabel(template % self.fps)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, max(1, self.total_frames))
        self.interval_spin.setValue(1)
        self.interval_spin.setStyleSheet(
            ChatbotDialogStyle.get_spinbox_style(
                up_arrow_url=new_icon_path("caret-up", "svg"),
                down_arrow_url=new_icon_path("caret-down", "svg"),
            )
        )
        self.interval_spin.setMinimumWidth(100)
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_spin)

        # Prefix input
        prefix_layout = QHBoxLayout()
        prefix_label = QLabel(self.tr("Filename prefix:"))
        self.prefix_edit = QLineEdit()
        base_style = ChatbotDialogStyle.get_settings_edit_style()
        self.prefix_edit.setStyleSheet(
            base_style
            + """
            QLineEdit {
                padding-top: 6px;
                padding-right: 8px;
                padding-bottom: 0px;
                padding-left: 8px;
                min-height: 28px;
            }
            """
        )
        self.prefix_edit.setText("frame_")
        prefix_layout.addWidget(prefix_label)
        prefix_layout.addWidget(self.prefix_edit)
        prefix_layout.setAlignment(Qt.AlignVCenter)

        # Sequence length input
        seq_layout = QHBoxLayout()
        seq_label = QLabel(self.tr("Number sequence length:"))
        self.seq_spin = QSpinBox()
        self.seq_spin.setRange(3, 10)
        self.seq_spin.setValue(5)
        self.seq_spin.setMinimumWidth(100)
        self.seq_spin.setStyleSheet(
            ChatbotDialogStyle.get_spinbox_style(
                up_arrow_url=new_icon_path("caret-up", "svg"),
                down_arrow_url=new_icon_path("caret-down", "svg"),
            )
        )
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
        template = self.tr("Example output: {example}")
        self.example_label.setText(template.format(example=example))

    def get_values(self):
        return (
            self.interval_spin.value(),
            self.prefix_edit.text(),
            self.seq_spin.value(),
        )


def extract_frames_from_video(self, input_file, out_dir):
    temp_video_path = None
    video_capture = None
    opened_successfully = False
    ffmpeg_path = None

    try:
        input_file_str = str(input_file)

        # Load video directly
        video_capture = cv2.VideoCapture(input_file_str)
        if video_capture.isOpened():
            opened_successfully = True
        else:
            video_capture.release()
            logger.warning(
                f"Loading video failed. Trying temporary file workaround."
            )

            try:
                with open(input_file, "rb") as f:
                    video_data = f.read()
                _, ext = osp.splitext(input_file)
                suffix = ext if ext else ".mp4"
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                )
                temp_video_path = temp_file.name
                temp_file.write(video_data)
                temp_file.close()
                logger.debug(
                    f"Writing video data to temporary file: {temp_video_path}"
                )

                video_capture = cv2.VideoCapture(temp_video_path)
                if video_capture.isOpened():
                    opened_successfully = True
                else:
                    video_capture.release()
                    logger.error(
                        f"Failed to open video via temporary file: {temp_video_path}"
                    )
            except Exception as e:
                logger.error(f"Error during temporary file workaround: {e}")
                if video_capture:
                    video_capture.release()

        if not opened_successfully:
            popup = Popup(
                f"Failed to open video file: {osp.basename(input_file)}",
                self,
                icon=new_icon_path("warning", "svg"),
            )
            popup.show_popup(self, position="center")
            return None

        # --- Proceed with frame extraction settings ---
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        # Handle cases where fps might be 0 or invalid
        if not fps or fps <= 0:
            logger.warning(
                f"Invalid or zero FPS ({fps}) detected for video. Defaulting FPS to 30 for calculations."
            )
            fps = 30.0  # Assign a default FPS
        logger.info(
            f"Video opened: Total Frames ~{total_frames}, FPS ~{fps:.2f}"
        )

        dialog = FrameExtractionDialog(self, total_frames, fps)
        if not dialog.exec_():
            logger.info(
                "Frame extraction cancelled by user in settings dialog."
            )
            # video_capture is released in the outer finally block
            return None

        interval, prefix, seq_len = dialog.get_values()
        os.makedirs(out_dir, exist_ok=True)

        # --- Check for ffmpeg ---
        ffmpeg_path = shutil.which("ffmpeg")

        # Inner try: Handle the actual extraction (ffmpeg or OpenCV)
        try:
            if ffmpeg_path:
                logger.info(f"Detected ffmpeg for extraction: {ffmpeg_path}")
                # --- FFMPEG Path ---
                # VideoCapture is no longer needed once ffmpeg takes over
                if video_capture and video_capture.isOpened():
                    video_capture.release()

                progress_dialog = QProgressDialog(
                    self.tr("Extracting frames using ffmpeg..."),
                    self.tr("Cancel"),
                    0,
                    0,
                    self,  # Range (0,0) makes it indeterminate
                )
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.setWindowTitle(self.tr("Progress"))
                progress_dialog.setMinimumWidth(400)
                progress_dialog.setMinimumHeight(150)
                progress_dialog.setStyleSheet(
                    get_progress_dialog_style(color="#1d1d1f", height=20)
                )
                progress_dialog.show()
                QApplication.processEvents()  # Ensure dialog is displayed

                video_source_path = (
                    temp_video_path if temp_video_path else input_file_str
                )
                output_pattern = osp.join(out_dir, f"{prefix}%0{seq_len}d.jpg")
                output_fps = (
                    fps / interval if interval > 0 else fps
                )  # Avoid division by zero

                cmd = [
                    ffmpeg_path,
                    "-i",
                    video_source_path,
                    "-vf",
                    f"fps={output_fps}",
                    "-qscale:v",
                    "2",  # High quality JPEG
                    "-start_number",
                    "0",
                    output_pattern,
                ]
                logger.info(f"Running ffmpeg command: {' '.join(cmd)}")

                ffmpeg_failed = False
                try:
                    # Using Popen for potential cancellation, though complex
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                    )
                    while process.poll() is None:  # While process is running
                        QApplication.processEvents()  # Keep UI responsive
                        if progress_dialog.wasCanceled():
                            logger.warning(
                                "Cancellation requested for ffmpeg process."
                            )
                            try:
                                process.terminate()  # Ask nicely first
                                process.wait(timeout=2)  # Wait a bit
                            except subprocess.TimeoutExpired:
                                logger.warning(
                                    "ffmpeg did not terminate gracefully, killing."
                                )
                                process.kill()  # Force kill
                            logger.warning(
                                "ffmpeg process terminated due to cancellation."
                            )
                            ffmpeg_failed = (
                                True  # Treat cancellation as failure for now
                            )
                            break

                    if not ffmpeg_failed:
                        progress_dialog.setLabelText(
                            self.tr("Verifying extracted frames...")
                        )
                        QApplication.processEvents()
                        time.sleep(0.5)

                        stdout, stderr = (
                            process.communicate()
                        )  # Get final output
                        if process.returncode != 0:
                            logger.error(
                                f"ffmpeg failed with exit code {process.returncode}"
                            )
                            logger.error(f"ffmpeg stderr: {stderr}")
                            logger.error(f"ffmpeg stdout: {stdout}")
                            progress_dialog.setLabelText(
                                self.tr("ffmpeg failed. Check logs.")
                            )
                            QApplication.processEvents()
                            time.sleep(0.3)
                            progress_dialog.close()
                            popup = Popup(
                                self.tr("ffmpeg failed. Check logs."),
                                self,
                                icon=new_icon_path("warning", "svg"),
                            )
                            popup.show_popup(self, position="center")
                            ffmpeg_failed = True
                        else:
                            logger.info(
                                "ffmpeg command completed successfully."
                            )
                            try:
                                saved_frame_count = len(
                                    [
                                        name
                                        for name in os.listdir(out_dir)
                                        if name.startswith(prefix)
                                        and name.endswith(".jpg")
                                    ]
                                )
                                logger.info(
                                    f"ffmpeg extracted approximately {saved_frame_count} frames to {out_dir}"
                                )
                                progress_dialog.setLabelText(
                                    self.tr(
                                        f"Completed! Extracted {saved_frame_count} frames."
                                    )
                                )
                                QApplication.processEvents()
                                time.sleep(0.3)
                            except Exception as count_e:
                                logger.warning(
                                    f"Could not count extracted frames: {count_e}"
                                )
                                progress_dialog.setLabelText(
                                    self.tr("Completed!")
                                )
                                QApplication.processEvents()
                                time.sleep(0.3)

                    if progress_dialog.isVisible():
                        progress_dialog.close()

                except FileNotFoundError:
                    logger.error(
                        f"ffmpeg command failed: {ffmpeg_path} not found or not executable."
                    )
                    if progress_dialog.isVisible():
                        progress_dialog.close()
                    popup = Popup(
                        self.tr("ffmpeg not found."),
                        self,
                        icon=new_icon_path("error", "svg"),
                    )
                    popup.show_popup(self, position="center")
                    ffmpeg_failed = True
                except Exception as e:
                    logger.exception(
                        f"An error occurred while running ffmpeg: {e}"
                    )
                    if progress_dialog.isVisible():
                        progress_dialog.close()
                    popup = Popup(
                        f"{self.tr('Error running ffmpeg')}: {e}",
                        self,
                        icon=new_icon_path("error", "svg"),
                    )
                    popup.show_popup(self, position="center")
                    ffmpeg_failed = True

                if ffmpeg_failed:
                    return None  # Indicate failure if ffmpeg path failed

            else:  # if not ffmpeg_path
                logger.info("ffmpeg not found. Using OpenCV for extraction.")
                # --- OpenCV Path ---
                estimated_frames = (
                    (total_frames + interval - 1) // interval
                    if total_frames > 0 and interval > 0
                    else 0
                )
                progress_dialog = QProgressDialog(
                    self.tr("Extracting frames (OpenCV)... Please wait..."),
                    self.tr("Cancel"),
                    0,
                    estimated_frames,
                    self,
                )
                progress_dialog.setWindowModality(Qt.WindowModal)
                progress_dialog.setWindowTitle(self.tr("Progress"))
                progress_dialog.setMinimumWidth(400)
                progress_dialog.setMinimumHeight(150)
                progress_dialog.setStyleSheet(
                    get_progress_dialog_style(color="#1d1d1f", height=20)
                )
                progress_dialog.setValue(0)
                progress_dialog.show()

                frame_count = 0
                saved_frame_count = 0
                extraction_cancelled = False
                while True:
                    if progress_dialog.wasCanceled():
                        logger.info(
                            "Frame extraction cancelled by user (OpenCV)."
                        )
                        extraction_cancelled = True
                        break

                    if not video_capture.isOpened():
                        logger.warning(
                            "Video capture became unopened during OpenCV processing."
                        )
                        break

                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    if frame_count % interval == 0:
                        frame_filename = osp.join(
                            out_dir,
                            f"{prefix}{str(saved_frame_count).zfill(seq_len)}.jpg",
                        )
                        try:
                            write_success = cv2.imwrite(frame_filename, frame)
                            if not write_success:
                                logger.error(
                                    f"Failed to write frame: {frame_filename}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error writing frame {frame_filename}: {e}"
                            )

                        saved_frame_count += 1
                        progress_dialog.setValue(saved_frame_count)

                    frame_count += 1
                    QApplication.processEvents()  # Keep UI responsive

                progress_dialog.close()

                if extraction_cancelled:
                    logger.warning(
                        f"Extraction cancelled. Frames saved so far (OpenCV): {saved_frame_count}"
                    )
                    # Decide if cancellation is an error or partial success. Currently returns out_dir.
                else:
                    logger.info(
                        f"OpenCV extraction finished. Saved frames: {saved_frame_count}"
                    )

            # --- Common success return (after ffmpeg or OpenCV) ---
            return out_dir

        # Except block for the *inner* try (extraction phase: ffmpeg or OpenCV)
        except Exception as extraction_e:
            logger.exception(
                f"An unexpected error occurred during frame extraction logic: {extraction_e}"
            )
            popup = Popup(
                f"An unexpected error occurred during extraction: {extraction_e}",
                self,
                icon=new_icon_path("warning", "svg"),
            )
            popup.show_popup(self, position="center")
            return None  # Indicate failure of extraction phase

    # Except block for the *outer* try (opening/setup phase)
    except Exception as opening_e:
        logger.exception(
            f"An unexpected error occurred during video opening/setup: {opening_e}"
        )
        # Use Popup instead of QMessageBox
        popup = Popup(
            f"An error occurred during setup: {opening_e}",
            self,
            icon=new_icon_path("error", "svg"),
        )
        popup.show_popup(self, position="center")
        return None  # Indicate failure

    # Finally block for the *outer* try (always runs)
    finally:
        # Release capture if it exists and is opened (mainly for OpenCV path or if ffmpeg failed early)
        if video_capture is not None and video_capture.isOpened():
            logger.info("Releasing video capture resource.")
            video_capture.release()
        # Clean up the temporary file if created
        if temp_video_path and osp.exists(temp_video_path):
            try:
                logger.debug(
                    f"Removing temporary video file: {temp_video_path}"
                )
                os.remove(temp_video_path)
            except OSError as e:
                logger.error(
                    f"Error removing temporary file {temp_video_path}: {e}"
                )


def open_video_file(self):
    if not self.may_continue():
        return

    filter = "Video Files (*.asf *.avi *.m4v *.mkv *.mov *.mp4 *.mpeg *.mpg *.ts *.wmv);;All Files (*)"
    input_file, _ = QFileDialog.getOpenFileName(
        self,
        self.tr("Open Video file"),
        "",
        filter,
    )

    if not input_file or not osp.exists(input_file):
        logger.warning(
            f"No valid video file selected or file does not exist: {input_file}"
        )
        return

    out_dir = osp.join(
        osp.dirname(input_file), osp.splitext(osp.basename(input_file))[0]
    )

    if osp.exists(out_dir):
        response = QMessageBox()
        response.setIcon(QMessageBox.Warning)
        response.setWindowTitle(self.tr("Warning"))
        response.setText(self.tr("Directory Already Exists"))

        template = (
            "Directory '{}' already exists. Do you want to overwrite it?"
        )
        translated_template = self.tr(template)
        final_text = translated_template.format(osp.basename(out_dir))
        response.setInformativeText(final_text)
        response.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        response.setDefaultButton(QMessageBox.Ok)
        response.setStyleSheet(get_msg_box_style())

        if response.exec_() != QMessageBox.Ok:
            logger.info(
                f"User chose not to overwrite existing directory: {out_dir}"
            )
            return

        logger.info(f"Removing existing directory: {out_dir}")
        try:
            shutil.rmtree(out_dir)
        except OSError as e:
            logger.error(f"Failed to remove directory {out_dir}: {e}")
            popup = Popup(
                f"Failed to remove existing directory: {e}",
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")
            return  # Don't proceed if removal fails

    # Extract frames from video
    logger.info(f"Starting frame extraction for: {input_file} -> {out_dir}")
    result_dir = extract_frames_from_video(self, input_file, out_dir)

    # Check if extraction process indicated success (returned the directory path)
    if result_dir:
        logger.info(
            f"✅ Frame extraction process finished for directory: {result_dir}"
        )
        # Update the canvas only if successful (or partially successful)
        self.import_image_folder(result_dir)
    else:
        logger.warning(
            f"Frame extraction failed or was cancelled for: {input_file}"
        )
        # Optional: Clean up empty output directory if extraction failed completely before starting
        if osp.exists(out_dir) and not os.listdir(out_dir):
            try:
                os.rmdir(out_dir)
                logger.info(f"Removed empty output directory: {out_dir}")
            except OSError as e:
                logger.error(
                    f"Failed to remove empty output directory {out_dir}: {e}"
                )
        elif osp.exists(out_dir):
            logger.info(
                f"Output directory {out_dir} may contain partial results from cancellation or failure."
            )
