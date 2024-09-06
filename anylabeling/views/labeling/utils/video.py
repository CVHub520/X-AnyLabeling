import os
import os.path as osp
import cv2
import shutil

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMessageBox,
    QInputDialog,
    QProgressDialog,
)


def get_output_directory(source_video_path):
    video_dir = os.path.dirname(source_video_path)
    folder_name = os.path.splitext(os.path.basename(source_video_path))[0]
    output_dir = os.path.join(video_dir, folder_name)
    return output_dir


def ask_overwrite_directory(parent, output_dir):
    if os.path.exists(output_dir):
        reply = QMessageBox.question(
            parent,
            "Directory Exists",
            f"The directory '{os.path.basename(output_dir)}' already exists. Do you want to overwrite it?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.No:
            return False
        else:
            shutil.rmtree(output_dir)
    return True


def get_frame_interval(parent, fps, total_frames):
    interval, ok = QInputDialog.getInt(
        parent,
        parent.tr("Frame Interval"),
        parent.tr(f"Enter the frame interval (FPS: {fps}):"),
        1,  # default value
        1,  # minimum value
        total_frames,  # maximum value
        1,  # step
    )
    if not ok:
        QMessageBox.warning(
            parent, "Cancelled", "Frame extraction was cancelled."
        )
    return interval if ok else None


def extract_frames_from_video(parent, source_video_path):
    output_dir = get_output_directory(source_video_path)

    if not ask_overwrite_directory(parent, output_dir):
        return None

    video_capture = cv2.VideoCapture(source_video_path)
    if not video_capture.isOpened():
        QMessageBox.critical(parent, "Error", "Failed to open video file.")
        return None

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    interval = get_frame_interval(parent, fps, total_frames)
    if interval is None:
        return None

    os.makedirs(output_dir)

    progress_dialog = QProgressDialog(
        parent.tr("Extracting frames. Please wait..."),
        parent.tr("Cancel"),
        0,
        total_frames // interval,
        parent,
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle("Progress")
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

    frame_count = 0
    saved_frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_filename = os.path.join(
                output_dir, f"{saved_frame_count:05}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            progress_dialog.setValue(saved_frame_count)
            if progress_dialog.wasCanceled():
                break

        frame_count += 1

    video_capture.release()
    progress_dialog.close()

    return output_dir
