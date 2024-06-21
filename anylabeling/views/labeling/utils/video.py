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


def extract_frames_from_video(parent, source_video_path):
    # Get the directory of the source video file
    video_dir = osp.dirname(source_video_path)

    # Create a folder in the current directory with the current video file_name
    folder_name = osp.splitext(osp.basename(source_video_path))[0]
    output_dir = osp.join(video_dir, folder_name)

    if osp.exists(output_dir):
        # Ask the user if they want to overwrite the existing directory
        reply = QMessageBox.question(
            parent,
            "Directory Exists",
            f"The directory '{folder_name}' already exists. Do you want to overwrite it?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            # If no, return None
            return None
        else:
            # If yes, delete the existing directory first
            shutil.rmtree(output_dir)

    # Open the video file
    video_capture = cv2.VideoCapture(source_video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Ask the user to input the frame interval
    interval, ok = QInputDialog.getInt(
        parent,
        parent.tr("Frame Interval"),
        parent.tr(f"Enter the frame interval (FPS: {fps}):"),
        1,  # default value
        1,  # minimum value
        total_frames,  # maximum value
        1  # step
    )

    if not ok:
        # If the user cancels the dialog, show a message box and return
        QMessageBox.warning(parent, "Cancelled", "Frame extraction was cancelled.")
        return None

    os.makedirs(output_dir)

    # Decode the video and save frames to the created folder
    progress_dialog = QProgressDialog(
        parent.tr("Extracting frames. Please wait..."),
        parent.tr("Cancel"),
        0,
        total_frames // interval
    )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle("Progress")
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

    frame_count = 0
    saved_frame_count = 0
    base_name = osp.splitext(osp.basename(source_video_path))[0]
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_filename = osp.join(
                output_dir, f"{base_name}-{saved_frame_count}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            progress_dialog.setValue(saved_frame_count)
            if progress_dialog.wasCanceled():
                break

        frame_count += 1

    video_capture.release()
    progress_dialog.close()

    # Return the path of the created folder
    return output_dir