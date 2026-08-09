import os
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6 import QtWidgets

from anylabeling.views.labeling.utils.export import (
    _export_mask_files,
    _show_yolo_export_error,
    export_yolo_annotation,
)
from anylabeling.views.labeling.label_converter import (
    PoseClassError,
    PoseGroupError,
)


@pytest.mark.parametrize(
    ("checked", "expected"),
    [
        (True, True),
        (False, False),
        ("false", False),
        (1, False),
        (None, False),
    ],
)
def test_export_mask_files_only_accepts_checked_true(
    tmp_path, checked, expected
):
    image_file = tmp_path / "image.png"
    label_file = tmp_path / "image.json"
    image_file.touch()
    label_file.touch()
    converter = mock.Mock()
    converter.read_json.return_value = {"checked": checked}
    progress_dialog = mock.Mock()

    _export_mask_files(
        converter,
        [str(image_file)],
        None,
        str(tmp_path / "masks"),
        {"type": "grayscale", "colors": {}},
        include_null_images=False,
        only_checked_images=True,
        progress_dialog=progress_dialog,
    )

    assert converter.custom_to_mask.called is expected


@pytest.mark.parametrize(
    ("include_null_images", "only_checked_images", "expected"),
    [
        (False, False, False),
        (True, False, True),
        (True, True, False),
    ],
)
def test_export_mask_files_handles_images_without_labels(
    tmp_path, include_null_images, only_checked_images, expected
):
    image_file = tmp_path / "image.png"
    image_file.touch()
    converter = mock.Mock()
    progress_dialog = mock.Mock()

    _export_mask_files(
        converter,
        [str(image_file)],
        None,
        str(tmp_path / "masks"),
        {"type": "grayscale", "colors": {}},
        include_null_images=include_null_images,
        only_checked_images=only_checked_images,
        progress_dialog=progress_dialog,
    )

    assert converter.custom_image_to_empty_mask.called is expected


def test_export_mask_files_stops_after_cancellation(tmp_path):
    image_files = [tmp_path / "first.png", tmp_path / "second.png"]
    for image_file in image_files:
        image_file.touch()
    converter = mock.Mock()
    progress_dialog = mock.Mock()
    progress_dialog.wasCanceled.return_value = True

    _export_mask_files(
        converter,
        [str(image_file) for image_file in image_files],
        None,
        str(tmp_path / "masks"),
        {"type": "grayscale", "colors": {}},
        include_null_images=True,
        only_checked_images=False,
        progress_dialog=progress_dialog,
    )

    converter.custom_image_to_empty_mask.assert_called_once()
    progress_dialog.setValue.assert_called_once_with(1)


def test_yolo_export_does_not_blame_last_image_for_popup_error(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_file = image_dir / "image.png"
    image_file.touch()
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("person\n", encoding="utf-8")

    widget = QtWidgets.QWidget()
    widget.filename = str(image_file)
    widget.image_list = [str(image_file)]
    widget.output_dir = str(image_dir)
    widget.may_continue = mock.Mock(return_value=True)
    converter = mock.Mock()
    converter.custom_to_yolo.return_value = False
    popup = mock.Mock()
    popup.show_popup.side_effect = RuntimeError("Popup failed")

    with (
        mock.patch.object(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            return_value=(str(classes_file), ""),
        ),
        mock.patch.object(QtWidgets.QDialog, "exec", return_value=1),
        mock.patch(
            "anylabeling.views.labeling.utils.export.LabelConverter",
            return_value=converter,
        ),
        mock.patch(
            "anylabeling.views.labeling.utils.export.Popup",
            return_value=popup,
        ),
        mock.patch(
            "anylabeling.views.labeling.utils.export._show_yolo_export_error"
        ) as show_export_error,
    ):
        export_yolo_annotation(widget, "hbb")

    show_export_error.assert_called_once()
    parent, image_file, error = show_export_error.call_args.args
    assert parent is widget
    assert image_file is None
    assert isinstance(error, RuntimeError)
    widget.close()
    app.processEvents()


def test_yolo_export_reports_failed_image(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    current_image = image_dir / "current.png"
    failed_image = image_dir / "failed.png"
    current_image.touch()
    failed_image.touch()
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("person\n", encoding="utf-8")

    widget = QtWidgets.QWidget()
    widget.filename = str(current_image)
    widget.image_list = [str(failed_image)]
    widget.output_dir = str(image_dir)
    widget.may_continue = mock.Mock(return_value=True)
    converter = mock.Mock()
    converter.custom_to_yolo.side_effect = PoseGroupError(
        "group_id is None for pose annotation"
    )

    with (
        mock.patch.object(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            return_value=(str(classes_file), ""),
        ),
        mock.patch.object(QtWidgets.QDialog, "exec", return_value=1),
        mock.patch(
            "anylabeling.views.labeling.utils.export.LabelConverter",
            return_value=converter,
        ),
        mock.patch(
            "anylabeling.views.labeling.utils.export._show_yolo_export_error"
        ) as show_export_error,
    ):
        export_yolo_annotation(widget, "pose")

    show_export_error.assert_called_once_with(
        widget, str(failed_image), converter.custom_to_yolo.side_effect
    )
    widget.close()
    app.processEvents()


@pytest.mark.parametrize(
    ("error", "guidance"),
    [
        (
            PoseGroupError("Invalid pose group"),
            "Reason: Pose instance grouping is incomplete or mismatched.\n"
            "Please ensure that each instance has one bounding box and that "
            "its bounding box and keypoints use the same numeric group ID.",
        ),
        (
            PoseClassError("Invalid pose class"),
            "Reason: The bounding box label is not defined in the pose "
            "configuration.\nPlease ensure that the bounding box label is "
            "listed under classes in the pose YAML file.",
        ),
        (RuntimeError("Unexpected error"), None),
    ],
)
def test_yolo_export_error_dialog_shows_actionable_guidance(
    tmp_path, error, guidance
):
    current_image = tmp_path / "current.png"
    failed_image = tmp_path / "failed.png"
    widget = mock.Mock()
    widget.filename = str(current_image)
    message_box = mock.Mock()

    with mock.patch(
        "anylabeling.views.labeling.utils.export.QtWidgets.QMessageBox",
        return_value=message_box,
    ) as message_box_class:
        _show_yolo_export_error(widget, str(failed_image), error)

    message_box_class.assert_called_once_with(widget)
    message_box.setWindowTitle.assert_called_once_with("Export Failed")
    expected_message = f"Failed on image: {failed_image}"
    if guidance:
        expected_message += f"\n\n{guidance}"
    message_box.setText.assert_called_once_with(expected_message)
    message_box.addButton.assert_called_once_with(
        message_box_class.StandardButton.Ok
    )
    widget.load_file.assert_called_once_with(str(failed_image))
