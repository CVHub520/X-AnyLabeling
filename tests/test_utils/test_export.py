import os
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6 import QtWidgets

from anylabeling.views.labeling.utils.export import (
    _export_mask_files,
    _show_yolo_export_error,
    _validate_yolo_export_path,
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
    converter.export_results = {}
    converter._export_inputs = {}
    converter._input_fingerprints.return_value = {}
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
    converter.export_results = {}
    converter._export_inputs = {}
    converter._input_fingerprints.return_value = {}
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
    converter.export_results = {}
    converter._export_inputs = {}
    converter._input_fingerprints.return_value = {}
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
    converter.export_results = {}
    converter._export_inputs = {}
    converter._input_fingerprints.return_value = {}
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
    converter.export_results = {}
    converter._export_inputs = {}
    converter._input_fingerprints.return_value = {}
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


@pytest.mark.parametrize("mode", ["hbb", "obb", "seg", "pose"])
def test_yolo_export_preserves_nested_directories(tmp_path, mode):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_dir = tmp_path / "images"
    first_image = image_dir / "first" / "image.png"
    second_image = image_dir / "second" / "image.png"
    for image_file in (first_image, second_image):
        image_file.parent.mkdir(parents=True, exist_ok=True)
        image_file.touch()
        image_file.with_suffix(".json").touch()
    config_file = tmp_path / ("pose.yaml" if mode == "pose" else "classes.txt")
    config_file.write_text(
        "classes:\n  person:\n    - nose\n" if mode == "pose" else "person\n",
        encoding="utf-8",
    )

    widget = QtWidgets.QWidget()
    widget.filename = str(first_image)
    widget.image_list = [str(first_image), str(second_image)]
    widget.last_open_dir = str(image_dir)
    widget.output_dir = None
    widget.may_continue = mock.Mock(return_value=True)
    converter = mock.Mock()
    converter.export_results = {}
    converter._export_inputs = {}
    converter._input_fingerprints.return_value = {}
    converter.custom_to_yolo.return_value = False
    converter.read_json.return_value = {
        "imageWidth": 100,
        "imageHeight": 100,
        "shapes": [],
    }

    with (
        mock.patch.object(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            return_value=(str(config_file), ""),
        ),
        mock.patch.object(QtWidgets.QDialog, "exec", return_value=1),
        mock.patch.object(
            QtWidgets.QCheckBox, "isChecked", side_effect=(True, False)
        ),
        mock.patch(
            "anylabeling.views.labeling.utils.export.LabelConverter",
            return_value=converter,
        ),
        mock.patch("anylabeling.views.labeling.utils.export.Popup"),
    ):
        export_yolo_annotation(widget, mode)

    save_path = tmp_path / "labels"
    expected_calls = [
        mock.call(
            str(first_image.with_suffix(".json")),
            str(save_path / "first" / "image.txt"),
            mode,
            skip_empty_files=False,
            obb_boundary_policy="keep",
        ),
        mock.call(
            str(second_image.with_suffix(".json")),
            str(save_path / "second" / "image.txt"),
            mode,
            skip_empty_files=False,
            obb_boundary_policy="keep",
        ),
    ]
    assert converter.custom_to_yolo.call_args_list == expected_calls
    assert (save_path / "first").is_dir()
    assert (save_path / "second").is_dir()
    assert (save_path / "first" / "image.png").is_file()
    assert (save_path / "second" / "image.png").is_file()
    widget.close()
    app.processEvents()


def test_yolo_export_rejects_conflicting_label_paths(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    first_image = image_dir / "image.jpg"
    second_image = image_dir / "image.png"
    first_image.touch()
    second_image.touch()
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("person\n", encoding="utf-8")

    widget = QtWidgets.QWidget()
    widget.filename = str(first_image)
    widget.image_list = [str(first_image), str(second_image)]
    widget.last_open_dir = str(image_dir)
    widget.output_dir = None
    widget.may_continue = mock.Mock(return_value=True)
    converter = mock.Mock()
    converter.export_results = {}
    converter._export_inputs = {}
    converter._input_fingerprints.return_value = {}

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
        export_yolo_annotation(widget, "hbb")

    show_export_error.assert_called_once()
    parent, image_file, error = show_export_error.call_args.args
    assert parent is widget
    assert image_file is None
    assert isinstance(error, ValueError)
    assert str(first_image) in str(error)
    assert str(second_image) in str(error)
    converter.custom_to_yolo.assert_not_called()
    assert not (tmp_path / "labels").exists()
    widget.close()
    app.processEvents()


@pytest.mark.parametrize("relative_path", [".", "nested", ".."])
def test_yolo_export_rejects_paths_overlapping_source(tmp_path, relative_path):
    source_root = tmp_path / "images"
    source_root.mkdir()
    save_path = source_root / relative_path

    with pytest.raises(ValueError):
        _validate_yolo_export_path(str(source_root), str(save_path))


def test_yolo_export_accepts_sibling_path(tmp_path):
    source_root = tmp_path / "images"
    source_root.mkdir()

    _validate_yolo_export_path(str(source_root), str(tmp_path / "labels"))


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
        (RuntimeError("Unexpected error"), "Reason: Unexpected error"),
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
