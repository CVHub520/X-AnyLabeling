import json
import os
from contextlib import ExitStack
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PyQt6 import QtWidgets

from anylabeling.views.labeling.utils.export import (
    _add_empty_label_options,
    _export_mask_files,
    _show_yolo_export_error,
    _validate_yolo_export_path,
    export_voc_annotation,
    export_yolo_annotation,
    is_confirmed_empty_label,
    should_skip_empty_export,
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
            QtWidgets.QCheckBox,
            "isChecked",
            lambda checkbox: checkbox.text() == "Save with images?",
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


@pytest.mark.parametrize(
    ("label_data", "expected"),
    [
        ({"checked": True, "shapes": []}, True),
        ({"checked": False, "shapes": []}, False),
        ({"checked": "failed", "shapes": []}, False),
        ({"checked": True, "shapes": [{"label": "cat"}]}, False),
        ({"checked": True}, False),
        ({"checked": 1, "shapes": []}, False),
        (None, False),
        ([], False),
    ],
)
def test_is_confirmed_empty_label(label_data, expected):
    assert is_confirmed_empty_label(label_data) is expected


@pytest.mark.parametrize(
    ("skip_empty", "is_empty", "keep_confirmed", "label_data", "expected"),
    [
        (False, True, True, {"checked": True, "shapes": []}, False),
        (True, False, True, {"checked": True, "shapes": []}, False),
        (True, True, False, {"checked": True, "shapes": []}, True),
        (True, True, True, {"checked": True, "shapes": []}, False),
        (True, True, True, {"checked": False, "shapes": []}, True),
        (True, True, True, {"checked": "failed", "shapes": []}, True),
        (
            True,
            True,
            True,
            {"checked": True, "shapes": [{"label": "cat"}]},
            True,
        ),
        (True, True, True, None, True),
    ],
)
def test_should_skip_empty_export(
    skip_empty, is_empty, keep_confirmed, label_data, expected
):
    assert (
        should_skip_empty_export(
            skip_empty, is_empty, keep_confirmed, label_data
        )
        is expected
    )


def test_keep_confirmed_empty_option_follows_skip_empty():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = QtWidgets.QDialog()
    layout = QtWidgets.QVBoxLayout(dialog)
    skip_empty, keep_confirmed = _add_empty_label_options(
        layout, lambda text: text
    )

    assert skip_empty.text() == "Skip empty labels?"
    assert keep_confirmed.text() == "Keep confirmed empty labels?"
    assert "Check Passed" in keep_confirmed.toolTip()
    assert not keep_confirmed.isEnabled()
    assert not keep_confirmed.isChecked()

    skip_empty.setChecked(True)
    assert keep_confirmed.isEnabled()
    keep_confirmed.setChecked(True)
    skip_empty.setChecked(False)
    assert not keep_confirmed.isEnabled()
    assert keep_confirmed.isChecked()

    dialog.close()
    app.processEvents()


def _checkbox_values(save_images, skip_empty, keep_confirmed):
    values = {
        "Save with images?": save_images,
        "Skip empty labels?": skip_empty,
        "Keep confirmed empty labels?": keep_confirmed,
    }

    def is_checked(checkbox):
        return values.get(checkbox.text(), False)

    return is_checked


def _write_xlabel(path, shapes, checked=False):
    path.write_text(
        json.dumps(
            {
                "imagePath": path.with_suffix(".png").name,
                "imageWidth": 20,
                "imageHeight": 20,
                "checked": checked,
                "shapes": shapes,
            }
        ),
        encoding="utf-8",
    )


def _prepare_export_samples(image_dir, write_image):
    rectangle = {
        "label": "cat",
        "shape_type": "rectangle",
        "points": [[1, 1], [10, 1], [10, 10], [1, 10]],
    }
    samples = [
        ("pos", [rectangle], True),
        ("confirmed", [], True),
        ("unchecked", [], False),
        ("failed", [], "failed"),
        ("missing", None, None),
    ]
    image_files = []
    for name, shapes, checked in samples:
        image_file = image_dir / f"{name}.png"
        write_image(image_file)
        image_files.append(str(image_file))
        if shapes is not None:
            _write_xlabel(image_file.with_suffix(".json"), shapes, checked)
    return image_files


def _run_annotation_export(
    widget,
    export_fn,
    mode,
    classes_file,
    save_images,
    skip_empty,
    keep_confirmed,
):
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                QtWidgets.QCheckBox,
                "isChecked",
                _checkbox_values(save_images, skip_empty, keep_confirmed),
            )
        )
        stack.enter_context(
            mock.patch.object(QtWidgets.QDialog, "exec", return_value=1)
        )
        stack.enter_context(
            mock.patch("anylabeling.views.labeling.utils.export.Popup")
        )
        if classes_file is not None:
            stack.enter_context(
                mock.patch.object(
                    QtWidgets.QFileDialog,
                    "getOpenFileName",
                    return_value=(classes_file, ""),
                )
            )
        export_fn(widget, mode)


@pytest.mark.parametrize(
    ("skip_empty", "keep_confirmed", "saved"),
    [
        (
            False,
            False,
            {"pos", "confirmed", "unchecked", "failed", "missing"},
        ),
        (True, False, {"pos"}),
        (True, True, {"pos", "confirmed"}),
    ],
)
def test_yolo_export_keeps_confirmed_empty_labels(
    tmp_path, skip_empty, keep_confirmed, saved
):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_files = _prepare_export_samples(
        image_dir, lambda path: path.touch()
    )
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\n", encoding="utf-8")

    widget = QtWidgets.QWidget()
    widget.filename = image_files[0]
    widget.image_list = image_files
    widget.last_open_dir = str(image_dir)
    widget.output_dir = None
    widget.may_continue = mock.Mock(return_value=True)

    _run_annotation_export(
        widget,
        export_yolo_annotation,
        "hbb",
        str(classes_file),
        save_images=True,
        skip_empty=skip_empty,
        keep_confirmed=keep_confirmed,
    )

    save_path = tmp_path / "labels"
    for name in ("pos", "confirmed", "unchecked", "failed", "missing"):
        label_file = save_path / f"{name}.txt"
        image_file = save_path / f"{name}.png"
        assert label_file.is_file() is (name in saved)
        assert image_file.is_file() is (name in saved)
        if name == "pos" and name in saved:
            assert label_file.read_text(encoding="utf-8").strip()
        if name == "confirmed" and name in saved:
            assert label_file.read_text(encoding="utf-8") == ""

    widget.close()
    app.processEvents()


def test_voc_export_keeps_confirmed_empty_labels(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    def write_image(path):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        assert cv2.imwrite(str(path), image)

    image_files = _prepare_export_samples(image_dir, write_image)
    widget = QtWidgets.QWidget()
    widget.filename = image_files[0]
    widget.image_list = image_files
    widget.output_dir = None
    widget.may_continue = mock.Mock(return_value=True)

    _run_annotation_export(
        widget,
        export_voc_annotation,
        "rectangle",
        None,
        save_images=True,
        skip_empty=True,
        keep_confirmed=True,
    )

    save_path = tmp_path / "Annotations"
    kept = {"pos", "confirmed"}
    for name in ("pos", "confirmed", "unchecked", "failed", "missing"):
        label_file = save_path / f"{name}.xml"
        image_file = save_path / f"{name}.png"
        assert label_file.is_file() is (name in kept)
        assert image_file.is_file() is (name in kept)
    positive = (save_path / "pos.xml").read_text(encoding="utf-8")
    confirmed = (save_path / "confirmed.xml").read_text(encoding="utf-8")
    assert "<name>cat</name>" in positive
    assert "<object>" not in confirmed

    widget.close()
    app.processEvents()


def test_confirmed_empty_label_translations():
    from PyQt6.QtCore import QTranslator

    import anylabeling.resources.resources  # noqa: F401

    labels = {
        "en_US": "Keep confirmed empty labels?",
        "zh_CN": "保持导出已确认的空标签？",
        "ja_JP": "確認済みの空ラベルも書き出しますか？",
        "ko_KR": "확인된 빈 레이블도 내보내시겠습니까?",
    }
    tooltips = {
        "en_US": (
            "When empty labels are skipped, still export images marked "
            "Check Passed that contain no shapes."
        ),
        "zh_CN": "跳过空标签时，仍导出已标记为合格且没有任何形状的图片。",
        "ja_JP": "空のラベルをスキップする場合でも、合格で形状のない画像は書き出します。",
        "ko_KR": "빈 레이블을 건너뛰더라도 합격이고 도형이 없는 이미지는 내보냅니다.",
    }
    tooltip_source = (
        "When empty labels are skipped, still export images marked "
        "Check Passed that contain no shapes."
    )
    for language, label in labels.items():
        translator = QTranslator()
        assert translator.load(f":/languages/translations/{language}.qm")
        assert (
            translator.translate(
                "LabelingWidget", "Keep confirmed empty labels?"
            )
            == label
        )
        assert (
            translator.translate("LabelingWidget", tooltip_source) == tooltips[language]
        )
