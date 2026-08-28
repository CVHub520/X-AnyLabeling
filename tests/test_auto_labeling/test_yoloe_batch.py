import json
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt6 import QtWidgets

from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.views.labeling.utils import batch


def make_app(tmp_path, manager, image_names):
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "labels"
    input_dir.mkdir()
    output_dir.mkdir()
    image_paths = []
    for name in image_names:
        path = input_dir / name
        Image.new("RGB", (32, 24), "white").save(path)
        image_paths.append(str(path))
    return (
        SimpleNamespace(
            auto_labeling_widget=SimpleNamespace(model_manager=manager),
            image=True,
            output_dir=str(output_dir),
            _config={"store_data": False},
            cancel_processing=False,
            image_index=0,
        ),
        image_paths,
        output_dir,
    )


def test_visual_prompt_batch_continues_after_broken_image(tmp_path):
    calls = []

    def predict(_image, filename, **kwargs):
        calls.append((filename, kwargs))
        if filename.endswith("broken.jpg"):
            raise RuntimeError("broken image")
        return AutoLabelingResult([], replace=True)

    manager = Mock()
    manager.predict_shapes.side_effect = predict
    app, images, output_dir = make_app(
        tmp_path, manager, ["one.jpg", "broken.jpg", "three.jpg"]
    )
    broken_label = output_dir / "broken.json"
    broken_label.write_text('{"sentinel": true}', encoding="utf-8")

    progress = []
    failures = []
    finished = []
    worker = batch.BatchProcessingThread(
        app,
        images,
        0,
        "yoloe",
        "",
        False,
        False,
        visual_prompt=True,
    )
    worker.progress_updated.connect(
        lambda value, _label: progress.append(value)
    )
    worker.image_failed.connect(
        lambda filename, message: failures.append((filename, message))
    )
    worker.processing_finished.connect(
        lambda count, cancelled: finished.append((count, cancelled))
    )

    worker.run()

    assert len(calls) == 3
    assert all(call[1]["visual_prompt"] for call in calls)
    assert progress == [1, 2, 3]
    assert finished == [(1, False)]
    assert failures == [(images[1], "broken image")]
    assert (output_dir / "one.json").is_file()
    assert (output_dir / "three.json").is_file()
    assert json.loads(broken_label.read_text(encoding="utf-8")) == {
        "sentinel": True
    }


def test_visual_prompt_batch_honors_cancel_before_next_image(tmp_path):
    manager = Mock()
    app, images, _output_dir = make_app(tmp_path, manager, ["one.jpg"])
    app.cancel_processing = True
    finished = []
    worker = batch.BatchProcessingThread(
        app,
        images,
        0,
        "yoloe",
        "",
        False,
        False,
        visual_prompt=True,
    )
    worker.processing_finished.connect(
        lambda count, cancelled: finished.append((count, cancelled))
    )

    worker.run()

    manager.predict_shapes.assert_not_called()
    assert finished == [(0, True)]


def test_visual_prompt_batch_stops_between_images_after_cancel(tmp_path):
    manager = Mock()
    app, images, _output_dir = make_app(
        tmp_path, manager, ["one.jpg", "two.jpg"]
    )

    def predict(*_args, **_kwargs):
        app.cancel_processing = True
        return AutoLabelingResult([])

    manager.predict_shapes.side_effect = predict
    progress = []
    finished = []
    worker = batch.BatchProcessingThread(
        app,
        images,
        0,
        "yoloe",
        "",
        False,
        False,
        visual_prompt=True,
    )
    worker.progress_updated.connect(
        lambda value, _label: progress.append(value)
    )
    worker.processing_finished.connect(
        lambda count, cancelled: finished.append((count, cancelled))
    )

    worker.run()

    assert manager.predict_shapes.call_count == 1
    assert progress == [1]
    assert finished == [(0, True)]


def test_yoloe_batch_dialog_prefers_ready_visual_prompt():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = batch.YoloeBatchPromptDialog(visual_prompt_ready=True)
    try:
        assert dialog.mode_combo.currentData() == dialog.VISUAL_PROMPT
        assert not dialog.text_input.isEnabled()
        dialog.mode_combo.setCurrentIndex(
            dialog.mode_combo.findData(dialog.TEXT_PROMPT)
        )
        app.processEvents()
        assert dialog.text_input.isEnabled()
    finally:
        dialog.close()


def test_run_all_images_visual_prompt_starts_at_folder_beginning():
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    model = Mock()
    model.has_visual_prompt.return_value = True
    manager = Mock()
    manager.loaded_model_config = {"type": "yoloe", "model": model}
    app = QtWidgets.QWidget()
    app._batch_processing_active = False
    app.image_list = ["one.jpg", "two.jpg", "three.jpg"]
    app.filename = "three.jpg"
    app.fn_to_index = {"one.jpg": 0, "two.jpg": 1, "three.jpg": 2}
    app.auto_labeling_widget = SimpleNamespace(model_manager=manager)
    selection = (batch.YoloeBatchPromptDialog.VISUAL_PROMPT, "")

    with (
        patch.object(
            batch.QtWidgets.QMessageBox,
            "exec",
            return_value=batch.QtWidgets.QMessageBox.StandardButton.Ok,
        ),
        patch.object(
            batch.YoloeBatchPromptDialog,
            "get_selection",
            return_value=selection,
        ),
        patch.object(batch, "_start_batch_processing") as start,
    ):
        batch.run_all_images(app)

    assert app.current_index == 2
    assert app.image_index == 0
    assert app.batch_visual_prompt
    start.assert_called_once_with(app)
    app.close()
    qt_app.processEvents()
