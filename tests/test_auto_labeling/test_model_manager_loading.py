import builtins
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from anylabeling.services.auto_labeling import model_manager
from anylabeling.services.auto_labeling.worker import GenericWorker


@pytest.fixture
def manager():
    return SimpleNamespace(
        loaded_model_config=None,
        loaded_model_config_lock=Lock(),
        model_configs=[{"type": "yolov5", "nested": {"value": 1}}],
        _cancel_event=Event(),
        download_progress=Mock(),
        new_model_status=Mock(),
        auto_segmentation_model_selected=Mock(),
        auto_segmentation_model_unselected=Mock(),
        request_next_files_requested=Mock(),
        tr=lambda text: text,
    )


@pytest.mark.parametrize(
    "model_type, selected, prefetch",
    [
        ("yolov5", False, False),
        ("remote_server", False, False),
        ("florence2", False, False),
        ("geco", False, False),
        ("yolov5_sam", True, True),
        ("yolov8_sam2", True, True),
        ("grounding_sam", True, True),
        ("grounding_sam2", True, True),
        ("open_vision", True, True),
        ("segment_anything", True, True),
        ("segment_anything_2", True, True),
        ("segment_anything_3", True, False),
        ("segment_anything_2_video", True, True),
        ("efficientvit_sam", True, True),
        ("sam_med2d", True, True),
        ("edge_sam", True, True),
        ("sam_hq", True, True),
    ],
)
def test_load_preserves_model_signals(
    manager, monkeypatch, model_type, selected, prefetch
):
    manager.model_configs[0]["type"] = model_type
    model = Mock()
    model_cls = Mock(return_value=model)
    resolve = Mock(return_value=model_cls)
    monkeypatch.setattr(model_manager, "_get_model_class", resolve)

    result = model_manager.ModelManager._load_model(manager, 0)

    resolve.assert_called_once_with(model_type)
    model_cls.assert_called_once_with(
        result, on_message=manager.new_model_status.emit
    )
    assert result is manager.loaded_model_config
    assert result["model"] is model
    assert result["_cancel_event"] is manager._cancel_event
    assert result["nested"] is not manager.model_configs[0]["nested"]
    assert "model" not in manager.model_configs[0]
    result["_on_progress"](10, 100)
    manager.download_progress.emit.assert_called_once_with(10, 100)
    assert manager.auto_segmentation_model_selected.emit.call_count == selected
    assert manager.auto_segmentation_model_unselected.emit.call_count == (
        not selected
    )
    assert manager.request_next_files_requested.emit.call_count == prefetch


@pytest.mark.parametrize("stage", ["import", "construct", "unknown"])
def test_failed_load_reports_error_and_finishes_worker(
    manager, monkeypatch, stage
):
    if stage == "import":
        monkeypatch.setattr(
            model_manager,
            "_get_model_class",
            Mock(side_effect=ModuleNotFoundError("missing dependency")),
        )
    elif stage == "construct":
        monkeypatch.setattr(
            model_manager,
            "_get_model_class",
            Mock(return_value=Mock(side_effect=RuntimeError("load failed"))),
        )
    else:
        manager.model_configs[0]["type"] = "unknown_model"
    worker = GenericWorker(model_manager.ModelManager._load_model, manager, 0)
    finished = Mock()
    worker.finished.connect(finished)

    worker.run()

    finished.assert_called_once_with()
    assert manager.loaded_model_config is None
    manager.new_model_status.emit.assert_called_once()
    assert manager.new_model_status.emit.call_args.args[0].startswith(
        "Error in loading model: "
    )
    manager.auto_segmentation_model_selected.emit.assert_not_called()
    manager.auto_segmentation_model_unselected.emit.assert_not_called()
    manager.request_next_files_requested.emit.assert_not_called()


@pytest.mark.parametrize("model_type", ["florence2", "geco"])
def test_timed_load_keeps_timeout_and_does_not_publish_failure(
    manager, monkeypatch, model_type
):
    manager.model_configs[0]["type"] = model_type
    model_cls = Mock()
    monkeypatch.setattr(
        model_manager, "_get_model_class", Mock(return_value=model_cls)
    )
    context = Mock()
    context.run.side_effect = TimeoutError("loading timed out")
    timeout = Mock()
    timeout.return_value.__enter__ = Mock(return_value=context)
    timeout.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(model_manager, "TimeoutContext", timeout)

    result = model_manager.ModelManager._load_model(manager, 0)

    assert result is None
    assert manager.loaded_model_config is None
    assert timeout.call_args.kwargs["timeout"] == 300
    context.run.assert_called_once()
    model_cls.assert_not_called()
    manager.new_model_status.emit.assert_called_once_with(
        "Error in loading model: loading timed out"
    )


def test_previous_model_is_unloaded_before_constructing(manager, monkeypatch):
    old_model = Mock()
    manager.loaded_model_config = {"model": old_model}

    def construct(config, on_message):
        old_model.unload.assert_called_once_with()
        assert manager.loaded_model_config is None
        manager.auto_segmentation_model_unselected.emit.assert_called_once()
        return Mock()

    monkeypatch.setattr(
        model_manager, "_get_model_class", Mock(return_value=construct)
    )

    result = model_manager.ModelManager._load_model(manager, 0)

    assert manager.loaded_model_config is result
    assert result["model"] is not old_model


@pytest.mark.parametrize(
    "model_type, class_name", [("yolov5", "YOLOv5"), ("geco", "GeCo")]
)
def test_resolver_only_imports_selected_model(
    monkeypatch, model_type, class_name
):
    model_cls = Mock()
    imports = []

    def import_model(name, globals=None, locals=None, fromlist=(), level=0):
        imports.append((name, fromlist, level))
        return SimpleNamespace(**{class_name: model_cls})

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", import_model)
        result = model_manager._get_model_class(model_type)

    assert result is model_cls
    assert imports == [(model_type, (class_name,), 1)]
