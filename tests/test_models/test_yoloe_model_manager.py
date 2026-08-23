from unittest.mock import Mock

import torch
import pytest

from anylabeling.services.auto_labeling.model_manager import ModelManager
from anylabeling.services.auto_labeling.types import AutoLabelingResult


class FakeYoloe:
    def __init__(self):
        self.marks = []
        self.visual_prompt_vpe = None
        self.visual_prompt_classes = []
        self.visual_prompt_reference = None
        self.state = "NO_VISUAL_PROMPT"

    def set_auto_labeling_marks(self, marks):
        self.marks = marks
        self.state = "REFERENCE_MARKS_READY" if marks else "NO_VISUAL_PROMPT"

    def build_visual_prompt(self, source):
        assert source == "reference.jpg"
        self.visual_prompt_vpe = torch.zeros(1, 1, 512)
        self.visual_prompt_classes = ["object"]
        self.visual_prompt_reference = {
            "image_path": source,
            "instance_count": len(self.marks),
        }
        self.marks = []
        self.state = "VISUAL_PROMPT_READY"
        return self.visual_prompt_vpe

    def clear_visual_prompt(self):
        self.visual_prompt_vpe = None
        self.visual_prompt_classes = []
        self.visual_prompt_reference = None
        self.marks = []
        self.state = "NO_VISUAL_PROMPT"

    def has_visual_prompt(self):
        return self.visual_prompt_vpe is not None

    def get_visual_prompt_state(self):
        return self.state


def make_manager(monkeypatch):
    monkeypatch.setattr(ModelManager, "load_model_configs", lambda self: None)
    manager = ModelManager()
    model = FakeYoloe()
    manager.loaded_model_config = {"type": "yoloe", "model": model}
    return manager, model


def test_visual_prompt_status_tracks_marks_build_and_clear(monkeypatch):
    manager, model = make_manager(monkeypatch)
    statuses = []
    messages = []
    manager.visual_prompt_status_changed.connect(statuses.append)
    manager.new_model_status.connect(messages.append)

    manager.set_auto_labeling_marks(
        [{"type": "rectangle", "data": [1, 2, 3, 4]}]
    )
    assert statuses[-1]["state"] == "REFERENCE_MARKS_READY"
    assert statuses[-1]["mark_count"] == 1

    manager.build_visual_prompt(reference_image=True, filename="reference.jpg")
    assert statuses[-1] == {
        "state": "VISUAL_PROMPT_READY",
        "ready": True,
        "classes": ["object"],
        "reference_image": "reference.jpg",
        "instance_count": 1,
        "vpe_shape": [1, 1, 512],
        "profile_name": None,
        "mark_count": 0,
    }
    assert "generated successfully" in messages[-1]

    assert manager.clear_visual_prompt()
    assert statuses[-1]["state"] == "NO_VISUAL_PROMPT"
    assert not statuses[-1]["ready"]
    assert not model.has_visual_prompt()


def test_visual_prompt_actions_reject_non_yoloe_model(monkeypatch):
    monkeypatch.setattr(ModelManager, "load_model_configs", lambda self: None)
    manager = ModelManager()
    manager.loaded_model_config = {"type": "yolov8", "model": Mock()}
    messages = []
    finished = []
    manager.new_model_status.connect(messages.append)
    manager.prediction_finished.connect(lambda: finished.append(True))

    manager.build_visual_prompt(reference_image=True, filename="reference.jpg")

    assert "Load a YOLOE model" in messages[-1]
    assert finished == [True]
    assert not manager.clear_visual_prompt()


def test_batch_visual_prompt_uses_cached_model_without_rebuilding(monkeypatch):
    monkeypatch.setattr(ModelManager, "load_model_configs", lambda self: None)
    manager = ModelManager()
    model = Mock()
    model.has_visual_prompt.return_value = True
    expected = AutoLabelingResult([])
    model.predict_shapes.return_value = expected
    manager.loaded_model_config = {"type": "yoloe", "model": model}

    result = manager.predict_shapes(
        image=True,
        filename="target.jpg",
        batch=True,
        visual_prompt=True,
    )

    assert result is expected
    model.predict_shapes.assert_called_once_with(
        True, "target.jpg", use_visual_prompt=True
    )
    model.build_visual_prompt.assert_not_called()
    model.set_visual_prompt.assert_not_called()


def test_batch_prediction_reraises_per_image_errors(monkeypatch):
    monkeypatch.setattr(ModelManager, "load_model_configs", lambda self: None)
    manager = ModelManager()
    model = Mock()
    model.has_visual_prompt.return_value = True
    model.predict_shapes.side_effect = RuntimeError("damaged target")
    manager.loaded_model_config = {"type": "yoloe", "model": model}

    with pytest.raises(RuntimeError, match="damaged target"):
        manager.predict_shapes(
            image=True,
            filename="broken.jpg",
            batch=True,
            visual_prompt=True,
        )
