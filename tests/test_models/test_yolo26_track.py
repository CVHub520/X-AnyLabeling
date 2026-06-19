import os

import pytest
import yaml

from anylabeling.services.auto_labeling.__base__ import yolo as yolo_module
from anylabeling.services.auto_labeling import model as model_module


class DummyOnnxBaseModel:
    def __init__(self, *args, **kwargs):
        pass

    def get_input_shape(self):
        return [1, 3, 640, 640]

    def get_metadata_info(self, field):
        return "[17, 3]" if field == "kpt_shape" else None


@pytest.fixture
def yolo_without_model(monkeypatch, tmp_path):
    model_file = tmp_path / "model.onnx"
    model_file.touch()
    monkeypatch.setattr(
        yolo_module.YOLO,
        "get_model_abs_path",
        lambda self, config, field: str(model_file),
    )
    monkeypatch.setattr(yolo_module, "OnnxBaseModel", DummyOnnxBaseModel)
    monkeypatch.setattr(model_module, "get_config", lambda: {})
    return yolo_module.YOLO


@pytest.mark.parametrize(
    ("model_type", "expected_task"),
    [
        ("yolo26_det_track", "det"),
        ("yolo26_seg_track", "seg"),
        ("yolo26_obb_track", "obb"),
        ("yolo26_pose_track", "pose"),
    ],
)
def test_yolo26_track_types_use_tracktrack(
    yolo_without_model, model_type, expected_task
):
    classes = {"person": ["nose"]} if expected_task == "pose" else ["person"]
    model = yolo_without_model(
        {
            "type": model_type,
            "name": model_type,
            "display_name": model_type,
            "model_path": "model.onnx",
            "classes": classes,
            "tracker": {
                "tracker_type": "tracktrack",
                "track_high_thresh": 0.6,
                "track_low_thresh": 0.25,
                "new_track_thresh": 0.7,
                "track_buffer": 30,
                "match_thresh": 0.7,
                "lost_match_thr": 0.0,
                "iou_weight": 0.5,
                "conf_weight": 0.1,
                "angle_weight": 0.05,
                "penalty_p": 0.2,
                "reduce_step": 0.05,
                "tai_thr": 0.55,
                "min_track_len": 3,
                "gmc_method": "none",
            },
        },
        on_message=lambda *args, **kwargs: None,
    )

    assert model.task == expected_task
    assert type(model.tracker).__name__ == "TRACKTRACK"


@pytest.mark.parametrize(
    ("config_file", "model_type"),
    [
        ("yolo26s_det_tracktrack.yaml", "yolo26_det_track"),
        ("yolo26s_seg_tracktrack.yaml", "yolo26_seg_track"),
        ("yolo26s_obb_tracktrack.yaml", "yolo26_obb_track"),
        ("yolo26s_pose_tracktrack.yaml", "yolo26_pose_track"),
    ],
)
def test_yolo26_tracktrack_configs_are_registered(config_file, model_type):
    config_path = os.path.join(
        "anylabeling", "configs", "auto_labeling", config_file
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["type"] == model_type
    assert config["tracker"]["tracker_type"] == "tracktrack"

    with open(
        os.path.join("anylabeling", "configs", "models.yaml"),
        "r",
        encoding="utf-8",
    ) as f:
        models = yaml.safe_load(f)

    assert any(item["config_file"] == f":/{config_file}" for item in models)
