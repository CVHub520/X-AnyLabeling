import os

import pytest
import yaml

from anylabeling.services.auto_labeling.__base__ import yolo as yolo_module
from anylabeling.services.auto_labeling import model as model_module


class DummyOnnxBaseModel:
    kpt_shape = "[17, 3]"

    def __init__(self, *args, **kwargs):
        pass

    def get_input_shape(self):
        return [1, 3, 640, 640]

    def get_metadata_info(self, field):
        return self.kpt_shape if field == "kpt_shape" else None


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
    "metadata",
    [
        "__import__('builtins').sum([20, 22])",
        "[17]",
        "[0, 3]",
        "[17, 4]",
        "[17, 3.0]",
        "[True, 3]",
    ],
)
def test_pose_rejects_invalid_kpt_shape_metadata(
    yolo_without_model, monkeypatch, metadata
):
    monkeypatch.setattr(DummyOnnxBaseModel, "kpt_shape", metadata)

    with pytest.raises(ValueError) as error:
        yolo_without_model(
            {
                "type": "yolo26_pose",
                "name": "yolo26_pose",
                "display_name": "YOLO26 Pose",
                "model_path": "model.onnx",
                "classes": {"person": ["nose"]},
            },
            on_message=lambda *args, **kwargs: None,
        )

    assert "kpt_shape" in str(error.value)
    assert "model.onnx" in str(error.value)


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


@pytest.mark.parametrize("end2end", [True, False])
@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "_seg",
        "_pose",
        "_obb",
        "_det_track",
        "_seg_track",
        "_pose_track",
        "_obb_track",
    ],
)
def test_yolo26_output_formats(yolo_without_model, end2end, suffix):
    import numpy as np

    model_type = "yolo26" + suffix
    classes = {"person": ["nose"]} if "pose" in suffix else ["person"]
    model = yolo_without_model(
        {
            "type": model_type,
            "name": model_type,
            "display_name": model_type,
            "model_path": "model.onnx",
            "classes": classes,
            "end2end": end2end,
        },
        on_message=lambda *args: None,
    )
    model.img_height = model.img_width = 640
    model.image_shape = (640, 640, 3)
    extra = {
        "det": [],
        "seg": [1.0] * 32,
        "pose": [320.0, 320.0, 1.0] * 17,
        "obb": [0.0],
    }[model.task]
    if end2end:
        box = (
            [320, 320, 100, 100]
            if model.task == "obb"
            else [270, 270, 370, 370]
        )
        predictions = np.array(
            [[box + [0.9, 0] + extra] * 2], dtype=np.float32
        )
    else:
        predictions = (
            np.array(
                [[[320, 320, 100, 100, 0.9] + extra] * 2], dtype=np.float32
            )
            .transpose(0, 2, 1)
            .copy()
        )
    outputs = [predictions]
    if model.task == "seg":
        outputs.append(np.ones((1, 32, 160, 160), dtype=np.float32))
    boxes, classes, scores, masks, keypoints = model.postprocess(outputs)
    count = 2 if end2end else 1
    assert len(boxes) == count
    np.testing.assert_allclose(scores, 0.9)
    np.testing.assert_equal(classes, 0)
    if model.task == "seg":
        assert masks.shape == (count, 640, 640)
    if model.task == "pose":
        assert keypoints.shape == (count, 17, 3)


@pytest.mark.parametrize(
    ("metadata", "override", "expected"),
    [
        (None, None, True),
        ("True", None, True),
        ("False", None, False),
        ("False", True, True),
        ("True", False, False),
    ],
)
def test_end2end_metadata_and_override(metadata, override, expected):
    from types import SimpleNamespace
    from anylabeling.services.auto_labeling.utils import resolve_end2end

    net = SimpleNamespace(get_metadata_info=lambda key: metadata)
    assert resolve_end2end(net, override) is expected


@pytest.mark.parametrize("end2end", [True, False])
def test_yolo26_sahi_output_formats(end2end):
    import numpy as np
    from anylabeling.services.auto_labeling.utils.sahi.models.yolo26_onnx import (
        Yolo26ONNX,
    )

    model = Yolo26ONNX.__new__(Yolo26ONNX)
    model.end2end = end2end
    model.input_shape = (640, 640)
    model.conf_thres, model.iou_thres, model.max_det = 0.25, 0.45, 300
    if end2end:
        output = np.array(
            [[[270, 270, 370, 370, 0.9, 0]] * 2], dtype=np.float32
        )
    else:
        output = (
            np.array([[[320, 320, 100, 100, 0.9]] * 2], dtype=np.float32)
            .transpose(0, 2, 1)
            .copy()
        )
    boxes, scores, classes = model.postprocess([output], (640, 640))
    assert len(boxes) == (2 if end2end else 1)
    np.testing.assert_allclose(boxes[0], [270, 270, 370, 370])
