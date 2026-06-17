import importlib.resources as pkg_resources

import numpy as np
import yaml
from PyQt6.QtCore import QFile

import anylabeling.configs as auto_labeling_configs
import anylabeling.resources.resources  # noqa: F401


def test_scrfd_distance_decoders():
    from anylabeling.services.auto_labeling.scrfd import (
        distance2bbox,
        distance2kps,
    )

    points = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    distances = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=np.float32,
    )
    keypoints = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        distance2bbox(points, distances),
        np.array([[9.0, 18.0, 13.0, 24.0], [25.0, 34.0, 37.0, 48.0]]),
    )
    np.testing.assert_allclose(
        distance2kps(points, keypoints),
        np.array([[11.0, 22.0, 13.0, 24.0], [35.0, 46.0, 37.0, 48.0]]),
    )


def test_scrfd_builtin_config_is_available():
    resource_path = pkg_resources.files(auto_labeling_configs).joinpath(
        "auto_labeling", "scrfd_10g_bnkps.yaml"
    )
    config = yaml.safe_load(resource_path.read_text(encoding="utf-8"))

    assert config["type"] == "scrfd"
    assert config["input_width"] == 640
    assert config["input_height"] == 640
    assert (
        config["model_path"]
        == "https://github.com/CVHub520/X-AnyLabeling/releases/download/v3.0.0/scrfd_10g_bnkps.onnx"
    )
    assert config["classes"] == {
        "face": [
            "left_eye",
            "right_eye",
            "nose",
            "left_mouth_corner",
            "right_mouth_corner",
        ]
    }


def test_scrfd_provider_icon_is_compiled_into_qt_resources():
    assert QFile.exists(":/images/images/insightface.png")
