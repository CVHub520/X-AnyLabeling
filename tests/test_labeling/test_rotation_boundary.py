import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from PyQt6 import QtCore

from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.batch import save_auto_labeling_result

pytest_plugins = ["tests.test_labeling.test_check_status"]


def rotation():
    shape = Shape(label="object", shape_type="rotation", direction=math.pi / 4)
    shape.points = [
        QtCore.QPointF(x, y)
        for x, y in [(-20, 20), (20, -20), (60, 20), (20, 60)]
    ]
    shape.close()
    return shape


@pytest.mark.parametrize("enabled", [False, True])
def test_auto_rotation_boundary(window, tmp_path, enabled):
    widget = window.labeling_widget.view
    path = str(tmp_path / "image.png")
    Image.new("RGB", (100, 80)).save(path)
    assert widget.load_file(path)
    widget.canvas.reset_out_of_bounds_rotation = enabled
    shape = rotation()
    widget.new_shapes_from_auto_labeling(AutoLabelingResult([shape]))
    assert widget.canvas.shapes[0].shape_type == "rotation"
    assert shape.direction == (0 if enabled else math.pi / 4)
    assert shape.points[0].x() == (0 if enabled else -20)
    assert widget.save_labels(widget.get_label_file())
    data = json.loads((tmp_path / "image.json").read_text())
    assert data["shapes"][0]["direction"] == shape.direction


@pytest.mark.parametrize("enabled", [False, True])
def test_batch_rotation_boundary(tmp_path, enabled):
    path = str(tmp_path / "image.png")
    Image.new("RGB", (100, 80)).save(path)
    widget = SimpleNamespace(
        output_dir=None,
        _config={
            "store_data": False,
            "canvas": {"rotation": {"reset_out_of_bounds": enabled}},
        },
    )
    save_auto_labeling_result(widget, path, AutoLabelingResult([rotation()]))
    data = json.loads((tmp_path / "image.json").read_text())
    shape = data["shapes"][0]
    assert shape["shape_type"] == "rotation"
    assert shape["direction"] == (0 if enabled else math.pi / 4)
    assert shape["points"][0] == ([0, 0] if enabled else [-20, 20])


def test_yolo26_obb_result_follows_saved_setting(window, tmp_path):
    from anylabeling.services.auto_labeling.yolo26_obb import YOLO26_OBB

    widget = window.labeling_widget.view
    path = str(tmp_path / "yolo.png")
    Image.new("RGB", (100, 80)).save(path)
    assert widget.load_file(path)
    model = SimpleNamespace(classes=["plane"], tracker=None)
    manager = widget.auto_labeling_widget.model_manager
    key = "canvas.rotation.reset_out_of_bounds"
    for enabled in (False, True, False):
        widget._settings_controller.update_field(key, enabled)
        widget._settings_controller.save_now()
        assert widget.canvas.reset_out_of_bounds_rotation is enabled
        shape = YOLO26_OBB.create_obb_shape(
            model, np.array([10, 30, 60, 20, math.pi / 4]), 0.9, 0, None
        )
        assert any(widget.canvas.out_off_pixmap(p) for p in shape.points)
        result = AutoLabelingResult([shape])
        result.image_path = path
        manager.new_auto_labeling_result.emit(result)
        assert widget.canvas.shapes == [shape]
        assert (
            any(widget.canvas.out_off_pixmap(p) for p in shape.points)
            is not enabled
        )
        assert (shape.direction == 0) is enabled
        assert shape.shape_type == "rotation"
        assert widget.save_labels(widget.get_label_file())
        saved = json.loads((tmp_path / "yolo.json").read_text())["shapes"][0]
        assert saved["direction"] == shape.direction
