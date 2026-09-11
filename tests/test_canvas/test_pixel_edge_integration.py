"""Exercise real window, canvas, selection, undo and configurable actions."""

import os
import time
import json

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

import anylabeling.resources.resources  # noqa: F401
from anylabeling import config
from anylabeling.views.mainwindow import MainWindow
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.cell_raster import fill_cell_polygon
from anylabeling.views.labeling.utils.pixel_cell_edges import (
    segment_box_to_edge_polygon,
)


@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(config, "_work_directory", str(tmp_path))
    old = QtCore.QSettings.defaultFormat()
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(
        QtCore.QSettings.Format.IniFormat,
        QtCore.QSettings.Scope.UserScope,
        str(tmp_path),
    )
    settings = config.get_default_config()
    monkeypatch.setattr(
        config, "current_config_file", str(tmp_path / ".xanylabelingrc")
    )
    settings["auto_save"] = False
    settings["canvas"]["edge_refinement"].update(
        search_radius=4, point_spacing=100
    )
    win = MainWindow(app, config=settings)
    view = win.labeling_widget.view
    image = QtGui.QImage(80, 50, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor(220, 220, 220))
    painter = QtGui.QPainter(image)
    painter.fillRect(10, 10, 20, 20, QtGui.QColor(100, 100, 100))
    painter.fillRect(45, 10, 20, 20, QtGui.QColor(100, 100, 100))
    painter.end()
    view.image = image
    view.filename = str(tmp_path / "image.png")
    view.image_path = view.filename
    image.save(view.filename)
    view.canvas.load_pixmap(QtGui.QPixmap.fromImage(image))
    view.canvas.set_editing(True)
    for offset in (0, 35):
        shape = Shape(label="object", shape_type="polygon")
        shape.points = [
            QtCore.QPointF(x + offset, y)
            for x, y in ((8, 8), (31, 8), (31, 31), (8, 31))
        ]
        shape.close()
        view.canvas.shapes.append(shape)
        view.add_label(shape)
    view.canvas.store_shapes()
    yield view, app
    view.cancel_pixel_edge_preview(switch_mode=False)
    view.dirty = False
    win.deleteLater()
    app.processEvents()
    QtCore.QSettings.setDefaultFormat(old)


def wait_preview(view, app):
    deadline = time.monotonic() + 10
    while (
        view._edge_existing_request is not None and time.monotonic() < deadline
    ):
        app.processEvents()
        time.sleep(0.005)
    assert (
        view._edge_existing_preview is not None
    ), view.pixel_edge_widget.status_label.text()


def wait_batch_preview(view, app):
    deadline = time.monotonic() + 10
    while view._edge_batch_request is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert (
        view._edge_batch_preview is not None
    ), view.pixel_edge_widget.status_label.text()


def test_confirm_a_then_cancel_b_keeps_a_and_has_no_transient_line(window):
    view, app = window
    a, b = view.canvas.shapes
    original_a, original_b = list(a.points), list(b.points)
    view.refine_existing_shape_to_edge(a)
    wait_preview(view, app)
    assert a.points == original_a
    action = view._settings_runtime_applier._shortcut_action_map[
        "shortcuts.confirm_pixel_edge"
    ]
    view._settings_runtime_applier.apply_shortcuts(
        "shortcuts.confirm_pixel_edge", "Alt+Return"
    )
    assert action.shortcut() == QtGui.QKeySequence("Alt+Return")
    action.trigger()
    committed = list(a.points)
    assert committed != original_a
    coordinates = np.array([[p.x(), p.y()] for p in committed])
    np.testing.assert_array_equal(coordinates.min(axis=0), [10, 10])
    np.testing.assert_array_equal(coordinates.max(axis=0), [30, 30])
    assert view.canvas.shapes[0] is a
    view.refine_existing_shape_to_edge(b)
    wait_preview(view, app)
    view.cancel_pixel_edge_preview(switch_mode=False)
    assert view.canvas.shapes == [a, b]
    assert a.points == committed
    assert b.points == original_b
    assert not view.canvas.edge_preview_shapes
    assert view.canvas.current is None
    assert not view.canvas.selected_shapes_copy
    assert view.canvas.h_vertex is None


def test_cell_mask_roundtrip_does_not_grow_by_one_pixel():
    image = np.zeros((15, 15), np.uint8)
    image[4:9, 5:11] = 220
    result = segment_box_to_edge_polygon(image, [[4, 3], [11, 9]])
    assert result.succeeded
    mask = np.zeros_like(image)
    fill_cell_polygon(mask, result.points, 220)
    np.testing.assert_array_equal(mask, image)


def test_model_toggle_has_exactly_one_receiver(window):
    view, app = window
    box_action = view._settings_runtime_applier._shortcut_action_map[
        "shortcuts.create_pixel_edge_box"
    ]
    view._settings_runtime_applier.apply_shortcuts(
        "shortcuts.create_pixel_edge_box", "Alt+Shift+E"
    )
    assert box_action.shortcut() == QtGui.QKeySequence("Alt+Shift+E")
    signal = view.auto_labeling_widget.model_manager.new_auto_labeling_result
    for enabled in (False, True, True, False, False, True):
        view.toggle_auto_label_edge_refine(enabled)
        assert view.auto_labeling_widget.model_manager.receivers(signal) == 1


def test_roi_batch_existing_confirm_autosaves_every_shape(window):
    view, app = window
    originals = [list(shape.points) for shape in view.canvas.shapes]
    view._config["auto_save"] = True
    view._refresh_existing_batch_preview(list(view.canvas.shapes))
    wait_batch_preview(view, app)
    assert all(
        shape.points == original
        for shape, original in zip(
            [item["shape"] for item in view._edge_batch_preview["items"]],
            originals,
        )
    )
    assert view.confirm_pixel_edge_preview()
    app.processEvents()
    output = os.path.splitext(view.filename)[0] + ".json"
    assert os.path.isfile(output)
    data = json.loads(open(output, encoding="utf-8").read())
    assert len(data["shapes"]) == 2
    assert all(
        shape["pixel_edge_geometry"] == "cell_boundary"
        and shape["pixel_edge_coordinates"] == "image_corner"
        for shape in data["shapes"]
    )
    for shape in data["shapes"]:
        points = np.asarray(shape["points"])
        np.testing.assert_array_equal(points, np.round(points))


def test_json_mask_export_uses_canvas_corner_coordinates(tmp_path):
    import json
    import cv2
    from anylabeling.views.labeling.label_converter import LabelConverter

    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            {
                "imageWidth": 8,
                "imageHeight": 8,
                "shapes": [
                    {
                        "label": "cell",
                        "shape_type": "polygon",
                        "points": [
                            [6.0, 6.0],
                            [8.0, 6.0],
                            [8.0, 8.0],
                            [6.0, 8.0],
                        ],
                        "pixel_edge_coordinates": "image_corner",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "mask.png"
    LabelConverter().custom_to_mask(
        str(source),
        str(output),
        {"type": "grayscale", "colors": {"cell": 255}},
    )
    mask = cv2.imdecode(
        np.fromfile(output, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
    )
    expected = np.zeros((8, 8), np.uint8)
    expected[6:8, 6:8] = 255
    np.testing.assert_array_equal(mask, expected)
