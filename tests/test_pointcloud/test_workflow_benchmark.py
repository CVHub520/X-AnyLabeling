import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import psutil
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.pointcloud.io import load_frame
from anylabeling.views.labeling.widgets import pointcloud_dialog as module


def _draw(app, window):
    rendered = []

    def presented():
        rendered.append(True)

    window.viewport._gl.frameSwapped.connect(presented)
    window.viewport._gl.update()
    deadline = time.monotonic() + 5
    while not rendered and time.monotonic() < deadline:
        app.processEvents()
        QtCore.QThread.msleep(1)
    window.viewport._gl.frameSwapped.disconnect(presented)
    assert rendered, "The rendered result was not presented"


def _wait_loaded(app, window, errors):
    deadline = time.monotonic() + 30
    while window._worker is not None and time.monotonic() < deadline:
        app.processEvents()
        QtCore.QThread.msleep(1)
    assert window._worker is None
    assert not errors, errors
    _draw(app, window)


def _mouse_event(kind, center):
    return QtGui.QMouseEvent(
        kind,
        QtCore.QPointF(*center),
        QtCore.QPointF(*center),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


def _brush_stroke(app, window, center):
    viewport = window.viewport
    start = time.perf_counter()
    viewport._mouse_press(
        _mouse_event(QtCore.QEvent.Type.MouseButtonPress, center)
    )
    viewport._mouse_move(
        _mouse_event(QtCore.QEvent.Type.MouseMove, center + [10, 0])
    )
    viewport._mouse_release(
        _mouse_event(QtCore.QEvent.Type.MouseButtonRelease, center + [10, 0])
    )
    _draw(app, window)
    return (time.perf_counter() - start) * 1000


def _cold_and_preview_trials(app, window, center):
    viewport = window.viewport
    cold = []
    for _ in range(5):
        viewport._selection_cache = None
        cold.append(_brush_stroke(app, window, center))
        window.undo()
        _draw(app, window)
    viewport._mouse_press(
        _mouse_event(QtCore.QEvent.Type.MouseButtonPress, center)
    )
    _draw(app, window)
    preview = []
    for step in range(1, 26):
        point = center + [step * 2, step]
        start = time.perf_counter()
        viewport._mouse_move(_mouse_event(QtCore.QEvent.Type.MouseMove, point))
        _draw(app, window)
        preview.append((time.perf_counter() - start) * 1000)
    viewport._mouse_release(
        _mouse_event(QtCore.QEvent.Type.MouseButtonRelease, point)
    )
    _draw(app, window)
    assert window.document.dirty
    window.undo()
    _draw(app, window)
    return cold, preview


@pytest.mark.skipif(
    os.environ.get("POINTCLOUD_GUI_BENCHMARK") != "1",
    reason="Opt-in native display workflow benchmark",
)
def test_native_workflow_benchmark(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    if QtGui.QGuiApplication.platformName() in ("offscreen", "minimal"):
        pytest.skip("A native OpenGL display is required")
    settings_class = QtCore.QSettings
    with patch.object(
        module.QtCore,
        "QSettings",
        lambda *args: settings_class(
            str(tmp_path / "benchmark.ini"), settings_class.Format.IniFormat
        ),
    ):
        window = module.PointCloudDialog()
    window.resize(1366, 768)
    errors = []
    window._error = errors.append
    window.show()
    app.processEvents()
    assert window.viewport._error is None
    window.viewport._gl.makeCurrent()
    gl = window.viewport._gl.functions
    graphics = {
        "vendor": gl.glGetString(0x1F00),
        "renderer": gl.glGetString(0x1F01),
        "version": gl.glGetString(0x1F02),
    }
    window.viewport._gl.doneCurrent()
    process = psutil.Process()
    rotation_seconds = float(
        os.environ.get("POINTCLOUD_GUI_ROTATION_SECONDS", "30")
    )
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "qt": QtCore.QT_VERSION_STR,
        "numpy": np.__version__,
        "device_pixel_ratio": window.devicePixelRatioF(),
        "viewport": [window.viewport.width(), window.viewport.height()],
        "point_size": window.point_size.value(),
        "rss_before_frames_mib": process.memory_info().rss / 2**20,
        "seed": 20260906,
        "frame_trials_seconds": rotation_seconds,
        "browse_motion": "yaw +1 degree per 16 ms timer; scale = initial * (1 + 0.2 * sin(yaw))",
        "scales": [],
        "graphics": graphics,
        "timing": "Qt frameSwapped; brush input to presented result; preview input to presented feedback",
    }

    try:
        module.save_classes(
            tmp_path / "pointcloud_classes.json",
            [
                *module.DEFAULT_CLASSES,
                module.ClassDefinition(10, "Vehicle", "#6496F5"),
            ],
        )
        for count in (100_000, 1_000_000):
            rng = np.random.default_rng(20260906)
            points = rng.normal(size=(count, 4)).astype("<f4")
            path = tmp_path / f"frame_{count}.bin"
            points.tofile(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            record = {
                "points": count,
                "sha256": digest,
                "load_ms": [],
                "fps_trials": [],
                "frame_interval_ms": [],
                "brush_ms": [],
                "brush_points": [],
                "rss_cycles_mib": [],
            }
            for _ in range(3):
                start = time.perf_counter()
                window.open_paths([path])
                _wait_loaded(app, window, errors)
                record["load_ms"].append((time.perf_counter() - start) * 1000)
            window.color_mode.setCurrentIndex(1)
            window.viewport.reset_view()
            _draw(app, window)
            for _ in range(3):
                frames = []
                initial_scale = window.viewport._scale

                def presented():
                    frames.append(time.perf_counter())

                def rotate():
                    window.viewport._yaw += 1.0
                    window.viewport._scale = initial_scale * (
                        1 + 0.2 * np.sin(np.deg2rad(window.viewport._yaw))
                    )
                    window.viewport._update()

                timer = QtCore.QTimer()
                timer.timeout.connect(rotate)
                window.viewport._gl.frameSwapped.connect(presented)
                start = time.perf_counter()
                timer.start(16)
                while time.perf_counter() - start < rotation_seconds:
                    app.processEvents()
                    QtCore.QThread.msleep(1)
                timer.stop()
                window.viewport._gl.frameSwapped.disconnect(presented)
                record["fps_trials"].append(
                    len(frames) / (time.perf_counter() - start)
                )
                record["frame_interval_ms"].append(
                    np.round(np.diff(frames) * 1000, 3).tolist()
                )
            window.viewport.reset_view()
            for row in range(window.class_list.count()):
                if (
                    window.class_list.item(row).data(
                        QtCore.Qt.ItemDataRole.UserRole
                    )
                    == 10
                ):
                    window.class_list.setCurrentRow(row)
                    break
            window._select_tool("brush")
            viewport = window.viewport
            center = np.array([viewport.width() / 2, viewport.height() / 2])
            for _ in range(20):
                record["brush_ms"].append(_brush_stroke(app, window, center))
                record["brush_points"].append(
                    int(np.count_nonzero(window.document.semantic == 10))
                )
                assert record["brush_points"][-1] > 0
                window.undo()
                _draw(app, window)
            record["cold_brush_ms"], record["preview_ms"] = (
                _cold_and_preview_trials(app, window, center)
            )
            window._leave_decision = lambda: "discard"
            for cycle in range(1, 51):
                window.open_paths([path])
                _wait_loaded(app, window, errors)
                window.viewport._yaw += cycle
                _draw(app, window)
                window.document.assign_semantic(
                    np.arange(cycle, cycle + 1000), 10, overwrite=True
                )
                window._refresh()
                assert window.save_work(), errors
                expected = window.document.labels.copy()
                np.testing.assert_array_equal(
                    load_frame(path, window.document.frame.label_path).labels,
                    expected,
                )
                if cycle % 10 == 0:
                    record["rss_cycles_mib"].append(
                        {
                            "cycle": cycle,
                            "rss": process.memory_info().rss / 2**20,
                        }
                    )
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            record["cycles"] = 50
            report["scales"].append(record)
            print("POINTCLOUD_GUI_SCALE " + json.dumps(record), flush=True)
        output = os.environ.get("POINTCLOUD_GUI_REPORT")
        if output:
            with open(output, "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
        print("POINTCLOUD_GUI_REPORT " + json.dumps(report), flush=True)
    finally:
        window.close_after_approval()
        app.processEvents()
