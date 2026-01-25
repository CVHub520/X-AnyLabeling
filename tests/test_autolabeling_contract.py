import os
from pathlib import Path

import pytest
from PyQt6 import QtCore, QtGui

from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.mainwindow import MainWindow


def _write_png(path: Path) -> None:
    image = QtGui.QImage(64, 64, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("white"))
    assert image.save(str(path))


class FakeModelManager(QtCore.QObject):
    prediction_started = QtCore.pyqtSignal()
    prediction_finished = QtCore.pyqtSignal()
    new_auto_labeling_result = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.loaded_model_config = {"type": "fake"}

    def predict_shapes_threading(self, image, filename, **kwargs):
        self.prediction_started.emit()
        shape = Shape(shape_type="rectangle")
        shape.label = "cat"
        shape.points = [
            QtCore.QPointF(5, 5),
            QtCore.QPointF(30, 5),
            QtCore.QPointF(30, 30),
            QtCore.QPointF(5, 30),
        ]
        shape.close()
        self.new_auto_labeling_result.emit(AutoLabelingResult([shape], replace=True))
        self.prediction_finished.emit()

    def set_auto_labeling_marks(self, marks):
        return None


def test_autolabeling_contract_run_prediction_adds_shapes(
    qtbot, qapp, tmp_path, xanylabeling_workdir
):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_png(images_dir / "a.png")

    win = MainWindow(qapp, config=xanylabeling_workdir)
    qtbot.addWidget(win)
    win.show()
    qapp.processEvents()
    view = win.labeling_widget.view

    view.import_image_folder(str(images_dir), load=True)
    view.async_exif_scanner.stop_scan()
    qtbot.waitUntil(lambda: view.canvas.pixmap is not None, timeout=2000)
    qtbot.waitUntil(lambda: view.image_path is not None, timeout=2000)

    widget = view.auto_labeling_widget
    fake = FakeModelManager()
    widget.model_manager = fake
    fake.new_auto_labeling_result.connect(
        lambda *args: view.new_shapes_from_auto_labeling(args[0])
    )

    assert len(view.canvas.shapes) == 0
    widget.run_prediction()
    qtbot.waitUntil(lambda: len(view.canvas.shapes) == 1, timeout=2000)
    assert view.canvas.shapes[0].label == "cat"


@pytest.mark.skipif(
    os.environ.get("XANYLABELING_RUN_MODEL_SMOKE") != "1",
    reason="Set XANYLABELING_RUN_MODEL_SMOKE=1 to run real model smoke",
)
def test_autolabeling_real_model_smoke():
    raise RuntimeError("Real model smoke is not configured in this environment")
