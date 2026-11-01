import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.classifier.utils import (
    MultiTaskBatch,
)
from anylabeling.views.labeling.shape import Shape


def test_batch_parallel_and_preserves_other_annotation_data(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    current = str(tmp_path / "current.jpg")
    other = str(tmp_path / "other.jpg")
    image = QtGui.QImage(80, 80, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("red"))
    assert image.save(other)
    shape = Shape(label="car", shape_type="rectangle")
    shape.points = [QtCore.QPointF(10, 10), QtCore.QPointF(60, 60)]
    shape.attributes = {"color": "blue"}
    annotation = {
        "shapes": [
            {
                "label": "car",
                "shape_type": "rectangle",
                "points": [[10, 10], [60, 60]],
                "attributes": {"color": "blue", "other": "keep"},
            }
        ],
        "custom": "keep",
    }
    label = tmp_path / "other.json"
    label.write_text(json.dumps(annotation))
    owner = QtWidgets.QWidget()
    owner.label_widget = SimpleNamespace(
        filename=current,
        image=image,
        canvas=SimpleNamespace(shapes=[shape]),
        attributes={"car": {"color": ["red", "blue"]}},
        attribute_widget_types={},
        output_dir=None,
    )
    path = QtGui.QPainterPath()
    path.addRect(10, 10, 50, 50)
    owner.image_label = SimpleNamespace(
        object_path=lambda shape: path, focus_selected_object=Mock()
    )
    owner.multitask_form = SimpleNamespace(
        set_value=lambda s, k, v: s.attributes.update({k: v})
    )
    owner.object_combo = SimpleNamespace(currentIndex=lambda: 0)
    owner.select_object = Mock()
    owner._ai_workers = set()
    owner._multitask_batch_errors = []
    stats = {"running": 0, "maximum": 0}

    class FakeWorker(QtCore.QObject):
        finished = QtCore.pyqtSignal(str, bool, str)
        completed = QtCore.pyqtSignal()

        def __init__(self, *args):
            super().__init__()

        def start(self):
            stats["running"] += 1
            stats["maximum"] = max(stats["maximum"], stats["running"])
            QtCore.QTimer.singleShot(10, self.done)

        def done(self):
            self.finished.emit('{"color":"red"}', True, "")
            stats["running"] -= 1
            self.completed.emit()

        def cancel(self):
            pass

    with (
        patch(
            "anylabeling.views.labeling.classifier.utils.AIWorkerThread",
            FakeWorker,
        ),
        patch.object(QtWidgets.QMessageBox, "information"),
    ):
        batch = MultiTaskBatch(
            owner, [current, other], "@image @object @attributes", 2
        )
        owner._multitask_batch = batch
        batch.start()
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(100, loop.quit)
        loop.exec()
        assert stats["maximum"] == 2
        assert batch.succeeded == 2
        owner.select_object.assert_called_with(0)
        owner.image_label.focus_selected_object.assert_called_once()
        assert shape.attributes["color"] == "red"
        saved = json.loads(label.read_text())
        assert saved["custom"] == "keep"
        assert saved["shapes"][0]["attributes"] == {
            "color": "red",
            "other": "keep",
        }
    owner.close()


def test_batch_detects_annotation_changes_before_writing(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image_path = str(tmp_path / "other.jpg")
    label_path = tmp_path / "other.json"
    original = {
        "shapes": [
            {
                "label": "car",
                "shape_type": "rectangle",
                "points": [[10, 10], [60, 60]],
                "attributes": {"color": "blue"},
            }
        ]
    }
    label_path.write_text(json.dumps(original))
    owner = QtWidgets.QWidget()
    owner.label_widget = SimpleNamespace(
        filename="current.jpg",
        canvas=SimpleNamespace(shapes=[]),
        attributes={"car": {"color": ["red", "blue"]}},
        attribute_widget_types={},
        output_dir=None,
    )
    owner._multitask_batch_errors = []
    batch = MultiTaskBatch(owner, [image_path], "@image", 1)
    job = batch.queue[0]
    worker = object()
    batch.running[worker] = job
    changed = dict(original, external="preserve")
    label_path.write_text(json.dumps(changed))
    batch.result(worker, '{"color":"red"}', True, "")
    assert batch.failed == 1
    assert json.loads(label_path.read_text()) == changed
    batch.running.clear()
    batch.progress.canceled.disconnect(batch.cancel)
    batch.progress.close()
    owner.close()


def test_batch_cancel_discards_pending_results():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    owner = QtWidgets.QWidget()
    owner.label_widget = SimpleNamespace(
        filename=None,
        canvas=SimpleNamespace(shapes=[]),
        attributes={},
        attribute_widget_types={},
    )
    owner._multitask_batch_errors = []
    batch = MultiTaskBatch(owner, [], "@image", 1)
    worker = Mock()
    owner._ai_workers = {worker}
    owner._multitask_batch = batch
    batch.running[worker] = ("test.jpg", None, {}, None)
    batch.queue.append(("pending.jpg", None, {}, None))
    batch.total = 2
    batch.cancel()
    worker.cancel.assert_called_once()
    assert not batch.queue
    batch.result(worker, '{"color":"red"}', True, "")
    assert batch.succeeded == 0
    with patch.object(QtWidgets.QMessageBox, "information"):
        batch.completed(worker)
    assert batch.finished
    owner.close()


def test_classification_batch_parallel_and_preserves_annotations(tmp_path):
    from anylabeling.views.labeling.classifier.utils import ClassificationBatch

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    paths = [
        str(tmp_path / f"{name}.jpg")
        for name in ("current", "existing", "new")
    ]
    image = QtGui.QImage(32, 32, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("red"))
    for path in paths:
        assert image.save(path)
    label = tmp_path / "existing.json"
    label.write_text(
        json.dumps(
            {
                "flags": {"cat": True, "dog": False, "other": True},
                "checked": True,
                "shapes": [{"label": "keep"}],
                "custom": "keep",
            }
        )
    )
    owner = QtWidgets.QWidget()
    owner.labels = ["cat", "dog"]
    owner.is_multiclass = True
    owner.label_widget = SimpleNamespace(
        filename=paths[0], output_dir=None, _config={"store_data": False}
    )
    owner.checkbox_group = SimpleNamespace(set_flags=Mock())
    owner.save_current_flags = Mock()
    owner._ai_workers = set()
    owner._multitask_batch_errors = []
    stats = {"running": 0, "maximum": 0}

    class FakeWorker(QtCore.QObject):
        finished = QtCore.pyqtSignal(str, bool, str)
        completed = QtCore.pyqtSignal()

        def __init__(self, *args):
            super().__init__()

        def start(self):
            stats["running"] += 1
            stats["maximum"] = max(stats["maximum"], stats["running"])
            QtCore.QTimer.singleShot(10, self.done)

        def done(self):
            self.finished.emit('{"cat":false,"dog":true}', True, "")
            stats["running"] -= 1
            self.completed.emit()

        def cancel(self):
            pass

    with (
        patch(
            "anylabeling.views.labeling.classifier.utils.AIWorkerThread",
            FakeWorker,
        ),
        patch.object(QtWidgets.QMessageBox, "information"),
    ):
        batch = ClassificationBatch(owner, paths, "@image classify", 2)
        owner._multitask_batch = batch
        batch.start()
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(120, loop.quit)
        loop.exec()
        assert stats["maximum"] == 2
        assert batch.succeeded == 3
        owner.checkbox_group.set_flags.assert_called_once_with(
            {"cat": False, "dog": True}
        )
        owner.save_current_flags.assert_called_once()
        existing = json.loads(label.read_text())
        assert existing["flags"] == {"cat": False, "dog": True, "other": True}
        assert existing["shapes"] == [{"label": "keep"}]
        assert existing["custom"] == "keep" and existing["checked"] is False
        created = json.loads((tmp_path / "new.json").read_text())
        assert created["flags"] == {"cat": False, "dog": True}
        assert created["imageWidth"] == 32
    owner.close()


def test_classification_batch_rejects_invalid_and_changed_results(tmp_path):
    from anylabeling.views.labeling.classifier.utils import ClassificationBatch

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path = str(tmp_path / "image.jpg")
    label_path = tmp_path / "image.json"
    original = {"flags": {"cat": True, "dog": False}, "shapes": []}
    label_path.write_text(json.dumps(original))
    owner = QtWidgets.QWidget()
    owner.labels = ["cat", "dog"]
    owner.is_multiclass = True
    owner.label_widget = SimpleNamespace(filename="other.jpg", output_dir=None)
    owner._multitask_batch_errors = []
    batch = ClassificationBatch(owner, [path], "@image", 1)
    worker = object()
    batch.running[worker] = batch.queue[0]
    batch.result(worker, '{"cat":true,"dog":true}', True, "")
    assert batch.failed == 1
    changed = dict(original, custom="external")
    label_path.write_text(json.dumps(changed))
    batch.result(worker, '{"cat":false,"dog":true}', True, "")
    assert batch.failed == 2
    assert json.loads(label_path.read_text()) == changed
    batch.running.clear()
    batch.progress.canceled.disconnect(batch.cancel)
    batch.progress.close()
    owner.close()
