from pathlib import Path

from PyQt6 import QtCore, QtGui
from PyQt6 import QtWidgets


def _write_png(path: Path) -> None:
    image = QtGui.QImage(32, 32, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("white"))
    assert image.save(str(path))


class FakeManager:
    def __init__(self):
        self.callbacks = []


def test_ultralytics_training_state_machine(
    qtbot, tmp_path, monkeypatch, xanylabeling_workdir
):
    from anylabeling.views.training import ultralytics_dialog as dialog_module

    monkeypatch.setattr(dialog_module, "get_training_manager", lambda: FakeManager())
    monkeypatch.setattr(dialog_module, "get_export_manager", lambda: FakeManager())

    img = tmp_path / "a.png"
    _write_png(img)

    class Parent(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.image_list = [str(img)]
            self.output_dir = str(tmp_path)
            self.supported_shape = ["rectangle", "polygon"]

    parent = Parent()

    dlg = dialog_module.UltralyticsDialog(parent=parent)
    dlg.show()

    dlg.on_training_event("training_started", {"total_epochs": 3})
    assert dlg.training_status == "training"
    assert dlg.stop_training_button.isHidden() is False
    assert dlg.start_training_button.isHidden() is True

    dlg.on_training_event("training_log", {"message": "epoch 1"})
    assert "epoch 1" in dlg.log_display.toPlainText()

    dlg.on_training_event("training_completed", {})
    assert dlg.training_status == "completed"
    assert dlg.export_button.isHidden() is False
    assert dlg.previous_button.isHidden() is False
    assert dlg.stop_training_button.isHidden() is True
    dlg.close()
    parent.close()
