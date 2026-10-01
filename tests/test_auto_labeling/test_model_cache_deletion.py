import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

import anylabeling.resources.resources  # noqa: F401
from anylabeling.services.auto_labeling import model_manager
from anylabeling.views.labeling.widgets.auto_labeling.auto_labeling import (
    AutoLabelingWidget,
)
from anylabeling.views.labeling.widgets.searchable_model_dropdown import (
    ModelItem,
    SearchableModelDropdownPopup,
)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_manager, "get_work_directory", lambda: str(tmp_path)
    )
    monkeypatch.setattr(
        model_manager.ModelManager, "load_model_configs", lambda self: None
    )
    manager = model_manager.ModelManager()
    manager.model_configs = [
        {
            "name": "demo",
            "encoder_model_path": "https://example.com/encoder.onnx",
            "decoder_model_path": "https://example.com/decoder.onnx?download=1",
            "local_model_path": str(tmp_path / "local.onnx"),
        }
    ]
    return manager


def write_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"model")
    return path


def test_delete_caches_preserves_local_and_other_files(manager, tmp_path):
    downloaded = [
        write_file(tmp_path / data_dir / "models/demo" / filename)
        for data_dir in ("xanylabeling_data", "anylabeling_data")
        for filename in ("encoder.onnx", "decoder.onnx")
    ]
    preserved = [
        write_file(tmp_path / "local.onnx"),
        write_file(tmp_path / "xanylabeling_data/models/other/encoder.onnx"),
        write_file(tmp_path / "xanylabeling_data/models/demo/notes.txt"),
    ]
    model = Mock()
    manager.loaded_model_config = {"name": "demo", "model": model}

    def unload():
        assert all(path.exists() for path in downloaded)

    model.unload.side_effect = unload
    manager.delete_downloaded_model_files("demo")
    model.unload.assert_called_once()
    assert manager.loaded_model_config is None
    assert not any(path.exists() for path in downloaded)
    assert all(path.exists() for path in preserved)
    assert manager.get_downloaded_model_files("demo") == []
    assert manager.model_configs[0]["name"] == "demo"


@pytest.mark.parametrize("stage", ["download", "inference"])
def test_busy_model_cannot_be_deleted(manager, tmp_path, stage):
    path = write_file(tmp_path / "xanylabeling_data/models/demo/encoder.onnx")
    if stage == "download":
        manager.model_download_thread = Mock()
    else:
        manager.model_execution_thread = Mock()
    with pytest.raises(RuntimeError):
        manager.delete_downloaded_model_files("demo")
    assert path.exists()


def test_cache_symlinks_are_not_followed(manager, tmp_path):
    external = write_file(tmp_path / "external/encoder.onnx")
    root = tmp_path / "xanylabeling_data/models"
    root.mkdir(parents=True)
    (root / "demo").symlink_to(external.parent, target_is_directory=True)
    assert manager.get_downloaded_model_files("demo") == []
    manager.delete_downloaded_model_files("demo")
    assert external.exists()


def test_deleting_other_cache_keeps_current_model(manager, tmp_path):
    path = write_file(tmp_path / "xanylabeling_data/models/demo/encoder.onnx")
    model = Mock()
    manager.loaded_model_config = {"name": "other", "model": model}
    manager.delete_downloaded_model_files("demo")
    assert not path.exists()
    model.unload.assert_not_called()
    assert manager.loaded_model_config["name"] == "other"


def test_confirm_deletion_resets_active_selection(
    app, manager, tmp_path, monkeypatch
):
    path = write_file(tmp_path / "xanylabeling_data/models/demo/encoder.onnx")
    manager.loaded_model_config = {"name": "demo", "model": Mock()}
    dropdown = Mock()
    dropdown.models_data = {"Alibaba": {"demo": {"selected": True}}}
    widget = SimpleNamespace(
        parent=Mock(),
        model_manager=manager,
        model_info={"demo": {"display_name": "Demo"}},
        model_dropdown=dropdown,
        refresh_downloaded_models=Mock(),
        clear_auto_labeling_action_requested=Mock(),
        hide_labeling_widgets=Mock(),
        model_selection_button=Mock(),
        tr=lambda text: text,
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes
    )
    popup = Mock()
    monkeypatch.setattr(
        "anylabeling.views.labeling.widgets.auto_labeling.auto_labeling.Popup",
        popup,
    )
    AutoLabelingWidget.on_downloaded_model_delete_requested(widget, "demo")
    assert not path.exists()
    assert not dropdown.models_data["Alibaba"]["demo"]["selected"]
    widget.hide_labeling_widgets.assert_called_once()
    widget.model_selection_button.setText.assert_called_once_with("No Model")
    widget.refresh_downloaded_models.assert_called_once()
    popup.return_value.show_popup.assert_called_once_with(widget.parent)


def test_cancel_confirmation_preserves_files(app, monkeypatch):
    manager = Mock()
    manager.get_downloaded_model_files.return_value = [
        "/cache/demo/model.onnx"
    ]
    widget = SimpleNamespace(
        model_manager=manager,
        model_info={"demo": {"display_name": "Demo"}},
        model_dropdown=Mock(),
        tr=lambda text: text,
    )
    question = Mock(return_value=QMessageBox.StandardButton.No)
    monkeypatch.setattr(QMessageBox, "question", question)
    AutoLabelingWidget.on_downloaded_model_delete_requested(widget, "demo")
    manager.delete_downloaded_model_files.assert_not_called()
    assert question.call_args.args[-1] == QMessageBox.StandardButton.No
    assert "/cache/demo/model.onnx" in question.call_args.args[2]


def test_delete_click_does_not_select_model(app):
    popup = SearchableModelDropdownPopup(
        {"Alibaba": {"demo": {"display_name": "Demo", "favorite": True}}}
    )
    popup.downloaded_models = {"demo"}
    popup.setup_model_list()
    selected = Mock()
    deleted = Mock()
    popup.modelSelected.connect(selected)
    popup.modelDownloadDeleteRequested.connect(deleted)
    popup.show()
    app.processEvents()
    rows = popup.findChildren(ModelItem)
    assert len(rows) == 2
    for row in rows:
        assert row.trash_button is not None
        assert not row.trash_button.icon().isNull()
        row.trash_button.show()
        QTest.mouseClick(row.trash_button, Qt.MouseButton.LeftButton)
    assert deleted.call_count == 2
    selected.assert_not_called()
    popup.close()
