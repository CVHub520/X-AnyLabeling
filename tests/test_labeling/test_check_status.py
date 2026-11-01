import copy
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6 import QtCore, QtWidgets

from anylabeling import config as app_config
from anylabeling.resources import resources  # noqa: F401
from anylabeling.views.labeling.settings.schema import load_template_config
from anylabeling.views.mainwindow import MainWindow
from PIL import Image


@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(app_config, "_work_directory", str(tmp_path))
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(app_config, "current_config_file", str(config_file))
    settings_class = QtCore.QSettings
    monkeypatch.setattr(
        QtCore,
        "QSettings",
        lambda *args: settings_class(
            str(tmp_path / "settings.ini"), settings_class.Format.IniFormat
        ),
    )
    window = MainWindow(app, config=copy.deepcopy(load_template_config()))
    window.resize(1600, 1000)
    window.show()
    app.processEvents()
    yield window
    window.labeling_widget.view.dirty = False
    window.close()
    app.removeEventFilter(window.labeling_widget.view)
    window.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_check_status_actions_save_and_reload(window, tmp_path):
    widget = window.labeling_widget.view
    path = str(tmp_path / "image.png")
    Image.new("RGB", (64, 64)).save(path)
    assert widget.load_file(path)
    actions = (
        (widget.actions.toggle_annotation_checked, True),
        (widget.actions.fail_annotation_checked, "failed"),
        (widget.actions.clear_annotation_checked, False),
    )
    for action, status in actions:
        for _ in range(2):
            action.trigger()
            assert widget._annotation_checked() == status
            assert action.isChecked()
            assert sum(a.isChecked() for a, _ in actions) == 1
            label_path = widget.get_label_file()
            with open(label_path, encoding="utf-8") as stream:
                assert json.load(stream)["checked"] == status
            assert widget._label_file_checked(label_path) == status
            item = widget._create_file_list_item(path, label_path)
            assert widget._file_item_annotation_checked(item) == bool(status)
            assert widget._annotation_checked_text() in item.toolTip()
            widget.filename = None
            assert widget.load_file(path)
            assert widget._annotation_checked() == status
