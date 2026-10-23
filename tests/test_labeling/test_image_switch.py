import copy
import json
import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

from anylabeling import config as app_config
from anylabeling.views.labeling.settings.schema import load_template_config
from anylabeling.views.mainwindow import MainWindow
from scripts.benchmark_image_switch import generate_images


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


def test_switch_drops_old_objects_and_batches_list_updates(window, tmp_path):
    app = QtWidgets.QApplication.instance()
    widget = window.labeling_widget.view
    paths = generate_images(tmp_path / "images", [0, 100, 300], 64)
    delegate = widget.label_list.itemDelegate()
    changes = QtTest.QSignalSpy(widget.label_list.item_changed)
    for count in (300, 100, 0, 300, 0):
        before = len(changes)
        with patch.object(
            widget,
            "_are_all_shapes_visible",
            wraps=widget._are_all_shapes_visible,
        ) as visibility_scan:
            assert widget.load_file(paths[count][0])
            app.processEvents()
            app.processEvents()
        assert visibility_scan.call_count <= 10
        assert len(changes) == before
        assert len(widget.label_list) == count
        assert len(widget.canvas.shapes) == count
        assert set(widget.canvas.visible) == set(widget.canvas.shapes)
        assert not delegate.findChildren(QtGui.QTextDocument)
        assert len(widget.canvas.shapes_backups) == 1
        assert len(widget.canvas.shapes_backups[0]) == count


def test_loaded_shapes_keep_filter_checkbox_lock_and_undo_behavior(
    window, tmp_path
):
    widget = window.labeling_widget.view
    path = generate_images(tmp_path / "images", [10], 64)[10][0]
    label_path = tmp_path / "images" / "polygons_010_0.json"
    data = json.loads(label_path.read_text(encoding="utf-8"))
    data["shapes"][0]["locked"] = True
    label_path.write_text(json.dumps(data), encoding="utf-8")
    assert widget.load_file(path)
    assert widget.label_list[0].is_locked()
    assert widget.canvas.shapes[0].locked
    changes = QtTest.QSignalSpy(widget.label_list.item_changed)
    item = widget.label_list[1]
    item.setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert len(changes) == 1
    assert not item.shape().visible
    assert not widget.canvas.is_visible(item.shape())
    item.setCheckState(QtCore.Qt.CheckState.Checked)
    assert item.shape().visible
    widget.set_label_filter_value("object_2")
    assert [s.label for s in widget.canvas.shapes if s.visible] == ["object_2"]
    widget.set_label_filter_value("")
    widget.set_gid_filter_value("1")
    assert [s.group_id for s in widget.canvas.shapes if s.visible] == [1] * 5
    widget.set_gid_filter_value("-1")
    assert all(s.visible for s in widget.canvas.shapes)
    widget.toggle_select_all()
    assert not any(s.visible for s in widget.canvas.shapes)
    widget.toggle_select_all()
    assert all(s.visible for s in widget.canvas.shapes)
    original = [s.to_dict() for s in widget.canvas.shapes]
    widget.canvas.shapes[1].points[0].setX(500)
    widget.canvas.store_shapes()
    widget.canvas.restore_shape()
    assert [s.to_dict() for s in widget.canvas.shapes] == original


def test_brightness_pixmap_refresh_preserves_visibility(window, tmp_path):
    widget = window.labeling_widget.view
    path = generate_images(tmp_path / "images", [1], 64)[1][0]
    assert widget.load_file(path)
    shape = widget.canvas.shapes[0]
    widget.label_list[0].setCheckState(QtCore.Qt.CheckState.Unchecked)
    widget.canvas.load_pixmap(
        QtGui.QPixmap.fromImage(widget.image), clear_shapes=False
    )
    assert widget.canvas.shapes == [shape]
    assert not widget.canvas.is_visible(shape)
    widget.canvas.load_pixmap(QtGui.QPixmap.fromImage(widget.image))
    assert widget.canvas.shapes == []
    assert widget.canvas.visible == {}
