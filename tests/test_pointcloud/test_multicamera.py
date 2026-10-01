from dataclasses import replace

import numpy as np
import pytest
from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

from anylabeling.views.labeling.pointcloud.camera import (
    CameraConfigurationDialog,
    CameraSource,
    image_files,
    project_points,
)
from anylabeling.views.labeling.pointcloud.cuboid import Cuboid
from .test_camera import calibration_data, save_calibration, save_image
from .test_dialog import app, window, cloud, open_cloud


def source(tmp_path, index, frames=("0000",)):
    directory = tmp_path / f"image_{index:02d}" / "data"
    directory.mkdir(parents=True)
    for frame in frames:
        save_image(directory / f"{frame}.png")
    data = calibration_data()
    data["camera_matrix"][0][2] += index * 4
    parameters = save_calibration(directory, data)
    return CameraSource(
        directory,
        image_files(directory),
        parameters,
        directory / "calibration.json",
    )


def wheel(view, delta=-120, modifiers=QtCore.Qt.KeyboardModifier.NoModifier):
    position = view.viewport().rect().center()
    event = QtGui.QWheelEvent(
        QtCore.QPointF(position),
        QtCore.QPointF(view.viewport().mapToGlobal(position)),
        QtCore.QPoint(),
        QtCore.QPoint(0, delta),
        QtCore.Qt.MouseButton.NoButton,
        modifiers,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(view.viewport(), event)
    assert event.isAccepted()


def test_blank_configuration_accepts_and_ignores_empty_rows(window, tmp_path):
    dialog = CameraConfigurationDialog(window.camera_panel, window)
    dialog.calibration_input.setText("ignored.json")
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
    assert dialog.configuration == []
    assert dialog.error_label.isHidden()
    window.camera_panel.configure_sources(dialog.configuration)
    assert window.camera_panel.isHidden()
    assert not window.camera_panel_action.isEnabled()
    dialog.deleteLater()


def test_multi_configuration_preview_and_scroll_after_three_sources(
    window, app, tmp_path
):
    sources = [source(tmp_path, index) for index in range(4)]
    broken = sources[0].directory / "000-broken.png"
    broken.write_bytes(b"invalid image")
    dialog = CameraConfigurationDialog(window.camera_panel, window)
    dialog.directory_input.setText(str(sources[0].directory))
    dialog.calibration_input.setText(str(sources[0].calibration_path))
    for camera in sources[1:3]:
        dialog.add_camera(camera)
    dialog._refresh_previews()
    dialog.show()
    app.processEvents()
    assert dialog.sources_scroll.verticalScrollBar().maximum() == 0

    def assert_button_alignment():
        def edges(widget):
            left = widget.mapTo(dialog, QtCore.QPoint()).x()
            return left, left + widget.width() - 1

        for entry in dialog.entries:
            for row in (1, 2):
                browse = entry.layout().itemAtPosition(row, 1).widget()
                field = entry.layout().itemAtPosition(row, 0).widget()
                assert edges(browse) == edges(dialog.ok_button)
                assert edges(field)[1] == edges(dialog.cancel_button)[1]

    assert_button_alignment()
    assert dialog.entries[0].title.text() == "image_00:"
    assert dialog.preview_caption.text() == "image_00"
    assert not dialog.preview.image_item.pixmap().isNull()
    height = dialog.sources_scroll.height()
    dialog.add_camera(sources[3])
    dialog._refresh_previews()
    app.processEvents()
    assert dialog.sources_scroll.height() == height
    assert dialog.sources_scroll.verticalScrollBar().maximum() > 0
    for width in (540, 720):
        dialog.resize(width, dialog.height())
        app.processEvents()
        assert_button_alignment()
    assert dialog.preview.camera_count == 4
    wheel(dialog.preview)
    assert dialog.preview_caption.text() == "image_01"
    dialog.preview.navigation[1].click()
    assert dialog.preview_caption.text() == "image_02"
    dialog.accept()
    assert [camera.name for camera in dialog.configuration] == [
        f"image_{i:02d}" for i in range(4)
    ]
    assert dialog.configuration[1].calibration["camera_matrix"][0, 2] == 24
    dialog.deleteLater()


def test_invalid_second_camera_does_not_apply_partial_changes(
    window, tmp_path
):
    cameras = [source(tmp_path, index) for index in range(2)]
    panel = window.camera_panel
    panel.configure_sources(cameras[:1])
    dialog = CameraConfigurationDialog(panel, window)
    dialog.add_camera(cameras[1])
    dialog.entries[1].calibration_input.setText(str(tmp_path / "missing.json"))
    dialog.accept()
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    assert not dialog.error_label.isHidden()
    assert len(panel.sources) == 1
    assert panel.directory == cameras[0].directory
    dialog.reject()
    dialog.deleteLater()


def test_switching_camera_updates_point_and_box_projection(
    window, app, tmp_path
):
    cameras = [source(tmp_path, index) for index in range(2)]
    path = cloud(tmp_path / "0000.bin")
    np.array([[0, 0, 5, 0], [1, 0, 5, 0]], dtype="<f4").tofile(path)
    open_cloud(window, app, path)
    window.sidebar_tabs.setCurrentIndex(0)
    box = Cuboid(1, 10, (0, 0, 5), (2, 2, 2))
    window.document.set_cuboid(box)
    window._refresh()
    window._autosave_timer.stop()
    window.show()
    panel = window.camera_panel
    panel.configure_sources(cameras)
    app.processEvents()
    first = panel._pixels.copy()
    first_box = panel.view.cuboid_item.paths[0][1].boundingRect()
    panel.view.navigation[1].click()
    assert panel.active_index == 1
    assert panel.filename.text() == "image_01"
    expected = project_points(
        window.document.frame.points, cameras[1].calibration, 40, 40
    )[1]
    np.testing.assert_array_equal(panel._pixels, expected)
    assert not np.array_equal(panel._pixels, first)
    second_box = panel.view.cuboid_item.paths[0][1].boundingRect()
    assert second_box.left() - first_box.left() == pytest.approx(4)
    assert window.document.cuboids == (box,)
    assert not window.document.labels_dirty
    scale = panel.view.transform().m11()
    wheel(panel.view, 120)
    assert panel.active_index == 1
    assert panel.view.transform().m11() == pytest.approx(scale * 1.2)
    wheel(panel.view)
    assert panel.active_index == 1
    assert panel.view.transform().m11() == pytest.approx(scale)
    np.testing.assert_array_equal(panel._pixels, expected)
    assert panel.view.cuboid_item.paths[0][1].boundingRect() == second_box
    wheel(panel.view, 120, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert panel.active_index == 1
    assert panel.view.transform().m11() > scale
    panel.view.navigation[0].click()
    assert panel.active_index == 0
    np.testing.assert_array_equal(panel._pixels, first)


def test_missing_frame_and_uncalibrated_camera_keep_navigation(
    window, app, tmp_path
):
    cameras = [source(tmp_path, 0), source(tmp_path, 1, ("0001",))]
    open_cloud(window, app, cloud(tmp_path / "0000.bin"))
    window.show()
    panel = window.camera_panel
    panel.configure_sources(cameras)
    app.processEvents()
    panel.view.navigation[1].click()
    assert panel._image.isNull()
    assert panel.view.image_item.pixmap().isNull()
    assert panel.view.isVisible()
    assert panel.message.isVisible()
    assert not panel.overlay_action.isEnabled()
    assert not panel.cuboid_action.isEnabled()
    panel.view.navigation[1].click()
    assert not panel._image.isNull()
    assert panel.message.isHidden()
    panel.configure_sources(
        [
            cameras[0],
            replace(cameras[0], calibration=None, calibration_path=None),
        ]
    )
    panel.step_camera(1)
    assert not panel.overlay_action.isEnabled()
    assert panel.view.overlay_item.pixmap().isNull()
    assert not panel.cuboid_action.isEnabled()


def test_hide_show_across_tasks_and_frames_keeps_camera_configuration(
    window, app, tmp_path
):
    cameras = [source(tmp_path, index, ("0000", "0001")) for index in range(2)]
    open_cloud(window, app, cloud(tmp_path / "0000.bin"))
    window.show()
    panel = window.camera_panel
    panel.configure_sources(cameras)
    panel.step_camera(1)
    assert window.camera_panel_action.isChecked()
    for task in (0, 1):
        window.sidebar_tabs.setCurrentIndex(task)
        assert window.camera_panel_button.isVisible()
        window.camera_panel_action.trigger()
        assert panel.isHidden()
        assert not window.camera_panel_action.isChecked()
        window.camera_panel_action.trigger()
        assert panel.isVisible()
        assert panel.active_index == 1
    window.camera_panel_action.trigger()
    open_cloud(window, app, cloud(tmp_path / "0001.bin"))
    assert panel.isHidden()
    window.camera_panel_action.trigger()
    assert panel.filename.text() == "image_01"
    assert panel.filename.toolTip().endswith("0001.png")
    assert not window.document.dirty
    panel.configure_sources([])
    assert panel.isHidden()
    assert not window.camera_panel_action.isEnabled()


def test_camera_resize_stays_anchored_and_respects_minimum(
    window, app, tmp_path
):
    window.resize(1400, 900)
    window.show()
    open_cloud(window, app, cloud(tmp_path / "0000.bin"))
    panel = window.camera_panel
    panel.configure_sources([source(tmp_path, 0)])
    app.processEvents()
    minimum = panel.minimum_image_size()
    assert panel.size() == minimum
    right = panel.geometry().right()
    panel.resize_image_region(minimum + QtCore.QSize(120, 100))
    assert panel.width() == minimum.width() + 120
    assert panel.height() == minimum.height() + 100
    assert panel.geometry().right() == right
    assert panel.y() == 8
    handle = panel.resize_handle
    start = handle.mapToGlobal(handle.rect().center())
    for event_type, position in (
        (QtCore.QEvent.Type.MouseButtonPress, start),
        (QtCore.QEvent.Type.MouseMove, start + QtCore.QPoint(-30, 20)),
        (
            QtCore.QEvent.Type.MouseButtonRelease,
            start + QtCore.QPoint(-30, 20),
        ),
    ):
        event = QtGui.QMouseEvent(
            event_type,
            QtCore.QPointF(handle.mapFromGlobal(position)),
            QtCore.QPointF(position),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(handle, event)
    assert panel.size() == minimum + QtCore.QSize(150, 120)
    panel.resize_image_region(QtCore.QSize(1, 1))
    assert panel.size() == minimum
    panel.resize_image_region(QtCore.QSize(9999, 9999))
    assert window.viewport.rect().contains(panel.geometry())
    assert panel.geometry().right() == right
