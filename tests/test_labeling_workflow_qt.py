from pathlib import Path

from PyQt6 import QtCore

from anylabeling.views.mainwindow import MainWindow


def _write_png(path: Path) -> None:
    from PyQt6 import QtGui

    image = QtGui.QImage(64, 64, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("white"))
    assert image.save(str(path))


def test_workflow_open_folder_navigate_and_draw_and_save(
    qtbot, qapp, tmp_path, xanylabeling_workdir
):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_png(images_dir / "a.png")
    _write_png(images_dir / "b.png")

    win = MainWindow(qapp, config=xanylabeling_workdir)
    qtbot.addWidget(win)
    win.show()
    qapp.processEvents()
    view = win.labeling_widget.view

    view.import_image_folder(str(images_dir), load=True)
    view.async_exif_scanner.stop_scan()
    qapp.processEvents()

    assert view.file_list_widget.count() == 2
    assert view.filename is not None
    qtbot.waitUntil(lambda: view.canvas.pixmap is not None, timeout=2000)

    view.open_next_image()
    assert view.filename is not None
    view.open_prev_image()
    assert view.filename is not None

    view.label_dialog.pop_up = lambda *args, **kwargs: (
        "cat",
        {"occluded": False},
        None,
        "",
        False,
        [],
    )

    view.toggle_draw_mode(edit=False, create_mode="rectangle")

    class _Ev:
        def __init__(self, p: QtCore.QPointF):
            self._p = p

        def position(self):
            return self._p

        def button(self):
            return QtCore.Qt.MouseButton.LeftButton

        def modifiers(self):
            return QtCore.Qt.KeyboardModifier.NoModifier

    offset = view.canvas.offset_to_center()
    scale = view.canvas.scale
    p1 = (offset + QtCore.QPointF(10, 10)) * scale
    p2 = (offset + QtCore.QPointF(40, 40)) * scale

    view.canvas.mousePressEvent(_Ev(p1))
    view.canvas.mouseMoveEvent(_Ev(p2))
    with qtbot.waitSignal(view.canvas.new_shape, timeout=2000):
        view.canvas.mousePressEvent(_Ev(p2))

    assert len(list(view.label_list)) == 1

    out = tmp_path / "out.json"
    view.save_labels(str(out))
    assert out.exists()
