from pathlib import Path

from PyQt6 import QtCore, QtGui

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.mainwindow import MainWindow


def _write_png(path: Path) -> None:
    image = QtGui.QImage(10, 10, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("white"))
    assert image.save(str(path))


def test_labeling_import_image_folder_populates_file_list(
    qtbot, qapp, tmp_path, xanylabeling_workdir
):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_png(images_dir / "a.png")
    _write_png(images_dir / "b.png")

    win = MainWindow(qapp, config=xanylabeling_workdir)
    qtbot.addWidget(win)
    view = win.labeling_widget.view

    view.import_image_folder(str(images_dir), load=False)
    view.async_exif_scanner.stop_scan()

    assert view.file_list_widget.count() == 2
    assert view.actions.open_next_image.isEnabled() is True


def test_labeling_import_dropped_image_files_dedup_and_checkstate(
    qtbot, qapp, tmp_path, xanylabeling_workdir
):
    import json

    from anylabeling.views.labeling.schema import create_xlabel_template

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img1 = images_dir / "a.png"
    img2 = images_dir / "b.png"
    _write_png(img1)
    _write_png(img2)
    (images_dir / "a.json").write_text(
        json.dumps(
            create_xlabel_template(
                image_path="a.png",
                image_data=None,
                image_height=10,
                image_width=10,
            )
        ),
        encoding="utf-8",
    )

    win = MainWindow(qapp, config=xanylabeling_workdir)
    qtbot.addWidget(win)
    view = win.labeling_widget.view

    view.import_dropped_image_files(
        [
            str(img1),
            str(img1),
            str(img2),
            str(images_dir / "not_image.txt"),
        ]
    )
    view.async_exif_scanner.stop_scan()

    assert view.file_list_widget.count() == 2
    item0 = view.file_list_widget.item(0)
    assert item0.checkState() == QtCore.Qt.CheckState.Checked


def test_labeling_undo_shape_edit_updates_label_list(qtbot, qapp, xanylabeling_workdir):
    win = MainWindow(qapp, config=xanylabeling_workdir)
    qtbot.addWidget(win)
    view = win.labeling_widget.view
    view._config["auto_save"] = False

    shape = Shape(shape_type="rectangle")
    shape.label = "a"
    shape.points = [
        QtCore.QPointF(1, 1),
        QtCore.QPointF(2, 1),
        QtCore.QPointF(2, 2),
        QtCore.QPointF(1, 2),
    ]
    shape.close()

    view.load_shapes([shape], replace=True)
    assert len(list(view.label_list)) == 1

    shape.points[0] = QtCore.QPointF(3, 3)
    view.canvas.store_shapes()
    assert view.canvas.is_shape_restorable is True

    view.undo_shape_edit()
    assert len(list(view.label_list)) == 1
    assert view.canvas.shapes[0].points[0] == QtCore.QPointF(1, 1)
