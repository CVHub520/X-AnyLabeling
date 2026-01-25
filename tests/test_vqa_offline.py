from pathlib import Path

from PyQt6 import QtGui

from anylabeling.views.labeling.widgets.vqa_dialog import VQADialog
from anylabeling.views.mainwindow import MainWindow


def _write_png(path: Path) -> None:
    image = QtGui.QImage(32, 32, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("white"))
    assert image.save(str(path))


def test_vqa_dialog_refreshes_from_main_window(qtbot, qapp, tmp_path, xanylabeling_workdir):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_png(images_dir / "a.png")
    _write_png(images_dir / "b.png")

    win = MainWindow(qapp, config=xanylabeling_workdir)
    qtbot.addWidget(win)
    view = win.labeling_widget.view
    view.import_image_folder(str(images_dir), load=True)
    view.async_exif_scanner.stop_scan()
    qtbot.waitUntil(lambda: view.filename is not None, timeout=2000)

    dlg = VQADialog(view)
    qtbot.addWidget(dlg)
    dlg.refresh_data()
    assert len(dlg.image_files) == 2
    dlg.close()

