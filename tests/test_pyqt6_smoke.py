from pathlib import Path

from PyQt6 import QtCore, QtGui

from anylabeling.views.mainwindow import MainWindow


def test_smoke_import_and_construct_core_widgets(qtbot):
    from anylabeling.views.labeling.widgets.canvas import Canvas
    from anylabeling.views.labeling.widgets.file_dialog_preview import (
        FileDialogPreview,
    )
    from anylabeling.views.labeling.widgets.label_dialog import LabelDialog
    from anylabeling.views.labeling.widgets.label_list_widget import (
        LabelListWidget,
    )
    from anylabeling.views.labeling.widgets.searchable_model_dropdown import (
        SearchableModelDropdownPopup,
    )

    canvas = Canvas(parent=None)
    qtbot.addWidget(canvas)
    qtbot.addWidget(LabelListWidget())
    qtbot.addWidget(FileDialogPreview())
    qtbot.addWidget(
        SearchableModelDropdownPopup(
            {
                "Provider": {
                    "model-a": {"display_name": "model-a", "favorite": False},
                }
            }
        )
    )
    qtbot.addWidget(
        LabelDialog(
            labels=["a", "b"],
            sort_labels=True,
            show_text_field=True,
            completion="startswith",
            fit_to_content={"row": False, "column": True},
            flags={},
        )
    )


def test_smoke_label_list_select_item(qtbot):
    from anylabeling.views.labeling.widgets.label_list_widget import (
        LabelListWidget,
        LabelListWidgetItem,
    )

    w = LabelListWidget()
    qtbot.addWidget(w)
    item = LabelListWidgetItem("a")
    w.add_iem(item)
    w.select_item(item)
    assert item in w.selected_items()


def test_smoke_canvas_paint_event(qtbot):
    from anylabeling.views.labeling.widgets.canvas import Canvas

    canvas = Canvas(parent=None)
    qtbot.addWidget(canvas)
    pixmap = QtGui.QPixmap(10, 10)
    pixmap.fill(QtGui.QColor("white"))
    canvas.load_pixmap(pixmap, clear_shapes=True)

    canvas.resize(10, 10)
    canvas.show()

    event = QtGui.QPaintEvent(QtCore.QRect(0, 0, 10, 10))
    canvas.paintEvent(event)


def test_smoke_canvas_bounded_move_shapes_pointf(qtbot):
    from anylabeling.views.labeling.widgets.canvas import Canvas
    from anylabeling.views.labeling.shape import Shape

    canvas = Canvas(parent=None)
    qtbot.addWidget(canvas)
    pixmap = QtGui.QPixmap(10, 10)
    pixmap.fill(QtGui.QColor("white"))
    canvas.load_pixmap(pixmap, clear_shapes=True)

    shape = Shape(shape_type="rectangle")
    shape.points = [
        QtCore.QPointF(1, 1),
        QtCore.QPointF(2, 1),
        QtCore.QPointF(2, 2),
        QtCore.QPointF(1, 2),
    ]
    shape.close()

    canvas.offsets = (QtCore.QPointF(0, 0), QtCore.QPointF(10, 10))
    canvas.prev_point = QtCore.QPointF(8, 8)
    assert canvas.bounded_move_shapes([shape], QtCore.QPointF(9, 9)) is True


def test_smoke_import_image_folder(qtbot, qapp, xanylabeling_workdir):
    win = MainWindow(qapp, config=xanylabeling_workdir)
    qtbot.addWidget(win)
    view = win.labeling_widget.view

    demo_dir = Path(__file__).resolve().parent.parent / "assets" / "demo"
    view.import_image_folder(str(demo_dir), load=False)
    view.async_exif_scanner.stop_scan()

    assert view.file_list_widget.count() > 0


def test_smoke_update_file_menu(qtbot, qapp, xanylabeling_workdir):
    win = MainWindow(qapp, config=xanylabeling_workdir)
    qtbot.addWidget(win)
    win.labeling_widget.view.update_file_menu()
