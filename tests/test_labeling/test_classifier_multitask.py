import os
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets.classifier_dialog import (
    ClassifierDialog,
)


def test_classifier_status_tracks_main_window():
    from types import SimpleNamespace

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    parent.image_list = []
    parent.filename = None
    parent.flag_widget = QtWidgets.QListWidget()
    parent.canvas = QtWidgets.QWidget()
    parent.canvas.shapes = []
    parent.image = QtGui.QImage(100, 100, QtGui.QImage.Format.Format_RGB32)
    parent.attributes = {}
    parent.attribute_widget_types = {}
    parent.canvas_adjustment = SimpleNamespace(
        status_buttons={
            status: QtWidgets.QRadioButton(parent)
            for status in (False, True, "failed")
        }
    )
    parent._annotation_checked = Mock(return_value=False)
    parent.set_annotation_checked = Mock(
        side_effect=lambda status: parent.canvas_adjustment.status_buttons[
            status
        ].setChecked(True)
    )
    dialog = ClassifierDialog(parent)
    parent.filename = "image.jpg"
    dialog.update_image_display()
    dialog.is_image_labeled = Mock(return_value=False)
    dialog.update_filename_label()
    assert "❌" in dialog.filename_label.text()
    dialog.switch_to_multitask()
    assert "❌" not in dialog.filename_label.text()
    dialog.switch_to_multiclass()
    assert "❌" in dialog.filename_label.text()
    assert dialog.status_buttons[False].isChecked()
    dialog.status_buttons[True].click()
    parent.set_annotation_checked.assert_called_once_with(True)
    assert parent.canvas_adjustment.status_buttons[True].isChecked()
    parent.canvas_adjustment.status_buttons["failed"].setChecked(True)
    assert dialog.status_buttons["failed"].isChecked()
    parent._annotation_checked.return_value = False
    dialog.update_image_display()
    assert dialog.status_buttons[False].isChecked()
    dialog.image_label.set_fit_mode("width")
    dialog.image_label.zoom_factor = 3.0
    dialog.image_label.pan_offset = QtCore.QPointF(50, 60)
    dialog.update_image_display()
    assert dialog.image_label.zoom_factor == 3.0
    parent.filename = "next.jpg"
    dialog.update_image_display()
    assert dialog.image_label.fit_mode == "window"
    assert dialog.image_label.zoom_factor == 1.0
    assert dialog.image_label.pan_offset == QtCore.QPointF()
    assert dialog.fit_window_button.isChecked()
    dialog.close()
    parent.close()


def test_missing_attributes_prompt_without_selected_object():
    from types import SimpleNamespace
    from anylabeling.views.labeling.classifier.widgets import MultiTaskForm

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shape = Shape(label="person", shape_type="rectangle")
    owner = SimpleNamespace(
        canvas=SimpleNamespace(shapes=[shape]),
        attributes={},
        attribute_widget_types={},
    )
    form = MultiTaskForm(owner)
    form.resize(500, 400)
    form.show()
    app.processEvents()
    label = form.findChild(QtWidgets.QLabel)
    assert label.text() == "Please load an object attributes configuration file first."
    assert label.alignment() == QtCore.Qt.AlignmentFlag.AlignCenter
    assert abs(label.geometry().center().y() - form.rect().center().y()) <= 1
    owner.attributes = {"person": {"color": ["red"]}}
    form.show_object(-1)
    assert not any(
        form.layout_box.itemAt(i).widget()
        for i in range(form.layout_box.count())
    )
    form.close()


def test_multitask_select_edit_and_preserve_unset_values():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    parent.image_list = []
    parent.filename = None
    parent.flag_widget = QtWidgets.QListWidget()
    parent.canvas = QtWidgets.QWidget()
    parent.image = QtGui.QImage(100, 100, QtGui.QImage.Format.Format_RGB32)
    parent.set_dirty = Mock()
    parent.attributes = {
        "person": {
            "color": ["red", "blue"],
            "note": "",
            "visible": ["yes", "no"],
        },
        "car": {"color": ["red", "blue"]},
    }
    parent.attribute_widget_types = {
        "person": {"note": "lineedit", "visible": "radiobutton"}
    }
    person = Shape(label="person", shape_type="rectangle", group_id=1)
    person.points = [
        QtCore.QPointF(5, 5),
        QtCore.QPointF(40, 5),
        QtCore.QPointF(40, 40),
        QtCore.QPointF(5, 40),
    ]
    car = Shape(label="car", shape_type="rectangle", group_id=2)
    car.points = [
        QtCore.QPointF(55, 55),
        QtCore.QPointF(95, 55),
        QtCore.QPointF(95, 95),
        QtCore.QPointF(55, 95),
    ]
    parent.canvas.shapes = [person, car]
    dialog = ClassifierDialog(parent)
    dialog.switch_to_multitask()
    dialog.image_label.setPixmap(QtGui.QPixmap.fromImage(parent.image))
    assert dialog.object_combo.count() == 3
    assert dialog.object_combo.currentIndex() == 0
    assert dialog.object_combo.itemText(0) == "All objects"
    assert dialog.object_combo.itemText(1) == "person (1)"
    assert dialog.object_label.text().endswith("(0/2)")
    assert dialog.image_label.selected_object == -1
    assert not dialog.image_label.mask_background
    dialog.mask_background_button.click()
    assert dialog.image_label.mask_background
    dialog.mask_background_button.click()
    assert not dialog.image_label.mask_background
    assert dialog.show_labels_button.isChecked()
    dialog.show_labels_button.click()
    assert not dialog.image_label.show_object_labels
    dialog.show_labels_button.click()
    assert dialog.image_label.show_object_labels
    assert person.attributes == {}
    assert not parent.set_dirty.called
    dialog.show()
    dialog.activateWindow()
    app.processEvents()
    popup_item_height = dialog.object_combo.view().sizeHintForRow(0)
    assert popup_item_height == dialog.object_combo.height()
    rect = dialog.image_label.image_rect()
    point = QtCore.QPoint(
        int(rect.left() + rect.width() * 0.75),
        int(rect.top() + rect.height() * 0.75),
    )
    QtTest.QTest.mouseClick(
        dialog.image_label, QtCore.Qt.MouseButton.LeftButton, pos=point
    )
    assert dialog.object_combo.currentIndex() == 2
    assert dialog.object_label.text().endswith("(2/2)")
    blank = QtCore.QPoint(
        int(rect.left() + rect.width() * 0.5),
        int(rect.top() + rect.height() * 0.5),
    )
    QtTest.QTest.mouseClick(
        dialog.image_label, QtCore.Qt.MouseButton.LeftButton, pos=blank
    )
    assert dialog.object_combo.currentIndex() == 0
    assert dialog.next_object_button.isEnabled()
    QtTest.QTest.mouseClick(
        dialog.image_label, QtCore.Qt.MouseButton.LeftButton, pos=point
    )
    assert dialog.object_combo.currentIndex() == 2
    parent.image_list = ["first.jpg", "second.jpg"]
    dialog.image_files = parent.image_list
    parent.filename = "first.jpg"
    dialog.switch_to_image = Mock()
    assert dialog.main_splitter.widget(0).minimumWidth() == 350
    dialog.activateWindow()
    app.processEvents()
    dialog.image_label.setFocus()
    QtTest.QTest.keyClick(
        dialog.image_label,
        QtCore.Qt.Key.Key_F,
        QtCore.Qt.KeyboardModifier.ControlModifier
        | QtCore.Qt.KeyboardModifier.ShiftModifier,
    )
    assert dialog.image_label.fit_mode == "width"
    assert dialog.image_label.selected_object == -1
    dialog.select_object(1)
    QtTest.QTest.keyClick(
        dialog.image_label,
        QtCore.Qt.Key.Key_F,
        QtCore.Qt.KeyboardModifier.ControlModifier,
    )
    assert dialog.image_label.fit_mode == "window"
    assert dialog.image_label.selected_object == -1
    dialog.select_object(1)
    QtTest.QTest.keyClick(dialog.image_label, QtCore.Qt.Key.Key_D)
    dialog.switch_to_image.assert_called_once_with(1)
    QtTest.QTest.keyClick(
        dialog.image_label,
        QtCore.Qt.Key.Key_A,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
    )
    assert dialog.object_combo.currentIndex() == 1
    QtTest.QTest.keyClick(
        dialog.image_label,
        QtCore.Qt.Key.Key_D,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
    )
    assert dialog.object_combo.currentIndex() == 2
    dialog.previous_object_button.click()
    assert dialog.object_combo.currentIndex() == 1
    dialog.previous_object_button.click()
    assert dialog.object_combo.currentIndex() == 2
    dialog.next_object_button.click()
    assert dialog.object_combo.currentIndex() == 1
    layout = dialog.multitask_form.layout_box
    radio_group = next(
        layout.itemAt(i).widget()
        for i in range(layout.count())
        if layout.itemAt(i).widget()
        and layout.itemAt(i).widget().findChildren(QtWidgets.QRadioButton)
    )
    app.processEvents()
    assert radio_group.height() == radio_group.sizeHint().height()
    combo = next(
        layout.itemAt(i).widget()
        for i in range(layout.count())
        if isinstance(layout.itemAt(i).widget(), QtWidgets.QComboBox)
    )
    combo.setCurrentText("blue")
    assert person.attributes == {"color": "blue"}
    search = next(
        layout.itemAt(i).widget().findChild(QtWidgets.QLineEdit)
        for i in range(layout.count())
        if layout.itemAt(i).widget()
        and layout.itemAt(i).widget().findChild(QtWidgets.QLineEdit)
    )
    search.setText("red")
    assert combo.view().isRowHidden(combo.findText("blue"))
    assert not combo.view().isRowHidden(combo.findText("red"))
    assert combo.currentText() == "blue"
    assert person.attributes == {"color": "blue"}
    search.clear()
    assert not combo.view().isRowHidden(combo.findText("blue"))
    assert parent.set_dirty.called
    line = next(
        layout.itemAt(i).widget()
        for i in range(layout.count())
        if isinstance(layout.itemAt(i).widget(), QtWidgets.QPlainTextEdit)
    )
    line.setFocus()
    app.processEvents()
    QtTest.QTest.keyClicks(line, "a1d")
    assert person.attributes["note"] == "a1d"
    QtTest.QTest.keyClick(line, QtCore.Qt.Key.Key_Return)
    QtTest.QTest.keyClicks(line, "more text")
    assert person.attributes["note"] == "a1d\nmore text"
    assert not dialog.object_shortcuts[0].isEnabled()
    assert line.lineWrapMode() == QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth
    assert combo.styleSheet() == dialog.object_combo.styleSheet()
    dialog.switch_to_multiclass()
    assert not dialog.is_multitask
    assert not dialog.image_label.object_shapes
    dialog.close()
    parent.close()


def test_preview_fit_zoom_and_rectangle_closure():
    from anylabeling.views.labeling.classifier.widgets import (
        ClassificationImagePreview,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    preview = ClassificationImagePreview()
    preview.resize(800, 400)
    preview.multitask_enabled = True
    preview.setPixmap(QtGui.QPixmap(200, 400))
    preview.show()
    app.processEvents()
    fit = preview.image_rect()
    assert fit.width() < fit.height()
    assert fit.height() <= preview.height()
    preview.set_fit_mode("width")
    width_fit = preview.image_rect()
    assert abs(width_fit.width() - (preview.width() - 21)) < 1
    shape = Shape(label="box", shape_type="rectangle")
    shape.points = [
        QtCore.QPointF(10, 10),
        QtCore.QPointF(40, 10),
        QtCore.QPointF(40, 40),
        QtCore.QPointF(10, 40),
    ]
    path = preview.object_path(shape)
    assert path.elementAt(path.elementCount() - 1).x == 10
    assert path.elementAt(path.elementCount() - 1).y == 10
    shape.points = [QtCore.QPointF(10, 10), QtCore.QPointF(40, 40)]
    assert preview.object_path(shape).contains(QtCore.QPointF(20, 20))
    preview.set_fit_mode("window")
    anchor = QtCore.QPointF(300, 180)
    before = preview.image_rect()
    relative_before = (anchor - before.topLeft()) / before.width()
    event = QtGui.QWheelEvent(
        anchor,
        anchor,
        QtCore.QPoint(),
        QtCore.QPoint(0, 120),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QtWidgets.QApplication.sendEvent(preview, event)
    after = preview.image_rect()
    relative_after = (anchor - after.topLeft()) / after.width()
    assert after.width() > before.width()
    assert (relative_after - relative_before).manhattanLength() < 1e-6
    preview.close()


def test_left_drag_pans_without_selecting_object():
    from anylabeling.views.labeling.classifier.widgets import (
        ClassificationImagePreview,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    preview = ClassificationImagePreview()
    preview.resize(400, 400)
    preview.multitask_enabled = True
    preview.setPixmap(QtGui.QPixmap(100, 100))
    shape = Shape(label="box", shape_type="rectangle")
    shape.points = [QtCore.QPointF(0, 0), QtCore.QPointF(100, 100)]
    preview.set_objects([shape], QtCore.QSize(100, 100), -1)
    selected = Mock()
    preview.object_selected.connect(selected)
    preview.show()
    app.processEvents()
    press = QtCore.QPointF(100, 100)
    move = QtCore.QPointF(130, 140)
    for event_type, point, button, buttons in (
        (
            QtCore.QEvent.Type.MouseButtonPress,
            press,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
        ),
        (
            QtCore.QEvent.Type.MouseMove,
            move,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.LeftButton,
        ),
        (
            QtCore.QEvent.Type.MouseButtonRelease,
            move,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.NoButton,
        ),
    ):
        event = QtGui.QMouseEvent(
            event_type,
            point,
            point,
            button,
            buttons,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(preview, event)
    assert preview.pan_offset == QtCore.QPointF(30, 40)
    selected.assert_not_called()
    QtTest.QTest.mouseClick(
        preview, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(120, 120)
    )
    selected.assert_called_once_with(0)
    preview.close()


def test_zoomed_preview_clips_image_and_shapes_to_rounded_viewport():
    from anylabeling.views.labeling.classifier.widgets import (
        ClassificationImagePreview,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    preview = ClassificationImagePreview()
    preview.resize(200, 200)
    preview.multitask_enabled = True
    pixmap = QtGui.QPixmap(100, 100)
    pixmap.fill(QtGui.QColor("white"))
    preview.setPixmap(pixmap)
    preview.zoom_factor = 3
    shape = Shape(label="box", shape_type="rectangle")
    shape.points = [QtCore.QPointF(0, 0), QtCore.QPointF(100, 100)]
    preview.set_objects([shape], QtCore.QSize(100, 100), 0)
    preview.show()
    app.processEvents()
    rendered = preview.grab().toImage()
    corner = rendered.pixelColor(11, 11)
    inside = rendered.pixelColor(30, 30)
    assert (corner.red(), corner.green(), corner.blue()) != (
        inside.red(),
        inside.green(),
        inside.blue(),
    )
    assert inside.red() > 160 and inside.green() > 180
    preview.close()


def test_label_background_is_visible_above_image_and_can_be_hidden():
    from anylabeling.views.labeling.classifier.widgets import (
        ClassificationImagePreview,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    preview = ClassificationImagePreview()
    preview.resize(300, 200)
    preview.multitask_enabled = True
    pixmap = QtGui.QPixmap(100, 100)
    pixmap.fill(QtGui.QColor("white"))
    preview.setPixmap(pixmap)
    shape = Shape(
        label="object", shape_type="rectangle", group_id=1, score=0.75
    )
    shape.line_color = QtGui.QColor("red")
    shape.points = [QtCore.QPointF(0, 0), QtCore.QPointF(60, 60)]
    preview.set_objects([shape], QtCore.QSize(100, 100), -1)
    preview.show()
    app.processEvents()
    image = preview.grab().toImage()
    x = int(preview.image_rect().left()) + 15
    shown = image.pixelColor(x, 5)
    assert shown.red() > 200 and shown.green() < 80
    preview.show_object_labels = False
    preview.update()
    hidden = preview.grab().toImage().pixelColor(x, 5)
    assert hidden != shown
    preview.close()


def test_classification_previews_follow_splitter_in_all_modes():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    parent.image_list = []
    parent.filename = None
    parent.flag_widget = QtWidgets.QListWidget()
    parent.canvas = QtWidgets.QWidget()
    parent.canvas.shapes = []
    parent.image = QtGui.QImage(100, 100, QtGui.QImage.Format.Format_RGB32)
    parent.attributes = {}
    parent.attribute_widget_types = {}
    dialog = ClassifierDialog(parent)
    dialog.resize(1600, 800)
    dialog.show()
    app.processEvents()
    for switch in (
        lambda: None,
        dialog.switch_to_multilabel,
        dialog.switch_to_multitask,
    ):
        switch()
        app.processEvents()
        row = dialog.object_row if dialog.is_multitask else dialog.category_row
        assert row.y() - dialog.export_button.geometry().bottom() - 1 >= 14
        if not dialog.is_multitask:
            search_right = dialog.category_search.mapTo(
                dialog.category_row.parentWidget(),
                dialog.category_search.rect().topRight(),
            ).x()
            assert search_right == dialog.multilabel_button.geometry().right()
        dialog.image_label.setPixmap(QtGui.QPixmap(1000, 500))
        dialog.main_splitter.setSizes([360, 800])
        app.processEvents()
        small = dialog.image_label.width()
        assert (
            dialog.image_label.geometry().right()
            <= dialog.image_container.width()
        )
        dialog.main_splitter.setSizes([650, 510])
        app.processEvents()
        assert dialog.image_label.width() > small
        image_rect = dialog.image_label.image_rect()
        assert image_rect.left() >= 0
        assert image_rect.right() <= dialog.image_label.width()
    dialog.close()
    parent.close()


def test_maximized_classifier_starts_with_compact_right_panel():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    parent.image_list = []
    parent.filename = None
    parent.flag_widget = QtWidgets.QListWidget()
    parent.canvas = QtWidgets.QWidget()
    parent.canvas.shapes = []
    parent.image = QtGui.QImage(100, 100, QtGui.QImage.Format.Format_RGB32)
    parent.attributes = {}
    parent.attribute_widget_types = {}
    dialog = ClassifierDialog(parent)
    dialog.show()
    QtTest.QTest.qWait(20)
    dialog.showMaximized()
    QtTest.QTest.qWait(20)
    dialog.resize(1600, 800)
    app.processEvents()
    with patch.object(dialog, "isMaximized", return_value=True):
        dialog.set_maximized_splitter_size()
    compact_width = dialog.main_splitter.sizes()[1]
    assert compact_width <= (
        dialog.main_splitter.widget(1).minimumSizeHint().width() + 1
    )
    dialog.main_splitter.setSizes([0, dialog.main_splitter.width()])
    assert dialog.main_splitter.sizes()[1] > compact_width
    dialog.close()
    parent.close()


def test_preview_uses_canvas_colors_opacity_and_selected_vertices():
    from types import SimpleNamespace
    from anylabeling.views.labeling.classifier.widgets import (
        ClassificationImagePreview,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    preview = ClassificationImagePreview()
    preview.resize(200, 200)
    preview.multitask_enabled = True
    preview.show_object_labels = False
    pixmap = QtGui.QPixmap(100, 100)
    pixmap.fill(QtGui.QColor("white"))
    preview.setPixmap(pixmap)
    preview.source_canvas = SimpleNamespace(
        display_mode="label",
        show_masks=True,
        mask_opacity=80,
        shape_opacity=1.0,
    )
    shape = Shape(label="box", shape_type="rectangle")
    shape.line_color = QtGui.QColor("#ff00ff")
    shape.vertex_fill_color = QtGui.QColor("#0088ff")
    shape.points = [
        QtCore.QPointF(20, 20),
        QtCore.QPointF(80, 20),
        QtCore.QPointF(80, 80),
        QtCore.QPointF(20, 80),
    ]
    preview.set_objects([shape], QtCore.QSize(100, 100), -1)
    preview.show()
    app.processEvents()
    image = preview.grab().toImage()
    center = image.pixelColor(100, 100)
    assert (
        center.red() > 240
        and 165 <= center.green() <= 180
        and center.blue() > 240
    )
    vertex = preview.object_transform().map(shape.points[0]).toPoint()
    before = image.pixelColor(vertex)
    preview.set_objects([shape], QtCore.QSize(100, 100), 0)
    selected = preview.grab().toImage()
    after = selected.pixelColor(vertex)
    assert after.blue() > 200 and after.red() < 40
    assert before != after
    assert selected.pixelColor(100, 100) == center
    preview.source_canvas.show_masks = False
    preview.update()
    assert preview.grab().toImage().pixelColor(100, 100) == QtGui.QColor(
        "white"
    )
    preview.close()


def test_selected_object_isolated_and_labels_only_show_class():
    from types import SimpleNamespace
    from anylabeling.views.labeling.classifier.widgets import (
        ClassificationImagePreview,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    preview = ClassificationImagePreview()
    preview.resize(200, 200)
    preview.multitask_enabled = True
    pixmap = QtGui.QPixmap(100, 100)
    pixmap.fill(QtGui.QColor("white"))
    preview.setPixmap(pixmap)
    preview.source_canvas = SimpleNamespace(
        display_mode="label",
        show_masks=True,
        mask_opacity=80,
        shape_opacity=1.0,
    )
    first = Shape(
        label="person", shape_type="rectangle", group_id=7, score=0.89
    )
    first.line_color = QtGui.QColor("red")
    first.points = [QtCore.QPointF(10, 10), QtCore.QPointF(40, 40)]
    second = Shape(label="car", shape_type="rectangle")
    second.line_color = QtGui.QColor("blue")
    second.points = [QtCore.QPointF(60, 60), QtCore.QPointF(90, 90)]
    preview.set_objects([first, second], QtCore.QSize(100, 100), 0)
    preview.show()
    app.processEvents()
    selected = preview.grab().toImage()
    second_pos = (
        preview.object_transform().map(QtCore.QPointF(75, 75)).toPoint()
    )
    assert selected.pixelColor(second_pos) == QtGui.QColor("white")
    inside_pos = preview.object_transform().map(QtCore.QPointF(25, 25)).toPoint()
    preview.mask_background = True
    masked = preview.grab().toImage()
    assert masked.pixelColor(second_pos) == QtGui.QColor("black")
    assert masked.pixelColor(inside_pos) == selected.pixelColor(inside_pos)
    preview.set_objects([first, second], QtCore.QSize(100, 100), -1)
    preview.source_canvas.show_masks = False
    assert preview.grab().toImage().pixelColor(second_pos) == QtGui.QColor("white")
    preview.source_canvas.show_masks = True
    preview.mask_background = False
    preview.set_objects([first, second], QtCore.QSize(100, 100), 0)
    painter = Mock()
    painter.font.return_value = QtGui.QFont("Arial", 8)
    preview.paint_object_labels(painter)
    painter.drawText.assert_not_called()
    preview.set_objects([first, second], QtCore.QSize(100, 100), -1)
    assert preview.grab().toImage().pixelColor(second_pos) != QtGui.QColor(
        "white"
    )
    painter.reset_mock()
    preview.paint_object_labels(painter)
    assert [call.args[1] for call in painter.drawText.call_args_list] == [
        "person",
        "car",
    ]
    preview.close()


def test_multitask_help_uses_interface_language():
    from types import SimpleNamespace
    from unittest.mock import patch

    dialog = SimpleNamespace(
        label_widget=SimpleNamespace(_config={"language": "zh_CN"})
    )
    with patch(
        "anylabeling.views.labeling.widgets.classifier_dialog.general.open_url"
    ) as open_url:
        ClassifierDialog.open_multitask_help(dialog)
        open_url.assert_called_once_with(
            "https://xanylabeling.com/zh-Hans/examples/classification/shape-level"
        )
        open_url.reset_mock()
        dialog.label_widget._config["language"] = "ja_JP"
        ClassifierDialog.open_multitask_help(dialog)
        open_url.assert_called_once_with(
            "https://xanylabeling.com/examples/classification/shape-level"
        )


def test_double_click_and_dropdown_focus_object():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    parent.image_list = []
    parent.filename = None
    parent.flag_widget = QtWidgets.QListWidget()
    parent.canvas = QtWidgets.QWidget()
    parent.image = QtGui.QImage(100, 100, QtGui.QImage.Format.Format_RGB32)
    parent.attributes = {}
    parent.attribute_widget_types = {}
    shape = Shape(label="person", shape_type="rectangle")
    shape.points = [QtCore.QPointF(10, 20), QtCore.QPointF(25, 35)]
    parent.canvas.shapes = [shape]
    dialog = ClassifierDialog(parent)
    dialog.switch_to_multitask()
    dialog.show()
    app.processEvents()
    dialog.image_label.setPixmap(QtGui.QPixmap.fromImage(parent.image))
    preview = dialog.image_label
    center = preview.object_transform().map(QtCore.QPointF(17.5, 27.5))
    QtTest.QTest.mouseDClick(
        preview, QtCore.Qt.MouseButton.LeftButton, pos=center.toPoint()
    )
    assert dialog.object_combo.currentIndex() == 1
    assert preview.zoom_factor > 1
    viewport = QtCore.QRectF(preview.rect()).adjusted(10.5, 10.5, -10.5, -10.5)
    centered = preview.object_transform().map(QtCore.QPointF(17.5, 27.5))
    assert (centered - viewport.center()).manhattanLength() < 0.01
    preview.set_fit_mode("window")
    dialog.select_object(-1)
    dialog.object_combo.setCurrentIndex(1)
    dialog.object_combo.activated.emit(1)
    centered = preview.object_transform().map(QtCore.QPointF(17.5, 27.5))
    assert preview.zoom_factor > 1
    assert (centered - viewport.center()).manhattanLength() < 0.01
    QtTest.QTest.mouseClick(
        preview, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(12, 12)
    )
    assert dialog.object_combo.currentIndex() == 0
    assert preview.fit_mode == "window"
    assert preview.zoom_factor == 1.0
    assert preview.pan_offset == QtCore.QPointF()
    assert dialog.fit_window_button.isChecked()
    dialog.next_object_button.click()
    assert dialog.object_combo.currentIndex() == 1
    assert preview.zoom_factor > 1
    centered = preview.object_transform().map(QtCore.QPointF(17.5, 27.5))
    assert (centered - viewport.center()).manhattanLength() < 0.01
    dialog.close()
    parent.close()
