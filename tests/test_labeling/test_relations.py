import json

from PIL import Image
from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

from anylabeling.services.auto_labeling.types import AutoLabelingResult
from anylabeling.views.labeling.shape import Shape

pytest_plugins = ["tests.test_labeling.test_check_status"]


def rectangle(label, shape_id, x):
    shape = Shape(label=label, shape_type="rectangle")
    shape.points = [
        QtCore.QPointF(x, 20),
        QtCore.QPointF(x + 30, 20),
        QtCore.QPointF(x + 30, 50),
        QtCore.QPointF(x, 50),
    ]
    shape.other_data["shape_id"] = shape_id
    shape.close()
    return shape


def test_relation_round_trip_edit_and_delete(window, tmp_path):
    widget = window.labeling_widget.view
    path = str(tmp_path / "image.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("person", "a", 20), rectangle("horse", "b", 190)]
    relation = {
        "subject_id": "a",
        "object_id": "b",
        "label": "riding",
        "score": 0.8,
    }
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=[relation])
    )
    assert widget.canvas.relations == [relation]
    assert widget.other_data.get("description", "") == ""
    assert widget.save_labels(widget.get_label_file())
    data = json.loads(open(widget.get_label_file(), encoding="utf-8").read())
    assert data["relations"] == [relation]
    widget.filename = None
    assert widget.load_file(path)
    assert widget.canvas.relations == [relation]
    assert len(widget.canvas._relation_segments()) == 1

    def change_label():
        dialog = QtWidgets.QApplication.activeModalWidget()
        dialog.findChild(QtWidgets.QLineEdit).setText("beside")
        dialog.accept()

    QtCore.QTimer.singleShot(0, change_label)
    widget.edit_relation(0)
    assert widget.canvas.relations[0]["label"] == "beside"
    assert widget.canvas.relations[0]["score"] is None
    assert widget.save_labels(widget.get_label_file())
    widget.filename = None
    assert widget.load_file(path)
    assert widget.canvas.relations[0]["label"] == "beside"

    def delete_relation():
        dialog = QtWidgets.QApplication.activeModalWidget()
        dialog.findChild(QtWidgets.QPushButton, "relationDeleteButton").click()

    QtCore.QTimer.singleShot(0, delete_relation)
    widget.edit_relation(0)
    assert widget.canvas.relations == []
    assert widget.save_labels(widget.get_label_file())
    assert (
        json.loads(open(widget.get_label_file(), encoding="utf-8").read())[
            "relations"
        ]
        == []
    )


def test_relation_canvas_direction_hit_and_copy(window):
    widget = window.labeling_widget.view
    canvas = widget.canvas
    canvas.pixmap = QtGui.QPixmap(300, 120)
    canvas.resize(300, 120)
    canvas.shapes = [rectangle("a", "a", 20), rectangle("b", "b", 190)]
    canvas.relations = [
        {"subject_id": "a", "object_id": "b", "label": "riding"},
        {"subject_id": "b", "object_id": "a", "label": "carrying"},
    ]
    segments = canvas._relation_segments()
    assert len(segments) == 2
    assert segments[0][1].x() < segments[0][2].x()
    assert segments[1][1].x() > segments[1][2].x()
    assert segments[0][1].y() != segments[1][1].y()
    copied = canvas.prepare_pasted_shapes(canvas.shapes)
    assert all("shape_id" not in shape.other_data for shape in copied)
    canvas.show()
    QtWidgets.QApplication.processEvents()
    assert not canvas.grab().isNull()


def test_create_relation_and_edit_by_double_click(window, tmp_path):
    widget = window.labeling_widget.view
    path = str(tmp_path / "manual.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("person", "a", 20), rectangle("horse", "b", 190)]
    widget.load_shapes(shapes)
    widget.shape_selection_changed(shapes)

    def create():
        dialog = QtWidgets.QApplication.activeModalWidget()
        dialog.findChild(QtWidgets.QLineEdit).setText("riding")
        dialog.accept()

    QtCore.QTimer.singleShot(0, create)
    widget.create_relation()
    assert widget.canvas.relations[0]["label"] == "riding"
    canvas = widget.canvas
    canvas.show()
    QtWidgets.QApplication.processEvents()
    _, start, end = canvas._relation_segments()[0]
    midpoint = (start + end) / 2
    click = (midpoint + canvas.offset_to_center()) * canvas.scale
    QtTest.QTest.mouseClick(
        canvas, QtCore.Qt.MouseButton.LeftButton, pos=click.toPoint()
    )
    assert canvas.selected_relation is canvas.relations[0]

    def edit():
        dialog = QtWidgets.QApplication.activeModalWidget()
        assert dialog is not None
        dialog.findChild(QtWidgets.QLineEdit).setText("beside")
        dialog.accept()

    QtCore.QTimer.singleShot(0, edit)
    QtTest.QTest.mouseDClick(
        canvas, QtCore.Qt.MouseButton.LeftButton, pos=click.toPoint()
    )
    assert canvas.relations[0]["label"] == "beside"


def test_relation_editor_requires_double_click_on_selected_label(
    window, tmp_path
):
    widget = window.labeling_widget.view
    path = str(tmp_path / "relation-hit.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("person", "a", 20), rectangle("bicycle", "b", 190)]
    relation = {
        "subject_id": "a",
        "object_id": "b",
        "label": "riding",
        "score": 0.77,
    }
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=[relation])
    )
    canvas = widget.canvas
    canvas.relation_edit_requested.disconnect(widget.edit_relation)
    requested = []
    canvas.relation_edit_requested.connect(requested.append)
    _, start, end, _, bounds = canvas._relation_layout()[0]
    to_widget = lambda point: (
        (point + canvas.offset_to_center()) * canvas.scale
    ).toPoint()
    label = to_widget(bounds.center())
    line = to_widget(start + (end - start) * 0.2)
    QtTest.QTest.mouseDClick(
        canvas, QtCore.Qt.MouseButton.LeftButton, pos=label
    )
    assert requested == []
    QtTest.QTest.mouseClick(
        canvas, QtCore.Qt.MouseButton.LeftButton, pos=label
    )
    assert canvas.selected_relation is relation
    QtTest.QTest.mouseDClick(
        canvas, QtCore.Qt.MouseButton.LeftButton, pos=line
    )
    assert requested == []
    QtTest.QTest.mouseClick(
        canvas, QtCore.Qt.MouseButton.LeftButton, pos=label
    )
    QtTest.QTest.mouseDClick(
        canvas, QtCore.Qt.MouseButton.LeftButton, pos=label
    )
    assert requested == [0]


def test_deleted_endpoint_is_not_saved_and_restored_shape_keeps_relation(
    window, tmp_path
):
    widget = window.labeling_widget.view
    path = str(tmp_path / "delete.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("a", "a", 20), rectangle("b", "b", 190)]
    relation = {
        "subject_id": "a",
        "object_id": "b",
        "label": "beside",
        "score": None,
    }
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=[relation])
    )
    widget.canvas.selected_shapes = [shapes[0]]
    widget.remove_labels(widget.canvas.delete_selected())
    assert not widget.canvas._relation_segments()
    assert widget.save_labels(widget.get_label_file())
    data = json.loads(open(widget.get_label_file(), encoding="utf-8").read())
    assert data["relations"] == []
    widget.undo_shape_edit()
    if len(widget.canvas.shapes) == 1:
        widget.undo_shape_edit()
    assert len(widget.canvas._relation_segments()) == 1


def test_manual_relation_assigns_distinct_ids(window, tmp_path):
    widget = window.labeling_widget.view
    path = str(tmp_path / "ids.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("a", "a", 20), rectangle("b", "b", 190)]
    for shape in shapes:
        shape.other_data.clear()
    widget.load_shapes(shapes)
    widget.shape_selection_changed(shapes)

    def create():
        dialog = QtWidgets.QApplication.activeModalWidget()
        dialog.findChild(QtWidgets.QLineEdit).setText("beside")
        dialog.accept()

    QtCore.QTimer.singleShot(0, create)
    widget.create_relation()
    assert shapes[0].other_data["shape_id"] != shapes[1].other_data["shape_id"]
    assert (
        widget.canvas.relations[0]["subject_id"]
        == shapes[0].other_data["shape_id"]
    )
    assert (
        widget.canvas.relations[0]["object_id"]
        == shapes[1].other_data["shape_id"]
    )


def test_manual_relation_survives_shape_undo(window, tmp_path):
    widget = window.labeling_widget.view
    path = str(tmp_path / "undo.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("a", "a", 20), rectangle("b", "b", 190)]
    for shape in shapes:
        shape.other_data.clear()
    widget.load_shapes(shapes)
    widget.shape_selection_changed(shapes)

    def create():
        dialog = QtWidgets.QApplication.activeModalWidget()
        dialog.findChild(QtWidgets.QLineEdit).setText("beside")
        dialog.accept()

    QtCore.QTimer.singleShot(0, create)
    widget.create_relation()
    shapes[0].move_by(QtCore.QPointF(10, 0))
    widget.canvas.store_shapes()
    widget.undo_shape_edit()
    assert len(widget.canvas._relation_segments()) == 1
    assert widget.canvas.shapes[0].points[0].x() == 20


def test_manage_relations_edit_overlapping_shapes(window, tmp_path):
    widget = window.labeling_widget.view
    path = str(tmp_path / "overlap.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("a", "a", 20), rectangle("b", "b", 20)]
    relation = {
        "subject_id": "a",
        "object_id": "b",
        "label": "beside",
        "score": 0.3,
    }
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=[relation])
    )
    assert widget.canvas._relation_segments() == []

    def manage():
        dialog = QtWidgets.QApplication.activeModalWidget()
        listing = dialog.findChild(QtWidgets.QListWidget)
        assert listing.count() == 1
        listing.setCurrentRow(0)

        def edit():
            editor = QtWidgets.QApplication.activeModalWidget()
            editor.findChild(QtWidgets.QLineEdit).setText("overlapping")
            editor.accept()

        QtCore.QTimer.singleShot(0, edit)
        dialog.findChild(QtWidgets.QPushButton, "relationEditButton").click()
        dialog.accept()

    QtCore.QTimer.singleShot(0, manage)
    widget.manage_relations()
    assert widget.canvas.relations[0]["label"] == "overlapping"


def test_relation_label_selection_and_fixed_handles(window, tmp_path):
    widget = window.labeling_widget.view
    path = str(tmp_path / "label.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("a", "a", 20), rectangle("b", "b", 190)]
    relation = {
        "subject_id": "a",
        "object_id": "b",
        "label": "riding",
        "score": 0.77,
    }
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=[relation])
    )
    canvas = widget.canvas
    canvas.show()
    QtWidgets.QApplication.processEvents()
    index, start, end, text, bounds = canvas._relation_layout()[0]
    assert text == "riding · 0.77"
    assert index == 0
    to_widget = lambda point: (
        (point + canvas.offset_to_center()) * canvas.scale
    ).toPoint()
    QtTest.QTest.mouseClick(
        canvas,
        QtCore.Qt.MouseButton.LeftButton,
        pos=to_widget(bounds.center()),
    )
    assert canvas.selected_relation is canvas.relations[0]
    old_points = [
        [QtCore.QPointF(p) for p in shape.points] for shape in canvas.shapes
    ]
    old_bounds = QtCore.QRectF(bounds)
    handle = to_widget(bounds.topLeft())
    QtTest.QTest.mousePress(
        canvas, QtCore.Qt.MouseButton.LeftButton, pos=handle
    )
    assert canvas._relation_press_active
    position = QtCore.QPointF(handle + QtCore.QPoint(30, 30))
    move = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        position,
        QtCore.QPointF(canvas.mapToGlobal(position.toPoint())),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(canvas, move)
    QtTest.QTest.mouseRelease(
        canvas,
        QtCore.Qt.MouseButton.LeftButton,
        pos=handle + QtCore.QPoint(30, 30),
    )
    assert not canvas._relation_press_active
    assert canvas._relation_layout()[0][4] == old_bounds
    assert [[p for p in shape.points] for shape in canvas.shapes] == old_points
    assert canvas.selected_relation is canvas.relations[0]


def test_create_relation_action_requires_exactly_two_selected(
    window, tmp_path
):
    widget = window.labeling_widget.view
    path = str(tmp_path / "selection.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [
        rectangle("a", "a", 20),
        rectangle("b", "b", 110),
        rectangle("c", "c", 200),
    ]
    widget.load_shapes(shapes)
    assert widget.actions.create_relation in widget.actions.tool
    for count in (0, 1, 2, 3, 2, 0):
        widget.shape_selection_changed(shapes[:count])
        assert widget.actions.create_relation.isEnabled() == (count == 2)
    assert not widget.actions.create_relation.icon().isNull()
    assert not widget.actions.manage_relations.icon().isNull()
    assert not widget.actions.create_line_strip_mode.icon().isNull()


def test_relation_label_center_and_font_stable_under_zoom(window, tmp_path):
    widget = window.labeling_widget.view
    path = str(tmp_path / "zoom.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("a", "a", 20), rectangle("b", "b", 190)]
    relations = [
        {
            "subject_id": "a",
            "object_id": "b",
            "label": "riding",
            "score": 0.77,
        },
        {"subject_id": "b", "object_id": "a", "label": "beside", "score": 0.5},
    ]
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=relations)
    )
    canvas = widget.canvas
    centers = None
    sizes = None
    for scale in (0.5, 1.0, 2.0):
        canvas.scale = scale
        layout = canvas._relation_layout()
        current_centers = [item[4].center() for item in layout]
        current_sizes = [
            (
                round(item[4].width() * scale, 4),
                round(item[4].height() * scale, 4),
            )
            for item in layout
        ]
        if centers is None:
            centers, sizes = current_centers, current_sizes
        assert current_centers == centers
        assert current_sizes == sizes
        for _, start, end, _, bounds in layout:
            assert bounds.center() == (start + end) / 2


def test_relation_dialogs_use_application_translations(window, tmp_path):
    widget = window.labeling_widget.view
    path = str(tmp_path / "translated.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("a", "a", 20), rectangle("b", "b", 190)]
    widget.load_shapes(shapes)
    widget.shape_selection_changed(shapes)
    widget.canvas.relations = [
        {"subject_id": "a", "object_id": "b", "label": "riding", "score": 0.77}
    ]
    app = QtWidgets.QApplication.instance()
    for language, expected in (
        (
            "zh_CN",
            ("创建关系", "确定", "取消", "管理关系", "编辑", "关闭", "确认"),
        ),
        (
            "ja_JP",
            (
                "関係を作成",
                "OK",
                "キャンセル",
                "関係を管理",
                "編集",
                "閉じる",
                "OK",
            ),
        ),
        (
            "ko_KR",
            ("관계 생성", "확인", "취소", "관계 관리", "편집", "닫기", "확인"),
        ),
    ):
        translator = QtCore.QTranslator()
        assert translator.load(f":/languages/translations/{language}.qm")
        app.installTranslator(translator)
        observed = []
        edit_observed = {}
        try:

            def inspect_editor():
                dialog = app.activeModalWidget()
                observed.extend(
                    [
                        dialog.windowTitle(),
                        dialog.findChild(
                            QtWidgets.QPushButton, "relationConfirmButton"
                        ).text(),
                        dialog.findChild(
                            QtWidgets.QPushButton, "relationCancelButton"
                        ).text(),
                    ]
                )
                dialog.reject()

            QtCore.QTimer.singleShot(0, inspect_editor)
            widget.create_relation()

            def inspect_existing_relation():
                dialog = app.activeModalWidget()
                edit_observed["title"] = dialog.windowTitle()
                edit_observed["labels"] = [
                    label.text()
                    for label in dialog.findChildren(QtWidgets.QLabel)
                ]
                edit_observed["buttons"] = [
                    button.text()
                    for button in dialog.findChildren(QtWidgets.QPushButton)
                ]
                edit_observed["positions"] = [
                    dialog.findChild(QtWidgets.QPushButton, name).x()
                    for name in (
                        "relationDeleteButton",
                        "relationCancelButton",
                        "relationConfirmButton",
                    )
                ]
                edit_observed["heights"] = [
                    dialog.findChild(QtWidgets.QPushButton, name).height()
                    for name in (
                        "relationDeleteButton",
                        "relationCancelButton",
                        "relationConfirmButton",
                    )
                ]
                dialog.reject()

            QtCore.QTimer.singleShot(0, inspect_existing_relation)
            widget.edit_relation(0)

            def inspect_manager():
                dialog = app.activeModalWidget()
                observed.extend(
                    [
                        dialog.windowTitle(),
                        dialog.findChild(
                            QtWidgets.QPushButton, "relationEditButton"
                        ).text(),
                        dialog.findChild(
                            QtWidgets.QPushButton, "relationCloseButton"
                        ).text(),
                    ]
                )
                dialog.reject()

            QtCore.QTimer.singleShot(0, inspect_manager)
            widget.manage_relations()

            def inspect_warning():
                dialog = app.activeModalWidget()
                observed.append(dialog.buttons()[0].text())
                dialog.accept()

            QtCore.QTimer.singleShot(0, inspect_warning)
            widget._show_relation_message(
                QtWidgets.QMessageBox.Icon.Warning, "test"
            )
            assert tuple(observed) == expected
            assert edit_observed["title"] == translator.translate(
                "LabelingWidget", "Edit Relation"
            )
            for key in ("Subject", "Object", "Relation"):
                assert (
                    translator.translate("LabelingWidget", key)
                    in edit_observed["labels"]
                )
            assert (
                translator.translate("LabelingWidget", "Confidence")
                not in edit_observed["labels"]
            )
            assert "0.770" not in edit_observed["labels"]
            for key in ("Cancel", "Delete"):
                assert (
                    translator.translate("LabelingWidget", key)
                    in edit_observed["buttons"]
                )
            assert (
                translator.translate("LabelDialog", "OK")
                in edit_observed["buttons"]
            )
            assert edit_observed["positions"] == sorted(
                edit_observed["positions"]
            )
            assert edit_observed["heights"] == [32, 32, 32]
        finally:
            app.removeTranslator(translator)


def test_relation_dialog_buttons_save_cancel_and_close(window, tmp_path):
    widget = window.labeling_widget.view
    path = str(tmp_path / "buttons.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("a", "a", 20), rectangle("b", "b", 190)]
    widget.load_shapes(shapes)
    widget.shape_selection_changed(shapes)
    app = QtWidgets.QApplication.instance()

    def choose_button(name, text=None):
        dialog = app.activeModalWidget()
        if text is not None:
            dialog.findChild(QtWidgets.QLineEdit).setText(text)
        dialog.findChild(QtWidgets.QPushButton, name).click()

    QtCore.QTimer.singleShot(
        0, lambda: choose_button("relationConfirmButton", "riding")
    )
    widget.create_relation()
    assert widget.canvas.relations[0]["label"] == "riding"
    QtCore.QTimer.singleShot(
        0, lambda: choose_button("relationCancelButton", "ignored")
    )
    widget.edit_relation(0)
    assert widget.canvas.relations[0]["label"] == "riding"
    QtCore.QTimer.singleShot(0, lambda: choose_button("relationCloseButton"))
    widget.manage_relations()


def test_delete_key_removes_only_selected_relation_and_persists(
    window, tmp_path
):
    widget = window.labeling_widget.view
    path = str(tmp_path / "delete_relation.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("person", "a", 20), rectangle("bicycle", "b", 190)]
    relations = [
        {
            "subject_id": "a",
            "object_id": "b",
            "label": "riding",
            "score": 0.77,
        },
        {
            "subject_id": "b",
            "object_id": "a",
            "label": "carrying",
            "score": 0.5,
        },
    ]
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=relations)
    )
    canvas = widget.canvas
    canvas.show()
    canvas.setFocus()
    QtWidgets.QApplication.processEvents()
    bounds = canvas._relation_layout()[0][4]
    point = (
        (bounds.center() + canvas.offset_to_center()) * canvas.scale
    ).toPoint()
    QtTest.QTest.mouseClick(
        canvas, QtCore.Qt.MouseButton.LeftButton, pos=point
    )
    assert canvas.selected_relation is relations[0]
    QtTest.QTest.keyClick(canvas, QtCore.Qt.Key.Key_Delete)
    assert canvas.selected_relation is None
    assert canvas.relations == [relations[1]]
    assert len(canvas.shapes) == 2
    assert widget.save_labels(widget.get_label_file())
    data = json.loads(open(widget.get_label_file(), encoding="utf-8").read())
    assert data["relations"] == [relations[1]]
    assert len(data["shapes"]) == 2
    widget.filename = None
    assert widget.load_file(path)
    assert widget.canvas.relations == [relations[1]]


def test_delete_key_keeps_object_deletion_after_switching_selection(
    window, tmp_path
):
    widget = window.labeling_widget.view
    path = str(tmp_path / "delete_object.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("person", "a", 20), rectangle("bicycle", "b", 190)]
    relation = {
        "subject_id": "a",
        "object_id": "b",
        "label": "riding",
        "score": 0.77,
    }
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=[relation])
    )
    canvas = widget.canvas
    canvas.show()
    canvas.setFocus()
    QtWidgets.QApplication.processEvents()
    bounds = canvas._relation_layout()[0][4]
    point = (
        (bounds.center() + canvas.offset_to_center()) * canvas.scale
    ).toPoint()
    QtTest.QTest.mouseClick(
        canvas, QtCore.Qt.MouseButton.LeftButton, pos=point
    )
    assert canvas.selected_relation is relation
    widget.shape_selection_changed([shapes[0]])
    assert canvas.selected_relation is None
    assert widget.actions.delete.isEnabled()
    QtTest.QTest.keyClick(canvas, QtCore.Qt.Key.Key_Delete)
    assert len(canvas.shapes) == 1
    assert canvas.shapes[0].label == "bicycle"


def test_relation_score_follows_show_scores_action(window, tmp_path):
    widget = window.labeling_widget.view
    path = str(tmp_path / "score-toggle.png")
    Image.new("RGB", (300, 120)).save(path)
    assert widget.load_file(path)
    shapes = [rectangle("person", "a", 20), rectangle("bicycle", "b", 190)]
    relation = {
        "subject_id": "a",
        "object_id": "b",
        "label": "riding",
        "score": 0.77,
    }
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=[relation])
    )
    if not widget.actions.show_scores.isChecked():
        widget.actions.show_scores.trigger()
    assert widget.canvas._relation_layout()[0][3] == "riding · 0.77"
    widget.actions.show_scores.trigger()
    assert not widget.canvas.show_scores
    assert widget.canvas._relation_layout()[0][3] == "riding"
    assert relation["score"] == 0.77
    assert widget.save_labels(widget.get_label_file())
    data = json.loads(open(widget.get_label_file(), encoding="utf-8").read())
    assert data["relations"][0]["score"] == 0.77
    widget.actions.show_scores.trigger()
    assert widget.canvas._relation_layout()[0][3] == "riding · 0.77"


def test_relation_labels_follow_show_labels_and_scores_actions(
    window, tmp_path
):
    widget = window.labeling_widget.view
    path = str(tmp_path / "label-toggle.png")
    Image.new("RGB", (300, 120), "white").save(path)
    assert widget.load_file(path)
    shapes = [rectangle("person", "a", 20), rectangle("bicycle", "b", 190)]
    relation = {
        "subject_id": "a",
        "object_id": "b",
        "label": "riding",
        "score": 0.77,
    }
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=[relation])
    )
    canvas = widget.canvas
    if not widget.actions.show_labels.isChecked():
        widget.actions.show_labels.trigger()
    if not widget.actions.show_scores.isChecked():
        widget.actions.show_scores.trigger()
    canvas.show()
    canvas.setFocus()
    QtWidgets.QApplication.processEvents()
    bounds = canvas._relation_layout()[0][4]
    point = (
        (bounds.center() + canvas.offset_to_center()) * canvas.scale
    ).toPoint()
    QtTest.QTest.mouseClick(
        canvas, QtCore.Qt.MouseButton.LeftButton, pos=point
    )
    assert canvas.selected_relation is relation
    widget.actions.show_labels.trigger()
    assert not canvas.show_labels
    assert canvas._relation_layout() == []
    assert len(canvas._relation_segments()) == 1
    assert canvas.selected_relation is None
    canvas.repaint()
    visible_line = canvas.grab().toImage()
    widget.actions.show_linking.trigger()
    canvas.repaint()
    hidden_line = canvas.grab().toImage()
    assert visible_line != hidden_line
    widget.actions.show_linking.trigger()
    widget.actions.show_scores.trigger()
    assert canvas._relation_layout() == []
    widget.actions.show_labels.trigger()
    assert canvas._relation_layout()[0][3] == "riding"
    widget.actions.show_scores.trigger()
    assert canvas._relation_layout()[0][3] == "riding · 0.77"
    assert relation["score"] == 0.77


def test_relation_selection_highlights_endpoints_without_changing_shapes(
    window, tmp_path
):
    widget = window.labeling_widget.view
    path = str(tmp_path / "relation-highlight.png")
    Image.new("RGB", (350, 120), "white").save(path)
    assert widget.load_file(path)
    shapes = [
        rectangle("a", "a", 20),
        rectangle("b", "b", 160),
        rectangle("c", "c", 260),
    ]
    relations = [
        {"subject_id": "a", "object_id": "b", "label": "beside", "score": 0.7},
        {"subject_id": "b", "object_id": "c", "label": "near", "score": 0.6},
    ]
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=relations)
    )
    canvas = widget.canvas
    canvas.show()
    canvas.cross_line_show = False
    QtWidgets.QApplication.processEvents()
    before = canvas.grab().toImage()
    state = [shape.to_dict() for shape in canvas.shapes]
    bounds = canvas._relation_layout()[0][4]
    point = (
        (bounds.center() + canvas.offset_to_center()) * canvas.scale
    ).toPoint()
    QtTest.QTest.mouseClick(
        canvas, QtCore.Qt.MouseButton.LeftButton, pos=point
    )
    QtWidgets.QApplication.processEvents()
    after = canvas.grab().toImage()
    assert canvas._focused_relation() is relations[0]
    assert before != after
    assert [shape.to_dict() for shape in canvas.shapes] == state
    assert canvas.selected_shapes == []
    assert all(not shape.selected for shape in canvas.shapes)
    canvas.selected_relation = None
    canvas.update()
    assert canvas._focused_relation() is None


def test_relation_manager_previews_and_restores_canvas_selection(
    window, tmp_path
):
    widget = window.labeling_widget.view
    path = str(tmp_path / "relation-preview.png")
    Image.new("RGB", (350, 120)).save(path)
    assert widget.load_file(path)
    shapes = [
        rectangle("a", "a", 20),
        rectangle("b", "b", 160),
        rectangle("c", "c", 260),
    ]
    relations = [
        {"subject_id": "a", "object_id": "b", "label": "beside", "score": 0.7},
        {"subject_id": "b", "object_id": "c", "label": "near", "score": 0.6},
    ]
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=relations)
    )
    canvas = widget.canvas
    canvas.selected_relation = relations[0]
    observed = []

    def inspect_manager():
        dialog = QtWidgets.QApplication.activeModalWidget()
        listing = dialog.findChild(QtWidgets.QListWidget)
        listing.setCurrentRow(1)
        observed.append(canvas._focused_relation() is relations[1])
        observed.append(canvas.selected_relation is relations[0])
        listing.clearSelection()
        observed.append(canvas._focused_relation() is relations[0])
        listing.setCurrentRow(1)
        dialog.findChild(QtWidgets.QPushButton, "relationCloseButton").click()

    QtCore.QTimer.singleShot(0, inspect_manager)
    widget.manage_relations()
    assert observed == [True, True, True]
    assert canvas.preview_relation is None
    assert canvas._focused_relation() is relations[0]
    assert canvas.selected_shapes == []


def test_relation_focus_highlights_only_endpoints_and_dims_other_edges(
    window, tmp_path
):
    widget = window.labeling_widget.view
    path = str(tmp_path / "relation-colors.png")
    Image.new("RGB", (350, 120), "white").save(path)
    assert widget.load_file(path)
    shapes = [
        rectangle("a", "a", 20),
        rectangle("b", "b", 160),
        rectangle("c", "c", 260),
    ]
    relations = [
        {
            "subject_id": "a",
            "object_id": "b",
            "label": "riding",
            "score": 0.77,
        },
        {"subject_id": "b", "object_id": "c", "label": "near", "score": 0.6},
    ]
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=relations)
    )
    canvas = widget.canvas
    canvas.scale = 1.0
    canvas.show_labels = False

    def render():
        image = QtGui.QImage(350, 120, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(image)
        canvas._paint_relations(painter)
        painter.end()
        return image

    before = render()
    canvas.selected_relation = relations[0]
    after = render()
    assert before.pixelColor(20, 25) == QtGui.QColor("white")
    assert after.pixelColor(20, 25) != QtGui.QColor("white")
    assert after.pixelColor(160, 25) != QtGui.QColor("white")
    assert after.pixelColor(16, 16) != QtGui.QColor("white")
    for pixel, shape in (
        (after.pixelColor(20, 25), shapes[0]),
        (after.pixelColor(160, 25), shapes[1]),
    ):
        assert pixel.hsvHue() == shape.line_color.hsvHue()
        assert pixel.hsvSaturation() >= shape.line_color.hsvSaturation()
    assert after.pixelColor(260, 25) == QtGui.QColor("white")
    before_color, after_color = before.pixelColor(210, 35), after.pixelColor(
        210, 35
    )
    assert sum(after_color.getRgb()[:3]) > sum(before_color.getRgb()[:3])


def test_relation_highlight_keeps_square_corners_at_multiple_zooms(
    window, tmp_path
):
    widget = window.labeling_widget.view
    path = str(tmp_path / "sharp-corners.png")
    Image.new("RGB", (300, 120), "white").save(path)
    assert widget.load_file(path)
    shapes = [rectangle("a", "a", 20), rectangle("b", "b", 190)]
    relation = {"subject_id": "a", "object_id": "b", "label": "near"}
    widget.new_shapes_from_auto_labeling(
        AutoLabelingResult(shapes, relations=[relation])
    )
    canvas = widget.canvas
    for scale in (0.5, 1.0, 2.0):
        canvas.scale = scale
        image = QtGui.QImage(700, 300, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(image)
        painter.scale(scale, scale)
        canvas._paint_relation_endpoints(painter, relation)
        painter.end()
        outer_corner = int(20 * scale - 4)
        assert image.pixelColor(outer_corner, outer_corner) != QtGui.QColor(
            "white"
        )
