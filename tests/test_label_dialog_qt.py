from anylabeling.views.labeling.widgets.label_dialog import LabelDialog


def test_label_dialog_popup_cancel_returns_empty_tuple(qtbot):
    dlg = LabelDialog(
        labels=["a", "b"],
        sort_labels=True,
        show_text_field=True,
        completion="startswith",
        fit_to_content={"row": False, "column": True},
        flags={},
    )
    qtbot.addWidget(dlg)
    dlg.exec = lambda: False

    assert dlg.pop_up(
        text="a",
        flags={},
        group_id=None,
        description="",
        difficult=False,
        kie_linking=[],
        move=False,
        move_mode="center",
    ) == (None, None, None, None, False, [])


def test_label_dialog_popup_ok_returns_values(qtbot):
    dlg = LabelDialog(
        labels=["a", "b"],
        sort_labels=True,
        show_text_field=True,
        completion="startswith",
        fit_to_content={"row": False, "column": True},
        flags={".*": ["occluded"]},
    )
    qtbot.addWidget(dlg)

    def _fake_exec():
        dlg.edit.setText("b")
        dlg.edit_group_id.setText("3")
        dlg.edit_description.setPlainText("desc")
        dlg.edit_difficult.setChecked(True)
        if dlg.flags_layout.count() > 0:
            dlg.flags_layout.itemAt(0).widget().setChecked(True)
        return True

    dlg.exec = _fake_exec

    text, flags, group_id, description, difficult, kie_linking = dlg.pop_up(
        text="a",
        flags={"occluded": False},
        group_id=None,
        description="",
        difficult=False,
        kie_linking=[],
        move=False,
        move_mode="center",
    )

    assert text == "b"
    assert flags == {"occluded": True}
    assert group_id == 3
    assert description == "desc"
    assert difficult is True
    assert kie_linking == []
