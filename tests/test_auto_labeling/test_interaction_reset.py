import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets, uic

from anylabeling.services.auto_labeling.types import AutoLabelingMode
from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.widgets.auto_labeling.auto_labeling import (
    AutoLabelingWidget,
)
from anylabeling.views.labeling.widgets.canvas import Canvas
from anylabeling.views.labeling.shape import Shape


@pytest.mark.parametrize("shape_type", ["point", "rectangle"])
@pytest.mark.parametrize(
    "edit_mode", [AutoLabelingMode.ADD, AutoLabelingMode.REMOVE]
)
def test_reset_prompt_interaction(shape_type, edit_mode):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = AutoLabelingWidget.__new__(AutoLabelingWidget)
    QtWidgets.QWidget.__init__(widget)
    uic.loadUi(
        str(
            Path(__file__).resolve().parents[2]
            / "anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui"
        ),
        widget,
    )
    canvas = Canvas(parent=None)
    parent = SimpleNamespace(
        auto_labeling_widget=widget,
        clear_auto_labeling_marks=Mock(),
        toggle_draw_mode=lambda edit: canvas.set_editing(edit),
        set_text_editing=Mock(),
        update_labeling_instruction=Mock(),
    )
    parent.set_edit_mode = lambda: LabelingWidget.set_edit_mode(parent)
    widget.parent = parent
    widget.auto_labeling_mode_changed.connect(canvas.set_auto_labeling_mode)
    widget.auto_labeling_mode_changed.connect(widget.update_button_colors)
    widget.auto_decode_mode_changed.connect(canvas.set_auto_decode_mode)
    widget.clear_auto_decode_requested.connect(canvas.reset_auto_decode_state)
    widget.auto_labeling_mode = AutoLabelingMode(edit_mode, shape_type)
    canvas.auto_labeling_mode = widget.auto_labeling_mode
    canvas.is_auto_labeling = True
    canvas.set_editing(False)
    canvas.current = Shape(shape_type=shape_type)
    canvas.line.points = [QtCore.QPointF(10, 10)]
    widget.skip_auto_prediction = True
    widget.button_auto_decode.setChecked(True)
    widget.on_auto_decode_toggled()
    canvas.auto_decode_timer.start(1000)

    widget.reset_auto_labeling_interaction()

    assert widget.auto_labeling_mode == AutoLabelingMode.NONE
    assert canvas.auto_labeling_mode == AutoLabelingMode.NONE
    assert not canvas.is_auto_labeling
    assert canvas.editing()
    assert canvas.current is None
    assert canvas.line.points == []
    assert not widget.skip_auto_prediction
    assert not widget.button_auto_decode.isChecked()
    assert not canvas.auto_decode_mode
    assert not canvas.auto_decode_timer.isActive()
    parent.clear_auto_labeling_marks.assert_called_once()
    canvas.restore_cursor()
    canvas.close()
    widget.close()
    app.processEvents()
