import pytest
from PyQt6 import QtCore, QtTest, QtWidgets

from anylabeling.views.labeling.utils.theme import get_mode, init_theme
from anylabeling.views.labeling.widgets.combo_box import ComboBox
from anylabeling.views.labeling.widgets.text_input import TextInput
from anylabeling.views.labeling.widgets.relation_dialog import (
    get_relation_dialog_style,
)


def test_dimensions_with_relation_dialog_stylesheet():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = QtWidgets.QDialog()
    dialog.setStyleSheet(get_relation_dialog_style())
    layout = QtWidgets.QVBoxLayout(dialog)
    combo = ComboBox(dialog)
    label = TextInput(parent=dialog)
    layout.addWidget(combo)
    layout.addWidget(label)
    combo.addItems(["person", "bicycle"])
    try:
        dialog.show()
        app.processEvents()
        assert combo.height() == 32
        assert combo.height() == label.height() == 32
        assert combo.lineEdit().geometry().left() - 2 == 8
        combo.showPopup()
        combo._filter("person")
        app.processEvents()
        assert combo._view.height() == 32
        assert combo._view.parentWidget().height() == 42
    finally:
        dialog.close()


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_search_selection_and_popup_position(mode):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    previous_mode = get_mode()
    init_theme(mode)
    host = QtWidgets.QWidget()
    host.resize(400, 200)
    combo = ComboBox(host)
    combo.setGeometry(20, 20, 240, 32)
    for i in range(20):
        combo.addItem(f"item {i}", i)
    combo.setCurrentIndex(17)
    try:
        host.show()
        combo._arrow.click()
        app.processEvents()
        assert combo._popup.isVisible()
        assert combo.lineEdit().text() == ""
        assert combo._view.currentIndex().row() == 17
        assert not combo._view.selectionModel().hasSelection()
        assert (
            combo._view.viewport()
            .rect()
            .intersects(combo._view.visualRect(combo._view.currentIndex()))
        )
        assert combo._view.verticalScrollBarPolicy() == (
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        assert (
            combo._view.visualRect(combo._view.currentIndex()).height() == 32
        )
        QtTest.QTest.keyClicks(combo.lineEdit(), "ITEM 3")
        assert combo._proxy.rowCount() == 1
        app.processEvents()
        assert combo._view.height() == 32
        QtTest.QTest.keyClick(combo.lineEdit(), QtCore.Qt.Key.Key_Return)
        assert combo.currentData() == 3
        combo._arrow.click()
        QtTest.QTest.keyClicks(combo.lineEdit(), "missing")
        assert combo._proxy.rowCount() == 0
        QtTest.QTest.keyClick(combo.lineEdit(), QtCore.Qt.Key.Key_Escape)
        assert combo.currentData() == 3
        assert combo.currentText() == "item 3"
    finally:
        host.close()
        init_theme(previous_mode)
