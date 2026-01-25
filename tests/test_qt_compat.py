from PyQt6 import QtCore, QtGui, QtWidgets


def test_qt5_compat_exports_common_symbols():
    assert hasattr(QtCore.Qt, "AlignCenter")
    assert hasattr(QtCore.Qt, "ElideNone")
    assert hasattr(QtCore.Qt, "QueuedConnection")
    assert hasattr(QtWidgets.QDialog, "exec_")
    assert hasattr(QtWidgets.QMessageBox, "Ok")
    assert hasattr(QtGui.QPainter, "Antialiasing")
    assert hasattr(QtWidgets.QDialogButtonBox, "Ok")
    assert hasattr(QtWidgets.QTextEdit, "WidgetWidth")
    assert hasattr(QtWidgets.QLineEdit, "Password")
    assert hasattr(QtWidgets.QSpinBox, "UpDownArrows")
    assert hasattr(QtWidgets.QFormLayout, "ExpandingFieldsGrow")
    assert hasattr(QtWidgets.QFormLayout, "DontWrapRows")
    assert hasattr(QtCore.QEvent, "Enter")
