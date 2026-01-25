from PyQt6 import QtCore, QtGui, QtWidgets


def _export_enum_members(enum_cls, target, *, allow_prefix=None):
    for member in enum_cls:
        name = member.name
        if allow_prefix is not None and not name.startswith(allow_prefix):
            continue
        if hasattr(target, name):
            continue
        try:
            setattr(target, name, member)
        except Exception:
            pass


def apply_qt5_compat():
    qt = QtCore.Qt

    if hasattr(qt, "AlignmentFlag"):
        _export_enum_members(qt.AlignmentFlag, qt, allow_prefix="Align")

    if hasattr(qt, "Orientation"):
        _export_enum_members(qt.Orientation, qt)

    if hasattr(qt, "ScrollBarPolicy"):
        _export_enum_members(qt.ScrollBarPolicy, qt, allow_prefix="ScrollBar")

    if hasattr(qt, "WindowType"):
        _export_enum_members(qt.WindowType, qt, allow_prefix="Window")
        _export_enum_members(qt.WindowType, qt, allow_prefix=None)

    if hasattr(qt, "WindowModality"):
        _export_enum_members(qt.WindowModality, qt, allow_prefix="Window")

    if hasattr(qt, "ItemDataRole"):
        _export_enum_members(qt.ItemDataRole, qt, allow_prefix=None)

    if hasattr(qt, "CheckState"):
        _export_enum_members(qt.CheckState, qt)

    if hasattr(qt, "AspectRatioMode"):
        _export_enum_members(qt.AspectRatioMode, qt, allow_prefix=None)

    if hasattr(qt, "TransformationMode"):
        _export_enum_members(qt.TransformationMode, qt, allow_prefix=None)

    if hasattr(qt, "CursorShape"):
        _export_enum_members(qt.CursorShape, qt, allow_prefix=None)

    if hasattr(qt, "WidgetAttribute"):
        _export_enum_members(qt.WidgetAttribute, qt, allow_prefix="WA_")

    if hasattr(qt, "FocusPolicy"):
        _export_enum_members(qt.FocusPolicy, qt, allow_prefix=None)

    if hasattr(qt, "ContextMenuPolicy"):
        _export_enum_members(qt.ContextMenuPolicy, qt, allow_prefix=None)

    if hasattr(qt, "MouseButton"):
        _export_enum_members(qt.MouseButton, qt, allow_prefix=None)

    if hasattr(qt, "DropAction"):
        _export_enum_members(qt.DropAction, qt, allow_prefix=None)

    if hasattr(qt, "KeyboardModifier"):
        _export_enum_members(qt.KeyboardModifier, qt, allow_prefix=None)

    if hasattr(qt, "Key"):
        _export_enum_members(qt.Key, qt, allow_prefix="Key_")

    if hasattr(qt, "MatchFlag"):
        _export_enum_members(qt.MatchFlag, qt, allow_prefix="Match")

    if hasattr(qt, "ConnectionType"):
        _export_enum_members(qt.ConnectionType, qt, allow_prefix=None)

    if hasattr(qt, "TextElideMode"):
        _export_enum_members(qt.TextElideMode, qt, allow_prefix="Elide")

    try:
        event_cls = QtCore.QEvent
        if hasattr(event_cls, "Type"):
            _export_enum_members(event_cls.Type, event_cls, allow_prefix=None)
    except Exception:
        pass

    try:
        dialog = QtWidgets.QDialog
        if not hasattr(dialog, "Accepted") and hasattr(dialog, "DialogCode"):
            dialog.Accepted = dialog.DialogCode.Accepted
        if not hasattr(dialog, "Rejected") and hasattr(dialog, "DialogCode"):
            dialog.Rejected = dialog.DialogCode.Rejected
        if not hasattr(dialog, "exec_") and hasattr(dialog, "exec"):
            dialog.exec_ = dialog.exec
    except Exception:
        pass

    try:
        painter = QtGui.QPainter
        if not hasattr(painter, "Antialiasing") and hasattr(painter, "RenderHint"):
            painter.Antialiasing = painter.RenderHint.Antialiasing
        if not hasattr(painter, "TextAntialiasing") and hasattr(
            painter, "RenderHint"
        ):
            painter.TextAntialiasing = painter.RenderHint.TextAntialiasing
        if not hasattr(painter, "SmoothPixmapTransform") and hasattr(
            painter, "RenderHint"
        ):
            painter.SmoothPixmapTransform = painter.RenderHint.SmoothPixmapTransform
    except Exception:
        pass

    try:
        line_edit = QtWidgets.QLineEdit
        if hasattr(line_edit, "EchoMode"):
            _export_enum_members(line_edit.EchoMode, line_edit, allow_prefix=None)
    except Exception:
        pass

    try:
        spin_base = QtWidgets.QAbstractSpinBox
        if hasattr(spin_base, "ButtonSymbols"):
            _export_enum_members(spin_base.ButtonSymbols, spin_base, allow_prefix=None)
    except Exception:
        pass

    try:
        form_layout = QtWidgets.QFormLayout
        if hasattr(form_layout, "FieldGrowthPolicy"):
            _export_enum_members(
                form_layout.FieldGrowthPolicy, form_layout, allow_prefix=None
            )
        if hasattr(form_layout, "RowWrapPolicy"):
            _export_enum_members(form_layout.RowWrapPolicy, form_layout, allow_prefix=None)
    except Exception:
        pass

    try:
        text_edit = QtWidgets.QTextEdit
        if not hasattr(text_edit, "WidgetWidth") and hasattr(text_edit, "LineWrapMode"):
            text_edit.WidgetWidth = text_edit.LineWrapMode.WidgetWidth
        if not hasattr(text_edit, "NoWrap") and hasattr(text_edit, "LineWrapMode"):
            text_edit.NoWrap = text_edit.LineWrapMode.NoWrap
    except Exception:
        pass

    try:
        msg = QtWidgets.QMessageBox
        if not hasattr(msg, "Ok") and hasattr(msg, "StandardButton"):
            msg.Ok = msg.StandardButton.Ok
        if not hasattr(msg, "Cancel") and hasattr(msg, "StandardButton"):
            msg.Cancel = msg.StandardButton.Cancel
        if not hasattr(msg, "Save") and hasattr(msg, "StandardButton"):
            msg.Save = msg.StandardButton.Save
        if not hasattr(msg, "Discard") and hasattr(msg, "StandardButton"):
            msg.Discard = msg.StandardButton.Discard
        if not hasattr(msg, "Yes") and hasattr(msg, "StandardButton"):
            msg.Yes = msg.StandardButton.Yes
        if not hasattr(msg, "No") and hasattr(msg, "StandardButton"):
            msg.No = msg.StandardButton.No
        if not hasattr(msg, "Warning") and hasattr(msg, "Icon"):
            msg.Warning = msg.Icon.Warning
        if not hasattr(msg, "Information") and hasattr(msg, "Icon"):
            msg.Information = msg.Icon.Information
        if not hasattr(msg, "Critical") and hasattr(msg, "Icon"):
            msg.Critical = msg.Icon.Critical
        if not hasattr(msg, "Question") and hasattr(msg, "Icon"):
            msg.Question = msg.Icon.Question
    except Exception:
        pass

    try:
        button_box = QtWidgets.QDialogButtonBox
        if not hasattr(button_box, "Ok") and hasattr(button_box, "StandardButton"):
            button_box.Ok = button_box.StandardButton.Ok
        if not hasattr(button_box, "Cancel") and hasattr(
            button_box, "StandardButton"
        ):
            button_box.Cancel = button_box.StandardButton.Cancel
    except Exception:
        pass

    try:
        spinbox = QtWidgets.QAbstractSpinBox
        if not hasattr(spinbox, "NoButtons") and hasattr(spinbox, "ButtonSymbols"):
            spinbox.NoButtons = spinbox.ButtonSymbols.NoButtons
    except Exception:
        pass

    try:
        item_view = QtWidgets.QAbstractItemView
        if not hasattr(item_view, "NoEditTriggers") and hasattr(
            item_view, "EditTrigger"
        ):
            item_view.NoEditTriggers = item_view.EditTrigger.NoEditTriggers
        if not hasattr(item_view, "NoSelection") and hasattr(
            item_view, "SelectionMode"
        ):
            item_view.NoSelection = item_view.SelectionMode.NoSelection
        if not hasattr(item_view, "SingleSelection") and hasattr(
            item_view, "SelectionMode"
        ):
            item_view.SingleSelection = item_view.SelectionMode.SingleSelection
        if not hasattr(item_view, "ExtendedSelection") and hasattr(
            item_view, "SelectionMode"
        ):
            item_view.ExtendedSelection = item_view.SelectionMode.ExtendedSelection
        if not hasattr(item_view, "InternalMove") and hasattr(
            item_view, "DragDropMode"
        ):
            item_view.InternalMove = item_view.DragDropMode.InternalMove
    except Exception:
        pass

    try:
        header_view = QtWidgets.QHeaderView
        if not hasattr(header_view, "ResizeToContents") and hasattr(
            header_view, "ResizeMode"
        ):
            header_view.ResizeToContents = header_view.ResizeMode.ResizeToContents
        if not hasattr(header_view, "Stretch") and hasattr(header_view, "ResizeMode"):
            header_view.Stretch = header_view.ResizeMode.Stretch
    except Exception:
        pass

    try:
        completer = QtWidgets.QCompleter
        if not hasattr(completer, "InlineCompletion") and hasattr(
            completer, "CompletionMode"
        ):
            completer.InlineCompletion = completer.CompletionMode.InlineCompletion
        if not hasattr(completer, "PopupCompletion") and hasattr(
            completer, "CompletionMode"
        ):
            completer.PopupCompletion = completer.CompletionMode.PopupCompletion
    except Exception:
        pass

    try:
        size_policy = QtWidgets.QSizePolicy
        if hasattr(size_policy, "Policy"):
            _export_enum_members(size_policy.Policy, size_policy, allow_prefix=None)
    except Exception:
        pass

    try:
        dock_widget = QtWidgets.QDockWidget
        if hasattr(dock_widget, "DockWidgetFeature"):
            _export_enum_members(
                dock_widget.DockWidgetFeature, dock_widget, allow_prefix=None
            )
    except Exception:
        pass

    try:
        frame = QtWidgets.QFrame
        if hasattr(frame, "Shape"):
            _export_enum_members(frame.Shape, frame, allow_prefix=None)
        if hasattr(frame, "Shadow"):
            _export_enum_members(frame.Shadow, frame, allow_prefix=None)
    except Exception:
        pass

    try:
        file_dialog = QtWidgets.QFileDialog
        if hasattr(file_dialog, "Option"):
            _export_enum_members(file_dialog.Option, file_dialog, allow_prefix=None)
    except Exception:
        pass
