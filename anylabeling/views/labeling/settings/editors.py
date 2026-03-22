from __future__ import annotations

from typing import Any

from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.utils.qt import new_icon
from anylabeling.views.labeling.utils.style import (
    get_double_spinbox_style,
    get_normal_button_style,
    get_spinbox_style,
)


def _pick_native_color(
    initial: QtGui.QColor | None = None,
    parent: QtWidgets.QWidget | None = None,
    show_alpha: bool = True,
) -> QtGui.QColor:
    anchor = parent.window() if parent is not None else None
    dialog_parent = anchor
    if dialog_parent is not None:
        translucent = dialog_parent.testAttribute(
            QtCore.Qt.WidgetAttribute.WA_TranslucentBackground
        )
        frameless = bool(
            dialog_parent.windowFlags()
            & QtCore.Qt.WindowType.FramelessWindowHint
        )
        if translucent or frameless:
            dialog_parent = None

    options = QtWidgets.QColorDialog.ColorDialogOption(0)
    if show_alpha:
        options |= QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel

    return QtWidgets.QColorDialog.getColor(
        initial if initial is not None else QtGui.QColor(),
        dialog_parent,
        "",
        options,
    )


class ColorRgbaEditor(QtWidgets.QWidget):
    value_changed = QtCore.pyqtSignal(list)

    def __init__(
        self, channels: int = 4, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        if channels not in (3, 4):
            raise ValueError("channels must be 3 or 4")
        self._channels = channels
        self._spinboxes: list[QtWidgets.QSpinBox] = []
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for _ in range(channels):
            spinbox = QtWidgets.QSpinBox(self)
            spinbox.setRange(0, 255)
            spinbox.setFixedHeight(30)
            spinbox.setStyleSheet(get_spinbox_style())
            spinbox.valueChanged.connect(self._on_changed)
            self._spinboxes.append(spinbox)
            layout.addWidget(spinbox)

    def _on_changed(self, _value: int) -> None:
        self.value_changed.emit(self.get_value())

    def set_value(self, value: Any) -> None:
        channels = list(value or [])
        if len(channels) != self._channels:
            channels = [0] * self._channels
        for idx, spinbox in enumerate(self._spinboxes):
            spinbox.blockSignals(True)
            spinbox.setValue(int(channels[idx]))
            spinbox.blockSignals(False)

    def get_value(self) -> list[int]:
        return [spinbox.value() for spinbox in self._spinboxes]


class Vector2Editor(QtWidgets.QWidget):
    value_changed = QtCore.pyqtSignal(list)

    def __init__(
        self,
        minimum: float | None = None,
        maximum: float | None = None,
        decimals: int = 2,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._spinboxes: list[QtWidgets.QDoubleSpinBox] = []
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        prefixes = ("X: ", "Y: ")
        for index in range(2):
            spinbox = QtWidgets.QDoubleSpinBox(self)
            spinbox.setRange(
                minimum if minimum is not None else -1000000.0,
                maximum if maximum is not None else 1000000.0,
            )
            spinbox.setDecimals(decimals)
            spinbox.setSingleStep(0.1)
            spinbox.setPrefix(prefixes[index])
            spinbox.setStyleSheet(get_double_spinbox_style())
            spinbox.setFixedHeight(30)
            spinbox.valueChanged.connect(self._on_changed)
            self._spinboxes.append(spinbox)
            layout.addWidget(spinbox)

    def _on_changed(self, _value: float) -> None:
        self.value_changed.emit(self.get_value())

    def set_value(self, value: Any) -> None:
        values = list(value or [0.0, 0.0])
        if len(values) != 2:
            values = [0.0, 0.0]
        for idx, spinbox in enumerate(self._spinboxes):
            spinbox.blockSignals(True)
            spinbox.setValue(float(values[idx]))
            spinbox.blockSignals(False)

    def get_value(self) -> list[float]:
        return [float(spinbox.value()) for spinbox in self._spinboxes]


class HexColorPickerEditor(QtWidgets.QWidget):
    value_changed = QtCore.pyqtSignal(object)

    def __init__(
        self,
        output_mode: str = "hex",
        fixed_alpha: int | None = 255,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        if output_mode not in {"hex", "rgb255", "rgba255"}:
            raise ValueError(
                "output_mode must be 'hex', 'rgb255', or 'rgba255'"
            )
        self._output_mode = output_mode
        if fixed_alpha is None:
            self._fixed_alpha = None
        else:
            self._fixed_alpha = int(max(0, min(255, fixed_alpha)))
        self._value = "#000000"
        self._alpha = (
            self._fixed_alpha if self._fixed_alpha is not None else 255
        )

        self._edit = QtWidgets.QLineEdit(self)
        self._edit.setReadOnly(True)
        self._edit.setFixedHeight(36)
        self._edit.setMinimumWidth(116)
        self._edit.setTextMargins(8, 0, 26, 0)

        self._button = QtWidgets.QToolButton(self._edit)
        self._base_button_icon = new_icon("color", "svg")
        self._button.setIcon(self._base_button_icon)
        self._button.setIconSize(QtCore.QSize(14, 14))
        self._button.setFixedSize(16, 16)
        self._button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._button.setStyleSheet(self._button_style())
        self._button.clicked.connect(self._on_pick_color)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._edit)
        self.setFixedHeight(36)

        self.set_value(self._value)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        x = self._edit.width() - self._button.width() - 8
        y = (self._edit.height() - self._button.height()) // 2
        self._button.move(x, y)

    def _input_style(self) -> str:
        return (
            "QLineEdit {"
            "border: 1px solid #C8C8CC;"
            "border-radius: 6px;"
            "padding: 0 8px;"
            "font-size: 11px;"
            "color: palette(text);"
            "}"
        )

    def _button_style(self) -> str:
        return (
            "QToolButton {"
            "border: 1px solid #C8C8CC;"
            "border-radius: 4px;"
            f"background: {self._value};"
            "padding: 0;"
            "}"
            "QToolButton:hover {"
            "border: 1px solid #ADADB3;"
            "}"
            "QToolButton:pressed {"
            "border: 1px solid #8E8E93;"
            "}"
        )

    def _button_icon_color(self) -> QtGui.QColor:
        color = QtGui.QColor(self._value)
        if not color.isValid():
            return QtGui.QColor("#1F1F1F")
        luminance = (
            0.299 * float(color.red())
            + 0.587 * float(color.green())
            + 0.114 * float(color.blue())
        ) / 255.0
        if luminance < 0.6:
            return QtGui.QColor("#FFFFFF")
        return QtGui.QColor("#1F1F1F")

    def _button_icon(self) -> QtGui.QIcon:
        source = self._base_button_icon.pixmap(self._button.iconSize())
        if source.isNull():
            return self._base_button_icon
        tinted = QtGui.QPixmap(source.size())
        tinted.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(tinted)
        painter.drawPixmap(0, 0, source)
        painter.setCompositionMode(
            QtGui.QPainter.CompositionMode.CompositionMode_SourceIn
        )
        painter.fillRect(tinted.rect(), self._button_icon_color())
        painter.end()
        return QtGui.QIcon(tinted)

    def _clamp_channel(self, value: Any, fallback: int) -> int:
        try:
            channel = int(value)
        except Exception:
            return fallback
        return max(0, min(255, channel))

    def _normalize_color(self, value: Any) -> tuple[str, int]:
        alpha = (
            self._fixed_alpha if self._fixed_alpha is not None else self._alpha
        )
        if self._output_mode in {"rgb255", "rgba255"}:
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                r = self._clamp_channel(value[0], 0)
                g = self._clamp_channel(value[1], 0)
                b = self._clamp_channel(value[2], 0)
                if (
                    self._output_mode == "rgba255"
                    and self._fixed_alpha is None
                ):
                    if len(value) >= 4:
                        alpha = self._clamp_channel(value[3], int(alpha))
                color = QtGui.QColor(r, g, b)
                if color.isValid():
                    return (
                        color.name(QtGui.QColor.NameFormat.HexRgb).upper(),
                        int(alpha),
                    )
            return self._value, int(alpha)
        text = str(value or "").strip()
        if not text:
            return self._value, int(alpha)
        if not text.startswith("#"):
            text = f"#{text}"
        color = QtGui.QColor(text)
        if not color.isValid():
            return self._value, int(alpha)
        if self._output_mode == "rgba255" and self._fixed_alpha is None:
            alpha = color.alpha()
        return color.name(QtGui.QColor.NameFormat.HexRgb).upper(), int(alpha)

    def _display_text(self) -> str:
        if self._output_mode != "rgba255":
            return self._value
        if self._fixed_alpha is not None:
            return self._value
        alpha = (
            self._fixed_alpha if self._fixed_alpha is not None else self._alpha
        )
        return f"{self._value}{int(alpha):02X}"

    def _refresh_ui(self) -> None:
        self._edit.setStyleSheet(self._input_style())
        self._edit.setText(self._display_text())
        self._button.setStyleSheet(self._button_style())
        self._button.setIcon(self._button_icon())

    def _on_pick_color(self) -> None:
        current = QtGui.QColor(self._value)
        if self._output_mode == "rgba255":
            alpha = (
                self._fixed_alpha
                if self._fixed_alpha is not None
                else self._alpha
            )
            current.setAlpha(int(alpha))
        color = _pick_native_color(
            initial=current,
            parent=self,
            show_alpha=self._output_mode == "rgba255"
            and self._fixed_alpha is None,
        )
        if not color.isValid():
            return
        if self._output_mode == "rgba255":
            if self._fixed_alpha is None:
                self._alpha = self._clamp_channel(color.alpha(), self._alpha)
            else:
                self._alpha = int(self._fixed_alpha)
        self._value = color.name(QtGui.QColor.NameFormat.HexRgb).upper()
        self._refresh_ui()
        self.value_changed.emit(self.get_value())

    def set_value(self, value: Any) -> None:
        self._value, self._alpha = self._normalize_color(value)
        self._refresh_ui()

    def get_value(self) -> str | list[int]:
        if self._output_mode == "rgb255":
            color = QtGui.QColor(self._value)
            return [color.red(), color.green(), color.blue()]
        if self._output_mode == "rgba255":
            color = QtGui.QColor(self._value)
            alpha = (
                self._fixed_alpha
                if self._fixed_alpha is not None
                else self._alpha
            )
            return [color.red(), color.green(), color.blue(), int(alpha)]
        return self._value


class ShortcutLineEditor(QtWidgets.QWidget):
    value_changed = QtCore.pyqtSignal(object)

    def __init__(
        self,
        allow_none: bool = False,
        allow_multiple: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._allow_none = allow_none
        self._allow_multiple = allow_multiple
        self._error = False
        self._edit = QtWidgets.QLineEdit(self)
        self._edit.setClearButtonEnabled(False)
        self._edit.setMinimumHeight(30)
        self._edit.editingFinished.connect(self._emit_current_value)
        self._edit.textChanged.connect(self._on_text_changed)
        self._edit.installEventFilter(self)

        self._clear_button = QtWidgets.QToolButton(self._edit)
        self._clear_button.setText("×")
        self._clear_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._clear_button.setStyleSheet("""
            QToolButton {
                border: none;
                background: transparent;
                color: rgb(153, 153, 158);
                font-size: 14px;
                padding: 0;
            }
            QToolButton:hover {
                color: rgb(90, 90, 95);
            }
            """)
        self._clear_button.setFixedSize(16, 16)
        self._clear_button.clicked.connect(self._clear_text)
        self._edit.setTextMargins(0, 0, 20, 0)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._edit)
        self._apply_style()

    def _normalize_single(self, text: str) -> str | None:
        sequence = QtGui.QKeySequence(text).toString(
            QtGui.QKeySequence.SequenceFormat.PortableText
        )
        sequence = sequence.strip()
        if not sequence:
            return None
        return sequence

    def _normalize_multiple(self, text: str) -> list[str]:
        chunks = [chunk.strip() for chunk in text.split(",")]
        normalized: list[str] = []
        for chunk in chunks:
            if not chunk:
                continue
            sequence = self._normalize_single(chunk)
            if sequence and sequence not in normalized:
                normalized.append(sequence)
        return normalized

    def _emit_current_value(self) -> None:
        value = self.get_value()
        self.value_changed.emit(value)

    def _on_text_changed(self, text: str) -> None:
        if text == "":
            if self._allow_multiple:
                self.value_changed.emit([])
            elif self._allow_none:
                self.value_changed.emit(None)
            else:
                self.value_changed.emit("")

    def _clear_text(self) -> None:
        self._edit.clear()
        self._emit_current_value()

    def _apply_key_event(self, event: QtGui.QKeyEvent) -> bool:
        key = event.key()
        if key == int(QtCore.Qt.Key.Key_Escape):
            self._clear_text()
            return True
        if key in {
            int(QtCore.Qt.Key.Key_Control),
            int(QtCore.Qt.Key.Key_Shift),
            int(QtCore.Qt.Key.Key_Alt),
            int(QtCore.Qt.Key.Key_Meta),
        }:
            return True
        modifiers = event.modifiers()
        modifier_value = int(getattr(modifiers, "value", modifiers))
        sequence = QtGui.QKeySequence(modifier_value | int(key)).toString(
            QtGui.QKeySequence.SequenceFormat.PortableText
        )
        sequence = sequence.strip()
        if not sequence:
            return True
        if self._allow_multiple:
            values = self._normalize_multiple(self._edit.text())
            if sequence not in values:
                values.append(sequence)
            text = ", ".join(values)
        else:
            text = sequence
        self._edit.blockSignals(True)
        self._edit.setText(text)
        self._edit.blockSignals(False)
        self._emit_current_value()
        return True

    def eventFilter(
        self, watched: QtCore.QObject, event: QtCore.QEvent
    ) -> bool:
        if (
            watched is self._edit
            and event.type() == QtCore.QEvent.Type.KeyPress
        ):
            return self._apply_key_event(event)
        return super().eventFilter(watched, event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        x = self._edit.width() - self._clear_button.width() - 6
        y = (self._edit.height() - self._clear_button.height()) // 2
        self._clear_button.move(x, y)

    def set_value(self, value: Any) -> None:
        if self._allow_multiple:
            if isinstance(value, (list, tuple)):
                text = ", ".join(str(item) for item in value if item)
            elif value:
                text = str(value)
            else:
                text = ""
        else:
            text = "" if value is None else str(value)
        self._edit.blockSignals(True)
        self._edit.setText(text)
        self._edit.blockSignals(False)

    def get_value(self) -> str | None | list[str]:
        text = self._edit.text().strip()
        if self._allow_multiple:
            return self._normalize_multiple(text)
        if not text:
            return None if self._allow_none else ""
        normalized = self._normalize_single(text)
        return normalized or (None if self._allow_none else "")

    def set_error_state(self, enabled: bool) -> None:
        self._error = enabled
        self._apply_style()

    def _apply_style(self) -> None:
        if self._error:
            self._edit.setStyleSheet(
                "QLineEdit { border: 1px solid #FF453A; border-radius: 8px; padding: 4px 8px; }"
            )
            return
        self._edit.setStyleSheet(
            "QLineEdit { border: 1px solid #C8C8CC; border-radius: 8px; padding: 4px 8px; }"
        )


class ShortcutEditor(QtWidgets.QWidget):
    value_changed = QtCore.pyqtSignal(object)

    def __init__(
        self,
        allow_none: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._allow_none = allow_none
        self._error = False
        self._edit = QtWidgets.QKeySequenceEdit(self)
        self._edit.setClearButtonEnabled(True)
        self._edit.keySequenceChanged.connect(self._on_sequence_changed)
        self._clear = QtWidgets.QPushButton(self.tr("Clear"), self)
        self._clear.setFixedHeight(30)
        self._clear.setEnabled(allow_none)
        self._clear.setStyleSheet(get_normal_button_style())
        self._clear.clicked.connect(self._clear_value)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._edit, 1)
        layout.addWidget(self._clear, 0)
        self._apply_edit_style()

    def _clear_value(self) -> None:
        self._edit.blockSignals(True)
        self._edit.setKeySequence(QtGui.QKeySequence())
        self._edit.blockSignals(False)
        self.value_changed.emit(None)

    def _on_sequence_changed(self, sequence: QtGui.QKeySequence) -> None:
        text = sequence.toString(
            QtGui.QKeySequence.SequenceFormat.PortableText
        )
        if not text and self._allow_none:
            self.value_changed.emit(None)
            return
        if not text:
            return
        self.value_changed.emit(text)

    def set_value(self, value: Any) -> None:
        sequence = QtGui.QKeySequence(str(value or ""))
        self._edit.blockSignals(True)
        self._edit.setKeySequence(sequence)
        self._edit.blockSignals(False)

    def get_value(self) -> str | None:
        text = self._edit.keySequence().toString(
            QtGui.QKeySequence.SequenceFormat.PortableText
        )
        if not text and self._allow_none:
            return None
        return text or None

    def set_error_state(self, enabled: bool) -> None:
        self._error = enabled
        self._apply_edit_style()

    def _apply_edit_style(self) -> None:
        if self._error:
            self._edit.setStyleSheet(
                "QKeySequenceEdit { border: 1px solid #FF453A; border-radius: 6px; padding: 4px; }"
            )
        else:
            self._edit.setStyleSheet("")


class MultiKeySequenceEditor(QtWidgets.QWidget):
    value_changed = QtCore.pyqtSignal(list)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._sequences: list[str] = []
        self._error = False

        self._edit = QtWidgets.QKeySequenceEdit(self)
        self._edit.setClearButtonEnabled(True)
        self._add = QtWidgets.QPushButton(self.tr("Add"), self)
        self._add.setFixedHeight(30)
        self._add.setStyleSheet(get_normal_button_style())
        self._add.clicked.connect(self._on_add_clicked)

        self._list = QtWidgets.QListWidget(self)
        self._list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._list.setMinimumHeight(88)
        self._list.itemDoubleClicked.connect(self._remove_selected)

        self._remove = QtWidgets.QPushButton(self.tr("Remove"), self)
        self._remove.setFixedHeight(30)
        self._remove.setStyleSheet(get_normal_button_style())
        self._remove.clicked.connect(self._remove_selected)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        top.addWidget(self._edit, 1)
        top.addWidget(self._add, 0)

        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addStretch(1)
        bottom.addWidget(self._remove, 0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addWidget(self._list)
        layout.addLayout(bottom)
        self._apply_edit_style()

    def _on_add_clicked(self) -> None:
        text = self._edit.keySequence().toString(
            QtGui.QKeySequence.SequenceFormat.PortableText
        )
        if not text:
            return
        if text not in self._sequences:
            self._sequences.append(text)
            self._refresh_list()
            self.value_changed.emit(self.get_value())
        self._edit.setKeySequence(QtGui.QKeySequence())

    def _remove_selected(self, _item: Any = None) -> None:
        current_item = self._list.currentItem()
        if current_item is None:
            return
        text = current_item.text()
        if text in self._sequences:
            self._sequences.remove(text)
            self._refresh_list()
            self.value_changed.emit(self.get_value())

    def _refresh_list(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        self._list.addItems(self._sequences)
        self._list.blockSignals(False)

    def set_value(self, value: Any) -> None:
        if isinstance(value, (list, tuple)):
            raw = [str(item) for item in value if item]
        elif value:
            raw = [str(value)]
        else:
            raw = []
        dedup: list[str] = []
        for text in raw:
            if text not in dedup:
                dedup.append(text)
        self._sequences = dedup
        self._refresh_list()

    def get_value(self) -> list[str]:
        return list(self._sequences)

    def set_error_state(self, enabled: bool) -> None:
        self._error = enabled
        self._apply_edit_style()

    def _apply_edit_style(self) -> None:
        if self._error:
            self._edit.setStyleSheet(
                "QKeySequenceEdit { border: 1px solid #FF453A; border-radius: 6px; padding: 4px; }"
            )
            self._list.setStyleSheet(
                "QListWidget { border: 1px solid #FF453A; border-radius: 6px; }"
            )
        else:
            self._edit.setStyleSheet("")
            self._list.setStyleSheet("")
