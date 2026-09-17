import re

from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.utils.qt import new_icon
from anylabeling.views.labeling.utils.style import get_dialog_style
from anylabeling.views.labeling.utils.theme import get_theme

from .model import ClassDefinition, MAX_ID


class PointSizeSlider(QtWidgets.QSlider):
    def __init__(self, parent=None):
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.setRange(1, 10)
        self.setFixedHeight(44)
        self.setAccessibleName(self.tr("Point size (px)"))
        self._wheel_delta = 0

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        track = QtCore.QRectF(2, 7, self.width() - 4, 30)
        x = 18 + (self.value() - 1) * (self.width() - 36) / 9
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(232, 231, 232))
        painter.drawRoundedRect(track, 15, 15)
        painter.save()
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(track, 15, 15)
        painter.setClipPath(clip)
        painter.fillRect(
            QtCore.QRectF(track.left(), track.top(), x - track.left(), 30),
            QtGui.QColor(58, 131, 247),
        )
        painter.restore()
        painter.setBrush(QtGui.QColor(181, 181, 183))
        for index in range(10):
            painter.drawEllipse(
                QtCore.QPointF(18 + index * (self.width() - 36) / 9, 22),
                2,
                2,
            )
        painter.setPen(QtGui.QPen(QtGui.QColor(232, 232, 232), 1))
        painter.setBrush(QtGui.QColor(255, 255, 255))
        painter.drawEllipse(QtCore.QPointF(x, 22), 16, 16)

    def _set_position(self, event):
        self.setValue(
            round(1 + (event.position().x() - 18) * 9 / (self.width() - 36))
        )

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            self.setSliderDown(True)
            self._set_position(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            self._set_position(event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setSliderDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self._wheel_delta += event.angleDelta().y() or event.angleDelta().x()
        steps = int(self._wheel_delta / 120)
        self._wheel_delta -= steps * 120
        self.setValue(self.value() + steps)
        event.accept()


class PointSizePopup(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(
            parent,
            QtCore.Qt.WindowType.Popup
            | QtCore.Qt.WindowType.FramelessWindowHint,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(270)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        self.title = QtWidgets.QLabel()
        self.title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)
        self.slider = PointSizeSlider()
        layout.addWidget(self.slider)
        hint = QtWidgets.QLabel(self.tr("Scroll to adjust point size"))
        hint.setObjectName("pointcloudMuted")
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)
        self.slider.valueChanged.connect(self._update_title)
        self._update_title(self.slider.value())

    def _update_title(self, value):
        self.title.setText(
            self.tr("Point size ({value} px)").format(value=value)
        )

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        theme = get_theme()
        painter.setPen(QtGui.QPen(QtGui.QColor(theme["border"]), 1))
        painter.setBrush(QtGui.QColor(theme["surface"]))
        painter.drawRoundedRect(
            QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1), 16, 16
        )

    def show_below(self, button):
        self.adjustSize()
        position = button.mapToGlobal(QtCore.QPoint(0, button.height() + 6))
        position.setX(position.x() + (button.width() - self.width()) // 2)
        available = button.screen().availableGeometry()
        position.setX(
            max(
                available.left(),
                min(position.x(), available.right() - self.width() + 1),
            )
        )
        position.setY(
            max(
                available.top(),
                min(position.y(), available.bottom() - self.height() + 1),
            )
        )
        self.move(position)
        self.show()
        self.slider.setFocus(QtCore.Qt.FocusReason.PopupFocusReason)

    def wheelEvent(self, event):
        self.slider.wheelEvent(event)


class PointCloudToolScrollArea(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pointcloudToolScroll")
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.setFixedHeight(42)
        self.horizontalScrollBar().rangeChanged.connect(self._update_height)

    def _update_height(self, minimum, maximum):
        self.setFixedHeight(
            42
            + (
                self.horizontalScrollBar().sizeHint().height()
                if maximum
                else 0
            )
        )

    def wheelEvent(self, event):
        bar = self.horizontalScrollBar()
        if bar.maximum() == 0:
            event.ignore()
            return
        pixels = event.pixelDelta()
        angles = event.angleDelta()
        delta = (
            pixels.x() or pixels.y()
            if not pixels.isNull()
            else (angles.x() or angles.y()) / 120 * bar.singleStep() * 3
        )
        bar.setValue(bar.value() - round(delta))
        event.accept()


class _RemoveDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent):
        super().__init__(parent)
        self._icons = {}

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(max(28, size.height()))
        return size

    def _icon(self, color, name="trash"):
        key = (color, name)
        if key not in self._icons:
            pixmap = new_icon(name, "svg").pixmap(32, 32)
            if not pixmap.isNull():
                painter = QtGui.QPainter(pixmap)
                painter.setCompositionMode(
                    QtGui.QPainter.CompositionMode.CompositionMode_SourceIn
                )
                painter.fillRect(pixmap.rect(), QtGui.QColor(color))
                painter.end()
            self._icons[key] = QtGui.QIcon(pixmap)
        return self._icons[key]

    def paint(self, painter, option, index):
        options = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        view = self.parent()
        style = view.style()
        colored = index.data(QtCore.Qt.ItemDataRole.BackgroundRole) is not None
        text_rect = style.subElementRect(
            QtWidgets.QStyle.SubElement.SE_ItemViewItemText, options, view
        )
        if (
            colored
            and index.data(QtCore.Qt.ItemDataRole.CheckStateRole) is not None
        ):
            text_rect.setLeft(view.visibility_rect(index).right() + 5)
        removable = index.data(view.REMOVABLE_ROLE) is not False
        if removable:
            text_rect.setRight(view.remove_rect(index).left() - 4)
        text = options.fontMetrics.elidedText(
            options.text,
            QtCore.Qt.TextElideMode.ElideRight,
            max(0, text_rect.width()),
        )
        options.text = ""
        painter.save()
        theme = get_theme()
        selected = options.state & QtWidgets.QStyle.StateFlag.State_Selected
        color = theme["selection_text"] if selected else theme["text"]
        if colored:
            background = options.backgroundBrush.color()
            painter.fillRect(options.rect, background)
            if selected:
                indicator = QtCore.QRect(options.rect)
                indicator.setWidth(3)
                painter.fillRect(indicator, QtGui.QColor("#000000"))
            color = (
                "#000000" if QtGui.qGray(background.rgb()) > 128 else "#ffffff"
            )
            if index.data(QtCore.Qt.ItemDataRole.CheckStateRole) is not None:
                name = (
                    "eye"
                    if options.checkState == QtCore.Qt.CheckState.Checked
                    else "eye-off"
                )
                self._icon(color, name).paint(
                    painter, view.visibility_rect(index).adjusted(3, 3, -3, -3)
                )
        else:
            style.drawControl(
                QtWidgets.QStyle.ControlElement.CE_ItemViewItem,
                options,
                painter,
                view,
            )
        if not options.state & QtWidgets.QStyle.StateFlag.State_Enabled:
            color = theme["text_secondary"]
        painter.setFont(options.font)
        painter.setPen(QtGui.QColor(color))
        painter.drawText(
            text_rect,
            int(
                options.displayAlignment | QtCore.Qt.AlignmentFlag.AlignVCenter
            ),
            text,
        )
        if (
            removable
            and options.state & QtWidgets.QStyle.StateFlag.State_MouseOver
        ):
            rect = view.remove_rect(index)
            self._icon(color).paint(painter, rect.adjusted(5, 5, -5, -5))
        painter.restore()


class PointCloudListWidget(QtWidgets.QListWidget):
    remove_requested = QtCore.pyqtSignal(QtWidgets.QListWidgetItem)
    REMOVABLE_ROLE = QtCore.Qt.ItemDataRole.UserRole.value + 1

    def __init__(
        self, parent=None, *, toggle_selection=False, remove_tooltip=""
    ):
        super().__init__(parent)
        self.toggle_selection = toggle_selection
        self.remove_tooltip = remove_tooltip
        self._pressed_remove = QtCore.QPersistentModelIndex()
        self._skip_release = False
        self.setMouseTracking(True)
        self.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)
        self.setItemDelegate(_RemoveDelegate(self))
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        if toggle_selection:
            self.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
            )

    def remove_rect(self, index):
        rect = self.visualRect(index)
        return QtCore.QRect(rect.right() - 27, rect.center().y() - 12, 24, 24)

    def visibility_rect(self, index):
        rect = self.visualRect(index)
        return QtCore.QRect(rect.left() + 3, rect.center().y() - 12, 24, 24)

    def _remove_at(self, position):
        index = self.indexAt(position)
        if (
            index.isValid()
            and index.data(self.REMOVABLE_ROLE) is not False
            and self.remove_rect(index).contains(position)
        ):
            return index
        return QtCore.QModelIndex()

    def mousePressEvent(self, event):
        self._skip_release = False
        self._pressed_remove = QtCore.QPersistentModelIndex()
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            index = self.indexAt(event.position().toPoint())
            if (
                index.isValid()
                and index.data(QtCore.Qt.ItemDataRole.BackgroundRole)
                is not None
                and self.visibility_rect(index).contains(
                    event.position().toPoint()
                )
            ):
                item = self.itemFromIndex(index)
                item.setCheckState(
                    QtCore.Qt.CheckState.Unchecked
                    if item.checkState() == QtCore.Qt.CheckState.Checked
                    else QtCore.Qt.CheckState.Checked
                )
                self._skip_release = True
                event.accept()
                return
            index = self._remove_at(event.position().toPoint())
            if index.isValid():
                self._pressed_remove = QtCore.QPersistentModelIndex(index)
                event.accept()
                return
            item = self.itemAt(event.position().toPoint())
            if (
                self.toggle_selection
                and event.modifiers() == QtCore.Qt.KeyboardModifier.NoModifier
                and (
                    item is None
                    or (item is self.currentItem() and item.isSelected())
                )
            ):
                self.clearSelection()
                self.setCurrentItem(
                    None, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate
                )
                self._skip_release = True
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pressed_remove.isValid():
            index = self._remove_at(event.position().toPoint())
            pressed = self._pressed_remove
            self._pressed_remove = QtCore.QPersistentModelIndex()
            if QtCore.QModelIndex(pressed) == index:
                self.remove_requested.emit(self.itemFromIndex(index))
            event.accept()
            return
        if self._skip_release:
            self._skip_release = False
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if self.toggle_selection and not self.selectedItems():
            self.setCurrentItem(
                None, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate
            )

    def mouseDoubleClickEvent(self, event):
        index = self.indexAt(event.position().toPoint())
        if (
            index.isValid()
            and index.data(QtCore.Qt.ItemDataRole.CheckStateRole) is not None
            and self.visibility_rect(index).contains(
                event.position().toPoint()
            )
        ):
            event.accept()
            return
        if self._remove_at(event.position().toPoint()).isValid():
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def viewportEvent(self, event):
        if event.type() == QtCore.QEvent.Type.ToolTip:
            index = self._remove_at(event.pos())
            if index.isValid() and self.remove_tooltip:
                QtWidgets.QToolTip.showText(
                    event.globalPos(),
                    self.remove_tooltip,
                    self.viewport(),
                    self.remove_rect(index),
                )
            else:
                QtWidgets.QToolTip.hideText()
            event.accept()
            return True
        return super().viewportEvent(event)


class ClassDefinitionDialog(QtWidgets.QDialog):
    def __init__(
        self, definition=None, used_ids=(), suggested_id=1, parent=None
    ):
        super().__init__(parent)
        self._original = definition
        self._used_ids = frozenset(used_ids)
        self.setWindowTitle(self.tr("Class definition"))
        self.setMinimumWidth(400)
        theme = get_theme()
        self.setStyleSheet(get_dialog_style() + f"""
            QLabel#classDefinitionError {{
                color: {theme["error"]};
            }}
            QPushButton#classColorPreview {{
                min-width: 0;
                min-height: 0;
                max-width: 22px;
                max-height: 22px;
                width: 22px;
                height: 22px;
                padding: 0;
                border: none;
                background-color: transparent;
            }}
            QPushButton#classDefinitionSave {{
                color: white;
                background-color: {theme["primary"]};
                border-color: {theme["primary"]};
            }}
            QPushButton#classDefinitionSave:hover {{
                background-color: {theme["primary_hover"]};
            }}
            QPushButton#classDefinitionSave:pressed {{
                background-color: {theme["primary_pressed"]};
            }}
            """)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)
        form.setColumnStretch(1, 1)
        self.id_input = QtWidgets.QSpinBox()
        self.id_input.setRange(0, MAX_ID)
        self.id_input.setValue(definition.id if definition else suggested_id)
        self.id_input.setEnabled(definition is None)
        if definition is not None:
            self.id_input.setButtonSymbols(
                QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
            )
        self.id_input.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.id_input.installEventFilter(self)
        self.name_input = QtWidgets.QLineEdit(
            definition.name if definition else ""
        )
        self.name_input.setPlaceholderText(self.tr("Class name"))
        self.color_input = QtWidgets.QLineEdit(
            definition.color if definition else "#60A5FA"
        )
        self.color_input.setPlaceholderText("#RRGGBB")
        self.color_input.setMaxLength(7)
        self.color_button = QtWidgets.QPushButton(self.color_input)
        self.color_button.setObjectName("classColorPreview")
        self.color_button.setFixedSize(22, 22)
        self.color_button.setIconSize(QtCore.QSize(22, 22))
        self.color_button.setAutoDefault(False)
        self.color_button.setToolTip(self.tr("Choose color"))
        self.color_button.setAccessibleName(self.tr("Choose color"))
        self.color_button.clicked.connect(self._choose_color)
        self.color_input.setTextMargins(0, 0, 32, 0)
        color_row = QtWidgets.QHBoxLayout(self.color_input)
        color_row.setContentsMargins(0, 0, 8, 0)
        color_row.setSpacing(0)
        color_row.addStretch()
        color_row.addWidget(
            self.color_button, 0, QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        for row, (text, widget) in enumerate(
            (
                (self.tr("Semantic ID"), self.id_input),
                (self.tr("Name"), self.name_input),
                (self.tr("Color"), self.color_input),
            )
        ):
            label = QtWidgets.QLabel(text)
            label.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            label.setBuddy(widget)
            form.addWidget(label, row, 0)
            form.addWidget(widget, row, 1)
        layout.addLayout(form)
        self.error_label = QtWidgets.QLabel()
        self.error_label.setObjectName("classDefinitionError")
        self.error_label.setWordWrap(True)
        self.error_label.setMinimumHeight(self.fontMetrics().height())
        form.addWidget(self.error_label, 3, 1)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(12)
        self.save_button = QtWidgets.QPushButton(self.tr("Save"))
        self.save_button.setObjectName("classDefinitionSave")
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"))
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.save_button)
        buttons.addStretch()
        buttons.addWidget(self.cancel_button)
        form.addLayout(buttons, 4, 1)
        self.save_button.setDefault(True)
        self.name_input.textChanged.connect(self.error_label.clear)
        self.id_input.valueChanged.connect(self.error_label.clear)
        self.color_input.textChanged.connect(self.error_label.clear)
        self.color_input.textChanged.connect(self._update_color)
        self._update_color()
        self.name_input.setFocus()

    def eventFilter(self, watched, event):
        if (
            watched is self.id_input
            and event.type() == QtCore.QEvent.Type.Wheel
            and not watched.hasFocus()
        ):
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def _update_color(self):
        color = QtGui.QColor(self.color_input.text().strip())
        pixmap = QtGui.QPixmap(48, 48)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(
            color if color.isValid() else QtGui.QColor(get_theme()["surface"])
        )
        painter.drawRoundedRect(QtCore.QRectF(1, 1, 46, 46), 8, 8)
        painter.end()
        self.color_button.setIcon(QtGui.QIcon(pixmap))

    def _choose_color(self):
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(self.color_input.text()),
            self,
            self.tr("Choose color"),
        )
        if color.isValid():
            self.color_input.setText(color.name().upper())

    def definition(self):
        return ClassDefinition(
            self.id_input.value(),
            self.name_input.text().strip(),
            self.color_input.text().strip().upper(),
        )

    def accept(self):
        definition = self.definition()
        if not definition.name:
            self.error_label.setText(self.tr("Enter a class name."))
            self.name_input.setFocus()
            return
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", definition.color):
            self.error_label.setText(
                self.tr("Enter a color in #RRGGBB format.")
            )
            self.color_input.setFocus()
            return
        if self._original is None and definition.id in self._used_ids:
            self.error_label.setText(
                self.tr("This semantic ID already has a definition.")
            )
            self.id_input.setFocus()
            return
        super().accept()


class ShortcutScrollBar(QtWidgets.QScrollBar):
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(225, 224, 226))
        center = self.width() / 2
        for y, direction in ((5, -1), (self.height() - 5, 1)):
            painter.drawPolygon(
                QtGui.QPolygonF(
                    [
                        QtCore.QPointF(center - 4, y - direction * 2),
                        QtCore.QPointF(center + 4, y - direction * 2),
                        QtCore.QPointF(center, y + direction * 3),
                    ]
                )
            )


class ShortcutsDialog(QtWidgets.QDialog):
    def __init__(self, groups, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle(self.tr("Keyboard shortcuts"))
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(600, 560)
        theme = get_theme()
        muted = (
            "#666666"
            if QtGui.QColor(theme["background"]).lightness() > 127
            else theme["text_secondary"]
        )
        self.setStyleSheet(f"""
            QFrame#shortcutCard {{
                background: {theme['background_secondary']};
                border: 1px solid {theme['border']}; border-radius: 20px;
            }}
            QLabel {{ color: {muted}; background: transparent; border: none; padding: 0; }}
            QLabel#shortcutTitle {{ color: {theme['text']}; font-size: 20px; font-weight: 600; }}
            QLabel#shortcutGroup {{ color: {theme['text']}; font-weight: 600; padding: 12px 0 0 0; }}
            QLabel#shortcutKey {{ background: rgb(235, 234, 235);
                color: #666666; border-radius: 7px; padding: 1px 7px; font-size: 12px; }}
            QLineEdit {{ background: {theme['background']}; color: {muted};
                border: 1px solid {theme['border']}; border-radius: 16px;
                padding: 0 10px 0 0; min-height: 30px; max-height: 30px; }}
            QLineEdit:focus {{ border-color: {theme['border']}; }}
            QToolButton {{ border: 1px solid transparent; background: transparent; padding: 3px;
                color: {theme['text']}; border-radius: 4px; }}
            QToolButton#shortcutCloseButton {{
                min-width: 18px; max-width: 18px;
                min-height: 18px; max-height: 18px;
                padding: 3px;
            }}
            QToolButton:hover {{ background: {theme['surface_hover']}; border-color: {theme['border']}; }}
            QToolButton:pressed {{ background: {theme['surface_pressed']}; }}
            QScrollArea, QWidget#shortcutContents {{ border: none; background: transparent; }}
            QScrollBar:vertical {{ background: transparent; width: 9px; margin: 13px 0; }}
            QScrollBar::handle:vertical {{ background: rgb(225, 224, 226);
                border-radius: 4px; min-height: 32px; }}
            QScrollBar::sub-line:vertical {{ height: 13px; subcontrol-origin: margin;
                subcontrol-position: top; background: transparent; border: none; }}
            QScrollBar::add-line:vertical {{ height: 13px; subcontrol-origin: margin;
                subcontrol-position: bottom; background: transparent; border: none; }}
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{ image: none; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        """)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        card = QtWidgets.QFrame()
        card.setObjectName("shortcutCard")
        outer.addWidget(card)
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(12)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(self.windowTitle())
        title.setObjectName("shortcutTitle")
        header.addWidget(title)
        header.addStretch()
        close = QtWidgets.QToolButton()
        close.setObjectName("shortcutCloseButton")
        close.setText("×")
        close.setAccessibleName(self.tr("Close"))
        close.setFixedSize(26, 26)
        close.clicked.connect(self.reject)
        header.addWidget(close)
        layout.addLayout(header)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText(self.tr("Search shortcuts"))
        self.search.setAccessibleName(self.tr("Search shortcuts"))
        self.search.setFixedHeight(32)
        self.search.setTextMargins(34, 0, 0, 0)
        search_icon = QtWidgets.QLabel(self.search)
        search_icon.setGeometry(12, 9, 14, 14)
        search_icon.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        pixmap = QtGui.QPixmap(28, 28)
        pixmap.setDevicePixelRatio(2)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor(muted), 1))
        painter.drawEllipse(QtCore.QRectF(1, 1, 9, 9))
        painter.drawLine(QtCore.QPointF(9, 9), QtCore.QPointF(13, 13))
        painter.end()
        search_icon.setPixmap(pixmap)
        layout.addWidget(self.search)
        scroll = QtWidgets.QScrollArea()
        scroll.setVerticalScrollBar(
            ShortcutScrollBar(QtCore.Qt.Orientation.Vertical, scroll)
        )
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        contents = QtWidgets.QWidget()
        contents.setObjectName("shortcutContents")
        rows_layout = QtWidgets.QVBoxLayout(contents)
        rows_layout.setContentsMargins(0, 0, 12, 0)
        rows_layout.setSpacing(4)
        self.groups = []
        for heading, entries in groups:
            label = QtWidgets.QLabel(heading)
            label.setObjectName("shortcutGroup")
            label.setIndent(0)
            rows_layout.addWidget(label)
            rows = []
            for text, keys in entries:
                row = QtWidgets.QWidget()
                row_layout = QtWidgets.QHBoxLayout(row)
                row_layout.setContentsMargins(0, 7, 0, 7)
                row_layout.setSpacing(6)
                description = QtWidgets.QLabel(text)
                description.setWordWrap(True)
                row_layout.addWidget(description, 1)
                for key in keys:
                    key_label = QtWidgets.QLabel(key)
                    key_label.setObjectName("shortcutKey")
                    row_layout.addWidget(key_label)
                rows_layout.addWidget(row)
                rows.append((row, " ".join([heading, text, *keys]).casefold()))
            self.groups.append((label, rows))
        self.empty = QtWidgets.QLabel(self.tr("No matching shortcuts"))
        self.empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty.hide()
        rows_layout.addWidget(self.empty)
        rows_layout.addStretch()
        scroll.setWidget(contents)
        layout.addWidget(scroll, 1)
        self.search.textChanged.connect(self._filter)
        self.search.setFocus()

    def _filter(self, text):
        terms = text.casefold().split()
        any_visible = False
        for heading, rows in self.groups:
            group_visible = False
            for row, searchable in rows:
                visible = all(term in searchable for term in terms)
                row.setVisible(visible)
                group_visible |= visible
            heading.setVisible(group_visible)
            any_visible |= group_visible
        self.empty.setVisible(not any_visible)
