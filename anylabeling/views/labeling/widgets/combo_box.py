from PyQt6 import QtCore, QtGui, QtWidgets

from ..utils.theme import get_mode, get_theme
from .text_input import paint_input_frame


class _ComboItemDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, combo, parent):
        super().__init__(parent)
        self._combo = combo

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(32)
        return size

    def paint(self, painter, option, index):
        option = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(option, index)
        painter.save()
        selected = index.model().mapToSource(index).row() == (
            self._combo.currentIndex()
        )
        if option.state & (
            QtWidgets.QStyle.StateFlag.State_Selected
            | QtWidgets.QStyle.StateFlag.State_MouseOver
        ):
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            color = "#262626" if get_mode() == "dark" else "#f5f5f5"
            painter.setBrush(QtGui.QColor(color))
            painter.drawRoundedRect(QtCore.QRectF(option.rect), 4, 4)
        painter.setPen(QtGui.QColor(get_theme()["text"]))
        painter.setFont(option.font)
        text_rect = option.rect.adjusted(8, 0, -32, 0)
        text = option.fontMetrics.elidedText(
            str(index.data(QtCore.Qt.ItemDataRole.DisplayRole) or ""),
            QtCore.Qt.TextElideMode.ElideRight,
            text_rect.width(),
        )
        painter.drawText(
            text_rect,
            QtCore.Qt.AlignmentFlag.AlignVCenter
            | QtCore.Qt.AlignmentFlag.AlignLeft,
            text,
        )
        if selected:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            pen = QtGui.QPen(QtGui.QColor(get_theme()["text"]), 1.5)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            x = option.rect.right() - 16
            y = option.rect.center().y()
            path = QtGui.QPainterPath()
            path.moveTo(x - 5, y)
            path.lineTo(x - 1, y + 4)
            path.lineTo(x + 6, y - 4)
            painter.drawPath(path)
        painter.restore()


class _ComboArrow(QtWidgets.QToolButton):
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor(get_theme()["text_secondary"]), 1.5)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        center = QtCore.QPointF(self.rect().center())
        path = QtGui.QPainterPath()
        path.moveTo(center.x() - 4, center.y() - 2)
        path.lineTo(center.x(), center.y() + 2)
        path.lineTo(center.x() + 4, center.y() - 2)
        painter.drawPath(path)


class ComboBox(QtWidgets.QComboBox):
    _shadow_margin = 8
    _row_height = 32
    _panel_padding = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.setCompleter(None)
        self.setFixedHeight(32)
        self.setMaxVisibleItems(8)
        self._selected_index = -1
        self._popup = QtWidgets.QWidget(
            self,
            QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self._popup.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TranslucentBackground
        )
        self._popup.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating
        )
        layout = QtWidgets.QVBoxLayout(self._popup)
        layout.setContentsMargins(*([self._shadow_margin] * 4))
        panel = QtWidgets.QFrame(self._popup)
        panel.setObjectName("comboPopupPanel")
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(*([self._panel_padding] * 4))
        self._view = QtWidgets.QListView(panel)
        self._view.setMinimumSize(0, 0)
        self._view.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        self._view.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._view.viewport().setAutoFillBackground(False)
        self._view.viewport().setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TranslucentBackground
        )
        self._view.setMouseTracking(True)
        self._view.setItemDelegate(_ComboItemDelegate(self, self._view))
        self._view.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._view.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self._view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._proxy = QtCore.QSortFilterProxyModel(self)
        self._proxy.setFilterCaseSensitivity(
            QtCore.Qt.CaseSensitivity.CaseInsensitive
        )
        self._proxy.setFilterKeyColumn(self.modelColumn())
        self._proxy.setSourceModel(self.model())
        self._view.setModel(self._proxy)
        self._view.clicked.connect(self._choose)
        self._view.entered.connect(
            lambda _index: self._view.selectionModel().clearSelection()
        )
        panel_layout.addWidget(self._view)
        layout.addWidget(panel)
        shadow = QtWidgets.QGraphicsDropShadowEffect(panel)
        shadow.setBlurRadius(12)
        shadow.setOffset(0, 3)
        shadow.setColor(QtGui.QColor(0, 0, 0, 45))
        panel.setGraphicsEffect(shadow)
        self._arrow = _ComboArrow(self)
        self._arrow.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._arrow.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self._arrow.clicked.connect(lambda: self.showPopup())
        self.lineEdit().textEdited.connect(self._filter)
        self.lineEdit().installEventFilter(self)
        self.currentIndexChanged.connect(self._remember_selection)
        t = get_theme()
        self.setStyleSheet(f"""
            QComboBox {{
                background: {t["background"]}; color: {t["text"]};
                border: 3px solid transparent; border-radius: 8px;
                min-height: 26px; max-height: 26px;
                padding: 0 7px; font-size: 14px;
            }}
            QComboBox::drop-down {{ border: none; width: 24px; }}
            QComboBox::down-arrow {{ image: none; }}
            QLineEdit {{ background: transparent; border: none; padding: 0; color: {t["text"]}; font-size: 14px; }}
            QToolButton {{ background: transparent; border: none; padding: 0; }}
        """)
        panel.setStyleSheet(f"""
            QFrame#comboPopupPanel {{
                background: {t["background"]};
                border: 1px solid {t["border_light"]}; border-radius: 9px;
            }}
            QListView, QListView QWidget {{ background: transparent; border: none; outline: none;
                color: {t["text"]}; font-size: 14px; }}
        """)

    def paintEvent(self, event):
        focused = self.lineEdit().hasFocus() or self._popup.isVisible()
        paint_input_frame(self, focused)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._arrow.setGeometry(self.width() - 30, 1, 28, self.height() - 2)

    def _remember_selection(self, index):
        if index >= 0:
            self._selected_index = index

    def _filter(self, text):
        self._proxy.setFilterFixedString(text)
        self._view.setCurrentIndex(QtCore.QModelIndex())
        if not self._popup.isVisible():
            self.showPopup(filter_text=text)
        else:
            self._resize_popup()

    def _resize_popup(self):
        count = min(self._proxy.rowCount(), self.maxVisibleItems())
        panel_height = max(1, count) * self._row_height + 2 * (
            self._panel_padding + 1
        )
        margin = self._shadow_margin
        height = panel_height + 2 * margin
        pos = self.mapToGlobal(
            QtCore.QPoint(-margin + 2, self.height() + 4 - margin)
        )
        screen = QtWidgets.QApplication.screenAt(pos) or self.screen()
        if pos.y() + height - margin > screen.availableGeometry().bottom() + 1:
            pos.setY(
                self.mapToGlobal(
                    QtCore.QPoint(0, -panel_height - 4 - margin)
                ).y()
            )
        self._popup.setGeometry(
            pos.x(), pos.y(), self.width() - 4 + 2 * margin, height
        )

    def showPopup(self, filter_text=""):
        if self._popup.isVisible():
            self.hidePopup()
            return
        self._proxy.setFilterFixedString(filter_text)
        self._view.setCurrentIndex(QtCore.QModelIndex())
        if not filter_text:
            self.setEditText("")
        self._resize_popup()
        self._popup.show()
        if not filter_text and self._selected_index >= 0:
            index = self._proxy.mapFromSource(
                self.model().index(self._selected_index, self.modelColumn())
            )
            self._view.setCurrentIndex(index)
            self._view.scrollTo(
                index, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter
            )
            self._view.selectionModel().clearSelection()
        self.lineEdit().setFocus()
        QtWidgets.QApplication.instance().installEventFilter(self)

    def hidePopup(self):
        if not self._popup.isVisible():
            return
        self._popup.hide()
        QtWidgets.QApplication.instance().removeEventFilter(self)
        self.setEditText(self.itemText(self._selected_index))

    def _choose(self, index):
        source = self._proxy.mapToSource(index)
        self.setCurrentIndex(source.row())
        self.activated.emit(source.row())
        self.hidePopup()

    def eventFilter(self, watched, event):
        if self._popup.isVisible():
            if event.type() == QtCore.QEvent.Type.ApplicationDeactivate:
                self.hidePopup()
            elif event.type() == QtCore.QEvent.Type.MouseButtonPress:
                pos = event.globalPosition().toPoint()
                if not self._popup.geometry().contains(
                    pos
                ) and not QtCore.QRect(
                    self.mapToGlobal(QtCore.QPoint()), self.size()
                ).contains(
                    pos
                ):
                    self.hidePopup()
            elif (
                event.type() == QtCore.QEvent.Type.KeyPress
                and watched is self.lineEdit()
            ):
                key = event.key()
                if key == QtCore.Qt.Key.Key_Escape:
                    self.hidePopup()
                    return True
                if key in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Up):
                    row = self._view.currentIndex().row()
                    row += 1 if key == QtCore.Qt.Key.Key_Down else -1
                    row = max(0, min(row, self._proxy.rowCount() - 1))
                    self._view.setCurrentIndex(self._proxy.index(row, 0))
                    self._view.scrollTo(self._view.currentIndex())
                    return True
                if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                    index = self._view.currentIndex()
                    if not index.isValid() and self._proxy.rowCount():
                        index = self._proxy.index(0, 0)
                    if index.isValid():
                        self._choose(index)
                    return True
        elif (
            watched is self.lineEdit()
            and event.type() == QtCore.QEvent.Type.MouseButtonPress
        ):
            self.showPopup()
        return super().eventFilter(watched, event)

    def hideEvent(self, event):
        self.hidePopup()
        super().hideEvent(event)
