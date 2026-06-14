from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QStyle


# https://stackoverflow.com/a/2039745/4158863
class HTMLDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        self.parent = parent
        super(HTMLDelegate, self).__init__()

    def _document_for_index(self, index, font):
        doc = QtGui.QTextDocument(self)
        doc.setDefaultFont(font)
        if index is not None and index.isValid():
            doc.setHtml(index.data(Qt.ItemDataRole.DisplayRole) or "")
        return doc

    def paint(self, painter, option, index):
        painter.save()

        options = QtWidgets.QStyleOptionViewItem(option)

        self.initStyleOption(options, index)
        doc = self._document_for_index(index, options.font)
        options.text = ""

        style = (
            QtWidgets.QApplication.style()
            if options.widget is None
            else options.widget.style()
        )
        style.drawControl(
            QStyle.ControlElement.CE_ItemViewItem, options, painter
        )

        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()

        if option.state & QStyle.StateFlag.State_Selected:
            ctx.palette.setColor(
                QPalette.ColorRole.Text,
                option.palette.color(
                    QPalette.ColorGroup.Active,
                    QPalette.ColorRole.HighlightedText,
                ),
            )
        else:
            ctx.palette.setColor(
                QPalette.ColorRole.Text,
                option.palette.color(
                    QPalette.ColorGroup.Active, QPalette.ColorRole.Text
                ),
            )

        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText, options
        )

        if index.column() != 0:
            text_rect.adjust(5, 0, 0, 0)

        margin = max(0, int((option.rect.height() - doc.size().height()) / 2))
        text_rect.setTop(text_rect.top() + margin)

        painter.translate(text_rect.topLeft())
        painter.setClipRect(text_rect.translated(-text_rect.topLeft()))
        doc.documentLayout().draw(painter, ctx)

        painter.restore()

    # QT Overload
    def sizeHint(self, option, index):
        font = option.font if option is not None else QtGui.QFont()
        doc = self._document_for_index(index, font)
        font_metrics = QtGui.QFontMetrics(font)
        return QtCore.QSize(
            int(doc.idealWidth()),
            max(int(doc.size().height()), font_metrics.height() + 4),
        )


class LabelListWidgetItem(QtGui.QStandardItem):
    def __init__(self, text=None, shape=None):
        super(LabelListWidgetItem, self).__init__()
        self.setText(text or "")
        self.set_shape(shape)

        self.setCheckable(True)
        self.setCheckState(Qt.CheckState.Checked)
        self.setEditable(False)
        self.setTextAlignment(Qt.AlignmentFlag.AlignBottom)

    def clone(self):
        return LabelListWidgetItem(self.text(), self.shape())

    def set_shape(self, shape):
        self.setData(shape, Qt.ItemDataRole.UserRole)

    def shape(self):
        return self.data(Qt.ItemDataRole.UserRole)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.text()!r}")'


class StandardItemModel(QtGui.QStandardItemModel):
    itemDropped = QtCore.pyqtSignal()

    # QT Overload
    def removeRows(self, *args, **kwargs):
        ret = super().removeRows(*args, **kwargs)
        self.itemDropped.emit()
        return ret


class LabelListWidget(QtWidgets.QListView):
    item_double_clicked = QtCore.pyqtSignal(LabelListWidgetItem)
    item_selection_changed = QtCore.pyqtSignal(list, list)

    def __init__(self):
        super().__init__()
        self._selected_items = []
        self._ignore_mouse_move_selection = False
        self._preserved_selected_items = []

        self.setWindowFlags(Qt.WindowType.Window)
        self.setModel(StandardItemModel())
        self.model().setItemPrototype(LabelListWidgetItem())
        self.setItemDelegate(HTMLDelegate())
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.NoDragDrop
        )

        self.doubleClicked.connect(self.item_double_clicked_event)
        self.selectionModel().selectionChanged.connect(
            self.item_selection_changed_event
        )

    def __len__(self):
        return self.model().rowCount()

    def __getitem__(self, i):
        return self.model().item(i)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def item_dropped(self):
        return self.model().itemDropped

    @property
    def item_changed(self):
        return self.model().itemChanged

    def item_selection_changed_event(self, selected, deselected):
        selected = [self.model().itemFromIndex(i) for i in selected.indexes()]
        deselected = [
            self.model().itemFromIndex(i) for i in deselected.indexes()
        ]
        self.item_selection_changed.emit(selected, deselected)

    def item_double_clicked_event(self, index):
        self.item_double_clicked.emit(self.model().itemFromIndex(index))

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.indexAt(event.pos())
            if index.isValid():
                self.item_double_clicked_event(index)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        self._preserved_selected_items = []
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.indexAt(event.pos())
            self._ignore_mouse_move_selection = (
                index.isValid() and self.selectionModel().isSelected(index)
            )
            if (
                self._ignore_mouse_move_selection
                and len(self.selectedIndexes()) > 1
                and event.modifiers() == Qt.KeyboardModifier.NoModifier
            ):
                self._preserved_selected_items = self.selected_items()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._ignore_mouse_move_selection
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._ignore_mouse_move_selection = False
        super().mouseReleaseEvent(event)
        if self._preserved_selected_items:
            self.clearSelection()
            for item in self._preserved_selected_items:
                self.select_item(item)
            self._preserved_selected_items = []

    def selected_items(self):
        return [self.model().itemFromIndex(i) for i in self.selectedIndexes()]

    def scroll_to_item(self, item):
        self.scrollTo(self.model().indexFromItem(item))

    def add_iem(self, item):
        if not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem")
        self.model().setItem(self.model().rowCount(), 0, item)

    def remove_item(self, item):
        index = self.model().indexFromItem(item)
        self.model().removeRows(index.row(), 1)

    def select_item(self, item):
        index = self.model().indexFromItem(item)
        self.selectionModel().select(
            index, QtCore.QItemSelectionModel.SelectionFlag.Select
        )

    def find_item_by_shape(self, shape):
        for row in range(self.model().rowCount()):
            item = self.model().item(row, 0)
            if item.shape() == shape:
                return item
        # NOTE: Handle the case when the shape is not found
        # This is a temporary solution to prevent a crash.
        # Further investigation and a more robust fix are recommended.
        return None
        # raise ValueError(f"cannot find shape: {shape}")

    def clear(self):
        self.model().clear()

    def item_at_index(self, index):
        return self.model().item(index, 0)
