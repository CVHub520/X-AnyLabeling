from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from anylabeling.views.labeling.utils.theme import get_theme

from .icons import themed_icon, theme_icon_color
from .style import get_segment_list_style
from .utils import color_for_label, ms_to_timecode


def _color_pixmap(color, size=12):
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setBrush(QColor(color))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return pm


class SegmentMenuItem(QWidget):
    clicked = pyqtSignal()

    def __init__(self, text, icon, parent=None):
        super().__init__(parent)
        self.setObjectName("XvaSegmentMenuItem")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(28)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 0, 12, 0)
        layout.setSpacing(8)

        icon_label = QLabel(self)
        icon_label.setFixedSize(16, 16)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setPixmap(icon.pixmap(QSize(14, 14)))
        icon_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )

        text_label = QLabel(text, self)
        text_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )

        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        layout.addStretch(1)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            return
        super().mouseReleaseEvent(event)


class SegmentListPanel(QWidget):
    segmentActivated = pyqtSignal(str)  # id (double-click → jump)
    segmentSelected = pyqtSignal(str)  # id (single click)
    segmentDeleted = pyqtSignal(str)  # id
    segmentRelabelRequested = pyqtSignal(str)  # id

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header_layout = QGridLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setColumnMinimumWidth(0, 72)
        header_layout.setColumnStretch(1, 1)
        header_layout.setColumnMinimumWidth(2, 72)
        header = QLabel(self.tr("Segments"))
        header.setObjectName("XvaPanelTitle")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(header, 0, 1)
        self.count_label = QLabel(self.tr("total: 0"))
        self.count_label.setStyleSheet(
            f"color: {get_theme()['text_secondary']}; font-size: 11px;"
        )
        header_layout.addWidget(
            self.count_label, 0, 2, Qt.AlignmentFlag.AlignRight
        )
        layout.addLayout(header_layout)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(get_segment_list_style())
        self.list_widget.setIconSize(QSize(12, 12))
        self.list_widget.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.list_widget.itemSelectionChanged.connect(
            self._on_selection_changed
        )
        self.list_widget.itemDoubleClicked.connect(self._on_double_clicked)
        self.list_widget.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.list_widget.customContextMenuRequested.connect(self._show_menu)
        layout.addWidget(self.list_widget, 1)

    def set_segments(self, segments, label_colors=None):
        label_colors = label_colors or {}
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for seg in segments:
            color = color_for_label(seg.label, label_colors)
            text = "{label}  |  {start} – {end}  ({dur})".format(
                label=seg.label or self.tr("(unlabeled)"),
                start=ms_to_timecode(seg.start_ms, with_ms=False),
                end=ms_to_timecode(seg.end_ms, with_ms=False),
                dur=ms_to_timecode(
                    max(0, seg.end_ms - seg.start_ms), with_ms=False
                ),
            )
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, seg.id)
            item.setIcon(QIcon(_color_pixmap(color, size=14)))
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)
        self.count_label.setText(
            self.tr("total: {n}").format(n=self.list_widget.count())
        )

    def select(self, seg_id):
        if not seg_id:
            self.list_widget.clearSelection()
            return
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == seg_id:
                self.list_widget.blockSignals(True)
                self.list_widget.setCurrentRow(i)
                self.list_widget.blockSignals(False)
                return

    def current_id(self):
        item = self.list_widget.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def _on_selection_changed(self):
        self.segmentSelected.emit(self.current_id())

    def _on_double_clicked(self, item):
        sid = item.data(Qt.ItemDataRole.UserRole)
        if sid:
            self.segmentActivated.emit(sid)

    def _show_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.setObjectName("XvaSegmentMenu")
        menu.setContentsMargins(0, 0, 0, 0)
        selected = {"action": None}
        jump = self._add_menu_button(
            menu,
            self.tr("Jump to start"),
            themed_icon(
                "arrow-right", "svg", theme_icon_color("text_secondary"), 14
            ),
            selected,
        )
        relabel = self._add_menu_button(
            menu,
            self.tr("Edit segment…"),
            themed_icon("edit", "svg", theme_icon_color("text_secondary"), 14),
            selected,
        )
        menu.addSeparator()
        delete = self._add_menu_button(
            menu,
            self.tr("Delete"),
            themed_icon("trash", "svg", theme_icon_color("error"), 14),
            selected,
        )
        self._sync_menu_item_widths(jump, relabel, delete)
        menu.exec(self.list_widget.viewport().mapToGlobal(pos))
        chosen = selected["action"]
        if chosen == jump:
            self.segmentActivated.emit(sid)
        elif chosen == relabel:
            self.segmentRelabelRequested.emit(sid)
        elif chosen == delete:
            self.segmentDeleted.emit(sid)

    def _add_menu_button(self, menu, text, icon, selected):
        action = QWidgetAction(menu)
        item = SegmentMenuItem(text, icon, menu)
        item.clicked.connect(
            lambda _checked=False, a=action: self._choose_menu_action(
                menu, selected, a
            )
        )
        action.setDefaultWidget(item)
        menu.addAction(action)
        return action

    def _sync_menu_item_widths(self, *actions):
        width = (
            max(
                action.defaultWidget().sizeHint().width() for action in actions
            )
            + 8
        )
        for action in actions:
            action.defaultWidget().setFixedWidth(width)

    def _choose_menu_action(self, menu, selected, action):
        selected["action"] = action
        menu.close()

    def delete_current(self):
        sid = self.current_id()
        if sid:
            self.segmentDeleted.emit(sid)
