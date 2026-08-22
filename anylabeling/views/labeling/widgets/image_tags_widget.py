from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from ..utils.image_tags import normalize_image_tag
from ..utils.qt import new_icon_path
from .popup import Popup

IMAGE_TAG_MIME_TYPE = "application/x-xanylabeling-image-tag"
IMAGE_TAG_HEIGHT = 28
IMAGE_TAG_SPACING = 6
IMAGE_TAG_MAX_ROWS = 3
IMAGE_TAG_RADIUS = IMAGE_TAG_HEIGHT // 2
IMAGE_TAG_CLOSE_SIZE = 12


class ImageTagsFlowLayout(QtWidgets.QLayout):
    def __init__(self, parent=None, margin=0, spacing=6):
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def insertWidget(self, index, widget):
        self.addChildWidget(widget)
        self._items.insert(index, QtWidgets.QWidgetItem(widget))
        self.invalidate()

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QtCore.QSize(
            margins.left() + margins.right(),
            margins.top() + margins.bottom(),
        )
        return size

    def clear(self):
        while self._items:
            item = self.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _do_layout(self, rect, test_only):
        margins = self.contentsMargins()
        area = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x = area.x()
        y = area.y()
        line_height = 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width()
            if next_x > area.right() + 1 and line_height:
                x = area.x()
                y += line_height + self.spacing()
                next_x = x + hint.width()
                line_height = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = next_x + self.spacing()
            line_height = max(line_height, hint.height())
        return y + line_height + margins.bottom() - rect.y()


class _TagLineEdit(QtWidgets.QLineEdit):
    escape_pressed = QtCore.pyqtSignal()
    focus_lost = QtCore.pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.escape_pressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.focus_lost.emit()


class ImageTagChip(QtWidgets.QFrame):
    edit_requested = QtCore.pyqtSignal(int)
    delete_requested = QtCore.pyqtSignal(int)
    selection_toggled = QtCore.pyqtSignal(int)
    drag_requested = QtCore.pyqtSignal(int, object)
    status_message = QtCore.pyqtSignal(str)

    def __init__(self, index, text, color, parent=None):
        super().__init__(parent)
        self.index = index
        self.text = text
        self._batch_mode = False
        self._selected = False
        self._hovered = False
        self._press_position = None
        self._color = color
        self._foreground = QtGui.QColor("#111111")
        self.setObjectName("imageTagChip")
        self.setFixedHeight(IMAGE_TAG_HEIGHT)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setAccessibleName(text)

        self.text_label = QtWidgets.QLabel(self)
        self.text_label.setTextFormat(Qt.TextFormat.PlainText)
        self.text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.NoTextInteraction
        )
        self.text_label.setMaximumWidth(220)

        self.delete_button = QtWidgets.QPushButton("×", self)
        self.delete_button.setObjectName("imageTagDeleteButton")
        self.delete_button.setFixedSize(
            IMAGE_TAG_CLOSE_SIZE, IMAGE_TAG_CLOSE_SIZE
        )
        self.delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_button.setToolTip(self.tr("Delete tag"))
        self.delete_button.setAccessibleName(self.tr("Delete tag"))
        self.delete_button.setStyleSheet(
            "QPushButton#imageTagDeleteButton {"
            "background-color: rgba(45, 45, 45, 190); color: white; "
            "border: none; border-radius: 6px; font-size: 10px; "
            "font-weight: bold; padding: 0; min-width: 12px; "
            "max-width: 12px; min-height: 12px; max-height: 12px; }"
            "QPushButton#imageTagDeleteButton:hover {"
            "background-color: rgb(205, 55, 55); }"
        )
        self.delete_button.clicked.connect(
            lambda: self.delete_requested.emit(self.index)
        )
        self._set_delete_visible(False)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(9, 0, 9, 0)
        layout.addWidget(self.text_label)
        self._update_text()
        self._update_style()

    def _update_text(self):
        metrics = QtGui.QFontMetrics(self.text_label.font())
        self.text_label.setText(
            metrics.elidedText(self.text, Qt.TextElideMode.ElideRight, 220)
        )

    def _update_style(self, drop_position=None):
        color = QtGui.QColor(*self._color)
        luminance = (
            0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
        )
        foreground = "#111111" if luminance > 160 else "#ffffff"
        self._foreground = QtGui.QColor(foreground)
        drop_style = ""
        if drop_position:
            highlight = (
                self.palette().color(QtGui.QPalette.ColorRole.Highlight).name()
            )
            drop_style = f"border-{drop_position}: 4px solid {highlight};"
        self.setStyleSheet(
            "QFrame#imageTagChip {"
            f"background-color: {color.name()}; color: {foreground}; "
            "border: none; "
            f"{drop_style} "
            f"border-radius: {IMAGE_TAG_RADIUS}px; }}"
            f"QFrame#imageTagChip QLabel {{ color: {foreground}; border: none; }}"
        )

    def _set_delete_visible(self, visible):
        self.delete_button.setVisible(visible and not self._batch_mode)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.delete_button.move(self.width() - IMAGE_TAG_CLOSE_SIZE, 0)
        self.delete_button.raise_()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._batch_mode or not self._selected:
            return
        text_rect = self.text_label.geometry()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(self._foreground, 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        y = text_rect.center().y()
        painter.drawLine(text_rect.left(), y, text_rect.right(), y)

    def set_batch_mode(self, enabled, selected=False):
        self._batch_mode = enabled
        self._selected = selected
        if enabled and self._hovered:
            self._hovered = False
            self.status_message.emit("")
        self._set_delete_visible(False)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._update_style()
        self.update()

    def set_drop_position(self, position=None):
        self._update_style(drop_position=position)

    def enterEvent(self, event):
        if not self._batch_mode:
            self._hovered = True
            self._set_delete_visible(True)
            self.status_message.emit(
                self.tr("Double-click a tag to edit its text.")
            )
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._set_delete_visible(False)
        if self._hovered:
            self._hovered = False
            self.status_message.emit("")
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._batch_mode
        ):
            self.edit_requested.emit(self.index)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_position = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._batch_mode or self._press_position is None:
            return
        if not event.buttons() & Qt.MouseButton.LeftButton:
            return
        distance = (
            event.position().toPoint() - self._press_position
        ).manhattanLength()
        if distance >= QtWidgets.QApplication.startDragDistance():
            self._press_position = None
            self.drag_requested.emit(self.index, self)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._batch_mode:
            self.selection_toggled.emit(self.index)
            event.accept()
            return
        self._press_position = None
        super().mouseReleaseEvent(event)


class _TagsDropWidget(QtWidgets.QWidget):
    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(IMAGE_TAG_MIME_TYPE):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(IMAGE_TAG_MIME_TYPE):
            self.panel.update_drag_target(event.position().toPoint())
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.panel.clear_drag_target()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasFormat(IMAGE_TAG_MIME_TYPE):
            source = int(bytes(event.mimeData().data(IMAGE_TAG_MIME_TYPE)))
            self.panel.finish_drag(source, event.position().toPoint())
            event.acceptProposedAction()


class ImageTagsWidget(QtWidgets.QFrame):
    tags_changed = QtCore.pyqtSignal(list)
    status_message = QtCore.pyqtSignal(str)

    def __init__(self, color_for_tag, parent=None):
        super().__init__(parent)
        self._color_for_tag = color_for_tag
        self._tags = []
        self._chips = []
        self._mode = "normal"
        self._selected = set()
        self._editing_index = None
        self._input = None
        self._drop_index = None
        self._interactions_enabled = False
        self._copy_all_button = None
        self._select_all_button = None
        self._delete_selected_button = None
        self._cancel_button = None
        self.setObjectName("imageTagsPanel")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )

        self._content = _TagsDropWidget(self)
        self._flow = ImageTagsFlowLayout(
            self._content,
            margin=0,
            spacing=IMAGE_TAG_SPACING,
        )
        self._flow.setContentsMargins(
            IMAGE_TAG_SPACING, 0, IMAGE_TAG_SPACING, 0
        )
        self._content.setLayout(self._flow)

        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidget(self._content)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._scroll)
        self.setStyleSheet(
            "QFrame#imageTagsPanel { border: none; background: transparent; }"
        )
        self._render()

    @property
    def tags(self):
        return list(self._tags)

    @property
    def mode(self):
        return self._mode

    def set_tags(self, tags):
        self.cancel_active_mode()
        self._tags = list(tags)
        self._render()

    def refresh_colors(self):
        for chip, tag in zip(self._chips, self._tags):
            if chip is None:
                continue
            chip._color = self._color_for_tag(tag)
            chip._update_style()

    def set_interactions_enabled(self, enabled):
        self._interactions_enabled = enabled
        self._content.setEnabled(enabled)

    def _render(self):
        self._flow.clear()
        self._chips = []
        self._copy_all_button = None
        self._select_all_button = None
        self._delete_selected_button = None
        self._cancel_button = None
        for index, tag in enumerate(self._tags):
            if self._mode == "edit" and index == self._editing_index:
                self._chips.append(None)
                continue
            chip = ImageTagChip(
                index, tag, self._color_for_tag(tag), self._content
            )
            chip.edit_requested.connect(self.start_edit)
            chip.delete_requested.connect(self.delete_tag)
            chip.selection_toggled.connect(self.toggle_selection)
            chip.drag_requested.connect(self.start_drag)
            chip.status_message.connect(self.status_message.emit)
            chip.set_batch_mode(self._mode == "batch", index in self._selected)
            self._flow.addWidget(chip)
            self._chips.append(chip)

        if self._mode == "add":
            self._input = self._create_input(self._next_default_tag())
            self._flow.addWidget(self._input)
            editor = self._input
            QtCore.QTimer.singleShot(
                0, lambda: self._focus_editor(editor, select_all=True)
            )
        elif self._mode == "edit" and self._editing_index is not None:
            self._input = self._create_input(self._tags[self._editing_index])
            self._flow.insertWidget(self._editing_index, self._input)
            editor = self._input
            QtCore.QTimer.singleShot(
                0, lambda: self._focus_editor(editor, select_all=False)
            )
        elif self._mode == "batch":
            self._input = None
            self._add_batch_action_buttons()
        else:
            self._input = None
            add_button = self._create_round_action_button(
                "+", "addImageTagButton", self.tr("Add image tag")
            )
            add_button.clicked.connect(self.start_add)
            self._flow.addWidget(add_button)
            if self._tags:
                batch_button = self._create_round_action_button(
                    "−",
                    "batchDeleteImageTagsButton",
                    self.tr("Batch Delete Tags"),
                )
                batch_button.clicked.connect(self.start_batch_mode)
                self._flow.addWidget(batch_button)
                self._copy_all_button = self._create_round_action_button(
                    "C",
                    "copyAllImageTagsButton",
                    self.tr("Copy All Tags"),
                )
                self._copy_all_button.setStyleSheet(
                    self._copy_all_button.styleSheet()
                    + "QPushButton#copyAllImageTagsButton {"
                    "font-size: 14px; font-weight: bold; }"
                )
                self._copy_all_button.clicked.connect(self.copy_all_tags)
                self._flow.addWidget(self._copy_all_button)

        if self._mode == "batch":
            self._update_batch_controls()
        QtCore.QTimer.singleShot(0, self._refresh_height)

    def _add_batch_action_buttons(self):
        self._select_all_button = self._create_batch_action_button(
            self.tr("Select All"), "imageTagsSelectAllButton"
        )
        self._delete_selected_button = self._create_batch_action_button(
            self.tr("Delete"), "imageTagsDeleteSelectedButton"
        )
        self._cancel_button = self._create_batch_action_button(
            self.tr("Cancel"), "imageTagsCancelButton"
        )
        select_all_width = max(
            self._select_all_button.fontMetrics().horizontalAdvance(text)
            for text in (self.tr("Select All"), self.tr("Deselect All"))
        )
        self._select_all_button.setFixedWidth(select_all_width + 32)
        self._delete_selected_button.setStyleSheet(
            self._delete_selected_button.styleSheet()
            + "QPushButton#imageTagsDeleteSelectedButton:enabled {"
            "background-color: rgb(210, 66, 66); color: white; }"
            "QPushButton#imageTagsDeleteSelectedButton:enabled:hover {"
            "background-color: rgb(190, 48, 48); }"
            "QPushButton#imageTagsDeleteSelectedButton:disabled {"
            "background-color: rgba(128, 128, 128, 35); "
            "color: palette(placeholder-text); }"
        )
        self._select_all_button.clicked.connect(self.toggle_select_all)
        self._delete_selected_button.clicked.connect(self.delete_selected)
        self._cancel_button.clicked.connect(self.cancel_batch_mode)
        for button in (
            self._select_all_button,
            self._delete_selected_button,
            self._cancel_button,
        ):
            self._flow.addWidget(button)

    def _create_batch_action_button(self, text, object_name):
        button = QtWidgets.QPushButton(text, self._content)
        button.setObjectName(object_name)
        button.setFixedHeight(IMAGE_TAG_HEIGHT)
        button.setCursor(Qt.CursorShape.PointingHandCursor)

        selector = f"QPushButton#{object_name}"
        button.setStyleSheet(
            f"{selector} {{"
            "background-color: rgba(128, 128, 128, 35); "
            "color: palette(text); "
            f"border: none; border-radius: {IMAGE_TAG_RADIUS}px; "
            f"padding: 0 14px; min-height: {IMAGE_TAG_HEIGHT}px; "
            f"max-height: {IMAGE_TAG_HEIGHT}px; }}"
            f"{selector}:hover {{"
            "background-color: rgba(128, 128, 128, 65); }"
            f"{selector}:pressed {{"
            "background-color: rgba(128, 128, 128, 90); }"
        )
        return button

    def _create_round_action_button(self, text, object_name, tooltip):
        button = QtWidgets.QPushButton(text, self._content)
        button.setObjectName(object_name)
        button.setFixedSize(IMAGE_TAG_HEIGHT, IMAGE_TAG_HEIGHT)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)

        selector = (
            "QPushButton#addImageTagButton, "
            "QPushButton#copyAllImageTagsButton, "
            "QPushButton#batchDeleteImageTagsButton"
        )
        hover_selector = (
            "QPushButton#addImageTagButton:hover, "
            "QPushButton#copyAllImageTagsButton:hover, "
            "QPushButton#batchDeleteImageTagsButton:hover"
        )
        pressed_selector = (
            "QPushButton#addImageTagButton:pressed, "
            "QPushButton#copyAllImageTagsButton:pressed, "
            "QPushButton#batchDeleteImageTagsButton:pressed"
        )

        button.setStyleSheet(
            f"{selector} {{"
            "background-color: rgba(128, 128, 128, 35); "
            "color: palette(text); "
            f"border: none; border-radius: {IMAGE_TAG_RADIUS}px; "
            "font-size: 18px; font-weight: bold; padding: 0; "
            f"min-width: {IMAGE_TAG_HEIGHT}px; "
            f"max-width: {IMAGE_TAG_HEIGHT}px; "
            f"min-height: {IMAGE_TAG_HEIGHT}px; "
            f"max-height: {IMAGE_TAG_HEIGHT}px; }}"
            f"{hover_selector} {{"
            "background-color: rgba(128, 128, 128, 65); }"
            f"{pressed_selector} {{"
            "background-color: rgba(128, 128, 128, 90); }"
        )

        return button

    def copy_all_tags(self):
        if (
            not self._interactions_enabled
            or self._mode != "normal"
            or not self._tags
        ):
            return
        parent = self.window()
        popup = Popup(
            self.tr("Copy Successful"),
            parent=parent,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(
            parent,
            copy_msg=",".join(self._tags),
            position="default",
        )

    def _create_input(self, text=""):
        editor = _TagLineEdit(self._content)
        editor.setObjectName("imageTagEditor")
        self._input = editor
        editor.setText(text)
        editor.setFixedHeight(IMAGE_TAG_HEIGHT)
        editor.setAccessibleName(self.tr("Image tag editor"))
        editor.textEdited.connect(self._show_input_hint)
        if self._mode == "add":
            editor.returnPressed.connect(self.commit_add)
            editor.focus_lost.connect(self.commit_add)
        else:
            editor.returnPressed.connect(self.commit_edit)
            editor.focus_lost.connect(self.commit_edit)
        editor.escape_pressed.connect(self.cancel_input)
        self._set_input_appearance(editor, self._color_for_tag(text))
        return editor

    def _next_default_tag(self):
        existing = set(self._tags)
        if "tag" not in existing:
            return "tag"
        suffix = 1
        while f"tag{suffix}" in existing:
            suffix += 1
        return f"tag{suffix}"

    def _set_input_appearance(self, editor, rgb):
        color = QtGui.QColor(*rgb)
        luminance = (
            0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
        )
        foreground = "#111111" if luminance > 160 else "#ffffff"
        editor.setStyleSheet(
            "QLineEdit#imageTagEditor {"
            f"background-color: {color.name()}; color: {foreground}; "
            f"border: none; border-radius: {IMAGE_TAG_RADIUS}px; "
            f"padding: 0 9px; min-height: {IMAGE_TAG_HEIGHT}px; "
            f"max-height: {IMAGE_TAG_HEIGHT}px; "
            f"selection-background-color: {foreground}; "
            f"selection-color: {color.name()}; }}"
        )
        metrics = QtGui.QFontMetrics(editor.font())
        editor.setFixedWidth(
            max(54, min(260, metrics.horizontalAdvance(editor.text()) + 18))
        )

    def _show_input_hint(self):
        if self._input:
            self._input.setToolTip("")
        if self._mode == "add":
            message = self.tr("Press Enter to add, or Esc to cancel.")
        elif self._mode == "edit":
            message = self.tr("Press Enter to save, or Esc to cancel.")
        else:
            return
        self.status_message.emit(message)

    def _focus_editor(self, editor, select_all):
        if editor is not self._input:
            return
        editor.setFocus()
        if select_all:
            editor.selectAll()
        else:
            editor.setCursorPosition(len(editor.text()))

    def _refresh_height(self):
        width = max(80, self._scroll.viewport().width())
        content_height = self._flow.heightForWidth(width)
        margins = self._flow.contentsMargins()
        vertical_margins = margins.top() + margins.bottom()
        minimum_height = IMAGE_TAG_HEIGHT + vertical_margins
        content_height = max(minimum_height, content_height)
        self._content.setMinimumHeight(content_height)
        maximum_height = (
            IMAGE_TAG_MAX_ROWS * IMAGE_TAG_HEIGHT
            + (IMAGE_TAG_MAX_ROWS - 1) * IMAGE_TAG_SPACING
            + vertical_margins
        )
        self._scroll.setFixedHeight(min(content_height, maximum_height))
        self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_height()

    def start_add(self):
        if not self._interactions_enabled or self._mode != "normal":
            return
        self._mode = "add"
        self._render()
        self._show_input_hint()

    def commit_add(self):
        if self._mode != "add" or not self._input:
            return False
        text = self._input.text()
        if not text.strip():
            return self.cancel_input()
        tag = self._validate_input(text, None)
        if tag is None:
            return False
        self._tags.append(tag)
        self._mode = "normal"
        self._input = None
        self._render()
        self.status_message.emit("")
        self.tags_changed.emit(self.tags)
        return True

    def start_edit(self, index):
        if not self._interactions_enabled or self._mode != "normal":
            return
        self._mode = "edit"
        self._editing_index = index
        self._render()
        self._show_input_hint()

    def commit_edit(self):
        if self._mode != "edit" or not self._input:
            return False
        index = self._editing_index
        text = self._input.text()
        if not text.strip():
            self._tags.pop(index)
            self._mode = "normal"
            self._editing_index = None
            self._input = None
            self._render()
            self.status_message.emit("")
            self.tags_changed.emit(self.tags)
            return True
        tag = self._validate_input(text, index)
        if tag is None:
            return False
        changed = tag != self._tags[index]
        if changed:
            self._tags[index] = tag
        self._mode = "normal"
        self._editing_index = None
        self._input = None
        self._render()
        self.status_message.emit("")
        if changed:
            self.tags_changed.emit(self.tags)
        return True

    def _validate_input(self, text, current_index):
        tag = normalize_image_tag(text)
        if tag is None:
            self._show_input_error(
                self.tr("Tags cannot be empty or contain line breaks.")
            )
            return None
        for index, existing in enumerate(self._tags):
            if existing == tag and index != current_index:
                self._show_input_error(self.tr("This tag already exists."))
                return None
        return tag

    def _show_input_error(self, message):
        if self._input:
            self._input.setToolTip(message)
        self.status_message.emit(message)

    def cancel_input(self):
        if self._mode not in ("add", "edit"):
            return False
        self._mode = "normal"
        self._editing_index = None
        self._input = None
        self._render()
        self.status_message.emit("")
        return True

    def delete_tag(self, index):
        if self._mode != "normal" or not self._confirm_single_delete(index):
            return
        self._tags.pop(index)
        self._render()
        self.tags_changed.emit(self.tags)

    def _confirm_single_delete(self, index):
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setWindowTitle(self.tr("Delete Image Tag"))
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(self.tr('Delete tag "%s"?') % self._tags[index])
        box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        return box.exec() == QtWidgets.QMessageBox.StandardButton.Yes

    def start_batch_mode(self):
        if not self._interactions_enabled or not self._tags:
            return
        self._mode = "batch"
        self._selected.clear()
        self._render()

    def toggle_selection(self, index):
        if self._mode != "batch":
            return
        if index in self._selected:
            self._selected.remove(index)
        else:
            self._selected.add(index)
        self._chips[index].set_batch_mode(True, index in self._selected)
        self._update_batch_controls()

    def toggle_select_all(self):
        if len(self._selected) == len(self._tags):
            self._selected.clear()
        else:
            self._selected = set(range(len(self._tags)))
        for index, chip in enumerate(self._chips):
            chip.set_batch_mode(True, index in self._selected)
        self._update_batch_controls()

    def _update_batch_controls(self):
        count = len(self._selected)
        self._delete_selected_button.setEnabled(count > 0)
        all_selected = bool(self._tags) and count == len(self._tags)
        self._select_all_button.setText(
            self.tr("Deselect All") if all_selected else self.tr("Select All")
        )

    def delete_selected(self):
        if self._mode != "batch" or not self._selected:
            return
        if not self._confirm_batch_delete():
            return
        selected = self._selected
        self._tags = [
            tag
            for index, tag in enumerate(self._tags)
            if index not in selected
        ]
        self._mode = "normal"
        self._selected = set()
        self._render()
        self.tags_changed.emit(self.tags)

    def _confirm_batch_delete(self):
        result = QtWidgets.QMessageBox.question(
            self,
            self.tr("Delete Image Tags"),
            self.tr("Delete %d selected tags?") % len(self._selected),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        return result == QtWidgets.QMessageBox.StandardButton.Yes

    def cancel_batch_mode(self):
        if self._mode != "batch":
            return False
        self._mode = "normal"
        self._selected.clear()
        self._render()
        return True

    def cancel_active_mode(self):
        if self._mode in ("add", "edit"):
            return self.cancel_input()
        if self._mode == "batch":
            return self.cancel_batch_mode()
        self.clear_drag_target()
        return False

    def finish_for_image_change(self):
        if self._mode == "add":
            if not self.commit_add():
                self.cancel_input()
        elif self._mode == "edit":
            if not self.commit_edit():
                self.cancel_input()
        elif self._mode == "batch":
            self.cancel_batch_mode()

    def start_drag(self, index, chip):
        if self._mode != "normal":
            return
        drag = QtGui.QDrag(chip)
        mime = QtCore.QMimeData()
        mime.setData(IMAGE_TAG_MIME_TYPE, str(index).encode("ascii"))
        drag.setMimeData(mime)
        source = chip.grab()
        preview = QtGui.QPixmap(source.size())
        preview.fill(Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(preview)
        painter.setOpacity(0.65)
        painter.drawPixmap(0, 0, source)
        painter.end()
        drag.setPixmap(preview)
        drag.setHotSpot(
            QtCore.QPoint(preview.width() // 2, preview.height() // 2)
        )
        drag.exec(Qt.DropAction.MoveAction)
        self.clear_drag_target()

    def _target_index(self, position):
        for index, chip in enumerate(self._chips):
            geometry = chip.geometry()
            center = geometry.center()
            if position.y() < center.y():
                return index
            same_row = geometry.top() <= position.y() <= geometry.bottom()
            if same_row and position.x() < center.x():
                return index
        return len(self._chips)

    def update_drag_target(self, position):
        self._drop_index = self._target_index(position)
        for index, chip in enumerate(self._chips):
            drop_position = None
            if index == self._drop_index:
                drop_position = "left"
            elif index == len(self._chips) - 1 and self._drop_index == len(
                self._chips
            ):
                drop_position = "right"
            chip.set_drop_position(drop_position)
        viewport_position = self._content.mapTo(
            self._scroll.viewport(), position
        )
        bar = self._scroll.verticalScrollBar()
        if viewport_position.y() < 20:
            bar.setValue(bar.value() - 18)
        elif viewport_position.y() > self._scroll.viewport().height() - 20:
            bar.setValue(bar.value() + 18)

    def clear_drag_target(self):
        self._drop_index = None
        for chip in self._chips:
            chip.set_drop_position()

    def finish_drag(self, source, position):
        target = self._target_index(position)
        if source < target:
            target -= 1
        self.clear_drag_target()
        if source == target:
            return
        tag = self._tags.pop(source)
        self._tags.insert(target, tag)
        self._render()
        self.tags_changed.emit(self.tags)
