from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_ok_btn_style,
)

from .icons import apply_button_icon, theme_icon_color
from .style import (
    get_icon_button_style,
    get_label_settings_dialog_style,
    get_segment_list_style,
)
from .utils import color_for_label, color_from_index


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


class LabelColorButton(QPushButton):
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(28, 28)
        self.setFlat(True)
        self._refresh_icon()

    def color(self):
        return self._color.name()

    def set_color(self, color):
        self._color = QColor(color)
        self._refresh_icon()

    def _refresh_icon(self):
        self.setIcon(self._make_icon())
        self.setIconSize(self.size())

    def _make_icon(self):
        from PyQt6.QtGui import QIcon

        return QIcon(_color_pixmap(self._color, size=18))


class LabelNameDialog(QDialog):
    def __init__(self, title, text="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedWidth(360)
        self.setStyleSheet(get_label_settings_dialog_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        self.name_edit = QLineEdit(text)
        self.name_edit.setFixedHeight(32)
        self.name_edit.selectAll()
        layout.addWidget(self.name_edit)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)
        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.setObjectName("XvaDialogSecondaryButton")
        cancel_btn.setStyleSheet(get_cancel_btn_style())
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton(self.tr("OK"))
        ok_btn.setObjectName("XvaDialogPrimaryButton")
        ok_btn.setStyleSheet(get_ok_btn_style())
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)

        self.name_edit.returnPressed.connect(self.accept)
        self.name_edit.setFocus()

    def text(self):
        return self.name_edit.text()


class LabelSettingsDialog(QDialog):
    def __init__(self, labels, colors, parent=None):
        super().__init__(parent)
        self._labels = list(labels or [])
        self._colors = dict(colors or {})
        self._changes = None

        self.setWindowTitle(self.tr("Label Settings"))
        self.setModal(True)
        self.resize(560, 320)
        self.setStyleSheet(get_label_settings_dialog_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.table = QTableWidget()
        self.table.setObjectName("XvaLabelSettingsTable")
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("#"),
                self.tr("Category"),
                self.tr("Delete"),
                self.tr("Name"),
                self.tr("Color"),
            ]
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(False)
        self.table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.horizontalHeader().setFixedHeight(34)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Fixed
        )
        self.table.setColumnWidth(1, 120)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        layout.addWidget(self.table, 1)

        for name in sorted(self._labels, key=str.casefold):
            self._append_row(name, name, self._colors.get(name))

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        upload_btn = QPushButton(self.tr("Upload"))
        upload_btn.setObjectName("XvaDialogSecondaryButton")
        upload_btn.setStyleSheet(get_cancel_btn_style())
        upload_btn.clicked.connect(self._upload_labels)
        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.setObjectName("XvaDialogSecondaryButton")
        cancel_btn.setStyleSheet(get_cancel_btn_style())
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton(self.tr("OK"))
        ok_btn.setObjectName("XvaDialogPrimaryButton")
        ok_btn.setStyleSheet(get_ok_btn_style())
        ok_btn.clicked.connect(self._accept_changes)
        buttons.addWidget(upload_btn)
        buttons.addStretch(1)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(ok_btn)
        layout.addLayout(buttons)

    def changes(self):
        return self._changes

    def _append_row(self, original, name, color, deleted=False):
        row = self.table.rowCount()
        self.table.insertRow(row)

        index_item = QTableWidgetItem(str(row + 1))
        index_item.setFlags(index_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
        index_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 0, index_item)

        original_item = QTableWidgetItem(original or self.tr("(new label)"))
        original_item.setData(Qt.ItemDataRole.UserRole, original)
        original_item.setFlags(
            original_item.flags() ^ Qt.ItemFlag.ItemIsEditable
        )
        self.table.setItem(row, 1, original_item)

        delete_checkbox = QCheckBox()
        delete_checkbox.setEnabled(bool(original))
        delete_checkbox.setChecked(deleted)
        delete_checkbox.stateChanged.connect(
            lambda _state, r=row: self._on_delete_changed(r)
        )
        delete_container = QWidget()
        delete_layout = QHBoxLayout(delete_container)
        delete_layout.setContentsMargins(0, 0, 0, 0)
        delete_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        delete_layout.addWidget(delete_checkbox)
        self.table.setCellWidget(row, 2, delete_container)

        name_edit = QLineEdit("" if original else name)
        name_edit.setFixedHeight(28)
        name_edit.setEnabled(not deleted)
        self.table.setCellWidget(row, 3, name_edit)

        color_button = LabelColorButton(
            color or color_for_label(original or name), self
        )
        color_button.setObjectName("XvaColorButton")
        color_button.clicked.connect(
            lambda _checked=False, btn=color_button: self._pick_color(btn)
        )
        color_container = QWidget()
        color_layout = QHBoxLayout(color_container)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        color_layout.addWidget(color_button)
        self.table.setCellWidget(row, 4, color_container)

    def _upload_labels(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Upload labels"),
            "",
            self.tr("Text files (*.txt);;All files (*)"),
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                labels = [line.strip() for line in f]
        except OSError as exc:
            QMessageBox.warning(
                self,
                self.tr("Upload labels"),
                self.tr("Failed to read labels file: {error}").format(
                    error=exc
                ),
            )
            return
        existing = {name for name in self._table_names() if name}
        added = 0
        for name in labels:
            if not name or name in existing:
                continue
            existing.add(name)
            self._append_row("", name, color_from_index(self.table.rowCount()))
            added += 1
        if added:
            self._sort_rows()
            return
        QMessageBox.information(
            self,
            self.tr("Upload labels"),
            self.tr("No new labels found."),
        )

    def _table_names(self):
        names = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            original = item.data(Qt.ItemDataRole.UserRole) if item else ""
            if original:
                names.append(original)
            name_edit = self._name_edit(row)
            names.append((name_edit.text() if name_edit else "").strip())
        return names

    def _sort_rows(self):
        rows = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            original = item.data(Qt.ItemDataRole.UserRole) if item else ""
            checkbox = self._delete_checkbox(row)
            name_edit = self._name_edit(row)
            color_button = self._color_button(row)
            name = (name_edit.text() if name_edit else "").strip()
            sort_name = name or original
            rows.append(
                {
                    "original": original,
                    "name": name,
                    "color": (
                        color_button.color()
                        if color_button
                        else color_for_label(sort_name)
                    ),
                    "sort_name": sort_name,
                    "deleted": bool(checkbox and checkbox.isChecked()),
                }
            )
        self.table.setRowCount(0)
        rows.sort(key=lambda row: row["sort_name"].casefold())
        for row in rows:
            self._append_row(
                row["original"],
                row["name"],
                row["color"],
                row["deleted"],
            )

    def _on_delete_changed(self, row):
        checkbox = self._delete_checkbox(row)
        name_edit = self._name_edit(row)
        if checkbox and name_edit:
            name_edit.setEnabled(not checkbox.isChecked())

    def _pick_color(self, button):
        color = QColorDialog.getColor(
            QColor(button.color()), self, self.tr("Pick color")
        )
        if color.isValid():
            button.set_color(color)

    def _delete_checkbox(self, row):
        container = self.table.cellWidget(row, 2)
        return container.findChild(QCheckBox) if container else None

    def _name_edit(self, row):
        return self.table.cellWidget(row, 3)

    def _color_button(self, row):
        container = self.table.cellWidget(row, 4)
        return container.findChild(LabelColorButton) if container else None

    def _accept_changes(self):
        changes = self._collect_changes()
        if changes is None:
            return
        if changes["deleted"]:
            names = ", ".join(changes["deleted"])
            choice = QMessageBox.warning(
                self,
                self.tr("Delete labels"),
                self.tr(
                    "Deleting these labels will also remove all associated "
                    "segments: {names}\n\nContinue?"
                ).format(names=names),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice != QMessageBox.StandardButton.Yes:
                return
        self._changes = changes
        self.accept()

    def _collect_changes(self):
        rows = []
        final_names = []
        deleted = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            original = item.data(Qt.ItemDataRole.UserRole) if item else ""
            checkbox = self._delete_checkbox(row)
            name_edit = self._name_edit(row)
            color_button = self._color_button(row)
            is_deleted = bool(checkbox and checkbox.isChecked())
            name = (name_edit.text() if name_edit else "").strip()
            final_name = name or original
            if is_deleted:
                if original:
                    deleted.append(original)
                continue
            if not final_name:
                QMessageBox.information(
                    self,
                    self.tr("Label Settings"),
                    self.tr("Label name cannot be empty."),
                )
                return None
            final_names.append(final_name)
            rows.append(
                {
                    "original": original,
                    "name": final_name,
                    "color": (
                        color_button.color()
                        if color_button
                        else color_for_label(final_name)
                    ),
                }
            )
        if len(final_names) != len(set(final_names)):
            QMessageBox.information(
                self,
                self.tr("Label Settings"),
                self.tr("Label names must be unique."),
            )
            return None
        rows.sort(key=lambda row: row["name"].casefold())
        return {"rows": rows, "deleted": deleted}


class LabelPanel(QWidget):
    """Right-side palette: add / rename / delete / pick-color / hotkey-select."""

    labelSelected = pyqtSignal(str)  # name (or "" if none)
    labelsChanged = pyqtSignal()  # whenever list mutates
    labelRenamed = pyqtSignal(str, str)  # (old, new)
    labelRemoved = pyqtSignal(str)  # name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels = []  # ordered names
        self._colors = {}  # name → hex

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        head = QGridLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setColumnMinimumWidth(0, 64)
        head.setColumnStretch(1, 1)
        head.setColumnMinimumWidth(2, 64)
        title = QLabel(self.tr("Labels"))
        title.setObjectName("XvaPanelTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head.addWidget(title, 0, 1)
        actions = QWidget()
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(2)
        self.new_btn = QPushButton()
        self.new_btn.setStyleSheet(get_icon_button_style())
        self.new_btn.setFixedSize(24, 24)
        self.new_btn.setToolTip(self.tr("Add label"))
        apply_button_icon(
            self.new_btn, "new", "svg", 18, theme_icon_color("text")
        )
        self.new_btn.clicked.connect(self._on_add_clicked)
        self.settings_btn = QPushButton()
        self.settings_btn.setStyleSheet(get_icon_button_style())
        self.settings_btn.setFixedSize(24, 24)
        self.settings_btn.setToolTip(self.tr("Manage labels"))
        apply_button_icon(
            self.settings_btn, "settings", "svg", 18, theme_icon_color("text")
        )
        self.settings_btn.clicked.connect(self._on_settings_clicked)
        actions_layout.addWidget(self.new_btn)
        actions_layout.addWidget(self.settings_btn)
        head.addWidget(actions, 0, 2, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(head)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(get_segment_list_style())
        self.list_widget.setIconSize(QSize(12, 12))
        self.list_widget.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.list_widget.setSelectionMode(
            self.list_widget.SelectionMode.SingleSelection
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

    # public API
    def set_labels(self, labels, colors=None):
        self._labels = list(labels or [])
        self._colors = dict(colors or {})
        self._sort_labels()
        self._ensure_unique_colors()
        self._refresh()

    def labels(self):
        return list(self._labels)

    def colors(self):
        return dict(self._colors)

    def color_for(self, name):
        return self._colors.get(name) or color_for_label(name)

    def active_label(self):
        item = self.list_widget.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def select_index(self, idx):
        if 0 <= idx < self.list_widget.count():
            self.list_widget.setCurrentRow(idx)

    def select_label(self, name):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == name:
                self.list_widget.setCurrentRow(i)
                return

    def add_label(self, name, color=None):
        name = (name or "").strip()
        if not name:
            return False
        if name in self._labels:
            self.select_label(name)
            return False
        self._labels.append(name)
        self._sort_labels()
        self._colors[name] = color or self._next_available_color()
        self._ensure_unique_colors()
        self._refresh()
        self.select_label(name)
        self.labelsChanged.emit()
        return True

    def rename_label(self, old, new):
        new = (new or "").strip()
        if not new or old == new or old not in self._labels:
            return False
        if new in self._labels:
            return False
        idx = self._labels.index(old)
        self._labels[idx] = new
        color = self._colors.pop(old, color_for_label(new))
        self._colors[new] = color
        self._sort_labels()
        self._ensure_unique_colors()
        self._refresh()
        self.select_label(new)
        self.labelRenamed.emit(old, new)
        self.labelsChanged.emit()
        return True

    def remove_label(self, name):
        if name not in self._labels:
            return False
        self._labels.remove(name)
        self._colors.pop(name, None)
        self._refresh()
        self.labelRemoved.emit(name)
        self.labelsChanged.emit()
        return True

    def set_color(self, name, color):
        if name not in self._labels:
            return
        self._colors[name] = color
        self._ensure_unique_colors()
        self._refresh()
        self.labelsChanged.emit()

    def apply_settings(self, changes):
        if not changes:
            return
        deleted = list(changes["deleted"])
        renamed = [
            (row["original"], row["name"])
            for row in changes["rows"]
            if row["original"] and row["original"] != row["name"]
        ]
        current = self.active_label()
        rename_map = dict(renamed)
        self._labels = [row["name"] for row in changes["rows"]]
        self._colors = {
            row["name"]: row["color"] or color_for_label(row["name"])
            for row in changes["rows"]
        }
        self._sort_labels()
        self._ensure_unique_colors()
        self._refresh()
        if current and current not in deleted:
            self.select_label(rename_map.get(current, current))
        for name in deleted:
            self.labelRemoved.emit(name)
        for old, new in renamed:
            self.labelRenamed.emit(old, new)
        self.labelsChanged.emit()

    # internals
    def _refresh(self):
        current = self.active_label()
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for index, name in enumerate(self._labels):
            color = self._colors.get(name) or color_for_label(name)
            display_name = f"{name}({index})" if index <= 9 else name
            item = QListWidgetItem(display_name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setIcon(self._make_icon(color))
            self.list_widget.addItem(item)
        # restore selection
        self.list_widget.blockSignals(False)
        if current:
            self.select_label(current)
        if not self.active_label() and self.list_widget.count():
            self.list_widget.setCurrentRow(0)

    def _make_icon(self, color):
        from PyQt6.QtGui import QIcon

        return QIcon(_color_pixmap(color, size=14))

    def _ensure_unique_colors(self):
        used = set()
        for index, name in enumerate(self._labels):
            qcolor = QColor(self._colors.get(name) or "")
            color = qcolor.name() if qcolor.isValid() else ""
            if not color or color in used:
                color = self._color_for_index(index, used)
            self._colors[name] = color
            used.add(color)

    def _next_available_color(self):
        return self._color_for_index(
            len(self._labels), set(self._colors.values())
        )

    def _color_for_index(self, index, used):
        color = color_from_index(index)
        step = 0
        while color in used:
            step += 1
            color = color_from_index(index + step)
        return color

    def _sort_labels(self):
        self._labels.sort(key=str.casefold)

    # slots
    def _on_selection_changed(self):
        self.labelSelected.emit(self.active_label())

    def _on_double_clicked(self, item):
        old = item.data(Qt.ItemDataRole.UserRole)
        self._rename_via_dialog(old)

    def _on_add_clicked(self):
        dialog = LabelNameDialog(self.tr("Add label"), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        text = dialog.text()
        if not self.add_label(text):
            if text and text.strip() in self._labels:
                QMessageBox.information(
                    self,
                    self.tr("Add label"),
                    self.tr("Label already exists."),
                )

    def _on_settings_clicked(self):
        dialog = LabelSettingsDialog(self._labels, self._colors, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_settings(dialog.changes())

    def _rename_via_dialog(self, old):
        dialog = LabelNameDialog(
            self.tr("Rename label"),
            text=old,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        text = dialog.text()
        new = (text or "").strip()
        if not new or new == old:
            return
        if new in self._labels:
            QMessageBox.information(
                self,
                self.tr("Rename label"),
                self.tr("Another label already uses that name."),
            )
            return
        self.rename_label(old, new)

    def _show_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        rename = menu.addAction(self.tr("Rename…"))
        change_color = menu.addAction(self.tr("Change color…"))
        menu.addSeparator()
        remove = menu.addAction(self.tr("Delete"))
        chosen = menu.exec(self.list_widget.viewport().mapToGlobal(pos))
        if chosen == rename:
            self._rename_via_dialog(name)
        elif chosen == change_color:
            c = QColorDialog.getColor(
                QColor(self.color_for(name)), self, self.tr("Pick color")
            )
            if c.isValid():
                self.set_color(name, c.name())
        elif chosen == remove:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle(self.tr("Delete label"))
            box.setText(
                self.tr(
                    "Deleting label '{name}' will also remove all associated "
                    "segments.\n\nContinue?"
                ).format(name=name)
            )
            box.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            box.setDefaultButton(QMessageBox.StandardButton.No)
            if box.exec() == QMessageBox.StandardButton.Yes:
                self.remove_label(name)

    # keyboard select via hotkey
    def select_by_digit(self, digit):
        if 0 <= digit <= 9 and digit < self.list_widget.count():
            self.list_widget.setCurrentRow(digit)
            return True
        return False
