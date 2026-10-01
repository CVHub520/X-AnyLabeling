from dataclasses import replace

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .controls import PointCloudListWidget
from .cuboid import Cuboid, MIN_SIZE
from .cuboid_viewport import CuboidViewport
from .icons import get_icon


class DetectionWorkspace(QtWidgets.QSplitter):
    def __init__(self, window, canvas):
        super().__init__(QtCore.Qt.Orientation.Vertical)
        self.window = window
        self.enabled = False
        self.selected_id = None
        self.hidden_ids = set()
        self.clipboard = None
        self._document = None
        self._syncing = False
        self._expanded_view = None
        self._expanded_sizes = None
        self.addWidget(window.viewport)
        self.orthographic = QtWidgets.QSplitter()
        self.views = [window.viewport]
        for name in ("top", "side", "front"):
            view = CuboidViewport(view=name)
            view.setMinimumSize(160, 140)
            self.orthographic.addWidget(view)
            self.views.append(view)
            view.expand_requested.connect(
                lambda view=view: self.toggle_expanded_view(view)
            )
        self.addWidget(self.orthographic)
        self.setChildrenCollapsible(False)
        self.orthographic.setChildrenCollapsible(False)
        self.setStretchFactor(0, 1)
        self.setStretchFactor(1, 0)
        self.setSizes([600, 0])
        self.orthographic.hide()
        self._view_sizes = None
        self._views_visible = True
        self.shortcuts = []
        self.views_button = QtWidgets.QToolButton(canvas)
        self.views_button.setObjectName("pointcloudIconButton")
        self.views_button.setProperty("panelHeader", True)
        self.views_button.setFixedSize(26, 26)
        self.views_button.setIconSize(QtCore.QSize(18, 18))
        self.views_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.views_button.clicked.connect(self.toggle_views)
        self.panel = self._build_panel()
        for view in self.views:
            view.cuboid_selected.connect(self.select)
            view.cuboid_preview.connect(
                lambda box, source=view: self.preview(box, source)
            )
            view.cuboid_edited.connect(self.commit)
            view.cuboid_created.connect(self.create)
            view.cancel_requested.connect(self.cancel)
            if view is not window.viewport:
                view.status_message.connect(window._status)
                for action in window.viewport.camera_actions.values():
                    view.addAction(action)
            for title, key, callback in (
                ("Draw cuboid", "N", self.start_creation),
                ("Delete cuboid", "Delete", self.delete),
                ("Duplicate cuboid", "Ctrl+D", self.duplicate),
                ("Copy cuboid", "Ctrl+C", self.copy),
                ("Paste cuboid", "Ctrl+V", self.paste),
                ("Focus cuboid", "G", self.focus),
            ):
                action = QtGui.QAction(self.tr(title), view)
                action.setShortcut(key)
                action.setShortcutContext(
                    QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
                )
                action.triggered.connect(
                    lambda checked=False, cb=callback: (
                        cb() if self.enabled else None
                    )
                )
                view.addAction(action)
                self.shortcuts.append(action)
        for action in (window.undo_action, window.redo_action):
            for view in self.views[1:]:
                view.addAction(action)

    @property
    def document(self):
        return self.window.document

    @property
    def selected(self):
        if self.document is None:
            return None
        return next(
            (
                box
                for box in self.document.cuboids
                if box.id == self.selected_id
            ),
            None,
        )

    @property
    def active(self):
        return any(
            view.selection_active or view.creating for view in self.views
        )

    def _build_panel(self):
        scroll, layout = self.window._scroll_page()
        self.labels, self.labels_heading, header = self._list_panel(
            layout, "Labels", "", allow_remove=False
        )
        self.classes_lock_action = self._header_action(
            header, "Lock all classes", self._lock_all, "unlock"
        )
        self.classes_visibility_action = self._header_action(
            header, "Toggle all classes", self._toggle_all_visibility, "eye"
        )
        self.labels.currentItemChanged.connect(self._label_selected)
        self.labels.itemChanged.connect(self._label_visibility_changed)
        self.labels.lock_requested.connect(self._lock_label)
        self.objects, self.objects_heading, header = self._list_panel(
            layout, "Objects", "Delete object"
        )
        self.lock_all_action = self._header_action(
            header, "Lock all objects", self._lock_all, "unlock"
        )
        self.visibility_action = self._header_action(
            header, "Toggle all objects", self._toggle_all_visibility, "eye"
        )
        self.objects.currentItemChanged.connect(self._select_item)
        self.objects.itemChanged.connect(self._visibility_changed)
        self.objects.itemDoubleClicked.connect(self.focus)
        self.objects.lock_requested.connect(self._lock_object)
        self.objects.remove_requested.connect(self._delete_object)
        self.objects.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.objects.customContextMenuRequested.connect(self._object_menu)
        self.draw_action = QtGui.QAction(self.tr("Draw cuboid (N)"), self)
        self.draw_action.setIcon(self.window._icon("draw-cuboid"))
        self.draw_action.setCheckable(True)
        self.draw_action.triggered.connect(self.start_creation)
        self.instance_action = QtGui.QAction(
            self.tr("Create box from current instance"), self
        )
        self.instance_action.triggered.connect(self.from_instance)
        self.edit_actions = []
        for title, callback in (
            ("Edit object", self.edit_properties),
            ("Focus (G)", self.focus),
            ("Fit to points", self.fit),
            ("Duplicate", self.duplicate),
            ("Delete", self.delete),
            ("Copy", self.copy),
            ("Paste", self.paste),
        ):
            action = QtGui.QAction(self.tr(title), self)
            action.triggered.connect(callback)
            self.edit_actions.append(action)
        self.crop = QtGui.QAction(
            self.tr("Crop depth to selected cuboid"), self
        )
        self.crop.setCheckable(True)
        self.crop.setChecked(True)
        self.crop.toggled.connect(self.sync_display)
        self._build_properties()
        return scroll

    def _list_panel(self, layout, title, remove_tooltip, allow_remove=True):
        panel = QtWidgets.QFrame()
        panel.setObjectName("pointcloudListPanel")
        body = QtWidgets.QVBoxLayout(panel)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 2)
        header.setSpacing(0)
        heading = self.window._heading(self.tr(title))
        header.addWidget(heading, 1)
        body.addLayout(header)
        listing = PointCloudListWidget(
            object_controls=True,
            remove_tooltip=self.tr(remove_tooltip),
            allow_remove=allow_remove,
        )
        listing.setObjectName("pointcloudList")
        listing.setAccessibleName(self.tr(title))
        listing.setProperty("integratedPanel", True)
        listing.setMinimumHeight(130)
        listing.setUniformItemSizes(True)
        listing.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        listing.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        listing.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        body.addWidget(listing, 1)
        listing.set_header(header)
        layout.addWidget(panel, 1)
        return listing, heading, header

    def _header_action(self, header, title, callback, icon):
        action = self.window._action(self.tr(title), callback, self, icon=icon)
        button = self.window._tool_button(action, header)
        button.setProperty("panelHeader", True)
        button.setFixedSize(24, 24)
        button.setIconSize(QtCore.QSize(16, 16))
        return action

    def _object_menu(self, position):
        item = self.objects.itemAt(position)
        self.objects.setCurrentItem(item)
        menu = QtWidgets.QMenu(self.objects)
        menu.addActions(self.edit_actions)
        menu.addSeparator()
        menu.addAction(self.crop)
        menu.exec(self.objects.viewport().mapToGlobal(position))

    def edit_properties(self):
        if self.selected is not None:
            self.properties.show()
            self.properties.raise_()
            self.properties.activateWindow()

    def _build_properties(self):
        self.properties = QtWidgets.QDialog(self.window)
        self.properties.setWindowTitle(self.tr("Edit object"))
        self.properties.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.properties.setMinimumWidth(360)
        layout = QtWidgets.QVBoxLayout(self.properties)
        self.class_combo = self.window._combo([])
        self.class_combo.setAccessibleName(self.tr("Cuboid class"))
        layout.addWidget(QtWidgets.QLabel(self.tr("Object class")))
        layout.addWidget(self.class_combo)
        self.class_combo.activated.connect(self._change_class)
        form = QtWidgets.QGridLayout()
        for column in range(3):
            form.setColumnStretch(column, 1)
        self.fields = {}
        for row, (name, title, labels) in enumerate(
            (
                ("center", "Center", ("X", "Y", "Z")),
                ("size", "Size", ("Length", "Width", "Height")),
                ("rotation", "Rotation (deg)", ("X", "Y", "Z")),
            )
        ):
            form.addWidget(QtWidgets.QLabel(self.tr(title)), row * 3, 0, 1, 3)
            self.fields[name] = []
            for column, label in enumerate(labels):
                form.addWidget(
                    QtWidgets.QLabel(self.tr(label)), row * 3 + 1, column
                )
                field = QtWidgets.QDoubleSpinBox()
                field.setDecimals(3)
                field.setRange(MIN_SIZE if name == "size" else -1e9, 1e9)
                field.setSingleStep(1 if name == "rotation" else 0.1)
                field.setKeyboardTracking(False)
                field.setMinimumWidth(0)
                field.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Ignored,
                    QtWidgets.QSizePolicy.Policy.Fixed,
                )
                self.window._protect_wheel(field)
                field.editingFinished.connect(
                    lambda n=name, i=column: self._edit_field(n, i)
                )
                form.addWidget(field, row * 3 + 2, column)
                self.fields[name].append(field)
        layout.addLayout(form)
        self.occluded = QtWidgets.QCheckBox(self.tr("Occluded"))
        self.locked = QtWidgets.QCheckBox(self.tr("Locked"))
        self.occluded.toggled.connect(
            lambda value: self._flag("occluded", value)
        )
        self.locked.toggled.connect(lambda value: self._flag("locked", value))
        confirm = QtWidgets.QPushButton(self.tr("OK"))
        confirm.setObjectName("pointcloudConfirmButton")
        confirm.setIcon(get_icon("confirm", "#ffffff", "#ffffff"))
        confirm.setIconSize(QtCore.QSize(16, 16))
        confirm.setDefault(True)
        confirm.clicked.connect(self.properties.accept)
        for column, widget in enumerate((self.occluded, self.locked, confirm)):
            widget.setAttribute(
                QtCore.Qt.WidgetAttribute.WA_LayoutUsesWidgetRect
            )
            form.addWidget(
                widget, 9, column, QtCore.Qt.AlignmentFlag.AlignBottom
            )

    def activate(self, enabled=True):
        self.cancel()
        if not enabled and self._expanded_view is not None:
            self.toggle_expanded_view(self._expanded_view)
        if self.enabled and not enabled:
            self._views_visible = not self.orthographic.isHidden()
            if self._views_visible and self.isVisible():
                self._view_sizes = self.sizes()
        self.enabled = enabled
        self.orthographic.setVisible(enabled and self._views_visible)
        if enabled:
            self.setSizes(self._view_sizes or [self.height(), 0])
        self.views_button.setVisible(enabled)
        self.draw_action.setVisible(enabled)
        for action in self.shortcuts:
            action.setEnabled(enabled)
        for view in self.views:
            view.detection_enabled = enabled
            view._update(scene=False)
        self.refresh()
        self._update_views_button()

    def toggle_views(self, checked=False, show=None):
        if self._expanded_view is not None:
            self.toggle_expanded_view(self._expanded_view)
        show = self.orthographic.isHidden() if show is None else show
        if show == (not self.orthographic.isHidden()):
            return
        if show:
            self.orthographic.show()
            if self._view_sizes:
                self.setSizes(self._view_sizes)
            self.sync_display()
        else:
            self._view_sizes = self.sizes()
            self.orthographic.hide()
        self._update_views_button()

    def toggle_expanded_view(self, view):
        self.cancel()
        previous = self._expanded_view
        if previous is not None:
            self._expanded_view = None
            previous.set_expanded(False)
            for member in self.views:
                member.show()
            vertical, horizontal = self._expanded_sizes
            self.setSizes(vertical)
            self.orthographic.setSizes(horizontal)
            self._expanded_sizes = None
            if previous is view:
                view.setFocus()
                return
        self._expanded_sizes = (self.sizes(), self.orthographic.sizes())
        self._expanded_view = view
        self.orthographic.show()
        for member in self.views:
            member.setVisible(member is view)
        view.set_expanded(True)
        view.setFocus()

    def _update_views_button(self):
        expanded = not self.orthographic.isHidden()
        title = (
            self.tr("Hide three views")
            if expanded
            else self.tr("Show three views")
        )
        self.views_button.setToolTip(title)
        self.views_button.setAccessibleName(title)
        self.views_button.setIcon(
            self.window._icon("panel-up" if expanded else "panel-down")
        )

    def cancel(self):
        for view in self.views:
            view.cancel_selection()
            view.creating = False
        self.draw_action.setChecked(False)

    def refresh(self):
        if self._syncing:
            return
        self._syncing = True
        if self.document is not self._document:
            self._document = self.document
            self.selected_id = None
            self.hidden_ids.clear()
            for view in self.views:
                view.creating = False
            self.draw_action.setChecked(False)
            if self.enabled:
                self.sync_display()
        box = self.selected
        if box is None:
            self.selected_id = None
        class_id = box.class_id if box else self.current_class()
        self.class_combo.clear()
        classes = {
            entry.id: entry.name
            for entry in self.window.class_definitions["detection"]
            if entry.id
        }
        if self.document is not None:
            for member in self.document.cuboids:
                classes.setdefault(member.class_id, f"Class {member.class_id}")
        for class_id_value, name in classes.items():
            self.class_combo.addItem(name, class_id_value)
        index = self.class_combo.findData(class_id)
        self.class_combo.setCurrentIndex(max(index, 0))
        self._refresh_lists(classes)
        editable = box is not None and not box.locked
        self.class_combo.setEnabled(box is None or editable)
        for name, fields in self.fields.items():
            for index, field in enumerate(fields):
                field.setEnabled(editable)
                value = getattr(box, name)[index] if box else 0
                field.setValue(
                    np.degrees(value) if name == "rotation" else value
                )
        for widget, name in (
            (self.occluded, "occluded"),
            (self.locked, "locked"),
        ):
            with QtCore.QSignalBlocker(widget):
                widget.setChecked(bool(box and getattr(box, name)))
            widget.setEnabled(
                box is not None and (name == "locked" or editable)
            )
        available = (
            box is not None,
            box is not None,
            editable,
            editable,
            editable,
            box is not None,
            self.clipboard is not None and self.document is not None,
        )
        for button, enabled in zip(self.edit_actions, available):
            button.setEnabled(self.enabled and enabled)
        self.draw_action.setEnabled(
            self.enabled and self.document is not None and bool(classes)
        )
        self.instance_action.setEnabled(
            self.enabled
            and self.document is not None
            and self.window._current_instance() is not None
        )
        self._syncing = False
        self.preview(None)
        if self.enabled:
            for view in self.views[1:]:
                view.align_cuboid(box)
            self.sync_display()

    def _refresh_lists(self, classes):
        members = self.document.cuboids if self.document else ()
        self.labels.blockSignals(True)
        self.labels.clear()
        by_class = {value: [] for value in classes}
        for member in members:
            by_class[member.class_id].append(member)
        colors = {
            entry.id: entry.color
            for entry in self.window.class_definitions["detection"]
        }
        for value, name in classes.items():
            group = by_class[value]
            item = self._list_item(
                f"{name} ({value})",
                value,
                colors.get(value, "#60A5FA"),
                bool(group) and all(member.locked for member in group),
                not group
                or any(member.id not in self.hidden_ids for member in group),
                not any(member.locked for member in group),
            )
            self.labels.addItem(item)
            if value == self.class_combo.currentData():
                self.labels.setCurrentItem(item)
        self.labels.blockSignals(False)
        self.objects.blockSignals(True)
        self.objects.clear()
        for member in members:
            item = self._list_item(
                f"#{member.id} · {classes[member.class_id]}",
                member.id,
                colors.get(member.class_id, "#60A5FA"),
                member.locked,
                member.id not in self.hidden_ids,
                not member.locked,
            )
            self.objects.addItem(item)
            if member.id == self.selected_id:
                self.objects.setCurrentItem(item)
        self.objects.blockSignals(False)
        self.labels_heading.setText(f"{self.tr('Labels')} ({len(classes)})")
        self.objects_heading.setText(f"{self.tr('Objects')} ({len(members)})")
        all_locked = bool(members) and all(member.locked for member in members)
        self.lock_all_action.setIcon(
            self.window._icon("lock" if all_locked else "unlock")
        )
        lock_title = (
            self.tr("Unlock all objects")
            if all_locked
            else self.tr("Lock all objects")
        )
        self.lock_all_action.setText(lock_title)
        self.lock_all_action.setToolTip(lock_title)
        self.lock_all_action.setEnabled(bool(members))
        self.classes_lock_action.setIcon(self.lock_all_action.icon())
        title = (
            self.tr("Unlock all classes")
            if all_locked
            else self.tr("Lock all classes")
        )
        self.classes_lock_action.setText(title)
        self.classes_lock_action.setToolTip(title)
        self.classes_lock_action.setEnabled(bool(members))
        self.visibility_action.setIcon(
            self.window._icon(
                "eye"
                if any(member.id not in self.hidden_ids for member in members)
                else "eye-off"
            )
        )
        self.visibility_action.setEnabled(bool(members))
        self.classes_visibility_action.setIcon(self.visibility_action.icon())
        self.classes_visibility_action.setEnabled(bool(members))

    def sync_display(self, *args):
        if (
            not self.enabled
            or self.document is None
            or self.orthographic.isHidden()
        ):
            return
        source = self.window.viewport
        box = self.selected
        for view in self.views[1:]:
            if view._points is not source._points:
                view.set_cloud(source._points)
            view.set_point_size(source._point_size)
            if not np.array_equal(view._colors, source._colors):
                view.set_colors(source._colors)
            self._sync_crop(view, box)

    def _sync_crop(self, view, box, cancel_selection=True):
        source = self.window.viewport
        visible = source._visible
        if (
            box is not None
            and self.crop.isChecked()
            and not view.creating
            and view.orthographic_view != "top"
        ):
            _, _, forward = view._basis()
            depth = (source._points[:, :3] - box.center) @ forward
            half = np.abs(box.matrix.T @ forward) @ (np.array(box.size) / 2)
            visible = visible & (np.abs(depth) <= half + 0.1)
        if not np.array_equal(view._visible, visible):
            view.set_visible_mask(visible, cancel_selection=cancel_selection)

    def preview(self, box, source=None):
        boxes = self.document.cuboids if self.document else ()
        if box is not None:
            boxes = tuple(
                box if member.id == box.id else member for member in boxes
            )
        visible = tuple(
            member
            for member in boxes
            if self.enabled and member.id not in self.hidden_ids
        )
        for view in self.views:
            previous = view.selected_cuboid
            view.class_colors = {
                entry.id: entry.color
                for entry in self.window.class_definitions["detection"]
            }
            view.class_names = {
                entry.id: entry.name
                for entry in self.window.class_definitions["detection"]
            }
            view.set_cuboids(visible, self.selected_id)
            selected = view.selected_cuboid
            if (
                view.orthographic_view is not None
                and previous is not None
                and selected is not None
                and previous.id == selected.id
            ):
                if (
                    previous.center != selected.center
                    and previous.size == selected.size
                    and previous.rotation == selected.rotation
                ):
                    view._focus_animation.stop()
                    if view is not source:
                        view._center += (
                            np.asarray(selected.center) - previous.center
                        )
                if (
                    previous.center != selected.center
                    or previous.rotation != selected.rotation
                    or previous.size != selected.size
                ):
                    self._sync_crop(view, selected, cancel_selection=False)

    def select(self, cuboid_id):
        changed = self.selected_id != cuboid_id
        if not changed:
            return
        self.cancel()
        cameras = [
            (view._center.copy(), view._scale) for view in self.views[1:]
        ]
        self.selected_id = cuboid_id
        self.refresh()
        for view, camera in zip(self.views[1:], cameras):
            view._center, view._scale = camera
            view.align_cuboid(self.selected, fit=True, animate=True)

    def _select_item(self, current, previous):
        if not self._syncing:
            self.select(
                current.data(QtCore.Qt.ItemDataRole.UserRole)
                if current
                else None
            )

    def current_class(self):
        item = self.labels.currentItem()
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None

    def _label_selected(self, current, previous):
        if not self._syncing:
            self.cancel()
            self.selected_id = None
            self.refresh()

    def _list_item(self, text, value, color, locked, visible, removable):
        item = QtWidgets.QListWidgetItem(text)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, value)
        item.setData(PointCloudListWidget.LOCKED_ROLE, locked)
        item.setData(PointCloudListWidget.REMOVABLE_ROLE, removable)
        item.setBackground(QtGui.QColor(color))
        item.setCheckState(
            QtCore.Qt.CheckState.Checked
            if visible
            else QtCore.Qt.CheckState.Unchecked
        )
        item.setToolTip(text)
        return item

    def _members(self, class_id=None):
        return tuple(
            box
            for box in (self.document.cuboids if self.document else ())
            if class_id is None or box.class_id == class_id
        )

    def _set_locked(self, members):
        if not members:
            return
        self.cancel()
        self.document.set_cuboids_locked(
            (box.id for box in members), not all(box.locked for box in members)
        )
        self.window._refresh()
        self.window._schedule_autosave()

    def _lock_object(self, item):
        value = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self._set_locked(
            tuple(box for box in self._members() if box.id == value)
        )

    def _lock_label(self, item):
        self._set_locked(
            self._members(item.data(QtCore.Qt.ItemDataRole.UserRole))
        )

    def _lock_all(self):
        self._set_locked(self._members())

    def _set_visible(self, ids, visible):
        self.cancel()
        if visible:
            self.hidden_ids.difference_update(ids)
        else:
            self.hidden_ids.update(ids)
        self.refresh()

    def _visibility_changed(self, item):
        if not self._syncing:
            self._set_visible(
                {item.data(QtCore.Qt.ItemDataRole.UserRole)},
                item.checkState() == QtCore.Qt.CheckState.Checked,
            )

    def _label_visibility_changed(self, item):
        if not self._syncing:
            self._set_visible(
                {
                    box.id
                    for box in self._members(
                        item.data(QtCore.Qt.ItemDataRole.UserRole)
                    )
                },
                item.checkState() == QtCore.Qt.CheckState.Checked,
            )

    def _toggle_all_visibility(self):
        ids = {box.id for box in self._members()}
        self._set_visible(ids, not bool(ids - self.hidden_ids))

    def _delete_object(self, item):
        value = item.data(QtCore.Qt.ItemDataRole.UserRole)
        box = next((box for box in self._members() if box.id == value), None)
        if box is not None and not box.locked:
            self._delete_ids({value})

    def _delete_ids(self, ids):
        self.cancel()
        self.document.delete_cuboids(ids)
        self.hidden_ids.difference_update(ids)
        if self.selected_id in ids:
            self.selected_id = None
        self.window._refresh()
        self.window._schedule_autosave()

    def _delete_label(self, item):
        if self.window.task_type is not None:
            return
        value = item.data(QtCore.Qt.ItemDataRole.UserRole)
        members = self._members(value)
        if any(box.locked for box in members):
            return
        if not self.window._confirm(
            self.tr(
                "Delete this class and its {count} objects in the current frame? Segmentation point labels and other frames are unchanged. Object deletion can be undone."
            ).format(count=len(members))
        ):
            return
        self.window.class_definitions["detection"] = [
            entry
            for entry in self.window.class_definitions["detection"]
            if entry.id != value
        ]
        if self.document is not None:
            self._delete_ids({box.id for box in members})
        else:
            self.window._refresh()
            self.window._schedule_autosave()

    def start_creation(self, checked=None):
        if (
            not self.enabled
            or self.document is None
            or self.current_class() is None
        ):
            return
        was_creating = self.views[0].creating
        self.cancel()
        if checked is False or (checked is None and was_creating):
            return
        if self._expanded_view is not None:
            self.toggle_expanded_view(self._expanded_view)
        self.window._select_tool("browse")
        if self.orthographic.isHidden():
            self.toggle_views(show=True)
        self.selected_id = None
        self.refresh()
        self.draw_action.setChecked(True)
        for view in self.views:
            view.creating = True
        main = self.views[0]
        main.setFocus()
        point = main.mapFromGlobal(QtGui.QCursor.pos())
        main._update_creation_preview((point.x(), point.y()))
        self.window._status(
            self.tr(
                "Move the preview in 3D and double-click to create; or draw a rectangle in Top, Front or Side. Esc cancels."
            )
        )

    def create(self, center, size):
        class_id = self.current_class()
        if self.document is None or class_id is None:
            return
        box = Cuboid(
            self.document.next_cuboid_id(),
            class_id,
            tuple(center),
            tuple(size) if size is not None else (4.5, 1.8, 1.6),
        )
        self.cancel()
        self.commit(box)
        self.focus()

    def commit(self, box):
        if self.document is None:
            return
        try:
            changed = self.document.set_cuboid(box)
        except ValueError as error:
            self.window._error(str(error))
            self.refresh()
            return
        self.selected_id = box.id
        if changed:
            self.window._refresh()
            self.window._schedule_autosave()
        else:
            self.refresh()

    def _edit_field(self, name, index):
        box = self.selected
        if self._syncing or box is None or box.locked:
            return
        values = list(getattr(box, name))
        value = self.fields[name][index].value()
        displayed = (
            np.degrees(values[index]) if name == "rotation" else values[index]
        )
        if abs(displayed - value) < 0.00051:
            return
        values[index] = np.radians(value) if name == "rotation" else value
        self.commit(replace(box, **{name: tuple(values)}))

    def _change_class(self):
        if self.selected is not None and not self.selected.locked:
            self.commit(
                replace(self.selected, class_id=self.class_combo.currentData())
            )

    def _flag(self, name, value):
        if not self._syncing and self.selected is not None:
            self.commit(replace(self.selected, **{name: value}))

    def focus(self, *args):
        box = self.selected
        for view in self.views:
            if box:
                view.align_cuboid(box, fit=True, animate=True)

    def fit(self):
        box = self.selected
        if box is None or box.locked:
            return
        points = self.document.frame.points
        try:
            self.commit(box.fitted(points[box.contains(points)]))
        except ValueError as error:
            self.window._status(str(error))

    def from_instance(self):
        if self.document is None:
            return
        key = self.window._current_instance()
        if key is None or key[0] == 0:
            return
        points = self.document.frame.points[
            self.window._instance_indices(key), :3
        ]
        low = points.min(axis=0).astype(np.float64)
        high = points.max(axis=0).astype(np.float64)
        self.commit(
            Cuboid(
                self.document.next_cuboid_id(),
                key[0],
                tuple((low + high) / 2),
                tuple(np.maximum(high - low, MIN_SIZE)),
            )
        )
        self.focus()

    def delete(self):
        box = self.selected
        if box is None or box.locked:
            return
        self._delete_ids({box.id})

    def duplicate(self):
        box = self.selected
        if box is not None:
            self.commit(
                replace(
                    box,
                    id=self.document.next_cuboid_id(),
                    center=tuple(
                        np.asarray(box.center) + box.matrix[:, 0] * 0.5
                    ),
                    locked=False,
                )
            )

    def copy(self):
        if self.selected is not None:
            self.clipboard = self.selected
            self.refresh()
            self.window._status(
                self.tr(
                    "Cuboid copied. Navigate to another frame and paste to reuse its position."
                )
            )

    def paste(self):
        if self.clipboard is not None and self.document is not None:
            self.commit(
                replace(
                    self.clipboard,
                    id=self.document.next_cuboid_id(),
                    locked=False,
                )
            )
            self.focus()
