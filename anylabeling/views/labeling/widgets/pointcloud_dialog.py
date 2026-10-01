import hashlib
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.config import get_work_directory
from anylabeling.views.labeling.pointcloud.io import (
    cuboid_source_path,
    default_label_path,
    discover_frames,
    label_candidates,
    load_classes,
    load_frame,
    save_classes,
    save_cuboids,
    save_labels,
)
from anylabeling.views.labeling.pointcloud.model import (
    DEFAULT_CLASSES,
    AnnotationDocument,
    ClassDefinition,
)
from anylabeling.views.labeling.pointcloud.controls import (
    ClassDefinitionDialog,
    PointCloudListWidget,
    PointCloudToolScrollArea,
    PointSizePopup,
    ShortcutsDialog,
)
from anylabeling.views.labeling.pointcloud.icons import center_pixmap, get_icon
from anylabeling.views.labeling.pointcloud.camera import (
    CameraConfigurationDialog,
    CameraPanel,
)
from anylabeling.views.labeling.pointcloud.cuboid_viewport import (
    CuboidViewport,
)
from anylabeling.views.labeling.pointcloud.detection import DetectionWorkspace
from anylabeling.views.labeling.pointcloud.export import ExportFrame
from anylabeling.views.labeling.pointcloud.export_dialog import (
    PointCloudExportDialog,
)
from anylabeling.views.labeling.pointcloud.import_dataset import ImportTarget
from anylabeling.views.labeling.pointcloud.import_dialog import (
    PointCloudImportDialog,
)
from anylabeling.views.labeling.pointcloud.style import get_pointcloud_style
from anylabeling.views.labeling.pointcloud.task_dialog import CreateTaskDialog
from anylabeling.views.labeling.utils.colormap import label_colormap
from anylabeling.views.labeling.utils.general import open_url
from anylabeling.views.labeling.utils.qt import new_icon
from anylabeling.views.labeling.utils.style import get_dock_style
from anylabeling.views.labeling.utils.theme import get_theme


class FrameLoader(QtCore.QThread):
    loaded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self, path, label_path, config_path, parent, output_path=None
    ):
        super().__init__(parent)
        self.path = path
        self.label_path = label_path
        self.config_path = config_path
        self.output_path = output_path

    def run(self):
        try:
            frame = load_frame(
                self.path,
                self.label_path,
                cuboid_path=(
                    self.output_path.with_suffix(".cuboids.json")
                    if self.output_path is not None
                    else None
                ),
            )
            if self.output_path is not None:
                frame.label_path = self.output_path
                frame.label_exists = False
            classes = (
                load_classes(self.config_path)
                if self.config_path is not None and self.config_path.exists()
                else None
            )
            if not self.isInterruptionRequested():
                self.loaded.emit((frame, classes))
        except (OSError, ValueError, MemoryError) as error:
            if not self.isInterruptionRequested():
                self.failed.emit(str(error))


class PointCloudDialog(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        palette = self.palette()
        palette.setColor(
            QtGui.QPalette.ColorRole.Window,
            QtGui.QColor(get_theme()["background"]),
        )
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.settings = QtCore.QSettings("anylabeling", "pointcloud")
        self.document = None
        self.task_type = None
        self.files = []
        self.frame_index = -1
        self.label_directory = None
        self.label_overrides = {}
        self.class_definitions = {
            "detection": [],
            "segmentation": list(DEFAULT_CLASSES),
        }
        self._saved_classes = {
            task: list(classes)
            for task, classes in self.class_definitions.items()
        }
        self.config_path = None
        self.dataset = None
        self._worker = None
        self._progress = None
        self._queued_frame = None
        self._navigation_focus = None
        self._pending = None
        self._load_result = None
        self._load_error = None
        self._close_approved = False
        self._refreshing = False
        self._visible = np.empty(0, dtype=bool)
        self._visible_count = 0
        self._display_signature = None
        self._display_revision = None
        self._palette_classes = None
        self._palette = None
        self._intensity_range = None
        self._task_color_modes = {False: "semantic", True: "rgb"}
        self._segmentation_tool = "browse"
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(350)
        self._autosave_timer.timeout.connect(self._autosave)
        self._build_ui()
        self.resize(1280, 800)
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if not any(
            screen.availableGeometry().intersects(self.frameGeometry())
            for screen in QtWidgets.QApplication.screens()
        ):
            self.move(
                QtWidgets.QApplication.primaryScreen()
                .availableGeometry()
                .topLeft()
            )
        self._refresh()

    def show(self):
        if not self.isVisible():
            self.ensurePolished()
            self.layout().activate()
            self.grab()
        super().show()

    @property
    def config_dirty(self):
        return self.class_definitions != self._saved_classes

    def _class_task(self):
        return "detection" if self.detection.enabled else "segmentation"

    def _icon(self, name):
        theme = get_theme()
        custom = get_icon(name, theme["text"], theme["primary"])
        if custom is not None:
            return custom
        source = new_icon(name, "svg").pixmap(QtCore.QSize(36, 36))
        if source.isNull():
            return new_icon(name, "svg")
        source = center_pixmap(source)
        icon = QtGui.QIcon()
        for mode, color in (
            (QtGui.QIcon.Mode.Normal, get_theme()["text"]),
            (QtGui.QIcon.Mode.Disabled, get_theme()["text_secondary"]),
        ):
            pixmap = QtGui.QPixmap(source.size())
            pixmap.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pixmap)
            painter.drawPixmap(0, 0, source)
            painter.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_SourceIn
            )
            painter.fillRect(pixmap.rect(), QtGui.QColor(color))
            painter.end()
            icon.addPixmap(pixmap, mode)
        return icon

    def _action(
        self,
        text,
        callback,
        container,
        shortcut=None,
        viewport=False,
        icon=None,
    ):
        action = QtGui.QAction(text, self)
        if icon:
            action.setIcon(self._icon(icon))
        action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
            if viewport:
                action.setShortcutContext(
                    QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
                )
                self.viewport.addAction(action)
            else:
                action.setShortcutContext(
                    QtCore.Qt.ShortcutContext.WindowShortcut
                )
                self.addAction(action)
            action.setToolTip(
                text
                + " ("
                + action.shortcut().toString(
                    QtGui.QKeySequence.SequenceFormat.NativeText
                )
                + ")"
            )
        container.addAction(action)
        return action

    def _button(self, text, callback, layout):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _tool_button(self, action, layout):
        button = QtWidgets.QToolButton()
        button.setDefaultAction(action)
        button.setAccessibleName(action.text())
        button.setIconSize(QtCore.QSize(18, 18))
        button.setObjectName("pointcloudIconButton")
        button.setFixedSize(30, 30)
        layout.addWidget(button)
        return button

    def _combo(self, items, layout=None):
        combo = QtWidgets.QComboBox()
        self._protect_wheel(combo)
        combo.setView(QtWidgets.QListView(combo))
        combo.view().setUniformItemSizes(True)
        combo.view().setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        combo.view().setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(10)
        combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        for title, value in items:
            combo.addItem(title, value)
        if layout is not None:
            layout.addWidget(combo)
        return combo

    def _protect_wheel(self, widget):
        widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Type.Wheel and not watched.hasFocus():
            event.ignore()
            parent = watched.parentWidget()
            while parent is not None and not isinstance(
                parent, QtWidgets.QAbstractScrollArea
            ):
                parent = parent.parentWidget()
            if parent is not None:
                viewport = parent.viewport()
                forwarded = QtGui.QWheelEvent(
                    QtCore.QPointF(
                        viewport.mapFromGlobal(
                            event.globalPosition().toPoint()
                        )
                    ),
                    event.globalPosition(),
                    event.pixelDelta(),
                    event.angleDelta(),
                    event.buttons(),
                    event.modifiers(),
                    event.phase(),
                    event.inverted(),
                    device=event.pointingDevice(),
                )
                QtWidgets.QApplication.sendEvent(viewport, forwarded)
                event.setAccepted(forwarded.isAccepted())
            return True
        return super().eventFilter(watched, event)

    def _label(self, text="", muted=False):
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        if muted:
            label.setObjectName("pointcloudMuted")
        return label

    def _heading(self, text):
        label = QtWidgets.QLabel(text)
        label.setObjectName("pointcloudSectionTitle")
        return label

    def _list(self, title, editable=False, toggle_selection=False):
        widget = (
            PointCloudListWidget(
                object_controls=True,
                toggle_selection=toggle_selection,
                remove_tooltip=(
                    self.tr("Delete instance")
                    if toggle_selection
                    else self.tr("Remove definition")
                ),
            )
            if editable
            else QtWidgets.QListWidget()
        )
        widget.setObjectName("pointcloudList")
        widget.setAccessibleName(title)
        widget.setUniformItemSizes(True)
        widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        widget.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        widget.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        widget.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        return widget

    def _build_ui(self):
        self.setObjectName("pointcloudWorkspace")
        self.setMinimumSize(960, 480)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.setChildrenCollapsible(False)
        self._panel_widths = {0: 280, 2: 280}
        self.setCentralWidget(self.splitter)
        self.viewport = CuboidViewport()
        self.viewport.selection_completed.connect(self._apply_selection)
        self.viewport.status_message.connect(self._status)
        self.viewport.renderer_error.connect(self._renderer_error)
        self.viewport.selection_started.connect(self._update_selection_actions)
        self.viewport.selection_cancelled.connect(
            self._update_selection_actions
        )
        self.viewport.selection_completed.connect(
            self._update_selection_actions
        )
        self._build_file_toolbar()
        self.splitter.addWidget(self._build_files_panel())
        center = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        view_header = QtWidgets.QHBoxLayout()
        view_header.setContentsMargins(0, 0, 0, 0)
        view_header.setSpacing(0)
        left_entry = QtWidgets.QWidget()
        left_entry.setFixedHeight(42)
        left_layout = QtWidgets.QHBoxLayout(left_entry)
        left_layout.setContentsMargins(3, 7, 0, 9)
        self.show_frames_button = self._panel_button(0, left_layout)
        view_header.addWidget(left_entry, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        left_entry.hide()
        view_header.addWidget(self._build_view_toolbar(), 1)
        right_entry = QtWidgets.QWidget()
        right_entry.setFixedHeight(42)
        right_layout = QtWidgets.QHBoxLayout(right_entry)
        right_layout.setContentsMargins(0, 5, 6, 5)
        self.show_annotation_button = self._panel_button(2, right_layout)
        right_entry.hide()
        center_layout.addLayout(view_header)
        self.detection = DetectionWorkspace(self, center)
        views_entry = QtWidgets.QFrame()
        views_entry.setFixedHeight(42)
        views_layout = QtWidgets.QHBoxLayout(views_entry)
        views_layout.setContentsMargins(0, 5, 0, 5)
        views_layout.setSpacing(0)
        self.camera_controls_action = self._action(
            self.tr("Show camera controls"),
            self._toggle_camera_controls,
            self,
            icon="keyboard",
        )
        self.camera_controls_action.setCheckable(True)
        self.camera_controls_button = self._tool_button(
            self.camera_controls_action, views_layout
        )
        self.camera_controls_button.setFixedSize(26, 26)
        self.camera_controls_button.setProperty("panelHeader", True)
        self.camera_controls_button.setProperty("cameraControls", True)
        self.camera_controls_button.setFocusPolicy(
            QtCore.Qt.FocusPolicy.NoFocus
        )
        self.camera_panel_action = self._action(
            self.tr("Show camera images"),
            lambda visible: self.camera_panel.set_images_visible(visible),
            self,
            icon="camera",
        )
        self.camera_panel_action.setCheckable(True)
        self.camera_panel_action.setEnabled(False)
        self.camera_panel_button = self._tool_button(
            self.camera_panel_action, views_layout
        )
        self.camera_panel_button.setFixedSize(26, 26)
        self.camera_panel_button.setProperty("panelHeader", True)
        self.camera_panel_button.setProperty("cameraControls", True)
        self.camera_panel_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        views_layout.addWidget(self.detection.views_button)
        view_header.addWidget(views_entry, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        view_header.addWidget(right_entry, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        center_layout.addWidget(self.detection, 1)
        self.camera_panel = CameraPanel(self)
        self.camera_panel.hide()
        self.splitter.addWidget(center)
        self.splitter.addWidget(self._build_annotation_panel())
        self.sidebar_tabs.insertTab(0, self.detection.panel, "Det")
        self.sidebar_tabs.setTabToolTip(0, self.tr("Detection"))
        toolbar_layout = self.view_tool_scroll.widget().layout()
        draw_button = self._tool_button(
            self.detection.draw_action, toolbar_layout
        )
        toolbar_layout.insertWidget(3, draw_button)
        self.point_size.valueChanged.connect(self.detection.sync_display)
        self.splitter.setSizes([280, 720, 280])
        self.splitter.setStretchFactor(1, 1)
        self.color_mode.currentIndexChanged.connect(self._refresh_display)
        self._tool_changed()
        self._bind_shortcuts()
        self.viewport.camera_view_changed.connect(self._camera_view_changed)
        self.summary = QtWidgets.QLabel()
        self.summary.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.statusBar().addPermanentWidget(self.summary)
        self.statusBar().setSizeGripEnabled(False)
        self.setStyleSheet(get_pointcloud_style())
        self.sidebar_tabs.currentChanged.connect(self._task_changed)
        self.sidebar_tabs.setCurrentWidget(self.detection.panel)

    def _task_changed(self, index):
        if self.task_type is not None:
            selected = 0 if self.task_type == "detection" else 1
            if index != selected:
                with QtCore.QSignalBlocker(self.sidebar_tabs):
                    self.sidebar_tabs.setCurrentIndex(selected)
                return
        enabled = self.sidebar_tabs.widget(index) is self.detection.panel
        self._task_color_modes[self.detection.enabled] = (
            self.color_mode.currentData()
        )
        if enabled:
            self._segmentation_tool = self._current_tool()
        self._select_tool("browse" if enabled else self._segmentation_tool)
        self.detection.activate(enabled)
        for name in ("brush", "polygon"):
            self.tool_actions[name].setEnabled(not enabled)
            self.tool_actions[name].setVisible(not enabled)
        for action in self.operation_actions.values():
            action.setVisible(not enabled)
        self.through_action.setVisible(not enabled)
        self.through_action.setEnabled(not enabled)
        with QtCore.QSignalBlocker(self.color_mode):
            self.color_mode.setCurrentIndex(
                self.color_mode.findData(self._task_color_modes[enabled])
            )
        self._sync_render_actions()
        for button in self.view_tool_scroll.findChildren(
            QtWidgets.QToolButton
        ):
            action = button.defaultAction()
            if action is not None:
                button.setVisible(action.isVisible())
        self._refresh_target()
        self._refresh_display()

    def createPopupMenu(self):
        return None

    def _build_file_toolbar(self):
        toolbar = self.addToolBar(self.tr("Point cloud files"))
        toolbar.setObjectName("pointcloudFileTools")
        toolbar.setMovable(False)
        toolbar.setIconSize(QtCore.QSize(18, 18))
        toolbar.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.create_task_action = self._action(
            self.tr("Create task"), self.create_task, toolbar, icon="new"
        )
        self.save_as_action = self._action(
            self.tr("Save as"), self.save_as, toolbar, icon="save-plus"
        )
        self.save_as_action.setToolTip(
            self.tr(
                "Save point labels and associated 3D cuboids to another location."
            )
        )
        self.import_action = self._action(
            self.tr("Upload"),
            self._import_dataset,
            toolbar,
            icon="pointcloud-upload",
        )
        self.import_action.setToolTip(
            self.tr("Import 3D objects into the current sequence.")
        )
        self.export_action = self._action(
            self.tr("Export"),
            self._export_dataset,
            toolbar,
            icon="pointcloud-download",
        )
        self.export_action.setToolTip(
            self.tr("Export 3D objects from all frames.")
        )
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        toolbar.addWidget(spacer)
        shortcuts_action = self._action(
            self.tr("Keyboard shortcuts"),
            self._show_shortcuts,
            toolbar,
            icon="command",
        )
        shortcuts_button = toolbar.widgetForAction(shortcuts_action)
        shortcuts_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        shortcuts_button.setAccessibleName(shortcuts_action.text())
        shortcuts_button.setObjectName("pointcloudHelpButton")
        shortcuts_button.setFixedSize(26, 26)
        shortcuts_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        help_action = self._action(
            self.tr("Help"), self._show_help, toolbar, icon="help-circle"
        )
        help_action.setMenuRole(QtGui.QAction.MenuRole.NoRole)
        help_button = toolbar.widgetForAction(help_action)
        help_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        help_button.setAccessibleName(help_action.text())
        help_button.setObjectName("pointcloudHelpButton")
        help_button.setFixedSize(26, 26)
        help_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def _build_files_panel(self):
        panel = QtWidgets.QFrame()
        panel.setObjectName("pointcloudPanel")
        panel.setMinimumWidth(184)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QtWidgets.QFrame()
        header.setObjectName("pointcloudPanelHeader")
        header.setFixedHeight(42)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(3, 5, 12, 5)
        header_layout.setSpacing(0)
        self._panel_button(0, header_layout)
        heading = self._heading(self.tr("Frames"))
        heading.setIndent(1)
        header_layout.addWidget(heading)
        header_layout.addStretch()
        self.frame_progress = QtWidgets.QLabel("0/0")
        self.frame_progress.setObjectName("pointcloudMuted")
        self.frame_progress.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        header_layout.addWidget(self.frame_progress)
        layout.addWidget(header)
        self.file_list = self._list(self.tr("Frames"))
        self.file_list.setObjectName("FileList")
        self.file_list.setStyleSheet(
            get_dock_style()
            + "QListWidget#FileList::item { padding-left: 3px; }"
            + "QListWidget#FileList::indicator { margin-right: 0; }"
        )
        self.file_list.setIconSize(QtCore.QSize(8, 8))
        self.file_list.currentRowChanged.connect(self._select_frame)
        self.file_list.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.file_list.customContextMenuRequested.connect(
            self._file_context_menu
        )
        layout.addWidget(self.file_list, 1)
        return panel

    def _panel_button(self, index, layout):
        button = self._tool_button(
            self._action(
                (
                    self.tr("Toggle Frames panel")
                    if index == 0
                    else self.tr("Toggle task panel")
                ),
                lambda: self._toggle_panel(index),
                self,
                icon="panel-left" if index == 0 else "panel-right",
            ),
            layout,
        )
        button.setProperty("panelHeader", True)
        button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        button.setFixedSize(26, 26)
        return button

    def _toggle_panel(self, index):
        panel = self.splitter.widget(index)
        sizes = self.splitter.sizes()
        show = panel.isHidden()
        if show:
            sizes[index] = self._panel_widths[index]
            sizes[1] = max(1, sizes[1] - sizes[index])
        else:
            self._panel_widths[index] = sizes[index]
            sizes[1] += sizes[index]
            sizes[index] = 0
        panel.setVisible(show)
        self.splitter.setSizes(sizes)
        button = (
            self.show_frames_button
            if index == 0
            else self.show_annotation_button
        )
        button.parentWidget().setVisible(not show)

    def _toggle_camera_controls(self, visible):
        self.viewport.set_camera_controls_visible(visible)
        title = (
            self.tr("Hide camera controls")
            if visible
            else self.tr("Show camera controls")
        )
        self.camera_controls_action.setText(title)
        self.camera_controls_action.setToolTip(title)
        self.camera_controls_button.setAccessibleName(title)

    def _build_view_toolbar(self):
        toolbar = QtWidgets.QFrame()
        toolbar.setObjectName("pointcloudViewTools")
        toolbar.setFixedHeight(42)
        toolbar.setAccessibleName(self.tr("Point cloud view"))
        layout = QtWidgets.QHBoxLayout(toolbar)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(2)
        self.tool_group = QtGui.QActionGroup(self)
        self.tool_actions = {}
        for title, name, shortcut, icon in [
            (
                self.tr("Browse: left drag to rotate; right drag to pan"),
                "browse",
                "V",
                "click",
            ),
            (
                self.tr("Brush (Ctrl+wheel adjusts size)"),
                "brush",
                "B",
                "brush",
            ),
            (
                self.tr("Polygon (Ctrl+left drag pans; wheel zooms)"),
                "polygon",
                "P",
                "polygon",
            ),
        ]:
            action = self._action(
                title, self._tool_changed, toolbar, shortcut, True, icon
            )
            action.setCheckable(True)
            action.setData(name)
            self.tool_group.addAction(action)
            self.tool_actions[name] = action
            self._tool_button(action, layout)
        self.tool_actions["browse"].setChecked(True)
        self.finish_action = self._action(
            self.tr("Finish polygon selection"),
            self.viewport.finish_polygon,
            toolbar,
            "Return",
            True,
        )
        self.cancel_action = self._action(
            self.tr("Cancel unfinished selection"),
            self.viewport.cancel_selection,
            toolbar,
            "Escape",
            True,
        )
        self.undo_action = self._action(
            self.tr("Undo"),
            self.undo,
            toolbar,
            QtGui.QKeySequence.StandardKey.Undo,
            True,
            "undo",
        )
        self.redo_action = self._action(
            self.tr("Redo"),
            self.redo,
            toolbar,
            QtGui.QKeySequence.StandardKey.Redo,
            True,
            "redo",
        )
        self.fit_action = self._action(
            self.tr("Fit all points"),
            self.viewport.fit_all,
            toolbar,
            "F",
            True,
            "fit",
        )
        self._tool_button(self.fit_action, layout)
        self.view_group = QtGui.QActionGroup(self)
        self.view_actions = {}
        for title, name in [
            (self.tr("Top view"), "top"),
            (self.tr("Front view"), "front"),
            (self.tr("Side view"), "side"),
        ]:
            action = self._action(
                title,
                lambda checked=False, view=name: self.viewport.set_view(view),
                toolbar,
                icon=name,
            )
            action.setCheckable(True)
            self.view_group.addAction(action)
            self.view_actions[name] = action
            self._tool_button(action, layout)
        reset_action = self._action(
            self.tr("Reset view"),
            self.viewport.reset_view,
            toolbar,
            icon="refresh",
        )
        self._tool_button(reset_action, layout)
        for action in (
            self.undo_action,
            self.redo_action,
        ):
            self._tool_button(action, layout)
        self.operation_group = QtGui.QActionGroup(self)
        self.operation_actions = {}
        for title, name, icon in [
            (
                self.tr("Assign semantic"),
                "assign",
                "semantic",
            ),
            (self.tr("Create instance"), "create", "instance-new"),
            (self.tr("Add to current instance"), "add", "instance-add"),
            (
                self.tr("Remove from current instance"),
                "remove",
                "instance-remove",
            ),
            (self.tr("Split current instance"), "split", "instance-split"),
        ]:
            action = self._action(
                title, self._target_changed, toolbar, icon=icon
            )
            action.setCheckable(True)
            action.setData(name)
            self.operation_group.addAction(action)
            self.operation_actions[name] = action
            self._tool_button(action, layout)
        self.operation_actions["assign"].setChecked(True)
        self.through_action = self._action(
            self.tr(
                "Through selection: select all depths; off selects the surface"
            ),
            self._depth_changed,
            toolbar,
            icon="through",
        )
        self.through_action.setCheckable(True)
        self._tool_button(self.through_action, layout)
        separator = QtWidgets.QFrame()
        separator.setObjectName("pointcloudToolSeparator")
        separator.setFixedSize(1, 22)
        layout.addSpacing(6)
        layout.addWidget(separator)
        layout.addSpacing(6)
        self.color_mode = self._combo(
            [
                (self.tr("Semantic"), "semantic"),
                (self.tr("Intensity"), "intensity"),
                (self.tr("Original"), "rgb"),
                (self.tr("Instance"), "instance"),
            ]
        )
        self.color_mode.setAccessibleName(self.tr("Color"))
        self.color_mode.setParent(toolbar)
        self.color_mode.hide()
        self.render_group = QtGui.QActionGroup(self)
        self.render_actions = {}
        for index in range(self.color_mode.count()):
            mode = self.color_mode.itemData(index)
            action = self._action(
                self.tr("Rendering mode")
                + ": "
                + self.color_mode.itemText(index),
                lambda checked=False, i=index: self.color_mode.setCurrentIndex(
                    i
                ),
                toolbar,
                icon="render-" + mode,
            )
            action.setCheckable(True)
            action.setData(mode)
            self.render_group.addAction(action)
            self.render_actions[mode] = action
            self._tool_button(action, layout)
        self.color_mode.setCurrentIndex(self.color_mode.findData("semantic"))
        self.render_actions["semantic"].setChecked(True)
        self.color_mode.currentIndexChanged.connect(self._sync_render_actions)
        layout.addSpacing(6)
        self.point_size_popup = PointSizePopup(self)
        self.point_size = self.point_size_popup.slider
        self.point_size.setValue(int(self.settings.value("point_size", 2)))
        self.point_size.valueChanged.connect(self.viewport.set_point_size)
        self.viewport.set_point_size(self.point_size.value())
        self.point_size_action = self._action(
            self.tr("Point size (px)"),
            self._show_point_size,
            toolbar,
            icon="point-size",
        )
        self.point_size_button = self._tool_button(
            self.point_size_action, layout
        )
        self.legend = QtWidgets.QLabel(self)
        self.legend.hide()
        layout.addStretch()
        layout.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetMinimumSize
        )
        self.view_tool_scroll = PointCloudToolScrollArea()
        self.view_tool_scroll.setWidget(toolbar)
        return self.view_tool_scroll

    def _configure_camera(self):
        dialog = CameraConfigurationDialog(self.camera_panel, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.camera_panel.configure_sources(dialog.configuration)
        dialog.deleteLater()

    def create_task(self):
        dialog = CreateTaskDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            configuration = dialog.configuration
            self._request_frame(
                list(configuration.files),
                0,
                configuration.output_directory,
                {},
                task_configuration=configuration,
            )
        dialog.deleteLater()

    def _show_point_size(self):
        self.point_size_popup.show_below(self.point_size_button)

    def _set_render_legend(self, text):
        self.legend.setText(text)
        action = self.render_actions[self.color_mode.currentData()]
        action.setToolTip(action.text() + "\n" + text)

    def _sync_render_actions(self):
        for mode, action in self.render_actions.items():
            action.setEnabled(
                self.color_mode.model()
                .item(self.color_mode.findData(mode))
                .isEnabled()
                and (
                    not self.detection.enabled
                    or mode not in ("semantic", "instance")
                )
            )
            action.setVisible(
                not self.detection.enabled
                or mode not in ("semantic", "instance")
            )
            action.setChecked(self.color_mode.currentData() == mode)

    def _current_tool(self):
        return self.tool_group.checkedAction().data()

    def _select_tool(self, tool):
        self.tool_actions[tool].setChecked(True)
        self._tool_changed()

    def _camera_view_changed(self, view):
        self.view_group.setExclusive(False)
        for name, action in self.view_actions.items():
            action.setChecked(name == view)
        self.view_group.setExclusive(True)

    def _scroll_page(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        page = QtWidgets.QWidget()
        page.setObjectName("pointcloudPage")
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        scroll.setWidget(page)
        return scroll, layout

    def _build_annotation_panel(self):
        self.sidebar_tabs = QtWidgets.QTabWidget()
        self.sidebar_tabs.setObjectName("pointcloudTabs")
        self.sidebar_tabs.tabBar().setFixedHeight(41)
        self.sidebar_tabs.setMinimumWidth(280)
        corner = QtWidgets.QWidget()
        corner.setFixedHeight(42)
        corner_layout = QtWidgets.QHBoxLayout(corner)
        corner_layout.setContentsMargins(4, 8, 6, 8)
        self._panel_button(2, corner_layout)
        self.sidebar_tabs.setCornerWidget(corner)
        page, layout = self._scroll_page()
        self.sidebar_tabs.addTab(page, "Seg")
        self.sidebar_tabs.setTabToolTip(0, self.tr("Segmentation"))
        self._build_class_list(layout)
        self._build_instance_list(layout)
        return self.sidebar_tabs

    def _build_class_list(self, layout):
        panel = QtWidgets.QFrame()
        panel.setObjectName("pointcloudListPanel")
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 2)
        header.setSpacing(0)
        self.classes_heading = self._heading(self.tr("Classes"))
        header.addWidget(self.classes_heading, 1)
        self.classes_lock_action = self._action(
            self.tr("Lock all classes"),
            self._toggle_all_class_locks,
            self,
            icon="unlock",
        )
        button = self._tool_button(self.classes_lock_action, header)
        button.setProperty("panelHeader", True)
        button.setFixedSize(24, 24)
        button.setIconSize(QtCore.QSize(16, 16))
        self.classes_visibility_action = self._visibility_action(
            header, self.tr("Toggle all classes"), lambda: self.class_list
        )
        panel_layout.addLayout(header)
        self.class_list = self._list(
            self.tr("Classes: ID / name / full-frame points"), editable=True
        )
        self.class_list.setProperty("integratedPanel", True)
        self.class_list.allow_remove = False
        self.class_list.setMinimumHeight(130)
        self.class_list.currentItemChanged.connect(self._target_changed)
        self.class_list.itemChanged.connect(self._filter_changed)
        panel_layout.addWidget(self.class_list, 1)
        self.class_list.set_header(header)
        self.class_list.lock_requested.connect(
            lambda item: self._toggle_seg_lock(self.class_list, item)
        )
        layout.addWidget(panel, 1)

    def _build_instance_list(self, layout):
        panel = QtWidgets.QFrame()
        panel.setObjectName("pointcloudListPanel")
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 2)
        header.setSpacing(0)
        self.instances_heading = self._heading(self.tr("Instances"))
        header.addWidget(self.instances_heading, 1)
        self.locate_action = self._action(
            self.tr("Locate instance"),
            self._locate_instance,
            self,
            icon="scan-search",
        )
        self.merge_action = self._action(
            self.tr("Merge into current instance"),
            self._merge_instances,
            self,
            icon="squares-unite",
        )
        for action in (self.locate_action, self.merge_action):
            button = self._tool_button(action, header)
            button.setProperty("panelHeader", True)
            button.setFixedSize(24, 24)
            button.setIconSize(QtCore.QSize(16, 16))
        self.instances_visibility_action = self._visibility_action(
            header, self.tr("Toggle all instances"), lambda: self.instance_list
        )
        panel_layout.addLayout(header)
        self.instance_list = self._list(
            self.tr("Instances: class / ID / full-frame points"),
            editable=True,
            toggle_selection=True,
        )
        self.instance_list.setProperty("integratedPanel", True)
        self.instance_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.instance_list.setMinimumHeight(120)
        self.instance_list.currentItemChanged.connect(self._target_changed)
        self.instance_list.itemSelectionChanged.connect(self._target_changed)
        self.instance_list.itemChanged.connect(self._filter_changed)
        self.instance_list.remove_requested.connect(self._delete_instance)
        panel_layout.addWidget(self.instance_list, 1)
        self.instance_list.set_header(header)
        self.instance_list.lock_requested.connect(
            lambda item: self._toggle_seg_lock(self.instance_list, item)
        )
        layout.addWidget(panel, 1)

    def _visibility_action(self, layout, title, get_list):
        action = self._action(
            title,
            lambda: self._toggle_list_visibility(get_list()),
            self,
            icon="eye",
        )
        action.setCheckable(True)
        action.setChecked(True)
        button = self._tool_button(action, layout)
        button.setProperty("panelHeader", True)
        button.setFixedSize(24, 24)
        button.setIconSize(QtCore.QSize(16, 16))
        return action

    def _toggle_list_visibility(self, widget):
        action = (
            self.classes_visibility_action
            if widget is self.class_list
            else self.instances_visibility_action
        )
        state = (
            QtCore.Qt.CheckState.Checked
            if action.isChecked()
            else QtCore.Qt.CheckState.Unchecked
        )
        with QtCore.QSignalBlocker(widget):
            for index in range(widget.count()):
                widget.item(index).setCheckState(state)
        self._refresh_display()

    def _sync_visibility_actions(self):
        visible = any(
            self.class_list.item(index).checkState()
            == QtCore.Qt.CheckState.Checked
            for index in range(self.class_list.count())
        )
        self.classes_visibility_action.setChecked(visible)
        for action, widget in (
            (self.classes_visibility_action, self.class_list),
            (self.instances_visibility_action, self.instance_list),
        ):
            action.setEnabled(widget.count() > 0)
            action.setIcon(
                self._icon("eye" if action.isChecked() else "eye-off")
            )

    def _bind_shortcuts(self):
        for key, delta in [("A", -1), ("D", 1), ("PgUp", -1), ("PgDown", 1)]:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(
                lambda d=delta: self._navigate_shortcut(d)
            )

    def _navigate_shortcut(self, delta):
        focus = (
            self._navigation_focus
            if self._worker is not None
            else QtWidgets.QApplication.focusWidget()
        )
        if any(
            focus is widget
            or (focus is not None and widget.isAncestorOf(focus))
            for widget in (
                self.viewport,
                self.detection,
                self.file_list,
                self.camera_panel.view,
            )
        ):
            self.navigate(delta)

    def _status(self, text):
        self.statusBar().showMessage(text, 12000)

    def _error(self, text):
        QtWidgets.QMessageBox.warning(self, self.tr("Point Cloud"), text)

    def _renderer_error(self, text):
        self._status(text)
        self._select_tool("browse")
        self.tool_group.setEnabled(False)
        self.create_task_action.setEnabled(False)
        if self._worker is not None:
            self._worker.requestInterruption()

    def _confirm(self, text):
        saving = self._autosave_timer.isActive()
        self._autosave_timer.stop()
        accepted = (
            QtWidgets.QMessageBox.question(
                self,
                self.tr("Point Cloud"),
                text,
                QtWidgets.QMessageBox.StandardButton.Ok
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            == QtWidgets.QMessageBox.StandardButton.Ok
        )
        if saving:
            self._schedule_autosave()
        return accepted

    def _leave_decision(self):
        self._autosave_timer.stop()
        self.detection.cancel()
        if self._worker is not None:
            self._worker.requestInterruption()
            self._status(
                self.tr(
                    "Waiting for loading to end safely. Please retry after loading stops."
                )
            )
            return "cancel"
        if not (self.config_dirty or (self.document and self.document.dirty)):
            return "continue"
        if self._autosave():
            return "continue"
        decision = QtWidgets.QMessageBox.warning(
            self,
            self.tr("Unsaved point cloud work"),
            self.tr(
                "Labels or class definitions have unsaved changes. Save before leaving?"
            ),
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if decision == QtWidgets.QMessageBox.StandardButton.Save:
            return (
                "continue" if self.save_work(only_changed=True) else "cancel"
            )
        if decision == QtWidgets.QMessageBox.StandardButton.Discard:
            return "discard"
        return "cancel"

    def can_close(self):
        return self._leave_decision() != "cancel"

    def close_after_approval(self):
        self._close_approved = True
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self._autosave_timer.stop()
        self.close()

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("point_size", self.point_size.value())
        if not self._close_approved:
            event.ignore()
            self.hide()
            return
        event.accept()

    def _recent(self):
        return str(self.settings.value("recent_directory", ""))

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Open point cloud"),
            self._recent(),
            self.tr("Point clouds (*.bin *.ply)"),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog
            | QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons,
        )
        if path:
            path = Path(path).resolve()
            overrides = dict(self.label_overrides)
            overrides[path] = default_label_path(path)
            self._request_frame([path], 0, None, overrides)

    def open_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Open point cloud directory"),
            self._recent(),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog
            | QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons,
        )
        if not directory:
            return
        try:
            files = discover_frames(directory)
        except (OSError, ValueError) as error:
            self._error(str(error))
            return
        self.open_paths(files)

    def open_paths(self, files):
        if files:
            dataset = self._dataset_path(Path(files[0]).resolve())
            directory = self.settings.value(
                self._output_directory_key(dataset)
            )
            self._request_frame(
                list(files),
                0,
                Path(directory) if directory else None,
                self.label_overrides,
            )

    @staticmethod
    def _output_directory_key(dataset):
        return (
            "output_directories/"
            + hashlib.sha256(str(dataset).encode()).hexdigest()
        )

    @staticmethod
    def _dataset_path(path):
        return (
            path.parent.parent
            if path.parent.name == "velodyne"
            else path.parent
        )

    def _resolve_label_path(self, path, directory, overrides):
        if path in overrides:
            return overrides[path]
        candidates = label_candidates(path, directory)
        if len(candidates) > 1:
            selected, accepted = QtWidgets.QInputDialog.getItem(
                self,
                self.tr("Choose label source"),
                self.tr(
                    "Multiple matching label files exist. Choose the source for this frame:"
                ),
                [str(candidate) for candidate in candidates],
                0,
                False,
            )
            if not accepted:
                return None
            return Path(selected)
        if candidates:
            return candidates[0]
        return (
            Path(directory) / (path.stem + ".label")
            if directory
            else default_label_path(path)
        )

    def _request_frame(
        self, files, index, directory, overrides, task_configuration=None
    ):
        decision = self._leave_decision()
        if decision == "cancel":
            self._restore_list_row()
            return
        path = Path(files[index]).resolve()
        if files is not self.files:
            files = [Path(item).resolve() for item in files]
        try:
            label_path = self._resolve_label_path(path, directory, overrides)
        except (OSError, ValueError) as error:
            self._error(str(error))
            self._restore_list_row()
            return
        if label_path is None:
            self._restore_list_row()
            return
        try:
            read_path = label_path
            output_path = None
            if (
                directory
                and label_path == Path(directory) / (path.stem + ".label")
                and not label_path.exists()
                and not label_path.is_symlink()
            ):
                sources = label_candidates(path)
                if sources:
                    read_path = sources[0]
                    output_path = label_path
        except (OSError, ValueError) as error:
            self._error(str(error))
            self._restore_list_row()
            return
        dataset = self._dataset_path(path)
        config_path = (
            self._default_class_path()
            if dataset != self.dataset
            and not (self.dataset is None and self.config_path is not None)
            else None
        )
        if task_configuration is not None:
            config_path = None
        self._pending = (
            files,
            index,
            directory,
            dict(overrides),
            dataset,
            config_path,
            decision,
            task_configuration,
        )
        self._load_result = None
        self._load_error = None
        self._worker = FrameLoader(
            path, read_path, config_path, self, output_path
        )
        self._worker.loaded.connect(self._loaded)
        self._worker.failed.connect(self._load_failed)
        self._worker.finished.connect(self._load_finished)
        self._progress = QtWidgets.QProgressDialog(
            self.tr("Loading and validating point cloud…"),
            self.tr("Cancel"),
            0,
            0,
            self,
        )
        self._progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(500)
        self._progress.canceled.connect(self._worker.requestInterruption)
        self._set_loading(True)
        self._worker.start()
        self._progress.setValue(0)

    def _loaded(self, result):
        self._load_result = result

    def _load_failed(self, error):
        self._load_error = error

    def _load_finished(self):
        created_task = False
        cancelled = self._worker.isInterruptionRequested()
        self._progress.close()
        self._progress.deleteLater()
        self._progress = None
        if self._load_result is not None and not cancelled:
            frame, loaded_classes = self._load_result
            (
                files,
                index,
                directory,
                overrides,
                dataset,
                config_path,
                decision,
                task_configuration,
            ) = self._pending
            previous = self.document
            try:
                document = AnnotationDocument(frame)
                self._display_signature = None
                self.viewport.set_cloud(frame.points)
                self.viewport.validate_rendering()
            except (RuntimeError, ValueError, MemoryError) as error:
                if previous is not None:
                    self.viewport.set_cloud(previous.frame.points)
                self._load_error = str(error)
            else:
                rebuild_files = (
                    files != self.files
                    or directory != self.label_directory
                    or overrides != self.label_overrides
                )
                self.document = document
                self.files, self.frame_index = files, index
                self.label_directory, self.label_overrides = (
                    directory,
                    overrides,
                )
                self.label_overrides[frame.path] = frame.label_path
                if task_configuration is not None or (
                    dataset != self.dataset and config_path is not None
                ):
                    definitions = loaded_classes or {}
                    self.class_definitions = {
                        "detection": list(definitions.get("detection", ())),
                        "segmentation": list(
                            definitions.get("segmentation", DEFAULT_CLASSES)
                        ),
                    }
                    self._saved_classes = {
                        task: list(classes)
                        for task, classes in self.class_definitions.items()
                    }
                    self.config_path = config_path
                elif decision == "discard":
                    self.class_definitions = {
                        task: list(classes)
                        for task, classes in self._saved_classes.items()
                    }
                self.dataset = dataset
                if task_configuration is not None:
                    created_task = True
                    self.task_type = task_configuration.task
                    self.detection.clipboard = None
                    self.class_definitions[task_configuration.task] = list(
                        task_configuration.classes
                    )
                    self._saved_classes = {
                        task: list(classes)
                        for task, classes in self.class_definitions.items()
                    }
                    self.settings.setValue(
                        self._output_directory_key(dataset), str(directory)
                    )
                    selected = 0 if self.task_type == "detection" else 1
                    self.sidebar_tabs.setTabEnabled(selected, True)
                    self.sidebar_tabs.setCurrentIndex(selected)
                    if self.task_type == "segmentation":
                        self.color_mode.setCurrentIndex(
                            self.color_mode.findData("semantic")
                        )
                    self.sidebar_tabs.setTabEnabled(1 - selected, False)
                    self.sidebar_tabs.setTabToolTip(
                        selected,
                        (
                            self.tr("Detection")
                            if selected == 0
                            else self.tr("Segmentation")
                        ),
                    )
                    self.sidebar_tabs.setTabToolTip(
                        1 - selected,
                        self.tr("Create a new task to change the task type."),
                    )
                self.settings.setValue("recent_directory", str(dataset))
                self._clear_filters()
                self.instance_list.clear()
                if rebuild_files:
                    self._rebuild_files()
                self.frame_progress.setText(
                    f"{self.frame_index + 1}/{len(self.files)}"
                )
                self.viewport.fit_all()
                self._status(
                    self.tr("Loaded {name}: {count} points.").format(
                        name=frame.path.name, count=len(frame.points)
                    )
                )
                if frame.warnings:
                    self._error("\n".join(frame.warnings))
        if self._load_error:
            self._error(self._load_error)
        elif cancelled:
            self._status(
                self.tr("Loading cancelled. Previous work has been retained.")
            )
        queued = self._queued_frame
        self._queued_frame = None
        self._pending = None
        self._load_result = None
        self._worker.deleteLater()
        self._worker = None
        self._set_loading(False)
        self._restore_list_row()
        self._refresh()
        if created_task:
            self.camera_panel.configure_sources(task_configuration.cameras)
            self._schedule_autosave()
        if queued is not None and not cancelled and not self._load_error:
            self._select_frame(queued)

    def _set_loading(self, loading):
        if loading:
            self._navigation_focus = QtWidgets.QApplication.focusWidget()
        for action in [
            self.create_task_action,
            self.save_as_action,
            self.export_action,
            self.import_action,
            self.undo_action,
            self.redo_action,
        ]:
            action.setEnabled(not loading)
        self.centralWidget().setEnabled(not loading)
        if not loading and self._navigation_focus is not None:
            self._navigation_focus.setFocus()
            self._navigation_focus = None
        for toolbar in self.findChildren(QtWidgets.QToolBar):
            toolbar.setEnabled(not loading)
        if self.viewport._error:
            self.create_task_action.setEnabled(False)

    def _restore_list_row(self):
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(self.frame_index)
        self.file_list.blockSignals(False)

    def _frame_label_path(self, path):
        if self.document is not None and path == self.document.frame.path:
            return self.document.frame.label_path
        if path in self.label_overrides:
            return self.label_overrides[path]
        if self.label_directory is not None:
            return self.label_directory / (path.stem + ".label")
        candidates = label_candidates(path)
        return candidates[0] if candidates else default_label_path(path)

    def _export_dataset(self):
        if (
            self.document is None
            or self._worker is not None
            or self.task_type == "segmentation"
        ):
            return
        self._autosave_timer.stop()
        self.detection.cancel()
        dialog = None
        try:
            frame = self.document.frame
            frames = [
                ExportFrame(
                    path,
                    cuboid_source_path(path, self._frame_label_path(path)),
                    (
                        tuple(self.document.cuboids)
                        if path == frame.path
                        else None
                    ),
                    self.camera_panel.files.get(path.stem),
                )
                for path in self.files
            ]
            dialog = PointCloudExportDialog(
                frames, self.class_definitions["detection"], self
            )
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                count, objects = dialog.export_result
                self.statusBar().showMessage(
                    self.tr(
                        "Exported {frames} frames and {objects} objects to {path}"
                    ).format(
                        frames=count, objects=objects, path=dialog.output_path
                    ),
                    12000,
                )
        except (OSError, ValueError) as error:
            self._error(str(error))
        finally:
            if dialog is not None:
                dialog.deleteLater()
            self._schedule_autosave()

    def _import_dataset(self):
        if (
            self.document is None
            or self._worker is not None
            or self.task_type == "segmentation"
        ):
            return
        self._autosave_timer.stop()
        self.detection.cancel()
        dialog = None
        try:
            frame = self.document.frame
            targets = []
            for path in self.files:
                label_path = self._frame_label_path(path)
                target = (
                    frame.cuboid_path
                    or label_path.with_suffix(".cuboids.json")
                    if path == frame.path
                    else label_path.with_suffix(".cuboids.json")
                )
                targets.append(
                    ImportTarget(
                        path,
                        label_path,
                        target,
                        (
                            tuple(self.document.cuboids)
                            if path == frame.path
                            else None
                        ),
                    )
                )
            dialog = PointCloudImportDialog(
                targets, self.class_definitions["detection"], self
            )
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            plan = dialog.import_plan
            self.class_definitions["detection"] = list(plan.classes)
            self._saved_classes = {
                task: list(classes)
                for task, classes in self.class_definitions.items()
            }
            self.config_path = (
                Path(dialog.config_path)
                if dialog.config_path is not None
                else None
            )
            for target in plan.targets:
                if target.path == frame.path:
                    self.document._commit_cuboids(target.cuboids)
                    self.document.mark_cuboids_saved(target.cuboid_path)
            self.detection.hidden_ids.clear()
            self.detection.select(None)
            self._rebuild_files()
            self.sidebar_tabs.setCurrentWidget(self.detection.panel)
            self._refresh()
            self._status(
                self.tr(
                    "Imported {objects} objects into {frames} frames."
                ).format(objects=plan.object_count, frames=len(plan.targets))
            )
        except (OSError, ValueError) as error:
            self._error(str(error))
        finally:
            if dialog is not None:
                dialog.deleteLater()
            self._schedule_autosave()

    def _review_signature(self, path, target):
        cuboid_path = target.with_suffix(".cuboids.json")
        if not path.exists() or not (target.exists() or cuboid_path.exists()):
            return ""
        source_stat = path.stat()
        signature = f"{source_stat.st_size}:{source_stat.st_mtime_ns}"
        for member in (target, cuboid_path):
            if member.exists():
                stat = member.stat()
                signature += f":{stat.st_size}:{stat.st_mtime_ns}"
        return signature

    def _review_key(self, target):
        return "reviewed/" + hashlib.sha256(str(target).encode()).hexdigest()

    def _update_file_item(self, index):
        path = self.files[index]
        item = self.file_list.item(index)
        target = self._frame_label_path(path)
        item.setText(path.name)
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
        )
        item.setCheckState(
            QtCore.Qt.CheckState.Checked
            if target.exists() or target.with_suffix(".cuboids.json").exists()
            else QtCore.Qt.CheckState.Unchecked
        )
        signature = self._review_signature(path, target)
        reviewed = (
            bool(signature)
            and self.settings.value(self._review_key(target)) == signature
        )
        if (
            self.document is not None
            and path == self.document.frame.path
            and self.document.dirty
        ):
            reviewed = False
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        color = QtGui.QColor(
            get_theme()["primary"]
            if reviewed
            else get_theme()["text_secondary"]
        )
        painter.setPen(QtGui.QPen(color, 2))
        painter.setBrush(color if reviewed else QtCore.Qt.BrushStyle.NoBrush)
        painter.drawEllipse(1, 1, 14, 14)
        painter.end()
        item.setIcon(QtGui.QIcon(pixmap))
        item.setData(QtCore.Qt.ItemDataRole.UserRole, reviewed)
        item.setToolTip(
            str(path)
            + "\n"
            + str(target)
            + "\n"
            + str(target.with_suffix(".cuboids.json"))
            + "\n"
            + (self.tr("Reviewed") if reviewed else self.tr("Not reviewed"))
        )

    def _rebuild_files(self):
        self.frame_progress.setText(
            f"{self.frame_index + 1}/{len(self.files)}"
        )
        with QtCore.QSignalBlocker(self.file_list):
            self.file_list.clear()
            for index, path in enumerate(self.files):
                self.file_list.addItem(path.name)
                self._update_file_item(index)
            self.file_list.setCurrentRow(self.frame_index)

    def _file_context_menu(self, position):
        item = self.file_list.itemAt(position)
        if item is None:
            return
        menu = QtWidgets.QMenu(self)
        title = (
            self.tr("Mark as not reviewed")
            if item.data(QtCore.Qt.ItemDataRole.UserRole)
            else self.tr("Mark as reviewed")
        )
        action = menu.addAction(title)
        action.triggered.connect(
            lambda: self._toggle_reviewed(self.file_list.row(item))
        )
        menu.exec(self.file_list.viewport().mapToGlobal(position))

    def _toggle_reviewed(self, index):
        path = self.files[index]
        target = self._frame_label_path(path)
        reviewed = self.file_list.item(index).data(
            QtCore.Qt.ItemDataRole.UserRole
        )
        if not reviewed and index == self.frame_index:
            if not self.save_work():
                return
            target = self._frame_label_path(path)
        signature = self._review_signature(path, target)
        if not reviewed and not signature:
            self._status(
                self.tr("Open this frame before marking it as reviewed.")
            )
            return
        self.settings.setValue(
            self._review_key(target), "" if reviewed else signature
        )
        self._update_file_item(index)

    def _select_frame(self, index):
        if self._worker is not None:
            if self._pending[0] == self.files and 0 <= index < len(self.files):
                self._queued_frame = index
            return
        if 0 <= index < len(self.files) and index != self.frame_index:
            self._request_frame(
                self.files, index, self.label_directory, self.label_overrides
            )

    def navigate(self, delta):
        current = self.frame_index
        if self._worker is not None:
            if self._pending[0] != self.files:
                return
            current = (
                self._queued_frame
                if self._queued_frame is not None
                else self._pending[1]
            )
        index = current + delta
        if 0 <= index < len(self.files):
            self._select_frame(index)

    def reload_frame(self):
        if self.document:
            overrides = dict(self.label_overrides)
            overrides[self.document.frame.path] = (
                self.document.frame.label_path
            )
            self._request_frame(
                self.files, self.frame_index, self.label_directory, overrides
            )

    def _choose_label_file(self):
        if not self.document:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Choose labels for current frame"),
            str(self.document.frame.label_path),
            self.tr("Point labels (*.label)"),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog
            | QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons,
        )
        if not path:
            return
        path = Path(path).resolve()
        if path.stem != self.document.frame.path.stem and not self._confirm(
            self.tr(
                "Bind {label} to {cloud}? Their file names differ."
            ).format(label=path, cloud=self.document.frame.path)
        ):
            return
        overrides = dict(self.label_overrides)
        overrides[self.document.frame.path] = path
        self._request_frame(
            self.files, self.frame_index, self.label_directory, overrides
        )

    def _choose_label_directory(self):
        if not self.document:
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Choose label directory"),
            str(self.document.frame.label_path.parent),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog
            | QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons,
        )
        if not directory:
            return
        if self.label_overrides and not self._confirm(
            self.tr(
                "Use this label directory for the entire sequence? This replaces {count} explicit frame associations."
            ).format(count=len(self.label_overrides))
        ):
            return
        self._request_frame(
            self.files, self.frame_index, Path(directory).resolve(), {}
        )

    def _change_output_directory(self):
        if self.document is None:
            return
        self._autosave_timer.stop()
        self.detection.cancel()
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Change output directory"),
            str(self.label_directory or self.document.frame.label_path.parent),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog
            | QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons,
        )
        if not directory:
            self._schedule_autosave()
            return
        directory = Path(directory).resolve()
        target = directory / (self.document.frame.path.stem + ".label")
        if not self._save_labels(target):
            return
        self.label_directory = directory
        self.settings.setValue(
            self._output_directory_key(self.dataset), str(directory)
        )
        self.label_overrides = {
            path: label
            for path, label in self.label_overrides.items()
            if path not in self.files
        }
        self.label_overrides[self.document.frame.path] = target
        self._rebuild_files()
        self._refresh_save_state()
        self._schedule_autosave()

    def _schedule_autosave(self):
        if self.config_dirty or (
            self.document is not None and self.document.dirty
        ):
            self._autosave_timer.start()

    def _autosave(self):
        if self._worker is not None:
            return False
        if self.detection.active:
            self._autosave_timer.start()
            return False
        return self.save_work(only_changed=True)

    def _save_labels(self, path, overwrite_confirmed=False):
        path = Path(path)
        frame = self.document.frame
        if (
            not overwrite_confirmed
            and path.exists()
            and (path != frame.label_path or not frame.label_exists)
        ):
            if not self._confirm(
                self.tr("Replace the existing label file at {path}?").format(
                    path=path
                )
            ):
                return False
        cuboid_path = path.with_suffix(".cuboids.json")
        save_boxes = bool(
            self.document.cuboids
            or frame.cuboid_exists
            or self.document.cuboids_dirty
        )
        if save_boxes and not self._confirm_cuboid_output(cuboid_path):
            return False
        try:
            save_labels(path, self.document.labels, self.document.frame.path)
        except (OSError, ValueError, MemoryError) as error:
            self._error(
                self.tr("Could not save labels to {path}: {error}").format(
                    path=path, error=error
                )
            )
            return False
        if save_boxes and not self._save_cuboids(cuboid_path, confirmed=True):
            return False
        self.document.mark_saved(path)
        frame.cuboid_path = cuboid_path
        self.label_overrides[self.document.frame.path] = Path(path)
        self._refresh_save_state()
        self._refresh_display()
        self._status(
            self.tr("Saved {count} labels to {path}.").format(
                count=len(self.document.labels), path=path
            )
        )
        return True

    def _confirm_cuboid_output(self, path):
        frame = self.document.frame
        return not (
            path.exists()
            and (path != frame.cuboid_path or not frame.cuboid_exists)
        ) or self._confirm(
            self.tr("Replace the existing cuboid file at {path}?").format(
                path=path
            )
        )

    def _save_cuboids(self, path=None, confirmed=False):
        frame = self.document.frame
        path = Path(
            path
            or frame.cuboid_path
            or frame.label_path.with_suffix(".cuboids.json")
        )
        if not confirmed and not self._confirm_cuboid_output(path):
            return False
        try:
            save_cuboids(path, self.document.cuboids, frame.path)
        except (OSError, ValueError, MemoryError) as error:
            self._error(
                self.tr("Could not save cuboids to {path}: {error}").format(
                    path=path, error=error
                )
            )
            return False
        self.document.mark_cuboids_saved(path)
        self._refresh_save_state()
        self._refresh_display()
        self._status(
            self.tr("Saved {count} cuboids to {path}.").format(
                count=len(self.document.cuboids), path=path
            )
        )
        return True

    def save_work(self, only_changed=False):
        self._autosave_timer.stop()
        self.detection.cancel()
        labels_saved = False
        if self.document and (not only_changed or self.document.labels_dirty):
            if not self._save_labels(self.document.frame.label_path):
                return False
            labels_saved = True
        if self.document and self.document.cuboids_dirty:
            if not self._save_cuboids():
                return False
        if self.config_dirty and not self._save_config():
            if labels_saved:
                self._status(
                    self.tr(
                        "Label save completed; class definitions remain unsaved."
                    )
                )
            return False
        return True

    def save_as(self):
        if not self.document:
            return False
        self._autosave_timer.stop()
        self.detection.cancel()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Save complete frame labels"),
            str(self.document.frame.label_path),
            self.tr("Point labels (*.label)"),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog
            | QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons,
        )
        if not path:
            self._schedule_autosave()
            return False
        path = Path(path).resolve()
        if path.stem != self.document.frame.path.stem and not self._confirm(
            self.tr(
                "Save labels for {cloud} as {label}? Keep this association when reopening."
            ).format(cloud=self.document.frame.path.name, label=path.name)
        ):
            self._schedule_autosave()
            return False
        if not self._save_labels(path, overwrite_confirmed=True):
            return False
        self._status(
            self.tr("Labels: {labels}. Class definitions: {config}.").format(
                labels=path,
                config=self.config_path or self.tr("Unlabeled only"),
            )
        )
        return True

    @staticmethod
    def _default_class_path():
        return (
            Path(get_work_directory())
            / "xanylabeling_data"
            / "pointcloud"
            / "pointcloud_classes.json"
        )

    def _save_config(self, choose_path=False):
        if choose_path and self.task_type is not None:
            return False
        task = self._class_task()
        path = self.config_path
        if path is None:
            path = self._default_class_path()
        if choose_path:
            selected, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                self.tr("Save class definitions"),
                str(path or Path(self._recent()) / "pointcloud_classes.json"),
                self.tr("Class definitions (*.json)"),
                options=QtWidgets.QFileDialog.Option.DontUseNativeDialog
                | QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons,
            )
            if not selected:
                return False
            path = Path(selected).resolve()
        try:
            if self.document and (
                path.resolve() == self.document.frame.path.resolve()
                or path.resolve() == self.document.frame.label_path.resolve()
            ):
                raise ValueError(
                    self.tr("Class definitions must use a separate JSON file.")
                )
            definitions = self.class_definitions
            if choose_path:
                definitions = load_classes(path) if path.exists() else {}
                definitions[task] = self.class_definitions[task]
            save_classes(path, definitions)
        except (OSError, ValueError, MemoryError) as error:
            self._error(
                self.tr(
                    "Could not save class definitions to {path}: {error}"
                ).format(path=path, error=error)
            )
            return False
        if not choose_path:
            self.config_path = path
            self._saved_classes = {
                task: list(classes)
                for task, classes in self.class_definitions.items()
            }
        elif (
            path.resolve()
            == (self.config_path or self._default_class_path()).resolve()
        ):
            self.config_path = path
            self._saved_classes[task] = list(self.class_definitions[task])
        self._refresh()
        self._status(
            self.tr("Class definitions saved to {path}.").format(path=path)
        )
        return True

    def _import_classes(self):
        if self.task_type is not None:
            return
        task = self._class_task()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Load class definitions"),
            self._recent(),
            self.tr("Class definitions (*.json)"),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog
            | QtWidgets.QFileDialog.Option.DontUseCustomDirectoryIcons,
        )
        if not path:
            return
        try:
            classes = load_classes(path)
            if task not in classes:
                raise ValueError(
                    self.tr(
                        "This file does not contain classes for the current task."
                    )
                )
        except (OSError, ValueError) as error:
            self._error(str(error))
            return
        if self.class_definitions[task] != self._saved_classes[
            task
        ] and not self._confirm(
            self.tr(
                "Replace the current task's unsaved class definitions with this file? Annotation IDs and the other task's classes are unchanged."
            )
        ):
            return
        self.viewport.cancel_selection()
        self.class_definitions[task] = list(classes[task])
        self._refresh()
        self._schedule_autosave()

    def _current_class(self):
        item = self.class_list.currentItem()
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None

    def _current_instance(self):
        item = self.instance_list.currentItem()
        return (
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            if item is not None and item.isSelected()
            else None
        )

    def _class_name(self, semantic_id, task="segmentation"):
        for definition in self.class_definitions[task]:
            if definition.id == semantic_id:
                return definition.name
        return "Unknown"

    def _next_class_name(self):
        names = {
            item.name for item in self.class_definitions[self._class_task()]
        }
        name = "Unknown"
        suffix = 0
        while name in names:
            suffix += 1
            name = f"Unknown({suffix})"
        return name

    def _next_class_color(self):
        used = {
            QtGui.QColor(item.color).name()
            for item in self.class_definitions[self._class_task()]
        }
        for rgb in label_colormap(33)[1:]:
            color = QtGui.QColor(*(int(value) for value in rgb)).name()
            if color not in used:
                return color
        value = 0x60A5FA
        while QtGui.QColor.fromRgb(value).name() in used:
            value = (value + 0x9E3779) & 0xFFFFFF
        return QtGui.QColor.fromRgb(value).name()

    def _edit_class(self, editing):
        if self.task_type is not None:
            return
        task = self._class_task()
        listing = (
            self.detection.labels
            if self.detection.enabled
            else self.class_list
        )
        current = listing.currentItem()
        semantic_id = (
            current.data(QtCore.Qt.ItemDataRole.UserRole)
            if editing and current
            else None
        )
        if editing and semantic_id is None:
            return
        definition = next(
            (
                item
                for item in self.class_definitions[task]
                if item.id == semantic_id
            ),
            None,
        )
        if editing and definition is None:
            definition = ClassDefinition(
                semantic_id, self._class_name(semantic_id, task), "#60A5FA"
            )
        used = {
            listing.item(index).data(QtCore.Qt.ItemDataRole.UserRole)
            for index in range(listing.count())
        }
        suggested = next(
            (value for value in range(1, 65536) if value not in used), 65535
        )
        dialog = ClassDefinitionDialog(
            definition, used, suggested, self, task=task
        )
        if definition is None:
            dialog.name_input.setText(self._next_class_name())
            dialog.color_input.setText(self._next_class_color())
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.viewport.cancel_selection()
        updated = dialog.definition()
        self.class_definitions[task] = sorted(
            [
                item
                for item in self.class_definitions[task]
                if item.id != updated.id
            ]
            + [updated],
            key=lambda item: item.id,
        )
        self._refresh()
        for index in range(listing.count()):
            item = listing.item(index)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == updated.id:
                listing.setCurrentItem(item)
                break
        self._schedule_autosave()

    def _remove_class(self, item=None):
        if self.task_type is not None:
            return
        semantic_id = (
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            if item is not None
            else self._current_class()
        )
        if semantic_id is None:
            return
        if self.document and (
            semantic_id in self.document.locked_classes
            or any(
                key[0] == semantic_id for key in self.document.locked_instances
            )
        ):
            self._status(
                self.tr(
                    "Unlock the class and its instances before deleting it."
                )
            )
            return
        if semantic_id == 0:
            self._error(
                self.tr(
                    "ID 0 is reserved for unlabeled points and cannot be removed."
                )
            )
            return
        count = (
            int(np.count_nonzero(self.document.semantic == semantic_id))
            if self.document
            else 0
        )
        if not self._confirm(
            self.tr(
                "Delete this class and clear semantic and instance labels from {count} points in the current frame? Other frames are unchanged. Point label changes can be undone."
            ).format(count=count)
        ):
            return
        self.viewport.cancel_selection()
        if self.document is not None:
            self.document.clear(
                np.flatnonzero(self.document.semantic_view == semantic_id)
            )
        self.class_definitions["segmentation"] = [
            item
            for item in self.class_definitions["segmentation"]
            if item.id != semantic_id
        ]
        self._refresh()
        self.class_list.setCurrentRow(-1)
        self._schedule_autosave()

    def _tool_changed(self):
        self.viewport.cancel_selection()
        if hasattr(self, "detection"):
            self.detection.cancel()
        tool = self._current_tool()
        self.viewport.set_tool(tool)
        if tool != "browse" and self._current_class() is None:
            self._status(
                self.tr("Create or select a class in Classes before painting.")
            )
        self._update_selection_actions()

    def _update_selection_actions(self, *args):
        active = self.viewport.selection_active
        self.finish_action.setEnabled(
            active and self._current_tool() == "polygon"
        )
        self.cancel_action.setEnabled(
            active or bool(len(self.viewport._preview))
        )

    def _update_instance_actions(self):
        if self._refreshing:
            return
        selected = (
            self.document is not None and self._current_instance() is not None
        )
        for name, action in self.operation_actions.items():
            enabled = self.document is not None and not self.detection.enabled
            if enabled and name in ("assign", "create"):
                enabled = (
                    self._current_class() not in self.document.locked_classes
                )
            if name == "create":
                enabled = enabled and self._current_class() not in (None, 0)
            elif name in ("add", "remove", "split"):
                enabled = (
                    selected
                    and not self.detection.enabled
                    and not self.document.instance_locked(
                        self._current_instance()
                    )
                )
            action.setEnabled(enabled)
            if not enabled and action.isChecked():
                self.operation_actions["assign"].setChecked(True)
        self.locate_action.setEnabled(selected)
        self.merge_action.setEnabled(
            selected
            and len(self.instance_list.selectedItems()) > 1
            and not any(
                self.document.instance_locked(
                    item.data(QtCore.Qt.ItemDataRole.UserRole)
                )
                for item in self.instance_list.selectedItems()
            )
        )

    def _depth_changed(self):
        self.viewport.cancel_selection()
        self.viewport.set_depth_mode(
            "through" if self.through_action.isChecked() else "surface"
        )
        if self.through_action.isChecked():
            self._status(
                self.tr(
                    "Through selection includes visible points at all depths. Hidden points remain excluded."
                )
            )

    def _target_changed(self, *args):
        if self._refreshing:
            return
        self.viewport.cancel_selection()
        self._refresh_target()
        self._refresh_display()

    def _refresh_target(self):
        self._update_instance_actions()
        self._update_selection_actions()
        if hasattr(self, "detection"):
            self.detection.instance_action.setEnabled(
                self.detection.enabled
                and self.document is not None
                and self._current_instance() is not None
            )

    def _apply_selection(self, indices):
        if (
            not self.document
            or self._worker is not None
            or self.detection.enabled
        ):
            return
        indices = np.asarray(indices, dtype=np.int64)
        indices = indices[self._visible[indices]]
        doc = self.document
        indices = doc.editable_indices(indices)
        if not len(indices):
            self._status(self.tr("No editable points selected."))
            return
        operation = self.operation_group.checkedAction().data()
        semantic_id = self._current_class()
        key = self._current_instance()
        before = doc.labels[indices].copy()
        selected_key = None
        excluded = 0
        try:
            if operation == "assign":
                if semantic_id is None:
                    raise ValueError(
                        self.tr("Select a class before painting.")
                    )
                if semantic_id == 0:
                    doc.clear(indices)
                else:
                    doc.assign_semantic(indices, semantic_id, overwrite=True)
            elif operation in ("create", "add"):
                target_class = (
                    semantic_id
                    if operation == "create"
                    else (key[0] if key else None)
                )
                if target_class in (None, 0):
                    raise ValueError(
                        self.tr(
                            "Select a nonzero class or a valid target instance first."
                        )
                    )
                effective = indices[doc.semantic_view[indices] == target_class]
                transferred = effective[doc.instance_view[effective] != 0]
                if operation == "add":
                    transferred = transferred[
                        doc.instance_view[transferred] != key[1]
                    ]
                if len(transferred) and not self._confirm(
                    self.tr(
                        "Transfer {count} points from their existing instances? Other semantic classes are excluded."
                    ).format(count=len(transferred))
                ):
                    return
                if operation == "create":
                    selected_key = doc.create_instance(indices, target_class)
                else:
                    doc.add_to_instance(indices, key)
                    selected_key = key
                excluded = len(indices) - len(effective)
            elif operation in ("remove", "split"):
                if key is None:
                    raise ValueError(self.tr("Select an instance first."))
                if operation == "remove":
                    doc.remove_from_instance(indices, key)
                else:
                    selected_key = doc.split_instance(indices, key)
        except ValueError as error:
            self._error(str(error))
            return
        changed = int(np.count_nonzero(before != doc.labels[indices]))
        self._refresh(selected_key)
        message = self.tr("Modified {count} points.").format(count=changed)
        if excluded:
            message += " " + self.tr(
                "Excluded {count} points belonging to other semantic classes."
            ).format(count=excluded)
        self._status(message)
        self._schedule_autosave()

    def _instance_indices(self, key):
        return np.flatnonzero(
            (self.document.semantic_view == key[0])
            & (self.document.instance_view == key[1])
        )

    def _delete_instance(self, item=None):
        key = (
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            if item is not None
            else self._current_instance()
        )
        if not self.document or key is None:
            return
        if self.document.instance_locked(key):
            self._status(
                self.tr("Unlock the class or instance before deleting it.")
            )
            return
        indices = self._instance_indices(key)
        hidden = int(np.count_nonzero(~self._visible[indices]))
        if not self._confirm(
            self.tr(
                "Delete instance {key} from all {count} points, including {hidden} hidden points? Semantic labels are retained."
            ).format(key=key, count=len(indices), hidden=hidden)
        ):
            return
        self.viewport.cancel_selection()
        self.document.delete_instance(key)
        self._refresh()
        self._schedule_autosave()

    def _merge_instances(self):
        target = self._current_instance()
        keys = [
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in self.instance_list.selectedItems()
        ]
        sources = [key for key in keys if key != target]
        if not self.document or target is None or not sources:
            self._error(
                self.tr(
                    "Select at least two instances. The current row is the retained target."
                )
            )
            return
        if any(key[0] != target[0] for key in sources):
            self._error(
                self.tr(
                    "Instances from different semantic classes cannot be merged."
                )
            )
            return
        if any(self.document.instance_locked(key) for key in keys):
            self._status(
                self.tr("Unlock the selected instances before merging.")
            )
            return
        indices = np.concatenate(
            [self._instance_indices(key) for key in sources]
        )
        hidden = int(np.count_nonzero(~self._visible[indices]))
        if not self._confirm(
            self.tr(
                "Merge {count} points, including {hidden} hidden points, into instance {target}? The target ID is retained."
            ).format(count=len(indices), hidden=hidden, target=target)
        ):
            return
        self.viewport.cancel_selection()
        try:
            self.document.merge_instances(target, sources)
        except ValueError as error:
            self._error(str(error))
            return
        self._refresh(target)
        self._schedule_autosave()

    def _locate_instance(self):
        key = self._current_instance()
        if not self.document or key is None:
            return
        indices = self._instance_indices(key)
        if not np.all(self._visible[indices]):
            if not self._confirm(
                self.tr(
                    "Adjust display filters to reveal the entire current instance?"
                )
            ):
                return
            self._clear_filters()
            self._refresh_display()
        self.viewport.focus_indices(indices)
        self._update_selection_actions()

    def undo(self):
        self.detection.cancel()
        if self.document and self.document.undo():
            self._refresh()
            self._schedule_autosave()

    def redo(self):
        self.detection.cancel()
        if self.document and self.document.redo():
            self._refresh()
            self._schedule_autosave()

    def _clear_filters(self):
        self.instances_visibility_action.setChecked(True)
        self._refreshing = True
        for index in range(self.instance_list.count()):
            self.instance_list.item(index).setCheckState(
                QtCore.Qt.CheckState.Unchecked
            )
        for index in range(self.class_list.count()):
            self.class_list.item(index).setCheckState(
                QtCore.Qt.CheckState.Checked
            )
        self._refreshing = False

    def _restore_all(self):
        self.viewport.cancel_selection()
        if self.detection.enabled:
            self.detection._set_visible(set(self.detection.hidden_ids), True)
            return
        self._clear_filters()
        self._refresh_display()

    def _filter_changed(self, *args):
        if not self._refreshing:
            if self.sender() is self.instance_list:
                self.instances_visibility_action.setChecked(True)
            self.viewport.cancel_selection()
            self._refresh_display()

    def _toggle_seg_lock(self, listing, item):
        if self.document is None:
            return
        value = item.data(QtCore.Qt.ItemDataRole.UserRole)
        classes = listing is self.class_list
        if not classes and value[0] in self.document.locked_classes:
            self._status(
                self.tr("Unlock the class before unlocking its instances.")
            )
            return
        self.viewport.cancel_selection()
        locked = (
            self.document.locked_classes
            if classes
            else self.document.locked_instances
        )
        if value in locked:
            locked.remove(value)
        else:
            locked.add(value)
        self._refresh()

    def _toggle_all_class_locks(self):
        if self.document is None:
            return
        ids = {
            self.class_list.item(index).data(QtCore.Qt.ItemDataRole.UserRole)
            for index in range(self.class_list.count())
        }
        self.viewport.cancel_selection()
        locked = self.document.locked_classes
        if ids <= locked:
            locked.difference_update(ids)
        else:
            locked.update(ids)
        self._refresh()

    def _set_seg_item_lock(self, item, value, instance=False):
        doc = self.document
        locked = bool(
            doc
            and (
                doc.instance_locked(value)
                if instance
                else value in doc.locked_classes
            )
        )
        removable = not locked and (instance or value != 0)
        if not instance and doc:
            removable &= not any(
                key[0] == value for key in doc.locked_instances
            )
        item.setData(PointCloudListWidget.LOCKED_ROLE, locked)
        item.setData(PointCloudListWidget.REMOVABLE_ROLE, removable)

    def _refresh(self, selected_instance=None):
        self._refreshing = True
        current_class = self._current_class()
        current_instance = selected_instance or self._current_instance()
        focused_instances = {
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in (
                self.instance_list.item(index)
                for index in range(self.instance_list.count())
            )
            if item.checkState() == QtCore.Qt.CheckState.Checked
        }
        selected_instances = (
            {selected_instance}
            if selected_instance is not None
            else {
                item.data(QtCore.Qt.ItemDataRole.UserRole)
                for item in self.instance_list.selectedItems()
            }
        )
        hidden = {
            self.class_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
            for i in range(self.class_list.count())
            if self.class_list.item(i).checkState()
            == QtCore.Qt.CheckState.Unchecked
        }
        counts = (
            self.document.semantic_counts()
            if self.document
            else np.zeros(65536, dtype=np.int64)
        )
        ids = sorted(
            {item.id for item in self.class_definitions["segmentation"]}
            | set(np.flatnonzero(counts))
        )
        self.class_list.clear()
        colors = {
            item.id: item.color
            for item in self.class_definitions["segmentation"]
        }
        for semantic_id in ids:
            item = QtWidgets.QListWidgetItem(
                f"{self._class_name(semantic_id)} ({semantic_id}) · {counts[semantic_id]:,}"
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, int(semantic_id))
            item.setFlags(
                item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                QtCore.Qt.CheckState.Unchecked
                if semantic_id in hidden
                else QtCore.Qt.CheckState.Checked
            )
            color = (
                QtGui.QColor(colors[semantic_id])
                if semantic_id in colors
                else QtGui.QColor(
                    int((semantic_id * 37 + 83) % 191 + 64),
                    int((semantic_id * 73 + 41) % 191 + 64),
                    int((semantic_id * 109 + 19) % 191 + 64),
                )
            )
            item.setBackground(color)
            self._set_seg_item_lock(item, int(semantic_id))
            self.class_list.addItem(item)
            if semantic_id == current_class:
                self.class_list.setCurrentItem(item)
        if current_class is None and self.class_list.count():
            self.class_list.setCurrentRow(0)
        self.instance_list.clear()
        if self.document:
            for key, count in sorted(self.document.instance_counts().items()):
                item = QtWidgets.QListWidgetItem(
                    f"{self._class_name(key[0])} · #{key[1]} · {count:,}"
                )
                item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
                self._set_seg_item_lock(item, key, instance=True)
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if key in focused_instances
                    else QtCore.Qt.CheckState.Unchecked
                )
                code = (key[1] << 16) | key[0]
                item.setBackground(
                    QtGui.QColor(
                        (code * 37 + 83) % 191 + 64,
                        (code * 73 + 41) % 191 + 64,
                        (code * 109 + 19) % 191 + 64,
                    )
                )
                self.instance_list.addItem(item)
                if key == current_instance:
                    self.instance_list.setCurrentItem(
                        item, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate
                    )
                item.setSelected(key in selected_instances)
        self.classes_heading.setText(
            f"{self.tr('Classes')} ({len(ids) - (0 in ids)})"
        )
        all_locked = bool(
            self.document and ids and set(ids) <= self.document.locked_classes
        )
        self.classes_lock_action.setEnabled(
            self.document is not None and bool(ids)
        )
        self.classes_lock_action.setIcon(
            self._icon("lock" if all_locked else "unlock")
        )
        title = (
            self.tr("Unlock all classes")
            if all_locked
            else self.tr("Lock all classes")
        )
        self.classes_lock_action.setText(title)
        self.classes_lock_action.setToolTip(title)
        self.instances_heading.setText(
            f"{self.tr('Instances')} ({self.instance_list.count()})"
        )
        self._refreshing = False
        self._refresh_target()
        self._refresh_save_state()
        self._refresh_display()
        self.detection.refresh()

    def _refresh_save_state(self):
        loaded = self.document is not None
        self.save_as_action.setEnabled(loaded)
        self.export_action.setEnabled(
            loaded and self.task_type != "segmentation"
        )
        self.import_action.setEnabled(
            loaded and self.task_type != "segmentation"
        )
        self.undo_action.setEnabled(loaded and self.document.can_undo)
        self.redo_action.setEnabled(loaded and self.document.can_redo)
        if loaded and self.file_list.count():
            self._update_file_item(self.frame_index)
        self._refresh_window_title()

    def _refresh_window_title(self):
        title = self.tr("Point Cloud")
        suffix = (
            " *"
            if self.config_dirty
            or (self.document is not None and self.document.dirty)
            else ""
        )
        if self.document is not None:
            title += " · "
            path = str(self.document.frame.path.parent)
            metrics = QtGui.QFontMetrics(
                QtWidgets.QApplication.font("QTitleBar")
            )
            width = int(self.width() * 0.9)
            if metrics.horizontalAdvance(title + path + suffix) > width:
                low, high = 0, len(path)
                while low < high:
                    count = (low + high + 1) // 2
                    left, right = (count + 1) // 2, count // 2
                    candidate = (
                        path[:left]
                        + "../../"
                        + (path[-right:] if right else "")
                    )
                    if (
                        metrics.horizontalAdvance(title + candidate + suffix)
                        <= width
                    ):
                        low = count
                    else:
                        high = count - 1
                left, right = (low + 1) // 2, low // 2
                path = (
                    path[:left] + "../../" + (path[-right:] if right else "")
                )
            title += path
        self.setWindowTitle(title + suffix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_window_title()

    def _class_palette(self):
        definitions = tuple(self.class_definitions["segmentation"])
        if self._palette_classes != definitions:
            ids = np.arange(65536, dtype=np.uint32)
            self._palette = (
                np.column_stack(
                    (
                        (ids * 37 + 83) % 191 + 64,
                        (ids * 73 + 41) % 191 + 64,
                        (ids * 109 + 19) % 191 + 64,
                    )
                ).astype(np.float32)
                / 255
            )
            for definition in definitions:
                color = QtGui.QColor(definition.color)
                self._palette[definition.id] = (
                    color.redF(),
                    color.greenF(),
                    color.blueF(),
                )
            self._palette_classes = definitions
        return self._palette

    def _intensity_colors(self, indices):
        values = self.document.frame.points[:, 3]
        if self._intensity_range is None:
            finite = np.isfinite(values)
            self._intensity_range = (
                (float(values[finite].min()), float(values[finite].max()))
                if finite.any()
                else (0, 1)
            )
        low, high = self._intensity_range
        normalized = np.nan_to_num(
            (values[indices].astype(np.float64) - low)
            / max(high - low, 1e-12),
            nan=0,
            posinf=1,
            neginf=0,
        ).astype(np.float32)
        self._set_render_legend(
            self.tr(
                "Intensity: {low:.3g} → {high:.3g} (display normalization only)"
            ).format(low=low, high=high)
        )
        return np.column_stack(
            (normalized, 0.25 + 0.65 * normalized, 1 - 0.8 * normalized)
        )

    def _display_colors(self, indices, selected):
        doc = self.document
        semantic = doc.semantic_view[indices]
        instance = doc.instance_view[indices]
        color_mode = self.color_mode.currentData()
        if color_mode == "intensity":
            colors = self._intensity_colors(indices)
        elif color_mode == "rgb":
            colors = (
                doc.frame.rgb[indices].astype(np.float32) / 255
                if doc.frame.rgb is not None
                else np.ones((len(semantic), 3), dtype=np.float32)
            )
            self._set_render_legend(
                self.tr(
                    "Original: source RGB colors or white when RGB is absent; labels are unchanged."
                )
            )
        else:
            palette = self._class_palette()
            colors = palette[semantic]
            if color_mode == "instance":
                codes = doc.labels[indices].astype(np.uint64)
                instance_colors = (
                    np.column_stack(
                        (
                            (codes * 37 + 83) % 191 + 64,
                            (codes * 73 + 41) % 191 + 64,
                            (codes * 109 + 19) % 191 + 64,
                        )
                    ).astype(np.float32)
                    / 255
                )
                has_instance = instance != 0
                colors *= 0.4
                colors[has_instance] = instance_colors[has_instance]
                colors[semantic == 0] = palette[0]
            self._set_render_legend(
                self.tr(
                    "Colors identify class IDs or (class, instance) pairs. Dimmed points have no instance."
                )
            )
        if selected is not None:
            colors[selected] = (
                colors[selected] * 0.4
                + np.array([1, 0.85, 0.2], dtype=np.float32) * 0.6
            )
        return colors

    def _refresh_display(self, *args):
        if self._refreshing:
            return
        self._sync_visibility_actions()
        if not self.document:
            return
        self.viewport.cancel_selection()
        doc = self.document
        with QtCore.QSignalBlocker(self.color_mode):
            self.color_mode.model().item(
                self.color_mode.findData("intensity")
            ).setEnabled(doc.frame.has_intensity)
            if (
                not doc.frame.has_intensity
                and self.color_mode.currentData() == "intensity"
            ):
                self.color_mode.setCurrentIndex(
                    self.color_mode.findData(
                        "rgb" if self.detection.enabled else "semantic"
                    )
                )
        self._sync_render_actions()
        hidden = tuple(
            self.class_list.item(index).data(QtCore.Qt.ItemDataRole.UserRole)
            for index in range(self.class_list.count())
            if self.class_list.item(index).checkState()
            == QtCore.Qt.CheckState.Unchecked
        )
        selected_codes = tuple(
            sorted(
                (key[1] << 16) | key[0]
                for item in self.instance_list.selectedItems()
                for key in [item.data(QtCore.Qt.ItemDataRole.UserRole)]
            )
        )
        focused_codes = tuple(
            (key[1] << 16) | key[0]
            for index in range(self.instance_list.count())
            for item in [self.instance_list.item(index)]
            if item.checkState() == QtCore.Qt.CheckState.Checked
            for key in [item.data(QtCore.Qt.ItemDataRole.UserRole)]
        )
        detection = self.detection.enabled
        if detection:
            hidden = selected_codes = focused_codes = ()
        instances_visible = (
            detection or self.instances_visibility_action.isChecked()
        )
        signature = (
            doc,
            tuple(self.class_definitions["segmentation"]),
            self.color_mode.currentData(),
            selected_codes,
            hidden,
            focused_codes,
            instances_visible,
            detection,
        )
        if (
            signature != self._display_signature
            or doc.revision != self._display_revision
        ):
            incremental = (
                signature == self._display_signature
                and doc.revision == self._display_revision + 1
            )
            indices = doc.last_changed_indices if incremental else None
            target = slice(None) if indices is None else indices
            semantic = doc.semantic_view[target]
            selected = (
                np.isin(doc.labels[target], selected_codes)
                if selected_codes
                else None
            )
            allowed = np.ones(65536, dtype=bool)
            allowed[list(hidden)] = False
            visible = allowed[semantic]
            if not instances_visible:
                visible &= doc.instance_view[target] == 0
            elif focused_codes:
                visible &= np.isin(doc.labels[target], focused_codes)
            if incremental:
                self._visible_count += int(np.count_nonzero(visible)) - int(
                    np.count_nonzero(self._visible[indices])
                )
                self._visible[indices] = visible
            else:
                self._visible = visible
                self._visible_count = int(np.count_nonzero(visible))
            if (
                self._display_signature is None
                or self._display_signature[0] is not doc
            ):
                self._intensity_range = None
            self.viewport.set_colors(
                self._display_colors(target, selected), indices=indices
            )
            self.viewport.set_visible_mask(visible, indices=indices)
            self._display_signature = signature
            self._display_revision = doc.revision
        self.camera_panel.refresh()
        self.detection.sync_display()
        state = (
            self.tr("Modified")
            if doc.dirty
            else (
                self.tr("Saved")
                if doc.frame.label_path.exists() or doc.frame.cuboid_exists
                else self.tr("No result file")
            )
        )
        self.summary.setText(
            self.tr(
                "{total} points · {visible} visible · {unlabeled} unlabeled · {state}"
            ).format(
                total=len(doc.labels),
                visible=self._visible_count,
                unlabeled=doc.semantic_counts()[0],
                state=state,
            )
        )
        if self.detection.enabled:
            self.summary.setText(
                self.summary.text()
                + self.tr(" · {count} cuboids").format(count=len(doc.cuboids))
            )
        if not self._visible_count:
            self._status(
                self.tr(
                    "No points match the display filters. Check the class and instance eye icons."
                )
            )

    def _show_shortcuts(self):
        def binding(action):
            return [
                key.toString(QtGui.QKeySequence.SequenceFormat.NativeText)
                for key in action.shortcuts()
            ]

        groups = [
            (
                self.tr("3D Detection"),
                [
                    (self.tr("Draw cuboid"), ["N"]),
                    (self.tr("Focus cuboid"), ["G"]),
                    (self.tr("Delete cuboid"), ["Delete"]),
                    (self.tr("Duplicate cuboid"), ["Ctrl+D"]),
                    (self.tr("Copy / Paste cuboid"), ["Ctrl+C", "Ctrl+V"]),
                    (self.tr("Pan 3D view"), [self.tr("Right drag")]),
                    (self.tr("Rotate 3D view"), [self.tr("Left drag")]),
                    (
                        self.tr("Edit cuboid in side views"),
                        [self.tr("Left or right drag on the box or handles")],
                    ),
                    (self.tr("Move cuboid"), [self.tr("Arrow keys")]),
                ],
            ),
            (
                self.tr("3D camera"),
                [
                    (action.text(), binding(action))
                    for action in self.viewport.camera_actions.values()
                ],
            ),
            (
                self.tr("Tools"),
                [
                    (self.tr("Browse"), binding(self.tool_actions["browse"])),
                    (self.tr("Brush"), binding(self.tool_actions["brush"])),
                    (
                        self.tr("Polygon"),
                        binding(self.tool_actions["polygon"]),
                    ),
                ],
            ),
            (
                self.tr("Editing"),
                [
                    (self.undo_action.text(), binding(self.undo_action)),
                    (self.redo_action.text(), binding(self.redo_action)),
                    (self.finish_action.text(), binding(self.finish_action)),
                    (self.cancel_action.text(), binding(self.cancel_action)),
                ],
            ),
            (
                self.tr("Navigation"),
                [
                    (self.tr("Previous frame"), ["A", "PgUp"]),
                    (self.tr("Next frame"), ["D", "PgDown"]),
                    (self.fit_action.text(), binding(self.fit_action)),
                ],
            ),
            (
                self.tr("Mouse controls"),
                [
                    (
                        self.tr("Pan (Browse)"),
                        [
                            self.tr("Right drag"),
                            self.tr("Middle drag"),
                        ],
                    ),
                    (
                        self.tr("Rotate"),
                        [self.tr("Left drag")],
                    ),
                    (self.tr("Zoom"), [self.tr("Wheel")]),
                    (self.tr("Adjust brush size"), [self.tr("Ctrl+Wheel")]),
                    (self.tr("Pan (Polygon)"), [self.tr("Ctrl+Left drag")]),
                    (
                        self.tr("Finish polygon selection"),
                        [self.tr("Double-click")],
                    ),
                ],
            ),
        ]
        ShortcutsDialog(groups, self).exec()

    def _show_help(self):
        config = getattr(self.parent(), "_config", {})
        locale = "/zh-Hans" if config.get("language") == "zh_CN" else ""
        open_url(
            f"https://xanylabeling.com{locale}/docs/"
            "x-anylabeling/point_cloud"
        )
