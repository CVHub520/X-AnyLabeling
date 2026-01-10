import functools
import html
import json
import math
import os
import os.path as osp
import re
import shutil
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWhatsThis,
    QWidget,
)

from anylabeling.services.auto_labeling.types import AutoLabelingMode
from anylabeling.services.auto_labeling import _THUMBNAIL_RENDER_MODELS
from anylabeling.views.training import UltralyticsDialog

from ...app_info import (
    __appname__,
    __version__,
    __preferred_device__,
)
from . import utils
from ...config import get_config, save_config
from .label_file import LabelFile, LabelFileError
from .logger import logger
from .shape import Shape
from .utils.file_search import (
    parse_search_pattern,
    matches_filename,
    matches_label_attribute,
)
from .widgets import (
    AboutDialog,
    AutoLabelingWidget,
    BrightnessContrastDialog,
    Canvas,
    ChatbotDialog,
    ClassifierDialog,
    VQADialog,
    CrosshairSettingsDialog,
    FileDialogPreview,
    ShapeModifyDialog,
    GroupIDFilterComboBox,
    LabelDialog,
    LabelFilterComboBox,
    LabelListWidget,
    LabelListWidgetItem,
    DigitShortcutDialog,
    LabelModifyDialog,
    GroupIDModifyDialog,
    OverviewDialog,
    SearchBar,
    ToolBar,
    UniqueLabelQListWidget,
    ZoomWidget,
    NavigatorDialog,
)

LABEL_COLORMAP = utils.label_colormap()
LABEL_OPACITY = 128


class LabelingWidget(LabelDialog):
    """The main widget for labeling images"""

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2
    next_files_changed = QtCore.pyqtSignal(list)

    def __init__(  # noqa: C901
        self,
        parent=None,
        config=None,
        filename=None,
        output=None,
        output_file=None,
        output_dir=None,
    ):
        self.parent = parent
        if output is not None:
            logger.warning(
                "argument output is deprecated, use output_file instead"
            )
            if output_file is None:
                output_file = output

        self.filename = None
        self.image_path = None
        self.image_data = None
        self.label_file = None
        self.other_data = {}
        self.classes_file = None
        self.attributes = {}
        self.attribute_widget_types = {}
        self.current_category = None
        self.selected_polygon_stack = []
        self.supported_shape = Shape.get_supported_shape()
        self.label_info = {}
        self.image_flags = []
        self.fn_to_index = {}
        self.cache_auto_label = None
        self.cache_auto_label_group_id = None

        # see configs/anylabeling_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config
        self.label_flags = self._config["label_flags"]
        self.label_loop_count = -1
        self.select_loop_count = -1
        self.digit_to_label = None
        self.drawing_digit_shortcuts = self._config.get("digit_shortcuts", {})

        # set default shape colors
        Shape.line_color = QtGui.QColor(*self._config["shape"]["line_color"])
        Shape.fill_color = QtGui.QColor(*self._config["shape"]["fill_color"])
        Shape.select_line_color = QtGui.QColor(
            *self._config["shape"]["select_line_color"]
        )
        Shape.select_fill_color = QtGui.QColor(
            *self._config["shape"]["select_fill_color"]
        )
        Shape.vertex_fill_color = QtGui.QColor(
            *self._config["shape"]["vertex_fill_color"]
        )
        Shape.hvertex_fill_color = QtGui.QColor(
            *self._config["shape"]["hvertex_fill_color"]
        )

        # Set point size from config file
        Shape.point_size = self._config["shape"]["point_size"]
        # Set line width from config file
        Shape.line_width = self._config["shape"]["line_width"]

        super(LabelDialog, self).__init__()

        # Whether we need to save or not.
        self.dirty = False

        self._no_selection_slot = False
        self._copied_shapes = None
        self._batch_edit_warning_shown = False

        self.brightness_contrast_dialog = BrightnessContrastDialog(
            self.on_new_brightness_contrast, parent=self
        )

        # Main widgets and related state.
        self.label_dialog = LabelDialog(
            parent=self,
            labels=self._config["labels"],
            sort_labels=self._config["sort_labels"],
            show_text_field=self._config["show_label_text_field"],
            completion=self._config["label_completion"],
            fit_to_content=self._config["fit_to_content"],
            flags=self.label_flags,
        )

        self.label_list = LabelListWidget()
        self.last_open_dir = None

        self.flag_dock = self.flag_widget = None
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("Flags")
        self.flag_widget = QtWidgets.QListWidget()
        if config["flags"]:
            self.image_flags = config["flags"]
            self.load_flags({k: False for k in self.image_flags})
        else:
            self.flag_dock.hide()
        self.flag_dock.setWidget(self.flag_widget)
        self.flag_widget.itemChanged.connect(self.set_dirty)
        self.flag_dock.setStyleSheet(
            "QDockWidget::title {" "text-align: center;" "padding: 0px;" "}"
        )

        # Create and add combobox for showing unique labels or group ids in group
        self.label_filter_combobox = LabelFilterComboBox(self)
        self.gid_filter_combobox = GroupIDFilterComboBox(self)

        # Create select all/none toggle button
        self.select_toggle_button = QPushButton(self.tr("Select"), self)
        self.select_toggle_button.setCheckable(True)
        self.select_toggle_button.clicked.connect(self.toggle_select_all)

        self.label_list.item_selection_changed.connect(
            self.label_selection_changed
        )
        self.label_list.item_double_clicked.connect(self.edit_label)
        self.label_list.item_changed.connect(self.label_item_changed)
        self.label_list.item_dropped.connect(self.label_order_changed)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Objects"), self)
        self.shape_dock.setWidget(self.label_list)
        self.shape_dock.setStyleSheet(
            "QDockWidget::title {" "text-align: center;" "padding: 0px;" "}"
        )
        self.shape_dock.setTitleBarWidget(QtWidgets.QWidget())

        self.unique_label_list = UniqueLabelQListWidget()
        self.unique_label_list.setToolTip(
            self.tr(
                "Select label to start annotating for it. "
                "Press 'Esc' to deselect."
            )
        )
        self.load_labels(self._config["labels"])
        self.label_dock = QtWidgets.QDockWidget(self.tr("Labels"), self)
        self.label_dock.setObjectName("Labels")
        self.label_dock.setWidget(self.unique_label_list)
        self.label_dock.setStyleSheet(
            "QDockWidget::title {" "text-align: center;" "padding: 0px;" "}"
        )

        self.file_search = SearchBar()
        self.file_search.setPlaceholderText(self.tr("Search files..."))
        self.file_search.setToolTip(
            self.tr(
                "Supported search modes:\n"
                "- Text: plain text search\n"
                "- Regex: <pattern> (e.g., <\\.png$>)\n"
                "- Attributes: difficult::1, gid::0, shape::1, label::xxx, type::xxx\n"
                "- Score range: score::[0,0.5], score::(0,0.6], score::[0,0.6), score::(0,0.6)\n"
                "- Description: description::1, description::true, description::yes\n"
                "Press Enter to search."
            )
        )
        self.file_search.returnPressed.connect(self.file_search_changed)
        self.file_list_widget = QtWidgets.QListWidget()
        self.file_list_widget.itemSelectionChanged.connect(
            self.file_selection_changed
        )
        file_list_layout = QtWidgets.QVBoxLayout()
        file_list_layout.setContentsMargins(0, 4, 0, 0)
        file_list_layout.setSpacing(4)
        file_list_layout.addWidget(self.file_search)
        file_list_layout.addWidget(self.file_list_widget)
        self.file_dock = QtWidgets.QDockWidget("", self)
        self.file_dock.setObjectName("Files")
        self.file_dock.setTitleBarWidget(QtWidgets.QWidget(self))
        file_list_widget = QtWidgets.QWidget()
        file_list_widget.setLayout(file_list_layout)
        self.file_dock.setWidget(file_list_widget)
        self.file_dock.setStyleSheet(
            "QDockWidget::title {" "text-align: center;" "padding: 0px;" "}"
        )

        self.zoom_widget = ZoomWidget()

        self.navigator_dialog = NavigatorDialog(self)
        self.navigator_dialog.navigator.navigation_requested.connect(
            self.on_navigator_request
        )
        self.navigator_dialog.closeEvent = self._navigator_close_event
        self.navigator_dialog.zoom_changed[int].connect(
            lambda zoom: self.on_navigator_zoom_changed(zoom, None)
        )
        self.navigator_dialog.zoom_changed[int, QtCore.QPoint].connect(
            self.on_navigator_zoom_changed
        )
        self.navigator_dialog.viewport_update_requested.connect(
            self.on_navigator_viewport_update_requested
        )
        self.async_exif_scanner = utils.AsyncExifScanner(self)
        self.async_exif_scanner.exif_detected.connect(self.on_exif_detected)

        self.setAcceptDrops(True)

        self.canvas = self.label_list.canvas = Canvas(
            parent=self,
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
            wheel_rectangle_editing=self._config["canvas"][
                "wheel_rectangle_editing"
            ],
            attributes=self._config["canvas"].get("attributes", {}),
            rotation=self._config["canvas"].get("rotation", {}),
            mask=self._config["canvas"].get("mask", {}),
        )
        self.canvas.zoom_request.connect(self.zoom_request)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)
        self.scroll_bars = {
            Qt.Vertical: scroll_area.verticalScrollBar(),
            Qt.Horizontal: scroll_area.horizontalScrollBar(),
        }
        self.scroll_bars[Qt.Vertical].valueChanged.connect(
            lambda: self.update_navigator_viewport()
        )
        self.scroll_bars[Qt.Horizontal].valueChanged.connect(
            lambda: self.update_navigator_viewport()
        )
        self.canvas.scroll_request.connect(self.scroll_request)
        self.canvas.new_shape.connect(self.new_shape)
        self.canvas.show_shape.connect(self.show_shape)
        self.canvas.shape_moved.connect(self.set_dirty)
        self.canvas.shape_rotated.connect(self.set_dirty)
        self.canvas.selection_changed.connect(self.shape_selection_changed)
        self.canvas.drawing_polygon.connect(self.toggle_drawing_sensitive)
        # [Feature] support for automatically switching to editing mode
        # when the cursor moves over an object
        self.canvas.h_shape_is_hovered = self._config.get(
            "auto_highlight_shape", False
        )
        if self._config["auto_switch_to_edit_mode"]:
            self.canvas.mode_changed.connect(self.set_edit_mode)

        # Crosshair
        self.crosshair_settings = self._config["canvas"]["crosshair"]
        self.canvas.set_cross_line(**self.crosshair_settings)

        self._central_widget = scroll_area

        features = QtWidgets.QDockWidget.DockWidgetFeatures()
        for dock in ["flag_dock", "label_dock", "shape_dock", "file_dock"]:
            if self._config[dock]["closable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetClosable
            if self._config[dock]["floatable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetFloatable
            if self._config[dock]["movable"]:
                features = features | QtWidgets.QDockWidget.DockWidgetMovable
            getattr(self, dock).setFeatures(features)
            if self._config[dock]["show"] is False:
                getattr(self, dock).setVisible(False)

        # Actions
        action = functools.partial(utils.new_action, self)
        shortcuts = self._config["shortcuts"]

        open_ = action(
            self.tr("Open File"),
            self.open_file,
            shortcuts["open"],
            "file",
            self.tr("Open image or label file"),
        )
        openvideo = action(
            self.tr("Open Video"),
            lambda: utils.open_video_file(self),
            shortcuts["open_video"],
            "video",
            self.tr("Open video file"),
        )
        opendir = action(
            self.tr("Open Dir"),
            self.open_folder_dialog,
            shortcuts["open_dir"],
            "open",
            self.tr("Open Dir"),
        )
        open_next_image = action(
            self.tr("Next Image"),
            self.open_next_image,
            shortcuts["open_next"],
            "next",
            self.tr("Open next image"),
            enabled=False,
        )
        open_prev_image = action(
            self.tr("Prev Image"),
            self.open_prev_image,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev image"),
            enabled=False,
        )
        open_next_unchecked_image = action(
            self.tr("Next Unchecked Image"),
            self.open_next_unchecked_image,
            shortcuts["open_next_unchecked"],
            "next",
            self.tr("Open next unchecked image"),
            enabled=False,
        )
        open_prev_unchecked_image = action(
            self.tr("Prev Unchecked Image"),
            self.open_prev_unchecked_image,
            shortcuts["open_prev_unchecked"],
            "prev",
            self.tr("Open previous unchecked image"),
            enabled=False,
        )
        save = action(
            self.tr("Save"),
            self.save_file,
            shortcuts["save"],
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )
        save_as = action(
            self.tr("Save As"),
            self.save_file_as,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=False,
        )
        run_all_images = action(
            self.tr("Auto Run"),
            lambda: utils.run_all_images(self),
            shortcuts["auto_run"],
            "auto-run",
            self.tr("Auto run all images at once"),
            checkable=True,
            enabled=False,
        )
        delete_file = action(
            self.tr("Delete File"),
            self.delete_file,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )
        delete_image_file = action(
            self.tr("Delete Image File"),
            self.delete_image_file,
            shortcuts["delete_image_file"],
            "delete",
            self.tr("Delete current image file"),
            enabled=True,
        )

        change_output_dir = action(
            self.tr("Change Output Dir"),
            slot=self.change_output_dir_dialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        save_auto = action(
            text=self.tr("Save Automatically"),
            slot=lambda x: self._config.update({"auto_save": x}),
            icon=None,
            tip=self.tr("Save automatically"),
            checkable=True,
            enabled=True,
            checked=self._config["auto_save"],
        )

        save_with_image_data = action(
            text=self.tr("Save With Image Data"),
            slot=lambda x: self._config.update({"store_data": x}),
            icon=None,
            tip=self.tr("Save image data in label file"),
            checkable=True,
            checked=self._config["store_data"],
        )

        close = action(
            self.tr("Close"),
            self.close_file,
            shortcuts["close"],
            "cancel",
            self.tr("Close current file"),
        )

        keep_prev_mode = action(
            self.tr("Keep Previous Annotation"),
            lambda x: self._config.update({"keep_prev": x}),
            shortcuts["toggle_keep_prev_mode"],
            None,
            self.tr('Toggle "Keep Previous Annotation" mode'),
            checkable=True,
            checked=self._config["keep_prev"],
        )

        auto_use_last_label_mode = action(
            self.tr("Auto Use Last Label"),
            lambda x: self._config.update({"auto_use_last_label": x}),
            shortcuts["toggle_auto_use_last_label"],
            None,
            self.tr('Toggle "Auto Use Last Label" mode'),
            checkable=True,
            checked=self._config["auto_use_last_label"],
        )

        auto_use_last_gid_mode = action(
            self.tr("Auto Use Last Group ID"),
            lambda x: self._config.update({"auto_use_last_gid": x}),
            shortcuts["toggle_auto_use_last_gid"],
            None,
            self.tr('Toggle "Auto Use Last Group ID" mode'),
            checkable=True,
            checked=self._config["auto_use_last_gid"],
        )

        use_system_clipboard = action(
            self.tr("Use System Clipboard"),
            self.toggle_system_clipboard,
            tip=self.tr("Use system clipboard for copy and paste"),
            checkable=True,
            checked=self._config["system_clipboard"],
            enabled=True,
        )

        visibility_shapes_mode = action(
            self.tr("Visibility Shapes"),
            self.toggle_visibility_shapes,
            shortcuts["toggle_visibility_shapes"],
            None,
            self.tr('Toggle "Visibility Shapes" mode'),
            checkable=True,
            checked=self._config["show_shapes"],
        )

        create_mode = action(
            self.tr("Create Polygons"),
            lambda: self.toggle_draw_mode(False, create_mode="polygon"),
            shortcuts["create_polygon"],
            "polygon",
            self.tr("Start drawing polygons"),
            enabled=False,
        )
        create_rectangle_mode = action(
            self.tr("Create Rectangle"),
            lambda: self.toggle_draw_mode(False, create_mode="rectangle"),
            shortcuts["create_rectangle"],
            "rectangle",
            self.tr("Start drawing rectangles"),
            enabled=False,
        )
        create_rotation_mode = action(
            self.tr("Create Rotation"),
            lambda: self.toggle_draw_mode(False, create_mode="rotation"),
            shortcuts["create_rotation"],
            "rotation",
            self.tr("Start drawing rotations"),
            enabled=False,
        )
        create_circle_mode = action(
            self.tr("Create Circle"),
            lambda: self.toggle_draw_mode(False, create_mode="circle"),
            shortcuts["create_circle"],
            "circle",
            self.tr("Start drawing circles"),
            enabled=False,
        )
        create_line_mode = action(
            self.tr("Create Line"),
            lambda: self.toggle_draw_mode(False, create_mode="line"),
            shortcuts["create_line"],
            "line",
            self.tr("Start drawing lines"),
            enabled=False,
        )
        create_point_mode = action(
            self.tr("Create Point"),
            lambda: self.toggle_draw_mode(False, create_mode="point"),
            shortcuts["create_point"],
            "point",
            self.tr("Start drawing points"),
            enabled=False,
        )
        create_line_strip_mode = action(
            self.tr("Create LineStrip"),
            lambda: self.toggle_draw_mode(False, create_mode="linestrip"),
            shortcuts["create_linestrip"],
            "line-strip",
            self.tr("Start drawing linestrip. Ctrl+LeftClick ends creation."),
            enabled=False,
        )
        digit_shortcut_0 = action(
            self.tr("Digit Shortcut 0"),
            lambda: self.create_digit_mode(0),
            "0",
            "digit0",
            enabled=False,
        )
        digit_shortcut_1 = action(
            self.tr("Digit Shortcut 1"),
            lambda: self.create_digit_mode(1),
            "1",
            "digit1",
            enabled=False,
        )
        digit_shortcut_2 = action(
            self.tr("Digit Shortcut 2"),
            lambda: self.create_digit_mode(2),
            "2",
            "digit2",
            enabled=False,
        )
        digit_shortcut_3 = action(
            self.tr("Digit Shortcut 3"),
            lambda: self.create_digit_mode(3),
            "3",
            "digit3",
            enabled=False,
        )
        digit_shortcut_4 = action(
            self.tr("Digit Shortcut 4"),
            lambda: self.create_digit_mode(4),
            "4",
            "digit4",
            enabled=False,
        )
        digit_shortcut_5 = action(
            self.tr("Digit Shortcut 5"),
            lambda: self.create_digit_mode(5),
            "5",
            "digit5",
            enabled=False,
        )
        digit_shortcut_6 = action(
            self.tr("Digit Shortcut 6"),
            lambda: self.create_digit_mode(6),
            "6",
            "digit6",
            enabled=False,
        )
        digit_shortcut_7 = action(
            self.tr("Digit Shortcut 7"),
            lambda: self.create_digit_mode(7),
            "7",
            "digit7",
            enabled=False,
        )
        digit_shortcut_8 = action(
            self.tr("Digit Shortcut 8"),
            lambda: self.create_digit_mode(8),
            "8",
            "digit8",
            enabled=False,
        )
        digit_shortcut_9 = action(
            self.tr("Digit Shortcut 9"),
            lambda: self.create_digit_mode(9),
            "9",
            "digit9",
            enabled=False,
        )
        edit_mode = action(
            self.tr("Edit Object"),
            self.set_edit_mode,
            shortcuts["edit_polygon"],
            "edit",
            self.tr("Move and edit the selected polygons"),
            enabled=False,
        )
        group_selected_shapes = action(
            self.tr("Group Selected Shapes"),
            self.group_selected_shapes,
            shortcuts["group_selected_shapes"],
            None,
            self.tr("Group shapes by assigning a same group_id"),
            enabled=True,
        )
        ungroup_selected_shapes = action(
            self.tr("Ungroup Selected Shapes"),
            self.ungroup_selected_shapes,
            shortcuts["ungroup_selected_shapes"],
            None,
            self.tr("Ungroup shapes"),
            enabled=True,
        )

        delete = action(
            self.tr("Delete"),
            self.delete_selected_shape,
            shortcuts["delete_polygon"],
            "cancel",
            self.tr("Delete the selected polygons"),
            enabled=False,
        )
        duplicate = action(
            self.tr("Duplicate Polygons"),
            self.duplicate_selected_shape,
            shortcuts["duplicate_polygon"],
            "copy",
            self.tr("Create a duplicate of the selected polygons"),
            enabled=False,
        )
        copy = action(
            self.tr("Copy Object"),
            self.copy_selected_shape,
            shortcuts["copy_polygon"],
            "copy",
            self.tr("Copy selected polygons to clipboard"),
            enabled=False,
        )
        paste = action(
            self.tr("Paste Object"),
            self.paste_selected_shape,
            shortcuts["paste_polygon"],
            "paste",
            self.tr("Paste copied polygons"),
            enabled=self._config["system_clipboard"],
        )
        undo_last_point = action(
            self.tr("Undo last point"),
            self.canvas.undo_last_point,
            shortcuts["undo_last_point"],
            "undo",
            self.tr("Undo last drawn point"),
            enabled=False,
        )
        remove_point = action(
            text=self.tr("Remove Selected Point"),
            slot=self.remove_selected_point,
            shortcut=shortcuts["remove_selected_point"],
            icon="edit",
            tip=self.tr("Remove selected point from polygon"),
            enabled=False,
        )

        undo = action(
            self.tr("Undo"),
            self.undo_shape_edit,
            shortcuts["undo"],
            "undo",
            self.tr("Undo last add and edit of shape"),
            enabled=False,
        )
        hide_selected_polygons = action(
            self.tr("Hide Selected Polygons"),
            self.hide_selected_polygons,
            shortcuts["hide_selected_polygons"],
            None,
            self.tr("Hide selected polygons"),
            enabled=True,
        )
        show_hidden_polygons = action(
            self.tr("Show Hidden Polygons"),
            self.show_hidden_polygons,
            shortcuts["show_hidden_polygons"],
            None,
            self.tr("Show hidden polygons"),
            enabled=True,
        )

        overview = action(
            self.tr("Overview"),
            self.overview,
            shortcuts["show_overview"],
            icon="overview",
            tip=self.tr("Show annotations statistics"),
        )
        save_crop = action(
            self.tr("Save Cropped Image"),
            lambda: utils.save_crop(self),
            icon="crop",
            tip=self.tr(
                "Save cropped image. (Support rectangle/rotation/polygon shape_type)"
            ),
        )
        digit_shortcut_manager = action(
            self.tr("Digit Shortcut Manager"),
            self.digit_shortcut_manager,
            shortcuts["edit_digit_shortcut"],
            icon="edit",
            tip=self.tr(
                "Manage Digit Shortcuts: Assign Drawing Modes and Labels to Number Keys"
            ),
        )
        label_manager = action(
            self.tr("Label Manager"),
            self.label_manager,
            shortcuts["edit_labels"],
            icon="edit",
            tip=self.tr(
                "Manage Labels: Rename, Delete, Hide/Show, Adjust Color"
            ),
        )
        gid_manager = action(
            self.tr("Group ID Manager"),
            self.gid_manager,
            shortcuts["edit_group_id"],
            icon="edit",
            tip=self.tr("Manage Group ID"),
        )
        shape_manager = action(
            self.tr("Shape Manager"),
            self.shape_manager,
            shortcuts["edit_shapes"],
            icon="edit",
            tip=self.tr("Manage Shapes: Add, Delete, Remove"),
            enabled=False,
        )
        copy_coordinates = action(
            self.tr("Copy Coordinates"),
            self.copy_shape_coordinates,
            icon="copy",
            tip=self.tr("Copy shape coordinates to clipboard"),
            enabled=False,
        )
        union_selection = action(
            self.tr("Union Selection"),
            self.union_selection,
            shortcuts["union_selected_shapes"],
            icon="union",
            tip=self.tr("Union multiple selected rectangle shapes"),
            enabled=False,
        )
        hbb_to_obb = action(
            self.tr("Convert HBB to OBB"),
            lambda: utils.shape_conversion(self, "hbb_to_obb"),
            icon="convert",
            tip=self.tr(
                "Perform conversion from horizontal bounding box to oriented bounding box"
            ),
        )
        obb_to_hbb = action(
            self.tr("Convert OBB to HBB"),
            lambda: utils.shape_conversion(self, "obb_to_hbb"),
            icon="convert",
            tip=self.tr(
                "Perform conversion from oriented bounding box to horizontal bounding box"
            ),
        )
        polygon_to_hbb = action(
            self.tr("Convert Polygon to HBB"),
            lambda: utils.shape_conversion(self, "polygon_to_hbb"),
            icon="convert",
            tip=self.tr(
                "Perform conversion from polygon to horizontal bounding box"
            ),
        )
        polygon_to_obb = action(
            self.tr("Convert Polygon to OBB"),
            lambda: utils.shape_conversion(self, "polygon_to_obb"),
            icon="convert",
            tip=self.tr(
                "Perform conversion from polygon to oriented bounding box"
            ),
        )
        circle_to_polygon = action(
            self.tr("Convert Circle to Polygon"),
            lambda: utils.shape_conversion(self, "circle_to_polygon"),
            icon="convert",
            tip=self.tr(
                "Perform conversion from circle to polygon with user-specified points"
            ),
        )
        open_chatbot = action(
            self.tr("ChatBot"),
            self.open_chatbot,
            shortcuts["open_chatbot"],
            icon="psyduck",
            tip=self.tr("Open chatbot dialog"),
        )
        open_vqa = action(
            self.tr("VQA"),
            self.open_vqa,
            shortcuts["open_vqa"],
            icon="husky",
            tip=self.tr("Open VQA dialog"),
        )
        open_classifier = action(
            self.tr("Classifier"),
            self.open_classifier,
            shortcuts["open_classifier"],
            icon="ragdoll",
            tip=self.tr("Open classifier dialog"),
        )
        documentation = action(
            self.tr("Documentation"),
            self.documentation,
            icon="docs",
            tip=self.tr("Show documentation"),
        )
        about = action(
            self.tr("About"),
            self.about,
            icon="help",
            tip=self.tr("Open about dialog"),
        )

        loop_thru_labels = action(
            self.tr("Loop Through Labels"),
            self.loop_thru_labels,
            shortcut=shortcuts["loop_thru_labels"],
            icon="loop",
            tip=self.tr("Loop through labels"),
            enabled=False,
        )
        loop_select_labels = action(
            self.tr("Loop Select Labels"),
            self.loop_select_labels,
            shortcut=shortcuts["loop_select_labels"],
            icon="circle-selection",
            tip=self.tr("Loop select labels"),
            enabled=False,
        )

        ultralytics_train = action(
            "Ultralytics",
            lambda: self.start_training("ultralytics"),
            icon="ultralytics",
        )

        zoom = QtWidgets.QWidgetAction(self)
        zoom.setDefaultWidget(self.zoom_widget)
        self.zoom_widget.setWhatsThis(
            str(
                self.tr(
                    "Zoom in or out of the image. Also accessible with "
                    "{} and {} from the canvas."
                )
            ).format(
                utils.fmt_shortcut(
                    f"{shortcuts['zoom_in']},{shortcuts['zoom_out']}"
                ),
                utils.fmt_shortcut(self.tr("Ctrl+Wheel")),
            )
        )
        self.zoom_widget.setEnabled(False)

        zoom_in = action(
            self.tr("Zoom In"),
            functools.partial(self.add_zoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoom_out = action(
            self.tr("Zoom Out"),
            functools.partial(self.add_zoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoom_org = action(
            self.tr("Original Size"),
            functools.partial(self.set_zoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keep_prev_scale = action(
            self.tr("Keep Previous Scale"),
            lambda x: self._config.update({"keep_prev_scale": x}),
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        keep_prev_brightness = action(
            self.tr("Keep Previous Brightness"),
            lambda x: self._config.update({"keep_prev_brightness": x}),
            tip=self.tr("Keep previous brightness"),
            checkable=True,
            checked=self._config["keep_prev_brightness"],
            enabled=True,
        )
        keep_prev_contrast = action(
            self.tr("Keep Previous Contrast"),
            lambda x: self._config.update({"keep_prev_contrast": x}),
            tip=self.tr("Keep previous contrast"),
            checkable=True,
            checked=self._config["keep_prev_contrast"],
            enabled=True,
        )
        fit_window = action(
            self.tr("Fit Window"),
            self.set_fit_window,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fit_width = action(
            self.tr("Fit Width"),
            self.set_fit_width,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightness_contrast = action(
            self.tr("Set Brightness Contrast"),
            self.brightness_contrast,
            None,
            "color",
            "Adjust brightness and contrast",
            enabled=False,
        )
        set_cross_line = action(
            self.tr("Set Cross Line"),
            self.set_cross_line,
            tip=self.tr("Adjust cross line for mouse position"),
            icon="cartesian",
        )
        show_groups = action(
            self.tr("Show Groups"),
            lambda x: self.set_canvas_params("show_groups", x),
            tip=self.tr("Show shape groups"),
            icon=None,
            checkable=True,
            checked=self._config["show_groups"],
            enabled=True,
            auto_trigger=True,
        )
        show_masks = action(
            self.tr("Show Masks"),
            lambda x: self.set_canvas_params("show_masks", x),
            shortcut=shortcuts["show_masks"],
            tip=self.tr("Show semi-transparent masks for shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_masks"],
            enabled=True,
            auto_trigger=True,
        )
        show_texts = action(
            self.tr("Show Texts"),
            lambda x: self.set_canvas_params("show_texts", x),
            shortcut=shortcuts["show_texts"],
            tip=self.tr("Show text above shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_texts"],
            enabled=True,
            auto_trigger=True,
        )
        show_labels = action(
            self.tr("Show Labels"),
            lambda x: self.set_canvas_params("show_labels", x),
            shortcut=shortcuts["show_labels"],
            tip=self.tr("Show label inside shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_labels"],
            enabled=True,
            auto_trigger=True,
        )
        show_scores = action(
            self.tr("Show Scores"),
            lambda x: self.set_canvas_params("show_scores", x),
            tip=self.tr("Show score inside shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_scores"],
            enabled=True,
            auto_trigger=True,
        )
        show_attributes = action(
            self.tr("Show Attributes"),
            lambda x: self.set_canvas_params("show_attributes", x),
            shortcut=shortcuts["show_attributes"],
            tip=self.tr("Show attribute inside shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_attributes"],
            enabled=True,
            auto_trigger=True,
        )
        show_degrees = action(
            self.tr("Show Degress"),
            lambda x: self.set_canvas_params("show_degrees", x),
            tip=self.tr("Show degrees above rotated shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_degrees"],
            enabled=True,
            auto_trigger=True,
        )
        show_linking = action(
            self.tr("Show KIE Linking"),
            lambda x: self.set_canvas_params("show_linking", x),
            shortcut=shortcuts["show_linking"],
            tip=self.tr("Show KIE linking between key and value"),
            icon=None,
            checkable=True,
            checked=self._config["show_linking"],
            enabled=True,
            auto_trigger=True,
        )

        # Languages
        select_lang_en = action(
            "English",
            functools.partial(self.set_language, "en_US"),
            icon="us",
            checkable=True,
            checked=self._config["language"] == "en_US",
            enabled=self._config["language"] != "en_US",
        )
        select_lang_zh = action(
            "中文",
            functools.partial(self.set_language, "zh_CN"),
            icon="cn",
            checkable=True,
            checked=self._config["language"] == "zh_CN",
            enabled=self._config["language"] != "zh_CN",
        )

        # Upload
        upload_image_flags_file = action(
            self.tr("Upload Image Flags File"),
            lambda: utils.upload_image_flags_file(self),
            None,
            icon="format_classify",
            tip=self.tr("Upload Custom Image Flags File"),
        )
        upload_label_flags_file = action(
            self.tr("Upload Label Flags File"),
            lambda: utils.upload_label_flags_file(self, LABEL_OPACITY),
            None,
            icon="format_classify",
            tip=self.tr("Upload Custom Label Flags File"),
        )
        upload_shape_attrs_file = action(
            self.tr("Upload Attributes File"),
            lambda: utils.upload_shape_attrs_file(self, LABEL_OPACITY),
            None,
            icon="format_classify",
            tip=self.tr("Upload Custom Attributes File"),
        )
        upload_label_classes_file = action(
            self.tr("Upload Label Classes File"),
            lambda: utils.upload_label_classes_file(self),
            None,
            icon="format_classify",
            tip=self.tr("Upload Custom Label Classes File"),
        )
        upload_yolo_hbb_annotation = action(
            self.tr("Upload YOLO-Hbb Annotations"),
            lambda: utils.upload_yolo_annotation(self, "hbb", LABEL_OPACITY),
            None,
            icon="format_yolo",
            tip=self.tr(
                "Upload Custom YOLO Horizontal Bounding Boxes Annotations"
            ),
        )
        upload_yolo_obb_annotation = action(
            self.tr("Upload YOLO-Obb Annotations"),
            lambda: utils.upload_yolo_annotation(self, "obb", LABEL_OPACITY),
            None,
            icon="format_yolo",
            tip=self.tr(
                "Upload Custom YOLO Oriented Bounding Boxes Annotations"
            ),
        )
        upload_yolo_seg_annotation = action(
            self.tr("Upload YOLO-Seg Annotations"),
            lambda: utils.upload_yolo_annotation(self, "seg", LABEL_OPACITY),
            None,
            icon="format_yolo",
            tip=self.tr("Upload Custom YOLO Segmentation Annotations"),
        )
        upload_yolo_pose_annotation = action(
            self.tr("Upload YOLO-Pose Annotations"),
            lambda: utils.upload_yolo_annotation(self, "pose", LABEL_OPACITY),
            None,
            icon="format_yolo",
            tip=self.tr("Upload Custom YOLO Pose Annotations"),
        )
        upload_voc_det_annotation = action(
            self.tr("Upload VOC Detection Annotations"),
            lambda: utils.upload_voc_annotation(self, "rectangle"),
            None,
            icon="format_voc",
            tip=self.tr("Upload Custom Pascal VOC Detection Annotations"),
        )
        upload_voc_seg_annotation = action(
            self.tr("Upload VOC Segmentation Annotations"),
            lambda: utils.upload_voc_annotation(self, "polygon"),
            None,
            icon="format_voc",
            tip=self.tr("Upload Custom Pascal VOC Segmentation Annotations"),
        )
        upload_coco_det_annotation = action(
            self.tr("Upload COCO Detection Annotations"),
            lambda: utils.upload_coco_annotation(self, "rectangle"),
            None,
            icon="format_coco",
            tip=self.tr("Upload Custom COCO Detection Annotations"),
        )
        upload_coco_seg_annotation = action(
            self.tr("Upload COCO Instance Segmentation Annotations"),
            lambda: utils.upload_coco_annotation(self, "polygon"),
            None,
            icon="format_coco",
            tip=self.tr(
                "Upload Custom COCO Instance Segmentation Annotations"
            ),
        )
        upload_coco_pose_annotation = action(
            self.tr("Upload COCO Keypoint Annotations"),
            lambda: utils.upload_coco_annotation(self, "pose"),
            None,
            icon="format_coco",
            tip=self.tr("Upload Custom COCO Keypoint Annotations"),
        )
        upload_dota_annotation = action(
            self.tr("Upload DOTA Annotations"),
            lambda: utils.upload_dota_annotation(self),
            None,
            icon="format_dota",
            tip=self.tr("Upload Custom DOTA Annotations"),
        )
        upload_mask_annotation = action(
            self.tr("Upload MASK Annotations"),
            lambda: utils.upload_mask_annotation(self, LABEL_OPACITY),
            None,
            icon="format_mask",
            tip=self.tr("Upload Custom MASK Annotations"),
        )
        upload_mot_annotation = action(
            self.tr("Upload MOT Annotations"),
            lambda: utils.upload_mot_annotation(self, LABEL_OPACITY),
            None,
            icon="format_mot",
            tip=self.tr("Upload Custom Multi-Object-Tracking Annotations"),
        )
        upload_odvg_annotation = action(
            self.tr("Upload ODVG Annotations"),
            lambda: utils.upload_odvg_annotation(self),
            None,
            icon="format_odvg",
            tip=self.tr(
                "Upload Custom Object Detection Visual Grounding Annotations"
            ),
        )
        upload_mmgd_annotation = action(
            self.tr("Upload MM-Grounding-DINO Annotations"),
            lambda: utils.upload_mmgd_annotation(self, LABEL_OPACITY),
            None,
            icon="format_mmgd",
            tip=self.tr("Upload Custom MM-Grounding-DINO Annotations"),
        )
        upload_ppocr_rec_annotation = action(
            self.tr("Upload PPOCR-Rec Annotations"),
            lambda: utils.upload_ppocr_annotation(self, "rec"),
            None,
            icon="format_ppocr",
            tip=self.tr("Upload Custom PPOCR Recognition Annotations"),
        )
        upload_ppocr_kie_annotation = action(
            self.tr("Upload PPOCR-KIE Annotations"),
            lambda: utils.upload_ppocr_annotation(self, "kie"),
            None,
            icon="format_ppocr",
            tip=self.tr(
                "Upload Custom PPOCR Key Information Extraction (KIE - Semantic Entity Recognition & Relation Extraction) Annotations"
            ),
        )
        upload_vlm_r1_ovd_annotation = action(
            self.tr("Upload VLM-R1 OVD Annotations"),
            lambda: utils.upload_vlm_r1_ovd_annotation(self),
            None,
            icon="format_vlm_r1_ovd",
            tip=self.tr("Upload Custom VLM-R1 OVD Annotations"),
        )

        # Export
        export_yolo_hbb_annotation = action(
            self.tr("Export YOLO-Hbb Annotations"),
            lambda: utils.export_yolo_annotation(self, "hbb"),
            None,
            icon="format_yolo",
            tip=self.tr(
                "Export Custom YOLO Horizontal Bounding Boxes Annotations"
            ),
        )
        export_yolo_obb_annotation = action(
            self.tr("Export YOLO-Obb Annotations"),
            lambda: utils.export_yolo_annotation(self, "obb"),
            None,
            icon="format_yolo",
            tip=self.tr(
                "Export Custom YOLO Oriented Bounding Boxes Annotations"
            ),
        )
        export_yolo_seg_annotation = action(
            self.tr("Export YOLO-Seg Annotations"),
            lambda: utils.export_yolo_annotation(self, "seg"),
            None,
            icon="format_yolo",
            tip=self.tr("Export Custom YOLO Segmentation Annotations"),
        )
        export_yolo_pose_annotation = action(
            self.tr("Export YOLO-Pose Annotations"),
            lambda: utils.export_yolo_annotation(self, "pose"),
            None,
            icon="format_yolo",
            tip=self.tr("Export Custom YOLO Pose Annotations"),
        )
        export_voc_det_annotation = action(
            self.tr("Export VOC Detection Annotations"),
            lambda: utils.export_voc_annotation(self, "rectangle"),
            None,
            icon="format_voc",
            tip=self.tr("Export Custom PASCAL VOC Detection Annotations"),
        )
        export_voc_seg_annotation = action(
            self.tr("Export VOC Segmentation Annotations"),
            lambda: utils.export_voc_annotation(self, "polygon"),
            None,
            icon="format_voc",
            tip=self.tr("Export Custom PASCAL VOC Segmentation Annotations"),
        )
        export_coco_det_annotation = action(
            self.tr("Export COCO Detection Annotations"),
            lambda: utils.export_coco_annotation(self, "rectangle"),
            None,
            icon="format_coco",
            tip=self.tr("Export Custom COCO Rectangle Annotations"),
        )
        export_coco_seg_annotation = action(
            self.tr("Export COCO Instance Segmentation Annotations"),
            lambda: utils.export_coco_annotation(self, "polygon"),
            None,
            icon="format_coco",
            tip=self.tr(
                "Export Custom COCO Instance Segmentation Annotations"
            ),
        )
        export_coco_pose_annotation = action(
            self.tr("Export COCO Keypoint Annotations"),
            lambda: utils.export_coco_annotation(self, "pose"),
            None,
            icon="format_coco",
            tip=self.tr("Export Custom COCO Keypoint Annotations"),
        )
        export_dota_annotation = action(
            self.tr("Export DOTA Annotations"),
            lambda: utils.export_dota_annotation(self),
            None,
            icon="format_dota",
            tip=self.tr("Export Custom DOTA Annotations"),
        )
        export_mask_annotation = action(
            self.tr("Export MASK Annotations"),
            lambda: utils.export_mask_annotation(self),
            None,
            icon="format_mask",
            tip=self.tr("Export Custom MASK Annotations - RGB/Gray"),
        )
        export_mot_annotation = action(
            self.tr("Export MOT Annotations"),
            lambda: utils.export_mot_annotation(self, "mot"),
            None,
            icon="format_mot",
            tip=self.tr("Export Custom Multi-Object-Tracking Annotations"),
        )
        export_mots_annotation = action(
            self.tr("Export MOTS Annotations"),
            lambda: utils.export_mot_annotation(self, "mots"),
            None,
            icon="format_mot",
            tip=self.tr(
                "Export Custom Multi-Object-Tracking-Segmentation Annotations"
            ),
        )
        export_odvg_annotation = action(
            self.tr("Export ODVG Annotations"),
            lambda: utils.export_odvg_annotation(self),
            None,
            icon="format_odvg",
            tip=self.tr(
                "Export Custom Object Detection Visual Grounding Annotations"
            ),
        )
        export_pporc_rec_annotation = action(
            self.tr("Export PPOCR-Rec Annotations"),
            lambda: utils.export_pporc_annotation(self, "rec"),
            None,
            icon="format_ppocr",
            tip=self.tr("Export Custom PPOCR Recognition Annotations"),
        )
        export_pporc_kie_annotation = action(
            self.tr("Export PPOCR-KIE Annotations"),
            lambda: utils.export_pporc_annotation(self, "kie"),
            None,
            icon="format_ppocr",
            tip=self.tr(
                "Export Custom PPOCR Key Information Extraction (KIE - Semantic Entity Recognition & Relation Extraction) Annotations"
            ),
        )
        export_vlm_r1_ovd_annotation = action(
            self.tr("Export VLM-R1 OVD Annotations"),
            lambda: utils.export_vlm_r1_ovd_annotation(self),
            None,
            icon="format_vlm_r1_ovd",
            tip=self.tr("Export Custom VLM-R1 OVD Annotations"),
        )

        # Group zoom controls into a list for easier toggling.
        zoom_actions = (
            self.zoom_widget,
            zoom_in,
            zoom_out,
            zoom_org,
            fit_window,
            fit_width,
        )
        self.zoom_mode = self.FIT_WINDOW
        fit_window.setChecked(Qt.Checked)
        self.scalers = {
            self.FIT_WINDOW: self.scale_fit_window,
            self.FIT_WIDTH: self.scale_fit_width,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(
            self.tr("Edit Label"),
            self.edit_label,
            shortcuts["edit_label"],
            "edit",
            self.tr("Modify the label of the selected polygon"),
            enabled=False,
        )

        fill_drawing = action(
            self.tr("Fill Drawing Polygon"),
            self.canvas.set_fill_drawing,
            None,
            "color",
            self.tr("Fill polygon while drawing"),
            checkable=True,
            enabled=True,
        )
        fill_drawing.trigger()

        show_navigator = action(
            self.tr("Navigator"),
            self.toggle_navigator,
            shortcuts["show_navigator"],
            "navigator",
            self.tr("Show/hide the navigator window"),
            checkable=True,
            enabled=True,
        )

        # AI Actions
        toggle_auto_labeling_widget = action(
            self.tr("Auto Labeling"),
            self.toggle_auto_labeling_widget,
            shortcuts["auto_label"],
            "brain",
            self.tr("Auto Labeling"),
        )

        # Label list context menu.
        label_menu = QtWidgets.QMenu()
        utils.add_actions(
            label_menu, (edit, delete, copy_coordinates, union_selection)
        )
        self.label_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_list.customContextMenuRequested.connect(
            self.pop_label_list_menu
        )

        # Store actions for further handling.
        self.actions = utils.Struct(
            save_auto=save_auto,
            save_with_image_data=save_with_image_data,
            change_output_dir=change_output_dir,
            save=save,
            save_as=save_as,
            open=open_,
            close=close,
            delete_file=delete_file,
            delete_image_file=delete_image_file,
            keep_prev_mode=keep_prev_mode,
            auto_use_last_label_mode=auto_use_last_label_mode,
            auto_use_last_gid_mode=auto_use_last_gid_mode,
            use_system_clipboard=use_system_clipboard,
            visibility_shapes_mode=visibility_shapes_mode,
            run_all_images=run_all_images,
            union_selection=union_selection,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
            copy_coordinates=copy_coordinates,
            paste=paste,
            undo_last_point=undo_last_point,
            undo=undo,
            remove_point=remove_point,
            create_mode=create_mode,
            edit_mode=edit_mode,
            create_rectangle_mode=create_rectangle_mode,
            create_rotation_mode=create_rotation_mode,
            create_circle_mode=create_circle_mode,
            create_line_mode=create_line_mode,
            create_point_mode=create_point_mode,
            create_line_strip_mode=create_line_strip_mode,
            digit_shortcut_0=digit_shortcut_0,
            digit_shortcut_1=digit_shortcut_1,
            digit_shortcut_2=digit_shortcut_2,
            digit_shortcut_3=digit_shortcut_3,
            digit_shortcut_4=digit_shortcut_4,
            digit_shortcut_5=digit_shortcut_5,
            digit_shortcut_6=digit_shortcut_6,
            digit_shortcut_7=digit_shortcut_7,
            digit_shortcut_8=digit_shortcut_8,
            digit_shortcut_9=digit_shortcut_9,
            upload_image_flags_file=upload_image_flags_file,
            upload_label_flags_file=upload_label_flags_file,
            upload_shape_attrs_file=upload_shape_attrs_file,
            upload_label_classes_file=upload_label_classes_file,
            upload_yolo_hbb_annotation=upload_yolo_hbb_annotation,
            upload_yolo_obb_annotation=upload_yolo_obb_annotation,
            upload_yolo_seg_annotation=upload_yolo_seg_annotation,
            upload_yolo_pose_annotation=upload_yolo_pose_annotation,
            upload_voc_det_annotation=upload_voc_det_annotation,
            upload_voc_seg_annotation=upload_voc_seg_annotation,
            upload_coco_det_annotation=upload_coco_det_annotation,
            upload_coco_seg_annotation=upload_coco_seg_annotation,
            upload_coco_pose_annotation=upload_coco_pose_annotation,
            upload_dota_annotation=upload_dota_annotation,
            upload_mask_annotation=upload_mask_annotation,
            upload_mot_annotation=upload_mot_annotation,
            upload_odvg_annotation=upload_odvg_annotation,
            upload_mmgd_annotation=upload_mmgd_annotation,
            upload_ppocr_rec_annotation=upload_ppocr_rec_annotation,
            upload_ppocr_kie_annotation=upload_ppocr_kie_annotation,
            upload_vlm_r1_ovd_annotation=upload_vlm_r1_ovd_annotation,
            export_yolo_hbb_annotation=export_yolo_hbb_annotation,
            export_yolo_obb_annotation=export_yolo_obb_annotation,
            export_yolo_seg_annotation=export_yolo_seg_annotation,
            export_yolo_pose_annotation=export_yolo_pose_annotation,
            export_voc_det_annotation=export_voc_det_annotation,
            export_voc_seg_annotation=export_voc_seg_annotation,
            export_coco_det_annotation=export_coco_det_annotation,
            export_coco_seg_annotation=export_coco_seg_annotation,
            export_coco_pose_annotation=export_coco_pose_annotation,
            export_dota_annotation=export_dota_annotation,
            export_mask_annotation=export_mask_annotation,
            export_mot_annotation=export_mot_annotation,
            export_mots_annotation=export_mots_annotation,
            export_odvg_annotation=export_odvg_annotation,
            export_pporc_rec_annotation=export_pporc_rec_annotation,
            export_pporc_kie_annotation=export_pporc_kie_annotation,
            export_vlm_r1_ovd_annotation=export_vlm_r1_ovd_annotation,
            zoom=zoom,
            zoom_in=zoom_in,
            zoom_out=zoom_out,
            zoom_org=zoom_org,
            keep_prev_scale=keep_prev_scale,
            keep_prev_brightness=keep_prev_brightness,
            keep_prev_contrast=keep_prev_contrast,
            fit_window=fit_window,
            fit_width=fit_width,
            brightness_contrast=brightness_contrast,
            set_cross_line=set_cross_line,
            show_groups=show_groups,
            show_masks=show_masks,
            show_texts=show_texts,
            show_labels=show_labels,
            show_scores=show_scores,
            show_degrees=show_degrees,
            show_attributes=show_attributes,
            show_linking=show_linking,
            show_navigator=show_navigator,
            zoom_actions=zoom_actions,
            open_next_image=open_next_image,
            open_prev_image=open_prev_image,
            open_next_unchecked_image=open_next_unchecked_image,
            open_prev_unchecked_image=open_prev_unchecked_image,
            open_chatbot=open_chatbot,
            open_vqa=open_vqa,
            open_classifier=open_classifier,
            shape_manager=shape_manager,
            loop_thru_labels=loop_thru_labels,
            loop_select_labels=loop_select_labels,
            file_menu_actions=(
                open_,
                openvideo,
                opendir,
                save,
                save_as,
                close,
            ),
            tool=(),
            # XXX: need to add some actions here to activate the shortcut
            editMenu=(
                edit,
                duplicate,
                delete,
                copy,
                paste,
                None,
                undo,
                undo_last_point,
                None,
                copy_coordinates,
                remove_point,
                union_selection,
                None,
                keep_prev_mode,
                auto_use_last_label_mode,
                auto_use_last_gid_mode,
                use_system_clipboard,
                visibility_shapes_mode,
            ),
            # menu shown at right click
            menu=(
                create_mode,
                create_rectangle_mode,
                create_rotation_mode,
                create_circle_mode,
                create_line_mode,
                create_point_mode,
                create_line_strip_mode,
                edit_mode,
                edit,
                copy_coordinates,
                union_selection,
                duplicate,
                copy,
                paste,
                delete,
                undo,
                undo_last_point,
                remove_point,
            ),
            on_load_active=(
                close,
                create_mode,
                create_rectangle_mode,
                create_rotation_mode,
                create_circle_mode,
                create_line_mode,
                create_point_mode,
                create_line_strip_mode,
                digit_shortcut_0,
                digit_shortcut_1,
                digit_shortcut_2,
                digit_shortcut_3,
                digit_shortcut_4,
                digit_shortcut_5,
                digit_shortcut_6,
                digit_shortcut_7,
                digit_shortcut_8,
                digit_shortcut_9,
                edit_mode,
                brightness_contrast,
                shape_manager,
                loop_thru_labels,
                loop_select_labels,
            ),
            on_shapes_present=(save_as, delete),
            hide_selected_polygons=hide_selected_polygons,
            show_hidden_polygons=show_hidden_polygons,
            group_selected_shapes=group_selected_shapes,
            ungroup_selected_shapes=ungroup_selected_shapes,
        )

        self.canvas.vertex_selected.connect(
            self.actions.remove_point.setEnabled
        )

        self.menus = utils.Struct(
            file=self.menu(self.tr("File")),
            edit=self.menu(self.tr("Edit")),
            view=self.menu(self.tr("View")),
            language=self.menu(self.tr("Language")),
            upload=self.menu(self.tr("Upload")),
            export=self.menu(self.tr("Export")),
            tool=self.menu(self.tr("Tool")),
            train=self.menu(self.tr("Train")),
            help=self.menu(self.tr("Help")),
            recent_files=QtWidgets.QMenu(self.tr("Open Recent")),
            label_list=label_menu,
        )

        utils.add_actions(
            self.menus.file,
            (
                open_,
                open_next_image,
                open_prev_image,
                open_next_unchecked_image,
                open_prev_unchecked_image,
                opendir,
                openvideo,
                self.menus.recent_files,
                save,
                save_as,
                save_auto,
                change_output_dir,
                save_with_image_data,
                close,
                delete_file,
                delete_image_file,
                None,
            ),
        )
        utils.add_actions(self.menus.train, (ultralytics_train,))
        utils.add_actions(
            self.menus.tool,
            (
                overview,
                None,
                save_crop,
                None,
                digit_shortcut_manager,
                label_manager,
                gid_manager,
                shape_manager,
                None,
                hbb_to_obb,
                obb_to_hbb,
                polygon_to_hbb,
                polygon_to_obb,
                circle_to_polygon,
            ),
        )
        utils.add_actions(
            self.menus.help,
            (
                documentation,
                None,
                about,
            ),
        )
        utils.add_actions(
            self.menus.language,
            (
                select_lang_en,
                select_lang_zh,
            ),
        )
        utils.add_actions(
            self.menus.upload,
            (
                upload_image_flags_file,
                upload_label_flags_file,
                upload_shape_attrs_file,
                upload_label_classes_file,
                None,
                upload_yolo_hbb_annotation,
                upload_yolo_obb_annotation,
                upload_yolo_seg_annotation,
                upload_yolo_pose_annotation,
                None,
                upload_voc_det_annotation,
                upload_voc_seg_annotation,
                None,
                upload_coco_det_annotation,
                upload_coco_seg_annotation,
                upload_coco_pose_annotation,
                None,
                upload_dota_annotation,
                upload_mask_annotation,
                upload_mot_annotation,
                upload_odvg_annotation,
                upload_mmgd_annotation,
                None,
                upload_ppocr_rec_annotation,
                upload_ppocr_kie_annotation,
                None,
                upload_vlm_r1_ovd_annotation,
            ),
        )
        utils.add_actions(
            self.menus.export,
            (
                export_yolo_hbb_annotation,
                export_yolo_obb_annotation,
                export_yolo_seg_annotation,
                export_yolo_pose_annotation,
                None,
                export_voc_det_annotation,
                export_voc_seg_annotation,
                None,
                export_coco_det_annotation,
                export_coco_seg_annotation,
                export_coco_pose_annotation,
                None,
                export_dota_annotation,
                export_mask_annotation,
                export_odvg_annotation,
                None,
                export_mot_annotation,
                export_mots_annotation,
                None,
                export_pporc_rec_annotation,
                export_pporc_kie_annotation,
                None,
                export_vlm_r1_ovd_annotation,
            ),
        )
        utils.add_actions(
            self.menus.view,
            (
                self.flag_dock.toggleViewAction(),
                self.label_dock.toggleViewAction(),
                self.shape_dock.toggleViewAction(),
                self.file_dock.toggleViewAction(),
                None,
                show_navigator,
                fill_drawing,
                loop_thru_labels,
                loop_select_labels,
                None,
                zoom_in,
                zoom_out,
                zoom_org,
                None,
                keep_prev_scale,
                keep_prev_brightness,
                keep_prev_contrast,
                None,
                fit_window,
                fit_width,
                None,
                brightness_contrast,
                set_cross_line,
                show_masks,
                show_texts,
                show_labels,
                show_scores,
                show_degrees,
                show_attributes,
                show_linking,
                show_groups,
                hide_selected_polygons,
                show_hidden_polygons,
                group_selected_shapes,
                ungroup_selected_shapes,
            ),
        )

        self.menus.file.aboutToShow.connect(self.update_file_menu)

        # Custom context menu for the canvas widget:
        utils.add_actions(self.canvas.menus[0], self.actions.menu)
        utils.add_actions(
            self.canvas.menus[1],
            (
                action("&Copy here", self.copy_shape),
                action("&Move here", self.move_shape),
            ),
        )

        self.tools = self.toolbar("Tools")
        # Menu buttons on Left
        self.actions.tool = (
            # open_,
            opendir,
            open_next_image,
            open_prev_image,
            save,
            delete_file,
            None,
            create_mode,
            self.actions.create_rectangle_mode,
            self.actions.create_rotation_mode,
            self.actions.create_circle_mode,
            self.actions.create_line_mode,
            self.actions.create_point_mode,
            self.actions.create_line_strip_mode,
            None,
            edit_mode,
            delete,
            undo,
            loop_thru_labels,
            loop_select_labels,
            run_all_images,
            toggle_auto_labeling_widget,
            None,
            open_chatbot,
            open_vqa,
            open_classifier,
            None,
            fit_width,
            zoom,
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.tools)
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        self.label_instruction = QLabel(self.get_labeling_instruction())
        self.label_instruction.setContentsMargins(0, 0, 0, 0)
        self.auto_labeling_widget = AutoLabelingWidget(self)
        self.auto_labeling_widget.auto_segmentation_requested.connect(
            self.on_auto_segmentation_requested
        )
        self.auto_labeling_widget.auto_segmentation_disabled.connect(
            self.on_auto_segmentation_disabled
        )
        self.canvas.auto_labeling_marks_updated.connect(
            self.auto_labeling_widget.on_new_marks
        )
        self.auto_labeling_widget.auto_labeling_mode_changed.connect(
            self.canvas.set_auto_labeling_mode
        )
        self.auto_labeling_widget.auto_decode_mode_changed.connect(
            self.canvas.set_auto_decode_mode
        )
        self.auto_labeling_widget.cropping_mode_changed.connect(
            self.auto_labeling_widget.model_manager.set_cropping_mode
        )
        self.auto_labeling_widget.clear_auto_decode_requested.connect(
            self.canvas.reset_auto_decode_state
        )
        self.canvas.auto_decode_requested.connect(
            self.on_auto_decode_requested
        )
        self.canvas.auto_decode_finish_requested.connect(
            self.auto_labeling_widget.on_finish_clicked
        )
        self.canvas.shape_hover_changed.connect(
            lambda: (
                self.update_navigator_shapes()
                if (
                    hasattr(self, "navigator_dialog")
                    and self.navigator_dialog.isVisible()
                )
                else None
            )
        )
        self.auto_labeling_widget.clear_auto_labeling_action_requested.connect(
            self.clear_auto_labeling_marks
        )
        self.auto_labeling_widget.finish_auto_labeling_object_action_requested.connect(
            self.finish_auto_labeling_object
        )
        self.auto_labeling_widget.cache_auto_label_changed.connect(
            self.set_cache_auto_label
        )
        self.auto_labeling_widget.model_manager.prediction_started.connect(
            lambda: self.canvas.set_loading(True, self.tr("Please wait..."))
        )
        self.auto_labeling_widget.model_manager.prediction_finished.connect(
            lambda: self.canvas.set_loading(False)
        )
        self.auto_labeling_widget.model_manager.prediction_finished.connect(
            self.update_thumbnail_display
        )
        self.auto_labeling_widget.model_manager.model_loaded.connect(
            self.update_thumbnail_display
        )
        self.next_files_changed.connect(
            self.auto_labeling_widget.model_manager.on_next_files_changed
        )
        # NOTE(jack): this is not needed for now
        # self.auto_labeling_widget.model_manager.request_next_files_requested.connect(
        #     lambda: self.inform_next_files(self.filename)
        # )
        self.auto_labeling_widget.hide()  # Hide by default
        central_layout.addWidget(self.label_instruction)
        central_layout.addSpacing(5)
        central_layout.addWidget(self.auto_labeling_widget)
        central_layout.addWidget(scroll_area)
        layout.addItem(central_layout)

        # Save central area for resize
        self._central_widget = scroll_area

        # Stretch central area (image view)
        layout.setStretch(1, 1)

        right_sidebar_layout = QVBoxLayout()
        right_sidebar_layout.setContentsMargins(0, 0, 0, 0)

        # Thumbnail image display
        self.thumbnail_pixmap = None
        self.thumbnail_container = QWidget()
        thumbnail_image_layout = QVBoxLayout()
        thumbnail_image_layout.setContentsMargins(2, 2, 2, 2)
        self.thumbnail_image_label = QLabel()
        self.thumbnail_image_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_image_label.mousePressEvent = utils.on_thumbnail_click(
            self
        )
        thumbnail_image_layout.addWidget(self.thumbnail_image_label)
        self.thumbnail_container.setLayout(thumbnail_image_layout)
        self.thumbnail_container.hide()
        right_sidebar_layout.addWidget(self.thumbnail_container)

        # Shape attributes
        self.shape_attributes = QLabel(self.tr("Attributes"))
        self.grid_layout = QGridLayout()
        self.scroll_area = QScrollArea()
        # Show vertical scrollbar as needed
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Disable horizontal scrollbar
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidgetResizable(True)
        # Create a container widget for the grid layout
        self.grid_layout_container = QWidget()
        self.grid_layout_container.setLayout(self.grid_layout)
        self.scroll_area.setWidget(self.grid_layout_container)
        if not self.attributes:
            self.shape_attributes.hide()
            self.scroll_area.hide()
        right_sidebar_layout.addWidget(
            self.shape_attributes, 0, Qt.AlignCenter
        )
        right_sidebar_layout.addWidget(self.scroll_area)

        # Shape text label with checkbox
        self.shape_text_label = QLabel("Object Text")
        self.shape_text_edit = QPlainTextEdit()
        self.description_checkbox = QCheckBox()
        self.description_checkbox.setChecked(True)
        self.description_checkbox.toggled.connect(
            self.toggle_description_visibility
        )

        description_header_layout = QHBoxLayout()
        description_header_layout.setContentsMargins(0, 2, 0, 2)
        description_header_layout.addStretch()
        description_header_layout.addWidget(self.shape_text_label)
        description_header_layout.addStretch()
        description_header_layout.addWidget(self.description_checkbox)
        description_header_widget = QWidget()
        description_header_widget.setLayout(description_header_layout)

        right_sidebar_layout.addWidget(description_header_widget)
        right_sidebar_layout.addWidget(self.shape_text_edit)
        right_sidebar_layout.addWidget(self.flag_dock)

        # Labels with checkbox
        self.labels_checkbox = QCheckBox()
        self.labels_checkbox.setChecked(True)
        self.labels_checkbox.toggled.connect(self.toggle_labels_visibility)

        labels_header_layout = QHBoxLayout()
        labels_header_layout.setContentsMargins(0, 2, 0, 2)
        labels_header_layout.addStretch()
        labels_title = QLabel(self.tr("Labels"))
        labels_header_layout.addWidget(labels_title)
        labels_header_layout.addStretch()
        labels_header_layout.addWidget(self.labels_checkbox)
        labels_header_widget = QWidget()
        labels_header_widget.setLayout(labels_header_layout)
        right_sidebar_layout.addWidget(labels_header_widget)

        # Hide the original dock title bar
        empty_widget = QWidget()
        empty_widget.setFixedHeight(0)
        self.label_dock.setTitleBarWidget(empty_widget)
        right_sidebar_layout.addWidget(self.label_dock)

        # Create a horizontal layout for the filters and select button
        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(5)
        filter_layout.addWidget(self.label_filter_combobox, 2)
        filter_layout.addWidget(self.gid_filter_combobox, 1)
        filter_layout.addWidget(self.select_toggle_button, 0)
        right_sidebar_layout.addLayout(filter_layout)
        right_sidebar_layout.addWidget(self.shape_dock)
        right_sidebar_layout.addWidget(self.file_dock)
        self.file_dock.setFeatures(QDockWidget.DockWidgetFloatable)
        dock_features = (
            ~QDockWidget.DockWidgetMovable
            | ~QDockWidget.DockWidgetFloatable
            | ~QDockWidget.DockWidgetClosable
        )
        rev_dock_features = ~dock_features
        self.label_dock.setFeatures(
            self.label_dock.features() & rev_dock_features
        )
        self.file_dock.setFeatures(
            self.file_dock.features() & rev_dock_features
        )
        self.flag_dock.setFeatures(
            self.flag_dock.features() & rev_dock_features
        )
        self.shape_dock.setFeatures(
            self.shape_dock.features() & rev_dock_features
        )

        self.shape_text_edit.textChanged.connect(self.shape_text_changed)

        layout.addItem(right_sidebar_layout)
        self.setLayout(layout)

        if output_file is not None and self._config["auto_save"]:
            logger.warning(
                "If `auto_save` argument is True, `output_file` argument "
                "is ignored and output filename is automatically "
                "set as IMAGE_BASENAME.json."
            )
        self.output_file = output_file
        self.output_dir = output_dir

        # Application state.
        self.image = QtGui.QImage()
        self.image_path = None
        self.recent_files = []
        self.max_recent = 7
        self.other_data = {}
        self.zoom_level = 100
        self.fit_window = False
        self.zoom_values = {}  # key=filename, value=(zoom_mode, zoom_value)
        self.brightness_contrast_values = {}
        self.scroll_values = {
            Qt.Horizontal: {},
            Qt.Vertical: {},
        }  # key=filename, value=scroll_value

        if filename is not None and osp.isdir(filename):
            self.import_image_folder(filename, load=False)
        else:
            self.filename = filename

        if config["file_search"]:
            self.file_search.setText(config["file_search"])
            self.file_search_changed()

        # XXX: Could be completely declarative.
        # Restore application settings.
        self.settings = QtCore.QSettings("anylabeling", "anylabeling")
        self.recent_files = self.settings.value("recent_files", []) or []
        size = self.settings.value("window/size", QtCore.QSize(600, 500))
        position = self.settings.value("window/position", QtCore.QPoint(0, 0))
        # state = self.settings.value("window/state", QtCore.QByteArray())
        self.resize(size)
        self.move(position)
        # or simply:
        # self.restoreGeometry(settings['window/geometry']

        # Populate the File menu dynamically.
        self.update_file_menu()

        # Since loading the file may take some time,
        # make sure it runs in the background.
        if self.filename is not None:
            self.queue_event(functools.partial(self.load_file, self.filename))

        # Callbacks:
        self.zoom_widget.valueChanged.connect(self.paint_canvas)

        self.populate_mode_actions()

        self.first_start = True
        if self.first_start:
            QWhatsThis.enterWhatsThisMode()

        self.set_text_editing(False)

        QtCore.QTimer.singleShot(100, self.restore_navigator_state)

    def restore_navigator_state(self) -> None:
        try:
            navigator_visible: bool = self.settings.value(
                "navigator/visible", False, type=bool
            )

            if navigator_visible:
                self.navigator_dialog.show()

                if hasattr(self, "image") and not self.image.isNull():
                    self.navigator_dialog.set_image(
                        QtGui.QPixmap.fromImage(self.image)
                    )
                    self.update_navigator_viewport()
                else:
                    self._should_restore_navigator = True

                # Restore geometry information
                geometry = self.settings.value("navigator/geometry")
                if geometry:
                    self.navigator_dialog.restoreGeometry(geometry)
                else:
                    # Fallback: restore position and size separately
                    saved_size = self.settings.value("navigator/size")
                    saved_position = self.settings.value("navigator/position")

                    if saved_size:
                        self.navigator_dialog.resize(saved_size)
                    if saved_position:
                        self.navigator_dialog.move(saved_position)

                if hasattr(self, "actions") and hasattr(
                    self.actions, "show_navigator"
                ):
                    self.actions.show_navigator.setChecked(True)

        except Exception as e:
            print(f"Error restoring navigator state: {e}")

    def _navigator_close_event(self, event: QtGui.QCloseEvent) -> None:
        if hasattr(self, "actions") and hasattr(
            self.actions, "show_navigator"
        ):
            self.actions.show_navigator.setChecked(False)

        self.settings.setValue("navigator/visible", False)

        NavigatorDialog.closeEvent(self.navigator_dialog, event)

    def set_language(self, language):
        if self._config["language"] == language:
            return
        self._config["language"] = language

        # Show dialog to restart application
        msg_box = QMessageBox()
        msg_box.setText(
            self.tr("Please restart the application to apply changes.")
        )
        msg_box.exec_()
        self.parent.parent.close()

    def get_labeling_instruction(self):
        text_mode = self.tr("Mode:")
        text_shortcuts = self.tr("Shortcuts:")
        text_chatbot = self.tr("Chatbot")
        text_vqa = self.tr("VQA")
        text_classifier = self.tr("Classifier")
        text_previous = self.tr("Previous")
        text_next = self.tr("Next")
        text_rectangle = self.tr("Rectangle")
        text_polygon = self.tr("Polygon")
        text_rotation = self.tr("Rotation")
        return (
            f"<b>{text_mode}</b> {self.canvas.get_mode()} | "
            f"<b>{text_shortcuts}</b>"
            f" {text_previous}(<b>A</b>),"
            f" {text_next}(<b>D</b>),"
            f" {text_rectangle}(<b>R</b>),"
            f" {text_polygon}(<b>P</b>),"
            f" {text_rotation}(<b>O</b>),"
            f" {text_chatbot}(<b>Ctrl+1</b>),"
            f" {text_vqa}(<b>Ctrl+2</b>),"
            f" {text_classifier}(<b>Ctrl+3</b>)"
        )

    @pyqtSlot()
    def on_auto_segmentation_requested(self):
        self.canvas.set_auto_labeling(True)
        self.label_instruction.setText(self.get_labeling_instruction())

    @pyqtSlot()
    def on_auto_segmentation_disabled(self):
        self.canvas.set_auto_labeling(False)
        self.label_instruction.setText(self.get_labeling_instruction())

    @pyqtSlot(list)
    def on_exif_detected(self, exif_files):
        if utils.ExifProcessingDialog.show_detection_dialog(
            self, len(exif_files)
        ):
            logger.info("Start processing EXIF orientation")
            utils.ExifProcessingDialog.process_exif_files_with_progress(
                self, exif_files
            )

    @pyqtSlot(list)
    def on_auto_decode_requested(self, marks):
        """Handle auto decode request"""
        self.auto_labeling_widget.model_manager.set_auto_labeling_marks(marks)
        self.auto_labeling_widget.run_prediction()

    def menu(self, title, actions=None):
        menu = self.parent.parent.menuBar().addMenu(title)
        if actions:
            utils.add_actions(menu, actions)
        return menu

    def central_widget(self):
        return self._central_widget

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(f"{title}ToolBar")
        toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        toolbar.setIconSize(QtCore.QSize(24, 24))
        toolbar.setMaximumWidth(40)
        if actions:
            utils.add_actions(toolbar, actions)
        return toolbar

    def statusBar(self):
        return self.parent.parent.statusBar()

    def no_shape(self):
        return len(self.label_list) == 0

    def populate_mode_actions(self):
        tool = self.actions.tool
        menu = self.actions.menu
        self.tools.clear()
        utils.add_actions(self.tools, tool)

        self.canvas.menus[0].clear()
        utils.add_actions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (
            self.actions.create_mode,
            self.actions.create_rectangle_mode,
            self.actions.create_rotation_mode,
            self.actions.create_circle_mode,
            self.actions.create_line_mode,
            self.actions.create_point_mode,
            self.actions.create_line_strip_mode,
            self.actions.digit_shortcut_0,
            self.actions.digit_shortcut_1,
            self.actions.digit_shortcut_2,
            self.actions.digit_shortcut_3,
            self.actions.digit_shortcut_4,
            self.actions.digit_shortcut_5,
            self.actions.digit_shortcut_6,
            self.actions.digit_shortcut_7,
            self.actions.digit_shortcut_8,
            self.actions.digit_shortcut_9,
            self.actions.edit_mode,
        )
        utils.add_actions(self.menus.edit, actions + self.actions.editMenu)

    def set_dirty(self):
        # Even if we autosave the file, we keep the ability to undo
        self.actions.undo.setEnabled(self.canvas.is_shape_restorable)

        if self._config["auto_save"]:
            label_file = osp.splitext(self.image_path)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = self.output_dir + "/" + label_file_without_path
            self.save_labels(label_file)
            if (
                hasattr(self, "navigator_dialog")
                and self.navigator_dialog.isVisible()
            ):
                self.update_navigator_shapes()
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
        if (
            hasattr(self, "navigator_dialog")
            and self.navigator_dialog.isVisible()
        ):
            self.update_navigator_shapes()
        title = __appname__
        if self.filename is not None:
            title = f"{title} - {self.filename}*"
        self.setWindowTitle(title)

    def set_clean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.union_selection.setEnabled(False)
        self.actions.create_mode.setEnabled(True)
        self.actions.create_rectangle_mode.setEnabled(True)
        self.actions.create_rotation_mode.setEnabled(True)
        self.actions.create_circle_mode.setEnabled(True)
        self.actions.create_line_mode.setEnabled(True)
        self.actions.create_point_mode.setEnabled(True)
        self.actions.create_line_strip_mode.setEnabled(True)
        self.actions.digit_shortcut_0.setEnabled(True)
        self.actions.digit_shortcut_1.setEnabled(True)
        self.actions.digit_shortcut_2.setEnabled(True)
        self.actions.digit_shortcut_3.setEnabled(True)
        self.actions.digit_shortcut_4.setEnabled(True)
        self.actions.digit_shortcut_5.setEnabled(True)
        self.actions.digit_shortcut_6.setEnabled(True)
        self.actions.digit_shortcut_7.setEnabled(True)
        self.actions.digit_shortcut_8.setEnabled(True)
        self.actions.digit_shortcut_9.setEnabled(True)
        title = __appname__
        if self.filename is not None:
            current_index, total_count = self.get_image_progress_info()
            basename = osp.basename(str(self.filename))
            title = f"{title} - {basename} [{current_index}/{total_count}]"
        self.parent.parent.setWindowTitle(title)

        if self.has_label_file():
            self.actions.delete_file.setEnabled(True)
        else:
            self.actions.delete_file.setEnabled(False)

    def get_image_progress_info(self):
        if self.filename and self.filename in self.fn_to_index:
            current_index = self.fn_to_index[str(self.filename)]
            total_count = len(self.image_list)
            return current_index + 1, total_count
        return 1, 1

    def toggle_actions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for action in self.actions.zoom_actions:
            action.setEnabled(value)
        for action in self.actions.on_load_active:
            action.setEnabled(value)

        if value and self.file_list_widget.count() > 0:
            self.actions.shape_manager.setEnabled(True)
        else:
            self.actions.shape_manager.setEnabled(False)

    def queue_event(self, function):
        QtCore.QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def reset_state(self):
        self.label_list.clear()
        self.filename = None
        self.image_path = None
        self.image_data = None
        self.label_file = None
        self.other_data = {}
        self.canvas.reset_state()
        self.label_filter_combobox.text_box.clear()
        self.gid_filter_combobox.gid_box.clear()
        self.select_toggle_button.setChecked(False)
        self.select_toggle_button.setText(self.tr("Select"))

    def toggle_select_all(self):
        if not self.canvas.shapes:
            if self.select_toggle_button.isChecked():
                self.select_toggle_button.setText(self.tr("Unselect"))
            else:
                self.select_toggle_button.setText(self.tr("Select"))
            return

        if self.select_toggle_button.isChecked():
            self.canvas.select_shapes(self.canvas.shapes)
            self.select_toggle_button.setText(self.tr("Unselect"))
        else:
            self.canvas.select_shapes([])
            self.select_toggle_button.setText(self.tr("Select"))

    def reset_attribute(self, text):
        # Skip validation for auto-labeling special constants
        if text in [
            AutoLabelingMode.OBJECT,
            AutoLabelingMode.ADD,
            AutoLabelingMode.REMOVE,
        ]:
            return text

        valid_labels = list(self.attributes.keys())
        if text not in valid_labels:
            most_similar_label = utils.find_most_similar_label(
                text, valid_labels
            )
            self.error_message(
                self.tr("Invalid label"),
                self.tr(
                    "Invalid label '{}' with validation type: {}!\n"
                    "Reset the label as {}."
                ).format(text, valid_labels, most_similar_label),
            )
            text = most_similar_label
        return text

    def current_item(self):
        items = self.label_list.selected_items()
        if items:
            return items[0]
        return None

    def add_recent_file(self, filename):
        if filename in self.recent_files:
            self.recent_files.remove(filename)
        elif len(self.recent_files) >= self.max_recent:
            self.recent_files.pop()
        self.recent_files.insert(0, filename)

    # Callbacks
    def undo_shape_edit(self):
        self.canvas.restore_shape()
        self.label_list.clear()
        self.load_shapes(self.canvas.shapes, update_last_label=False)
        self.actions.undo.setEnabled(self.canvas.is_shape_restorable)
        self.set_dirty()

    def get_label_file_list(self):
        label_file_list = []
        if not self.image_list and self.filename:
            dir_path, filename = osp.split(self.filename)
            label_file = osp.join(
                dir_path, osp.splitext(filename)[0] + ".json"
            )
            if osp.exists(label_file):
                label_file_list = [label_file]
        elif self.image_list and not self.output_dir and self.filename:
            file_list = os.listdir(osp.dirname(self.filename))
            for file_name in file_list:
                if not file_name.endswith(".json"):
                    continue
                label_file_list.append(
                    osp.join(osp.dirname(self.filename), file_name)
                )
        if self.output_dir:
            for file_name in os.listdir(self.output_dir):
                if not file_name.endswith(".json"):
                    continue
                label_file_list.append(osp.join(self.output_dir, file_name))
        return label_file_list

    def copy_shape_coordinates(self):
        item = self.current_item()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return

        points = shape.points
        if shape.shape_type == "rectangle":
            if len(points) >= 2:
                x1, y1 = points[0].x(), points[0].y()
                x2, y2 = points[2].x(), points[2].y()
                coordinates = [x1, y1, x2, y2]
                coordinates = list(map(int, coordinates))
            else:
                return
        else:
            coordinates = []
            for point in points:
                coordinates.extend([point.x(), point.y()])

        coordinates_str = str(coordinates)
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(coordinates_str)

    def union_selection(self):
        rectangle_shapes, polygon_shapes = [], []
        for shape in self.canvas.selected_shapes:
            points = shape.points
            if shape.shape_type == "rectangle":
                xmin, ymin = (points[0].x(), points[0].y())
                xmax, ymax = (points[2].x(), points[2].y())
                rectangle_shapes.append([xmin, ymin, xmax, ymax])
            else:
                polygon_shapes.append([(p.x(), p.y()) for p in points])

        union_shape = shape.copy()

        if len(rectangle_shapes) > 0:
            min_x = min([bbox[0] for bbox in rectangle_shapes])
            min_y = min([bbox[1] for bbox in rectangle_shapes])
            max_x = max([bbox[2] for bbox in rectangle_shapes])
            max_y = max([bbox[3] for bbox in rectangle_shapes])

            union_shape.points[0].setX(min_x)
            union_shape.points[0].setY(min_y)
            union_shape.points[1].setX(max_x)
            union_shape.points[1].setY(min_y)
            union_shape.points[2].setX(max_x)
            union_shape.points[2].setY(max_y)
            union_shape.points[3].setX(min_x)
            union_shape.points[3].setY(max_y)
        else:
            # Create a blank mask
            min_x = min([min(p[0] for p in poly) for poly in polygon_shapes])
            min_y = min([min(p[1] for p in poly) for poly in polygon_shapes])
            max_x = max([max(p[0] for p in poly) for poly in polygon_shapes])
            max_y = max([max(p[1] for p in poly) for poly in polygon_shapes])

            width = int(max_x - min_x + 10)
            height = int(max_y - min_y + 10)
            mask = np.zeros((height, width), dtype=np.uint8)

            # Draw all polygons on the mask
            for polygon in polygon_shapes:
                contour = np.array(polygon, dtype=np.int32)
                shifted_contour = contour - np.array(
                    [min_x - 5, min_y - 5], dtype=np.int32
                )
                shifted_contour = shifted_contour.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [shifted_contour], 255)

            # Find contours of the merged shape
            merged_contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if merged_contours:
                largest_contour = max(merged_contours, key=cv2.contourArea)
                epsilon = 0.001 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(
                    largest_contour, epsilon, True
                )
                approx_contour = approx_contour.reshape(-1, 2) + np.array(
                    [min_x - 5, min_y - 5], dtype=np.int32
                )
                union_shape.points = [
                    QtCore.QPointF(float(x), float(y))
                    for x, y in approx_contour
                ]

        # Append merged shape and remove selected shapes
        self.add_label(union_shape)
        self.remove_labels(self.canvas.delete_selected())
        self.set_dirty()

        # Update UI state
        if self.no_shape():
            for action in self.actions.on_shapes_present:
                action.setEnabled(False)

    # Trainer
    def start_training(self, mode):
        if mode == "ultralytics":
            dialog = UltralyticsDialog(self)
        else:
            return

        try:
            _ = dialog.exec_()
        except Exception as e:
            self.error_message(
                "Start Error", f"Failed to start training dialog: {str(e)}"
            )

    # Tools
    def overview(self):
        if self.filename:
            OverviewDialog(parent=self)

    def digit_shortcut_manager(self):
        digit_shortcut_dialog = DigitShortcutDialog(parent=self)
        result = digit_shortcut_dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            self._config["digit_shortcuts"] = self.drawing_digit_shortcuts
            save_config(self._config)

    def label_manager(self):
        modify_label_dialog = LabelModifyDialog(
            parent=self, opacity=LABEL_OPACITY
        )
        result = modify_label_dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            if self.filename:
                self.load_file(self.filename)

    def gid_manager(self):
        modify_gid_dialog = GroupIDModifyDialog(parent=self)
        result = modify_gid_dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            self.load_file(self.filename)

    def shape_manager(self):
        modify_shape_dialog = ShapeModifyDialog(parent=self)
        result = modify_shape_dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            if modify_shape_dialog.need_reload and self.filename:
                self.load_file(self.filename)

    def open_chatbot(self):
        dialog = ChatbotDialog(self)
        _ = dialog.exec_()

    def open_vqa(self):
        if not self.image_list:
            self.error_message(
                self.tr("No images loaded"),
                self.tr(
                    "Please load an image folder before opening the VQA dialog."
                ),
            )
            return

        if not hasattr(self, "vqa_window") or self.vqa_window is None:
            self.vqa_window = VQADialog(self)
            self.vqa_window.setAttribute(Qt.WA_DeleteOnClose, False)
        if self.vqa_window.isVisible():
            self.vqa_window.raise_()
            self.vqa_window.activateWindow()
        else:
            self.vqa_window.show()

    def open_classifier(self):
        if not self.image_list:
            self.error_message(
                self.tr("No images loaded"),
                self.tr(
                    "Please load an image folder before opening the Classification dialog."
                ),
            )
            return

        main_window = self
        while True:
            try:
                parent = main_window.parent()
            except TypeError:
                parent = getattr(main_window, "parent", None)
            if parent is None:
                break
            main_window = parent
        main_window.hide()

        dialog = ClassifierDialog(self)
        dialog.exec_()

    # Help
    def documentation(self):
        url = (
            "https://github.com/CVHub520/X-AnyLabeling/tree/main/docs"  # NOQA
        )
        utils.general.open_url(url)

    def about(self):
        about_dialog = AboutDialog(self)
        _ = about_dialog.exec_()

    def loop_thru_labels(self):
        self.label_loop_count += 1
        if len(self.label_list) == 0 or self.label_loop_count >= len(
            self.label_list
        ):
            # If we go through all the things go back to 100%
            self.label_loop_count = -1
            self.set_zoom(int(100 * self.scale_fit_window()))
            return

        width = self.central_widget().width() - 2.0
        height = self.central_widget().height() - 2.0

        im_width = self.canvas.pixmap.width()
        im_height = self.canvas.pixmap.height()

        zoom_scale = 4

        item = self.label_list[self.label_loop_count]
        xs = []
        ys = []
        # loop through all points on this label
        for point in item.shape().points:
            xs.append(point.x())
            ys.append(point.y())

        # Set minimum label width to 30px this should handle point
        # lables and very tiny labels gracefully
        label_width = max(int(max(xs) - min(xs)), 30)
        x = (max(xs) + min(xs)) / 2
        y = (max(ys) + min(ys)) / 2

        zoom = int(100 * width / (zoom_scale * label_width))
        # Don't go past the max zoom which is 1000
        zoom = min(1000, zoom)

        self.set_zoom(zoom)

        x_range = self.scroll_bars[Qt.Horizontal].maximum()
        x_step = self.scroll_bars[Qt.Horizontal].pageStep()

        y_range = self.scroll_bars[Qt.Vertical].maximum()
        # QT docs says Document length = maximum() - minimum() + pageStep().
        # so there's a weird pageStep thing we gotta add
        y_step = self.scroll_bars[Qt.Vertical].pageStep()
        screen_width = width / (zoom / 100)
        # add half a screen to this
        x_scroll = int((x - screen_width / 2) / im_width * (x_range + x_step))
        x_scroll = min(max(0, x_scroll), x_range)

        screen_height = height / (zoom / 100)

        y_scroll = int(
            (y - screen_height / 2) / (im_height) * (y_range + y_step)
        )
        y_scroll = min(max(0, y_scroll), y_range)

        self.set_scroll(Qt.Horizontal, x_scroll)
        self.set_scroll(Qt.Vertical, y_scroll)
        for shape in self.canvas.selected_shapes:
            shape.selected = False
        self.canvas.prev_h_shape = self.canvas.h_hape = item.shape()
        self.canvas.update()

    def loop_select_labels(self):
        self.select_loop_count += 1
        if len(self.label_list) == 0 or self.select_loop_count >= len(
            self.label_list
        ):
            self.select_loop_count = -1
            self.canvas.deselect_shape()
            return

        item = self.label_list[self.select_loop_count]
        shape = item.shape()
        self.canvas.select_shapes([shape])

    def copy_to_clipboard(self, text):
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(
            self,
            self.tr("Copied"),
            self.tr("The information has been copied to the clipboard."),
        )

    # General
    def toggle_drawing_sensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.edit_mode.setEnabled(not drawing)
        self.actions.undo_last_point.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)
        self.actions.union_selection.setEnabled(not drawing)

    def create_digit_mode(self, digit_num):
        if self.drawing_digit_shortcuts is None:
            return

        data = self.drawing_digit_shortcuts.get(digit_num, None)
        if not data:
            return

        label = data.get("label", "object")
        create_mode = data.get("mode", None)

        if create_mode not in [
            "polygon",
            "rectangle",
            "rotation",
            "circle",
            "line",
            "point",
            "linestrip",
        ]:
            return

        self.digit_to_label = label
        self.toggle_draw_mode(edit=False, create_mode=create_mode)

    def toggle_draw_mode(
        self, edit=True, create_mode="rectangle", disable_auto_labeling=True
    ):
        # Disable auto labeling if needed
        if (
            disable_auto_labeling
            and self.auto_labeling_widget.auto_labeling_mode
            != AutoLabelingMode.NONE
        ):
            self.clear_auto_labeling_marks()
            self.auto_labeling_widget.set_auto_labeling_mode(None)

        self.set_text_editing(False)

        self.canvas.set_editing(edit)
        self.canvas.create_mode = create_mode
        if edit:
            self.actions.create_mode.setEnabled(True)
            self.actions.create_rectangle_mode.setEnabled(True)
            self.actions.create_rotation_mode.setEnabled(True)
            self.actions.create_circle_mode.setEnabled(True)
            self.actions.create_line_mode.setEnabled(True)
            self.actions.create_point_mode.setEnabled(True)
            self.actions.create_line_strip_mode.setEnabled(True)
            self.actions.digit_shortcut_0.setEnabled(True)
            self.actions.digit_shortcut_1.setEnabled(True)
            self.actions.digit_shortcut_2.setEnabled(True)
            self.actions.digit_shortcut_3.setEnabled(True)
            self.actions.digit_shortcut_4.setEnabled(True)
            self.actions.digit_shortcut_5.setEnabled(True)
            self.actions.digit_shortcut_6.setEnabled(True)
            self.actions.digit_shortcut_7.setEnabled(True)
            self.actions.digit_shortcut_8.setEnabled(True)
            self.actions.digit_shortcut_9.setEnabled(True)
        else:
            self.hide_attributes_panel()
            self.actions.union_selection.setEnabled(False)
            if create_mode == "polygon":
                self.actions.create_mode.setEnabled(False)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "rectangle":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(False)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "line":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(False)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "point":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(False)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "circle":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(False)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            elif create_mode == "linestrip":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(True)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(False)
            elif create_mode == "rotation":
                self.actions.create_mode.setEnabled(True)
                self.actions.create_rectangle_mode.setEnabled(True)
                self.actions.create_rotation_mode.setEnabled(False)
                self.actions.create_circle_mode.setEnabled(True)
                self.actions.create_line_mode.setEnabled(True)
                self.actions.create_point_mode.setEnabled(True)
                self.actions.create_line_strip_mode.setEnabled(True)
            else:
                raise ValueError(f"Unsupported create_mode: {create_mode}")
        self.actions.edit_mode.setEnabled(not edit)
        self.label_instruction.setText(self.get_labeling_instruction())

    def set_edit_mode(self):
        # Disable auto labeling
        self.clear_auto_labeling_marks()
        self.auto_labeling_widget.set_auto_labeling_mode(None)

        self.toggle_draw_mode(True)
        self.set_text_editing(True)
        self.label_instruction.setText(self.get_labeling_instruction())

    def update_file_menu(self):
        current = self.filename

        def exists(filename):
            return osp.exists(str(filename))

        menu = self.menus.recent_files
        menu.clear()
        files = [f for f in self.recent_files if f != current and exists(f)]
        for i, f in enumerate(files):
            icon = utils.new_icon("labels")
            action = QtWidgets.QAction(
                icon, "&%d %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(functools.partial(self.load_recent, f))
            menu.addAction(action)

    def pop_label_list_menu(self, point):
        self.menus.label_list.exec_(self.label_list.mapToGlobal(point))

    def validate_label(self, label):
        # no validation
        if self._config["validate_label"] is None:
            return True

        for i in range(self.unique_label_list.count()):
            label_i = self.unique_label_list.item(i).data(Qt.UserRole)
            if self._config["validate_label"] in ["exact"]:
                if label_i == label:
                    return True
        return False

    def batch_edit_labels(self, shapes):
        if not self._batch_edit_warning_shown:
            reply = QtWidgets.QMessageBox.question(
                self,
                self.tr("Batch Edit"),
                self.tr(
                    "You are about to edit multiple shapes in batch mode. "
                    "This operation cannot be undone.\n\n"
                    "This warning will only be shown once. Do you want to continue?"
                ),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )

            if reply != QtWidgets.QMessageBox.Yes:
                return

            self._batch_edit_warning_shown = True

        first_shape = shapes[0]
        result = self.label_dialog.pop_up(
            text=first_shape.label,
            flags=first_shape.flags,
            group_id=first_shape.group_id,
            description=first_shape.description,
            difficult=first_shape.difficult,
            kie_linking=first_shape.kie_linking,
            move_mode="center",
        )

        if result[0] is None:
            return

        text, flags, group_id, description, difficult, kie_linking = result

        if not self.validate_label(text):
            self.error_message(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        for shape in shapes:
            if self.attributes and text:
                text = self.reset_attribute(text)

            shape.label = text
            shape.flags = flags
            shape.group_id = group_id
            shape.description = description
            shape.difficult = difficult
            shape.kie_linking = kie_linking

            self._update_shape_color(shape)

            item = self.label_list.find_item_by_shape(shape)
            if item is not None:
                if shape.group_id is None:
                    color = shape.fill_color.getRgb()[:3]
                    item.setText("{}".format(html.escape(shape.label)))
                    item.setBackground(QtGui.QColor(*color, LABEL_OPACITY))
                else:
                    item.setText(f"{shape.label} ({shape.group_id})")

        self.label_dialog.add_label_history(text)

        if not self.unique_label_list.find_items_by_label(text):
            unique_label_item = self.unique_label_list.create_item_from_label(
                text
            )
            self.unique_label_list.addItem(unique_label_item)
            rgb = self._get_rgb_by_label(text)
            self.unique_label_list.set_item_label(
                unique_label_item, text, rgb, LABEL_OPACITY
            )

        self.set_dirty()
        self.update_combo_box()
        self.update_gid_box()

    def edit_label(self, item=None):
        if item and not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem type")

        if not self.canvas.editing():
            return

        selected_shapes = self.canvas.selected_shapes
        if not selected_shapes:
            return

        if len(selected_shapes) > 1:
            return self.batch_edit_labels(selected_shapes)

        if not item:
            item = self.current_item()
        if item is None:
            return
        shape = item.shape()
        if shape is None:
            return
        (
            text,
            flags,
            group_id,
            description,
            difficult,
            kie_linking,
        ) = self.label_dialog.pop_up(
            text=shape.label,
            flags=shape.flags,
            group_id=shape.group_id,
            description=shape.description,
            difficult=shape.difficult,
            kie_linking=shape.kie_linking,
            move_mode=self._config.get("move_mode", "auto"),
        )
        if text is None:
            return
        if not self.validate_label(text):
            self.error_message(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return
        if self.attributes and text:
            text = self.reset_attribute(text)
        shape.label = text
        shape.flags = flags
        shape.group_id = group_id
        shape.description = description
        shape.difficult = difficult
        shape.kie_linking = kie_linking

        # Add to label history
        self.label_dialog.add_label_history(shape.label)

        # Update last group_id
        if group_id is not None:
            self.label_dialog._last_gid = group_id

        # Update unique label list
        if not self.unique_label_list.find_items_by_label(shape.label):
            unique_label_item = self.unique_label_list.create_item_from_label(
                shape.label
            )
            self.unique_label_list.addItem(unique_label_item)
            rgb = self._get_rgb_by_label(shape.label)
            self.unique_label_list.set_item_label(
                unique_label_item, shape.label, rgb, LABEL_OPACITY
            )

        self._update_shape_color(shape)
        if shape.group_id is None:
            color = shape.fill_color.getRgb()[:3]
            item.setText("{}".format(html.escape(shape.label)))
            item.setBackground(QtGui.QColor(*color, LABEL_OPACITY))
        else:
            item.setText(f"{shape.label} ({shape.group_id})")
        self.set_dirty()
        self.update_combo_box()
        self.update_gid_box()

    def file_search_changed(self):
        search_text = self.file_search.text()
        self.import_image_folder(
            self.last_open_dir,
            pattern=search_text,
            load=False,
        )

    def file_selection_changed(self):
        items = self.file_list_widget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.may_continue():
            return

        current_index = self.fn_to_index[str(item.text())]
        if current_index < len(self.image_list):
            filename = self.image_list[current_index]
            if filename:
                self.load_file(filename)
                if self.attributes:
                    # Clear the history widgets from the QGridLayout
                    self.grid_layout = QGridLayout()
                    self.grid_layout_container = QWidget()
                    self.grid_layout_container.setLayout(self.grid_layout)
                    self.scroll_area.setWidget(self.grid_layout_container)
                    self.scroll_area.setWidgetResizable(True)
                    # Create a container widget for the grid layout
                    self.grid_layout_container = QWidget()
                    self.grid_layout_container.setLayout(self.grid_layout)
                    self.scroll_area.setWidget(self.grid_layout_container)

    def attribute_selection_changed(self, i, property, combo):
        selected_option = combo.currentText()
        if i < len(self.canvas.shapes):
            if not self.canvas.shapes[i].attributes:
                self.canvas.shapes[i].attributes = {}
            self.canvas.shapes[i].attributes[property] = selected_option
            self.save_attributes(self.canvas.shapes)

    def attribute_radio_changed(self, i, property, option, checked):
        if checked and i < len(self.canvas.shapes):
            if not self.canvas.shapes[i].attributes:
                self.canvas.shapes[i].attributes = {}
            self.canvas.shapes[i].attributes[property] = option
            self.save_attributes(self.canvas.shapes)

    def update_selected_options(self, selected_options):
        if not isinstance(selected_options, dict):
            return

        row_count = self.grid_layout.rowCount()
        for row in range(row_count):
            category_label = None
            property_widget = None
            if self.grid_layout.itemAtPosition(row, 0):
                category_label = self.grid_layout.itemAtPosition(
                    row, 0
                ).widget()
            if self.grid_layout.itemAtPosition(row, 1):
                property_widget = self.grid_layout.itemAtPosition(
                    row, 1
                ).widget()
            if category_label and property_widget:
                category = category_label.text()
                if category in selected_options:
                    selected_option = selected_options[category]

                    if isinstance(property_widget, QComboBox):
                        index = property_widget.findText(selected_option)
                        if index >= 0:
                            property_widget.setCurrentIndex(index)
                    elif isinstance(property_widget, QWidget):
                        for child in property_widget.findChildren(
                            QRadioButton
                        ):
                            if child.text() == selected_option:
                                child.setChecked(True)
                                break
        return

    def update_attributes(self, shape_index):
        if shape_index >= len(self.canvas.shapes) or shape_index < 0:
            self.hide_attributes_panel()
            return

        update_shape = self.canvas.shapes[shape_index]
        update_category = update_shape.label
        if update_category not in self.attributes:
            self.hide_attributes_panel()
            return

        current_attibute = self.attributes[update_category]
        if not update_shape.attributes:
            update_shape.attributes = {}

        self.grid_layout = QGridLayout()
        row_counter = 0

        for property, options in current_attibute.items():
            widget_type = self.attribute_widget_types.get(
                update_category, {}
            ).get(property, "combobox")
            current_value = update_shape.attributes.get(property, None)
            if hasattr(self, "grid_layout_container"):
                font_metrics = QFontMetrics(self.grid_layout_container.font())
            else:
                font_metrics = QLabel().font()
            available_width = self.scroll_area.width() - 30
            property_display = property
            if font_metrics.width(property) > available_width:
                while (
                    font_metrics.width(property_display + "...")
                    > available_width
                    and len(property_display) > 1
                ):
                    property_display = property_display[:-1]
                property_display += "..."

            property_label = QLabel(property_display)
            if property_display != property:
                property_label.setToolTip(property)

            self.grid_layout.addWidget(property_label, row_counter, 0, 1, 2)
            row_counter += 1

            if widget_type == "radiobutton":
                radio_group = QButtonGroup()
                radio_container = QWidget()
                main_layout = QVBoxLayout()
                main_layout.setContentsMargins(0, 0, 0, 0)
                main_layout.setSpacing(2)

                def get_truncated_text(text, max_width):
                    if font_metrics.width(text) <= max_width:
                        return text, text
                    truncated = text
                    while (
                        font_metrics.width(truncated + "...") > max_width
                        and len(truncated) > 1
                    ):
                        truncated = truncated[:-1]
                    return truncated + "...", text

                def get_button_width(text):
                    return font_metrics.width(text) + 30

                def create_radio_button_with_handler(
                    display_text, original_text, prop, shape_idx
                ):
                    radio_button = QRadioButton(display_text)
                    if display_text != original_text:
                        radio_button.setToolTip(original_text)
                    radio_group.addButton(radio_button)

                    def handler(checked):
                        if checked:
                            self.attribute_radio_changed(
                                shape_idx, prop, original_text, checked
                            )

                    radio_button.toggled.connect(handler)
                    return radio_button

                buttons_data = []
                for option in options:
                    display_text, original_text = get_truncated_text(
                        option, available_width
                    )
                    button_width = get_button_width(display_text)
                    buttons_data.append(
                        (display_text, original_text, button_width)
                    )

                current_row_buttons = []
                current_row_width = 0

                idx = 0
                while idx < len(buttons_data):
                    display_text, original_text, button_width = buttons_data[
                        idx
                    ]

                    if not current_row_buttons:
                        current_row_buttons.append(
                            (display_text, original_text)
                        )
                        current_row_width = button_width
                        idx += 1
                        continue

                    if current_row_width + button_width <= available_width:
                        current_row_buttons.append(
                            (display_text, original_text)
                        )
                        current_row_width += button_width
                        idx += 1
                    else:
                        if len(current_row_buttons) == 1:
                            first_display, first_original = (
                                current_row_buttons[0]
                            )
                            first_truncated, _ = get_truncated_text(
                                first_original, available_width - button_width
                            )
                            first_truncated_width = get_button_width(
                                first_truncated
                            )

                            if (
                                first_truncated_width + button_width
                                <= available_width
                            ):
                                current_row_buttons = [
                                    (first_truncated, first_original),
                                    (display_text, original_text),
                                ]
                                current_row_width = (
                                    first_truncated_width + button_width
                                )
                                idx += 1
                            else:
                                row_layout = QHBoxLayout()
                                row_layout.setContentsMargins(0, 0, 0, 0)
                                row_layout.setSpacing(4)

                                for (
                                    btn_display,
                                    btn_original,
                                ) in current_row_buttons:
                                    radio_button = (
                                        create_radio_button_with_handler(
                                            btn_display,
                                            btn_original,
                                            property,
                                            shape_index,
                                        )
                                    )
                                    row_layout.addWidget(radio_button)
                                    if current_value == btn_original or (
                                        current_value is None
                                        and btn_original == options[0]
                                    ):
                                        radio_button.setChecked(True)
                                        if current_value is None:
                                            update_shape.attributes[
                                                property
                                            ] = btn_original

                                row_layout.addStretch()
                                row_widget = QWidget()
                                row_widget.setLayout(row_layout)
                                main_layout.addWidget(row_widget)

                                current_row_buttons = []
                                current_row_width = 0
                                continue
                        else:
                            row_layout = QHBoxLayout()
                            row_layout.setContentsMargins(0, 0, 0, 0)
                            row_layout.setSpacing(4)
                            for (
                                btn_display,
                                btn_original,
                            ) in current_row_buttons:
                                radio_button = (
                                    create_radio_button_with_handler(
                                        btn_display,
                                        btn_original,
                                        property,
                                        shape_index,
                                    )
                                )
                                row_layout.addWidget(radio_button)
                                if current_value == btn_original or (
                                    current_value is None
                                    and btn_original == options[0]
                                ):
                                    radio_button.setChecked(True)
                                    if current_value is None:
                                        update_shape.attributes[property] = (
                                            btn_original
                                        )

                            row_layout.addStretch()
                            row_widget = QWidget()
                            row_widget.setLayout(row_layout)
                            main_layout.addWidget(row_widget)

                            current_row_buttons = []
                            current_row_width = 0
                            continue

                if current_row_buttons:
                    row_layout = QHBoxLayout()
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    row_layout.setSpacing(4)
                    for btn_display, btn_original in current_row_buttons:
                        radio_button = create_radio_button_with_handler(
                            btn_display, btn_original, property, shape_index
                        )
                        row_layout.addWidget(radio_button)
                        if current_value == btn_original or (
                            current_value is None
                            and btn_original == options[0]
                        ):
                            radio_button.setChecked(True)
                            if current_value is None:
                                update_shape.attributes[property] = (
                                    btn_original
                                )
                    row_layout.addStretch()
                    row_widget = QWidget()
                    row_widget.setLayout(row_layout)
                    main_layout.addWidget(row_widget)

                radio_container.setLayout(main_layout)
                self.grid_layout.addWidget(
                    radio_container, row_counter, 0, 1, 2
                )
                row_counter += 1
            else:
                property_combo = QComboBox()
                property_combo.addItems(options)
                if current_value:
                    index = property_combo.findText(current_value)
                    if index >= 0:
                        property_combo.setCurrentIndex(index)
                else:
                    update_shape.attributes[property] = options[0]
                property_combo.currentIndexChanged.connect(
                    lambda _, prop=property, combo=property_combo, shape_idx=shape_index: self.attribute_selection_changed(
                        shape_idx, prop, combo
                    )
                )
                self.grid_layout.addWidget(
                    property_combo, row_counter, 0, 1, 2
                )
                row_counter += 1

        self.grid_layout_container = QWidget()
        self.grid_layout_container.setLayout(self.grid_layout)
        self.scroll_area.setWidget(self.grid_layout_container)
        self.scroll_area.setWidgetResizable(True)
        if shape_index < len(self.canvas.shapes):
            self.canvas.shapes[shape_index] = update_shape
            self.save_attributes(self.canvas.shapes)
        self.show_attributes_panel()

    def show_attributes_panel(self):
        if hasattr(self, "scroll_area"):
            self.scroll_area.setVisible(True)

    def hide_attributes_panel(self):
        if hasattr(self, "scroll_area"):
            self.scroll_area.setVisible(False)

    def save_attributes(self, _shapes):
        filename = osp.splitext(self.image_path)[0] + ".json"
        if self.output_dir:
            label_file_without_path = osp.basename(filename)
            filename = osp.join(self.output_dir, label_file_without_path)
        label_file = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            info = {
                "label": s.label,
                "points": [(p.x(), p.y()) for p in s.points],
                "group_id": s.group_id,
                "description": s.description,
                "difficult": s.difficult,
                "shape_type": s.shape_type,
                "flags": s.flags,
                "attributes": s.attributes,
                "kie_linking": s.kie_linking,
            }
            if s.shape_type == "rotation":
                info["direction"] = s.direction
            data.update(info)

            return data

        # Get current shapes
        # Excluding auto labeling special shapes
        shapes = [
            format_shape(shape)
            for shape in _shapes
            if shape.label
            not in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]
        ]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            image_path = osp.relpath(self.image_path, osp.dirname(filename))
            image_data = (
                self.image_data if self._config["store_data"] else None
            )
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            label_file.save(
                filename=filename,
                shapes=shapes,
                image_path=image_path,
                image_data=image_data,
                image_height=self.image.height(),
                image_width=self.image.width(),
                other_data=self.other_data,
                flags=flags,
            )
            self.label_file = label_file
            items = self.file_list_widget.findItems(
                self.image_path, Qt.MatchExactly
            )
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.error_message(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    # React to canvas signals.
    def shape_selection_changed(self, selected_shapes):
        self._no_selection_slot = True
        for shape in self.canvas.selected_shapes:
            shape.selected = False
        self.label_list.clearSelection()
        self.canvas.selected_shapes = selected_shapes
        allow_merge_shape_type = {"rectangle": 0, "polygon": 0}
        for shape in self.canvas.selected_shapes:
            shape.selected = True
            if shape.shape_type in ["rectangle", "polygon"]:
                allow_merge_shape_type[shape.shape_type] += 1
            item = self.label_list.find_item_by_shape(shape)
            # NOTE: Handle the case when the shape is not found
            if item is not None:
                self.label_list.select_item(item)
                self.label_list.scroll_to_item(item)
        self._no_selection_slot = False
        n_selected = len(selected_shapes)
        same_type = (
            len(set(shape.shape_type for shape in selected_shapes)) <= 1
        )
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected >= 1 and same_type)
        self.actions.copy_coordinates.setEnabled(n_selected == 1)
        self.actions.union_selection.setEnabled(
            not all(value > 0 for value in allow_merge_shape_type.values())
            and (
                allow_merge_shape_type["rectangle"] > 1
                or allow_merge_shape_type["polygon"] > 1
            )
        )
        self.set_text_editing(True)

        selected_count = len(self.canvas.selected_shapes)
        is_drawing_mode = (
            hasattr(self.canvas, "current") and self.canvas.current is not None
        )
        if self.attributes and selected_count == 1 and not is_drawing_mode:
            for i in range(len(self.canvas.shapes)):
                if self.canvas.shapes[i].selected:
                    self.update_attributes(i)
                    break
        else:
            self.hide_attributes_panel()

    def add_label(self, shape, update_last_label=True):
        if shape.group_id is None:
            text = shape.label
        else:
            text = f"{shape.label} ({shape.group_id})"
        label_list_item = LabelListWidgetItem(text, shape)
        self.label_list.add_iem(label_list_item)
        if not self.unique_label_list.find_items_by_label(shape.label):
            item = self.unique_label_list.create_item_from_label(shape.label)
            self.unique_label_list.addItem(item)
            rgb = self._get_rgb_by_label(shape.label)
            self.unique_label_list.set_item_label(
                item, shape.label, rgb, LABEL_OPACITY
            )

        if shape.label not in self.label_info:
            rgb = self._get_rgb_by_label(shape.label)
            self.label_info[shape.label] = dict(
                delete=False,
                value=None,
                color=list(rgb),
                opacity=LABEL_OPACITY,
                visible=True,
            )

        # Add label to history if it is not a special label
        if shape.label not in [
            AutoLabelingMode.OBJECT,
            AutoLabelingMode.ADD,
            AutoLabelingMode.REMOVE,
        ]:
            self.label_dialog.add_label_history(
                shape.label, update_last_label=update_last_label
            )
            if update_last_label and shape.group_id is not None:
                self.label_dialog._last_gid = shape.group_id

        for action in self.actions.on_shapes_present:
            action.setEnabled(True)

        self._update_shape_color(shape)
        color = shape.fill_color.getRgb()[:3]
        label_list_item.setText("{}".format(html.escape(text)))
        label_list_item.setBackground(QtGui.QColor(*color, LABEL_OPACITY))
        self.update_combo_box()
        self.update_gid_box()

    def load_labels(self, labels, clear_existing=True):
        """
        Load labels to the unique label list widget.

        Args:
            labels (list): List of label names to load
            clear_existing (bool): Whether to clear existing labels before loading new ones
        """
        if not labels:
            return

        if clear_existing:
            self.unique_label_list.clear()

        for label in labels:
            # Check if label already exists to avoid duplicates
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label, skip_label_info=False):
        if label == "AUTOLABEL_ADD":
            return (144, 238, 144)
        if label == "AUTOLABEL_REMOVE":
            return (255, 182, 193)
        if label in self.label_info and not skip_label_info:
            return tuple(self.label_info[label]["color"])
        if self._config["shape_color"] == "auto":
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
            item = self.unique_label_list.find_items_by_label(label)[0]
            label_id = self.unique_label_list.indexFromItem(item).row() + 1
            label_id += self._config["shift_auto_shape_color"]
            return LABEL_COLORMAP[label_id % len(LABEL_COLORMAP)]
        if (
            self._config["shape_color"] == "manual"
            and self._config["label_colors"]
            and label in self._config["label_colors"]
        ):
            return self._config["label_colors"][label]
        if self._config["default_shape_color"]:
            return self._config["default_shape_color"]
        return (0, 255, 0)

    def remove_labels(self, shapes):
        for shape in shapes:
            item = self.label_list.find_item_by_shape(shape)
            self.label_list.remove_item(item)
        self.update_combo_box()
        self.update_gid_box()

    def load_shapes(self, shapes, replace=True, update_last_label=True):
        self._no_selection_slot = True
        for shape in shapes:
            self.add_label(shape, update_last_label=update_last_label)
        self.label_list.clearSelection()
        self._no_selection_slot = False
        self.canvas.load_shapes(shapes, replace=replace)
        self.apply_label_visibility()

    def load_flags(self, flags):
        self.flag_widget.clear()
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)

    def apply_label_visibility(self):
        for item in self.label_list:
            label = item.shape().label
            if label in self.label_info:
                is_visible = self.label_info[label].get("visible", True)
            else:
                is_visible = True
            if is_visible:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

    def update_combo_box(self):
        # Get the unique labels and add them to the Combobox.
        labels_list = []
        for item in self.label_list:
            label = item.shape().label
            labels_list.append(str(label))
        unique_labels_list = list(set(labels_list))

        # Add a null row for showing all the labels
        unique_labels_list.append("")
        unique_labels_list.sort()
        self.label_filter_combobox.update_items(unique_labels_list)

    def update_gid_box(self):
        # Get the unique group ids and add them to the Combobox.
        gid_list = []
        for item in self.label_list:
            gid = item.shape().group_id
            if gid is not None:
                gid_list.append(str(gid))
        unique_gid_list = list(set(gid_list))

        # Add a null row for showing all the labels
        unique_gid_list.append("-1")
        unique_gid_list.sort()
        self.gid_filter_combobox.update_items(unique_gid_list)

    def save_labels(self, filename):
        label_file = LabelFile()
        # Get current shapes
        # Excluding auto labeling special shapes
        shapes = [
            item.shape().to_dict()
            for item in self.label_list
            if item.shape().label
            not in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]
        ]
        flags = {}
        for i in range(self.flag_widget.count()):
            item = self.flag_widget.item(i)
            key = item.text()
            flag = item.checkState() == Qt.Checked
            flags[key] = flag
        try:
            image_path = osp.relpath(self.image_path, osp.dirname(filename))
            image_data = (
                self.image_data if self._config["store_data"] else None
            )
            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))

            label_file.save(
                filename=filename,
                shapes=shapes,
                image_path=image_path,
                image_data=image_data,
                image_height=self.image.height(),
                image_width=self.image.width(),
                other_data=self.other_data,
                flags=flags,
            )
            self.label_file = label_file
            items = self.file_list_widget.findItems(
                self.image_path, Qt.MatchExactly
            )
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
            # disable allows next and previous image to proceed
            # self.filename = filename
            return True
        except LabelFileError as e:
            self.error_message(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def duplicate_selected_shape(self):
        added_shapes = self.canvas.duplicate_selected_shapes()
        self.label_list.clearSelection()
        for shape in added_shapes:
            self.add_label(shape)
        self.set_dirty()

    def paste_selected_shape(self):
        if self._config["system_clipboard"]:
            clipboard = QtWidgets.QApplication.clipboard()
            json_str = clipboard.text()
            shapes = []
            try:
                shapeDicts = json.loads(json_str)
                for shapeDict in shapeDicts:
                    shapes.append(Shape().load_from_dict(shapeDict))
            except json.JSONDecodeError as e:
                self.error_message(
                    self.tr("Error pasting shapes"),
                    self.tr("Error decoding shapes: %s") % str(e),
                )
                return
            self.load_shapes(shapes, replace=False)
        else:
            self.load_shapes(self._copied_shapes, replace=False)
        self.set_dirty()

    def toggle_system_clipboard(self, system_clipboard):
        self._config["system_clipboard"] = system_clipboard
        self.actions.paste.setEnabled(
            bool(system_clipboard or self._copied_shapes)
        )

    def copy_selected_shape(self):
        if self._config["system_clipboard"]:
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setText(
                json.dumps([s.to_dict() for s in self.canvas.selected_shapes])
            )
        else:
            self._copied_shapes = [
                s.copy() for s in self.canvas.selected_shapes
            ]
            self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def text_selection_changed(self, index):
        label = self.label_filter_combobox.text_box.itemText(index)
        for item in self.label_list:
            item_label = item.shape().label
            if label in ["", item_label]:
                if item_label in self.label_info:
                    is_visible = self.label_info[item_label].get(
                        "visible", True
                    )
                    item.setCheckState(
                        Qt.Checked if is_visible else Qt.Unchecked
                    )
                else:
                    item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

    def gid_selection_changed(self, index):
        gid = self.gid_filter_combobox.gid_box.itemText(index)
        for item in self.label_list:
            if item.shape().group_id is not None:
                checked_gid = ["-1", str(item.shape().group_id)]
            else:
                checked_gid = ["-1"]
            if str(gid) in checked_gid:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

    def label_selection_changed(self):
        if self._no_selection_slot:
            return
        if self.canvas.editing():
            selected_shapes = []
            for item in self.label_list.selected_items():
                selected_shapes.append(item.shape())
            if selected_shapes:
                self.canvas.select_shapes(selected_shapes)
            else:
                self.canvas.deselect_shape()

    def label_item_changed(self, item):
        shape = item.shape()
        shape.visible = item.checkState() == Qt.Checked
        self.canvas.set_shape_visible(shape, item.checkState() == Qt.Checked)
        if (
            hasattr(self, "navigator_dialog")
            and self.navigator_dialog.isVisible()
        ):
            self.update_navigator_shapes()

    def label_order_changed(self):
        self.set_dirty()
        self.canvas.load_shapes([item.shape() for item in self.label_list])

    # Callback functions:
    def new_shape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        items = self.unique_label_list.selectedItems()
        text = None
        if items:
            text = items[0].data(Qt.UserRole)
        flags = {}
        group_id = None
        description = ""
        difficult = False
        kie_linking = []

        if self.canvas.shapes[-1].label in [
            AutoLabelingMode.ADD,
            AutoLabelingMode.REMOVE,
        ]:
            text = self.canvas.shapes[-1].label
        elif (
            self._config["display_label_popup"]
            or not text
            or self.canvas.shapes[-1].label == AutoLabelingMode.OBJECT
        ):
            last_label = self.find_last_label()
            last_gid = (
                self.find_last_gid()
                if self._config["auto_use_last_gid"]
                else None
            )
            if self.digit_to_label is not None:
                text = self.digit_to_label
                self.digit_to_label = None
                if last_gid is not None:
                    group_id = last_gid
            elif self._config["auto_use_last_label"] and last_label:
                text = last_label
                if last_gid is not None:
                    group_id = last_gid
            else:
                previous_text = self.label_dialog.edit.text()
                (
                    text,
                    flags,
                    group_id,
                    description,
                    difficult,
                    kie_linking,
                ) = self.label_dialog.pop_up(
                    text,
                    group_id=last_gid,
                    move_mode=self._config.get("move_mode", "auto"),
                )
                if not text:
                    self.label_dialog.edit.setText(previous_text)

        if text and not self.validate_label(text):
            self.error_message(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            text = ""
            return

        if self.attributes and text:
            text = self.reset_attribute(text)

        if text:
            self.label_list.clearSelection()
            shape = self.canvas.set_last_label(text, flags, group_id)
            shape.group_id = group_id
            shape.description = description
            if text not in [AutoLabelingMode.ADD, AutoLabelingMode.REMOVE]:
                shape.label = text
            shape.difficult = difficult
            shape.kie_linking = kie_linking
            self.add_label(shape)
            self.actions.edit_mode.setEnabled(True)
            self.actions.undo_last_point.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.set_dirty()

            if self.attributes and text in self.attributes:
                shape.selected = True
                self.shape_attributes.show()
                self.scroll_area.show()
                for i, canvas_shape in enumerate(self.canvas.shapes):
                    if canvas_shape is shape:
                        self.update_attributes(i)
                        break
        else:
            self.canvas.undo_last_line()
            self.canvas.shapes_backups.pop()

    def show_shape(self, shape_height, shape_width, pos):
        """Display annotation width and height while hovering inside.

        Parameters:
        - shape_height (float): The height of the shape.
        - shape_width (float): The width of the shape.
        - pos (QPointF): The current mouse coordinates inside the shape.
        """
        num_images = len(self.image_list)
        if shape_height > 0 and shape_width > 0:
            if num_images and self.filename in self.image_list:
                self.status(
                    str(self.tr("X: %d, Y: %d | H: %d, W: %d"))
                    % (
                        int(pos.x()),
                        int(pos.y()),
                        shape_height,
                        shape_width,
                    )
                )
            else:
                self.status(
                    str(self.tr("X: %d, Y: %d | H: %d, W: %d"))
                    % (int(pos.x()), int(pos.y()), shape_height, shape_width)
                )
        elif self.image_path:
            if num_images and self.filename in self.image_list:
                self.status(
                    str(self.tr("X: %d, Y: %d"))
                    % (
                        int(pos.x()),
                        int(pos.y()),
                    )
                )
            else:
                self.status(
                    str(self.tr("X: %d, Y: %d")) % (int(pos.x()), int(pos.y()))
                )

    def on_navigator_request(self, x_ratio, y_ratio):
        """Handle navigation request from navigator widget."""
        if not hasattr(self, "image") or self.image.isNull():
            return

        scroll_area = self._central_widget
        canvas_size = self.canvas.size()
        scroll_area_size = scroll_area.viewport().size()

        target_x = x_ratio * canvas_size.width() - scroll_area_size.width() / 2
        target_y = (
            y_ratio * canvas_size.height() - scroll_area_size.height() / 2
        )

        self.set_scroll(Qt.Horizontal, target_x)
        self.set_scroll(Qt.Vertical, target_y)

    def update_navigator_viewport(self):
        """Update the viewport rectangle in the navigator."""
        if not hasattr(self, "navigator_dialog") or not hasattr(self, "image"):
            return

        if not self.navigator_dialog.isVisible():
            return

        if self.image.isNull():
            return

        scroll_area = self._central_widget
        canvas_size = self.canvas.size()
        scroll_area_size = scroll_area.viewport().size()
        if canvas_size.width() <= 0 or canvas_size.height() <= 0:
            return

        h_scroll = self.scroll_bars[Qt.Horizontal].value()
        v_scroll = self.scroll_bars[Qt.Vertical].value()
        x_ratio = max(0.0, h_scroll / canvas_size.width())
        y_ratio = max(0.0, v_scroll / canvas_size.height())
        width_ratio = min(1.0, scroll_area_size.width() / canvas_size.width())
        height_ratio = min(
            1.0, scroll_area_size.height() / canvas_size.height()
        )

        self.navigator_dialog.set_viewport(
            x_ratio, y_ratio, width_ratio, height_ratio
        )
        self.update_navigator_shapes()

    def update_navigator_shapes(self):
        """Update shapes overlay in navigator."""
        if (
            not hasattr(self, "navigator_dialog")
            or not self.navigator_dialog.isVisible()
        ):
            return

        shapes = getattr(self.canvas, "shapes", [])
        canvas_visible = getattr(self.canvas, "visible", {})
        h_shape = getattr(self.canvas, "h_hape", None)
        for shape in shapes:
            shape._is_highlighted = shape == h_shape
        self.navigator_dialog.set_shapes(shapes, canvas_visible)

    def on_navigator_zoom_changed(
        self, zoom_percentage: int, mouse_pos: Optional[QtCore.QPoint] = None
    ) -> None:
        """Handle zoom change from navigator controls."""

        if not hasattr(self, "image") or self.image.isNull():
            return

        if mouse_pos is not None:
            canvas_pos = self._convert_navigator_pos_to_canvas(mouse_pos)
            if canvas_pos:
                canvas_width_old = self.canvas.width()

                self.zoom_widget.setValue(zoom_percentage)
                self.zoom_mode = self.MANUAL_ZOOM
                self.zoom_values[self.filename] = (
                    self.zoom_mode,
                    zoom_percentage,
                )
                self.paint_canvas()

                canvas_width_new = self.canvas.width()
                if canvas_width_old != canvas_width_new:
                    canvas_scale_factor = canvas_width_new / canvas_width_old
                    x_shift = round(
                        canvas_pos.x() * canvas_scale_factor - canvas_pos.x()
                    )
                    y_shift = round(
                        canvas_pos.y() * canvas_scale_factor - canvas_pos.y()
                    )
                    self.set_scroll(
                        QtCore.Qt.Horizontal,
                        self.scroll_bars[QtCore.Qt.Horizontal].value()
                        + x_shift,
                    )
                    self.set_scroll(
                        QtCore.Qt.Vertical,
                        self.scroll_bars[QtCore.Qt.Vertical].value() + y_shift,
                    )

                return

        # Handle direct zoom changes
        if (
            hasattr(self, "canvas")
            and hasattr(self.canvas, "width")
            and hasattr(self.canvas, "height")
        ):
            if hasattr(self.navigator_dialog, "navigator"):
                nav_widget = self.navigator_dialog.navigator
                if (
                    hasattr(nav_widget, "viewport_rect")
                    and not nav_widget.viewport_rect.isEmpty()
                ):
                    nav_rect_center_x = nav_widget.viewport_rect.center().x()
                    nav_rect_center_y = nav_widget.viewport_rect.center().y()
                    canvas_pos = self._convert_navigator_pos_to_canvas(
                        QtCore.QPoint(
                            int(nav_rect_center_x), int(nav_rect_center_y)
                        )
                    )

                    if canvas_pos:
                        canvas_width_old = self.canvas.width()

                        self.zoom_widget.setValue(zoom_percentage)
                        self.zoom_mode = self.MANUAL_ZOOM
                        self.zoom_values[self.filename] = (
                            self.zoom_mode,
                            zoom_percentage,
                        )
                        self.paint_canvas()

                        canvas_width_new = self.canvas.width()
                        if canvas_width_old != canvas_width_new:
                            canvas_scale_factor = (
                                canvas_width_new / canvas_width_old
                            )
                            x_shift = round(
                                canvas_pos.x() * canvas_scale_factor
                                - canvas_pos.x()
                            )
                            y_shift = round(
                                canvas_pos.y() * canvas_scale_factor
                                - canvas_pos.y()
                            )
                            self.set_scroll(
                                QtCore.Qt.Horizontal,
                                self.scroll_bars[QtCore.Qt.Horizontal].value()
                                + x_shift,
                            )
                            self.set_scroll(
                                QtCore.Qt.Vertical,
                                self.scroll_bars[QtCore.Qt.Vertical].value()
                                + y_shift,
                            )
                        return

            self.zoom_widget.setValue(zoom_percentage)
            self.zoom_mode = self.MANUAL_ZOOM
            self.zoom_values[self.filename] = (self.zoom_mode, zoom_percentage)
            self.paint_canvas()
        else:
            self.zoom_widget.setValue(zoom_percentage)
            self.zoom_mode = self.MANUAL_ZOOM
            self.zoom_values[self.filename] = (self.zoom_mode, zoom_percentage)
            self.paint_canvas()

    def _convert_navigator_pos_to_canvas(
        self, navigator_pos: QtCore.QPoint
    ) -> Optional[QtCore.QPoint]:
        """Convert navigator mouse position to canvas coordinates."""
        if (
            not hasattr(self, "navigator_dialog")
            or not self.navigator_dialog.isVisible()
        ):
            return None

        navigator_widget = self.navigator_dialog.navigator
        if (
            not navigator_widget.image_rect
            or navigator_widget.image_rect.isEmpty()
        ):
            return None

        relative_x = navigator_pos.x() - navigator_widget.image_rect.x()
        relative_y = navigator_pos.y() - navigator_widget.image_rect.y()
        if (
            relative_x < 0
            or relative_x > navigator_widget.image_rect.width()
            or relative_y < 0
            or relative_y > navigator_widget.image_rect.height()
        ):
            return None

        # Convert to ratio (0.0 to 1.0)
        x_ratio = relative_x / navigator_widget.image_rect.width()
        y_ratio = relative_y / navigator_widget.image_rect.height()

        # Convert to canvas coordinates
        canvas_x = int(x_ratio * self.canvas.width())
        canvas_y = int(y_ratio * self.canvas.height())

        return QtCore.QPoint(canvas_x, canvas_y)

    def on_navigator_viewport_update_requested(self):
        """Handle viewport update request from navigator resize"""
        QtCore.QTimer.singleShot(50, self.update_navigator_viewport)

    def toggle_navigator(self):
        """Toggle the navigator window visibility"""
        if self.navigator_dialog.isVisible():
            self.navigator_dialog.hide()
            if hasattr(self, "actions") and hasattr(
                self.actions, "show_navigator"
            ):
                self.actions.show_navigator.setChecked(False)
        else:
            self.navigator_dialog.show()
            if hasattr(self, "image") and not self.image.isNull():
                self.navigator_dialog.set_image(
                    QtGui.QPixmap.fromImage(self.image)
                )
                self.update_navigator_viewport()
            if hasattr(self, "actions") and hasattr(
                self.actions, "show_navigator"
            ):
                self.actions.show_navigator.setChecked(True)

    def scroll_request(self, delta, orientation, mode):
        scroll_bar = self.scroll_bars[orientation]
        units = -delta * (0.1 if mode == 0 else 1)
        step = scroll_bar.singleStep() if mode == 0 else scroll_bar.maximum()
        value = scroll_bar.value() + step * units
        self.set_scroll(orientation, value)

    def set_scroll(self, orientation, value):
        self.scroll_bars[orientation].setValue(round(value))
        self.scroll_values[orientation][self.filename] = value
        self.update_navigator_viewport()

    def set_zoom(self, value):
        self.actions.fit_width.setChecked(False)
        self.actions.fit_window.setChecked(False)
        self.zoom_mode = self.MANUAL_ZOOM
        self.zoom_widget.setValue(value)
        self.zoom_values[self.filename] = (self.zoom_mode, value)
        if hasattr(self, "navigator_dialog"):
            self.navigator_dialog.set_zoom_value(value)

    def add_zoom(self, increment=1.1):
        zoom_value = self.zoom_widget.value() * increment
        if increment > 1:
            zoom_value = math.ceil(zoom_value)
        else:
            zoom_value = math.floor(zoom_value)
        self.set_zoom(zoom_value)

    def zoom_request(self, delta, pos):
        canvas_width_old = self.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.add_zoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor - pos.x())
            y_shift = round(pos.y() * canvas_scale_factor - pos.y())

            self.set_scroll(
                Qt.Horizontal,
                self.scroll_bars[Qt.Horizontal].value() + x_shift,
            )
            self.set_scroll(
                Qt.Vertical,
                self.scroll_bars[Qt.Vertical].value() + y_shift,
            )

    def set_fit_window(self, value=True):
        if value:
            self.actions.fit_width.setChecked(False)
        self.zoom_mode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjust_scale()

    def set_fit_width(self, value=True):
        if value:
            self.actions.fit_window.setChecked(False)
        self.zoom_mode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjust_scale()

    def set_cross_line(self):
        crosshair_dialog = CrosshairSettingsDialog(**self.crosshair_settings)
        if crosshair_dialog.exec_() == QtWidgets.QDialog.Accepted:
            crosshair_settings = crosshair_dialog.get_settings()
            show = crosshair_settings["show"]
            width = crosshair_settings["width"]
            color = crosshair_settings["color"]
            opacity = crosshair_settings["opacity"]
            self.canvas.set_cross_line(show, width, color, opacity)
            self._config["canvas"]["crosshair"] = crosshair_settings

    def set_canvas_params(self, key, value):
        self._config[key] = value
        assert hasattr(self.canvas, key), f"Canvas has no attribute {key}"
        setattr(self.canvas, key, value)
        self.canvas.update()

    def on_new_brightness_contrast(self, qimage):
        self.canvas.load_pixmap(
            QtGui.QPixmap.fromImage(qimage), clear_shapes=False
        )

    def brightness_contrast(self, _):
        self.brightness_contrast_dialog.update_image(
            utils.img_data_to_pil(self.image_data)
        )

        brightness, contrast = self.brightness_contrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            self.brightness_contrast_dialog.slider_brightness.setValue(
                brightness
            )
        if contrast is not None:
            self.brightness_contrast_dialog.slider_contrast.setValue(contrast)

        self.brightness_contrast_dialog.exec_()

        brightness = self.brightness_contrast_dialog.slider_brightness.value()
        contrast = self.brightness_contrast_dialog.slider_contrast.value()
        self.brightness_contrast_values[self.filename] = (brightness, contrast)

    def hide_selected_polygons(self):
        shapes_to_hide = []
        for item in self.label_list:
            if item.shape().selected:
                item.setCheckState(Qt.Unchecked)
                item.shape().visible = False
                shapes_to_hide.append(item.shape())

        self.selected_polygon_stack.extend(shapes_to_hide)
        self.canvas.update()
        if (
            hasattr(self, "navigator_dialog")
            and self.navigator_dialog.isVisible()
        ):
            self.update_navigator_shapes()

    def show_hidden_polygons(self):
        if self.selected_polygon_stack:
            shape_to_show = self.selected_polygon_stack.pop()
            item = self.label_list.find_item_by_shape(shape_to_show)
            if item:
                item.setCheckState(Qt.Checked)
                shape_to_show.visible = True
                self.canvas.update()
                if (
                    hasattr(self, "navigator_dialog")
                    and self.navigator_dialog.isVisible()
                ):
                    self.update_navigator_shapes()
            else:
                logger.warning(
                    f"Shape associated with the hidden item was not found in label list, could not show."
                )

    def get_next_files(self, filename, num_files):
        """Get the next files in the list."""
        if not self.image_list:
            return []
        filenames = []
        current_index = 0
        if filename is not None:
            try:
                current_index = self.fn_to_index[str(filename)]
            except ValueError:
                return []
            filenames.append(filename)
        for _ in range(num_files):
            if current_index + 1 < len(self.image_list):
                filenames.append(self.image_list[current_index + 1])
                current_index += 1
            else:
                filenames.append(self.image_list[-1])
                break
        return filenames

    def inform_next_files(self, filename):
        """Inform the next files to be annotated.
        This list can be used by the user to preload the next files
        or running a background process to process them
        """
        next_files = self.get_next_files(filename, 5)
        if next_files:
            self.next_files_changed.emit(next_files)

    def load_file(self, filename=None):  # noqa: C901
        """Load the specified file, or the last opened file if None."""

        # NOTE(jack): Does we need to save the config here?
        # save_config(self._config)

        # For auto labeling, clear the previous marks
        # and inform the next files to be annotated
        # NOTE(jack): this is not needed for now
        # self.clear_auto_labeling_marks()
        # self.inform_next_files(filename)

        # Changing file_list_widget loads file
        if filename in self.image_list and (
            self.file_list_widget.currentRow()
            != self.fn_to_index[str(filename)]
        ):
            self.file_list_widget.setCurrentRow(
                self.fn_to_index[str(filename)]
            )
            self.file_list_widget.repaint()
            return False

        self.reset_state()
        self.canvas.setEnabled(False)
        if filename is None:
            filename = self.settings.value("filename", "")
        filename = str(filename)
        if not QtCore.QFile.exists(filename):
            self.error_message(
                self.tr("Error opening file"),
                self.tr("No such file: <b>%s</b>") % filename,
            )
            return False

        # assumes same name, but json extension
        label_file = osp.splitext(filename)[0] + ".json"
        image_dir = None
        if self.output_dir:
            image_dir = osp.dirname(filename)
            label_file_without_path = osp.basename(label_file)
            label_file = self.output_dir + "/" + label_file_without_path
        if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
            label_file
        ):
            try:
                self.label_file = LabelFile(label_file, image_dir)
            except LabelFileError as e:
                self.error_message(
                    self.tr("Error opening file"),
                    self.tr(
                        "<p><b>%s</b></p>"
                        "<p>Make sure <i>%s</i> is a valid label file."
                    )
                    % (e, label_file),
                )
                self.status(self.tr("Error reading %s") % label_file)
                return False
            self.image_data = self.label_file.image_data
            self.image_path = osp.join(
                osp.dirname(label_file),
                self.label_file.image_path,
            )
            self.other_data = self.label_file.other_data
            self.shape_text_edit.textChanged.disconnect()
            self.shape_text_edit.setPlainText(
                self.other_data.get("description", "")
            )
            self.shape_text_edit.textChanged.connect(self.shape_text_changed)
        else:
            self.image_data = LabelFile.load_image_file(filename)
            if self.image_data:
                self.image_path = filename
            self.label_file = None

        # Reset the label loop count
        self.label_loop_count = -1
        self.select_loop_count = -1

        # TODO(jack): icc profile issue warning
        # - qt.gui.icc: fromIccProfile: failed minimal tag size sanity
        # - qt.gui.icc: fromIccProfile: invalid tag offset alignment
        image = QtGui.QImage.fromData(self.image_data)

        if image.isNull():
            formats = [
                f"*.{fmt.data().decode()}"
                for fmt in QtGui.QImageReader.supportedImageFormats()
            ]
            self.error_message(
                self.tr("Error opening file"),
                self.tr(
                    "<p>Make sure <i>{0}</i> is a valid image file.<br/>"
                    "Supported image formats: {1}</p>"
                ).format(filename, ",".join(formats)),
            )
            self.status(self.tr("Error reading %s") % filename)
            return False
        self.image = image
        self.filename = filename

        if (
            hasattr(self, "navigator_dialog")
            and self.navigator_dialog.isVisible()
        ):
            self.navigator_dialog.set_image(QtGui.QPixmap.fromImage(image))
            self.update_navigator_shapes()
        if (
            hasattr(self, "_should_restore_navigator")
            and self._should_restore_navigator
        ):
            self._should_restore_navigator = False
            if self.navigator_dialog.isVisible():
                self.update_navigator_viewport()
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        self.canvas.load_pixmap(QtGui.QPixmap.fromImage(image))

        # load label flags
        flags = {k: False for k in self.image_flags or []}
        if self.label_file:
            for shape in self.label_file.shapes:
                default_flags = {}
                if self._config["label_flags"]:
                    for pattern, keys in self._config["label_flags"].items():
                        if re.match(pattern, shape.label):
                            for key in keys:
                                default_flags[key] = False
                    shape.flags = {
                        **default_flags,
                        **shape.flags,
                    }
            self.update_combo_box()
            self.update_gid_box()
            self.load_shapes(self.label_file.shapes, update_last_label=False)
            if self.label_file.flags is not None:
                flags.update(self.label_file.flags)
        self.load_flags(flags)

        # load shapes
        if self._config["keep_prev"] and self.no_shape():
            self.load_shapes(
                prev_shapes, replace=False, update_last_label=False
            )
            self.set_dirty()
        else:
            self.set_clean()
        self.canvas.setEnabled(True)
        # set zoom values
        is_initial_load = not self.zoom_values
        if self.filename in self.zoom_values:
            self.zoom_mode = self.zoom_values[self.filename][0]
            self.set_zoom(self.zoom_values[self.filename][1])
        elif is_initial_load or not self._config["keep_prev_scale"]:
            self.adjust_scale(initial=True)
        # set scroll values
        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.set_scroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        # set brightness contrast values
        self.brightness_contrast_dialog.update_image(
            utils.img_data_to_pil(self.image_data)
        )

        brightness, contrast = self.brightness_contrast_values.get(
            self.filename, (None, None)
        )
        if self._config["keep_prev_brightness"] and self.recent_files:
            brightness, _ = self.brightness_contrast_values.get(
                self.recent_files[0], (None, None)
            )
        if self._config["keep_prev_contrast"] and self.recent_files:
            _, contrast = self.brightness_contrast_values.get(
                self.recent_files[0], (None, None)
            )
        if brightness is not None:
            self.brightness_contrast_dialog.slider_brightness.setValue(
                brightness
            )
        if contrast is not None:
            self.brightness_contrast_dialog.slider_contrast.setValue(contrast)
        self.brightness_contrast_values[self.filename] = (brightness, contrast)
        if brightness is not None or contrast is not None:
            self.brightness_contrast_dialog.on_new_value()

        self.paint_canvas()
        self.add_recent_file(self.filename)
        self.toggle_actions(True)
        self.canvas.setFocus()
        self.update_thumbnail_display()
        return True

    # QT Overload
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            event.accept()
            return
        super(LabelingWidget, self).keyPressEvent(event)

    def resizeEvent(self, _):
        if (
            self.canvas
            and not self.image.isNull()
            and self.zoom_mode != self.MANUAL_ZOOM
        ):
            self.adjust_scale()
        self.update_thumbnail_pixmap()

    def paint_canvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoom_widget.value()
        self.canvas.adjustSize()
        self.canvas.update()
        self.update_navigator_viewport()

    def adjust_scale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoom_mode]()
        value = int(100 * value)
        self.zoom_widget.setValue(value)
        self.zoom_values[self.filename] = (self.zoom_mode, value)
        if hasattr(self, "navigator_dialog"):
            self.navigator_dialog.set_zoom_value(value)

    def scale_fit_window(self):
        """Figure out the size of the pixmap to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.central_widget().width() - e
        h1 = self.central_widget().height() - e
        wh_ratio1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        wh_ratio2 = w2 / h2
        return w1 / w2 if wh_ratio2 >= wh_ratio1 else h1 / h2

    def scale_fit_width(self):
        # The epsilon does not seem to work too well here.
        w = self.central_widget().width() - 2.0
        return w / self.canvas.pixmap.width()

    # QT Overload
    def closeEvent(self, event):
        if not self.may_continue():
            event.ignore()
        self.settings.setValue(
            "filename", self.filename if self.filename else ""
        )
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/position", self.pos())
        self.settings.setValue("window/state", self.parent.parent.saveState())
        self.settings.setValue("recent_files", self.recent_files)

        if hasattr(self, "navigator_dialog"):
            navigator_visible = self.navigator_dialog.isVisible()
            self.settings.setValue("navigator/visible", navigator_visible)
            if navigator_visible:
                self.settings.setValue(
                    "navigator/geometry", self.navigator_dialog.saveGeometry()
                )
                self.settings.setValue(
                    "navigator/size", self.navigator_dialog.size()
                )
                self.settings.setValue(
                    "navigator/position", self.navigator_dialog.pos()
                )

        save_config(self._config)

        if hasattr(self, "async_exif_scanner") and self.async_exif_scanner:
            try:
                self.async_exif_scanner.stop_scan()
            except (RuntimeError, AttributeError):
                pass

        # ask the use for where to save the labels
        # self.settings.setValue('window/geometry', self.saveGeometry())

    # QT Overload
    def dragEnterEvent(self, event):
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        if event.mimeData().hasUrls():
            items = [i.toLocalFile() for i in event.mimeData().urls()]
            if any(i.lower().endswith(tuple(extensions)) for i in items):
                event.accept()
        else:
            event.ignore()

    # QT Overload
    def dropEvent(self, event):
        if not self.may_continue():
            event.ignore()
            return
        items = [i.toLocalFile() for i in event.mimeData().urls()]
        self.import_dropped_image_files(items)

    def load_recent(self, filename):
        if self.may_continue():
            self.load_file(filename)

    def open_checked_image(self, end_index, step, load=True):
        if not self.may_continue():
            return
        current_index = self.fn_to_index[str(self.filename)]
        for i in range(current_index + step, end_index, step):
            if self.file_list_widget.item(i).checkState() == Qt.Checked:
                self.filename = self.image_list[i]
                if self.filename and load:
                    self.load_file(self.filename)
                break

    def open_prev_unchecked_image(self):
        if self._config["switch_to_checked"]:
            self.open_checked_image(-1, -1)
            return

        if (
            not self.may_continue()
            or len(self.image_list) <= 0
            or self.filename is None
        ):
            return

        current_index = self.fn_to_index[str(self.filename)]
        for i in range(current_index - 1, -1, -1):
            if self.file_list_widget.item(i).checkState() == Qt.Unchecked:
                filename = self.image_list[i]
                if filename:
                    self.load_file(filename)
                break

    def open_next_unchecked_image(self, _value=False):
        if self._config["switch_to_checked"]:
            self.open_checked_image(self.file_list_widget.count(), 1)
            return

        if (
            not self.may_continue()
            or len(self.image_list) <= 0
            or self.filename is None
        ):
            return

        current_index = self.fn_to_index[str(self.filename)]
        for i in range(current_index + 1, len(self.image_list)):
            if self.file_list_widget.item(i).checkState() == Qt.Unchecked:
                filename = self.image_list[i]
                if filename:
                    self.load_file(filename)
                break

    def open_prev_image(self, _value=False):
        if not self.may_continue():
            return

        if len(self.image_list) <= 0:
            return

        if self.filename is None:
            return

        current_index = self.fn_to_index[str(self.filename)]
        if current_index - 1 >= 0:
            filename = self.image_list[current_index - 1]
            if filename:
                self.load_file(filename)

    def open_next_image(self, _value=False, load=True):
        if not self.may_continue():
            return

        if len(self.image_list) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.image_list[0]
        else:
            current_index = self.fn_to_index[str(self.filename)]
            if current_index + 1 < len(self.image_list):
                filename = self.image_list[current_index + 1]
            else:
                filename = self.image_list[-1]
        self.filename = filename

        if self.filename and load:
            self.load_file(self.filename)

    # File
    def open_file(self, _value=False):
        if not self.may_continue():
            return
        path = osp.dirname(str(self.filename)) if self.filename else "."
        formats = [
            f"*.{fmt.data().decode()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = self.tr("Image & Label files (%s)") % " ".join(
            formats + [f"*{LabelFile.suffix}"]
        )
        file_dialog = FileDialogPreview(self)
        file_dialog.setFileMode(FileDialogPreview.ExistingFile)
        file_dialog.setNameFilter(filters)
        file_dialog.setWindowTitle(
            self.tr("%s - Choose Image or Label file") % __appname__,
        )
        file_dialog.setWindowFilePath(path)
        file_dialog.setViewMode(FileDialogPreview.Detail)
        if file_dialog.exec_():
            filename = file_dialog.selectedFiles()[0]
            if filename:
                self.file_list_widget.clear()
                self.fn_to_index.clear()
                self.load_file(filename)

    def change_output_dir_dialog(self, _value=False):
        default_output_dir = self.output_dir
        if default_output_dir is None and self.filename:
            default_output_dir = osp.dirname(self.filename)
        if default_output_dir is None:
            default_output_dir = self.current_path()

        output_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Save/Load Annotations in Directory") % __appname__,
            default_output_dir,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        output_dir = str(output_dir)

        if not output_dir:
            return

        self.output_dir = output_dir

        self.statusBar().showMessage(
            self.tr("%s . Annotations will be saved/loaded in %s")
            % ("Change Annotations Dir", self.output_dir)
        )
        self.statusBar().show()

        current_filename = self.filename
        self.import_image_folder(self.last_open_dir, load=False)

        if current_filename in self.image_list:
            # retain currently selected file
            self.file_list_widget.setCurrentRow(
                self.fn_to_index[str(current_filename)]
            )
            self.file_list_widget.repaint()

    def save_file(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        if self.label_file:
            # DL20180323 - overwrite when in directory
            self._save_file(self.label_file.filename)
        elif self.output_file:
            self._save_file(self.output_file)
            self.close()
        else:
            self._save_file(self.save_file_dialog())

    def save_file_as(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._save_file(self.save_file_dialog())

    def save_file_dialog(self):
        caption = self.tr("%s - Choose File") % __appname__
        filters = self.tr("Label files (*%s)") % LabelFile.suffix
        if self.output_dir:
            file_dialog = QtWidgets.QFileDialog(
                self, caption, self.output_dir, filters
            )
        else:
            file_dialog = QtWidgets.QFileDialog(
                self, caption, self.current_path(), filters
            )
        file_dialog.setDefaultSuffix(LabelFile.suffix[1:])
        file_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        file_dialog.setOption(
            QtWidgets.QFileDialog.DontConfirmOverwrite, False
        )
        file_dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        basename = osp.basename(osp.splitext(self.filename)[0])
        if self.output_dir:
            default_labelfile_name = osp.join(
                self.output_dir, basename + LabelFile.suffix
            )
        else:
            default_labelfile_name = osp.join(
                self.current_path(), basename + LabelFile.suffix
            )
        filename = file_dialog.getSaveFileName(
            self,
            self.tr("Choose File"),
            default_labelfile_name,
            self.tr("Label files (*%s)") % LabelFile.suffix,
        )
        if isinstance(filename, tuple):
            filename, _ = filename
        return filename

    def _save_file(self, filename):
        if filename and self.save_labels(filename):
            self.add_recent_file(filename)
            self.set_clean()

    def close_file(self, _value=False):
        if not self.may_continue():
            return
        self.reset_state()
        self.set_clean()
        self.toggle_actions(False)
        self.canvas.setEnabled(False)
        self.actions.save_as.setEnabled(False)

    def get_label_file(self):
        if self.label_file:
            return self.label_file.filename
        base = self.image_path if self.image_path else self.filename
        if base.lower().endswith(".json"):
            return base
        lf = osp.splitext(base)[0] + ".json"
        if self.output_dir:
            lf = osp.join(self.output_dir, osp.basename(lf))
        return lf

    def get_image_file(self):
        if not self.filename.lower().endswith(".json"):
            image_file = self.filename
        else:
            image_file = self.image_path

        return image_file

    def delete_file(self):
        mb = QtWidgets.QMessageBox
        if self._config.get("keep_prev", False):
            mb.warning(
                self,
                self.tr("Attention"),
                self.tr(
                    "Please disable 'Keep Previous Annotation' before deleting the label file."
                ),
                mb.Ok,
            )
            return

        msg = self.tr(
            "You are about to permanently delete this label file, "
            "proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        label_file = self.get_label_file()
        if osp.exists(label_file):
            os.remove(label_file)
            logger.info(f"Label file is removed: {label_file}")

            item = self.file_list_widget.currentItem()
            item.setCheckState(Qt.Unchecked)

            filename = self.filename
            self.reset_state()
            self.filename = filename
            if self.filename:
                self.load_file(self.filename)

    def delete_image_file(self):
        if len(self.image_list) < 2:
            return

        mb = QtWidgets.QMessageBox
        if self._config.get("keep_prev", False):
            mb.warning(
                self,
                self.tr("Attention"),
                self.tr(
                    "Please disable 'Keep Previous Annotation' before deleting the image file."
                ),
                mb.Ok,
            )
            return

        msg = self.tr(
            "You are about to permanently delete this image file, "
            "proceed anyway?"
        )
        answer = mb.warning(self, self.tr("Attention"), msg, mb.Yes | mb.No)
        if answer != mb.Yes:
            return

        image_file = self.get_image_file()
        if osp.exists(image_file):
            image_path, image_name = osp.split(image_file)
            save_path = osp.join(image_path, "..", "_delete_")
            os.makedirs(save_path, exist_ok=True)
            save_file = osp.join(save_path, image_name)
            shutil.move(image_file, save_file)
            logger.info(f"Image file is moved to: {osp.realpath(save_file)}")

            label_dir_path = osp.dirname(self.filename)
            if self.output_dir:
                label_dir_path = self.output_dir
            label_name = osp.splitext(image_name)[0] + ".json"
            label_file = osp.join(label_dir_path, label_name)
            if not osp.exists(label_file):
                label_file = osp.join(osp.dirname(image_file), label_name)
            if osp.exists(label_file):
                os.remove(label_file)
                logger.info(f"Label file is removed: {image_file}")

            filename = None
            if self.filename is None:
                filename = self.image_list[0]
            else:
                current_index = self.fn_to_index[str(self.filename)]
                if current_index + 1 < len(self.image_list):
                    filename = self.image_list[current_index + 1]
                else:
                    filename = self.image_list[0]

            self.reset_state()
            if osp.isfile(image_path):
                image_path = osp.dirname(image_path)
            self.import_image_folder(image_path)

            self.filename = filename
            if self.filename:
                self.load_file(self.filename)

    # Message Dialogs. #
    def has_labels(self):
        if self.no_shape():
            self.error_message(
                "No objects labeled",
                "You must label at least one object to save the file.",
            )
            return False
        return True

    def has_label_file(self):
        if self.filename is None:
            return False

        label_file = self.get_label_file()
        return osp.exists(label_file)

    def may_continue(self):
        if not self.dirty:
            return True
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            f'Save annotations to "{self.filename!r}" before closing?'
        )
        answer = mb.question(
            self,
            self.tr("Save annotations?"),
            msg,
            mb.Save | mb.Discard | mb.Cancel,
            mb.Save,
        )
        if answer == mb.Discard:
            return True
        if answer == mb.Save:
            self.save_file()
            return True
        # answer == mb.Cancel
        return False

    def error_message(self, title, message):
        return QtWidgets.QMessageBox.critical(
            self, title, f"<p><b>{title}</b></p>{message}"
        )

    def current_path(self):
        return osp.dirname(str(self.filename)) if self.filename else "."

    def toggle_visibility_shapes(self, value):
        for index, item in enumerate(self.label_list):
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)
            self.label_list[index].shape().visible = True if value else False
        self._config["show_shapes"] = value
        if (
            hasattr(self, "navigator_dialog")
            and self.navigator_dialog.isVisible()
        ):
            self.update_navigator_shapes()

    def remove_selected_point(self):
        self.canvas.remove_selected_point()
        self.canvas.update()
        if self.canvas.h_hape is not None and not self.canvas.h_hape.points:
            self.canvas.delete_shape(self.canvas.h_hape)
            self.remove_labels([self.canvas.h_hape])
            self.set_dirty()
            if self.no_shape():
                for action in self.actions.on_shapes_present:
                    action.setEnabled(False)

    def delete_selected_shape(self):
        self.remove_labels(self.canvas.delete_selected())
        self.set_dirty()
        if self.no_shape():
            for action in self.actions.on_shapes_present:
                action.setEnabled(False)

    def copy_shape(self):
        self.canvas.end_move(copy=True)
        for shape in self.canvas.selected_shapes:
            self.add_label(shape)
        self.label_list.clearSelection()
        self.set_dirty()

    def move_shape(self):
        self.canvas.end_move(copy=False)
        self.set_dirty()

    def open_folder_dialog(self, _value=False, dirpath=None):
        if not self.may_continue():
            return

        default_open_dir_path = dirpath if dirpath else "."
        if self.last_open_dir and osp.exists(self.last_open_dir):
            default_open_dir_path = self.last_open_dir
        else:
            default_open_dir_path = (
                osp.dirname(self.filename) if self.filename else "."
            )

        target_dir_path = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("%s - Open Directory") % __appname__,
                default_open_dir_path,
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        )
        self.import_image_folder(target_dir_path)

    @property
    def image_list(self):
        lst = []
        for i in range(self.file_list_widget.count()):
            item = self.file_list_widget.item(i)
            lst.append(item.text())
        return lst

    def import_dropped_image_files(self, image_files):
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        self.filename = None
        valid_files = []
        for file in image_files:
            if file in self.image_list or not file.lower().endswith(
                tuple(extensions)
            ):
                continue
            valid_files.append(file)
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = self.output_dir + "/" + label_file_without_path
            item = QtWidgets.QListWidgetItem(file)
            flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            if self._config.get("file_list_checkbox_editable", False):
                flags |= Qt.ItemIsUserCheckable
            item.setFlags(flags)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
                label_file
            ):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.file_list_widget.addItem(item)
            self.fn_to_index[file] = self.file_list_widget.count() - 1

        if len(self.image_list) > 1:
            self.actions.open_next_image.setEnabled(True)
            self.actions.open_prev_image.setEnabled(True)
            self.actions.open_next_unchecked_image.setEnabled(True)
            self.actions.open_prev_unchecked_image.setEnabled(True)

        self.toggle_actions(True)
        self.open_next_image()

        if valid_files and self._config.get("exif_scan_enabled", True):
            self.async_exif_scanner.start_scan(valid_files)

    def import_image_folder(self, dirpath, pattern=None, load=True):
        if not self.may_continue() or not dirpath:
            return

        self.last_open_dir = dirpath
        self.filename = None
        self.file_list_widget.clear()
        image_files = []

        search_pattern = parse_search_pattern(pattern) if pattern else None

        for filename in utils.scan_all_images(dirpath):
            if search_pattern:
                if not matches_filename(filename, search_pattern):
                    continue

                if search_pattern.mode == "attribute":
                    label_file = osp.splitext(filename)[0] + ".json"
                    if self.output_dir:
                        label_file_without_path = osp.basename(label_file)
                        label_file = (
                            self.output_dir + "/" + label_file_without_path
                        )

                    if not matches_label_attribute(
                        filename, label_file, search_pattern
                    ):
                        continue

            image_files.append(filename)
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = self.output_dir + "/" + label_file_without_path
            item = QtWidgets.QListWidgetItem(filename)
            flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            if self._config.get("file_list_checkbox_editable", False):
                flags |= Qt.ItemIsUserCheckable
            item.setFlags(flags)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
                label_file
            ):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.file_list_widget.addItem(item)
            self.fn_to_index[filename] = self.file_list_widget.count() - 1

        self.actions.open_next_image.setEnabled(True)
        self.actions.open_prev_image.setEnabled(True)
        self.actions.open_next_unchecked_image.setEnabled(True)
        self.actions.open_prev_unchecked_image.setEnabled(True)
        self.toggle_actions(True)
        self.open_next_image(load=load)

        if image_files and self._config.get("exif_scan_enabled", True):
            self.async_exif_scanner.start_scan(image_files)

    def toggle_auto_labeling_widget(self):
        """Toggle auto labeling widget visibility."""
        if self.auto_labeling_widget.isVisible():
            self.auto_labeling_widget.hide()
            self.actions.run_all_images.setEnabled(False)
        else:
            self.auto_labeling_widget.show()
            self.actions.run_all_images.setEnabled(True)
        self.update_thumbnail_display()

    @pyqtSlot()
    def new_shapes_from_auto_labeling(self, auto_labeling_result):
        """Apply auto labeling results to the current image."""
        if not self.image or not self.image_path:
            return

        # Clear existing shapes
        if auto_labeling_result.replace:
            self.load_shapes([], replace=True)
            self.label_list.clear()
            self.load_shapes(auto_labeling_result.shapes, replace=True)
        else:  # Just update existing shapes
            # Remove shapes with label AutoLabelingMode.OBJECT
            for shape in self.canvas.shapes:
                if shape.label == AutoLabelingMode.OBJECT:
                    item = self.label_list.find_item_by_shape(shape)
                    self.label_list.remove_item(item)
            self.load_shapes(auto_labeling_result.shapes, replace=False)

        # Set image description
        if auto_labeling_result.description:
            description = auto_labeling_result.description
            self.shape_text_label.setText(self.tr("Image Description"))
            self.shape_text_edit.setPlainText(description)
            self.other_data["description"] = description
            self.shape_text_edit.setDisabled(False)

        self.set_dirty()

    def clear_auto_labeling_marks(self):
        """Clear auto labeling marks from the current image."""
        # Clean up label list
        for shape in self.canvas.shapes:
            if shape.label in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]:
                try:
                    item = self.label_list.find_item_by_shape(shape)
                    self.label_list.remove_item(item)
                except ValueError:
                    pass

        # Clean up unique label list
        for shape_label in [
            AutoLabelingMode.OBJECT,
            AutoLabelingMode.ADD,
            AutoLabelingMode.REMOVE,
        ]:
            for item in self.unique_label_list.find_items_by_label(
                shape_label
            ):
                self.unique_label_list.takeItem(
                    self.unique_label_list.row(item)
                )

        # Remove shapes from the canvas
        self.canvas.shapes = [
            shape
            for shape in self.canvas.shapes
            if shape.label
            not in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]
        ]
        self.canvas.update()

    def find_last_label(self):
        """
        Find the last label in the label list.
        Exclude labels for auto labeling.
        """

        # Get from dialog history
        last_label = self.label_dialog.get_last_label()
        if last_label:
            return last_label

        # Get selected label from the label list
        items = self.label_list.selected_items()
        if items:
            shape = items[0].data(Qt.UserRole)
            return shape.label

        # Get the last label from the label list
        for item in reversed(self.label_list):
            shape = item.data(Qt.UserRole)
            if shape.label not in [
                AutoLabelingMode.OBJECT,
                AutoLabelingMode.ADD,
                AutoLabelingMode.REMOVE,
            ]:
                return shape.label

        # No label is found
        return ""

    def find_last_gid(self):
        last_gid = self.label_dialog.get_last_gid()
        if last_gid is not None:
            return last_gid

        for item in reversed(self.label_list):
            shape = item.data(Qt.UserRole)
            if (
                shape.label
                not in [
                    AutoLabelingMode.OBJECT,
                    AutoLabelingMode.ADD,
                    AutoLabelingMode.REMOVE,
                ]
                and shape.group_id is not None
            ):
                return shape.group_id
        return None

    def set_cache_auto_label(self):
        self.auto_labeling_widget.on_cache_auto_label_changed(
            self.cache_auto_label, self.cache_auto_label_group_id
        )

    def finish_auto_labeling_object(self):
        """Finish auto labeling object."""
        has_object, cache_label = False, None
        for shape in self.canvas.shapes:
            if shape.label == AutoLabelingMode.OBJECT:
                cache_label = shape.cache_label
                cache_description = shape.cache_description
                has_object = True
                break

        # If there is no object, do nothing
        if not has_object:
            return

        # Ask a label for the object
        text, flags, group_id, description, difficult, kie_linking = (
            "",
            {},
            None,
            None,
            False,
            [],
        )
        last_label = self.find_last_label()
        last_gid = (
            self.find_last_gid() if self._config["auto_use_last_gid"] else None
        )
        if self._config["auto_use_last_label"] and last_label:
            text = last_label
            if last_gid is not None:
                group_id = last_gid
        elif cache_label is not None:
            text = cache_label
            description = cache_description
        else:
            previous_text = self.label_dialog.edit.text()
            (
                text,
                flags,
                group_id,
                description,
                difficult,
                kie_linking,
            ) = self.label_dialog.pop_up(
                text=self.find_last_label(),
                flags={},
                group_id=last_gid,
                description=None,
                difficult=False,
                kie_linking=[],
                move_mode=self._config.get("move_mode", "auto"),
            )
            if not text:
                self.label_dialog.edit.setText(previous_text)
                return

        self.cache_auto_label = text
        self.cache_auto_label_group_id = group_id
        if not self.validate_label(text):
            self.error_message(
                self.tr("Invalid label"),
                self.tr("Invalid label '{}' with validation type '{}'").format(
                    text, self._config["validate_label"]
                ),
            )
            return

        if self.attributes and text:
            text = self.reset_attribute(text)

        # Add to label history
        self.label_dialog.add_label_history(text)

        # Update label for the object
        updated_shapes = False
        for shape in self.canvas.shapes:
            if shape.label == AutoLabelingMode.OBJECT:
                updated_shapes = True
                shape.label = text
                shape.flags = flags
                shape.group_id = group_id
                shape.description = description
                shape.difficult = difficult
                shape.kie_linking = kie_linking
                # Update unique label list
                if not self.unique_label_list.find_items_by_label(shape.label):
                    unique_label_item = (
                        self.unique_label_list.create_item_from_label(
                            shape.label
                        )
                    )
                    self.unique_label_list.addItem(unique_label_item)
                    rgb = self._get_rgb_by_label(shape.label)
                    self.unique_label_list.set_item_label(
                        unique_label_item, shape.label, rgb, LABEL_OPACITY
                    )

                # Update label list
                self._update_shape_color(shape)
                item = self.label_list.find_item_by_shape(shape)
                if shape.group_id is None:
                    color = shape.fill_color.getRgb()[:3]
                    item.setText(
                        '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                            html.escape(shape.label), *color
                        )
                    )
                else:
                    item.setText(f"{shape.label} ({shape.group_id})")

        # Clean up auto labeling objects
        self.clear_auto_labeling_marks()

        # Update shape colors
        for shape in self.canvas.shapes:
            self._update_shape_color(shape)
            color = shape.fill_color.getRgb()[:3]
            item = self.label_list.find_item_by_shape(shape)
            item.setText("{}".format(html.escape(shape.label)))
            item.setBackground(QtGui.QColor(*color, LABEL_OPACITY))
            self.unique_label_list.update_item_color(
                shape.label, color, LABEL_OPACITY
            )

        if updated_shapes:
            self.set_dirty()

    def shape_text_changed(self):
        description = self.shape_text_edit.toPlainText()
        if self.canvas.current is not None:
            self.canvas.current.description = description
        elif self.canvas.editing() and len(self.canvas.selected_shapes) == 1:
            self.canvas.selected_shapes[0].description = description
        else:
            self.other_data["description"] = description
        self.set_dirty()

    def set_text_editing(self, enable):
        """Set text editing."""
        if enable:
            # Enable text editing and set shape text from selected shape
            if len(self.canvas.selected_shapes) == 1:
                self.shape_text_label.setText(self.tr("Object Description"))
                self.shape_text_edit.textChanged.disconnect()
                self.shape_text_edit.setPlainText(
                    self.canvas.selected_shapes[0].description
                )
                self.shape_text_edit.textChanged.connect(
                    self.shape_text_changed
                )
            else:
                self.shape_text_label.setText(self.tr("Image Description"))
                self.shape_text_edit.textChanged.disconnect()
                self.shape_text_edit.setPlainText(
                    self.other_data.get("description", "")
                )
                self.shape_text_edit.textChanged.connect(
                    self.shape_text_changed
                )
            self.shape_text_edit.setDisabled(False)
        else:
            self.shape_text_edit.setDisabled(True)
            self.shape_text_label.setText(self.tr("Description"))
            self.shape_text_edit.textChanged.disconnect()
            self.shape_text_edit.setPlainText("")
            self.shape_text_edit.textChanged.connect(self.shape_text_changed)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.shape_text_edit.setFont(font)
        self.shape_text_label.setFont(font)

    def group_selected_shapes(self):
        self.canvas.group_selected_shapes()
        self.set_dirty()
        self.load_file(self.filename)

    def ungroup_selected_shapes(self):
        self.canvas.ungroup_selected_shapes()
        self.set_dirty()
        self.load_file(self.filename)

    def update_thumbnail_pixmap(self):
        if self.thumbnail_pixmap and not self.thumbnail_pixmap.isNull():
            width = self.thumbnail_image_label.width()
            if width > 0:
                self.thumbnail_image_label.setPixmap(
                    self.thumbnail_pixmap.scaledToWidth(
                        width, QtCore.Qt.SmoothTransformation
                    )
                )

    def update_thumbnail_display(self):
        self.thumbnail_pixmap = None
        self.thumbnail_image_label.clear()
        self.thumbnail_container.hide()

        model_config = (
            self.auto_labeling_widget.model_manager.loaded_model_config
        )
        supported_model_list = list(_THUMBNAIL_RENDER_MODELS.keys())
        if not (
            model_config
            and model_config.get("type") in supported_model_list
            and self.image_list
        ):
            return

        try:
            image_dir = osp.dirname(self.filename)
            parent_dir = osp.dirname(image_dir)
            base_name = osp.splitext(osp.basename(self.filename))[0]
            save_dir, _thumbnail_file_ext = _THUMBNAIL_RENDER_MODELS[
                model_config["type"]
            ]
            thumbnail_dir = osp.join(parent_dir, save_dir)
            thumbnail_path = osp.join(
                thumbnail_dir, base_name + _thumbnail_file_ext
            )
            if not osp.exists(thumbnail_path):
                return

            self.thumbnail_pixmap = QtGui.QPixmap(thumbnail_path)
            if not self.thumbnail_pixmap.isNull():
                self.thumbnail_container.show()
                self.update_thumbnail_pixmap()

        except Exception as e:
            logger.error(f"Failed to load thumbnail image: {str(e)}")

    def toggle_description_visibility(self, checked):
        self.shape_text_edit.setVisible(checked)

    def toggle_labels_visibility(self, checked):
        if checked:
            self.label_dock.widget().setVisible(True)
            self.label_dock.setMinimumHeight(2)
            self.label_dock.setMaximumHeight(16777215)
        else:
            self.label_dock.widget().setVisible(False)
            self.label_dock.setMinimumHeight(2)
            self.label_dock.setMaximumHeight(2)
