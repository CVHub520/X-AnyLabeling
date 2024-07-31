import functools
import html
import math
import json
import os
import os.path as osp
import shutil
import pathlib
import cv2
import re
import yaml
import webbrowser
from difflib import SequenceMatcher

import imgviz
import natsort
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QDockWidget,
    QGridLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWhatsThis,
    QWidget,
    QMessageBox,
    QProgressDialog,
    QScrollArea,
)

from anylabeling.services.auto_labeling.types import AutoLabelingMode

from ...app_info import (
    __appname__,
    __version__,
    __preferred_device__,
)
from . import utils
from ...config import get_config, save_config
from .label_file import LabelFile, LabelFileError
from .label_converter import LabelConverter
from .logger import logger
from .shape import Shape
from .widgets import (
    AutoLabelingWidget,
    BrightnessContrastDialog,
    Canvas,
    CrosshairSettingsDialog,
    FileDialogPreview,
    TextInputDialog,
    LabelDialog,
    LabelFilterComboBox,
    LabelListWidget,
    LabelListWidgetItem,
    LabelModifyDialog,
    OverviewDialog,
    ToolBar,
    UniqueLabelQListWidget,
    ZoomWidget,
)

LABEL_COLORMAP = imgviz.label_colormap()

# Green for the first label
LABEL_COLORMAP[2] = LABEL_COLORMAP[1]
LABEL_COLORMAP[1] = [0, 180, 33]
LABEL_OPACITY = 128


class LabelingWidget(LabelDialog):
    """The main widget for labeling images"""

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2
    next_files_changed = QtCore.pyqtSignal(list)

    def __init__(
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
        self.current_category = None
        self.selected_polygon_stack = []
        self.supported_shape = Shape.get_supported_shape()
        self.hidden_cls = []
        self.label_info = {}
        self.image_flags = []

        # see configs/anylabeling_config.yaml for valid configuration
        if config is None:
            config = get_config()
        self._config = config
        self.label_flags = self._config["label_flags"]

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
            "QDockWidget::title {"
            "text-align: center;"
            "padding: 0px;"
            "background-color: #f0f0f0;"
            "}"
        )

        # Create and add combobox for showing unique labels in group
        self.label_filter_combobox = LabelFilterComboBox(self)

        self.label_list.item_selection_changed.connect(
            self.label_selection_changed
        )
        self.label_list.item_double_clicked.connect(self.edit_label)
        self.label_list.item_changed.connect(self.label_item_changed)
        self.label_list.item_dropped.connect(self.label_order_changed)
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Objects"), self)
        self.shape_dock.setWidget(self.label_list)
        self.shape_dock.setStyleSheet(
            "QDockWidget::title {"
            "text-align: center;"
            "padding: 0px;"
            "background-color: #f0f0f0;"
            "}"
        )
        self.shape_dock.setTitleBarWidget(QtWidgets.QWidget())

        self.unique_label_list = UniqueLabelQListWidget()
        self.unique_label_list.setToolTip(
            self.tr(
                "Select label to start annotating for it. "
                "Press 'Esc' to deselect."
            )
        )
        if self._config["labels"]:
            for label in self._config["labels"]:
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )
        self.label_dock = QtWidgets.QDockWidget(self.tr("Labels"), self)
        self.label_dock.setObjectName("Labels")
        self.label_dock.setWidget(self.unique_label_list)
        self.label_dock.setStyleSheet(
            "QDockWidget::title {"
            "text-align: center;"
            "padding: 0px;"
            "background-color: #f0f0f0;"
            "}"
        )
        self.file_search = QtWidgets.QLineEdit()
        self.file_search.setPlaceholderText(self.tr("Search Filename"))
        self.file_search.textChanged.connect(self.file_search_changed)
        self.file_list_widget = QtWidgets.QListWidget()
        self.file_list_widget.itemSelectionChanged.connect(
            self.file_selection_changed
        )
        file_list_layout = QtWidgets.QVBoxLayout()
        file_list_layout.setContentsMargins(0, 0, 0, 0)
        file_list_layout.setSpacing(0)
        file_list_layout.addWidget(self.file_search)
        file_list_layout.addWidget(self.file_list_widget)
        self.file_dock = QtWidgets.QDockWidget(self.tr("Files"), self)
        self.file_dock.setObjectName("Files")
        file_list_widget = QtWidgets.QWidget()
        file_list_widget.setLayout(file_list_layout)
        self.file_dock.setWidget(file_list_widget)
        self.file_dock.setStyleSheet(
            "QDockWidget::title {"
            "text-align: center;"
            "padding: 0px;"
            "background-color: #f0f0f0;"
            "}"
        )

        self.zoom_widget = ZoomWidget()
        self.setAcceptDrops(True)

        self.canvas = self.label_list.canvas = Canvas(
            parent=self,
            epsilon=self._config["epsilon"],
            double_click=self._config["canvas"]["double_click"],
            num_backups=self._config["canvas"]["num_backups"],
        )
        self.canvas.zoom_request.connect(self.zoom_request)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)
        self.scroll_bars = {
            Qt.Vertical: scroll_area.verticalScrollBar(),
            Qt.Horizontal: scroll_area.horizontalScrollBar(),
        }
        self.canvas.scroll_request.connect(self.scroll_request)
        self.canvas.new_shape.connect(self.new_shape)
        self.canvas.show_shape.connect(self.show_shape)
        self.canvas.shape_moved.connect(self.set_dirty)
        self.canvas.shape_rotated.connect(self.set_dirty)
        self.canvas.selection_changed.connect(self.shape_selection_changed)
        self.canvas.drawing_polygon.connect(self.toggle_drawing_sensitive)
        # [Feature] support for automatically switching to editing mode 
        # when the cursor moves over an object
        self.canvas.h_shape_is_hovered = self._config.get("auto_highlight_shape", False)
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
            self.tr("&Open File"),
            self.open_file,
            shortcuts["open"],
            "file",
            self.tr("Open image or label file"),
        )
        openvideo = action(
            self.tr("&Open Video"),
            self.open_video_file,
            shortcuts["open_video"],
            "video",
            self.tr("Open video file"),
        )
        opendir = action(
            self.tr("&Open Dir"),
            self.open_folder_dialog,
            shortcuts["open_dir"],
            "open",
            self.tr("Open Dir"),
        )
        open_next_image = action(
            self.tr("&Next Image"),
            self.open_next_image,
            shortcuts["open_next"],
            "next",
            self.tr("Open next (hold Ctrl+Shift to move to the next labeled image)"),
            enabled=False,
        )
        open_prev_image = action(
            self.tr("&Prev Image"),
            self.open_prev_image,
            shortcuts["open_prev"],
            "prev",
            self.tr("Open prev (hold Ctrl+Shift to move to the prev labeled image)"),
            enabled=False,
        )
        save = action(
            self.tr("&Save"),
            self.save_file,
            shortcuts["save"],
            "save",
            self.tr("Save labels to file"),
            enabled=False,
        )
        save_as = action(
            self.tr("&Save As"),
            self.save_file_as,
            shortcuts["save_as"],
            "save-as",
            self.tr("Save labels to a different file"),
            enabled=False,
        )
        run_all_images = action(
            self.tr("&Auto Run"),
            self.run_all_images,
            shortcuts["auto_run"],
            "auto-run",
            self.tr("Auto run all images at once"),
            checkable=True,
            enabled=False,
        )
        delete_file = action(
            self.tr("&Delete File"),
            self.delete_file,
            shortcuts["delete_file"],
            "delete",
            self.tr("Delete current label file"),
            enabled=False,
        )
        delete_image_file = action(
            self.tr("&Delete Image File"),
            self.delete_image_file,
            shortcuts["delete_image_file"],
            "delete",
            self.tr("Delete current image file"),
            enabled=True,
        )

        change_output_dir = action(
            self.tr("&Change Output Dir"),
            slot=self.change_output_dir_dialog,
            shortcut=shortcuts["save_to"],
            icon="open",
            tip=self.tr("Change where annotations are loaded/saved"),
        )

        save_auto = action(
            text=self.tr("Save &Automatically"),
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
            self.tr("&Close"),
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
            self.canvas.group_selected_shapes,
            shortcuts["group_selected_shapes"],
            None,
            self.tr("Group shapes by assigning a same group_id"),
            enabled=True,
        )
        ungroup_selected_shapes = action(
            self.tr("Ungroup Selected Shapes"),
            self.canvas.ungroup_selected_shapes,
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
            enabled=False,
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
            self.tr("&Overview"),
            self.overview,
            shortcuts["show_overview"],
            icon="overview",
            tip=self.tr("Show annotations statistics"),
        )
        save_crop = action(
            self.tr("&Save Cropped Image"),
            self.save_crop,
            icon="crop",
            tip=self.tr(
                "Save cropped image. (Support rectangle/rotation/polygon shape_type)"
            ),
        )
        label_manager = action(
            self.tr("&Label Manager"),
            self.label_manager,
            icon="edit",
            tip=self.tr("Manage Labels: Rename, Delete, Show/Hide, Adjust Color"),
        )
        union_selection = action(
            self.tr("&Union Selection"),
            self.union_selection,
            icon="union",
            tip=self.tr("Union multiple selected rectangle shapes"),
        )
        hbb_to_obb = action(
            self.tr("&Convert HBB to OBB"),
            self.hbb_to_obb,
            icon="convert",
            tip=self.tr(
                "Perform conversion from horizontal bounding box to oriented bounding box"
            ),
        )
        obb_to_hbb = action(
            self.tr("&Convert OBB to HBB"),
            self.obb_to_hbb,
            icon="convert",
            tip=self.tr(
                "Perform conversion from oriented bounding box to horizontal bounding box"
            ),
        )
        polygon_to_hbb = action(
            self.tr("&Convert Polygon to HBB"),
            self.polygon_to_hbb,
            icon="convert",
            tip=self.tr(
                "Perform conversion from polygon to horizontal bounding box"
            ),
        )

        documentation = action(
            self.tr("&Documentation"),
            self.documentation,
            icon="docs",
            tip=self.tr("Show documentation"),
        )
        contact = action(
            self.tr("&Contact me"),
            self.contact,
            icon="contact",
            tip=self.tr("Show contact page"),
        )
        information = action(
            self.tr("&Information"),
            self.information,
            icon="help",
            tip=self.tr("Show system information"),
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
            self.tr("Zoom &In"),
            functools.partial(self.add_zoom, 1.1),
            shortcuts["zoom_in"],
            "zoom-in",
            self.tr("Increase zoom level"),
            enabled=False,
        )
        zoom_out = action(
            self.tr("&Zoom Out"),
            functools.partial(self.add_zoom, 0.9),
            shortcuts["zoom_out"],
            "zoom-out",
            self.tr("Decrease zoom level"),
            enabled=False,
        )
        zoom_org = action(
            self.tr("&Original size"),
            functools.partial(self.set_zoom, 100),
            shortcuts["zoom_to_original"],
            "zoom",
            self.tr("Zoom to original size"),
            enabled=False,
        )
        keep_prev_scale = action(
            self.tr("&Keep Previous Scale"),
            lambda x: self._config.update({"keep_prev_scale": x}),
            tip=self.tr("Keep previous zoom scale"),
            checkable=True,
            checked=self._config["keep_prev_scale"],
            enabled=True,
        )
        keep_prev_brightness = action(
            self.tr("&Keep Previous Brightness"),
            lambda x: self._config.update({"keep_prev_brightness": x}),
            tip=self.tr("Keep previous brightness"),
            checkable=True,
            checked=self._config["keep_prev_brightness"],
            enabled=True,
        )
        keep_prev_contrast = action(
            self.tr("&Keep Previous Contrast"),
            lambda x: self._config.update({"keep_prev_contrast": x}),
            tip=self.tr("Keep previous contrast"),
            checkable=True,
            checked=self._config["keep_prev_contrast"],
            enabled=True,
        )
        fit_window = action(
            self.tr("&Fit Window"),
            self.set_fit_window,
            shortcuts["fit_window"],
            "fit-window",
            self.tr("Zoom follows window size"),
            checkable=True,
            enabled=False,
        )
        fit_width = action(
            self.tr("Fit &Width"),
            self.set_fit_width,
            shortcuts["fit_width"],
            "fit-width",
            self.tr("Zoom follows window width"),
            checkable=True,
            enabled=False,
        )
        brightness_contrast = action(
            self.tr("&Set Brightness Contrast"),
            self.brightness_contrast,
            None,
            "color",
            "Adjust brightness and contrast",
            enabled=False,
        )
        set_cross_line = action(
            self.tr("&Set Cross Line"),
            self.set_cross_line,
            tip=self.tr("Set cross line for mouse position"),
            icon="cartesian",
        )
        show_groups = action(
            self.tr("&Show Groups"),
            lambda x: self.set_canvas_params("show_groups", x),
            tip=self.tr("Show shape groups"),
            icon=None,
            checkable=True,
            checked=self._config["show_groups"],
            enabled=True,
            auto_trigger=True,
        )
        show_texts = action(
            self.tr("&Show Texts"),
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
            self.tr("&Show Labels"),
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
            self.tr("&Show Scores"),
            lambda x: self.set_canvas_params("show_scores", x),
            tip=self.tr("Show score inside shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_scores"],
            enabled=True,
            auto_trigger=True,
        )
        show_degrees = action(
            self.tr("&Show Degress"),
            lambda x: self.set_canvas_params("show_degrees", x),
            tip=self.tr("Show degrees above rotated shapes"),
            icon=None,
            checkable=True,
            checked=self._config["show_degrees"],
            enabled=True,
            auto_trigger=True,
        )
        show_linking = action(
            self.tr("&Show KIE Linking"),
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
            self.tr("&Upload Image Flags File"),
            self.upload_image_flags_file,
            None,
            icon="format_classify",
            tip=self.tr("Upload Custom Image Flags File"),
        )
        upload_label_flags_file = action(
            self.tr("&Upload Label Flags File"),
            self.upload_label_flags_file,
            None,
            icon="format_classify",
            tip=self.tr("Upload Custom Label Flags File"),
        )
        upload_shape_attrs_file = action(
            self.tr("&Upload Attributes File"),
            self.upload_shape_attrs_file,
            None,
            icon="format_classify",
            tip=self.tr("Upload Custom Attributes File"),
        )
        upload_yolo_hbb_annotation = action(
            self.tr("&Upload YOLO-Hbb Annotations"),
            lambda: self.upload_yolo_annotation("hbb"),
            None,
            icon="format_yolo",
            tip=self.tr(
                "Upload Custom YOLO Horizontal Bounding Boxes Annotations"
            ),
        )
        upload_yolo_obb_annotation = action(
            self.tr("&Upload YOLO-Obb Annotations"),
            lambda: self.upload_yolo_annotation("obb"),
            None,
            icon="format_yolo",
            tip=self.tr(
                "Upload Custom YOLO Oriented Bounding Boxes Annotations"
            ),
        )
        upload_yolo_seg_annotation = action(
            self.tr("&Upload YOLO-Seg Annotations"),
            lambda: self.upload_yolo_annotation("seg"),
            None,
            icon="format_yolo",
            tip=self.tr("Upload Custom YOLO Segmentation Annotations"),
        )
        upload_yolo_pose_annotation = action(
            self.tr("&Upload YOLO-Pose Annotations"),
            lambda: self.upload_yolo_annotation("pose"),
            None,
            icon="format_yolo",
            tip=self.tr("Upload Custom YOLO Pose Annotations"),
        )
        upload_voc_det_annotation = action(
            self.tr("&Upload VOC Detection Annotations"),
            lambda: self.upload_voc_annotation("rectangle"),
            None,
            icon="format_voc",
            tip=self.tr("Upload Custom Pascal VOC Detection Annotations"),
        )
        upload_voc_seg_annotation = action(
            self.tr("&Upload VOC Segmentation Annotations"),
            lambda: self.upload_voc_annotation("polygon"),
            None,
            icon="format_voc",
            tip=self.tr("Upload Custom Pascal VOC Segmentation Annotations"),
        )
        upload_coco_det_annotation = action(
            self.tr("&Upload COCO Detection Annotations"),
            lambda: self.upload_coco_annotation("rectangle"),
            None,
            icon="format_coco",
            tip=self.tr("Upload Custom COCO Detection Annotations"),
        )
        upload_coco_seg_annotation = action(
            self.tr("&Upload COCO Segmentation Annotations"),
            lambda: self.upload_coco_annotation("polygon"),
            None,
            icon="format_coco",
            tip=self.tr("Upload Custom COCO Segmentation Annotations"),
        )
        upload_dota_annotation = action(
            self.tr("&Upload DOTA Annotations"),
            self.upload_dota_annotation,
            None,
            icon="format_dota",
            tip=self.tr("Upload Custom DOTA Annotations"),
        )
        upload_mask_annotation = action(
            self.tr("&Upload MASK Annotations"),
            self.upload_mask_annotation,
            None,
            icon="format_mask",
            tip=self.tr("Upload Custom MASK Annotations"),
        )
        upload_mot_annotation = action(
            self.tr("&Upload MOT Annotations"),
            self.upload_mot_annotation,
            None,
            icon="format_mot",
            tip=self.tr("Upload Custom Multi-Object-Tracking Annotations"),
        )
        upload_odvg_annotation = action(
            self.tr("&Upload ODVG Annotations"),
            self.upload_odvg_annotation,
            None,
            icon="format_odvg",
            tip=self.tr("Upload Custom Object Detection Visual Grounding Annotations"),
        )
        upload_ppocr_rec_annotation = action(
            self.tr("&Upload PPOCR-Rec Annotations"),
            lambda: self.upload_ppocr_annotation("rec"),
            None,
            icon="format_ppocr",
            tip=self.tr("Upload Custom PPOCR Recognition Annotations"),
        )
        upload_ppocr_kie_annotation = action(
            self.tr("&Upload PPOCR-KIE Annotations"),
            lambda: self.upload_ppocr_annotation("kie"),
            None,
            icon="format_ppocr",
            tip=self.tr("Upload Custom PPOCR Key Information Extraction (KIE - Semantic Entity Recognition & Relation Extraction) Annotations"),
        )

        # Export
        export_yolo_hbb_annotation = action(
            self.tr("&Export YOLO-Hbb Annotations"),
            lambda: self.export_yolo_annotation("hbb"),
            None,
            icon="format_yolo",
            tip=self.tr(
                "Export Custom YOLO Horizontal Bounding Boxes Annotations"
            ),
        )
        export_yolo_obb_annotation = action(
            self.tr("&Export YOLO-Obb Annotations"),
            lambda: self.export_yolo_annotation("obb"),
            None,
            icon="format_yolo",
            tip=self.tr(
                "Export Custom YOLO Oriented Bounding Boxes Annotations"
            ),
        )
        export_yolo_seg_annotation = action(
            self.tr("&Export YOLO-Seg Annotations"),
            lambda: self.export_yolo_annotation("seg"),
            None,
            icon="format_yolo",
            tip=self.tr("Export Custom YOLO Segmentation Annotations"),
        )
        export_yolo_pose_annotation = action(
            self.tr("&Export YOLO-Pose Annotations"),
            lambda: self.export_yolo_annotation("pose"),
            None,
            icon="format_yolo",
            tip=self.tr("Export Custom YOLO Pose Annotations"),
        )
        export_voc_det_annotation = action(
            self.tr("&Export VOC Detection Annotations"),
            lambda: self.export_voc_annotation("rectangle"),
            None,
            icon="format_voc",
            tip=self.tr("Export Custom PASCAL VOC Detection Annotations"),
        )
        export_voc_seg_annotation = action(
            self.tr("&Export VOC Segmentation Annotations"),
            lambda: self.export_voc_annotation("polygon"),
            None,
            icon="format_voc",
            tip=self.tr("Export Custom PASCAL VOC Segmentation Annotations"),
        )
        export_coco_det_annotation = action(
            self.tr("&Export COCO Detection Annotations"),
            lambda: self.export_coco_annotation("rectangle"),
            None,
            icon="format_coco",
            tip=self.tr("Export Custom COCO Rectangle Annotations"),
        )
        export_coco_seg_annotation = action(
            self.tr("&Export COCO Segmentation Annotations"),
            lambda: self.export_coco_annotation("polygon"),
            None,
            icon="format_coco",
            tip=self.tr("Export Custom COCO Segmentation Annotations"),
        )
        export_dota_annotation = action(
            self.tr("&Export DOTA Annotations"),
            self.export_dota_annotation,
            None,
            icon="format_dota",
            tip=self.tr("Export Custom DOTA Annotations"),
        )
        export_mask_annotation = action(
            self.tr("&Export MASK Annotations"),
            self.export_mask_annotation,
            None,
            icon="format_mask",
            tip=self.tr("Export Custom MASK Annotations - RGB/Gray"),
        )
        export_mot_annotation = action(
            self.tr("&Export MOT Annotations"),
            self.export_mot_annotation,
            None,
            icon="format_mot",
            tip=self.tr("Export Custom Multi-Object-Tracking Annotations"),
        )
        export_odvg_annotation = action(
            self.tr("&Export ODVG Annotations"),
            self.export_odvg_annotation,
            None,
            icon="format_odvg",
            tip=self.tr("Export Custom Object Detection Visual Grounding Annotations"),
        )
        export_pporc_rec_annotation = action(
            self.tr("&Export PPOCR-Rec Annotations"),
            lambda: self.export_pporc_annotation("rec"),
            None,
            icon="format_ppocr",
            tip=self.tr("Export Custom PPOCR Recognition Annotations"),
        )
        export_pporc_kie_annotation = action(
            self.tr("&Export PPOCR-KIE Annotations"),
            lambda: self.export_pporc_annotation("kie"),
            None,
            icon="format_ppocr",
            tip=self.tr("Export Custom PPOCR Key Information Extraction (KIE - Semantic Entity Recognition & Relation Extraction) Annotations"),
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
            self.tr("&Edit Label"),
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

        # AI Actions
        toggle_auto_labeling_widget = action(
            self.tr("&Auto Labeling"),
            self.toggle_auto_labeling_widget,
            shortcuts["auto_label"],
            "brain",
            self.tr("Auto Labeling"),
        )

        # Label list context menu.
        label_menu = QtWidgets.QMenu()
        utils.add_actions(label_menu, (edit, delete))
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
            visibility_shapes_mode=visibility_shapes_mode,
            run_all_images=run_all_images,
            union_selection=union_selection,
            delete=delete,
            edit=edit,
            duplicate=duplicate,
            copy=copy,
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
            upload_image_flags_file=upload_image_flags_file,
            upload_label_flags_file=upload_label_flags_file,
            upload_shape_attrs_file=upload_shape_attrs_file,
            upload_yolo_hbb_annotation=upload_yolo_hbb_annotation,
            upload_yolo_obb_annotation=upload_yolo_obb_annotation,
            upload_yolo_seg_annotation=upload_yolo_seg_annotation,
            upload_yolo_pose_annotation=upload_yolo_pose_annotation,
            upload_voc_det_annotation=upload_voc_det_annotation,
            upload_voc_seg_annotation=upload_voc_seg_annotation,
            upload_coco_det_annotation=upload_coco_det_annotation,
            upload_coco_seg_annotation=upload_coco_seg_annotation,
            upload_dota_annotation=upload_dota_annotation,
            upload_mask_annotation=upload_mask_annotation,
            upload_mot_annotation=upload_mot_annotation,
            upload_odvg_annotation=upload_odvg_annotation,
            upload_ppocr_rec_annotation=upload_ppocr_rec_annotation,
            upload_ppocr_kie_annotation=upload_ppocr_kie_annotation,
            export_yolo_hbb_annotation=export_yolo_hbb_annotation,
            export_yolo_obb_annotation=export_yolo_obb_annotation,
            export_yolo_seg_annotation=export_yolo_seg_annotation,
            export_yolo_pose_annotation=export_yolo_pose_annotation,
            export_voc_det_annotation=export_voc_det_annotation,
            export_voc_seg_annotation=export_voc_seg_annotation,
            export_coco_det_annotation=export_coco_det_annotation,
            export_coco_seg_annotation=export_coco_seg_annotation,
            export_dota_annotation=export_dota_annotation,
            export_mask_annotation=export_mask_annotation,
            export_mot_annotation=export_mot_annotation,
            export_odvg_annotation=export_odvg_annotation,
            export_pporc_rec_annotation=export_pporc_rec_annotation,
            export_pporc_kie_annotation=export_pporc_kie_annotation,
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
            show_texts=show_texts,
            show_labels=show_labels,
            show_scores=show_scores,
            show_degrees=show_degrees,
            show_linking=show_linking,
            zoom_actions=zoom_actions,
            open_next_image=open_next_image,
            open_prev_image=open_prev_image,
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
                remove_point,
                None,
                keep_prev_mode,
                auto_use_last_label_mode,
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
                edit_mode,
                brightness_contrast,
            ),
            on_shapes_present=(save_as,),
            hide_selected_polygons=hide_selected_polygons,
            show_hidden_polygons=show_hidden_polygons,
            group_selected_shapes=group_selected_shapes,
            ungroup_selected_shapes=ungroup_selected_shapes,
        )

        self.canvas.vertex_selected.connect(
            self.actions.remove_point.setEnabled
        )

        self.menus = utils.Struct(
            file=self.menu(self.tr("&File")),
            edit=self.menu(self.tr("&Edit")),
            view=self.menu(self.tr("&View")),
            language=self.menu(self.tr("&Language")),
            upload=self.menu(self.tr("&Upload")),
            export=self.menu(self.tr("&Export")),
            tool=self.menu(self.tr("&Tool")),
            help=self.menu(self.tr("&Help")),
            recent_files=QtWidgets.QMenu(self.tr("Open &Recent")),
            label_list=label_menu,
        )

        utils.add_actions(
            self.menus.file,
            (
                open_,
                open_next_image,
                open_prev_image,
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
        utils.add_actions(
            self.menus.tool,
            (
                overview,
                None,
                save_crop,
                None,
                label_manager,
                None,
                hbb_to_obb,
                obb_to_hbb,
                polygon_to_hbb,
            ),
        )
        utils.add_actions(
            self.menus.help,
            (
                documentation,
                contact,
                information,
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
                None,
                upload_dota_annotation,
                upload_mask_annotation,
                upload_mot_annotation,
                upload_odvg_annotation,
                None,
                upload_ppocr_rec_annotation,
                upload_ppocr_kie_annotation,
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
                None,
                export_dota_annotation,
                export_mask_annotation,
                export_mot_annotation,
                export_odvg_annotation,
                None,
                export_pporc_rec_annotation,
                export_pporc_kie_annotation,
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
                fill_drawing,
                None,
                None,
                zoom_in,
                zoom_out,
                zoom_org,
                keep_prev_scale,
                keep_prev_brightness,
                keep_prev_contrast,
                None,
                fit_window,
                fit_width,
                None,
                brightness_contrast,
                set_cross_line,
                show_texts,
                show_labels,
                show_scores,
                show_degrees,
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
            edit_mode,
            delete,
            undo,
            None,
            zoom,
            fit_width,
            toggle_auto_labeling_widget,
            run_all_images,
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
        self.auto_labeling_widget.clear_auto_labeling_action_requested.connect(
            self.clear_auto_labeling_marks
        )
        self.auto_labeling_widget.finish_auto_labeling_object_action_requested.connect(
            self.finish_auto_labeling_object
        )
        self.auto_labeling_widget.model_manager.prediction_started.connect(
            lambda: self.canvas.set_loading(True, self.tr("Please wait..."))
        )
        self.auto_labeling_widget.model_manager.prediction_finished.connect(
            lambda: self.canvas.set_loading(False)
        )
        self.next_files_changed.connect(
            self.auto_labeling_widget.model_manager.on_next_files_changed
        )
        self.auto_labeling_widget.model_manager.request_next_files_requested.connect(
            lambda: self.inform_next_files(self.filename)
        )
        self.auto_labeling_widget.hide()  # Hide by default
        central_layout.addWidget(self.label_instruction)
        central_layout.addWidget(self.auto_labeling_widget)
        central_layout.addWidget(scroll_area)
        layout.addItem(central_layout)

        # Save central area for resize
        self._central_widget = scroll_area

        # Stretch central area (image view)
        layout.setStretch(1, 1)

        right_sidebar_layout = QVBoxLayout()
        right_sidebar_layout.setContentsMargins(0, 0, 0, 0)

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

        # Shape text label
        self.shape_text_label = QLabel("Object Text")
        self.shape_text_edit = QPlainTextEdit()
        right_sidebar_layout.addWidget(
            self.shape_text_label, 0, Qt.AlignCenter
        )
        right_sidebar_layout.addWidget(self.shape_text_edit)
        right_sidebar_layout.addWidget(self.flag_dock)
        right_sidebar_layout.addWidget(self.label_dock)
        right_sidebar_layout.addWidget(self.label_filter_combobox)
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
            f" {text_rotation}(<b>O</b>)"
        )

    @pyqtSlot()
    def on_auto_segmentation_requested(self):
        self.canvas.set_auto_labeling(True)
        self.label_instruction.setText(self.get_labeling_instruction())

    @pyqtSlot()
    def on_auto_segmentation_disabled(self):
        self.canvas.set_auto_labeling(False)
        self.label_instruction.setText(self.get_labeling_instruction())

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
            return
        self.dirty = True
        self.actions.save.setEnabled(True)
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
        title = __appname__
        if self.filename is not None:
            title = f"{title} - {self.filename}"
        self.setWindowTitle(title)

        if self.has_label_file():
            self.actions.delete_file.setEnabled(True)
        else:
            self.actions.delete_file.setEnabled(False)

    def toggle_actions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for action in self.actions.zoom_actions:
            action.setEnabled(value)
        for action in self.actions.on_load_active:
            action.setEnabled(value)

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
        self.label_filter_combobox.combo_box.clear()

    def reset_attribute(self, text):
        valid_labels = list(self.attributes.keys())
        if text not in valid_labels:
            most_similar_label = self.find_most_similar_label(
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
        self.load_shapes(self.canvas.shapes)
        self.actions.undo.setEnabled(self.canvas.is_shape_restorable)

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

    def union_selection(self):
        """
        Merges selected shapes into one shape.
        """
        if len(self.canvas.selected_shapes) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please select at least two shapes to perform union."),
                QtWidgets.QMessageBox.Ok,
            )
            return

        # Get rectangle of all selected shapes
        rectangle_shapes = []
        for shape in self.canvas.selected_shapes:
            points = shape.points
            # Convert QPointF objects to tuples
            xmin, ymin = (points[0].x(), points[0].y())
            xmax, ymax = (points[2].x(), points[2].y())
            rectangle_shapes.append([xmin, ymin, xmax, ymax])

        # Calculate the rectangle
        min_x = min([bbox[0] for bbox in rectangle_shapes])
        min_y = min([bbox[1] for bbox in rectangle_shapes])
        max_x = max([bbox[2] for bbox in rectangle_shapes])
        max_y = max([bbox[3] for bbox in rectangle_shapes])

        # Create a new rectangle shape representing the union
        union_shape = shape.copy()
        union_shape.points[0].setX(min_x)
        union_shape.points[0].setY(min_y)
        union_shape.points[1].setX(max_x)
        union_shape.points[1].setY(min_y)
        union_shape.points[2].setX(max_x)
        union_shape.points[2].setY(max_y)
        union_shape.points[3].setX(min_x)
        union_shape.points[3].setY(max_y)
        self.add_label(union_shape)

        # clear selected shapes
        self.remove_labels(self.canvas.delete_selected())
        self.set_dirty()

        # Update UI state
        if self.no_shape():
            for action in self.actions.on_shapes_present:
                action.setEnabled(False)

    # Tools
    def overview(self):
        if self.filename:
            OverviewDialog(parent=self)

    def save_crop(self):
        if not self.filename:
            return

        image_file_list, label_dir_path = [], ""
        if not self.image_list and self.filename:
            image_file_list = [self.filename]
            dir_path, filename = osp.split(self.filename)
            label_file = osp.join(
                dir_path, osp.splitext(filename)[0] + ".json"
            )
            if osp.exists(label_file):
                label_dir_path = dir_path
        elif self.image_list and not self.output_dir and self.filename:
            image_file_list = self.image_list
            label_dir_path = osp.dirname(self.filename)
        if self.output_dir:
            label_dir_path = self.output_dir
        save_path = osp.join(
            osp.dirname(self.filename), "..", "x-anylabeling-crops"
        )
        if osp.exists(save_path):
            shutil.rmtree(save_path)

        progress_dialog = QProgressDialog(
            self.tr("Processing..."),
            self.tr("Cancel"),
            0,
            len(image_file_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet(
            """
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """
        )
        label_to_count = {}
        try:
            for i, image_file in enumerate(image_file_list):
                image_name = osp.basename(image_file)
                label_name = osp.splitext(image_name)[0] + ".json"
                label_file = osp.join(label_dir_path, label_name)
                if not osp.exists(label_file):
                    continue

                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                shapes = data["shapes"]
                for shape in shapes:
                    label = shape["label"]
                    points = shape["points"]
                    shape_type = shape["shape_type"]
                    if shape_type not in ["rectangle", "polygon", "rotation"]:
                        continue
                    if (
                        (shape_type == "polygon" and len(points) < 3)
                        or (shape_type == "rotation" and len(points) != 4)
                        or (shape_type == "rectangle" and len(points) != 4)
                    ):
                        progress_dialog.close()
                        error_dialog = QMessageBox()
                        error_dialog.setIcon(QMessageBox.Critical)
                        error_dialog.setText(self.tr("Existing invalid shape!"))
                        error_dialog.setInformativeText(label_file)
                        error_dialog.setWindowTitle(self.tr("Critical"))
                        error_dialog.exec_()
                        progress_dialog.close()
                        return

                    points = np.array(points).astype(np.int32)
                    x, y, w, h = cv2.boundingRect(points)
                    xmin = int(x)
                    ymin = int(y)
                    xmax = int(w) + xmin
                    ymax = int(h) + ymin

                    dst_path = osp.join(save_path, label)
                    if not osp.exists(dst_path):
                        label_to_count[label] = 0
                        os.makedirs(dst_path)
                    else:
                        label_to_count[label] += 1

                    image = cv2.imread(image_file)
                    height, width = image.shape[:2]
                    xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                    xmin = max(0, min(xmin, width))
                    ymin = max(0, min(ymin, height))
                    xmax = max(xmin, min(xmax, width))
                    ymax = max(ymin, min(ymax, height))
                    crop_image = image[ymin:ymax, xmin:xmax]
                    dst_file = osp.join(
                        dst_path, f"{label_to_count[label]}-{shape_type}.jpg"
                    )
                    cv2.imencode(".jpg", crop_image)[1].tofile(dst_file)
                # Update progress bar
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break
            # Hide the progress dialog after processing is done
            progress_dialog.close()

            # # Show success message
            save_path = osp.realpath(save_path)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(self.tr("Cropping completed successfully!"))
            msg_box.setInformativeText(
                self.tr(f"Cropped images have been saved to:\n{save_path}")
            )
            msg_box.setWindowTitle(self.tr("Success"))
            msg_box.exec_()

        except Exception as e:
            progress_dialog.close()
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while saving cropped image.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def label_manager(self):
        modify_label_dialog = LabelModifyDialog(parent=self, opacity=LABEL_OPACITY)
        result = modify_label_dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            self.load_file(self.filename)

    def hbb_to_obb(self):
        label_file_list = self.get_label_file_list()
        if len(label_file_list) == 0:
            return

        response = QtWidgets.QMessageBox.warning(
            self,
            self.tr("Current annotation will be changed"),
            self.tr(
                "You are about to start a transformation. Continue?"
            ),
            QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
        )
        if response != QtWidgets.QMessageBox.Ok:
            return

        progress_dialog = QProgressDialog(
            self.tr("Converting..."),
            self.tr("Cancel"),
            0,
            len(label_file_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet("""
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """)

        try:
            for i, label_file in enumerate(label_file_list):
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for j in range(len(data["shapes"])):
                    if data["shapes"][j]["shape_type"] == "rectangle":
                        data["shapes"][j]["shape_type"] = "rotation"
                        data["shapes"][j]["direction"] = 0
                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                # Update progress bar
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break
            # Hide the progress dialog after processing is done
            progress_dialog.close()
            # Reload the file after processing all label files
            self.load_file(self.filename)

        except Exception as e:
            progress_dialog.close()
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(self.tr("Error occurred while updating labels."))
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def obb_to_hbb(self):
        label_file_list = self.get_label_file_list()
        if len(label_file_list) == 0:
            return

        response = QtWidgets.QMessageBox.warning(
            self,
            self.tr("Current annotation will be changed"),
            self.tr(
                "You are about to start a transformation. Continue?"
            ),
            QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
        )
        if response != QtWidgets.QMessageBox.Ok:
            return

        progress_dialog = QProgressDialog(
            self.tr("Converting..."),
            self.tr("Cancel"),
            0,
            len(label_file_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet("""
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """)

        try:
            for i, label_file in enumerate(label_file_list):
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for j in range(len(data["shapes"])):
                    if data["shapes"][j]["shape_type"] == "rotation":
                        del data["shapes"][j]["direction"]
                        data["shapes"][j]["shape_type"] = "rectangle"
                        points = np.array(data["shapes"][j]["points"])
                        if len(points) != 4:
                            continue
                        xmin = int(np.min(points[:, 0]))
                        ymin = int(np.min(points[:, 1]))
                        xmax = int(np.max(points[:, 0]))
                        ymax = int(np.max(points[:, 1]))
                        data["shapes"][j]["points"] = [
                            [xmin, ymin],
                            [xmax, ymin],
                            [xmax, ymax],
                            [xmin, ymax],
                        ]
                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                # Update progress bar
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break
            # Hide the progress dialog after processing is done
            progress_dialog.close()
            # Reload the file after processing all label files
            self.load_file(self.filename)

        except Exception as e:
            progress_dialog.close()
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(self.tr("Error occurred while updating labels."))
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def polygon_to_hbb(self):
        label_file_list = self.get_label_file_list()
        if len(label_file_list) == 0:
            return

        response = QtWidgets.QMessageBox.warning(
            self,
            self.tr("Current annotation will be changed"),
            self.tr(
                "You are about to start a transformation. Continue?"
            ),
            QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
        )
        if response != QtWidgets.QMessageBox.Ok:
            return

        progress_dialog = QProgressDialog(
            self.tr("Converting..."),
            self.tr("Cancel"),
            0,
            len(label_file_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet("""
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """)

        try:
            for i, label_file in enumerate(label_file_list):
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for j in range(len(data["shapes"])):
                    if data["shapes"][j]["shape_type"] == "polygon":
                        data["shapes"][j]["shape_type"] = "rectangle"
                        points = np.array(data["shapes"][j]["points"])
                        if len(points) < 3:
                            continue
                        xmin = int(np.min(points[:, 0]))
                        ymin = int(np.min(points[:, 1]))
                        xmax = int(np.max(points[:, 0]))
                        ymax = int(np.max(points[:, 1]))
                        data["shapes"][j]["points"] = [
                            [xmin, ymin],
                            [xmax, ymin],
                            [xmax, ymax],
                            [xmin, ymax],
                        ]
                with open(label_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                # Update progress bar
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break
            # Hide the progress dialog after processing is done
            progress_dialog.close()
            # Reload the file after processing all label files
            self.load_file(self.filename)

        except Exception as e:
            progress_dialog.close()
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(self.tr("Error occurred while updating labels."))
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    # Help
    def documentation(self):
        url = (
            "https://github.com/CVHub520/X-AnyLabeling/tree/main/docs"  # NOQA
        )
        webbrowser.open(url)

    def contact(self):
        url = "https://github.com/CVHub520/X-AnyLabeling/tree/main/"  # NOQA
        webbrowser.open(url)

    def information(self):
        msg = "App name: {0} \nApp version: {1} \nDevice: {2} ".format(
            __appname__,
            __version__,
            __preferred_device__,
        )
        QMessageBox.information(self, "Information", msg)

    # General
    def toggle_drawing_sensitive(self, drawing=True):
        """Toggle drawing sensitive.

        In the middle of drawing, toggling between modes should be disabled.
        """
        self.actions.edit_mode.setEnabled(not drawing)
        self.actions.undo_last_point.setEnabled(drawing)
        self.actions.undo.setEnabled(not drawing)
        self.actions.delete.setEnabled(not drawing)

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
        else:
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

    def edit_label(self, item=None):
        if item and not isinstance(item, LabelListWidgetItem):
            raise TypeError("item must be LabelListWidgetItem type")

        if not self.canvas.editing():
            return
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
            kie_linking=shape.kie_linking
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

    def file_search_changed(self):
        self.import_image_folder(
            self.last_open_dir,
            pattern=self.file_search.text(),
            load=False,
        )

    def file_selection_changed(self):
        items = self.file_list_widget.selectedItems()
        if not items:
            return
        item = items[0]

        if not self.may_continue():
            return

        current_index = self.image_list.index(str(item.text()))
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
        # This function is called when the user changes the value in a QComboBox
        # It updates the shape's attributes and saves them immediately
        selected_option = combo.currentText()
        self.canvas.shapes[i].attributes[property] = selected_option
        self.save_attributes(self.canvas.shapes)

    def update_selected_options(self, selected_options):
        if not isinstance(selected_options, dict):
            # Handle the case where `selected_options`` is not valid
            return
        for row in range(len(selected_options)):
            category_label = None
            property_combo = None
            if self.grid_layout.itemAtPosition(row, 0):
                category_label = self.grid_layout.itemAtPosition(
                    row, 0
                ).widget()
            if self.grid_layout.itemAtPosition(row, 1):
                property_combo = self.grid_layout.itemAtPosition(
                    row, 1
                ).widget()
            if category_label and property_combo:
                category = category_label.text()
                if category in selected_options:
                    selected_option = selected_options[category]
                    index = property_combo.findText(selected_option)
                    if index >= 0:
                        property_combo.setCurrentIndex(index)
        return

    def update_attributes(self, i):
        selected_options = {}
        update_shape = self.canvas.shapes[i]
        update_category = update_shape.label
        update_attribute = update_shape.attributes
        current_attibute = self.attributes[update_category]
        # Clear the existing widgets from the QGridLayout
        self.grid_layout = QGridLayout()
        # Repopulate the QGridLayout with the updated data
        for row, (property, options) in enumerate(current_attibute.items()):
            property_label = QLabel(property)
            property_combo = QComboBox()
            property_combo.addItems(options)
            property_combo.currentIndexChanged.connect(
                lambda _, property=property, combo=property_combo: self.attribute_selection_changed(
                    i, property, combo
                )
            )
            self.grid_layout.addWidget(property_label, row, 0)
            self.grid_layout.addWidget(property_combo, row, 1)
            selected_options[property] = options[0]
        # Ensure the scroll_area updates its contents
        self.grid_layout_container = QWidget()
        self.grid_layout_container.setLayout(self.grid_layout)
        self.scroll_area.setWidget(self.grid_layout_container)
        self.scroll_area.setWidgetResizable(True)

        if update_attribute:
            for property, option in update_attribute.items():
                selected_options[property] = option
            self.update_selected_options(selected_options)
        else:
            update_shape.attributes = selected_options
            self.canvas.shapes[i] = update_shape
            self.save_attributes(self.canvas.shapes)

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
        is_mergeable = True
        for shape in self.canvas.selected_shapes:
            shape.selected = True
            if shape.shape_type != "rectangle":
                is_mergeable = False
            item = self.label_list.find_item_by_shape(shape)
            # NOTE: Handle the case when the shape is not found
            if item is not None:
                self.label_list.select_item(item)
                self.label_list.scroll_to_item(item)
        self._no_selection_slot = False
        n_selected = len(selected_shapes)
        self.actions.delete.setEnabled(n_selected)
        self.actions.duplicate.setEnabled(n_selected)
        self.actions.copy.setEnabled(n_selected)
        self.actions.edit.setEnabled(n_selected == 1)
        self.actions.union_selection.setEnabled(is_mergeable and n_selected > 1)
        self.set_text_editing(True)
        if self.attributes:
            # TODO: For future optimization(add parm to monitor selected_shape status)
            for i in range(len(self.canvas.shapes)):
                if self.canvas.shapes[i].selected:
                    self.update_attributes(i)
                    break

    def add_label(self, shape):
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

        # Add label to history if it is not a special label
        if shape.label not in [
            AutoLabelingMode.OBJECT,
            AutoLabelingMode.ADD,
            AutoLabelingMode.REMOVE,
        ]:
            self.label_dialog.add_label_history(shape.label)

        for action in self.actions.on_shapes_present:
            action.setEnabled(True)

        self._update_shape_color(shape)
        color = shape.fill_color.getRgb()[:3]
        label_list_item.setText("{}".format(html.escape(text)))
        label_list_item.setBackground(QtGui.QColor(*color, LABEL_OPACITY))
        self.update_combo_box()

    def shape_text_changed(self):
        description = self.shape_text_edit.toPlainText()
        if self.canvas.current is not None:
            self.canvas.current.description = description
        elif self.canvas.editing() and len(self.canvas.selected_shapes) == 1:
            self.canvas.selected_shapes[0].description = description
        else:
            self.other_data["image_description"] = description
        self.set_dirty()

    def _update_shape_color(self, shape):
        r, g, b = self._get_rgb_by_label(shape.label)
        shape.line_color = QtGui.QColor(r, g, b)
        shape.vertex_fill_color = QtGui.QColor(r, g, b)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(r, g, b, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(r, g, b, 155)

    def _get_rgb_by_label(self, label, skip_label_info=False):
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

    def load_shapes(self, shapes, replace=True):
        self._no_selection_slot = True
        for shape in shapes:
            self.add_label(shape)
        self.label_list.clearSelection()
        self._no_selection_slot = False
        self.canvas.load_shapes(shapes, replace=replace)

    def load_labels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            score = shape.get("score", None)
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            group_id = shape["group_id"]
            description = shape.get("description", "")
            difficult = shape.get("difficult", False)
            attributes = shape.get("attributes", {})
            direction = shape.get("direction", 0)
            kie_linking = shape.get("kie_linking", [])
            other_data = shape["other_data"]

            if label in self.hidden_cls or not points:
                # skip point-empty shape
                continue

            shape = Shape(
                label=label,
                score=score,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
                difficult=difficult,
                direction=direction,
                attributes=attributes,
                kie_linking=kie_linking,
            )
            for x, y in points:
                shape.add_point(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self.label_flags:
                for pattern, keys in self.label_flags.items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            if flags:
                shape.flags.update(flags)
            shape.other_data = other_data

            s.append(shape)
        self.update_combo_box()
        self.load_shapes(s)

    def load_flags(self, flags):
        self.flag_widget.clear()
        for key, flag in flags.items():
            item = QtWidgets.QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if flag else Qt.Unchecked)
            self.flag_widget.addItem(item)

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

    def save_labels(self, filename):
        label_file = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            info = {
                "label": s.label,
                "score": s.score,
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
            format_shape(item.shape())
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
        self.load_shapes(self._copied_shapes, replace=False)
        self.set_dirty()

    def copy_selected_shape(self):
        self._copied_shapes = [s.copy() for s in self.canvas.selected_shapes]
        self.actions.paste.setEnabled(len(self._copied_shapes) > 0)

    def combo_selection_changed(self, index):
        label = self.label_filter_combobox.combo_box.itemText(index)
        for item in self.label_list:
            if label in ["", item.shape().label]:
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
        self.canvas.set_shape_visible(shape, item.checkState() == Qt.Checked)

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
            if self._config["auto_use_last_label"] and last_label:
                text = last_label
            else:
                previous_text = self.label_dialog.edit.text()
                (
                    text,
                    flags,
                    group_id,
                    description,
                    difficult,
                    kie_linking,
                ) = self.label_dialog.pop_up(text)
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
            shape = self.canvas.set_last_label(text, flags)
            shape.group_id = group_id
            shape.description = description
            shape.label = text
            shape.difficult = difficult
            shape.kie_linking = kie_linking
            self.add_label(shape)
            self.actions.edit_mode.setEnabled(True)
            self.actions.undo_last_point.setEnabled(False)
            self.actions.undo.setEnabled(True)
            self.set_dirty()
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
        basename = osp.basename(str(self.filename))
        if shape_height > 0 and shape_width > 0:
            if num_images and self.filename in self.image_list:
                current_index = self.image_list.index(self.filename) + 1
                self.status(
                    str(self.tr("X: %d, Y: %d | H: %d, W: %d [%s: %d/%d]"))
                    % (
                        int(pos.x()),
                        int(pos.y()),
                        shape_height,
                        shape_width,
                        basename,
                        current_index,
                        num_images,
                    )
                )
            else:
                self.status(
                    str(self.tr("X: %d, Y: %d | H: %d, W: %d"))
                    % (int(pos.x()), int(pos.y()), shape_height, shape_width)
                )
        elif self.image_path:
            if num_images and self.filename in self.image_list:
                current_index = self.image_list.index(self.filename) + 1
                self.status(
                    str(self.tr("X: %d, Y: %d [%s: %d/%d]"))
                    % (
                        int(pos.x()),
                        int(pos.y()),
                        basename,
                        current_index,
                        num_images,
                    )
                )
            else:
                self.status(
                    str(self.tr("X: %d, Y: %d")) % (int(pos.x()), int(pos.y()))
                )

    def scroll_request(self, delta, orientation):
        units = -delta * 0.1  # natural scroll
        scroll_bar = self.scroll_bars[orientation]
        value = scroll_bar.value() + scroll_bar.singleStep() * units
        self.set_scroll(orientation, value)

    def set_scroll(self, orientation, value):
        self.scroll_bars[orientation].setValue(round(value))
        self.scroll_values[orientation][self.filename] = value

    def set_zoom(self, value):
        self.actions.fit_width.setChecked(False)
        self.actions.fit_window.setChecked(False)
        self.zoom_mode = self.MANUAL_ZOOM
        self.zoom_widget.setValue(value)
        self.zoom_values[self.filename] = (self.zoom_mode, value)

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
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.image_data),
            self.on_new_brightness_contrast,
            parent=self,
        )
        brightness, contrast = self.brightness_contrast_values.get(
            self.filename, (None, None)
        )
        if brightness is not None:
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        dialog.exec_()

        brightness = dialog.slider_brightness.value()
        contrast = dialog.slider_contrast.value()
        self.brightness_contrast_values[self.filename] = (brightness, contrast)

    def hide_selected_polygons(self):
        for index, item in enumerate(self.label_list):
            if item.shape().selected:
                item.setCheckState(Qt.Unchecked)
                self.selected_polygon_stack.append(index)
                self.label_list[index].shape().visible = False

    def show_hidden_polygons(self):
        if self.selected_polygon_stack:
            index = self.selected_polygon_stack.pop()
            item = self.label_list.item_at_index(index)
            item.setCheckState(Qt.Checked)
            self.label_list[index].shape().visible = True

    def get_next_files(self, filename, num_files):
        """Get the next files in the list."""
        if not self.image_list:
            return []
        filenames = []
        current_index = 0
        if filename is not None:
            try:
                current_index = self.image_list.index(filename)
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

        save_config(self._config)
        # For auto labeling, clear the previous marks
        # and inform the next files to be annotated
        self.clear_auto_labeling_marks()
        self.inform_next_files(filename)

        # Changing file_list_widget loads file
        if filename in self.image_list and (
            self.file_list_widget.currentRow()
            != self.image_list.index(filename)
        ):
            self.file_list_widget.setCurrentRow(
                self.image_list.index(filename)
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
        self.status(
            str(self.tr("Loading %s...")) % osp.basename(str(filename))
        )
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
                self.other_data.get("image_description", "")
            )
            self.shape_text_edit.textChanged.connect(self.shape_text_changed)
        else:
            self.image_data = LabelFile.load_image_file(filename)
            if self.image_data:
                self.image_path = filename
            self.label_file = None
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
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        self.canvas.load_pixmap(QtGui.QPixmap.fromImage(image))
        flags = {k: False for k in self.image_flags or []}
        if self.label_file:
            self.load_labels(self.label_file.shapes)
            if self.label_file.flags is not None:
                flags.update(self.label_file.flags)
        self.load_flags(flags)
        if self._config["keep_prev"] and self.no_shape():
            self.load_shapes(prev_shapes, replace=False)
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
        dialog = BrightnessContrastDialog(
            utils.img_data_to_pil(self.image_data),
            self.on_new_brightness_contrast,
            parent=self,
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
            dialog.slider_brightness.setValue(brightness)
        if contrast is not None:
            dialog.slider_contrast.setValue(contrast)
        self.brightness_contrast_values[self.filename] = (brightness, contrast)
        if brightness is not None or contrast is not None:
            dialog.on_new_value()
        self.paint_canvas()
        self.add_recent_file(self.filename)
        self.toggle_actions(True)
        self.canvas.setFocus()
        basename = osp.basename(str(filename))
        if self.image_list and filename in self.image_list:
            num_images = len(self.image_list)
            current_index = self.image_list.index(filename) + 1
            msg = str(self.tr("Loaded %s [%d/%d]")) % (
                basename,
                current_index,
                num_images,
            )
        else:
            msg = str(self.tr("Loaded %s")) % basename
        self.status(msg)
        return True

    # QT Overload
    def resizeEvent(self, _):
        if (
            self.canvas
            and not self.image.isNull()
            and self.zoom_mode != self.MANUAL_ZOOM
        ):
            self.adjust_scale()

    def paint_canvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoom_widget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjust_scale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoom_mode]()
        value = int(100 * value)
        self.zoom_widget.setValue(value)
        self.zoom_values[self.filename] = (self.zoom_mode, value)

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
        save_config(self._config)
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

    def open_prev_image(self, _value=False):
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self.open_labeled_image(-1, -1)
            return

        if not self.may_continue():
            return

        if len(self.image_list) <= 0:
            return

        if self.filename is None:
            return

        current_index = self.image_list.index(self.filename)
        if current_index - 1 >= 0:
            filename = self.image_list[current_index - 1]
            if filename:
                self.load_file(filename)

    def open_next_image(self, _value=False, load=True):
        if QtWidgets.QApplication.keyboardModifiers() == (
            Qt.ControlModifier | Qt.ShiftModifier
        ):
            self.open_labeled_image(self.file_list_widget.count(), 1, load)
            return

        if not self.may_continue():
            return

        if len(self.image_list) <= 0:
            return

        filename = None
        if self.filename is None:
            filename = self.image_list[0]
        else:
            current_index = self.image_list.index(self.filename)
            if current_index + 1 < len(self.image_list):
                filename = self.image_list[current_index + 1]
            else:
                filename = self.image_list[-1]
        self.filename = filename

        if self.filename and load:
            self.load_file(self.filename)

    def open_labeled_image(self, end_index, step, load=True):
        if not self.may_continue():
            return
        current_index = self.image_list.index(self.filename)
        for i in range(current_index + step, end_index, step):
            if self.file_list_widget.item(i).checkState() == Qt.Checked:
                self.filename = self.image_list[i]
                if self.filename and load:
                    self.load_file(self.filename)
                break

    # Uplaod
    def upload_image_flags_file(self):
        filter = "Image Flags Files (*.txt);;All Files (*)"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific flags file"),
            "",
            filter,
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # Each line in the file is an image-level flag
                self.image_flags = f.read().splitlines()
                self.load_flags({k: False for k in self.image_flags})
            self.flag_dock.show()
            # update and refresh the current canvas
            self.load_file(self.filename)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(self.tr("Success!"))
            msg_box.setInformativeText(self.tr("Uploading successfully!"))
            msg_box.setWindowTitle(self.tr("Information"))
            msg_box.exec_()
        except Exception as e:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while uploading flags.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def upload_label_flags_file(self):
        filter = "Label Flags Files (*.yaml);;All Files (*)"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific flags file"),
            "",
            filter,
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # Each line in the file is an flag-level flag
                self.label_flags = yaml.safe_load(f)
                for label in list(self.label_flags.keys()):
                    if not self.unique_label_list.find_items_by_label(label):
                        item = self.unique_label_list.create_item_from_label(label)
                        self.unique_label_list.addItem(item)
                        rgb = self._get_rgb_by_label(label)
                        self.unique_label_list.set_item_label(
                            item, label, rgb, LABEL_OPACITY
                        )
            self.label_dialog.upload_flags(self.label_flags)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(self.tr("Success!"))
            msg_box.setInformativeText(self.tr("Uploading successfully!"))
            msg_box.setWindowTitle(self.tr("Information"))
            msg_box.exec_()
        except Exception as e:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while uploading flags.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def upload_shape_attrs_file(self):
        filter = "Attribute Files (*.json);;All Files (*)"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific attributes file"),
            "",
            filter,
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.attributes = json.load(f)
                for label in list(self.attributes.keys()):
                    if not self.unique_label_list.find_items_by_label(label):
                        item = self.unique_label_list.create_item_from_label(label)
                        self.unique_label_list.addItem(item)
                        rgb = self._get_rgb_by_label(label)
                        self.unique_label_list.set_item_label(
                            item, label, rgb, LABEL_OPACITY
                        )
            self.shape_attributes.show()
            self.scroll_area.show()
            self.canvas.h_shape_is_hovered = False
            self.canvas.mode_changed.disconnect(self.set_edit_mode)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(self.tr("Success!"))
            msg_box.setInformativeText(self.tr("Uploading successfully!"))
            msg_box.setWindowTitle(self.tr("Information"))
            msg_box.exec_()
        except Exception as e:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while uploading attributes.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def upload_yolo_annotation(self, mode, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please load an image folder before proceeding!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        if mode == "pose":
            filter = "Classes Files (*.yaml);;All Files (*)"
            self.yaml_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("Select a specific yolo-pose config file"),
                "",
                filter,
            )
            if not self.yaml_file:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Warning"),
                    self.tr("Please select a specific config file!"),
                    QtWidgets.QMessageBox.Ok,
                )
                return
            labels = []
            with open(self.yaml_file, "r", encoding="utf-8") as f:
                import yaml

                data = yaml.safe_load(f)
                for class_name, keypoint_name in data["classes"].items():
                    labels.append(class_name)
                    labels.extend(keypoint_name)
            converter = LabelConverter(pose_cfg_file=self.yaml_file)
        elif mode in ["hbb", "obb", "seg"]:
            filter = "Classes Files (*.txt);;All Files (*)"
            self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("Select a specific classes file"),
                "",
                filter,
            )
            if not self.classes_file:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Warning"),
                    self.tr("Please select a specific classes file!"),
                    QtWidgets.QMessageBox.Ok,
                )
                return
            with open(self.classes_file, "r", encoding="utf-8") as f:
                labels = f.read().splitlines()
            converter = LabelConverter(classes_file=self.classes_file)

        # Initialize unique labels
        for label in labels:
            if not self.unique_label_list.find_items_by_label(label):
                item = self.unique_label_list.create_item_from_label(label)
                self.unique_label_list.addItem(item)
                rgb = self._get_rgb_by_label(label)
                self.unique_label_list.set_item_label(
                    item, label, rgb, LABEL_OPACITY
                )

        default_open_dir_path = dirpath if dirpath else "."
        if self.last_open_dir and osp.exists(self.last_open_dir):
            default_open_dir_path = self.last_open_dir
        else:
            default_open_dir_path = (
                osp.dirname(self.filename) if self.filename else "."
            )
        image_dir_path = osp.dirname(self.filename)
        label_dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Open Directory") % __appname__,
            default_open_dir_path,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )

        response = QtWidgets.QMessageBox.warning(
            self,
            self.tr("Current annotation will be lost"),
            self.tr(
                "You are going to upload new annotations to this task. Continue?"
            ),
            QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
        )
        if response != QtWidgets.QMessageBox.Ok:
            return

        image_file_list = os.listdir(image_dir_path)
        label_file_list = os.listdir(label_dir_path)
        output_dir_path = image_dir_path
        if self.output_dir:
            output_dir_path = self.output_dir

        progress_dialog = QProgressDialog(
            self.tr("Uploading..."),
            self.tr("Cancel"),
            0,
            len(image_file_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet(
            """
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """
        )

        try:
            for i, image_filename in enumerate(image_file_list):
                if image_filename.endswith(".json"):
                    continue
                label_filename = osp.splitext(image_filename)[0] + ".txt"
                data_filename = osp.splitext(image_filename)[0] + ".json"
                if label_filename not in label_file_list:
                    continue
                input_file = osp.join(label_dir_path, label_filename)
                output_file = osp.join(output_dir_path, data_filename)
                image_file = osp.join(image_dir_path, image_filename)
                if mode in ["hbb", "seg"]:
                    converter.yolo_to_custom(
                        input_file=input_file,
                        output_file=output_file,
                        image_file=image_file,
                        mode=mode,
                    )
                elif mode == "obb":
                    converter.yolo_obb_to_custom(
                        input_file=input_file,
                        output_file=output_file,
                        image_file=image_file,
                    )
                elif mode == "pose":
                    converter.yolo_pose_to_custom(
                        input_file=input_file,
                        output_file=output_file,
                        image_file=image_file,
                    )
                # Update progress bar
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break
            # Hide the progress dialog after processing is done
            progress_dialog.close()
            # update and refresh the current canvas
            self.load_file(self.filename)

        except Exception as e:
            progress_dialog.close()
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while uploading annotations.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def upload_voc_annotation(self, mode, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please load an image folder before proceeding!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        default_open_dir_path = dirpath if dirpath else "."
        if self.last_open_dir and osp.exists(self.last_open_dir):
            default_open_dir_path = self.last_open_dir
        else:
            default_open_dir_path = (
                osp.dirname(self.filename) if self.filename else "."
            )
        image_dir_path = osp.dirname(self.filename)
        label_dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Open Directory") % __appname__,
            default_open_dir_path,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )

        response = QtWidgets.QMessageBox.warning(
            self,
            self.tr("Current annotation will be lost"),
            self.tr(
                "You are going to upload new annotations to this task. Continue?"
            ),
            QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
        )

        if response != QtWidgets.QMessageBox.Ok:
            return

        converter = LabelConverter(classes_file=self.classes_file)
        image_file_list = os.listdir(image_dir_path)
        label_file_list = os.listdir(label_dir_path)
        output_dir_path = image_dir_path
        if self.output_dir:
            output_dir_path = self.output_dir

        progress_dialog = QProgressDialog(
            self.tr("Uploading..."),
            self.tr("Cancel"),
            0,
            len(image_file_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet(
            """
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """
        )

        try:
            for i, image_filename in enumerate(image_file_list):
                if image_filename.endswith(".json"):
                    continue
                label_filename = osp.splitext(image_filename)[0] + ".xml"
                data_filename = osp.splitext(image_filename)[0] + ".json"
                if label_filename not in label_file_list:
                    continue
                input_file = osp.join(label_dir_path, label_filename)
                output_file = osp.join(output_dir_path, data_filename)
                converter.voc_to_custom(
                    input_file=input_file,
                    output_file=output_file,
                    image_filename=image_filename,
                    mode=mode,
                )

                # Update progress bar
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break
            # Hide the progress dialog after processing is done
            progress_dialog.close()
            # update and refresh the current canvas
            self.load_file(self.filename)

        except Exception as e:
            progress_dialog.close()
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while uploading annotations.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def upload_coco_annotation(self, mode, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please load an image folder before proceeding!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        filter = "Attribute Files (*.json);;All Files (*)"
        input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a custom coco annotation file"),
            "",
            filter,
        )

        if (
            not input_file
            or QtWidgets.QMessageBox.warning(
                self,
                self.tr("Current annotation will be lost"),
                self.tr(
                    "You are going to upload new annotations to this task. Continue?"
                ),
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
            )
            != QtWidgets.QMessageBox.Ok
        ):
            return

        try:
            converter = LabelConverter()
            converter.coco_to_custom(
                input_file=input_file,
                image_path=osp.dirname(self.filename),
                mode=mode,
            )
            # update and refresh the current canvas
            self.load_file(self.filename)
        except Exception as e:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while uploading annotations.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def upload_dota_annotation(self, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please load an image folder before proceeding!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        default_open_dir_path = dirpath if dirpath else "."
        if self.last_open_dir and osp.exists(self.last_open_dir):
            default_open_dir_path = self.last_open_dir
        else:
            default_open_dir_path = (
                osp.dirname(self.filename) if self.filename else "."
            )
        image_dir_path = osp.dirname(self.filename)
        label_dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Open Directory") % __appname__,
            default_open_dir_path,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )

        response = QtWidgets.QMessageBox.warning(
            self,
            self.tr("Current annotation will be lost"),
            self.tr(
                "You are going to upload new annotations to this task. Continue?"
            ),
            QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
        )

        if response != QtWidgets.QMessageBox.Ok:
            return

        converter = LabelConverter()
        image_file_list = os.listdir(image_dir_path)
        label_file_list = os.listdir(label_dir_path)
        output_dir_path = image_dir_path
        if self.output_dir:
            output_dir_path = self.output_dir

        progress_dialog = QProgressDialog(
            self.tr("Exporting..."),
            self.tr("Cancel"),
            0,
            len(image_file_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet(
            """
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """
        )

        try:
            for i, image_filename in enumerate(image_file_list):
                if image_filename.endswith(".json"):
                    continue
                label_filename = osp.splitext(image_filename)[0] + ".txt"
                data_filename = osp.splitext(image_filename)[0] + ".json"
                if label_filename not in label_file_list:
                    continue
                input_file = osp.join(label_dir_path, label_filename)
                output_file = osp.join(output_dir_path, data_filename)
                image_file = osp.join(image_dir_path, image_filename)
                converter.dota_to_custom(
                    input_file=input_file,
                    output_file=output_file,
                    image_file=image_file,
                )
                # Update progress bar
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break
            # Hide the progress dialog after processing is done
            progress_dialog.close()
            # update and refresh the current canvas
            self.load_file(self.filename)

        except Exception as e:
            progress_dialog.close()
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while uploading annotations.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def upload_mask_annotation(self, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please load an image folder before proceeding!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        filter = "JSON Files (*.json);;All Files (*)"
        color_map_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific color_map file"),
            "",
            filter,
        )
        if not color_map_file:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please select a specific color_map file!"),
                QtWidgets.QMessageBox.Ok,
            )
            self._config["save_mode"] = "default"
            return
        with open(color_map_file, "r", encoding="utf-8") as f:
            mapping_table = json.load(f)
            classes = list(mapping_table["colors"].keys())
            for label in classes:
                if not self.unique_label_list.find_items_by_label(label):
                    item = self.unique_label_list.create_item_from_label(label)
                    self.unique_label_list.addItem(item)
                    rgb = self._get_rgb_by_label(label)
                    self.unique_label_list.set_item_label(
                        item, label, rgb, LABEL_OPACITY
                    )

        default_open_dir_path = dirpath if dirpath else "."
        if self.last_open_dir and osp.exists(self.last_open_dir):
            default_open_dir_path = self.last_open_dir
        else:
            default_open_dir_path = (
                osp.dirname(self.filename) if self.filename else "."
            )
        image_dir_path = osp.dirname(self.filename)
        label_dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - Open Directory") % __appname__,
            default_open_dir_path,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )

        response = QtWidgets.QMessageBox.warning(
            self,
            self.tr("Current annotation will be lost"),
            self.tr(
                "You are going to upload new annotations to this task. Continue?"
            ),
            QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
        )

        if response != QtWidgets.QMessageBox.Ok:
            return

        converter = LabelConverter()
        image_file_list = os.listdir(image_dir_path)
        label_file_list = os.listdir(label_dir_path)
        output_dir_path = image_dir_path
        if self.output_dir:
            output_dir_path = self.output_dir

        progress_dialog = QProgressDialog(
            self.tr("Uploading..."),
            self.tr("Cancel"),
            0,
            len(image_file_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet(
            """
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """
        )

        try:
            for i, image_filename in enumerate(image_file_list):
                if image_filename.endswith(".json"):
                    continue
                label_filename = osp.splitext(image_filename)[0] + ".png"
                data_filename = osp.splitext(image_filename)[0] + ".json"
                if label_filename not in label_file_list:
                    continue
                input_file = osp.join(label_dir_path, label_filename)
                output_file = osp.join(output_dir_path, data_filename)
                image_file = osp.join(image_dir_path, image_filename)
                converter.mask_to_custom(
                    input_file=input_file,
                    output_file=output_file,
                    image_file=image_file,
                    mapping_table=mapping_table,
                )

                # Update progress bar
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break
            # Hide the progress dialog after processing is done
            progress_dialog.close()
            # update and refresh the current canvas
            self.load_file(self.filename)

        except Exception as e:
            progress_dialog.close()
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while uploading annotations.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def upload_mot_annotation(self, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            return

        filter = "Classes Files (*.txt);;All Files (*)"
        self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific classes file"),
            "",
            filter,
        )
        if not self.classes_file:
            return
        with open(self.classes_file, "r", encoding="utf-8") as f:
            labels = f.read().splitlines()
            for label in labels:
                if not self.unique_label_list.find_items_by_label(label):
                    item = self.unique_label_list.create_item_from_label(label)
                    self.unique_label_list.addItem(item)
                    rgb = self._get_rgb_by_label(label)
                    self.unique_label_list.set_item_label(
                        item, label, rgb, LABEL_OPACITY
                    )

        filter = "Attribute Files (*.txt);;All Files (*)"
        input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a custom mot annotation file (gt.txt)"),
            "",
            filter,
        )

        if (
            not input_file
            or QtWidgets.QMessageBox.warning(
                self,
                self.tr("Current annotation will be lost"),
                self.tr(
                    "You are going to upload new annotations to this task. Continue?"
                ),
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
            )
            != QtWidgets.QMessageBox.Ok
        ):
            return

        try:
            image_dir_path = osp.dirname(self.filename)
            output_dir_path = image_dir_path
            if self.output_dir:
                output_dir_path = self.output_dir
            converter = LabelConverter(classes_file=self.classes_file)
            converter.mot_to_custom(
                input_file=input_file,
                output_path=output_dir_path,
                image_path=image_dir_path,
            )
            # update and refresh the current canvas
            self.load_file(self.filename)
        except Exception as e:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while uploading annotations.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def upload_odvg_annotation(self, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            return

        filter = "OD Files (*.json *.jsonl);;All Files (*)"
        input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific OD file"),
            "",
            filter,
        )

        if (
            not input_file
            or QtWidgets.QMessageBox.warning(
                self,
                self.tr("Current annotation will be lost"),
                self.tr(
                    "You are going to upload new annotations to this task. Continue?"
                ),
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
            )
            != QtWidgets.QMessageBox.Ok
        ):
            return

        try:
            output_dir_path = osp.dirname(self.filename)
            if self.output_dir:
                output_dir_path = self.output_dir
            converter = LabelConverter()
            converter.odvg_to_custom(
                input_file=input_file,
                output_path=output_dir_path,
            )
            # update and refresh the current canvas
            self.load_file(self.filename)
        except Exception as e:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while uploading annotations.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def upload_ppocr_annotation(self, mode, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please load an image folder before proceeding!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        if mode == "rec":
            filter = "Attribute Files (*.txt);;All Files (*)"
            input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("Select a custom annotation file (Label.txt)"),
                "",
                filter,
            )
        elif mode == "kie":
            filter = "Attribute Files (*.json);;All Files (*)"
            input_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("Select a custom annotation file (ppocr_kie.json)"),
                "",
                filter,
            )

        if (
            not input_file
            or QtWidgets.QMessageBox.warning(
                self,
                self.tr("Current annotation will be lost"),
                self.tr(
                    "You are going to upload new annotations to this task. Continue?"
                ),
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
            )
            != QtWidgets.QMessageBox.Ok
        ):
            return

        try:
            image_dir_path = osp.dirname(self.filename)
            output_dir_path = image_dir_path
            if self.output_dir:
                output_dir_path = self.output_dir
            converter = LabelConverter(classes_file=self.classes_file)
            converter.ppocr_to_custom(
                input_file=input_file,
                output_path=output_dir_path,
                image_path=image_dir_path,
                mode=mode,
            )
            # update and refresh the current canvas
            self.load_file(self.filename)
        except Exception as e:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while uploading annotations.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    # Export
    def export_yolo_annotation(self, mode, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please load an image folder before proceeding!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        if mode == "pose":
            filter = "Classes Files (*.yaml);;All Files (*)"
            self.yaml_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("Select a specific yolo-pose config file"),
                "",
                filter,
            )
            if not self.yaml_file:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Warning"),
                    self.tr("Please select a specific config file!"),
                    QtWidgets.QMessageBox.Ok,
                )
                return
            converter = LabelConverter(pose_cfg_file=self.yaml_file)
        elif mode in ["hbb", "obb", "seg"]:
            filter = "Classes Files (*.txt);;All Files (*)"
            self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("Select a specific classes file"),
                "",
                filter,
            )
            if not self.classes_file:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Warning"),
                    self.tr("Please select a specific classes file!"),
                    QtWidgets.QMessageBox.Ok,
                )
                return
            converter = LabelConverter(classes_file=self.classes_file)

        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle(self.tr("Options"))

        layout = QVBoxLayout()

        save_images_checkbox = QtWidgets.QCheckBox(self.tr("Save images?"))
        save_images_checkbox.setChecked(False)
        layout.addWidget(save_images_checkbox)

        skip_empty_files_checkbox = QtWidgets.QCheckBox(self.tr("Skip empty labels?"))
        skip_empty_files_checkbox.setChecked(False)
        layout.addWidget(skip_empty_files_checkbox)

        button_box = QtWidgets.QPushButton(self.tr("OK"))
        button_box.clicked.connect(dialog.accept)
        layout.addWidget(button_box)

        dialog.setLayout(layout)
        dialog.exec_()

        save_images = save_images_checkbox.isChecked()
        skip_empty_files = skip_empty_files_checkbox.isChecked()

        label_dir_path = osp.dirname(self.filename)
        if self.output_dir:
            label_dir_path = self.output_dir
        image_list = self.image_list
        if not image_list:
            image_list = [self.filename]
        save_path = osp.realpath(osp.join(label_dir_path, "..", "labels"))

        if osp.exists(save_path):
            response = QtWidgets.QMessageBox.warning(
                self,
                self.tr("Output Directory Exist!"),
                self.tr(
                    "You are going to export new annotations to this task. Continue?"
                ),
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
            )

            if response != QtWidgets.QMessageBox.Ok:
                return
            else:
                shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        progress_dialog = QProgressDialog(
            self.tr("Exporting..."),
            self.tr("Cancel"),
            0,
            len(image_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet(
            """
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """
        )

        try:
            for i, image_file in enumerate(image_list):
                image_file_name = osp.basename(image_file)
                label_file_name = osp.splitext(image_file_name)[0] + ".json"
                dst_file_name = osp.splitext(image_file_name)[0] + ".txt"
                dst_file = osp.join(save_path, dst_file_name)
                src_file = osp.join(label_dir_path, label_file_name)
                is_emtpy_file = converter.custom_to_yolo(
                    src_file, dst_file, mode, skip_empty_files
                )
                if save_images and not (skip_empty_files and is_emtpy_file):
                    image_dst = osp.join(save_path, image_file_name)
                    shutil.copy(image_file, image_dst)
                if skip_empty_files and is_emtpy_file and osp.exists(dst_file):
                    os.remove(dst_file)
                # Update progress bar
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break
            # Hide the progress dialog after processing is done
            progress_dialog.close()

            # # Show success message
            save_path = osp.realpath(save_path)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(self.tr("Exporting annotations successfully!"))
            msg_box.setInformativeText(
                self.tr(f"Results have been saved to:\n{save_path}")
            )
            msg_box.setWindowTitle(self.tr("Success"))
            msg_box.exec_()

        except Exception as e:
            progress_dialog.close()
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while exporting annotations.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def export_voc_annotation(self, mode, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please load an image folder before proceeding!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle(self.tr("Options"))

        layout = QVBoxLayout()

        save_images_checkbox = QtWidgets.QCheckBox(self.tr("Save images?"))
        save_images_checkbox.setChecked(False)
        layout.addWidget(save_images_checkbox)

        skip_empty_files_checkbox = QtWidgets.QCheckBox(self.tr("Skip empty labels?"))
        skip_empty_files_checkbox.setChecked(False)
        layout.addWidget(skip_empty_files_checkbox)

        button_box = QtWidgets.QPushButton(self.tr("OK"))
        button_box.clicked.connect(dialog.accept)
        layout.addWidget(button_box)

        dialog.setLayout(layout)
        dialog.exec_()

        save_images = save_images_checkbox.isChecked()
        skip_empty_files = skip_empty_files_checkbox.isChecked()

        label_dir_path = osp.dirname(self.filename)
        if self.output_dir:
            label_dir_path = self.output_dir
        image_list = self.image_list
        if not image_list:
            image_list = [self.filename]
        save_path = osp.realpath(osp.join(label_dir_path, "..", "Annotations"))

        if osp.exists(save_path):
            response = QtWidgets.QMessageBox.warning(
                self,
                self.tr("Output Directory Exist!"),
                self.tr(
                    "You are going to export new annotations to this task. Continue?"
                ),
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
            )

            if response != QtWidgets.QMessageBox.Ok:
                return
            else:
                shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        converter = LabelConverter()
        label_file_list = os.listdir(label_dir_path)

        progress_dialog = QProgressDialog(
            self.tr("Exporting..."),
            self.tr("Cancel"),
            0,
            len(label_file_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet(
            """
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """
        )

        try:
            for i, image_file in enumerate(image_list):
                image_file_name = osp.basename(image_file)
                label_file_name = osp.splitext(image_file_name)[0] + ".json"
                dst_file_name = osp.splitext(image_file_name)[0] + ".xml"
                src_file = osp.join(label_dir_path, label_file_name)
                dst_file = osp.join(save_path, dst_file_name)
                is_emtpy_file = converter.custom_to_voc(
                    image_file, src_file, dst_file, mode, skip_empty_files
                )
                if save_images and not (skip_empty_files and is_emtpy_file):
                    image_dst = osp.join(save_path, image_file_name)
                    shutil.copyfile(image_file, image_dst)
                if skip_empty_files and is_emtpy_file and osp.exists(dst_file):
                    os.remove(dst_file)
                # Update progress bar
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break
            # Hide the progress dialog after processing is done
            progress_dialog.close()

            # # Show success message
            save_path = osp.realpath(save_path)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(self.tr("Exporting annotations successfully!"))
            msg_box.setInformativeText(
                self.tr(f"Results have been saved to:\n{save_path}")
            )
            msg_box.setWindowTitle(self.tr("Success"))
            msg_box.exec_()

        except Exception as e:
            progress_dialog.close()
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while exporting annotations.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def export_coco_annotation(self, mode, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please load an image folder before proceeding!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        filter = "Classes Files (*.txt);;All Files (*)"
        self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific classes file"),
            "",
            filter,
        )
        if not self.classes_file:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please select a specific classes file!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        label_dir_path = osp.dirname(self.filename)
        if self.output_dir:
            label_dir_path = self.output_dir
        save_path = osp.realpath(osp.join(label_dir_path, "..", "annotations"))
        os.makedirs(save_path, exist_ok=True)
        converter = LabelConverter(classes_file=self.classes_file)

        try:
            converter.custom_to_coco(label_dir_path, save_path, mode)
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Success"),
                self.tr(
                    f"Annotation exported successfully!\n"
                    f"Check the results in: {save_path}."
                ),
                QtWidgets.QMessageBox.Ok,
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr(f"{e}"),
                QtWidgets.QMessageBox.Ok,
            )
            return

    def export_dota_annotation(self, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please load an image folder before proceeding!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        label_dir_path = osp.dirname(self.filename)
        if self.output_dir:
            label_dir_path = self.output_dir
        image_list = self.image_list
        if not image_list:
            image_list = [self.filename]
        save_path = osp.realpath(osp.join(label_dir_path, "..", "labelTxt"))
        os.makedirs(save_path, exist_ok=True)
        converter = LabelConverter(classes_file=self.classes_file)
        label_file_list = os.listdir(label_dir_path)

        progress_dialog = QProgressDialog(
            self.tr("Exporting..."),
            self.tr("Cancel"),
            0,
            len(image_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet(
            """
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """
        )

        try:
            for i, image_file in enumerate(image_list):
                image_file_name = osp.basename(image_file)
                label_file_name = osp.splitext(image_file_name)[0] + ".json"
                dst_file_name = osp.splitext(image_file_name)[0] + ".txt"
                dst_file = osp.join(save_path, dst_file_name)
                if label_file_name not in label_file_list:
                    pathlib.Path(dst_file).touch()
                else:
                    src_file = osp.join(label_dir_path, label_file_name)
                    converter.custom_to_dota(src_file, dst_file)
                # Update progress bar
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break
            # Hide the progress dialog after processing is done
            progress_dialog.close()

            # # Show success message
            save_path = osp.realpath(save_path)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(self.tr("Exporting annotations successfully!"))
            msg_box.setInformativeText(
                self.tr(f"Results have been saved to:\n{save_path}")
            )
            msg_box.setWindowTitle(self.tr("Success"))
            msg_box.exec_()

        except Exception as e:
            progress_dialog.close()
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while exporting annotations.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def export_mask_annotation(self, _value=False, dirpath=None):
        if not self.may_continue():
            return

        filter = "JSON Files (*.json);;All Files (*)"
        color_map_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Select a specific color_map file"),
            "",
            filter,
        )
        if not color_map_file:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please select a specific color_map file!"),
                QtWidgets.QMessageBox.Ok,
            )
            self._config["save_mode"] = "default"
            return

        with open(color_map_file, "r", encoding="utf-8") as f:
            mapping_table = json.load(f)

        label_dir_path = osp.dirname(self.filename)
        if self.output_dir:
            label_dir_path = self.output_dir
        selected_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select a directory to save the mask annotations"),
            label_dir_path,
            QtWidgets.QFileDialog.ShowDirsOnly
        )

        if not selected_dir:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("No directory selected! Operation cancelled."),
                QtWidgets.QMessageBox.Ok,
            )
            return
        save_path = osp.realpath(selected_dir)
        converter = LabelConverter(classes_file=self.classes_file)
        label_file_list = os.listdir(label_dir_path)

        progress_dialog = QProgressDialog(
            self.tr("Exporting..."),
            self.tr("Cancel"),
            0,
            len(label_file_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet(
            """
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """
        )

        try:
            for i, src_file_name in enumerate(label_file_list):
                if not src_file_name.endswith(".json"):
                    continue
                dst_file_name = osp.splitext(src_file_name)[0] + ".png"
                src_file = osp.join(label_dir_path, src_file_name)
                dst_file = osp.join(save_path, dst_file_name)
                converter.custom_to_mask(src_file, dst_file, mapping_table)
                # Update progress bar
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break
            # Hide the progress dialog after processing is done
            progress_dialog.close()

            # # Show success message
            save_path = osp.realpath(save_path)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(self.tr("Exporting annotations successfully!"))
            msg_box.setInformativeText(
                self.tr(f"Results have been saved to:\n{save_path}")
            )
            msg_box.setWindowTitle(self.tr("Success"))
            msg_box.exec_()

        except Exception as e:
            progress_dialog.close()
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(
                self.tr("Error occurred while exporting annotations.")
            )
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

    def export_mot_annotation(self, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please load an image folder before proceeding!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        if not self.classes_file:
            filter = "Classes Files (*.txt);;All Files (*)"
            self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("Select a specific classes file"),
                "",
                filter,
            )
            if not self.classes_file:
                return

        label_dir_path = osp.dirname(self.filename)
        if self.output_dir:
            label_dir_path = self.output_dir

        selected_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select a directory to save the mot annotations"),
            label_dir_path,
            QtWidgets.QFileDialog.ShowDirsOnly
        )

        if not selected_dir:
            return

        save_path = osp.realpath(selected_dir)
        os.makedirs(save_path, exist_ok=True)
        converter = LabelConverter(classes_file=self.classes_file)
        try:
            converter.custom_to_mot(label_dir_path, save_path)
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Success"),
                self.tr(
                    f"Annotation exported successfully!\n"
                    f"Check the results in: {save_path}."
                ),
                QtWidgets.QMessageBox.Ok,
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr(f"{e}"),
                QtWidgets.QMessageBox.Ok,
            )
            return

    def export_odvg_annotation(self, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please load an image folder before proceeding!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        if not self.classes_file:
            filter = "Classes Files (*.txt);;All Files (*)"
            self.classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("Select a specific classes file"),
                "",
                filter,
            )
            if not self.classes_file:
                return

        label_dir_path = osp.dirname(self.filename)
        if self.output_dir:
            label_dir_path = self.output_dir

        selected_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select a directory to save the odvg annotations"),
            label_dir_path,
            QtWidgets.QFileDialog.ShowDirsOnly
        )
        if not selected_dir:
            return

        save_path = osp.realpath(selected_dir)
        image_list = self.image_list
        if not image_list:
            image_list = [self.filename]
        os.makedirs(save_path, exist_ok=True)
        converter = LabelConverter(classes_file=self.classes_file)
        try:
            converter.custom_to_odvg(image_list, label_dir_path, save_path)
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Success"),
                self.tr(
                    f"Annotation exported successfully!\n"
                    f"Check the results in: {save_path}."
                ),
                QtWidgets.QMessageBox.Ok,
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr(f"{e}"),
                QtWidgets.QMessageBox.Ok,
            )
            return

    def export_pporc_annotation(self, mode, _value=False, dirpath=None):
        if not self.may_continue():
            return

        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please load an image folder before proceeding!"),
                QtWidgets.QMessageBox.Ok,
            )
            return

        converter = LabelConverter(classes_file=self.classes_file)
        label_dir_path = osp.dirname(self.filename)
        if self.output_dir:
            label_dir_path = self.output_dir
        image_list = self.image_list
        if not image_list:
            image_list = [self.filename]
        save_path = osp.realpath(osp.join(label_dir_path, "..", f"ppocr-{mode}"))

        if osp.exists(save_path):
            response = QtWidgets.QMessageBox.warning(
                self,
                self.tr("Output Directory Exist!"),
                self.tr(
                    "You are going to export new annotations to this task. Continue?"
                ),
                QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok,
            )

            if response != QtWidgets.QMessageBox.Ok:
                return
            else:
                shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)
        if mode == "rec":
            save_crop_img_path = osp.join(save_path, "crop_img")
            if osp.exists(save_crop_img_path):
                shutil.rmtree(save_crop_img_path)
            os.makedirs(save_crop_img_path, exist_ok=True)
        elif mode == "kie":
            total_class_set = set()
            class_list_file = osp.join(save_path, "class_list.txt")

        progress_dialog = QProgressDialog(
            self.tr("Exporting..."),
            self.tr("Cancel"),
            0,
            len(image_list),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet("""
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """)

        try:
            for i, image_file in enumerate(image_list):
                image_file_name = osp.basename(image_file)
                label_file_name = osp.splitext(image_file_name)[0] + ".json"
                label_file = osp.join(label_dir_path, label_file_name)
                if mode == "rec":
                    converter.custom_to_pporc(
                        image_file, label_file, save_path, mode
                    )
                elif mode == "kie":
                    class_set = converter.custom_to_pporc(
                        image_file, label_file, save_path, mode
                    )
                    total_class_set = total_class_set.union(class_set)
                # Update progress bar
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break   
            
            if mode == "kie":
                with open(class_list_file, 'w') as f:
                    for c in total_class_set:
                        f.writelines(f'{c.upper()}\n')

            # Hide the progress dialog after processing is done
            progress_dialog.close()

            # # Show success message
            save_path = osp.realpath(save_path)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(self.tr("Exporting annotations successfully!"))
            msg_box.setInformativeText(self.tr(f"Results have been saved to:\n{save_path}"))
            msg_box.setWindowTitle(self.tr("Success"))
            msg_box.exec_()

        except Exception as e:
            progress_dialog.close()
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(self.tr("Error occurred while exporting annotations."))
            error_dialog.setInformativeText(str(e))
            error_dialog.setWindowTitle(self.tr("Error"))
            error_dialog.exec_()

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
                self.image_list.index(current_filename)
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
        if self.filename.lower().endswith(".json"):
            label_file = self.filename
        else:
            label_file = osp.splitext(self.filename)[0] + ".json"

        return label_file

    def get_image_file(self):
        if not self.filename.lower().endswith(".json"):
            image_file = self.filename
        else:
            image_file = self.image_path

        return image_file

    def delete_file(self):
        mb = QtWidgets.QMessageBox
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
            logger.info("Label file is removed: %s", label_file)

            item = self.file_list_widget.currentItem()
            item.setCheckState(Qt.Unchecked)

            filename = self.filename
            self.reset_state()
            self.filename = filename
            if self.filename:
                self.load_file(self.filename)

    def delete_image_file(self):
        if len(self.image_list) <= 0:
            return

        mb = QtWidgets.QMessageBox
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
            logger.info("Image file is moved to: %s", osp.realpath(save_file))

            label_dir_path = osp.dirname(self.filename)
            if self.output_dir:
                label_dir_path = self.output_dir
            label_name = osp.splitext(image_name)[0] + ".json"
            label_file = osp.join(label_dir_path, label_name)
            if osp.exists(label_file):
                os.remove(label_file)
                logger.info("Label file is removed: %s", image_file)

            filename = None
            if self.filename is None:
                filename = self.image_list[0]
            else:
                current_index = self.image_list.index(self.filename)
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
        for item in self.label_list:
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)
        self._config["show_shapes"] = value

    def run_all_images(self):
        if len(self.image_list) <= 0:
            return

        if self.auto_labeling_widget.model_manager.loaded_model_config is None:
            self.auto_labeling_widget.model_manager.new_model_status.emit(
                self.tr("Model is not loaded. Choose a mode to continue.")
            )
            return

        marks_model_list = [
            "segment_anything",
            "sam_med2d",
            "sam_hq",
            "efficientvit_sam",
            "edge_sam",
        ]

        if (
            self.auto_labeling_widget.model_manager.loaded_model_config["type"]
            in marks_model_list
        ):
            self.auto_labeling_widget.model_manager.new_model_status.emit(
                self.tr(
                    "Invalid model type, please choose a valid model_type to run."
                )
            )
            return

        reply = QMessageBox.question(
            self,
            self.tr("Confirmation"),
            self.tr("Do you want to process all images?"),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.current_index = self.image_list.index(self.filename)
            self.image_index = self.current_index
            self.text_prompt = ""
            if self.auto_labeling_widget.model_manager.loaded_model_config[
                "type"
            ] in [
                "grounding_dino",
                "grounding_sam",
            ]:
                text_input_dialog = TextInputDialog(parent=self)
                self.text_prompt = text_input_dialog.get_input_text()
                if self.text_prompt:
                    self.show_progress_dialog_and_process()
            else:
                self.show_progress_dialog_and_process()

    def show_progress_dialog_and_process(self):
        self.cancel_processing = False
        progress_dialog = QProgressDialog(
            self.tr("Inferencing..."),
            self.tr("Cancel"),
            self.image_index,
            len(self.image_list),
            self,
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setStyleSheet(
            """
        QProgressDialog QProgressBar {
            border: 1px solid grey;
            border-radius: 5px;
            text-align: center;
        }
        QProgressDialog QProgressBar::chunk {
            background-color: orange;
        }
        """
        )
        progress_dialog.canceled.connect(self.cancel_operation)
        self.process_next_image(progress_dialog)

    def process_next_image(self, progress_dialog):
        if self.image_index < len(self.image_list):
            filename = self.image_list[self.image_index]
            self.filename = filename
            self.load_file(self.filename)
            if self.text_prompt:
                self.auto_labeling_widget.model_manager.predict_shapes(
                    self.image, self.filename, self.text_prompt
                )
            else:
                self.auto_labeling_widget.model_manager.predict_shapes(
                    self.image, self.filename
                )

            # Update the progress dialog
            progress_dialog.setValue(self.image_index)

            self.image_index += 1
            if not self.cancel_processing:
                delay_ms = 1
                QtCore.QTimer.singleShot(
                    delay_ms, lambda: self.process_next_image(progress_dialog)
                )
            else:
                self.cancel_operation()
        else:
            self.finish_processing(progress_dialog)

    def cancel_operation(self):
        self.cancel_processing = True

    def finish_processing(self, progress_dialog):
        self.filename = self.image_list[self.current_index]
        self.load_file(self.filename)
        del self.text_prompt
        del self.image_index
        del self.current_index
        progress_dialog.close()

    def remove_selected_point(self):
        self.canvas.remove_selected_point()
        self.canvas.update()
        if not self.canvas.h_hape.points:
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

    def open_video_file(self, _value=False):
        if not self.may_continue():
            return
        default_open_video_path = (
            osp.dirname(str(self.filename)) if self.filename else "."
        )
        supportedVideoFormats = (
            "*.asf *.avi *.m4v *.mkv *.mov *.mp4 *.mpeg *.mpg *.ts *.wmv"
        )
        source_video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("%s - Open Video file") % __appname__,
            default_open_video_path,
            supportedVideoFormats,
        )

        # Check if the path contains Chinese characters
        if utils.is_chinese(source_video_path):
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr(
                    "File path contains Chinese characters, invalid path!"
                ),
                QMessageBox.Ok,
            )
            return

        if osp.exists(source_video_path):
            target_dir_path = utils.extract_frames_from_video(
                self, source_video_path
            )
            self.import_image_folder(target_dir_path)

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
        for file in image_files:
            if file in self.image_list or not file.lower().endswith(
                tuple(extensions)
            ):
                continue
            label_file = osp.splitext(file)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = self.output_dir + "/" + label_file_without_path
            item = QtWidgets.QListWidgetItem(file)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
                label_file
            ):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.file_list_widget.addItem(item)

        if len(self.image_list) > 1:
            self.actions.open_next_image.setEnabled(True)
            self.actions.open_prev_image.setEnabled(True)

        self.open_next_image()

    def import_image_folder(self, dirpath, pattern=None, load=True):
        self.actions.open_next_image.setEnabled(True)
        self.actions.open_prev_image.setEnabled(True)

        if not self.may_continue() or not dirpath:
            return

        self.last_open_dir = dirpath
        self.filename = None
        self.file_list_widget.clear()
        for filename in self.scan_all_images(dirpath):
            if pattern and pattern not in filename:
                continue
            label_file = osp.splitext(filename)[0] + ".json"
            if self.output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = self.output_dir + "/" + label_file_without_path
            item = QtWidgets.QListWidgetItem(filename)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if QtCore.QFile.exists(label_file) and LabelFile.is_label_file(
                label_file
            ):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.file_list_widget.addItem(item)
        self.open_next_image(load=load)

    def scan_all_images(self, folder_path):
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        images = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relative_path = osp.join(root, file)
                    images.append(relative_path)
        images = natsort.os_sorted(images)
        return images

    def toggle_auto_labeling_widget(self):
        """Toggle auto labeling widget visibility."""
        if self.auto_labeling_widget.isVisible():
            self.auto_labeling_widget.hide()
            self.actions.run_all_images.setEnabled(False)
        else:
            self.auto_labeling_widget.show()
            self.actions.run_all_images.setEnabled(True)

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

    @staticmethod
    def find_most_similar_label(text, valid_labels):
        max_similarity = 0
        most_similar_label = valid_labels[0]

        for label in valid_labels:
            similarity = SequenceMatcher(None, text, label).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_label = label

        return most_similar_label

    def finish_auto_labeling_object(self):
        """Finish auto labeling object."""
        has_object, cache_label = False, None
        for shape in self.canvas.shapes:
            if shape.label == AutoLabelingMode.OBJECT:
                cache_label = shape.cache_label
                has_object = True
                break

        # If there is no object, do nothing
        if not has_object:
            return

        # Ask a label for the object
        text, flags, group_id, description, difficult = (
            "",
            {},
            None,
            None,
            False,
        )
        last_label = self.find_last_label()
        if self._config["auto_use_last_label"] and last_label:
            text = last_label
        elif cache_label is not None:
            text = cache_label
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
                group_id=None,
                description=None,
                difficult=False,
                kie_linking=[],
            )
            if not text:
                self.label_dialog.edit.setText(previous_text)
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
                    self.other_data.get("image_description", "")
                )
                self.shape_text_edit.textChanged.connect(
                    self.shape_text_changed
                )
            self.shape_text_edit.setDisabled(False)
        else:
            self.shape_text_edit.setDisabled(True)
            self.shape_text_label.setText(
                self.tr("Switch to Edit mode for description editing")
            )
            self.shape_text_edit.textChanged.disconnect()
            self.shape_text_edit.setPlainText("")
            self.shape_text_edit.textChanged.connect(self.shape_text_changed)
