from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import zipfile

from PyQt6.QtCore import QSize, QThread, QTimer, Qt
from PyQt6.QtGui import (
    QColor,
    QFontMetrics,
    QIcon,
    QIntValidator,
    QPainter,
    QPixmap,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSplitterHandle,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PIL import Image, ImageColor, ImageDraw, ImageFont

from anylabeling.views.labeling.ppocr.config import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_TITLE,
    LEFT_PANEL_WIDTH,
    MIN_DIALOG_HEIGHT,
    MIN_DIALOG_WIDTH,
    PPOCR_API_MODEL_ID,
    PPOCR_API_MODEL_LABEL,
    PPOCR_API_MODEL_SERVER_ID,
    PPOCR_COLOR_TEXT,
    PPOCR_FILE_TYPE_ALL,
    PPOCR_FILE_TYPE_IMAGE,
    PPOCR_FILE_TYPE_PDF,
    PPOCR_SORT_NEWEST,
    PPOCR_SORT_OLDEST,
    PPOCR_STATUS_ERROR,
    PPOCR_STATUS_PARSED,
    PPOCR_STATUS_PENDING,
)
from anylabeling.views.labeling.ppocr.data_manager import (
    PPOCRDataManager,
    PPOCRFileRecord,
)
from anylabeling.views.labeling.ppocr.dialogs import (
    PPOCRApiSettingsDialog,
    PPOCRFilterDialog,
)
from anylabeling.views.labeling.ppocr.editors import create_ppocr_block_editor
from anylabeling.views.labeling.ppocr.pipeline import (
    PPOCRParsingProgress,
    PPOCRPipeline,
    PPOCRPipelineWorker,
)
from anylabeling.views.labeling.ppocr.render import (
    FORMULA_BLOCK_LABELS,
    build_document_markdown_assets,
    document_page_count,
    extract_blocks,
    get_block_copy_text,
    get_document_copy_text,
    normalize_block_label,
)
from anylabeling.views.labeling.ppocr.style import (
    get_dialog_style,
    get_icon_button_style,
    get_model_combo_style,
    get_page_control_style,
    get_primary_button_style,
    get_preview_panel_style,
    get_result_header_style,
    get_secondary_button_style,
    get_source_file_info_style,
    get_sidebar_panel_style,
    get_sidebar_search_style,
    get_sidebar_tab_style,
)
from anylabeling.views.labeling.ppocr.widgets import (
    PPOCRBlockCard,
    PPOCRJsonViewer,
    PPOCRPreviewCanvas,
    PPOCRRecentsListWidget,
    PPOCRStatusBanner,
    resolve_qcolor,
)
from anylabeling.views.labeling.utils.qt import new_icon, new_icon_path
from anylabeling.views.labeling.utils.theme import get_theme
from anylabeling.views.labeling.widgets.popup import Popup

PPOCR_WORKSPACE_HANDLE_WIDTH = 16
PPOCR_WORKSPACE_MIN_RATIO = 0.2
PPOCR_WORKSPACE_GRIP_WIDTH = 2
PPOCR_WORKSPACE_GRIP_LENGTH = 34
PPOCR_WORKSPACE_GRIP_HOVER_LENGTH = 58


class PPOCRWorkspaceSplitterHandle(QSplitterHandle):
    def __init__(self, orientation: Qt.Orientation, parent: QSplitter) -> None:
        super().__init__(orientation, parent)
        if orientation == Qt.Orientation.Horizontal:
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.SplitVCursor)
        self._hovered = False
        self._grip = QFrame(self)
        self._grip.setObjectName("PPOCRWorkspaceSplitterGrip")
        self._apply_grip_style(False)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_grip_geometry()

    def enterEvent(self, event) -> None:
        self._hovered = True
        self._update_grip_geometry()
        self._apply_grip_style(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self._update_grip_geometry()
        self._apply_grip_style(False)
        super().leaveEvent(event)

    def _update_grip_geometry(self) -> None:
        target_length = (
            PPOCR_WORKSPACE_GRIP_HOVER_LENGTH
            if self._hovered
            else PPOCR_WORKSPACE_GRIP_LENGTH
        )
        if self.orientation() == Qt.Orientation.Horizontal:
            grip_length = min(
                target_length,
                max(20, self.height() - 24),
            )
            grip_x = (self.width() - PPOCR_WORKSPACE_GRIP_WIDTH) // 2
            grip_y = (self.height() - grip_length) // 2
            self._grip.setGeometry(
                grip_x,
                grip_y,
                PPOCR_WORKSPACE_GRIP_WIDTH,
                grip_length,
            )
            return
        grip_length = min(
            target_length,
            max(20, self.width() - 24),
        )
        grip_x = (self.width() - grip_length) // 2
        grip_y = (self.height() - PPOCR_WORKSPACE_GRIP_WIDTH) // 2
        self._grip.setGeometry(
            grip_x,
            grip_y,
            grip_length,
            PPOCR_WORKSPACE_GRIP_WIDTH,
        )

    def _apply_grip_style(self, hovered: bool) -> None:
        grip_color = "rgb(70, 88, 255)" if hovered else "rgb(204, 212, 229)"
        self._grip.setStyleSheet(
            "QFrame#PPOCRWorkspaceSplitterGrip {"
            f" background: {grip_color};"
            " border-radius: 1px;"
            "}"
        )


class PPOCRWorkspaceSplitter(QSplitter):
    def createHandle(self) -> QSplitterHandle:
        return PPOCRWorkspaceSplitterHandle(self.orientation(), self)


class PPOCRDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setModal(False)
        self.resize(*DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(MIN_DIALOG_WIDTH, MIN_DIALOG_HEIGHT)
        self.setAcceptDrops(True)
        self.setStyleSheet(get_dialog_style())

        parent_config = getattr(parent, "_config", {}) if parent else {}
        self.data_manager = PPOCRDataManager()
        self.pipeline = PPOCRPipeline(self.data_manager, parent_config)

        self.records: list[PPOCRFileRecord] = []
        self.record_map: dict[str, PPOCRFileRecord] = {}
        self.current_record: PPOCRFileRecord | None = None
        self.current_data = {}
        self.current_blocks = []
        self.current_page_paths = []
        self.current_page_no = 1
        self.current_fit_scale = 1.0
        self.hovered_block_key = ""
        self.selected_block_key = ""
        self.selected_block_source = ""
        self.editing_block_key = ""
        self.card_widgets = {}
        self.page_anchor_widgets = {}

        self.worker_thread = None
        self.worker = None
        self.cancel_event = threading.Event()
        self.queued_filenames = set()
        self.parsing_filenames = set()
        self.parsing_progress_map: dict[str, PPOCRParsingProgress] = {}
        self._pending_status_text = ""
        self._pending_status_animation_enabled = False
        self._pending_status_dot_phase = 0
        self._pending_status_cancellable = False
        self._pending_status_cancel_enabled = False
        self._pending_status_timer = QTimer(self)
        self._pending_status_timer.setInterval(420)
        self._pending_status_timer.timeout.connect(
            self._tick_pending_status_animation
        )
        self.active_sort_mode = PPOCR_SORT_NEWEST
        self.active_file_type = PPOCR_FILE_TYPE_ALL
        self.active_status = PPOCR_FILE_TYPE_ALL
        self.search_expanded = False
        self.filter_popover = None
        self.current_sidebar_tab = "recents"
        self.workspace_splitter: PPOCRWorkspaceSplitter | None = None
        self._updating_workspace_splitter = False
        self._prompting_api_settings = False

        self.init_ui()
        self.refresh_service_state()
        self.refresh_records()
        self.select_initial_record()

    def init_ui(self) -> None:
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        left_widget = self.build_left_panel()
        workspace_widget = self.build_workspace_panel()
        root_layout.addWidget(left_widget)
        root_layout.addWidget(workspace_widget, 1)

    def build_left_panel(self) -> QWidget:
        left_widget = QWidget()
        left_widget.setObjectName("PPOCRSidebar")
        left_widget.setFixedWidth(LEFT_PANEL_WIDTH)
        left_widget.setStyleSheet(get_sidebar_panel_style())
        layout = QVBoxLayout(left_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.new_parsing_button = QPushButton(self.tr("+ New Parsing"))
        self.new_parsing_button.setStyleSheet(get_secondary_button_style())
        self.new_parsing_button.setAutoDefault(False)
        self.new_parsing_button.setDefault(False)
        self.new_parsing_button.clicked.connect(self.on_new_parsing_clicked)
        layout.addWidget(self.new_parsing_button)

        divider = QFrame()
        divider.setObjectName("PPOCRSidebarDivider")
        divider.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(divider)

        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(8)

        self.recents_tab_button = QPushButton(self.tr("Recents"))
        self.recents_tab_button.setStyleSheet(get_sidebar_tab_style(True))
        self.recents_tab_button.clicked.connect(
            lambda: self.switch_sidebar_tab("recents")
        )
        toolbar_row.addWidget(self.recents_tab_button)

        self.favorites_tab_button = QPushButton(self.tr("Favorites"))
        self.favorites_tab_button.setStyleSheet(get_sidebar_tab_style(False))
        self.favorites_tab_button.clicked.connect(
            lambda: self.switch_sidebar_tab("favorites")
        )
        toolbar_row.addWidget(self.favorites_tab_button)
        toolbar_row.addStretch()

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(2)

        self.search_button = QPushButton()
        self.search_button.setIcon(new_icon("search", "svg"))
        self.search_button.setFixedSize(28, 28)
        self.search_button.setStyleSheet(get_icon_button_style())
        self.search_button.clicked.connect(self.toggle_search)
        actions_row.addWidget(self.search_button)

        self.filter_button = QPushButton()
        self.filter_button.setIcon(new_icon("import-export", "svg"))
        self.filter_button.setFixedSize(28, 28)
        self.filter_button.setStyleSheet(get_icon_button_style())
        self.filter_button.clicked.connect(self.open_filter_dialog)
        actions_row.addWidget(self.filter_button)
        toolbar_row.addLayout(actions_row)
        layout.addLayout(toolbar_row)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(self.tr("Search by Name"))
        self.search_input.setStyleSheet(get_sidebar_search_style())
        self.search_input.textChanged.connect(self.on_search_or_filter_changed)
        self.search_input.setVisible(False)
        layout.addWidget(self.search_input)

        self.recents_list = PPOCRRecentsListWidget()
        self.recents_list.fileSelected.connect(self.on_record_selected)
        self.recents_list.deleteRequested.connect(self.on_delete_requested)
        self.recents_list.favoriteToggled.connect(self.on_favorite_toggled)
        layout.addWidget(self.recents_list, 1)
        return left_widget

    def build_workspace_panel(self) -> QWidget:
        splitter = PPOCRWorkspaceSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("PPOCRWorkspaceSplitter")
        splitter.setHandleWidth(PPOCR_WORKSPACE_HANDLE_WIDTH)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("""
            QSplitter#PPOCRWorkspaceSplitter::handle {
                background: rgb(247, 249, 255);
                border: none;
            }
            """)

        middle_widget = self.build_preview_panel()
        right_widget = self.build_result_panel()
        splitter.addWidget(middle_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 700])
        splitter.splitterMoved.connect(self.on_workspace_splitter_moved)
        self.workspace_splitter = splitter
        QTimer.singleShot(0, self.enforce_workspace_splitter_ratio)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter)
        return container

    def on_workspace_splitter_moved(self, _pos: int, _index: int) -> None:
        self.enforce_workspace_splitter_ratio()

    def enforce_workspace_splitter_ratio(self) -> None:
        splitter = self.workspace_splitter
        if splitter is None or self._updating_workspace_splitter:
            return
        sizes = splitter.sizes()
        if len(sizes) < 2:
            return
        left_size, right_size = sizes[:2]
        total_size = left_size + right_size
        if total_size <= 0:
            return
        min_size = int(total_size * PPOCR_WORKSPACE_MIN_RATIO)
        if left_size >= min_size and right_size >= min_size:
            return
        adjusted_left = left_size
        adjusted_right = right_size
        if adjusted_left < min_size:
            adjusted_left = min_size
            adjusted_right = total_size - adjusted_left
        if adjusted_right < min_size:
            adjusted_right = min_size
            adjusted_left = total_size - adjusted_right
        self._updating_workspace_splitter = True
        splitter.setSizes([adjusted_left, adjusted_right])
        self._updating_workspace_splitter = False

    def build_preview_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("PPOCRPreviewPanel")
        panel.setStyleSheet(get_preview_panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 0, 0)
        layout.setSpacing(0)

        self.file_source_label = QLabel(self.tr("Source File"))
        self.file_source_label.setObjectName("PPOCRSourceFileTitle")
        self.file_source_label.setStyleSheet(get_source_file_info_style())
        layout.addWidget(self.file_source_label, 0, Qt.AlignmentFlag.AlignLeft)

        self.preview_frame = QFrame()
        self.preview_frame.setObjectName("PPOCRPreviewFrame")
        self.preview_frame.setStyleSheet(get_preview_panel_style())
        preview_layout = QVBoxLayout(self.preview_frame)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)

        self.file_info_frame = QFrame()
        self.file_info_frame.setObjectName("PPOCRSourceFileInfoFrame")
        self.file_info_frame.setStyleSheet(get_source_file_info_style())
        self.file_info_frame.setFixedHeight(44)
        file_row_layout = QHBoxLayout(self.file_info_frame)
        file_row_layout.setContentsMargins(11, 0, 11, 0)
        file_row_layout.setSpacing(0)
        self.file_icon_label = QLabel()
        self.file_icon_label.setFixedSize(16, 16)
        self.file_name_label = QLabel(self.tr("No file selected"))
        self.file_name_label.setObjectName("PPOCRSourceFileName")
        self.file_name_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.file_name_label.setCursor(Qt.CursorShape.IBeamCursor)
        self.file_size_label = QLabel("")
        self.file_size_label.setObjectName("PPOCRSourceFileSize")
        file_row_layout.addWidget(self.file_icon_label)
        file_row_layout.addSpacing(6)
        file_row_layout.addWidget(self.file_name_label)
        file_row_layout.addSpacing(18)
        file_row_layout.addWidget(self.file_size_label)
        file_row_layout.addStretch()
        preview_layout.addWidget(self.file_info_frame)

        self.file_info_divider = QFrame()
        self.file_info_divider.setObjectName("PPOCRSourceFileDivider")
        preview_layout.addWidget(self.file_info_divider)

        self.preview_content_frame = QFrame()
        self.preview_content_frame.setObjectName("PPOCRPreviewContentFrame")
        preview_content_layout = QVBoxLayout(self.preview_content_frame)
        preview_content_layout.setContentsMargins(0, 0, 0, 0)
        preview_content_layout.setSpacing(0)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setObjectName("PPOCRPreviewScrollArea")
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.preview_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.preview_canvas = PPOCRPreviewCanvas()
        self.preview_canvas.blockHovered.connect(self.on_canvas_block_hovered)
        self.preview_canvas.blockSelected.connect(
            self.on_canvas_block_selected
        )
        self.preview_canvas.blockCopyRequested.connect(
            self.on_canvas_block_copy_requested
        )
        self.preview_canvas.canvasCleared.connect(self.on_canvas_cleared)
        self.preview_canvas.scaleChanged.connect(self.on_canvas_scale_changed)
        self.preview_scroll.setWidget(self.preview_canvas)
        self.preview_scroll.horizontalScrollBar().rangeChanged.connect(
            lambda _min, _max: self.update_page_control_position()
        )
        preview_content_layout.addWidget(self.preview_scroll, 1)

        self.page_control = QWidget(self.preview_content_frame)
        self.page_control.setObjectName("PageControl")
        self.page_control.setFixedHeight(44)
        self.page_control.setFixedWidth(286)
        self.page_control.setStyleSheet(get_page_control_style())
        control_layout = QHBoxLayout(self.page_control)
        control_layout.setContentsMargins(10, 4, 10, 4)
        control_layout.setSpacing(4)

        self.prev_page_button = QPushButton()
        self.prev_page_button.setIcon(QIcon(new_icon("arrow-left", "svg")))
        self.prev_page_button.setIconSize(QSize(14, 14))
        self.prev_page_button.setFixedSize(26, 26)
        self.prev_page_button.setStyleSheet(get_icon_button_style())
        self.prev_page_button.setToolTip(self.tr("Previous Page"))
        self.prev_page_button.clicked.connect(self.goto_previous_page)

        self.page_input = QLineEdit("1")
        self.page_input.setFixedSize(48, 26)
        self.page_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_input.setValidator(QIntValidator(1, 9999, self))
        self.page_input.editingFinished.connect(self.on_page_input_finished)

        self.page_separator_label = QLabel("/")
        self.page_separator_label.setFixedWidth(10)
        self.page_separator_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.page_total_label = QLabel("1")
        self.page_total_label.setFixedWidth(20)
        self.page_total_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        self.next_page_button = QPushButton()
        self.next_page_button.setIcon(QIcon(new_icon("arrow-right", "svg")))
        self.next_page_button.setIconSize(QSize(14, 14))
        self.next_page_button.setFixedSize(26, 26)
        self.next_page_button.setStyleSheet(get_icon_button_style())
        self.next_page_button.setToolTip(self.tr("Next Page"))
        self.next_page_button.clicked.connect(self.goto_next_page)

        self.zoom_out_button = QPushButton()
        self.zoom_out_button.setIcon(QIcon(new_icon("zoom-out", "svg")))
        self.zoom_out_button.setIconSize(QSize(16, 16))
        self.zoom_out_button.setFixedSize(26, 26)
        self.zoom_out_button.setStyleSheet(get_icon_button_style())
        self.zoom_out_button.setToolTip(self.tr("Zoom Out"))
        self.zoom_out_button.clicked.connect(lambda: self.apply_zoom(-0.1))

        self.zoom_in_button = QPushButton()
        self.zoom_in_button.setIcon(QIcon(new_icon("zoom-in", "svg")))
        self.zoom_in_button.setIconSize(QSize(16, 16))
        self.zoom_in_button.setFixedSize(26, 26)
        self.zoom_in_button.setStyleSheet(get_icon_button_style())
        self.zoom_in_button.setToolTip(self.tr("Zoom In"))
        self.zoom_in_button.clicked.connect(lambda: self.apply_zoom(0.1))

        self.reset_zoom_button = QPushButton()
        self.reset_zoom_button.setIcon(QIcon(new_icon("refresh", "svg")))
        self.reset_zoom_button.setIconSize(QSize(14, 14))
        self.reset_zoom_button.setFixedSize(26, 26)
        self.reset_zoom_button.setStyleSheet(get_icon_button_style())
        self.reset_zoom_button.setToolTip(self.tr("Reset Zoom"))
        self.reset_zoom_button.clicked.connect(self.reset_zoom)

        page_divider_one = QFrame()
        page_divider_one.setObjectName("PPOCRPageControlDivider")
        page_divider_one.setFixedSize(1, 16)

        page_divider_two = QFrame()
        page_divider_two.setObjectName("PPOCRPageControlDivider")
        page_divider_two.setFixedSize(1, 16)

        control_layout.addWidget(self.prev_page_button)
        control_layout.addWidget(self.page_input)
        control_layout.addWidget(self.page_separator_label)
        control_layout.addWidget(self.page_total_label)
        control_layout.addWidget(self.next_page_button)
        control_layout.addWidget(page_divider_one)
        control_layout.addWidget(self.zoom_out_button)
        control_layout.addWidget(self.zoom_in_button)
        control_layout.addWidget(page_divider_two)
        control_layout.addWidget(self.reset_zoom_button)

        preview_layout.addWidget(self.preview_content_frame, 1)
        QTimer.singleShot(0, self.update_page_control_position)

        layout.addWidget(self.preview_frame, 1)
        return panel

    def build_result_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("PPOCRResultPanel")
        panel.setStyleSheet(get_result_header_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 24, 24, 0)
        layout.setSpacing(0)

        self.model_header_frame = QFrame()
        self.model_header_frame.setObjectName("PPOCRParsingModelHeader")
        header_height = (
            self.file_source_label.sizeHint().height()
            if hasattr(self, "file_source_label")
            else 44
        )
        self.model_header_frame.setFixedHeight(header_height)
        model_row = QHBoxLayout(self.model_header_frame)
        model_row.setContentsMargins(11, 0, 11, 0)
        model_row.setSpacing(24)
        self.model_label = QLabel(self.tr("Parsing model"))
        self.model_label.setObjectName("PPOCRParsingModelTitle")
        model_row.addWidget(self.model_label)
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("PPOCRParsingModelCombo")
        self.model_combo.setStyleSheet(get_model_combo_style())
        self.model_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.model_combo.setMinimumWidth(96)
        self.model_combo.setMaximumWidth(320)
        self.model_combo.setMaxVisibleItems(6)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_row.addWidget(self.model_combo)
        model_row.addStretch()
        layout.addWidget(
            self.model_header_frame,
            0,
            Qt.AlignmentFlag.AlignLeft,
        )

        self.result_info_frame = QFrame()
        self.result_info_frame.setObjectName("PPOCRResultInfoFrame")
        self.result_info_frame.setFixedHeight(44)
        result_info_layout = QHBoxLayout(self.result_info_frame)
        result_info_layout.setContentsMargins(11, 0, 11, 0)
        result_info_layout.setSpacing(8)

        self.document_parsing_button = QPushButton(self.tr("Document parsing"))
        self.document_parsing_button.setObjectName("PPOCRResultModeButton")
        self.document_parsing_button.setCheckable(True)
        self.document_parsing_button.setIcon(
            QIcon(new_icon("document", "svg"))
        )
        self.document_parsing_button.setIconSize(QSize(14, 14))
        self.document_parsing_button.clicked.connect(
            lambda: self.switch_result_view("document")
        )
        result_info_layout.addWidget(self.document_parsing_button)

        self.json_button = QPushButton(self.tr("JSON"))
        self.json_button.setObjectName("PPOCRResultModeButton")
        self.json_button.setCheckable(True)
        self.json_button.setIcon(QIcon(new_icon("json", "svg")))
        self.json_button.setIconSize(QSize(14, 14))
        self.json_button.clicked.connect(
            lambda: self.switch_result_view("json")
        )
        result_info_layout.addWidget(self.json_button)
        result_info_layout.addStretch()

        self.result_actions_frame = QFrame()
        self.result_actions_frame.setObjectName("PPOCRResultActionsFrame")
        self.result_actions_frame.setFixedHeight(26)
        actions_layout = QHBoxLayout(self.result_actions_frame)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(2)

        self.api_settings_button = QPushButton()
        self.api_settings_button.setObjectName("PPOCRResultActionButton")
        self.api_settings_button.setIcon(QIcon(new_icon("settings", "svg")))
        self.api_settings_button.setIconSize(QSize(14, 14))
        self.api_settings_button.setFixedSize(26, 26)
        self.api_settings_button.setToolTip(self.tr("Settings"))
        self.api_settings_button.clicked.connect(self.open_api_settings_dialog)
        actions_layout.addWidget(self.api_settings_button)

        self.reshape_button = QPushButton()
        self.reshape_button.setObjectName("PPOCRResultActionButton")
        self.reshape_button.setIcon(QIcon(new_icon("refresh", "svg")))
        self.reshape_button.setIconSize(QSize(14, 14))
        self.reshape_button.setFixedSize(26, 26)
        self.reshape_button.setToolTip(self.tr("Reparse"))
        self.reshape_button.clicked.connect(self.on_retry_requested)
        actions_layout.addWidget(self.reshape_button)

        self.copy_result_button = QPushButton()
        self.copy_result_button.setObjectName("PPOCRResultActionButton")
        self.copy_result_button.setIcon(QIcon(new_icon("copy", "svg")))
        self.copy_result_button.setIconSize(QSize(14, 14))
        self.copy_result_button.setFixedSize(26, 26)
        self.copy_result_button.setToolTip(self.tr("Copy"))
        self.copy_result_button.clicked.connect(self.on_copy_result_requested)
        actions_layout.addWidget(self.copy_result_button)

        self.download_result_button = QPushButton()
        self.download_result_button.setObjectName("PPOCRResultActionButton")
        self.download_result_button.setIcon(QIcon(new_icon("download", "svg")))
        self.download_result_button.setIconSize(QSize(14, 14))
        self.download_result_button.setFixedSize(26, 26)
        self.download_result_button.setToolTip(self.tr("Download"))
        self.download_result_button.clicked.connect(
            self.on_download_result_requested
        )
        actions_layout.addWidget(self.download_result_button)

        result_info_layout.addWidget(self.result_actions_frame)
        layout.addWidget(self.result_info_frame)

        self.result_info_divider = QFrame()
        self.result_info_divider.setObjectName("PPOCRResultDivider")
        layout.addWidget(self.result_info_divider)

        self.document_tab = QWidget()
        self.document_tab.setObjectName("PPOCRResultDocumentTab")
        document_layout = QVBoxLayout(self.document_tab)
        document_layout.setContentsMargins(0, 0, 0, 0)
        document_layout.setSpacing(0)

        self.document_stack = QStackedWidget()

        self.cards_scroll = QScrollArea()
        self.cards_scroll.setObjectName("PPOCRResultCardsScrollArea")
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.cards_container = QWidget()
        self.cards_container.setObjectName("PPOCRResultCardsContainer")
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(12, 12, 12, 12)
        self.cards_layout.setSpacing(4)
        self.cards_scroll.setWidget(self.cards_container)

        self.status_banner = PPOCRStatusBanner()
        self.status_banner.retryRequested.connect(self.on_retry_requested)
        self.status_banner.copyLogRequested.connect(
            self.copy_text_to_clipboard
        )
        self.status_banner.cancelRequested.connect(
            self.on_pending_cancel_requested
        )

        self.document_stack.addWidget(self.cards_scroll)
        self.document_stack.addWidget(self.status_banner)
        document_layout.addWidget(self.document_stack)

        self.json_viewer = PPOCRJsonViewer()
        self.json_viewer.setObjectName("PPOCRResultJsonViewer")
        self.result_content_stack = QStackedWidget()
        self.result_content_stack.setObjectName("PPOCRResultContentStack")
        self.result_content_stack.addWidget(self.document_tab)
        self.result_content_stack.addWidget(self.json_viewer)
        layout.addWidget(self.result_content_stack, 1)

        self.switch_result_view("document")
        self.reshape_button.setEnabled(False)
        self.copy_result_button.setEnabled(False)
        self.download_result_button.setEnabled(False)
        return panel

    def refresh_service_state(self) -> None:
        service_probe = self.pipeline.probe_service()
        selected_model = str(
            self.model_combo.currentData() or self.pipeline.pipeline_model
        ).strip()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem(
            PPOCR_API_MODEL_LABEL,
            userData=PPOCR_API_MODEL_ID,
        )
        seen_model_ids = {PPOCR_API_MODEL_ID}
        for model_info in service_probe.pipeline_models:
            model_id = str(model_info.model_id or "").strip()
            if not model_id or model_id in seen_model_ids:
                continue
            if model_id == PPOCR_API_MODEL_SERVER_ID:
                continue
            self.model_combo.addItem(
                model_info.display_name,
                userData=model_id,
            )
            seen_model_ids.add(model_id)
        selected_index = self.model_combo.findData(selected_model)
        if selected_index < 0:
            selected_index = 0
        self.model_combo.setCurrentIndex(selected_index)
        self.model_combo.setEnabled(True)
        self.pipeline.set_pipeline_model(
            str(self.model_combo.currentData() or PPOCR_API_MODEL_ID)
        )
        for index in range(self.model_combo.count()):
            self.model_combo.setItemData(
                index,
                int(Qt.AlignmentFlag.AlignCenter),
                Qt.ItemDataRole.TextAlignmentRole,
            )
        self.update_model_combo_width()
        self.model_combo.blockSignals(False)
        QTimer.singleShot(0, self.update_model_combo_width)
        QTimer.singleShot(60, self.update_model_combo_width)

    def open_api_settings_dialog(self, _checked: bool = False) -> None:
        if self._prompting_api_settings:
            return
        self._prompting_api_settings = True
        try:
            dialog = PPOCRApiSettingsDialog(
                self,
                current_api_url=self.pipeline.api_url,
                current_api_key=self.pipeline.api_key,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            api_url = dialog.get_api_url()
            api_key = dialog.get_api_key()
            self.pipeline.update_api_settings(
                api_url,
                api_key,
            )
            self.refresh_service_state()
        finally:
            self._prompting_api_settings = False

    def prompt_api_settings_if_needed(self) -> None:
        if self.pipeline.has_required_api_settings():
            return
        self.open_api_settings_dialog()

    def update_model_combo_width(self) -> None:
        self.model_combo.updateGeometry()
        view = self.model_combo.view()
        item_texts = [
            self.model_combo.itemText(index).strip()
            for index in range(self.model_combo.count())
        ]
        current_text = self.model_combo.currentText().strip()
        candidate_texts = [
            text for text in item_texts + [current_text] if text
        ] or ["Model"]
        text_metrics = QFontMetrics(self.model_combo.font())
        content_width = max(
            text_metrics.horizontalAdvance(text) for text in candidate_texts
        )
        target_width = max(
            content_width + 28,
            self.model_combo.minimumSizeHint().width(),
            self.model_combo.sizeHint().width(),
        )
        if view is not None:
            column_width = view.sizeHintForColumn(0)
            if column_width > 0:
                target_width = max(target_width, column_width + 28)
        layout = self.model_header_frame.layout()
        if layout is not None:
            margins = layout.contentsMargins()
            host_width = self.model_header_frame.width()
            parent_widget = self.model_header_frame.parentWidget()
            if parent_widget is not None and parent_widget.width() > 0:
                host_width = parent_widget.width()
            available_width = (
                host_width
                - margins.left()
                - margins.right()
                - self.model_label.sizeHint().width()
                - layout.spacing()
                - 12
            )
            if available_width > 0:
                target_width = min(target_width, available_width)
        target_width = max(96, min(520, target_width))
        self.model_combo.setFixedWidth(target_width)
        if view is not None:
            view.setMinimumWidth(target_width)
            view.setTextElideMode(Qt.TextElideMode.ElideNone)

    def refresh_records(self, preferred_filename: str = "") -> None:
        records = self.data_manager.query_records(
            search_text=self.search_input.text().strip(),
            sort_mode=self.active_sort_mode or PPOCR_SORT_NEWEST,
            file_type=self.active_file_type or PPOCR_FILE_TYPE_ALL,
            status=self.active_status or PPOCR_FILE_TYPE_ALL,
        )
        favorites = self.data_manager.list_favorites()
        for record in records:
            record.favorite = record.filename in favorites
        if self.current_sidebar_tab == "favorites":
            records = [record for record in records if record.favorite]
        self.records = records
        self.record_map = {record.filename: record for record in self.records}
        selected_name = preferred_filename or (
            self.current_record.filename if self.current_record else ""
        )
        if selected_name not in self.record_map and self.records:
            selected_name = self.records[0].filename
        if not self.records:
            selected_name = ""
        self.recents_list.render_records(self.records, selected_name)
        if selected_name:
            self.current_record = self.record_map[selected_name]
        else:
            self.current_record = None

    def select_initial_record(self) -> None:
        if self.current_record is not None:
            self.on_record_selected(self.current_record.filename)
            return
        self.show_empty_state()

    def show_empty_state(self) -> None:
        self.current_record = None
        self.current_data = {}
        self.current_blocks = []
        self.current_page_paths = []
        self.current_page_no = 1
        self.hovered_block_key = ""
        self.selected_block_key = ""
        self.selected_block_source = ""
        self.editing_block_key = ""
        self.file_icon_label.clear()
        self.file_name_label.setText(self.tr("No file selected"))
        self.file_size_label.setText("")
        self.preview_canvas.set_page(QPixmap(), [])
        self.update_page_control_state()
        self.status_banner.set_pending_state(
            self.tr("Use New Parsing to import images or PDFs.")
        )
        self.document_stack.setCurrentWidget(self.status_banner)
        self.json_viewer.setPlainText("")
        self.reshape_button.setEnabled(False)
        self.copy_result_button.setEnabled(False)
        self.download_result_button.setEnabled(False)

    def on_search_or_filter_changed(self) -> None:
        current_name = (
            self.current_record.filename if self.current_record else ""
        )
        self.refresh_records(current_name)
        if self.current_record is not None:
            self.on_record_selected(self.current_record.filename)
        else:
            self.show_empty_state()

    def toggle_search(self) -> None:
        self.search_expanded = not self.search_expanded
        self.search_input.setVisible(self.search_expanded)
        if self.search_expanded:
            self.search_input.setFocus()
            self.search_input.selectAll()
            return
        if self.search_input.text():
            self.search_input.clear()

    def switch_sidebar_tab(self, tab_name: str) -> None:
        if tab_name == self.current_sidebar_tab:
            return
        self.current_sidebar_tab = tab_name
        self.recents_tab_button.setStyleSheet(
            get_sidebar_tab_style(tab_name == "recents")
        )
        self.favorites_tab_button.setStyleSheet(
            get_sidebar_tab_style(tab_name == "favorites")
        )
        current_name = (
            self.current_record.filename if self.current_record else ""
        )
        self.refresh_records(current_name)
        if self.current_record is not None:
            self.on_record_selected(self.current_record.filename)
        else:
            self.show_empty_state()

    def open_filter_dialog(self) -> None:
        if self.filter_popover is not None and self.filter_popover.isVisible():
            self.filter_popover.close()
            self.filter_popover = None
            return
        self.filter_popover = PPOCRFilterDialog(
            self.active_sort_mode,
            self.active_file_type,
            self.active_status,
            self,
        )
        self.filter_popover.filtersConfirmed.connect(self.apply_filters)
        self.filter_popover.adjustSize()
        self.filter_popover.set_anchor_offset(
            self.filter_button.width() // 2
            + self.filter_popover.panel_margin()
        )
        button_pos = self.filter_button.mapToGlobal(
            self.filter_button.rect().bottomLeft()
        )
        self.filter_popover.move(
            button_pos.x() - self.filter_popover.panel_margin(),
            button_pos.y() + 8,
        )
        self.filter_popover.show()

    def apply_filters(
        self,
        sort_mode: str,
        file_type: str,
        status: str,
    ) -> None:
        self.active_sort_mode = sort_mode
        self.active_file_type = file_type
        self.active_status = status
        self.on_search_or_filter_changed()

    def on_favorite_toggled(self, filename: str, favorite: bool) -> None:
        self.data_manager.set_favorite(filename, favorite)
        current_name = (
            self.current_record.filename if self.current_record else ""
        )
        self.refresh_records(current_name)
        if (
            self.current_record is not None
            and self.current_record.filename in self.record_map
        ):
            self.current_record = self.record_map[self.current_record.filename]
        elif self.records:
            self.current_record = self.records[0]
        else:
            self.current_record = None
        if self.current_record is not None:
            self.on_record_selected(self.current_record.filename)
        else:
            self.show_empty_state()

    def on_new_parsing_clicked(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            self.tr("New Parsing"),
            "",
            self.tr(
                "PaddleOCR Inputs (*.pdf *.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp *.cif)"
            ),
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if file_paths:
            self.import_source_paths(file_paths)

    def import_source_paths(self, file_paths: list[str]) -> None:
        imported_records, errors = self.data_manager.import_files(file_paths)
        if errors:
            QMessageBox.warning(
                self,
                self.tr("Import Failed"),
                "\n".join(errors),
            )
        if not imported_records:
            self.refresh_records()
            if self.current_record is None:
                self.show_empty_state()
            return
        first_name = imported_records[0].filename
        self.refresh_records(first_name)
        self.on_record_selected(first_name)
        self.start_parsing(imported_records)

    def on_record_selected(self, filename: str) -> None:
        record = self.record_map.get(filename)
        if record is None:
            return
        self.current_record = record
        self.recents_list.set_selected_name(filename)
        self.hovered_block_key = ""
        self.selected_block_key = ""
        self.selected_block_source = ""
        self.editing_block_key = ""
        self.current_page_no = 1
        self.load_current_record()
        if (
            record.status == PPOCR_STATUS_PENDING
            and record.filename not in self.parsing_filenames
            and record.filename not in self.queued_filenames
        ):
            self.start_parsing([record])

    def load_current_record(self) -> None:
        if self.current_record is None:
            self.show_empty_state()
            return
        record = self.current_record
        self.current_data = self.data_manager.load_record_data(record)
        self.current_blocks = extract_blocks(self.current_data)
        self.current_page_paths = []

        if record.file_type == PPOCR_FILE_TYPE_IMAGE:
            icon_name, icon_ext = "image", "svg"
        elif record.file_type == PPOCR_FILE_TYPE_PDF:
            icon_name, icon_ext = "pdf", "svg"
        else:
            icon_name, icon_ext = "file", "png"
        self.file_icon_label.setPixmap(
            QPixmap(new_icon_path(icon_name, icon_ext)).scaled(
                16,
                16,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.file_name_label.setText(record.filename)
        self.file_size_label.setText(self.format_size(record.size_bytes))

        try:
            self.current_page_paths = self.data_manager.get_preview_pages(
                record
            )
        except Exception as exc:
            self.current_page_paths = []
            if record.status == PPOCR_STATUS_PENDING:
                self.data_manager.mark_error(record, str(exc))
                self.refresh_records(record.filename)
                self.current_record = self.record_map.get(
                    record.filename, record
                )
                self.current_data = self.data_manager.load_record_data(
                    self.current_record
                )
                self.current_blocks = extract_blocks(self.current_data)

        self.current_page_no = max(
            1,
            min(
                self.current_page_no,
                max(1, len(self.current_page_paths) or record.page_count),
            ),
        )
        self.render_preview_page()
        self.render_right_panel()
        self.update_page_control_state()

    def render_preview_page(self) -> None:
        if not self.current_page_paths:
            self.preview_canvas.set_page(QPixmap(), [])
            self.current_fit_scale = 1.0
            self.reset_zoom_button.setEnabled(False)
            return
        page_path = self.current_page_paths[self.current_page_no - 1]
        pixmap = QPixmap(str(page_path))
        page_blocks = [
            block
            for block in self.current_blocks
            if block.page_no == self.current_page_no
        ]
        self.preview_canvas.set_page(pixmap, page_blocks)
        self.current_fit_scale = self.preview_canvas.set_fit_width(
            self.preview_scroll.viewport().width()
        )
        if self.hovered_block_key:
            hovered_block = self.block_map().get(self.hovered_block_key)
            if hovered_block and hovered_block.page_no == self.current_page_no:
                self.preview_canvas.set_hovered_blocks(
                    self.linked_block_keys(self.hovered_block_key)
                )
        if self.selected_block_key:
            current_block = self.block_map().get(self.selected_block_key)
            if current_block and current_block.page_no == self.current_page_no:
                self.preview_canvas.set_selected_block(
                    self.selected_block_key,
                    selected_locally=self.selected_block_source == "canvas",
                )
        self.reset_zoom_button.setEnabled(False)

    def render_right_panel(self) -> None:
        self.cleanup_stale_parsing_state()
        if self.sync_current_record_status_from_disk():
            return
        self.json_viewer.setPlainText(self.current_json_text())
        self.reshape_button.setEnabled(self.current_record is not None)
        self.copy_result_button.setEnabled(
            bool(self.json_viewer.toPlainText().strip())
        )
        self.download_result_button.setEnabled(self.current_record is not None)
        if self.current_record is None:
            self._stop_pending_status_animation()
            self.document_stack.setCurrentWidget(self.status_banner)
            return
        if self.current_record.status == PPOCR_STATUS_ERROR:
            self._stop_pending_status_animation()
            self.status_banner.set_error_state(
                self.current_record.error_message or self.tr("Parsing failed.")
            )
            self.document_stack.setCurrentWidget(self.status_banner)
            return
        if (
            self.current_record.status == PPOCR_STATUS_PENDING
            or self.current_record.filename in self.parsing_filenames
            or self.current_record.filename in self.queued_filenames
        ):
            batch_active = bool(
                self.worker_thread is not None
                and self.worker_thread.isRunning()
            )
            is_cancelling = batch_active and self.cancel_event.is_set()
            can_cancel = batch_active
            cancel_enabled = can_cancel and not is_cancelling
            if self.current_record.filename in self.parsing_filenames:
                if is_cancelling:
                    text = self.tr(
                        "Cancelling parsing task. Waiting for current step to finish."
                    )
                    animate = True
                else:
                    progress = self.parsing_progress_map.get(
                        self.current_record.filename
                    )
                    if progress is not None:
                        text = self.tr(
                            "File {0}/{1}: {2}  Page {3}/{4}"
                        ).format(
                            progress.index,
                            progress.total,
                            progress.filename,
                            progress.page_no,
                            progress.page_total,
                        )
                    else:
                        text = self.tr("Parsing is running for this file.")
                    animate = True
            elif self.current_record.filename in self.queued_filenames:
                if is_cancelling:
                    text = self.tr(
                        "Cancelling parsing task. Waiting for current step to finish."
                    )
                    animate = True
                else:
                    text = self.tr("Waiting in the parsing queue.")
                    animate = False
            else:
                text = self.tr("Waiting to parse this file.")
                animate = False
            self._set_pending_status_banner(
                text,
                animate=animate,
                cancellable=can_cancel,
                cancel_enabled=cancel_enabled,
            )
            self.document_stack.setCurrentWidget(self.status_banner)
            return
        self._stop_pending_status_animation()
        self.rebuild_cards()
        self.document_stack.setCurrentWidget(self.cards_scroll)

    def cleanup_stale_parsing_state(self) -> None:
        batch_active = bool(
            self.worker_thread is not None and self.worker_thread.isRunning()
        )
        if batch_active:
            return
        if (
            not self.queued_filenames
            and not self.parsing_filenames
            and not self.parsing_progress_map
        ):
            return
        self.queued_filenames = set()
        self.parsing_filenames = set()
        self.parsing_progress_map.clear()

    def sync_current_record_status_from_disk(self) -> bool:
        if self.current_record is None:
            return False
        if self.current_record.status != PPOCR_STATUS_PENDING:
            return False
        if self.queued_filenames or self.parsing_filenames:
            return False
        data = self.data_manager.load_record_data(self.current_record)
        if not isinstance(data, dict):
            return False
        meta = data.get("_ppocr_meta") or {}
        disk_status = str(meta.get("status") or "").strip()
        if disk_status not in {PPOCR_STATUS_PARSED, PPOCR_STATUS_ERROR}:
            return False
        self.current_record.status = disk_status
        self.current_record.error_message = str(
            meta.get("error_message") or ""
        )
        self.load_current_record()
        return True

    def _set_pending_status_banner(
        self,
        text: str,
        animate: bool,
        cancellable: bool = False,
        cancel_enabled: bool = True,
    ) -> None:
        self._pending_status_text = (text or "").strip()
        self._pending_status_animation_enabled = animate
        self._pending_status_cancellable = cancellable
        self._pending_status_cancel_enabled = cancel_enabled
        if animate:
            if not self._pending_status_timer.isActive():
                self._pending_status_timer.start()
        else:
            self._pending_status_timer.stop()
            self._pending_status_dot_phase = 0
        self._apply_pending_status_text()

    def _apply_pending_status_text(self) -> None:
        text = self._pending_status_text or self.tr("Parsing")
        if self._pending_status_animation_enabled:
            dot_count = self._pending_status_dot_phase % 4
            suffix = "" if dot_count == 0 else f" {'.' * dot_count}"
            display_text = f"{text}{suffix}"
        else:
            display_text = text
        self.status_banner.set_pending_state(
            display_text,
            cancellable=self._pending_status_cancellable,
            cancel_enabled=self._pending_status_cancel_enabled,
        )

    def _tick_pending_status_animation(self) -> None:
        batch_active = bool(
            self.worker_thread is not None and self.worker_thread.isRunning()
        )
        if (
            self.current_record is None
            or not batch_active
            or (
                self.current_record.filename not in self.parsing_filenames
                and self.current_record.filename not in self.queued_filenames
            )
            or not self._pending_status_animation_enabled
            or self.document_stack.currentWidget() is not self.status_banner
        ):
            self._stop_pending_status_animation()
            return
        self._pending_status_dot_phase = (
            self._pending_status_dot_phase + 1
        ) % 4
        self._apply_pending_status_text()

    def _stop_pending_status_animation(self) -> None:
        self._pending_status_animation_enabled = False
        self._pending_status_dot_phase = 0
        self._pending_status_cancellable = False
        self._pending_status_cancel_enabled = False
        self._pending_status_timer.stop()

    def on_pending_cancel_requested(self) -> None:
        if self.cancel_event.is_set():
            return
        if self.worker_thread is None or not self.worker_thread.isRunning():
            return
        self.cancel_event.set()
        self.render_right_panel()

    def switch_result_view(self, view_mode: str) -> None:
        show_document = view_mode != "json"
        self.document_parsing_button.setChecked(show_document)
        self.json_button.setChecked(not show_document)
        self.update_result_mode_button_icons(show_document)
        if show_document:
            self.result_content_stack.setCurrentWidget(self.document_tab)
            return
        self.result_content_stack.setCurrentWidget(self.json_viewer)

    def _build_result_mode_icon(
        self,
        icon_name: str,
        icon_ext: str,
        color: QColor,
    ) -> QIcon:
        base_icon = QIcon(new_icon(icon_name, icon_ext))
        source_pixmap = base_icon.pixmap(QSize(14, 14))
        if source_pixmap.isNull():
            return base_icon
        source_pixmap = source_pixmap.scaled(
            14,
            14,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        tinted_pixmap = QPixmap(source_pixmap.size())
        tinted_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(tinted_pixmap)
        painter.drawPixmap(0, 0, source_pixmap)
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_SourceIn
        )
        painter.fillRect(tinted_pixmap.rect(), color)
        painter.end()
        return QIcon(tinted_pixmap)

    def update_result_mode_button_icons(self, show_document: bool) -> None:
        theme = get_theme()
        active_color = resolve_qcolor(PPOCR_COLOR_TEXT)
        inactive_color = QColor(theme["text_secondary"])
        document_color = active_color if show_document else inactive_color
        json_color = inactive_color if show_document else active_color
        self.document_parsing_button.setIcon(
            self._build_result_mode_icon(
                "document",
                "svg",
                document_color,
            )
        )
        self.json_button.setIcon(
            self._build_result_mode_icon(
                "json",
                "svg",
                json_color,
            )
        )

    def on_copy_result_requested(self) -> None:
        if self.result_content_stack.currentWidget() is self.json_viewer:
            text = self.current_json_text()
            if not text.strip():
                return
            self.copy_text_to_clipboard(text)
            return
        text = get_document_copy_text(
            self.current_blocks,
            self.current_data,
            self.data_manager.root_dir,
        )
        if not text:
            return
        self.copy_text_to_clipboard(text)

    def on_download_result_requested(self) -> None:
        if self.current_record is None:
            return
        if self.result_content_stack.currentWidget() is self.json_viewer:
            self.download_json_result()
            return
        self.download_document_result()

    @staticmethod
    def _sanitize_export_token(value: str) -> str:
        token = value or ""
        for char in ("\\", "/", ":", "*", "?", '"', "<", ">", "|"):
            token = token.replace(char, "_")
        return token.strip() or "model"

    def _current_parsing_model_token(self) -> str:
        meta = (
            self.current_data.get("_ppocr_meta") if self.current_data else {}
        )
        model_id = ""
        if isinstance(meta, dict):
            model_id = str(
                meta.get("pipeline_model")
                or meta.get("content_recognizer_model")
                or ""
            )
        if not model_id:
            model_id = str(
                self.model_combo.currentData()
                or self.model_combo.currentText()
            ).strip()
        return self._sanitize_export_token(model_id)

    def _document_export_pages(self) -> list[Path]:
        if self.current_page_paths:
            return [Path(path) for path in self.current_page_paths]
        if self.current_record is None:
            return []
        try:
            return self.data_manager.get_preview_pages(self.current_record)
        except Exception:
            return []

    def save_layout_detection_image(
        self,
        page_path: Path,
        page_no: int,
        output_path: Path,
    ) -> None:
        with Image.open(page_path) as page_image:
            page_image.load()
            if "A" in page_image.getbands():
                canvas = Image.new("RGB", page_image.size, (255, 255, 255))
                rgba_page = page_image.convert("RGBA")
                canvas.paste(rgba_page, mask=rgba_page.getchannel("A"))
            else:
                canvas = page_image.convert("RGB")
        rgba_canvas = canvas.convert("RGBA")
        overlay = Image.new("RGBA", rgba_canvas.size, (255, 255, 255, 0))
        overlay_drawer = ImageDraw.Draw(overlay, "RGBA")
        font = ImageFont.load_default()
        label_specs: list[
            tuple[tuple[int, int, int, int], str, tuple[int, int, int]]
        ] = []
        for block in self.current_blocks:
            if block.page_no != page_no or not block.points:
                continue
            polygon = [
                (int(round(float(x))), int(round(float(y))))
                for x, y in block.points
            ]
            if len(polygon) < 2:
                continue
            try:
                red, green, blue = ImageColor.getrgb(block.category_color)
            except Exception:
                red, green, blue = (255, 92, 92)
            if len(polygon) >= 3:
                overlay_drawer.polygon(
                    polygon,
                    fill=(red, green, blue, 64),
                    outline=(red, green, blue, 235),
                    width=2,
                )
            else:
                overlay_drawer.line(
                    polygon,
                    fill=(red, green, blue, 235),
                    width=2,
                )
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            label_specs.append(
                (
                    (
                        max(0, min(xs)),
                        max(0, min(ys)),
                        min(rgba_canvas.width, max(xs)),
                        min(rgba_canvas.height, max(ys)),
                    ),
                    block.display_label or block.label,
                    (red, green, blue),
                )
            )
        rgba_canvas = Image.alpha_composite(rgba_canvas, overlay)
        drawer = ImageDraw.Draw(rgba_canvas)
        for bounding_box, label_text, label_color in label_specs:
            if not label_text:
                continue
            red, green, blue = label_color
            x1, y1, _, _ = bounding_box
            left, top, right, bottom = drawer.textbbox(
                (0, 0),
                label_text,
                font=font,
            )
            text_width = right - left
            text_height = bottom - top
            padding_x = 8
            padding_y = 4
            label_width = text_width + padding_x * 2
            label_height = text_height + padding_y * 2
            label_x = max(0, min(x1, rgba_canvas.width - label_width - 1))
            label_y = y1 - label_height
            if label_y < 0:
                label_y = y1
            drawer.rectangle(
                (
                    label_x,
                    label_y,
                    label_x + label_width,
                    label_y + label_height,
                ),
                fill=(red, green, blue, 235),
            )
            drawer.text(
                (label_x + padding_x, label_y + padding_y),
                label_text,
                fill=(255, 255, 255, 255),
                font=font,
            )
        rgba_canvas.convert("RGB").save(
            output_path,
            format="JPEG",
            quality=95,
        )

    def download_document_result(self) -> None:
        markdown_text, image_assets = build_document_markdown_assets(
            self.current_blocks,
            self.current_data,
            self.data_manager.root_dir,
            image_dir_name="imgs",
        )
        page_paths = self._document_export_pages()
        if not markdown_text and not image_assets and not page_paths:
            return
        default_name = (
            f"{Path(self.current_record.filename).stem}_document_parsing.zip"
        )
        export_path = QFileDialog.getSaveFileName(
            self,
            self.tr("Download Document Parsing"),
            default_name,
            self.tr("ZIP Files (*.zip)"),
            options=QFileDialog.Option.DontUseNativeDialog,
        )[0]
        if not export_path:
            return
        export_path = Path(export_path)
        if export_path.suffix.lower() != ".zip":
            export_path = export_path.with_suffix(".zip")
        try:
            with tempfile.TemporaryDirectory(
                prefix="ppocr_export_"
            ) as temp_dir:
                bundle_dir = Path(temp_dir) / "package"
                bundle_dir.mkdir(parents=True, exist_ok=True)
                (bundle_dir / "doc_0.md").write_text(
                    markdown_text,
                    encoding="utf-8",
                )
                imgs_dir = bundle_dir / "imgs"
                imgs_dir.mkdir(parents=True, exist_ok=True)
                for asset in image_assets:
                    target_path = imgs_dir / asset.output_name
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(asset.source_path) as asset_image:
                        asset_image.convert("RGB").save(
                            target_path,
                            format="JPEG",
                            quality=95,
                        )
                for page_index, page_path in enumerate(page_paths):
                    self.save_layout_detection_image(
                        page_path,
                        page_index + 1,
                        bundle_dir / f"layout_det_res_{page_index}.jpg",
                    )
                with zipfile.ZipFile(
                    export_path,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                ) as archive:
                    for file_path in sorted(bundle_dir.rglob("*")):
                        if file_path.is_file():
                            archive.write(
                                file_path,
                                file_path.relative_to(bundle_dir).as_posix(),
                            )
        except Exception as exc:
            QMessageBox.warning(self, self.tr("Save Failed"), str(exc))
            return
        self.show_toast(self.tr("Download Successful"), "copy-green")

    def download_json_result(self) -> None:
        json_text = self.current_json_text()
        if not json_text.strip():
            return
        filename = (
            f"{self.current_record.filename}_by_"
            f"{self._current_parsing_model_token()}.json"
        )
        export_path = QFileDialog.getSaveFileName(
            self,
            self.tr("Download JSON"),
            filename,
            self.tr("JSON Files (*.json)"),
            options=QFileDialog.Option.DontUseNativeDialog,
        )[0]
        if not export_path:
            return
        export_file = Path(export_path)
        if export_file.suffix.lower() != ".json":
            export_file = export_file.with_suffix(".json")
        try:
            export_file.write_text(json_text, encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, self.tr("Save Failed"), str(exc))
            return
        self.show_toast(self.tr("Download Successful"), "copy-green")

    def current_json_text(self) -> str:
        if not self.current_data:
            return ""
        return json.dumps(self.current_data, ensure_ascii=False, indent=2)

    @staticmethod
    def _teardown_card_widget(widget: QWidget) -> None:
        actions_widget = getattr(widget, "actions_widget", None)
        if actions_widget is not None:
            actions_widget.hide()
            actions_widget.deleteLater()
        widget.hide()
        widget.deleteLater()

    def _create_block_editor(self, block):
        editor = create_ppocr_block_editor(
            block.label,
            block.content,
            image_path=(
                self.data_manager.root_dir / block.image_path
                if block.image_path
                else None
            ),
        )
        editor.saveRequested.connect(
            lambda text, block_data=block, editor_widget=editor: self.save_block_edit(
                block_data,
                text,
                editor_widget,
            )
        )
        editor.cancelRequested.connect(self.cancel_block_edit)
        return editor

    def _create_block_card(
        self,
        block,
        formula_render_delay_ms: int = 0,
    ) -> PPOCRBlockCard:
        card = PPOCRBlockCard(
            block,
            self.data_manager.root_dir,
            formula_render_delay_ms=formula_render_delay_ms,
        )
        card.copyRequested.connect(self.on_copy_block_requested)
        card.correctRequested.connect(self.on_correct_block_requested)
        card.blockHovered.connect(self.on_card_block_hovered)
        card.blockSelected.connect(self.on_card_block_selected)
        card.refresh_hover_state(self.hovered_block_key)
        card.refresh_selection_state(
            self.selected_block_key,
            selected_locally=self.selected_block_source == "card",
        )
        return card

    def _create_page_transition_divider(self, page_no: int) -> QWidget:
        divider = QWidget()
        divider.setObjectName("PPOCRPageDivider")
        layout = QHBoxLayout(divider)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(10)

        left_line = QFrame(divider)
        left_line.setObjectName("PPOCRPageDividerLineLeft")
        left_line.setFixedHeight(1)
        layout.addWidget(left_line, 1)

        page_label = QLabel(self.tr("Page {0}").format(page_no), divider)
        page_label.setObjectName("PPOCRPageDividerText")
        page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(page_label, 0)

        right_line = QFrame(divider)
        right_line.setObjectName("PPOCRPageDividerLineRight")
        right_line.setFixedHeight(1)
        layout.addWidget(right_line, 1)
        return divider

    def _replace_card_widget(
        self, block_key: str, replacement: QWidget
    ) -> bool:
        current_widget = self.card_widgets.get(block_key)
        if current_widget is None:
            return False
        index = self.cards_layout.indexOf(current_widget)
        if index < 0:
            return False
        self.cards_layout.removeWidget(current_widget)
        self._teardown_card_widget(current_widget)
        self.cards_layout.insertWidget(index, replacement)
        self.card_widgets[block_key] = replacement
        return True

    def _restore_card_from_editor(self, block_key: str) -> bool:
        current_widget = self.card_widgets.get(block_key)
        if current_widget is None:
            return False
        if isinstance(current_widget, PPOCRBlockCard):
            return True
        block = self.block_map().get(block_key)
        if block is None:
            return False
        return self._replace_card_widget(
            block_key,
            self._create_block_card(block),
        )

    def rebuild_cards(self, ensure_current_page_visible: bool = True) -> None:
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                self._teardown_card_widget(widget)
        self.card_widgets = {}
        self.page_anchor_widgets = {}
        previous_page = None
        formula_block_count = 0
        for block in self.panel_blocks():
            if self.current_record.file_type == PPOCR_FILE_TYPE_PDF:
                if (
                    previous_page is not None
                    and block.page_no != previous_page
                ):
                    self.cards_layout.addWidget(
                        self._create_page_transition_divider(previous_page)
                    )
            if block.block_key == self.editing_block_key:
                editor = self._create_block_editor(block)
                self.cards_layout.addWidget(editor)
                self.card_widgets[block.block_key] = editor
                self.page_anchor_widgets.setdefault(block.page_no, editor)
                previous_page = block.page_no
                continue
            formula_render_delay_ms = 0
            if normalize_block_label(block.label) in FORMULA_BLOCK_LABELS:
                formula_block_count += 1
                if formula_block_count > 4:
                    formula_render_delay_ms = min(
                        800,
                        (formula_block_count - 4) * 10,
                    )
            card = self._create_block_card(
                block,
                formula_render_delay_ms=formula_render_delay_ms,
            )
            self.cards_layout.addWidget(card)
            self.card_widgets[block.block_key] = card
            self.page_anchor_widgets.setdefault(block.page_no, card)
            previous_page = block.page_no
        if (
            self.current_record.file_type == PPOCR_FILE_TYPE_PDF
            and previous_page is not None
        ):
            self.cards_layout.addWidget(
                self._create_page_transition_divider(previous_page)
            )
        self.cards_layout.addStretch()
        if (
            ensure_current_page_visible
            and self.current_record
            and self.current_record.file_type == PPOCR_FILE_TYPE_PDF
            and self.current_page_no in self.page_anchor_widgets
        ):
            self.cards_scroll.ensureWidgetVisible(
                self.page_anchor_widgets[self.current_page_no]
            )

    def block_map(self) -> dict[str, object]:
        return {block.block_key: block for block in self.current_blocks}

    def panel_blocks(self) -> list[object]:
        return [
            block
            for block in self.current_blocks
            if not getattr(block, "hidden_in_panel", False)
        ]

    def representative_block_key(self, block_key: str) -> str:
        block = self.block_map().get(block_key) if block_key else None
        if block is None:
            return block_key
        representative_key = getattr(
            block,
            "representative_block_key",
            "",
        )
        return representative_key or block.block_key

    def linked_block_keys(self, block_key: str) -> list[str]:
        representative_key = self.representative_block_key(block_key)
        if not representative_key:
            return []
        representative = self.block_map().get(representative_key)
        if representative is None:
            return [representative_key]
        linked_keys = list(
            getattr(representative, "linked_block_keys", ()) or ()
        )
        linked_keys = [key for key in linked_keys if key]
        return linked_keys or [representative_key]

    def on_canvas_block_hovered(self, block_key: str) -> None:
        self.set_hovered_block(block_key)

    def on_card_block_hovered(self, block_key: str) -> None:
        self.set_hovered_block(block_key)

    def on_canvas_block_selected(self, block_key: str) -> None:
        selected_key = self.representative_block_key(block_key)
        self.selected_block_key = selected_key
        self.selected_block_source = "canvas"
        self.preview_canvas.set_selected_block(
            selected_key,
            selected_locally=True,
        )
        self.refresh_cards_selection()
        card = self.card_widgets.get(selected_key)
        if card is not None:
            self.cards_scroll.ensureWidgetVisible(card)

    def on_canvas_cleared(self) -> None:
        self.selected_block_key = ""
        self.selected_block_source = ""
        self.preview_canvas.set_selected_block("")
        self.refresh_cards_selection()

    def on_card_block_selected(self, block_key: str) -> None:
        selected_key = self.representative_block_key(block_key)
        block = self.block_map().get(selected_key)
        if block is None:
            return
        self.selected_block_key = selected_key
        self.selected_block_source = "card"
        if block.page_no != self.current_page_no:
            self.current_page_no = block.page_no
            self.render_preview_page()
            self.update_page_control_state()
        self.preview_canvas.set_selected_block(
            selected_key,
            selected_locally=False,
        )
        self.ensure_preview_block_visible(selected_key)
        self.refresh_cards_selection()

    def refresh_cards_selection(self) -> None:
        for widget in self.card_widgets.values():
            if hasattr(widget, "refresh_selection_state"):
                widget.refresh_selection_state(
                    self.selected_block_key,
                    selected_locally=self.selected_block_source == "card",
                )

    def ensure_preview_block_visible(self, block_key: str) -> None:
        block_rect = self.preview_canvas.block_view_rect(block_key)
        if block_rect.isNull():
            return
        center = block_rect.center()
        self.preview_scroll.ensureVisible(
            center.x(),
            center.y(),
            max(32, block_rect.width() // 2),
            max(32, block_rect.height() // 2),
        )

    def refresh_cards_hover(self) -> None:
        for widget in self.card_widgets.values():
            if hasattr(widget, "refresh_hover_state"):
                widget.refresh_hover_state(self.hovered_block_key)

    def set_hovered_block(
        self,
        block_key: str,
        ensure_visible: bool = False,
    ) -> None:
        resolved_key = self.representative_block_key(block_key)
        self.hovered_block_key = resolved_key
        hovered_block = (
            self.block_map().get(resolved_key) if resolved_key else None
        )
        if hovered_block and hovered_block.page_no == self.current_page_no:
            self.preview_canvas.set_hovered_blocks(
                self.linked_block_keys(resolved_key)
            )
        else:
            self.preview_canvas.set_hovered_blocks([])
        self.refresh_cards_hover()
        if ensure_visible:
            card = self.card_widgets.get(resolved_key)
            if card is not None:
                self.cards_scroll.ensureWidgetVisible(card)

    def on_copy_block_requested(self, block) -> None:
        text = get_block_copy_text(block, self.data_manager.root_dir)
        self.copy_text_to_clipboard(text)

    def on_canvas_block_copy_requested(self, block_key: str) -> None:
        copy_key = self.representative_block_key(block_key)
        block = self.block_map().get(copy_key)
        if block is None:
            return
        self.on_copy_block_requested(block)

    def on_correct_block_requested(self, block) -> None:
        scroll_value = self.cards_scroll.verticalScrollBar().value()
        previous_editing_key = self.editing_block_key
        self.editing_block_key = block.block_key
        self.selected_block_key = block.block_key
        self.selected_block_source = "card"
        editor = None

        if (
            previous_editing_key
            and previous_editing_key != block.block_key
            and not self._restore_card_from_editor(previous_editing_key)
        ):
            self.rebuild_cards(ensure_current_page_visible=False)

        current_widget = self.card_widgets.get(block.block_key)
        if current_widget is not None:
            if isinstance(current_widget, PPOCRBlockCard):
                next_editor = self._create_block_editor(block)
                if self._replace_card_widget(block.block_key, next_editor):
                    editor = next_editor
            else:
                editor = current_widget

        if editor is None:
            self.rebuild_cards(ensure_current_page_visible=False)
            editor = self.card_widgets.get(block.block_key)

        self.refresh_cards_selection()
        if editor is not None:
            QTimer.singleShot(
                0,
                lambda editor_widget=editor, value=scroll_value: self.activate_block_editor(
                    editor_widget,
                    value,
                ),
            )

    def activate_block_editor(self, editor, scroll_value: int) -> None:
        scroll_bar = self.cards_scroll.verticalScrollBar()
        scroll_bar.setValue(min(scroll_value, scroll_bar.maximum()))
        text_editor = getattr(editor, "editor", None)
        if text_editor is None:
            return
        if hasattr(text_editor, "textCursor") and hasattr(
            text_editor, "setTextCursor"
        ):
            cursor = text_editor.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            text_editor.setTextCursor(cursor)
            if hasattr(text_editor, "ensureCursorVisible"):
                text_editor.ensureCursorVisible()
        text_editor.setFocus(Qt.FocusReason.MouseFocusReason)
        QTimer.singleShot(
            0,
            lambda value=scroll_value: scroll_bar.setValue(
                min(value, scroll_bar.maximum())
            ),
        )

    def save_block_edit(self, block, text: str, editor) -> None:
        if self.current_record is None:
            return
        if not editor.content_height_valid():
            self.editing_block_key = ""
            self.show_toast(
                self.tr("Block content exceeds the available editor height"),
                "warning",
            )
            self.load_current_record()
            return
        try:
            self.current_data = self.data_manager.save_block_content(
                self.current_record,
                block.page_no,
                block.block_uid,
                text,
            )
        except Exception as exc:
            QMessageBox.warning(self, self.tr("Save Failed"), str(exc))
            return
        self.editing_block_key = ""
        self.current_blocks = extract_blocks(self.current_data)
        for current_block in self.current_blocks:
            if current_block.block_key == block.block_key:
                current_block.edited = True
                break
        self.render_preview_page()
        self.render_right_panel()
        self.refresh_records(self.current_record.filename)

    def cancel_block_edit(self) -> None:
        self.editing_block_key = ""
        self.load_current_record()

    def on_retry_requested(self) -> None:
        if self.current_record is None:
            return
        if self.current_record.filename in self.parsing_filenames:
            return
        self.data_manager.clear_error_and_edits(self.current_record)
        self.refresh_records(self.current_record.filename)
        self.current_record = self.record_map.get(
            self.current_record.filename,
            self.current_record,
        )
        self.start_parsing([self.current_record])

    def on_delete_requested(self, filename: str) -> None:
        record = self.record_map.get(filename)
        if record is None:
            return
        result = QMessageBox.question(
            self,
            self.tr("Delete File"),
            self.tr(
                "Are you sure you want to delete this file? This action "
                "cannot be undone and will remove all associated data."
            )
            + f"\n\n{filename}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return
        current_index = next(
            (
                index
                for index, item in enumerate(self.records)
                if item.filename == filename
            ),
            -1,
        )
        self.data_manager.delete_record(record)
        fallback_name = ""
        remaining_records = [
            item for item in self.records if item.filename != filename
        ]
        if remaining_records:
            if current_index < len(remaining_records):
                fallback_name = remaining_records[current_index].filename
            else:
                fallback_name = remaining_records[-1].filename
        self.refresh_records(fallback_name)
        if fallback_name:
            self.on_record_selected(fallback_name)
        else:
            self.show_empty_state()

    def start_parsing(self, records: list[PPOCRFileRecord]) -> None:
        self.cleanup_stale_parsing_state()
        if self.worker_thread is not None and self.worker_thread.isRunning():
            return
        if (
            self.pipeline.pipeline_model == PPOCR_API_MODEL_ID
            and not self.pipeline.has_required_api_settings()
        ):
            self.prompt_api_settings_if_needed()
            if not self.pipeline.has_required_api_settings():
                return
        records = [
            record
            for record in records
            if record.filename not in self.parsing_filenames
        ]
        if not records:
            return
        self.cancel_event = threading.Event()
        self.queued_filenames = {record.filename for record in records}
        self.parsing_filenames = set()
        self.worker_thread = QThread()
        self.worker = PPOCRPipelineWorker(
            self.pipeline,
            records,
            self.cancel_event,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progressChanged.connect(self.on_parsing_progress_changed)
        self.worker.recordStarted.connect(self.on_parse_record_started)
        self.worker.recordFinished.connect(self.on_parse_record_finished)
        self.worker.recordFailed.connect(self.on_parse_record_failed)
        self.worker.batchFinished.connect(self.on_parse_batch_finished)
        self.worker.batchCancelled.connect(self.on_parse_batch_cancelled)
        self.worker_thread.start()
        self.render_right_panel()

    def on_parsing_progress_changed(self, progress) -> None:
        if isinstance(progress, PPOCRParsingProgress):
            self.parsing_progress_map[progress.filename] = progress
            if (
                self.current_record is not None
                and self.current_record.filename == progress.filename
            ):
                self.render_right_panel()

    def on_parse_record_started(self, filename: str) -> None:
        self.queued_filenames.discard(filename)
        self.parsing_filenames.add(filename)
        self.parsing_progress_map.pop(filename, None)
        self.refresh_records(
            self.current_record.filename if self.current_record else ""
        )
        if self.current_record and self.current_record.filename == filename:
            self.render_right_panel()

    def on_parse_record_finished(
        self, filename: str, document_data: dict
    ) -> None:
        self.parsing_filenames.discard(filename)
        self.queued_filenames.discard(filename)
        self.parsing_progress_map.pop(filename, None)
        self.refresh_records(
            self.current_record.filename if self.current_record else ""
        )
        if self.current_record and self.current_record.filename == filename:
            self.current_record = self.record_map.get(
                filename, self.current_record
            )
            self.current_data = (
                document_data
                or self.data_manager.load_record_data(self.current_record)
            )
            self.current_blocks = extract_blocks(self.current_data)
            self.current_page_paths = self.data_manager.get_preview_pages(
                self.current_record
            )
            self.current_page_no = 1
            self.hovered_block_key = ""
            self.selected_block_key = ""
            self.selected_block_source = ""
            self.editing_block_key = ""
            self.render_preview_page()
            self.render_right_panel()
            self.update_page_control_state()

    def on_parse_record_failed(
        self, filename: str, error_message: str
    ) -> None:
        self.parsing_filenames.discard(filename)
        self.queued_filenames.discard(filename)
        self.parsing_progress_map.pop(filename, None)
        self.refresh_records(
            self.current_record.filename if self.current_record else ""
        )
        if self.current_record and self.current_record.filename == filename:
            self.current_record = self.record_map.get(
                filename, self.current_record
            )
            self.load_current_record()

    def on_parse_batch_finished(self) -> None:
        self.parsing_progress_map.clear()
        self.stop_worker()
        self.refresh_records(
            self.current_record.filename if self.current_record else ""
        )
        if self.current_record:
            self.current_record = self.record_map.get(
                self.current_record.filename,
                self.current_record,
            )
            self.load_current_record()

    def on_parse_batch_cancelled(self) -> None:
        self.parsing_progress_map.clear()
        self.stop_worker()
        self.queued_filenames = set()
        self.parsing_filenames = set()
        self.refresh_records(
            self.current_record.filename if self.current_record else ""
        )
        if self.current_record:
            self.current_record = self.record_map.get(
                self.current_record.filename,
                self.current_record,
            )
            self.load_current_record()

    def stop_worker(
        self, wait: bool = True, wait_timeout_ms: int = 2000
    ) -> None:
        self._stop_pending_status_animation()
        worker_thread = self.worker_thread
        worker = self.worker
        self.worker_thread = None
        self.worker = None
        if worker_thread is not None:
            worker_thread.quit()
            if wait:
                worker_thread.wait(wait_timeout_ms)
            if worker_thread.isFinished():
                worker_thread.deleteLater()
            else:
                worker_thread.finished.connect(worker_thread.deleteLater)
        if worker is not None:
            worker.deleteLater()

    def goto_previous_page(self) -> None:
        if self.current_page_no <= 1:
            return
        self.current_page_no -= 1
        self.editing_block_key = ""
        self.render_preview_page()
        self.update_page_control_state()
        self.render_right_panel()

    def goto_next_page(self) -> None:
        total_pages = max(1, len(self.current_page_paths))
        if self.current_page_no >= total_pages:
            return
        self.current_page_no += 1
        self.editing_block_key = ""
        self.render_preview_page()
        self.update_page_control_state()
        self.render_right_panel()

    def on_page_input_finished(self) -> None:
        total_pages = max(1, len(self.current_page_paths))
        text = self.page_input.text().strip()
        if not text.isdigit():
            self.page_input.setText(str(self.current_page_no))
            return
        self.current_page_no = max(1, min(total_pages, int(text)))
        self.editing_block_key = ""
        self.render_preview_page()
        self.update_page_control_state()
        self.render_right_panel()

    def apply_zoom(self, delta: float) -> None:
        self.preview_canvas.set_scale(
            self.preview_canvas.current_scale() + delta
        )

    def reset_zoom(self) -> None:
        self.current_fit_scale = self.preview_canvas.set_fit_width(
            self.preview_scroll.viewport().width()
        )
        self.reset_zoom_button.setEnabled(False)

    def on_canvas_scale_changed(self, scale: float) -> None:
        self.reset_zoom_button.setEnabled(
            abs(scale - self.current_fit_scale) > 1e-3
        )

    def update_page_control_state(self) -> None:
        total_pages = max(
            1,
            len(self.current_page_paths)
            or (
                self.current_record.page_count
                if self.current_record is not None
                else 1
            ),
        )
        if self.current_data:
            total_pages = max(
                total_pages,
                document_page_count(self.current_data, default=total_pages),
            )
        self.page_input.blockSignals(True)
        self.page_input.setText(str(self.current_page_no))
        self.page_input.blockSignals(False)
        self.page_total_label.setText(str(total_pages))
        page_controls_enabled = total_pages > 1
        self.prev_page_button.setEnabled(
            page_controls_enabled and self.current_page_no > 1
        )
        self.next_page_button.setEnabled(
            page_controls_enabled and self.current_page_no < total_pages
        )
        self.page_input.setEnabled(page_controls_enabled)
        self.update_page_control_position()

    def update_page_control_position(self) -> None:
        if not hasattr(self, "page_control") or not hasattr(
            self, "preview_content_frame"
        ):
            return
        if self.page_control is None or self.preview_content_frame is None:
            return
        host_width = self.preview_content_frame.width()
        host_height = self.preview_content_frame.height()
        if host_width <= 0 or host_height <= 0:
            return
        control_width = self.page_control.width()
        control_height = self.page_control.height()
        bottom_gap = 16
        if hasattr(self, "preview_scroll") and self.preview_scroll is not None:
            horizontal_scrollbar = self.preview_scroll.horizontalScrollBar()
            if horizontal_scrollbar.maximum() > horizontal_scrollbar.minimum():
                bottom_gap += (
                    horizontal_scrollbar.height()
                    or horizontal_scrollbar.sizeHint().height()
                )
        x = max(0, (host_width - control_width) // 2)
        y = max(0, host_height - control_height - bottom_gap)
        self.page_control.move(x, y)
        self.page_control.raise_()

    def on_model_changed(self, _index: int) -> None:
        self.update_model_combo_width()
        model_id = str(
            self.model_combo.currentData() or self.model_combo.currentText()
        ).strip()
        self.pipeline.set_pipeline_model(model_id)
        if (
            model_id == PPOCR_API_MODEL_ID
            and not self.pipeline.has_required_api_settings()
        ):
            QTimer.singleShot(0, self.prompt_api_settings_if_needed)

    def copy_text_to_clipboard(self, text: str) -> None:
        self.show_toast(
            self.tr("Copy Successful"), "copy-green", copy_text=text
        )

    def show_toast(
        self,
        text: str,
        icon_name: str,
        copy_text: str = "",
    ) -> None:
        popup = Popup(
            text,
            parent=self,
            icon=new_icon_path(icon_name, "svg"),
        )
        popup.show_popup(self, copy_msg=copy_text, position="default")

    def format_size(self, size_bytes: int) -> str:
        size = float(size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                if unit == "B":
                    return f"{int(size)}{unit}"
                return f"{size:.2f}{unit}"
            size /= 1024
        return f"{size_bytes}B"

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.enforce_workspace_splitter_ratio()
        self.update_page_control_position()
        self.update_model_combo_width()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self.refresh_service_state)
        QTimer.singleShot(0, self.prompt_api_settings_if_needed)
        QTimer.singleShot(0, self.update_model_combo_width)
        QTimer.singleShot(60, self.update_model_combo_width)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            file_paths = [
                url.toLocalFile()
                for url in event.mimeData().urls()
                if url.isLocalFile()
            ]
            if file_paths:
                self.import_source_paths(file_paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def closeEvent(self, event) -> None:
        self.cancel_event.set()
        self.stop_worker(wait=False)
        super().closeEvent(event)
