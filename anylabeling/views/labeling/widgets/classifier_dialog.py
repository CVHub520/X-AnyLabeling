import os
import re
import json

from PyQt6.QtCore import (
    QCoreApplication,
    QEvent,
    Qt,
    QTimer,
    QSize,
    QTemporaryFile,
    QPoint,
    QSettings,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLayout,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSizePolicy,
    QSplitter,
    QStyle,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import (
    QColor,
    QIcon,
    QIntValidator,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)

from anylabeling.views.labeling.classifier import *
from anylabeling.views.labeling.chatbot.provider import save_model_selection
from anylabeling.views.labeling.classifier.widgets import (
    MultiTaskForm,
    ObjectComboBox,
    ObjectItemDelegate,
    ResizableTextEdit,
    configure_object_combo,
)
from anylabeling.views.labeling.classifier.utils import MultiTaskBatch
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils import general
from anylabeling.views.labeling.utils.qt import new_icon, new_icon_path
from anylabeling.views.labeling.utils.style import (
    get_dock_style,
    get_progress_dialog_style,
    get_scrollbar_style,
    get_spinbox_style,
)
from anylabeling.views.labeling.utils.theme import get_theme
from anylabeling.views.labeling.widgets.popup import Popup
from anylabeling.views.labeling.vqa.dialogs import (
    AILoadingDialog,
    AIPromptDialog,
)
from anylabeling.views.labeling.vqa.style import get_page_input_style
from anylabeling.views.labeling.vqa.utils import AIWorkerThread

PANEL_BUTTON_SIZE = 28
PANEL_ICON_SIZE = 16
PANEL_ICON_RENDER_SCALE = 2


def _new_panel_icon(icon_name, color="text_secondary"):
    real_size = PANEL_ICON_SIZE * PANEL_ICON_RENDER_SCALE
    source_pixmap = QIcon(new_icon_path(icon_name, "svg")).pixmap(
        QSize(real_size, real_size)
    )
    if source_pixmap.isNull():
        return QIcon(new_icon(icon_name, "svg"))

    tinted_pixmap = QPixmap(source_pixmap.size())
    tinted_pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(tinted_pixmap)
    painter.drawPixmap(0, 0, source_pixmap)
    painter.setCompositionMode(
        QPainter.CompositionMode.CompositionMode_SourceIn
    )
    painter.fillRect(tinted_pixmap.rect(), QColor(get_theme()[color]))
    painter.end()

    icon = QIcon()
    icon.addPixmap(tinted_pixmap, QIcon.Mode.Normal)
    disabled = QPixmap(tinted_pixmap.size())
    disabled.fill(Qt.GlobalColor.transparent)
    painter = QPainter(disabled)
    painter.setOpacity(0.35)
    painter.drawPixmap(0, 0, tinted_pixmap)
    painter.end()
    icon.addPixmap(disabled, QIcon.Mode.Disabled)
    return icon


def _object_navigation_icon(up):
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(get_theme()["text"]), 2))
    painter.drawEllipse(4, 4, 24, 24)
    if up:
        painter.drawLine(10, 19, 16, 13)
        painter.drawLine(16, 13, 22, 19)
    else:
        painter.drawLine(10, 13, 16, 19)
        painter.drawLine(16, 19, 22, 13)
    painter.end()
    return QIcon(pixmap)


class ModelItemDelegate(ObjectItemDelegate):
    def paint(self, painter, option, index):
        option = QStyleOptionViewItem(option)
        option.state &= ~QStyle.StateFlag.State_HasFocus
        super().paint(painter, option, index)


class OverwriteConfirmationDialog(QDialog):
    def __init__(self, parent, text):
        super().__init__(parent)
        self.setWindowTitle(self.tr("AI Assistant"))
        layout = QVBoxLayout(self)
        content = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(
            self.style()
            .standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            .pixmap(32, 32)
        )
        content.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
        label = QLabel(text)
        label.setWordWrap(True)
        content.addWidget(label, 1)
        layout.addLayout(content)
        buttons = QHBoxLayout()
        self.checkbox = QCheckBox(self.tr("Don't show again"))
        buttons.addWidget(self.checkbox)
        buttons.addStretch()
        cancel = QPushButton(self.tr("Cancel"))
        cancel.setStyleSheet(get_dialog_button_style("secondary", "medium"))
        cancel.clicked.connect(self.reject)
        confirm = QPushButton(self.tr("Confirm"))
        confirm.setStyleSheet(get_dialog_button_style("primary", "medium"))
        confirm.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        layout.addLayout(buttons)
        self.setMinimumWidth(440)


class MultiTaskAIDialog(QDialog):
    def __init__(self, parent, prompt, batch=False, classification=False):
        super().__init__(parent)
        self.owner = parent
        self.batch = batch
        self.classification = classification
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
        )
        theme = get_theme()
        self.setStyleSheet(
            "MultiTaskAIDialog {"
            f" background-color: {theme['background']};"
            f" border: 1px solid {theme['border']};"
            "}"
        )
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        header = QHBoxLayout()
        header.addWidget(QLabel(self.tr("Current model")))
        header.addStretch()
        self.model_search = QLineEdit()
        self.model_search.setPlaceholderText(self.tr("Search models..."))
        self.model_search.setClearButtonEnabled(True)
        self.model_search.setMaximumWidth(220)
        self.model_search.setStyleSheet(get_dock_style())
        header.addWidget(self.model_search)
        settings = QPushButton()
        settings.setIcon(_new_panel_icon("settings", "text"))
        settings.setFixedSize(24, 24)
        settings.setStyleSheet(get_panel_button_style())
        settings.setToolTip(self.tr("Chatbot Settings"))
        settings.clicked.connect(self.open_settings)
        header.addWidget(settings)
        layout.addLayout(header)
        self.models = ObjectComboBox()
        configure_object_combo(self.models, 28)
        self.models.setItemDelegate(ModelItemDelegate(self.models))
        layout.addWidget(self.models)
        layout.addWidget(QLabel(self.tr("Prompt")))
        self.prompt_input = ResizableTextEdit()
        self.prompt_input.setFixedHeight(150)
        self.prompt_input.setPlainText(prompt)
        self.prompt_input.verticalScrollBar().setStyleSheet(
            get_scrollbar_style()
        )
        layout.addWidget(self.prompt_input)
        self.concurrency = QSpinBox()
        self.concurrency.setRange(1, 8)
        self.concurrency.setValue(2)
        self.concurrency.setStyleSheet(get_spinbox_style())
        self.concurrency.setFixedWidth(76)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        if batch:
            buttons.addWidget(QLabel(self.tr("Concurrent requests")))
            buttons.addWidget(self.concurrency)
        buttons.addStretch()
        cancel = QPushButton(self.tr("Cancel"))
        cancel.setStyleSheet(get_dialog_button_style("secondary", "medium"))
        cancel.clicked.connect(self.reject)
        self.generate = QPushButton(self.tr("Generate"))
        self.generate.setStyleSheet(
            get_dialog_button_style("primary", "medium")
        )
        self.generate.clicked.connect(self.confirm_generate)
        buttons.addWidget(cancel)
        buttons.addWidget(self.generate)
        layout.addLayout(buttons)
        self.models.currentIndexChanged.connect(self.select_model)
        self.model_search.textChanged.connect(self.filter_models)
        self.prompt_input.textChanged.connect(self.update_generate)
        self.refresh_models()

    def refresh_models(self):
        config = AIWorkerThread("", "", {}).load_models_config() or {}
        settings = config.get("settings", {})
        selected = (settings.get("provider"), settings.get("model_id"))
        models = config.get("models_data", {})
        entries = [
            (provider, model)
            for provider, provider_models in models.items()
            for model in provider_models
        ]
        if all(selected) and selected not in entries:
            entries.insert(0, selected)
        self.models.blockSignals(True)
        self.models.clear()
        self.models.addItem(self.tr("Select a model"), None)
        for provider, model in entries:
            self.models.addItem(f"{model} ({provider})", (provider, model))
        index = entries.index(selected) + 1 if all(selected) else 0
        self.models.setCurrentIndex(index)
        self.models.blockSignals(False)
        self.filter_models(self.model_search.text())
        self.update_generate()

    def filter_models(self, text):
        query = text.casefold().strip()
        for index in range(1, self.models.count()):
            self.models.view().setRowHidden(
                index, query not in self.models.itemText(index).casefold()
            )

    def select_model(self):
        selected = self.models.currentData()
        if selected:
            try:
                save_model_selection(*selected)
            except (OSError, ValueError, KeyError) as error:
                QMessageBox.warning(self, self.tr("Error"), str(error))
                self.refresh_models()
        self.update_generate()

    def update_generate(self):
        self.generate.setEnabled(
            bool(self.models.currentData()) and bool(self.get_prompt())
        )

    def get_prompt(self):
        return self.prompt_input.toPlainText().strip()

    def confirm_generate(self):
        settings = getattr(self.owner.label_widget, "settings", None)
        if settings is None:
            settings = QSettings("anylabeling", "anylabeling")
        key = (
            "classifier/classification_ai_hide_batch_overwrite_warning"
            if self.classification
            else (
                "classifier/multitask_ai_hide_batch_overwrite_warning"
                if self.batch
                else "classifier/multitask_ai_hide_overwrite_warning"
            )
        )
        if not settings.value(key, False, type=bool):
            text = (
                self.tr(
                    "AI results will overwrite existing classification values for images in the selected range."
                )
                if self.classification
                else (
                    self.tr(
                        "AI results will overwrite existing attribute values for all objects in the selected range."
                    )
                    if self.batch
                    else self.tr(
                        "AI results will overwrite existing values for the selected object's attributes."
                    )
                )
            )
            message = OverwriteConfirmationDialog(self, text)
            if message.exec() != QDialog.DialogCode.Accepted:
                return
            if message.checkbox.isChecked():
                settings.setValue(key, True)
        self.accept()

    def open_settings(self):
        self.owner.label_widget.open_chatbot()
        self.refresh_models()


class ClassifierDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label_widget = parent
        self.setWindowTitle(self.tr(DEFAULT_WINDOW_TITLE))
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.resize(*DEFAULT_WINDOW_SIZE)

        self.image_files = []
        self.current_image_index = 0
        self.labels = []
        self.is_multiclass = True
        self.is_multitask = False
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.switching_image = False
        self._preview_filename = None
        self._ai_workers = set()

        self.init_ui()
        self.load_initial_data()
        self.setup_shortcuts()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(20)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(3)
        self.main_splitter.setStyleSheet(get_main_splitter_style())
        self.main_splitter.setChildrenCollapsible(False)

        left_widget = QWidget()
        left_widget.setMinimumWidth(350)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.setSpacing(10)

        self.filename_label = QLabel(self.tr("No image loaded"))
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename_label.setFixedHeight(DEFAULT_COMPONENT_HEIGHT)
        self.filename_label.setStyleSheet(get_filename_label_style())
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        self.preview_title_spacer = QWidget()
        self.preview_title_spacer.setFixedWidth(96)
        self.preview_title_spacer.hide()
        title_row.addWidget(self.preview_title_spacer)
        title_row.addWidget(self.filename_label, 1)
        self.preview_controls = QWidget()
        preview_controls_layout = QHBoxLayout(self.preview_controls)
        preview_controls_layout.setContentsMargins(0, 0, 0, 0)
        preview_controls_layout.setSpacing(0)
        self.show_labels_button = QPushButton()
        self.show_labels_button.setIcon(new_icon("tag"))
        self.show_labels_button.setToolTip(self.tr("Show Labels"))
        self.mask_background_button = QPushButton()
        self.mask_background_button.setIcon(new_icon("contrast"))
        self.mask_background_button.setToolTip(self.tr("Mask Background"))
        self.fit_window_button = QPushButton()
        self.fit_width_button = QPushButton()
        self.fit_window_button.setIcon(new_icon("fit-window"))
        self.fit_width_button.setIcon(new_icon("fit-width"))
        for button in (
            self.show_labels_button,
            self.mask_background_button,
            self.fit_window_button,
            self.fit_width_button,
        ):
            button.setIconSize(QSize(PANEL_ICON_SIZE, PANEL_ICON_SIZE))
            button.setFixedSize(24, 24)
            button.setStyleSheet(get_panel_button_style())
            button.setCheckable(True)
            preview_controls_layout.addWidget(button)
        self.show_labels_button.setStyleSheet(
            get_panel_button_style()
            + f"QPushButton:checked {{ background-color: {get_theme()['surface_pressed']}; }}"
        )
        self.show_labels_button.setChecked(True)
        self.show_labels_button.toggled.connect(self.toggle_object_labels)
        self.mask_background_button.setStyleSheet(
            get_panel_button_style()
            + f"QPushButton:checked {{ background-color: {get_theme()['surface_pressed']}; }}"
        )
        self.mask_background_button.toggled.connect(
            self.toggle_background_mask
        )
        self.fit_window_button.setChecked(True)
        self.fit_window_button.clicked.connect(
            lambda: self.reset_preview_fit("window")
        )
        self.fit_width_button.clicked.connect(
            lambda: self.reset_preview_fit("width")
        )
        title_row.addWidget(self.preview_controls)
        self.preview_controls.hide()
        left_layout.addLayout(title_row)

        self.image_container = QWidget()
        self.image_container.setStyleSheet(get_image_container_style())
        container_layout = QVBoxLayout(self.image_container)
        container_layout.setContentsMargins(8, 8, 8, 8)

        self.image_label = ClassificationImagePreview()
        self.image_label.source_canvas = self.label_widget.canvas
        self.image_label.label_font_size = getattr(
            self.label_widget.canvas, "label_font_size", 8
        )
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self.image_label.setScaledContents(False)
        container_layout.addWidget(self.image_label, 1)

        self.image_label.object_selected.connect(self.select_object)
        self.image_label.background_clicked.connect(
            lambda: self.set_preview_fit("window")
        )
        self.overlay_label = ClassificationOverlay(self.image_label)
        self.image_label.resized.connect(
            lambda: self.overlay_label.position_overlay(self.image_label)
        )

        left_layout.addWidget(self.image_container, 1)

        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(20)
        status_layout.addStretch()
        self.status_buttons = {}
        theme = get_theme()
        adjustment = getattr(self.label_widget, "canvas_adjustment", None)
        for status, text, color in (
            (False, "Pending", theme["primary"]),
            (True, "Passed", theme["success"]),
            ("failed", "Failed", theme["error"]),
        ):
            button = QRadioButton(
                QCoreApplication.translate("CanvasAdjustmentWidget", text)
            )
            button.setStyleSheet(
                "QRadioButton { spacing: 4px; }"
                "QRadioButton::indicator { width: 12px; height: 12px;"
                f" border: 1px solid {theme['text_secondary']};"
                f" border-radius: 7px; background: {theme['background']}; }}"
                "QRadioButton::indicator:checked { width: 6px; height: 6px;"
                f" border: 4px solid {color}; border-radius: 7px; }}"
            )
            button.clicked.connect(
                lambda _checked, value=status: (
                    self.label_widget.set_annotation_checked(value)
                )
            )
            if adjustment is not None:
                adjustment.status_buttons[status].toggled.connect(
                    button.setChecked
                )
            self.status_buttons[status] = button
            status_layout.addWidget(button)
        status_layout.addStretch()
        left_layout.addLayout(status_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        action_widget = QWidget()
        action_layout = QGridLayout(action_widget)
        action_layout.setContentsMargins(10, 4, 0, 0)
        action_layout.setHorizontalSpacing(8)
        action_layout.setVerticalSpacing(0)
        action_layout.setRowMinimumHeight(1, 14)
        for column in range(5):
            action_layout.setColumnStretch(column, 1)
            action_layout.setColumnMinimumWidth(column, PANEL_BUTTON_SIZE * 3)

        self.export_button = QPushButton(self.tr("Export"))
        self.export_button.setStyleSheet(
            get_dialog_button_style("primary", "medium")
        )
        self.export_button.setToolTip(
            self.tr("Export classified images to folders by category")
        )
        self.export_button.clicked.connect(self.export_images)

        self.multiclass_button = QPushButton(self.tr("MultiClass"))
        self.multiclass_button.setStyleSheet(
            get_dialog_button_style("light_green", "medium")
        )
        self.multiclass_button.setToolTip(
            self.tr("Single-label classification mode")
        )
        self.multiclass_button.clicked.connect(self.switch_to_multiclass)

        self.multilabel_button = QPushButton(self.tr("MultiLabel"))
        self.multilabel_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        self.multilabel_button.setToolTip(
            self.tr("Multi-label classification mode")
        )
        self.multilabel_button.clicked.connect(self.switch_to_multilabel)

        self.multitask_button = QPushButton(self.tr("MultiTask"))
        self.multitask_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        self.multitask_button.clicked.connect(self.switch_to_multitask)

        self.auto_run_button = QPushButton(self.tr("AutoRun"))
        self.auto_run_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        self.auto_run_button.setToolTip(
            self.tr("Use AI to automatically classify all images in batch")
        )
        self.auto_run_button.clicked.connect(self.auto_run_batch)

        action_layout.addWidget(self.export_button, 0, 0)
        action_layout.addWidget(self.multiclass_button, 0, 1)
        action_layout.addWidget(self.multilabel_button, 0, 2)
        action_layout.addWidget(self.multitask_button, 0, 3)
        action_layout.addWidget(self.auto_run_button, 0, 4)

        right_layout.addWidget(action_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameStyle(QFrame.Shape.NoFrame)
        self.scroll_area.verticalScrollBar().setStyleSheet(get_dock_style())

        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll_layout.setSpacing(15)

        self.category_row = QWidget()
        category_row_layout = QHBoxLayout(self.category_row)
        category_row_layout.setContentsMargins(0, 0, 0, 0)
        category_row_layout.setSpacing(8)
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)
        title_layout.addStretch()

        title_label = QLabel(self.tr("Category"))
        self.category_title = title_label
        title_label.setStyleSheet(get_filename_label_style())
        title_label.setToolTip(
            self.tr("Use number keys (0-9) to quickly select categories")
        )
        category_row_layout.addWidget(title_label)
        self.category_search = QLineEdit()
        self.category_search.setPlaceholderText(
            self.tr("Search categories...")
        )
        self.category_search.setClearButtonEnabled(True)
        self.category_search.setStyleSheet(get_dock_style())
        self.category_search.textChanged.connect(self.filter_categories)
        category_row_layout.addWidget(self.category_search, 1)
        self.object_row = QWidget()
        object_row_layout = QHBoxLayout(self.object_row)
        object_row_layout.setContentsMargins(0, 0, 0, 0)
        object_row_layout.setSpacing(8)
        self.object_label = QLabel(self.tr("Object"))
        self.object_label.setStyleSheet(get_filename_label_style())
        object_row_layout.addWidget(self.object_label)
        self.object_combo = ObjectComboBox()
        combo_height = self.category_search.sizeHint().height()
        configure_object_combo(self.object_combo, combo_height)
        self.object_combo.setMaximumWidth(280)
        self.object_combo.currentIndexChanged.connect(
            lambda index: self.select_object(index - 1)
        )
        self.object_combo.activated.connect(
            self.image_label.focus_selected_object
        )
        object_row_layout.addWidget(self.object_combo, 1)
        object_row_layout.addStretch()

        self.ai_button = QPushButton()
        self.ai_button.setIcon(_new_panel_icon("wand"))
        self.ai_button.setIconSize(QSize(PANEL_ICON_SIZE, PANEL_ICON_SIZE))
        self.ai_button.setFixedSize(24, 24)
        self.ai_button.setStyleSheet(get_panel_button_style())
        self.ai_button.setToolTip(self.tr("AI Assistant"))
        self.ai_button.clicked.connect(self.ai_classify_current)
        title_layout.addWidget(self.ai_button)

        self.new_button = QPushButton()
        self.new_button.setIcon(_new_panel_icon("new"))
        self.new_button.setIconSize(QSize(PANEL_ICON_SIZE, PANEL_ICON_SIZE))
        self.new_button.setFixedSize(24, 24)
        self.new_button.setStyleSheet(get_panel_button_style())
        self.new_button.setToolTip(self.tr("Add Label"))
        self.new_button.clicked.connect(self.add_new_label)
        title_layout.addWidget(self.new_button)

        self.delete_button = QPushButton()
        self.delete_button.setIcon(_new_panel_icon("minus"))
        self.delete_button.setIconSize(QSize(PANEL_ICON_SIZE, PANEL_ICON_SIZE))
        self.delete_button.setFixedSize(24, 24)
        self.delete_button.setStyleSheet(get_panel_button_style())
        self.delete_button.setToolTip(self.tr("Delete Label"))
        self.delete_button.clicked.connect(self.delete_label)
        title_layout.addWidget(self.delete_button)

        self.edit_button = QPushButton()
        self.edit_button.setIcon(_new_panel_icon("edit"))
        self.edit_button.setIconSize(QSize(PANEL_ICON_SIZE, PANEL_ICON_SIZE))
        self.edit_button.setFixedSize(24, 24)
        self.edit_button.setStyleSheet(get_panel_button_style())
        self.edit_button.setToolTip(self.tr("Edit Label"))
        self.edit_button.clicked.connect(self.edit_label)
        title_layout.addWidget(self.edit_button)

        self.view_button = QPushButton()
        self.view_button.setIcon(_new_panel_icon("view"))
        self.view_button.setIconSize(QSize(PANEL_ICON_SIZE, PANEL_ICON_SIZE))
        self.view_button.setFixedSize(24, 24)
        self.view_button.setStyleSheet(get_panel_button_style())
        self.view_button.setToolTip(self.tr("View Statistics"))
        self.view_button.clicked.connect(self.view_statistics)
        title_layout.addWidget(self.view_button)

        action_layout.addWidget(self.category_row, 2, 0, 1, 3)
        action_layout.addWidget(title_widget, 2, 3, 1, 2)
        self.category_tools = title_widget
        self.object_tools = QWidget()
        object_tools_layout = QHBoxLayout(self.object_tools)
        object_tools_layout.setContentsMargins(0, 0, 0, 0)
        object_tools_layout.setSpacing(0)
        self.previous_object_button = QPushButton()
        self.previous_object_button.setIcon(_object_navigation_icon(True))
        self.next_object_button = QPushButton()
        self.next_object_button.setIcon(_object_navigation_icon(False))
        self.multitask_ai_button = QPushButton()
        self.multitask_ai_button.setIcon(_new_panel_icon("wand", "text"))
        self.multitask_ai_button.setToolTip(self.tr("AI Assistant"))
        self.multitask_ai_button.clicked.connect(self.ai_multitask_current)
        self.upload_attributes_button = QPushButton()
        self.upload_attributes_button.setIcon(
            _new_panel_icon("upload", "text")
        )
        self.multitask_help_button = QPushButton()
        self.multitask_help_button.setIcon(new_icon("book-open", "svg"))
        self.multitask_help_button.setToolTip(self.tr("User Manual"))
        self.multitask_help_button.clicked.connect(self.open_multitask_help)
        for button in (
            self.previous_object_button,
            self.next_object_button,
            self.multitask_ai_button,
            self.upload_attributes_button,
            self.multitask_help_button,
        ):
            button.setIconSize(QSize(PANEL_ICON_SIZE, PANEL_ICON_SIZE))
            button.setFixedSize(24, 24)
            button.setStyleSheet(get_panel_button_style())
            object_tools_layout.addWidget(button)
        self.previous_object_button.setToolTip(
            self.tr("Previous object (Shift+A)")
        )
        self.next_object_button.setToolTip(self.tr("Next object (Shift+D)"))
        self.upload_attributes_button.setToolTip(self.tr("Load Attributes"))
        self.previous_object_button.clicked.connect(
            lambda: self.move_object(-1)
        )
        self.next_object_button.clicked.connect(lambda: self.move_object(1))
        self.upload_attributes_button.clicked.connect(self.load_attributes)
        object_row_layout.addWidget(self.object_tools)
        action_layout.addWidget(self.object_row, 2, 0, 1, 5)
        self.object_row.hide()
        self.multitask_form = MultiTaskForm(self.label_widget, combo_height)
        self.scroll_layout.addWidget(self.multitask_form, 1)
        self.multitask_form.hide()

        self.checkbox_group = None
        self.scroll_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_widget)
        right_layout.addWidget(self.scroll_area, 1)

        nav_widget = QWidget()
        nav_widget.setFixedHeight(DEFAULT_COMPONENT_HEIGHT)
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(10, 0, 10, 0)
        nav_layout.setSpacing(8)

        self.prev_button = QPushButton()
        self.prev_button.setIcon(QIcon(new_icon("arrow-left", "svg")))
        self.prev_button.setFixedSize(*ICON_SIZE_NORMAL)
        self.prev_button.setStyleSheet(get_button_style())
        self.prev_button.setToolTip(
            self.tr("Previous image (A) | Previous unlabeled image (Ctrl+A)")
        )
        self.prev_button.clicked.connect(self.prev_image)

        self.page_input = PageInputLineEdit()
        self.page_input.classifier_dialog = self
        self.page_input.setFixedSize(68, DEFAULT_COMPONENT_HEIGHT)
        self.page_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_input.setStyleSheet(get_page_input_style())
        self.page_input.textChanged.connect(self.validate_page_input)
        self.page_input.editingFinished.connect(self.on_page_input_finished)

        self.next_button = QPushButton()
        self.next_button.setIcon(QIcon(new_icon("arrow-right", "svg")))
        self.next_button.setFixedSize(*ICON_SIZE_NORMAL)
        self.next_button.setStyleSheet(get_button_style())
        self.next_button.setToolTip(
            self.tr("Next image (D) | Next unlabeled image (Ctrl+D)")
        )
        self.next_button.clicked.connect(self.next_image)

        nav_layout.addStretch()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.page_input)
        nav_layout.addWidget(self.next_button)
        nav_layout.addStretch()

        right_layout.addWidget(nav_widget)

        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(right_widget)

        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 1)
        self.target_left_width = PANEL_SIZE

        main_layout.addWidget(self.main_splitter)

        self.update_navigation_state()

    def setup_shortcuts(self):
        self.classification_shortcuts = []
        prev_shortcut = QShortcut(QKeySequence("A"), self)
        prev_shortcut.activated.connect(self.prev_image)

        next_shortcut = QShortcut(QKeySequence("D"), self)
        next_shortcut.activated.connect(self.next_image)

        prev_unlabeled_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)
        prev_unlabeled_shortcut.activated.connect(self.prev_unlabeled_image)

        next_unlabeled_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        next_unlabeled_shortcut.activated.connect(self.next_unlabeled_image)

        self.classification_shortcuts.extend(
            [
                prev_shortcut,
                next_shortcut,
                prev_unlabeled_shortcut,
                next_unlabeled_shortcut,
            ]
        )
        self.image_shortcuts = [prev_shortcut, next_shortcut]
        shortcuts = getattr(self.label_widget, "_config", {}).get(
            "shortcuts", {}
        )
        self.fit_shortcuts = []
        for key, mode, button, title in (
            (
                shortcuts.get("fit_window", "Ctrl+F"),
                "window",
                self.fit_window_button,
                self.tr("Fit Window"),
            ),
            (
                shortcuts.get("fit_width", "Ctrl+Shift+F"),
                "width",
                self.fit_width_button,
                self.tr("Fit Width"),
            ),
        ):
            sequence = QKeySequence(key)
            button.setToolTip(
                f"{title} ({sequence.toString(QKeySequence.SequenceFormat.NativeText)})"
            )
            shortcut = QShortcut(sequence, self)
            shortcut.activated.connect(
                lambda selected=mode: self.reset_preview_fit(selected)
            )
            shortcut.setEnabled(False)
            self.fit_shortcuts.append(shortcut)
        self.object_shortcuts = []
        for key, direction in (("Shift+A", -1), ("Shift+D", 1)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(
                lambda step=direction: self.move_object(step)
            )
            shortcut.setEnabled(False)
            self.object_shortcuts.append(shortcut)
        QApplication.instance().focusChanged.connect(
            self.update_shortcut_focus
        )
        for i in range(10):
            shortcut = QShortcut(QKeySequence(str(i)), self)
            shortcut.activated.connect(
                lambda idx=i: self.select_by_number(idx)
            )
            self.classification_shortcuts.append(shortcut)

    def update_shortcut_focus(self, _old, current):
        if not self.is_multitask:
            return
        editing = isinstance(current, (QLineEdit, QPlainTextEdit))
        for shortcut in self.image_shortcuts + self.object_shortcuts:
            shortcut.setEnabled(not editing)

    def move_object(self, step):
        count = self.object_combo.count() - 1
        if not count:
            return
        current = self.object_combo.currentIndex()
        if current == 0:
            index = 1 if step > 0 else count
        else:
            index = (current - 1 + step) % count + 1
        self.object_combo.setCurrentIndex(index)
        self.image_label.focus_selected_object()

    def select_by_number(self, number):
        if (
            not self.is_multitask
            and self.checkbox_group
            and number < len(self.labels)
        ):
            label = self.labels[number]
            if label in self.checkbox_group.checkboxes:
                checkbox = self.checkbox_group.checkboxes[label]
                if self.is_multiclass:
                    if checkbox.isChecked():
                        checkbox.setChecked(False)
                    else:
                        checkbox.setChecked(True)
                else:
                    checkbox.setChecked(not checkbox.isChecked())
                self.save_current_flags()

    def load_initial_data(self):
        if self.label_widget.image_list:
            self.image_files = self.label_widget.image_list
            self.update_image_display()
            self.update_navigation_state()
            self.page_input.setValidator(
                QIntValidator(1, len(self.image_files))
            )

        # Reset flags from current data
        self.labels = []
        if self.label_widget.filename:
            flag_widget = self.label_widget.flag_widget
            self.labels = [
                flag_widget.item(i).text() for i in range(flag_widget.count())
            ]

        if self.labels:
            self.label_widget.image_flags = self.labels[:]
            self.create_checkbox_group()

        self.load_current_flags()

    def create_checkbox_group(self):
        if self.checkbox_group:
            try:
                self.scroll_layout.removeWidget(self.checkbox_group)
                self.checkbox_group.deleteLater()
            except RuntimeError:
                pass
            self.checkbox_group = None

        if not self.labels:
            return

        self.checkbox_group = ClassificationCheckBoxGroup(
            self.labels, self.is_multiclass, self
        )

        for checkbox in self.checkbox_group.checkboxes.values():
            checkbox.toggled.connect(self.on_checkbox_changed)

        stretch_item = None
        if self.scroll_layout.count() > 0:
            stretch_item = self.scroll_layout.itemAt(
                self.scroll_layout.count() - 1
            )
            if stretch_item.spacerItem():
                self.scroll_layout.removeItem(stretch_item)

        self.scroll_layout.addWidget(self.checkbox_group)

        if stretch_item:
            self.scroll_layout.addStretch()

        self.filter_categories(self.category_search.text())

    def filter_categories(self, text):
        if not self.checkbox_group:
            return

        terms = text.casefold().split()
        for label, checkbox in self.checkbox_group.checkboxes.items():
            checkbox.setVisible(
                all(term in label.casefold() for term in terms)
            )
        self.scroll_area.verticalScrollBar().setValue(0)

    def reconnect_save_signals(self):
        if self.checkbox_group:
            for checkbox in self.checkbox_group.checkboxes.values():
                checkbox.toggled.connect(self.on_checkbox_changed)

    def on_checkbox_changed(self):
        if not self.switching_image and self.checkbox_group:
            if self.is_multiclass:
                sender = self.sender()
                if sender and sender.isChecked():
                    for (
                        label,
                        checkbox,
                    ) in self.checkbox_group.checkboxes.items():
                        if checkbox != sender and checkbox.isChecked():
                            checkbox.blockSignals(True)
                            checkbox.setChecked(False)
                            checkbox.blockSignals(False)
            self.save_current_flags()
        self.update_filename_label()

    def update_filename_label(self):
        if self.label_widget.filename:
            filename = os.path.basename(self.label_widget.filename)
            status_icon = ""
            if not self.is_multitask:
                status_icon = (
                    " ✅"
                    if self.is_image_labeled(self.label_widget.filename)
                    else " ❌"
                )

            if (
                self.image_files
                and self.label_widget.filename in self.image_files
            ):
                current_index = self.image_files.index(
                    self.label_widget.filename
                )
                total_count = len(self.image_files)
                self.filename_label.setText(
                    f"{filename} ({current_index + 1}/{total_count}){status_icon}"
                )
            else:
                self.filename_label.setText(f"{filename}{status_icon}")

    def add_new_label(self):
        dialog = NewLabelDialog(self.labels, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_labels = dialog.get_labels()
            self.labels.extend(new_labels)
            self.label_widget.image_flags.extend(new_labels)

            if hasattr(self.label_widget, "flag_widget"):
                self.ensure_labels_in_flag_widget(new_labels)

            self.create_checkbox_group()
            if self.label_widget.filename:
                self.load_current_flags()
            self.update_label_files()
            self.update_filename_label()

    def delete_label(self):
        if not self.labels:
            return

        dialog = DeleteLabelDialog(self.labels, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_labels = dialog.get_selected_labels()
            for label in selected_labels:
                self.labels.remove(label)
                self.label_widget.image_flags.remove(label)
            self.update_label_files()
            self.update_filename_label()

    def edit_label(self):
        if not self.labels:
            return

        dialog = EditLabelDialog(self.labels, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            labels_map = dialog.get_edit_info()

            new_labels = []
            for old_label in self.labels:
                new_labels.append(labels_map[old_label])

            self.labels = new_labels
            self.label_widget.image_flags = new_labels[:]

            reversed_labels_map = {v: k for k, v in labels_map.items()}
            self.update_label_files(reversed_labels_map)
            if hasattr(self.label_widget, "flag_widget"):
                self.update_main_flag_widget_labels(labels_map)
            self.create_checkbox_group()
            if self.label_widget.filename:
                self.load_current_flags()
            self.update_filename_label()

    def view_statistics(self):
        if not self.labels:
            QMessageBox.information(
                self, self.tr("Info"), self.tr("Please set labels first.")
            )
            return

        if not self.image_files:
            QMessageBox.information(
                self,
                self.tr("Info"),
                self.tr("No images loaded for statistics."),
            )
            return

        dialog = StatisticsViewDialog(
            self.labels, self.image_files, self.label_widget.output_dir, self
        )
        dialog.exec()

    def switch_to_multiclass(self):
        if self.is_multitask:
            self.set_multitask_mode(False)
        if self.is_multiclass:
            return

        reply = QMessageBox.question(
            self,
            self.tr("Switch Mode"),
            self.tr(
                "Switching to MultiClass mode will only keep the first label for each image. "
                "Other labels will be discarded. Continue?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.is_multiclass = True
            self.multiclass_button.setStyleSheet(
                get_dialog_button_style("light_green", "medium")
            )
            self.multilabel_button.setStyleSheet(
                get_dialog_button_style("secondary", "medium")
            )
            self.create_checkbox_group()
            self.load_current_flags()

    def switch_to_multilabel(self):
        if self.is_multitask:
            self.set_multitask_mode(False)
        if not self.is_multiclass:
            return

        self.is_multiclass = False
        self.multilabel_button.setStyleSheet(
            get_dialog_button_style("light_green", "medium")
        )
        self.multiclass_button.setStyleSheet(
            get_dialog_button_style("secondary", "medium")
        )
        self.create_checkbox_group()
        self.load_current_flags()

    def toggle_object_labels(self, visible):
        self.image_label.show_object_labels = visible
        self.image_label.update()

    def toggle_background_mask(self, enabled):
        self.image_label.mask_background = enabled
        self.image_label.update()

    def set_preview_fit(self, mode):
        self.image_label.set_fit_mode(mode)
        self.fit_window_button.setChecked(mode == "window")
        self.fit_width_button.setChecked(mode == "width")

    def reset_preview_fit(self, mode):
        self.select_object(-1)
        self.set_preview_fit(mode)

    def switch_to_multitask(self):
        if not self.is_multitask:
            self.set_multitask_mode(True)

    def set_multitask_mode(self, enabled):
        self.is_multitask = enabled
        for shortcut in self.classification_shortcuts:
            shortcut.setEnabled(not enabled)
        for shortcut in self.image_shortcuts + self.object_shortcuts:
            shortcut.setEnabled(
                enabled
                and not isinstance(
                    QApplication.focusWidget(), (QLineEdit, QPlainTextEdit)
                )
            )
        if not enabled:
            for shortcut in self.image_shortcuts:
                shortcut.setEnabled(True)
        self.preview_controls.setVisible(enabled)
        self.preview_title_spacer.setVisible(enabled)
        for shortcut in self.fit_shortcuts:
            shortcut.setEnabled(enabled)
        self.image_label.multitask_enabled = enabled
        self.set_preview_fit("window")
        self.update_image_display()
        self.export_button.setEnabled(not enabled)
        self.auto_run_button.setEnabled(True)
        self.auto_run_button.setToolTip(
            self.tr("Use AI to process objects in all images")
            if enabled
            else self.tr(
                "Use AI to automatically classify all images in batch"
            )
        )
        self.category_title.setText(
            self.tr("Object") if enabled else self.tr("Category")
        )
        self.category_row.setVisible(not enabled)
        self.category_tools.setVisible(not enabled)
        self.object_row.setVisible(enabled)
        self.multitask_form.setVisible(enabled)
        if self.checkbox_group:
            self.checkbox_group.setVisible(not enabled)
        self.multitask_button.setStyleSheet(
            get_dialog_button_style(
                "light_green" if enabled else "secondary", "medium"
            )
        )
        self.multiclass_button.setStyleSheet(
            get_dialog_button_style(
                (
                    "light_green"
                    if not enabled and self.is_multiclass
                    else "secondary"
                ),
                "medium",
            )
        )
        self.multilabel_button.setStyleSheet(
            get_dialog_button_style(
                (
                    "light_green"
                    if not enabled and not self.is_multiclass
                    else "secondary"
                ),
                "medium",
            )
        )
        if enabled:
            self.overlay_label.hide()
            self.refresh_objects()
        else:
            self.image_label.set_objects([], QSize())
            self.update_overlay_text()

    def refresh_objects(self):
        shapes = self.label_widget.canvas.shapes
        self.object_combo.blockSignals(True)
        self.object_combo.clear()
        self.object_combo.addItem(self.tr("All objects"))
        for shape in shapes:
            group = (
                f" ({shape.group_id})" if shape.group_id is not None else ""
            )
            self.object_combo.addItem(f"{shape.label}{group}")
        self.object_combo.blockSignals(False)
        self.object_combo.setEnabled(bool(shapes))
        self.select_object(-1)

    def select_object(self, index):
        if not self.is_multitask:
            return
        shapes = self.label_widget.canvas.shapes
        if index < 0 or index >= len(shapes):
            index = -1
        self.object_combo.blockSignals(True)
        self.object_combo.setCurrentIndex(index + 1)
        self.object_combo.blockSignals(False)
        self.object_label.setText(
            f"{self.tr('Object')} ({index + 1}/{len(shapes)})"
        )
        self.previous_object_button.setEnabled(bool(shapes))
        self.next_object_button.setEnabled(bool(shapes))
        targets = [shapes[index]] if index >= 0 else shapes
        self.multitask_ai_button.setEnabled(
            bool(self.label_widget.filename)
            and any(self.label_widget.attributes.get(s.label) for s in targets)
        )
        self.multitask_ai_button.setToolTip(
            self.tr("AI Assistant")
            if self.multitask_ai_button.isEnabled()
            else self.tr("Select an object with configured attributes first.")
        )
        self.multitask_form.show_object(index)
        self.image_label.set_objects(
            shapes, self.label_widget.image.size(), index
        )

    def load_attributes(self):
        self.label_widget.actions.upload_shape_attrs_file.trigger()
        self.select_object(self.object_combo.currentIndex() - 1)

    def start_multitask_batch(self, all_images):
        paths = (
            self.select_batch_image_range()
            if all_images
            else [self.label_widget.filename]
        )
        if paths is None:
            return
        paths = [path for path in paths if path]
        if not paths:
            QMessageBox.warning(
                self, self.tr("Warning"), self.tr("No images loaded!")
            )
            return
        prompt = (
            "@image\nDescribe the selected object in this crop. "
            "Return only a JSON object with the specified attribute names. "
            "For choice attributes use exactly one listed option; for text attributes use a string. "
            "Use null when uncertain. Do not infer information that is not visible.\n"
            "Object category: @object\nAttributes: @attributes"
        )
        dialog = MultiTaskAIDialog(self, prompt, batch=True)
        if (
            dialog.exec() != QDialog.DialogCode.Accepted
            or not dialog.get_prompt()
        ):
            return
        self._multitask_batch_errors = []
        self._multitask_batch = MultiTaskBatch(
            self, paths, dialog.get_prompt(), dialog.concurrency.value()
        )
        self._multitask_batch.start()

    def ai_multitask_current(self):
        index = self.object_combo.currentIndex() - 1
        shapes = self.label_widget.canvas.shapes
        if index < 0:
            self.start_multitask_batch(False)
            return
        if not 0 <= index < len(shapes) or not self.label_widget.filename:
            return
        shape = shapes[index]
        properties = self.label_widget.attributes.get(shape.label, {})
        if not properties:
            return
        types = self.label_widget.attribute_widget_types.get(shape.label, {})
        schema = {}
        for name, options in properties.items():
            kind = types.get(name, "combobox")
            if kind == "group_id":
                continue
            schema[name] = "text" if kind == "lineedit" else list(options)
        if not schema:
            QMessageBox.information(
                self,
                self.tr("AI Assistant"),
                self.tr("No supported attributes available."),
            )
            return
        bounds = self.image_label.object_path(shape).boundingRect()
        margin = max(bounds.width(), bounds.height(), 20) * 0.1
        rect = bounds.adjusted(
            -margin, -margin, margin, margin
        ).toAlignedRect()
        rect = rect.intersected(self.label_widget.image.rect())
        if rect.isEmpty():
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Cannot crop the selected object."),
            )
            return
        prompt = (
            "@image\nDescribe the selected object in this crop. "
            "Return only a JSON object with the specified attribute names. "
            "For choice attributes use exactly one listed option; for text attributes use a string. "
            "Use null when uncertain. Do not infer information that is not visible.\n"
            f"Object category: {shape.label}\n"
            f"Attributes: {json.dumps(schema, ensure_ascii=False)}"
        )
        dialog = MultiTaskAIDialog(self, prompt)
        dialog.adjustSize()
        anchor = self.multitask_ai_button.mapToGlobal(
            QPoint(
                self.multitask_ai_button.width(),
                self.multitask_ai_button.height(),
            )
        )
        screen = QApplication.screenAt(anchor) or self.screen()
        if screen:
            bounds = screen.availableGeometry()
            dialog.move(
                max(
                    bounds.left(),
                    min(
                        anchor.x() - dialog.width(),
                        bounds.right() - dialog.width() + 1,
                    ),
                ),
                max(
                    bounds.top(),
                    min(anchor.y(), bounds.bottom() - dialog.height() + 1),
                ),
            )
        if (
            dialog.exec() != QDialog.DialogCode.Accepted
            or not dialog.get_prompt()
        ):
            return
        crop = QTemporaryFile()
        if not crop.open():
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Cannot crop the selected object."),
            )
            return
        path = crop.fileName()
        crop.close()
        if not self.label_widget.image.copy(rect).save(path, "JPEG"):
            QMessageBox.warning(
                self,
                self.tr("Error"),
                self.tr("Cannot crop the selected object."),
            )
            return
        self.loading_msg = AILoadingDialog(self)
        worker = AIWorkerThread(dialog.get_prompt(), "", {}, path, [], None)
        worker.crop_file = crop
        filename = self.label_widget.filename
        worker.finished.connect(
            lambda result, success, error: self.handle_multitask_ai_result(
                result, success, error, filename, shape, schema
            )
        )
        self.ai_worker = worker
        self._ai_workers.add(worker)
        worker.completed.connect(self._on_ai_worker_completed)
        self.loading_msg.cancel_button.clicked.connect(
            self.cancel_ai_processing
        )
        worker.start()
        if self.loading_msg.exec() == QDialog.DialogCode.Rejected:
            self.cancel_ai_processing()

    def handle_multitask_ai_result(
        self, result, success, error, filename, shape, schema
    ):
        self.loading_msg.close()
        if not success:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                error or self.tr("AI classification failed"),
            )
            return
        if self.label_widget.filename != filename or not any(
            item is shape for item in self.label_widget.canvas.shapes
        ):
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("The selected object is no longer available."),
            )
            return
        try:
            match = re.search(r"```json\s*(.*?)\s*```", result, re.DOTALL)
            values = json.loads(match.group(1) if match else result)
            if not isinstance(values, dict):
                raise ValueError
        except (ValueError, TypeError):
            QMessageBox.warning(
                self, self.tr("Error"), self.tr("Failed to parse AI result")
            )
            return
        suggestions = {}
        skipped = []
        current_properties = self.label_widget.attributes.get(shape.label, {})
        for name, value in values.items():
            if value is None:
                continue
            options = schema.get(name)
            current_kind = self.label_widget.attribute_widget_types.get(
                shape.label, {}
            ).get(name, "combobox")
            valid = (
                name in current_properties
                and isinstance(value, str)
                and bool(value.strip())
            )
            if options == "text":
                valid = valid and current_kind == "lineedit"
            elif isinstance(options, list):
                valid = (
                    valid
                    and current_kind in ("combobox", "radiobutton")
                    and value in options
                    and value in current_properties[name]
                )
            else:
                valid = False
            if not valid:
                skipped.append(str(name))
            else:
                suggestions[name] = value
        if not suggestions:
            QMessageBox.information(
                self,
                self.tr("AI Assistant"),
                self.tr("No applicable suggestions.")
                + (
                    "\n"
                    + self.tr("Skipped attributes:")
                    + " "
                    + ", ".join(skipped)
                    if skipped
                    else ""
                ),
            )
            return
        for name, value in suggestions.items():
            self.multitask_form.set_value(shape, name, value)
        self.select_object(self.object_combo.currentIndex() - 1)
        if skipped:
            QMessageBox.information(
                self,
                self.tr("AI Assistant"),
                self.tr("Skipped attributes:") + " " + ", ".join(skipped),
            )

    def open_multitask_help(self):
        locale = (
            "/zh-Hans"
            if self.label_widget._config["language"] == "zh_CN"
            else ""
        )
        general.open_url(
            f"https://xanylabeling.com{locale}/examples/classification/shape-level"
        )

    def export_images(self):
        if not self.labels:
            QMessageBox.warning(
                self, self.tr("Warning"), self.tr("No labels configured!")
            )
            return

        if not self.is_multiclass:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Export only supports multi-class tasks!"),
            )
            return

        dialog = ExportPathDialog(self.output_dir, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.output_dir = dialog.get_path()

        if not self.image_files:
            QMessageBox.warning(
                self, self.tr("Warning"), self.tr("No images loaded!")
            )
            return

        exported_count = 0
        base_dir = (
            os.path.dirname(self.image_files[0]) if self.image_files else ""
        )
        output_path = os.path.join(base_dir, self.output_dir)

        for image_path in self.image_files:
            label_path = get_label_file_path(
                image_path, getattr(self.label_widget, "output_dir", None)
            )
            flags = load_flags_from_json(label_path)

            if flags:
                first_true = get_first_true_flag(flags)
                if first_true:
                    try:
                        export_image_to_category(
                            image_path, first_true, output_path
                        )
                    except ValueError as e:
                        logger.error(f"Error exporting image: {e}")
                        QMessageBox.warning(
                            self,
                            self.tr("Warning"),
                            self.tr("Invalid category name: %s") % first_true,
                        )
                        return
                    exported_count += 1

        template = self.tr("Exported %d images to %s")
        message_text = template % (exported_count, output_path)
        QMessageBox.information(self, self.tr("Success"), message_text)

    def ai_classify_current(self):
        if not self.labels:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please configure labels first!"),
            )
            return

        if not self.label_widget.filename:
            QMessageBox.warning(
                self, self.tr("Warning"), self.tr("No image loaded!")
            )
            return

        prompt = create_ai_prompt_template(self.labels, self.is_multiclass)
        dialog = AIPromptDialog(self)
        dialog.prompt_input.setPlainText(prompt)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            final_prompt = dialog.get_prompt()
            if final_prompt:
                self.loading_msg = AILoadingDialog(self)

                self.ai_worker = AIWorkerThread(
                    final_prompt,
                    "",
                    {},
                    self.label_widget.filename,
                    [],
                    self.label_widget,
                )
                self.ai_worker.finished.connect(self.handle_ai_result)
                self._ai_workers.add(self.ai_worker)
                self.ai_worker.completed.connect(self._on_ai_worker_completed)
                self.loading_msg.cancel_button.clicked.connect(
                    self.cancel_ai_processing
                )

                self.ai_worker.start()
                if self.loading_msg.exec() == QDialog.DialogCode.Rejected:
                    self.cancel_ai_processing()

    def handle_ai_result(self, result, success, error_message):
        if hasattr(self, "loading_msg"):
            self.loading_msg.close()

        if success and result:
            try:
                json_match = re.search(
                    r"```json\s*(.*?)\s*```", result, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = result

                flags = json.loads(json_str)
                if self.checkbox_group:
                    self.checkbox_group.set_flags(flags)
                    self.save_current_flags()
                    QMessageBox.information(
                        self,
                        self.tr("Success"),
                        self.tr("AI classification completed!"),
                    )
            except (json.JSONDecodeError, Exception):
                QMessageBox.warning(
                    self,
                    self.tr("Error"),
                    self.tr("Failed to parse AI result"),
                )
        else:
            QMessageBox.warning(
                self,
                self.tr("Error"),
                error_message or self.tr("AI classification failed"),
            )

    def cancel_ai_processing(self):
        if hasattr(self, "ai_worker") and self.ai_worker.isRunning():
            self.ai_worker.cancel()
        if hasattr(self, "loading_msg"):
            self.loading_msg.close()

    def _on_ai_worker_completed(self):
        worker = self.sender()
        if hasattr(worker, "crop_file"):
            worker.crop_file.remove()
        self._ai_workers.discard(worker)

    def auto_run_batch(self):
        if self.is_multitask:
            self.start_multitask_batch(True)
            return
        if not self.labels:
            QMessageBox.warning(
                self,
                self.tr("Warning"),
                self.tr("Please configure labels first!"),
            )
            return

        if not self.image_files:
            QMessageBox.warning(
                self, self.tr("Warning"), self.tr("No images loaded!")
            )
            return

        paths = self.select_batch_image_range()
        if paths is None:
            return

        prompt = create_ai_prompt_template(self.labels, self.is_multiclass)
        dialog = MultiTaskAIDialog(
            self, prompt, batch=True, classification=True
        )
        if (
            dialog.exec() != QDialog.DialogCode.Accepted
            or not dialog.get_prompt()
        ):
            return
        self._multitask_batch_errors = []
        self._multitask_batch = ClassificationBatch(
            self, paths, dialog.get_prompt(), dialog.concurrency.value()
        )
        self._multitask_batch.start()

    def select_batch_image_range(self):
        from anylabeling.views.labeling.utils.batch import BatchRangeDialog

        if not self.image_files:
            QMessageBox.warning(
                self, self.tr("Warning"), self.tr("No images loaded!")
            )
            return None
        try:
            start = self.image_files.index(self.label_widget.filename) + 1
        except ValueError:
            start = 1
        dialog = BatchRangeDialog(len(self.image_files), start, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return self.image_files[
            dialog.from_input.value() - 1 : dialog.to_input.value()
        ]

    def prev_image(self):
        if not self.image_files:
            return

        current_file = self.label_widget.filename
        if not current_file or current_file not in self.image_files:
            return

        current_index = self.image_files.index(current_file)
        if current_index > 0:
            self.switch_to_image(current_index - 1)

    def next_image(self):
        if not self.image_files:
            return

        current_file = self.label_widget.filename
        if not current_file or current_file not in self.image_files:
            return

        current_index = self.image_files.index(current_file)
        if current_index < len(self.image_files) - 1:
            self.switch_to_image(current_index + 1)

    def is_image_labeled(self, image_path):
        if not self.labels:
            return False

        label_path = get_label_file_path(
            image_path, self.label_widget.output_dir
        )
        flags = load_flags_from_json(label_path)
        if not flags:
            return False

        return any(flags.get(label, False) for label in self.labels)

    def prev_unlabeled_image(self):
        if not self.image_files or not self.labels:
            return

        current_file = self.label_widget.filename
        if not current_file or current_file not in self.image_files:
            return

        current_index = self.image_files.index(current_file)
        for i in range(current_index - 1, -1, -1):
            if not self.is_image_labeled(self.image_files[i]):
                self.switch_to_image(i)
                return

        for i in range(len(self.image_files) - 1, current_index, -1):
            if not self.is_image_labeled(self.image_files[i]):
                self.switch_to_image(i)
                return

        QMessageBox.information(
            self, self.tr("Info"), self.tr("No unlabeled images found.")
        )

    def next_unlabeled_image(self):
        if not self.image_files or not self.labels:
            return

        current_file = self.label_widget.filename
        if not current_file or current_file not in self.image_files:
            return

        current_index = self.image_files.index(current_file)
        for i in range(current_index + 1, len(self.image_files)):
            if not self.is_image_labeled(self.image_files[i]):
                self.switch_to_image(i)
                return

        for i in range(0, current_index):
            if not self.is_image_labeled(self.image_files[i]):
                self.switch_to_image(i)
                return

        QMessageBox.information(
            self, self.tr("Info"), self.tr("No unlabeled images found.")
        )

    def switch_to_image(self, index):
        if index < 0 or index >= len(self.image_files):
            return

        new_file = self.image_files[index]

        self.switching_image = True
        self.save_current_flags()
        self.label_widget.load_file(new_file)

        self.update_image_display()
        self.update_navigation_state()
        self.load_current_flags()
        if self.is_multitask:
            self.refresh_objects()

        self.switching_image = False

    def jump_to_page(self, page_num):
        if not self.image_files:
            return

        if page_num < 1 or page_num > len(self.image_files):
            return

        self.switch_to_image(page_num - 1)

    def validate_page_input(self, text):
        if not text:
            return

        try:
            if not text.isdigit():
                cursor_pos = self.page_input.cursorPosition()
                clean_text = "".join(c for c in text if c.isdigit())
                self.page_input.setText(clean_text)
                self.page_input.setCursorPosition(
                    min(cursor_pos, len(clean_text))
                )
                return

            page_num = int(text)
            max_pages = len(self.image_files) if self.image_files else 1
            if page_num > max_pages:
                self.page_input.setText(str(max_pages))
                self.page_input.setCursorPosition(len(str(max_pages)))

        except ValueError:
            pass

    def on_page_input_finished(self):
        text = self.page_input.text().strip()

        if not text:
            self.restore_current_page_number()
            return

        try:
            page_num = int(text)
            max_pages = len(self.image_files) if self.image_files else 1

            if page_num < 1:
                self.page_input.setText("1")
            elif page_num > max_pages:
                self.page_input.setText(str(max_pages))

        except ValueError:
            self.restore_current_page_number()

    def restore_current_page_number(self):
        if self.label_widget.filename and self.image_files:
            try:
                current_index = self.image_files.index(
                    self.label_widget.filename
                )
                self.page_input.setText(str(current_index + 1))
            except (ValueError, AttributeError):
                self.page_input.setText("1")
        else:
            self.page_input.setText("1")

    def update_image_display(self):
        if self.label_widget.filename != self._preview_filename:
            self._preview_filename = self.label_widget.filename
            self.set_preview_fit("window")
        checked = getattr(
            self.label_widget, "_annotation_checked", lambda: False
        )()
        self.status_buttons[checked].setChecked(True)
        for button in self.status_buttons.values():
            button.setEnabled(bool(self.label_widget.filename))
        if self.label_widget.filename:
            self.update_filename_label()

            if (
                self.image_files
                and self.label_widget.filename in self.image_files
            ):
                current_index = self.image_files.index(
                    self.label_widget.filename
                )
                self.page_input.setText(str(current_index + 1))
            else:
                self.page_input.setText("1")

            pixmap = QPixmap(self.label_widget.filename)
            if not pixmap.isNull():
                self.image_label.setPixmap(pixmap)

            self.update_overlay_text()
        else:
            self.filename_label.setText(self.tr("No image loaded"))
            self.image_label.clear()
            self.page_input.setText("0")
            self.overlay_label.hide()

    def update_overlay_text(self):
        if self.is_multitask:
            self.overlay_label.hide()
            return
        if self.checkbox_group:
            try:
                flags = self.checkbox_group.get_selected_flags()
                text = get_display_text_for_flags(flags, self.labels)
                self.overlay_label.update_text(text)
                self.overlay_label.position_overlay(self.image_label)
            except RuntimeError:
                pass

    def update_navigation_state(self):
        if self.image_files and self.label_widget.filename:
            try:
                current_index = self.image_files.index(
                    self.label_widget.filename
                )
                self.prev_button.setEnabled(current_index > 0)
                self.next_button.setEnabled(
                    current_index < len(self.image_files) - 1
                )
            except (ValueError, AttributeError):
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(False)
        else:
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)

    def load_current_flags(self):
        if not self.checkbox_group:
            return

        if self.label_widget.filename:
            flags = {}
            if hasattr(self.label_widget, "flag_widget") and self.labels:
                flag_widget = self.label_widget.flag_widget

                for label in self.labels:
                    found = False
                    for i in range(flag_widget.count()):
                        item = flag_widget.item(i)
                        if item.text() == label:
                            flags[label] = (
                                item.checkState() == Qt.CheckState.Checked
                            )
                            found = True
                            break
                    if not found:
                        flags[label] = False
            else:
                flags = {label: False for label in self.labels}

            self.switching_image = True
            try:
                self.checkbox_group.set_flags(flags)
                self.update_overlay_text()
            except RuntimeError:
                pass
            finally:
                self.switching_image = False

    def save_current_flags(self):
        if (
            self.is_multitask
            or self.switching_image
            or not self.checkbox_group
        ):
            return

        try:
            flags = self.checkbox_group.get_selected_flags()
        except RuntimeError:
            return

        if self.label_widget.filename:
            if hasattr(self.label_widget, "flags"):
                self.label_widget.flags = flags

            if hasattr(self.label_widget, "flag_widget"):
                self.sync_to_main_flag_widget(flags)

            if hasattr(self.label_widget, "set_dirty"):
                self.label_widget.set_dirty()

        self.update_overlay_text()

    def ensure_labels_in_flag_widget(self, new_labels):
        flag_widget = self.label_widget.flag_widget

        for label in new_labels:
            found = False
            for i in range(flag_widget.count()):
                item = flag_widget.item(i)
                if item.text() == label:
                    found = True
                    break

            if not found:
                item = QListWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                flag_widget.addItem(item)

    def sync_to_main_flag_widget(self, flags):
        try:
            flag_widget = self.label_widget.flag_widget

            for i in range(flag_widget.count()):
                item = flag_widget.item(i)
                key = item.text()
                if key in flags:
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if flags[key]
                        else Qt.CheckState.Unchecked
                    )
        except Exception as e:
            logger.error(f"Error syncing to flag_widget: {e}")

    def update_main_flag_widget_labels(self, labels_map):
        """更新主flag_widget中的标签名称"""
        try:
            flag_widget = self.label_widget.flag_widget

            for i in range(flag_widget.count()):
                item = flag_widget.item(i)
                current_label = item.text()
                if current_label in labels_map:
                    new_label = labels_map[current_label]
                    if new_label != current_label:
                        # 保存当前的勾选状态
                        check_state = item.checkState()
                        item.setText(new_label)
                        item.setCheckState(check_state)
        except Exception as e:
            logger.error(f"Error updating flag_widget labels: {e}")

    def _delayed_update_ui(self):
        self.create_checkbox_group()
        self.load_current_flags()
        self.update_filename_label()

    def update_label_files(self, labels_map={}):
        progress_dialog = QProgressDialog(
            self.tr("Updating label files..."),
            self.tr("Cancel"),
            0,
            len(self.image_files),
            self,
        )
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setWindowTitle(self.tr("Progress"))
        progress_dialog.setMinimumWidth(500)
        progress_dialog.setMinimumHeight(150)
        progress_dialog.setStyleSheet(
            get_progress_dialog_style(color="#1d1d1f", height=20)
        )

        try:
            for i, image_file in enumerate(self.image_files):
                label_path = get_label_file_path(
                    image_file, self.label_widget.output_dir
                )
                if not os.path.exists(label_path):
                    continue

                flags = load_flags_from_json(label_path)
                save_flags = {}
                for label in self.labels:
                    check_label = labels_map[label] if labels_map else label
                    save_flags[label] = flags.get(check_label, False) is True
                save_flags_to_json(label_path, save_flags)

                progress_dialog.setValue(i + 1)
                if progress_dialog.wasCanceled():
                    break

            progress_dialog.close()
            message = self.tr("Labels update successfully!")
            popup = Popup(
                message,
                self,
                icon=new_icon_path("copy-green", "svg"),
            )
            popup.show_popup(self, popup_height=65, position="center")

        except Exception as e:
            progress_dialog.close()
            template = self.tr("Error occurred while updating label files: %s")
            logger.error(template % str(e))
            popup = Popup(
                template % str(e),
                self,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self, position="center")

        finally:
            QTimer.singleShot(100, self._delayed_update_ui)
            if self.image_files:
                self.page_input.setValidator(
                    QIntValidator(1, len(self.image_files))
                )

    def changeEvent(self, event):
        super().changeEvent(event)
        if (
            event.type() == QEvent.Type.WindowStateChange
            and self.isMaximized()
        ):
            QTimer.singleShot(0, self.set_maximized_splitter_size)
        if (
            event.type() == QEvent.Type.ActivationChange
            and self.isActiveWindow()
            and hasattr(self, "page_input")
        ):
            self.image_files = self.label_widget.image_list
            self.update_image_display()
            self.update_navigation_state()
            self.load_current_flags()
            if self.is_multitask:
                selected = self.object_combo.currentIndex() - 1
                previous_shapes = getattr(
                    self.image_label, "object_shapes", []
                )
                shape = (
                    previous_shapes[selected]
                    if 0 <= selected < len(previous_shapes)
                    else None
                )
                self.refresh_objects()
                if shape in self.label_widget.canvas.shapes:
                    self.select_object(
                        self.label_widget.canvas.shapes.index(shape)
                    )

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(10, self.set_initial_splitter_size)
        QTimer.singleShot(100, self.update_image_display)

    def set_initial_splitter_size(self):
        if self.isMaximized():
            self.set_maximized_splitter_size()
            return
        total_width = self.main_splitter.width()
        right_width = total_width - self.target_left_width
        if right_width > 0:
            self.main_splitter.setSizes([self.target_left_width, right_width])

    def set_maximized_splitter_size(self):
        if self.isMaximized():
            right_width = (
                self.main_splitter.widget(1).minimumSizeHint().width()
            )
            left_width = self.main_splitter.width() - right_width
            self.main_splitter.moveSplitter(left_width, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
