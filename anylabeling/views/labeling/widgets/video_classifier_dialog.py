import base64
import json
import mimetypes
import os
import re
import subprocess
import tempfile
from copy import deepcopy

import requests

from PyQt6.QtCore import QEvent, QThread, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QDesktopServices,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QScrollBar,
    QSlider,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.chatbot.config import (
    get_models_config_path,
    get_providers_config_path,
)
from anylabeling.views.labeling.utils.style import (
    get_cancel_btn_style,
    get_ok_btn_style,
    get_progress_dialog_style,
    get_settings_combo_style,
)
from anylabeling.views.labeling.utils.theme import get_theme

from anylabeling.views.labeling.video_classifier.config import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_TITLE,
    SUPPORTED_VIDEO_EXTS,
    SUPPORTED_VIDEO_FILTER,
)
from anylabeling.views.labeling.video_classifier.drop_zone import DropZone
from anylabeling.views.labeling.video_classifier.export_dialog import (
    ExportDialog,
)
from anylabeling.views.labeling.video_classifier.exporter import (
    ExporterController,
)
from anylabeling.views.labeling.video_classifier.label_panel import (
    LabelNameDialog,
    LabelPanel,
)
from anylabeling.views.labeling.video_classifier.player import VideoPlayer
from anylabeling.views.labeling.video_classifier.segment_list import (
    SegmentListPanel,
)
from anylabeling.views.labeling.video_classifier.sidecar import (
    Segment,
    SidecarData,
    load_sidecar,
    save_sidecar,
)
from anylabeling.views.labeling.video_classifier.icons import (
    apply_button_icon,
    theme_icon_color,
)
from anylabeling.views.labeling.video_classifier.style import (
    get_dialog_style,
    get_icon_button_style,
    get_panel_frame_style,
    get_slider_style,
    get_timeline_frame_style,
)
from anylabeling.views.labeling.video_classifier.timeline import TimelineWidget
from anylabeling.views.labeling.video_classifier.utils import (
    color_for_label,
    detect_ffmpeg,
    extract_video_thumbnails,
    ms_to_seconds,
    ms_to_timecode,
    probe_video,
)

VIDEO_DESCRIPTION_PROMPT = "Describe this video."
VIDEO_SEGMENTATION_PROMPT = """请你描述视频中的人物的一系列动作，并以 JSON 格式输出开始时间、结束时间和事件描述。
请只返回 JSON，不要返回 Markdown 或额外解释。
时间戳请使用 HH:mm:ss 格式。

示例输出：
{
  "events": [
    {
      "start_time": "00:00:00",
      "end_time": "00:00:05",
      "event": "人物手持一个纸箱走向桌子，并将纸箱放在桌上。"
    },
    {
      "start_time": "00:00:05",
      "end_time": "00:00:15",
      "event": "人物拿起扫描枪，对准纸箱上的标签进行扫描。"
    }
  ]
}"""
VIDEO_SEGMENTATION_PROMPT_EN = """Describe the sequence of actions performed by the people in the video, and output the start time, end time, and event description in JSON format.
Return JSON only. Do not return Markdown or any extra explanation.
Use HH:mm:ss timestamps.

Example output:
{
  "events": [
    {
      "start_time": "00:00:00",
      "end_time": "00:00:05",
      "event": "A person walks toward the table holding a cardboard box and places the box on the table."
    },
    {
      "start_time": "00:00:05",
      "end_time": "00:00:15",
      "event": "The person picks up a scanner and scans the label on the cardboard box."
    }
  ]
}"""
REQUEST_TIMEOUT = 120
AI_CANCELLED = "__cancelled__"


def _vertical_scrollbar_style():
    t = get_theme()
    return f"""
        QScrollBar:vertical {{
            background: {t["background_secondary"]};
            width: 12px;
            margin: 12px 0px 12px 0px;
            border: none;
        }}
        QScrollBar:vertical:disabled {{
            background: transparent;
        }}
        QScrollBar::handle:vertical {{
            background: {t["scrollbar"]};
            min-height: 34px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {t["scrollbar_hover"]};
        }}
        QScrollBar::handle:vertical:disabled {{
            background: transparent;
        }}
        QScrollBar::add-line:vertical {{
            background: {t["background_secondary"]};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: bottom;
            height: 12px;
            image: url(:/images/images/caret-down.svg);
        }}
        QScrollBar::sub-line:vertical {{
            background: {t["background_secondary"]};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: top;
            height: 12px;
            image: url(:/images/images/caret-up.svg);
        }}
        QScrollBar::add-line:vertical:disabled,
        QScrollBar::sub-line:vertical:disabled {{
            background: transparent;
            image: none;
        }}
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
    """


class VideoDescriptionWorker(QThread):
    resultReady = pyqtSignal(str, bool, str)
    progressChanged = pyqtSignal(str)

    def __init__(self, video_path, segment, prompt, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.segment = deepcopy(segment) if segment else None
        self.prompt = prompt
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        temp_path = ""
        try:
            models_config = self._load_json(get_models_config_path())
            providers_config = self._load_json(get_providers_config_path())
            if not models_config or not providers_config:
                self.resultReady.emit(
                    "", False, "Configuration files not found"
                )
                return

            settings = models_config.get("settings", {})
            provider = settings.get("provider")
            model_id = settings.get("model_id")
            if not provider or not model_id:
                self.resultReady.emit(
                    "",
                    False,
                    "Please configure model and provider in Chatbot (Ctrl+0)",
                )
                return

            provider_info = providers_config.get(provider, {})
            api_address = provider_info.get("api_address")
            api_key = provider_info.get("api_key")
            if not api_address or not api_key:
                self.resultReady.emit(
                    "",
                    False,
                    f"Please configure API key for {provider} in Chatbot (Ctrl+0)",
                )
                return

            video_url, temp_path = self._video_url()
            if self._cancelled:
                self.resultReady.emit("", False, AI_CANCELLED)
                return

            result = self._call_openai_api(
                api_address,
                api_key,
                model_id,
                settings,
                video_url,
            )
            if self._cancelled:
                self.resultReady.emit("", False, AI_CANCELLED)
                return
            self.resultReady.emit(result, True, "")
        except Exception as exc:
            if self._cancelled:
                self.resultReady.emit("", False, AI_CANCELLED)
            else:
                self.resultReady.emit("", False, f"API call failed: {exc}")
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _load_json(self, path):
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _video_url(self):
        source_path = self.video_path
        temp_path = ""
        if self.segment:
            self.progressChanged.emit("Preparing selected segment...")
            temp_path = self._extract_segment()
            if temp_path:
                source_path = temp_path

        self.progressChanged.emit("Encoding video...")
        mime = mimetypes.guess_type(source_path)[0] or "video/mp4"
        with open(source_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{data}", temp_path

    def _extract_segment(self):
        ffmpeg = detect_ffmpeg()
        if not ffmpeg:
            return ""
        duration_ms = max(0, self.segment.end_ms - self.segment.start_ms)
        if duration_ms <= 0:
            return ""

        fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{self.segment.start_ms / 1000.0:.3f}",
            "-t",
            f"{duration_ms / 1000.0:.3f}",
            "-i",
            self.video_path,
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            out_path,
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=True,
            )
        except Exception:
            try:
                os.remove(out_path)
            except OSError:
                pass
            return ""
        return out_path

    def _call_openai_api(
        self, api_address, api_key, model_id, settings, video_url
    ):
        if not api_address.endswith("/"):
            api_address += "/"
        if not api_address.endswith("chat/completions"):
            api_address += "chat/completions"

        messages = []
        system_prompt = settings.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                        "fps": 2,
                    },
                    {"type": "text", "text": self.prompt},
                ],
            }
        )

        temperature = settings.get("temperature", 0.7)
        data = {
            "model": model_id,
            "messages": messages,
            "temperature": (
                temperature / 100.0 if temperature > 2 else temperature
            ),
        }
        max_tokens = settings.get("max_length")
        if max_tokens:
            data["max_tokens"] = max_tokens

        self.progressChanged.emit("Waiting for AI response...")
        response = requests.post(
            api_address,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=data,
            timeout=(10, REQUEST_TIMEOUT),
        )
        if not response.ok:
            try:
                error_message = response.json().get("error", {}).get("message")
            except Exception:
                error_message = response.text
            raise RuntimeError(error_message or response.reason)
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()


class SegmentEditDialog(QDialog):
    def __init__(self, title, labels, current="", description="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(520, 360)
        self.setMinimumSize(420, 260)
        self.setStyleSheet(get_dialog_style() + get_settings_combo_style())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        self.combo = QComboBox(self)
        self.combo.addItems(labels)
        self.combo.setFixedHeight(32)
        if current in labels:
            self.combo.setCurrentIndex(labels.index(current))
        self._prepare_combo_popup()
        layout.addWidget(self.combo)

        self.description_edit = QPlainTextEdit(self)
        self.description_edit.setObjectName("XvaSegmentDescription")
        self.description_edit.setPlainText(description or "")
        self.description_edit.setPlaceholderText(self.tr("Description"))
        self.description_edit.setMinimumHeight(200)
        self.description_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.description_edit.verticalScrollBar().setStyleSheet(
            _vertical_scrollbar_style()
        )
        layout.addWidget(self.description_edit, 1)

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

    def text(self):
        return self.combo.currentText()

    def description(self):
        return self.description_edit.toPlainText()

    def _prepare_combo_popup(self):
        from PyQt6.QtWidgets import QAbstractItemView, QListView

        popup_view = QListView(self.combo)
        self.combo.setView(popup_view)
        self.combo.setMaxVisibleItems(min(12, max(1, self.combo.count())))
        popup_view.setUniformItemSizes(True)
        popup_view.setMouseTracking(True)
        popup_view.viewport().setMouseTracking(True)
        popup_view.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        popup_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )


class AIDescriptionDialog(QDialog):
    def __init__(
        self,
        model_text,
        prompt,
        parent=None,
        title=None,
        show_full_video_option=False,
        full_video_checked=False,
        full_video_enabled=True,
    ):
        super().__init__(parent)
        self.setWindowTitle(title or self.tr("AI Description"))
        self.setModal(True)
        self.resize(560, 380)
        self.setMinimumSize(480, 340)
        self.setStyleSheet(
            get_dialog_style()
            + get_settings_combo_style()
            + get_panel_frame_style()
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        model_label = QLabel(
            self.tr("Current model: {model}").format(model=model_text)
        )
        model_label.setWordWrap(False)
        model_label.setObjectName("XvaMetaLabel")
        model_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        model_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(model_label)

        self.full_video_checkbox = None
        if show_full_video_option:
            self.full_video_checkbox = QCheckBox(self.tr("Use full video"))
            self.full_video_checkbox.setChecked(full_video_checked)
            self.full_video_checkbox.setEnabled(full_video_enabled)
            header.addWidget(self.full_video_checkbox, 0)
        layout.addLayout(header)

        self.prompt_edit = QPlainTextEdit(self)
        self.prompt_edit.setObjectName("XvaSegmentDescription")
        self.prompt_edit.setPlainText(prompt)
        self.prompt_edit.setMinimumHeight(160)
        self.prompt_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.prompt_edit.verticalScrollBar().setStyleSheet(
            _vertical_scrollbar_style()
        )
        layout.addWidget(self.prompt_edit, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        hint = QLabel(
            self.tr(
                "Tip: The AI model is configured in Chatbot.\n"
                "Review the prompt before generating."
            )
        )
        hint.setWordWrap(True)
        hint.setObjectName("XvaMetaLabel")
        buttons.addWidget(hint, 1)
        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.setObjectName("XvaDialogSecondaryButton")
        cancel_btn.setStyleSheet(get_cancel_btn_style())
        cancel_btn.clicked.connect(self.reject)
        generate_btn = QPushButton(self.tr("Generate"))
        generate_btn.setObjectName("XvaDialogPrimaryButton")
        generate_btn.setStyleSheet(get_ok_btn_style())
        generate_btn.setDefault(True)
        generate_btn.clicked.connect(self.accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(generate_btn)
        layout.addLayout(buttons)

    def prompt(self):
        return self.prompt_edit.toPlainText().strip()

    def use_full_video(self):
        return bool(
            self.full_video_checkbox and self.full_video_checkbox.isChecked()
        )


class VideoClassifierDialog(QDialog):
    """Independent dialog for video clip classification labelling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VideoClassifierDialog")
        self.setWindowTitle(self.tr(DEFAULT_WINDOW_TITLE))
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setStyleSheet(get_dialog_style())
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(*DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(960, 640)
        self.setAcceptDrops(True)

        self._video_path = ""
        self._sidecar = SidecarData()
        self._dirty = False
        self._pending_in_ms = None
        self._pending_out_ms = None
        self._exporter = None
        self._selected_segment_id = ""
        self._undo_stack = []
        self._redo_stack = []
        self._editing_undo_segment_id = ""
        self._description_undo_segment_id = ""
        self._updating_description_editor = False
        self._description_worker = None
        self._cancelled_description_workers = []
        self._segmentation_worker = None
        self._cancelled_segmentation_workers = []
        self._ai_segmentation_replace = False
        self._ai_segmentation_target_segment_id = ""
        self._ai_segmentation_offset_ms = 0
        self._ai_segmentation_max_ms = 0
        self._ai_progress_dialog = None
        self._released = False

        self._build_ui()
        self._wire_signals()
        self._install_shortcuts()
        self._install_wheel_zoom_filters()
        self._refresh_actions()
        self._refresh_status_bar()

    # UI
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setHandleWidth(8)
        top_splitter.setChildrenCollapsible(False)
        preview_panel = self._build_preview_panel()
        right_panel = self._build_right_panel()
        top_splitter.addWidget(preview_panel)
        top_splitter.addWidget(right_panel)
        top_splitter.setSizes(
            [DEFAULT_WINDOW_SIZE[0], right_panel.minimumWidth()]
        )

        root.addWidget(top_splitter, 1)
        root.addWidget(self._build_timeline_panel(), 0)

        self.status_label = QLabel("", self)
        self.status_label.hide()

    def _build_preview_panel(self):
        panel = QFrame()
        panel.setObjectName("XvaPreviewPanel")
        panel.setStyleSheet(get_dialog_style())
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("XvaPreviewHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 6, 8, 6)
        header_layout.setSpacing(8)

        self.title_label = QLabel(self.tr("(no video loaded)"))
        self.title_label.setObjectName("XvaFileTitle")
        self.meta_label = QLabel("")
        self.meta_label.setObjectName("XvaMetaLabel")
        self.btn_export_frame = self._icon_button(
            "image",
            "svg",
            self.tr("Save frame"),
            self._save_current_frame,
            "text",
        )
        self.btn_shortcuts = self._icon_button(
            "shortcuts",
            "svg",
            self._shortcuts_tip_text(),
            self._show_shortcuts_menu,
            "text",
            14,
        )
        self.btn_tutorial = self._icon_button(
            "help-circle",
            "svg",
            self._tutorial_tip_text(),
            self._open_tutorial,
            "text",
            14,
        )
        header_actions = QWidget()
        header_actions_layout = QHBoxLayout(header_actions)
        header_actions_layout.setContentsMargins(0, 0, 0, 0)
        header_actions_layout.setSpacing(2)
        header_actions_layout.addWidget(self.btn_shortcuts, 0)
        header_actions_layout.addWidget(self.btn_tutorial, 0)
        header_layout.addWidget(self.title_label, 0)
        header_layout.addWidget(self.meta_label, 1)
        header_layout.addWidget(header_actions, 0)
        panel_layout.addWidget(header, 0)

        self.stack = QStackedWidget()
        self.drop_zone = DropZone()
        self.player = VideoPlayer(fps_provider=lambda: self._sidecar.fps)
        self.stack.addWidget(self.drop_zone)
        self.stack.addWidget(self.player)
        self.stack.setCurrentWidget(self.drop_zone)
        panel_layout.addWidget(self.stack, 1)

        panel_layout.addWidget(self._build_preview_footer(), 0)
        panel_layout.addWidget(self._build_preview_zoom_panel(), 0)
        return panel

    def _build_preview_footer(self):
        footer = QFrame()
        footer.setObjectName("XvaPreviewFooter")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)

        self.current_time_label = QLabel("00:00:00:00")
        self.current_time_label.setObjectName("XvaCurrentTimeLabel")
        self.duration_label = QLabel("00:00:00:00")
        self.duration_label.setObjectName("XvaDurationLabel")
        layout.addWidget(self.current_time_label)
        layout.addSpacing(8)
        layout.addWidget(self.duration_label)
        layout.addStretch(1)

        self.btn_back_s = self._icon_button(
            "backward",
            "svg",
            self.tr("Back 1s"),
            lambda: self.player.step(-1000),
            "text",
        )
        self.btn_prev_frame = self._icon_button(
            "left",
            "svg",
            self.tr("Previous frame"),
            lambda: self.player.step_frames(-1),
            "text",
        )
        self.btn_play = self._icon_button(
            "play",
            "svg",
            self.tr("Play/Pause"),
            self.player.toggle_play,
            "text",
        )
        self.btn_next_frame = self._icon_button(
            "right",
            "svg",
            self.tr("Next frame"),
            lambda: self.player.step_frames(1),
            "text",
        )
        self.btn_forward_s = self._icon_button(
            "forward",
            "svg",
            self.tr("Forward 1s"),
            lambda: self.player.step(1000),
            "text",
        )
        for button in (
            self.btn_back_s,
            self.btn_prev_frame,
            self.btn_play,
            self.btn_next_frame,
            self.btn_forward_s,
        ):
            layout.addWidget(button)

        layout.addStretch(1)
        self.btn_zoom = self._icon_button(
            "fit",
            "svg",
            self.tr("Preview zoom"),
            self._toggle_preview_zoom_panel,
        )
        self.btn_open = self._icon_button(
            "open", "svg", self.tr("Open video"), self._on_open_clicked
        )
        self.btn_ai_segment = self._icon_button(
            "wand",
            "svg",
            self.tr("Auto segment video with AI"),
            self._on_ai_segment_clicked,
        )
        self.btn_export = self._icon_button(
            "export",
            "svg",
            self.tr("Export dataset"),
            self._on_export_clicked,
            "text",
        )
        layout.addWidget(self.btn_zoom)
        layout.addWidget(self.btn_open)
        layout.addWidget(self.btn_ai_segment)
        layout.addWidget(self.btn_export_frame)
        layout.addWidget(self.btn_export)
        return footer

    def _build_preview_zoom_panel(self):
        panel = QFrame()
        panel.setObjectName("XvaZoomPanel")
        panel.hide()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)
        self.preview_zoom_panel = panel
        self.btn_preview_zoom_out = self._icon_button(
            "zoom-out",
            "svg",
            self.tr("Zoom out"),
            lambda: self._step_preview_zoom(-10),
            "text",
        )
        self.preview_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_zoom_slider.setRange(0, 200)
        self.preview_zoom_slider.setValue(100)
        self.preview_zoom_slider.setTickInterval(100)
        self.preview_zoom_slider.setTickPosition(
            QSlider.TickPosition.TicksBelow
        )
        self.preview_zoom_slider.setStyleSheet(get_slider_style())
        self.preview_zoom_slider.valueChanged.connect(
            self._on_preview_zoom_changed
        )
        self.preview_zoom_label = QLabel("100%")
        self.preview_zoom_label.setObjectName("XvaMetaLabel")
        self.btn_preview_zoom_in = self._icon_button(
            "zoom-in",
            "svg",
            self.tr("Zoom in"),
            lambda: self._step_preview_zoom(10),
            "text",
        )
        layout.addWidget(self.btn_preview_zoom_out)
        layout.addWidget(self.preview_zoom_slider, 1)
        layout.addWidget(self.preview_zoom_label)
        layout.addWidget(self.btn_preview_zoom_in)
        return panel

    def _build_right_panel(self):
        right = QWidget()
        right.setMinimumWidth(320)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        label_frame = QFrame()
        label_frame.setObjectName("XvaPanel")
        label_frame.setStyleSheet(get_panel_frame_style())
        lf_layout = QVBoxLayout(label_frame)
        lf_layout.setContentsMargins(12, 12, 12, 12)
        self.label_panel = LabelPanel()
        lf_layout.addWidget(self.label_panel)
        right_layout.addWidget(label_frame, 1)

        seg_frame = QFrame()
        seg_frame.setObjectName("XvaPanel")
        seg_frame.setStyleSheet(get_panel_frame_style())
        sf_layout = QVBoxLayout(seg_frame)
        sf_layout.setContentsMargins(12, 12, 12, 12)
        self.segment_list = SegmentListPanel()
        sf_layout.addWidget(self.segment_list)
        right_layout.addWidget(seg_frame, 1)

        desc_frame = QFrame()
        desc_frame.setObjectName("XvaPanel")
        desc_frame.setStyleSheet(get_panel_frame_style())
        desc_layout = QVBoxLayout(desc_frame)
        desc_layout.setContentsMargins(12, 12, 12, 12)
        desc_layout.setSpacing(8)
        desc_header = QGridLayout()
        desc_header.setContentsMargins(0, 0, 0, 0)
        desc_header.setColumnMinimumWidth(0, 64)
        desc_header.setColumnStretch(1, 1)
        desc_header.setColumnMinimumWidth(2, 64)
        desc_title = QLabel(self.tr("Description"))
        desc_title.setObjectName("XvaPanelTitle")
        desc_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_header.addWidget(desc_title, 0, 1)
        self.btn_ai_description = self._icon_button(
            "wand",
            "svg",
            self.tr("Generate description with AI"),
            self._on_ai_description_clicked,
            "text",
        )
        desc_header.addWidget(
            self.btn_ai_description, 0, 2, Qt.AlignmentFlag.AlignRight
        )
        self.segment_description_edit = QPlainTextEdit()
        self.segment_description_edit.setObjectName("XvaSegmentDescription")
        self.segment_description_edit.setPlaceholderText(
            self.tr("Select a segment to edit its description.")
        )
        self.segment_description_edit.setFixedHeight(96)
        self.segment_description_edit.setEnabled(False)
        desc_layout.addLayout(desc_header)
        desc_layout.addWidget(self.segment_description_edit)
        right_layout.addWidget(desc_frame, 0)
        return right

    def _build_timeline_panel(self):
        frame = QFrame()
        frame.setObjectName("XvaTimelinePanel")
        frame.setStyleSheet(get_timeline_frame_style())
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.timeline_toolbar = QFrame()
        self.timeline_toolbar.setObjectName("XvaTimelineToolbar")
        toolbar_layout = QHBoxLayout(self.timeline_toolbar)
        toolbar_layout.setContentsMargins(10, 6, 10, 6)
        toolbar_layout.setSpacing(2)
        self.timeline = TimelineWidget()

        self.btn_undo = self._icon_button(
            "undo",
            "svg",
            self.tr("Undo"),
            self._undo_last_segment,
            "text",
        )
        self.btn_reset = self._icon_button(
            "redo",
            "svg",
            self.tr("Redo"),
            self._redo_last_segment,
            "text",
        )
        self.btn_split = self._icon_button(
            "scissors",
            "svg",
            self.tr("Split at playhead"),
            self._split_selected_segment,
            "text",
        )
        self.btn_export_selected = self._icon_button(
            "export",
            "svg",
            self.tr("Export segment"),
            self._export_selected_segment,
            "text",
        )

        for button in (
            self.btn_undo,
            self.btn_reset,
            self.btn_split,
            self.btn_export_selected,
        ):
            toolbar_layout.addWidget(button)
        toolbar_layout.addStretch(1)
        self.timeline_hint_label = QLabel("", self.timeline_toolbar)
        self.timeline_hint_label.setObjectName("XvaTimelineHint")
        self.timeline_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timeline_hint_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.timeline_hint_label.hide()

        self.btn_timeline_zoom_out = self._icon_button(
            "zoom-out",
            "svg",
            self.tr("Zoom timeline out"),
            lambda: self._step_timeline_zoom(-1),
            "text",
        )
        self.timeline_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_zoom_slider.setRange(0, 100)
        self.timeline_zoom_slider.setValue(0)
        self.timeline_zoom_slider.setTickInterval(10)
        self.timeline_zoom_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.timeline_zoom_slider.setFixedWidth(150)
        self.timeline_zoom_slider.setStyleSheet(get_slider_style())
        self.timeline_zoom_slider.valueChanged.connect(
            self._on_timeline_zoom_changed
        )
        self.btn_timeline_zoom_in = self._icon_button(
            "zoom-in",
            "svg",
            self.tr("Zoom timeline in"),
            lambda: self._step_timeline_zoom(1),
            "text",
        )
        toolbar_layout.addWidget(self.btn_timeline_zoom_out)
        toolbar_layout.addWidget(self.timeline_zoom_slider)
        toolbar_layout.addWidget(self.btn_timeline_zoom_in)
        layout.addWidget(self.timeline_toolbar, 0)

        layout.addWidget(self.timeline, 0)
        self.timeline_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.timeline_scrollbar.setObjectName("XvaTimelineScrollBar")
        self.timeline_scrollbar.setFixedHeight(8)
        self.timeline_scrollbar.valueChanged.connect(
            self.timeline.set_view_start
        )
        layout.addWidget(self.timeline_scrollbar, 0)
        return frame

    def _icon_button(
        self, icon, ext, tip, slot, color_role="text", icon_size=18
    ):
        button = QPushButton()
        button.setStyleSheet(get_icon_button_style())
        button.setToolTip(tip)
        button.setFixedSize(24, 24)
        apply_button_icon(
            button, icon, ext, icon_size, theme_icon_color(color_role)
        )
        if button.icon().isNull():
            button.setText(icon[:1].upper())
        button.clicked.connect(slot)
        return button

    def _shortcuts_tip_text(self):
        return (
            "查看快捷键"
            if self._tutorial_language() == "zh_cn"
            else "Show shortcuts"
        )

    def _shortcut_rows(self):
        if self._tutorial_language() == "zh_cn":
            return [
                ("Space", "播放/暂停"),
                ("Left/Right", "上一帧/下一帧"),
                ("Shift+Left/Right", "后退/前进 1 秒"),
                (",/.", "上一个/下一个片段"),
                ("I/O", "标记入点/出点"),
                ("Enter", "根据入点和出点创建片段"),
                ("Delete/Backspace", "删除选中片段"),
                ("Ctrl+S", "保存标注"),
                ("Ctrl+Z", "撤销"),
                ("Ctrl+Shift+Z", "重做"),
                ("0-9", "选择对应序号的标签"),
            ]
        return [
            ("Space", "Play/Pause"),
            ("Left/Right", "Previous/Next frame"),
            ("Shift+Left/Right", "Back/Forward 1 second"),
            (",/.", "Previous/Next segment"),
            ("I/O", "Mark in/out"),
            ("Enter", "Create segment from marks"),
            ("Delete/Backspace", "Delete selected segment"),
            ("Ctrl+S", "Save annotations"),
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Shift+Z", "Redo"),
            ("0-9", "Select label by number"),
        ]

    def _show_shortcuts_menu(self):
        menu = QMenu(self)
        menu.setObjectName("XvaShortcutMenu")
        action = QWidgetAction(menu)
        panel = QWidget(menu)
        panel.setObjectName("XvaShortcutPanel")
        grid = QGridLayout(panel)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)

        title = QLabel(
            "快捷键" if self._tutorial_language() == "zh_cn" else "Shortcuts"
        )
        title.setObjectName("XvaShortcutTitle")
        grid.addWidget(title, 0, 0, 1, 2)

        key_header = QLabel(
            "按键" if self._tutorial_language() == "zh_cn" else "Key"
        )
        key_header.setObjectName("XvaShortcutHeader")
        action_header = QLabel(
            "功能" if self._tutorial_language() == "zh_cn" else "Action"
        )
        action_header.setObjectName("XvaShortcutHeader")
        grid.addWidget(key_header, 1, 0)
        grid.addWidget(action_header, 1, 1)

        for row, (key, text) in enumerate(self._shortcut_rows(), 2):
            key_label = QLabel(key)
            key_label.setObjectName("XvaShortcutKey")
            text_label = QLabel(text)
            text_label.setObjectName("XvaShortcutAction")
            grid.addWidget(key_label, row, 0)
            grid.addWidget(text_label, row, 1)

        action.setDefaultWidget(panel)
        menu.addAction(action)
        pos = self.btn_shortcuts.mapToGlobal(
            self.btn_shortcuts.rect().bottomRight()
        )
        pos.setX(pos.x() - menu.sizeHint().width())
        pos.setY(pos.y() + 4)
        menu.exec(pos)

    def _tutorial_language(self):
        config = getattr(self, "_config", {})
        if not isinstance(config, dict) and self.parent():
            config = getattr(self.parent(), "_config", {})
        language = config.get("language") if isinstance(config, dict) else ""
        return "zh_cn" if language == "zh_CN" else "en"

    def _tutorial_tip_text(self):
        return (
            "打开教程"
            if self._tutorial_language() == "zh_cn"
            else "Open tutorial"
        )

    def _open_tutorial(self):
        language = self._tutorial_language()
        url = (
            "https://github.com/CVHub520/X-AnyLabeling/blob/main/"
            f"docs/{language}/video_classifier.md"
        )
        QDesktopServices.openUrl(QUrl(url))

    # Signals
    def _wire_signals(self):
        self.drop_zone.videoSelected.connect(self.load_video)

        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.playbackStateChanged.connect(self._sync_play_button)
        self.player.markInRequested.connect(self._mark_in)
        self.player.markOutRequested.connect(self._mark_out)
        self.player.commitSegmentRequested.connect(self._commit_from_marks)
        self.player.errorOccurred.connect(self._on_player_error)

        self.timeline.seekRequested.connect(self._on_timeline_seek)
        self.timeline.segmentRequested.connect(self._on_timeline_create)
        self.timeline.segmentCreationBlocked.connect(
            self._on_timeline_create_blocked
        )
        self.timeline.segmentSelected.connect(self._on_timeline_select)
        self.timeline.segmentDoubleClicked.connect(self._relabel_segment)
        self.timeline.segmentEdited.connect(self._on_timeline_edit)
        self.timeline.segmentEditFinished.connect(
            self._on_timeline_edit_finished
        )
        self.timeline.hoverMs.connect(self._on_timeline_hover)
        self.timeline.viewChanged.connect(self._sync_timeline_scrollbar)

        self.label_panel.labelsChanged.connect(self._on_labels_changed)
        self.label_panel.labelSelected.connect(self._on_active_label_changed)
        self.label_panel.labelRenamed.connect(self._on_label_renamed)
        self.label_panel.labelRemoved.connect(self._on_label_removed)

        self.segment_list.segmentActivated.connect(self._on_segment_jump)
        self.segment_list.segmentSelected.connect(self._on_segment_picked)
        self.segment_list.segmentDeleted.connect(self._delete_segment)
        self.segment_list.segmentRelabelRequested.connect(
            self._relabel_segment
        )
        self.segment_description_edit.textChanged.connect(
            self._on_description_changed
        )

    def _install_shortcuts(self):
        def sc(seq, slot, ctx=Qt.ShortcutContext.WindowShortcut):
            s = QShortcut(QKeySequence(seq), self)
            s.setContext(ctx)
            s.activated.connect(slot)
            return s

        sc("Space", self.player.toggle_play)
        sc("Right", lambda: self.player.step_frames(1))
        sc("Left", lambda: self.player.step_frames(-1))
        sc("Shift+Right", lambda: self.player.step(1000))
        sc("Shift+Left", lambda: self.player.step(-1000))
        sc(",", self._jump_prev_segment)
        sc(".", self._jump_next_segment)
        sc("I", self._mark_in)
        sc("O", self._mark_out)
        sc("Return", self._commit_from_marks)
        sc("Enter", self._commit_from_marks)
        sc("Delete", self._delete_selected_segment)
        sc("Backspace", self._delete_selected_segment)
        sc("Ctrl+S", self._save_sidecar)
        sc("Ctrl+Z", self._undo_last_segment)
        sc("Ctrl+Shift+Z", self._redo_last_segment)
        for i in range(10):
            sc(str(i), lambda d=i: self.label_panel.select_by_digit(d))

    def _install_wheel_zoom_filters(self):
        for widget in (
            self.player,
            self.player.video_widget,
            self.timeline,
            self.timeline_scrollbar,
            self.timeline_toolbar,
        ):
            widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.timeline:
            if event.type() == QEvent.Type.MouseMove:
                self._show_timeline_hint(
                    self.timeline.hover_hint(event.position().toPoint())
                )
            elif event.type() == QEvent.Type.Leave:
                self._hide_timeline_hint()
        elif obj is self.timeline_scrollbar:
            if event.type() in (QEvent.Type.Enter, QEvent.Type.MouseMove):
                self._hide_timeline_hint()
        elif (
            obj is self.timeline_toolbar and event.type() == QEvent.Type.Resize
        ):
            self._position_timeline_hint()
        if event.type() == QEvent.Type.Wheel:
            steps = self._wheel_steps(event)
            if not steps:
                return super().eventFilter(obj, event)
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                if obj in (self.player, self.player.video_widget):
                    anchor = self._preview_zoom_anchor(obj, event)
                    self._step_preview_zoom(steps * 10, anchor)
                    event.accept()
                    return True
                if obj in (self.timeline, self.timeline_scrollbar):
                    self._step_timeline_zoom(steps)
                    event.accept()
                    return True
            elif obj in (self.timeline, self.timeline_scrollbar):
                if self._scroll_timeline_with_wheel(steps):
                    event.accept()
                    return True
        return super().eventFilter(obj, event)

    def _show_timeline_hint(self, hint):
        if not self._video_path or not hint:
            self._hide_timeline_hint()
            return
        if hint == "segment":
            text = self.tr(
                "Double-click to edit label and description; hold left button to drag"
            )
        elif hint == "ruler":
            text = self.tr(
                "Right-drag the timeline ruler, then release to create a segment"
            )
        else:
            self._hide_timeline_hint()
            return
        if self.timeline_hint_label.text() != text:
            self.timeline_hint_label.setText(text)
        self._position_timeline_hint()
        self.timeline_hint_label.show()
        self.timeline_hint_label.raise_()

    def _hide_timeline_hint(self):
        self.timeline_hint_label.hide()

    def _position_timeline_hint(self):
        if not hasattr(self, "timeline_hint_label"):
            return
        self.timeline_hint_label.adjustSize()
        margin = 8
        hint = self.timeline_hint_label.sizeHint()
        width = min(
            hint.width() + 12,
            max(0, self.timeline_toolbar.width() - margin * 2),
        )
        height = hint.height()
        self.timeline_hint_label.setGeometry(
            (self.timeline_toolbar.width() - width) // 2,
            (self.timeline_toolbar.height() - height) // 2,
            width,
            height,
        )

    @staticmethod
    def _wheel_steps(event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if not delta:
            return 0
        direction = 1 if delta > 0 else -1
        steps = max(1, abs(delta) // 120)
        return direction * steps

    def _preview_zoom_anchor(self, obj, event):
        pos = event.position().toPoint()
        if obj is self.player.video_widget:
            return pos
        return self.player.video_widget.mapFrom(obj, pos)

    def _scroll_timeline_with_wheel(self, steps):
        if not self.timeline_scrollbar.isEnabled():
            return False
        amount = max(1, self.timeline_scrollbar.pageStep() // 6)
        value = self.timeline_scrollbar.value() - int(steps) * amount
        value = max(
            self.timeline_scrollbar.minimum(),
            min(self.timeline_scrollbar.maximum(), value),
        )
        if value == self.timeline_scrollbar.value():
            return False
        self.timeline_scrollbar.setValue(value)
        return True

    # Video lifecycle
    def load_video(self, path):
        if not path or not os.path.isfile(path):
            return
        if not path.lower().endswith(SUPPORTED_VIDEO_EXTS):
            QMessageBox.warning(
                self,
                self.tr("Unsupported file"),
                self.tr("This file type is not supported."),
            )
            return
        if not self._save_dirty_sidecar():
            return

        self._video_path = path
        self.title_label.setText(os.path.basename(path))
        self.title_label.setToolTip(path)
        # Probe metadata via ffprobe / opencv to capture fps / size early.
        info = probe_video(path) or {}
        existing = load_sidecar(path)
        if existing:
            sidecar = existing
            # refresh probe info if missing
            if not sidecar.fps and info.get("fps"):
                sidecar.fps = info["fps"]
            if not sidecar.duration_ms and info.get("duration_ms"):
                sidecar.duration_ms = info["duration_ms"]
            if not sidecar.width and info.get("width"):
                sidecar.width = info["width"]
            if not sidecar.height and info.get("height"):
                sidecar.height = info["height"]
        else:
            sidecar = SidecarData(
                video=os.path.basename(path),
                fps=info.get("fps", 0.0),
                duration_ms=info.get("duration_ms", 0),
                width=info.get("width", 0),
                height=info.get("height", 0),
            )

        # ensure colors are populated for labels
        for idx, label in enumerate(sidecar.labels):
            if label not in sidecar.label_colors:
                sidecar.label_colors[label] = color_for_label(label, index=idx)

        self._sidecar = sidecar
        self._dirty = False
        self._pending_in_ms = None
        self._pending_out_ms = None
        self._selected_segment_id = ""
        self._undo_stack = []
        self._redo_stack = []
        self._editing_undo_segment_id = ""
        self._description_undo_segment_id = ""
        self.preview_zoom_slider.setValue(100)
        self.timeline_zoom_slider.setValue(0)
        self.timeline.set_fps(sidecar.fps)

        self.stack.setCurrentWidget(self.player)
        self.player.show()
        self.player.video_widget.show()

        QApplication.processEvents()
        self.player.load(path)
        self.player.seek(0)
        # Note: QMediaPlayer reports its own duration asynchronously
        if sidecar.duration_ms:
            self.timeline.set_duration(sidecar.duration_ms)
        self.timeline.set_playhead(0)
        self.timeline.set_view_start(0)
        self.timeline.set_thumbnails(extract_video_thumbnails(path))

        self.label_panel.set_labels(sidecar.labels, sidecar.label_colors)
        self._sync_timeline_active_label()
        self._refresh_timeline()
        self._refresh_segment_list()
        self._refresh_description_editor()
        self._refresh_actions()
        self._refresh_video_meta()
        self._update_time_labels()
        self._refresh_status_bar()

    def _on_open_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open video"),
            os.path.dirname(self._video_path) if self._video_path else "",
            SUPPORTED_VIDEO_FILTER,
        )
        if path:
            self.load_video(path)

    def _on_close_video(self):
        if not self._save_dirty_sidecar():
            return
        self.player.stop()
        try:
            self.player.load("")
        except Exception:
            pass
        self.stack.setCurrentWidget(self.drop_zone)
        self._video_path = ""
        self._sidecar = SidecarData()
        self._dirty = False
        self._undo_stack = []
        self._redo_stack = []
        self._editing_undo_segment_id = ""
        self._description_undo_segment_id = ""
        self.title_label.setText(self.tr("(no video loaded)"))
        self.title_label.setToolTip("")
        self.meta_label.setText("")
        self.timeline_hint_label.hide()
        self._selected_segment_id = ""
        self.preview_zoom_slider.setValue(100)
        self.timeline_zoom_slider.setValue(0)
        self.timeline.set_duration(0)
        self.timeline.set_playhead(0)
        self.timeline.set_view_start(0)
        self.timeline.set_fps(0)
        self.timeline.set_segments([])
        self.timeline.set_thumbnails([])
        self.label_panel.set_labels([], {})
        self._sync_timeline_active_label()
        self._refresh_segment_list()
        self._refresh_description_editor()
        self._refresh_actions()
        self._update_time_labels()
        self._refresh_status_bar()

    def _on_position_changed(self, pos_ms):
        self.timeline.set_playhead(pos_ms)
        self._update_time_labels()
        if not self.player.is_playing():
            self._refresh_split_action()

    def _on_duration_changed(self, dur_ms):
        # QMediaPlayer reports its own duration; trust whichever is largest
        cur = self._sidecar.duration_ms or 0
        if dur_ms and dur_ms > cur:
            self._sidecar.duration_ms = int(dur_ms)
        self.timeline.set_duration(self._sidecar.duration_ms or dur_ms)
        self._refresh_video_meta()
        self._update_time_labels()

    def _on_player_error(self, message):
        QMessageBox.critical(
            self,
            self.tr("Playback error"),
            self.tr(
                "Failed to play this video. The system may lack a suitable "
                "Qt multimedia codec.\n\nDetails: {msg}"
            ).format(msg=message or ""),
        )

    def _refresh_video_meta(self):
        if not self._video_path:
            self.meta_label.setText("")
            return
        sd = self._sidecar
        self.meta_label.setText(
            self.tr("FPS: {f:.2f} | Size: {w}x{h}").format(
                f=sd.fps or 0.0,
                w=sd.width or "?",
                h=sd.height or "?",
            )
        )

    def _update_time_labels(self):
        if not hasattr(self, "current_time_label"):
            return
        current = self._format_frame_timecode(self.player.position())
        if self.current_time_label.text() != current:
            self.current_time_label.setText(current)
        duration = self._sidecar.duration_ms or self.player.duration()
        duration = self._format_frame_timecode(duration)
        if self.duration_label.text() != duration:
            self.duration_label.setText(duration)

    def _format_frame_timecode(self, ms):
        fps = float(self._sidecar.fps or 0.0)
        if fps <= 0:
            return ms_to_timecode(ms or 0)
        fps_int = max(1, int(round(fps)))
        total_frames = int(round(float(ms or 0) / 1000.0 * fps))
        total_seconds, frame = divmod(total_frames, fps_int)
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame:02d}"

    def _sync_play_button(self, *_args):
        if not hasattr(self, "btn_play"):
            return
        if self.player.is_playing():
            apply_button_icon(
                self.btn_play, "pause", "svg", 18, theme_icon_color("text")
            )
        else:
            apply_button_icon(
                self.btn_play, "play", "svg", 18, theme_icon_color("text")
            )
        self._refresh_actions()

    def _toggle_preview_zoom_panel(self):
        self.preview_zoom_panel.setVisible(
            not self.preview_zoom_panel.isVisible()
        )

    def _on_preview_zoom_changed(self, value):
        if value <= 100:
            percent = 25 + value * 0.75
        else:
            percent = 100 + (value - 100) * 3
        self.preview_zoom_label.setText(f"{int(round(percent))}%")
        anchor = getattr(self, "_pending_preview_zoom_anchor", None)
        self._pending_preview_zoom_anchor = None
        self.player.set_zoom(percent / 100.0, anchor)

    def _step_preview_zoom(self, step, anchor=None):
        value = max(
            self.preview_zoom_slider.minimum(),
            min(
                self.preview_zoom_slider.maximum(),
                self.preview_zoom_slider.value() + int(step),
            ),
        )
        if value == self.preview_zoom_slider.value():
            return
        self._pending_preview_zoom_anchor = anchor
        self.preview_zoom_slider.setValue(value)
        if self._pending_preview_zoom_anchor is anchor:
            self._pending_preview_zoom_anchor = None

    def _step_timeline_zoom(self, step):
        self.timeline_zoom_slider.setValue(
            max(
                self.timeline_zoom_slider.minimum(),
                min(
                    self.timeline_zoom_slider.maximum(),
                    self.timeline_zoom_slider.value() + int(step) * 5,
                ),
            )
        )

    def _on_timeline_zoom_changed(self, value):
        max_zoom = self._max_timeline_zoom_factor()
        if value <= 0 or max_zoom <= 1:
            self.timeline.set_zoom_factor(1.0)
            return
        factor = max_zoom ** (float(value) / 100.0)
        self.timeline.set_zoom_factor(factor)

    def _max_timeline_zoom_factor(self):
        duration = max(1, self.timeline.duration_ms())
        fps = float(self._sidecar.fps or 30.0)
        min_visible = max(1000, int(round(36 * 1000.0 / max(1.0, fps))))
        min_visible = min(duration, min_visible)
        return max(1.0, float(duration) / float(max(1, min_visible)))

    def _sync_timeline_scrollbar(self, start_ms, visible_ms, duration_ms):
        if not hasattr(self, "timeline_scrollbar"):
            return
        max_start = max(0, int(duration_ms) - int(visible_ms))
        self.timeline_scrollbar.blockSignals(True)
        self.timeline_scrollbar.setRange(0, max_start)
        self.timeline_scrollbar.setPageStep(max(1, int(visible_ms)))
        self.timeline_scrollbar.setSingleStep(max(1, int(visible_ms) // 20))
        self.timeline_scrollbar.setValue(max(0, min(int(start_ms), max_start)))
        self.timeline_scrollbar.setEnabled(max_start > 0)
        self.timeline_scrollbar.blockSignals(False)

    def _save_current_frame(self):
        if not self._video_path:
            return
        image = self.player.current_image()
        if image.isNull():
            QMessageBox.information(
                self,
                self.tr("Save current frame"),
                self.tr("No decoded frame is available yet."),
            )
            return
        stem = os.path.splitext(os.path.basename(self._video_path))[0]
        frame = self._format_frame_timecode(self.player.position()).replace(
            ":", "-"
        )
        default_path = os.path.join(
            os.path.dirname(self._video_path), f"{stem}_{frame}.png"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save current frame"),
            default_path,
            self.tr("PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)"),
        )
        if path and not image.save(path):
            QMessageBox.warning(
                self,
                self.tr("Save current frame"),
                self.tr("Failed to save the current frame."),
            )

    # Sidecar / dirty state
    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            self._refresh_actions()

    def _save_sidecar(self):
        if not self._video_path:
            return True
        try:
            self._sidecar.labels = self.label_panel.labels()
            self._sidecar.label_colors = self.label_panel.colors()
            for seg in self._sidecar.segments:
                seg.refresh_frames(self._sidecar.fps)
            path = save_sidecar(self._video_path, self._sidecar)
        except Exception as exc:
            QMessageBox.critical(self, self.tr("Save failed"), str(exc))
            return False
        self._dirty = False
        self._refresh_actions()
        self._refresh_status_bar(extra=self.tr("Saved: {p}").format(p=path))
        return True

    def _save_dirty_sidecar(self):
        if not self._dirty:
            return True
        return self._save_sidecar()

    def _push_undo_snapshot(self, selected_segment_id=None):
        self._undo_stack.append(self._segment_snapshot(selected_segment_id))
        self._redo_stack = []
        self._refresh_actions()

    def _segment_snapshot(self, selected_segment_id=None):
        selected = (
            self._selected_segment_id
            if selected_segment_id is None
            else selected_segment_id
        )
        return {
            "segments": deepcopy(self._sidecar.segments),
            "selected_segment_id": selected,
        }

    def _restore_segment_snapshot(self, snapshot):
        self._sidecar.segments = deepcopy(snapshot["segments"])
        selected = snapshot.get("selected_segment_id") or ""
        if selected and not self._find_segment(selected):
            selected = ""
        self._selected_segment_id = selected
        self._editing_undo_segment_id = ""
        self._description_undo_segment_id = ""
        self._refresh_timeline()
        self._refresh_segment_list()
        self.timeline.set_selected(selected)
        self.segment_list.select(selected)
        self._refresh_description_editor()

    # Timeline events
    def _on_timeline_seek(self, ms):
        self.player.seek(int(ms))

    def _on_timeline_hover(self, ms):
        self._refresh_status_bar(hover_ms=ms)

    def _on_timeline_create(self, start_ms, end_ms):
        label = self.label_panel.active_label()
        if not label:
            self._on_timeline_create_blocked()
            return
        if self._has_segment_overlap(start_ms, end_ms):
            return
        self._push_undo_snapshot()
        seg = Segment.new(label, start_ms, end_ms, fps=self._sidecar.fps)
        self._sidecar.segments.append(seg)
        self._sidecar.upsert_label(label, self.label_panel.color_for(label))
        self._selected_segment_id = seg.id
        self._mark_dirty()
        self._refresh_timeline()
        self._refresh_segment_list()
        self.timeline.set_selected(seg.id)
        self.segment_list.select(seg.id)
        self._refresh_description_editor()

    def _on_timeline_select(self, seg_id):
        self._selected_segment_id = seg_id or ""
        self.segment_list.select(seg_id)
        self._refresh_description_editor()
        self._refresh_actions()

    def _on_timeline_edit(self, seg_id, start_ms, end_ms):
        seg = self._find_segment(seg_id)
        if not seg:
            return
        if self._has_segment_overlap(start_ms, end_ms, ignore_id=seg_id):
            self._refresh_timeline()
            return
        if self._editing_undo_segment_id != seg_id:
            self._push_undo_snapshot()
            self._editing_undo_segment_id = seg_id
        seg.start_ms = int(start_ms)
        seg.end_ms = int(end_ms)
        seg.refresh_frames(self._sidecar.fps)
        self._mark_dirty()
        self._refresh_segment_list_no_focus()
        self._refresh_status_bar()

    def _on_timeline_edit_finished(self):
        self._editing_undo_segment_id = ""

    # Label panel
    def _on_labels_changed(self):
        self._undo_stack = []
        self._redo_stack = []
        self._editing_undo_segment_id = ""
        self._sidecar.labels = self.label_panel.labels()
        self._sidecar.label_colors = self.label_panel.colors()
        self._sync_timeline_active_label()
        self._refresh_timeline()
        self._refresh_segment_list()
        self._mark_dirty()
        self._refresh_actions()

    def _on_active_label_changed(self, name):
        self._sync_timeline_active_label(name)

    def _sync_timeline_active_label(self, name=None):
        active = name if name is not None else self.label_panel.active_label()
        self.timeline.set_segment_creation_enabled(bool(active))
        self.timeline.set_current_label_color(
            self.label_panel.color_for(active) if active else None
        )

    def _on_timeline_create_blocked(self):
        if not self.label_panel.labels():
            QMessageBox.information(
                self,
                self.tr("Add segment"),
                self.tr("Create a label first."),
            )
            return
        QMessageBox.information(
            self,
            self.tr("Add segment"),
            self.tr("Select a label first."),
        )

    def _on_label_renamed(self, old, new):
        for seg in self._sidecar.segments:
            if seg.label == old:
                seg.label = new
        self._refresh_timeline()
        self._refresh_segment_list()
        self._mark_dirty()

    def _on_label_removed(self, name):
        before = len(self._sidecar.segments)
        self._sidecar.segments = [
            s for s in self._sidecar.segments if s.label != name
        ]
        if len(self._sidecar.segments) != before:
            self._mark_dirty()
        if self._selected_segment_id and not self._find_segment(
            self._selected_segment_id
        ):
            self._selected_segment_id = ""
        self._refresh_timeline()
        self._refresh_segment_list()
        self._refresh_description_editor()

    def _ensure_active_label_or_prompt(self):
        active = self.label_panel.active_label()
        if active:
            return active
        dialog = LabelNameDialog(self.tr("Pick a label"), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return ""
        name = dialog.text().strip()
        if not name:
            return ""
        self.label_panel.add_label(name)
        return name

    # Segment list
    def _refresh_segment_list(self):
        self.segment_list.set_segments(
            self._sidecar.segments, self.label_panel.colors()
        )

    def _refresh_segment_list_no_focus(self):
        sid = self.segment_list.current_id()
        self._refresh_segment_list()
        if sid:
            self.segment_list.select(sid)

    def _refresh_timeline(self):
        self.timeline.set_segments(
            self._sidecar.segments, self.label_panel.colors()
        )
        self._refresh_split_action()

    def _on_segment_jump(self, seg_id):
        seg = self._find_segment(seg_id)
        if not seg:
            return
        self.player.seek(seg.start_ms)
        self.timeline.set_selected(seg_id)

    def _on_segment_picked(self, seg_id):
        self._selected_segment_id = seg_id or ""
        self.timeline.set_selected(seg_id)
        seg = self._find_segment(seg_id)
        if seg:
            self.player.seek(seg.start_ms)
        self._refresh_description_editor()
        self._refresh_actions()

    def _delete_segment(self, seg_id):
        if not self._find_segment(seg_id):
            return
        self._push_undo_snapshot()
        self._sidecar.segments = [
            s for s in self._sidecar.segments if s.id != seg_id
        ]
        if self._selected_segment_id == seg_id:
            self._selected_segment_id = ""
        self._mark_dirty()
        self.timeline.set_selected("")
        self._refresh_description_editor()
        self._refresh_timeline()
        self._refresh_segment_list()
        self._refresh_actions()

    def _delete_selected_segment(self):
        sid = self.segment_list.current_id()
        if sid:
            self._delete_segment(sid)

    def _relabel_segment(self, seg_id):
        seg = self._find_segment(seg_id)
        if not seg:
            return
        labels = self.label_panel.labels()
        if not labels:
            QMessageBox.information(
                self,
                self.tr("Edit segment"),
                self.tr("Define some labels first."),
            )
            return
        dialog = SegmentEditDialog(
            self.tr("Edit segment"),
            labels,
            seg.label,
            seg.description,
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new = dialog.text()
        description = dialog.description()
        if new and (new != seg.label or description != seg.description):
            self._push_undo_snapshot()
            seg.label = new
            seg.description = description
            self._mark_dirty()
            self._refresh_timeline()
            self._refresh_segment_list()
            self._refresh_description_editor()

    def _refresh_description_editor(self):
        seg = self._find_segment(self._selected_segment_id)
        self._updating_description_editor = True
        self.segment_description_edit.setPlainText(
            seg.description if seg else ""
        )
        self.segment_description_edit.setEnabled(bool(seg))
        self.btn_ai_description.setEnabled(
            bool(seg)
            and self._description_worker is None
            and self._segmentation_worker is None
        )
        self._updating_description_editor = False
        self._description_undo_segment_id = ""

    def _on_description_changed(self):
        if self._updating_description_editor:
            return
        seg = self._find_segment(self._selected_segment_id)
        if not seg:
            return
        description = self.segment_description_edit.toPlainText()
        if description == seg.description:
            return
        if self._description_undo_segment_id != seg.id:
            self._push_undo_snapshot()
            self._description_undo_segment_id = seg.id
        seg.description = description
        self._mark_dirty()

    def _on_ai_segment_clicked(self):
        if not self._video_path:
            QMessageBox.information(
                self,
                self.tr("AI Segmentation"),
                self.tr("Please open a video first."),
            )
            return
        if (
            self._description_worker is not None
            or self._segmentation_worker is not None
        ):
            return

        selected_seg = self._find_segment(self._selected_segment_id)

        dialog = AIDescriptionDialog(
            self._current_ai_model_text(),
            self._default_ai_segmentation_prompt(),
            self,
            self.tr("AI Segmentation"),
            True,
            selected_seg is None,
            selected_seg is not None,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        prompt = dialog.prompt()
        if not prompt:
            QMessageBox.information(
                self,
                self.tr("AI Segmentation"),
                self.tr("Prompt cannot be empty."),
            )
            return

        use_full_video = dialog.use_full_video()
        target_seg = None if use_full_video else selected_seg
        replace_segments = False
        if use_full_video and self._sidecar.segments:
            choice = QMessageBox.warning(
                self,
                self.tr("AI Segmentation"),
                self.tr(
                    "Existing segments will be overwritten by AI-generated "
                    "segments. Continue?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice != QMessageBox.StandardButton.Yes:
                return
            replace_segments = True
        elif target_seg is not None:
            choice = QMessageBox.warning(
                self,
                self.tr("AI Segmentation"),
                self.tr(
                    "The selected segment will be replaced by AI-generated "
                    "segments. Continue?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice != QMessageBox.StandardButton.Yes:
                return

        self.btn_ai_segment.setEnabled(False)
        self.btn_ai_description.setEnabled(False)
        self._ai_segmentation_replace = replace_segments
        self._ai_segmentation_target_segment_id = (
            target_seg.id if target_seg else ""
        )
        self._ai_segmentation_offset_ms = (
            target_seg.start_ms if target_seg else 0
        )
        self._ai_segmentation_max_ms = (
            target_seg.end_ms
            if target_seg
            else int(self._sidecar.duration_ms or 0)
        )
        self._segmentation_worker = VideoDescriptionWorker(
            self._video_path, target_seg, prompt, self
        )
        self._segmentation_worker.resultReady.connect(
            self._on_ai_segmentation_finished
        )
        self._segmentation_worker.progressChanged.connect(
            self._on_ai_segmentation_progress
        )
        self._segmentation_worker.start()
        self._show_ai_progress_dialog(self.tr("Generating segments..."))
        self._refresh_status_bar(extra=self.tr("Generating segments..."))

    def _on_ai_description_clicked(self):
        seg = self._find_segment(self._selected_segment_id)
        if not self._video_path or not seg:
            QMessageBox.information(
                self,
                self.tr("AI Description"),
                self.tr("Please select a segment first."),
            )
            return
        if (
            self._description_worker is not None
            or self._segmentation_worker is not None
        ):
            return

        dialog = AIDescriptionDialog(
            self._current_ai_model_text(),
            self._default_ai_description_prompt(seg),
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        prompt = dialog.prompt()
        if not prompt:
            QMessageBox.information(
                self,
                self.tr("AI Description"),
                self.tr("Prompt cannot be empty."),
            )
            return

        self.btn_ai_description.setEnabled(False)
        self.btn_ai_segment.setEnabled(False)
        self._description_worker = VideoDescriptionWorker(
            self._video_path, seg, prompt, self
        )
        self._description_worker.resultReady.connect(
            self._on_ai_description_finished
        )
        self._description_worker.progressChanged.connect(
            self._on_ai_description_progress
        )
        self._description_worker.start()
        self._show_ai_progress_dialog(self.tr("Generating description..."))
        self._refresh_status_bar(extra=self.tr("Generating description..."))

    def _current_ai_model_text(self):
        try:
            with open(get_models_config_path(), "r", encoding="utf-8") as f:
                settings = json.load(f).get("settings", {})
        except Exception:
            return self.tr("Not configured")

        provider = settings.get("provider") or self.tr("Not configured")
        model_id = settings.get("model_id") or self.tr("Not configured")
        return f"{model_id} ({provider})"

    def _default_ai_description_prompt(self, seg):
        return VIDEO_DESCRIPTION_PROMPT

    def _default_ai_segmentation_prompt(self):
        if self._tutorial_language() == "zh_cn":
            return VIDEO_SEGMENTATION_PROMPT
        return VIDEO_SEGMENTATION_PROMPT_EN

    def _show_ai_progress_dialog(self, text):
        progress = QProgressDialog(
            text,
            self.tr("Cancel"),
            0,
            0,
            self,
        )
        progress.setWindowTitle(self.tr("AI Description"))
        progress.setWindowModality(Qt.WindowModality.NonModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setStyleSheet(get_progress_dialog_style())
        progress.canceled.connect(self._cancel_ai_description)
        self._ai_progress_dialog = progress
        progress.show()

    def _on_ai_segmentation_progress(self, text):
        if self.sender() is not self._segmentation_worker:
            return
        if self._ai_progress_dialog is not None:
            self._ai_progress_dialog.setLabelText(self.tr(text))

    def _on_ai_description_progress(self, text):
        if self.sender() is not self._description_worker:
            return
        if self._ai_progress_dialog is not None:
            self._ai_progress_dialog.setLabelText(self.tr(text))

    def _cancel_ai_description(self):
        worker = self._description_worker
        if worker is not None:
            worker.cancel()
            self._cancelled_description_workers.append(worker)
            self._description_worker = None
        worker = self._segmentation_worker
        if worker is not None:
            worker.cancel()
            self._cancelled_segmentation_workers.append(worker)
            self._segmentation_worker = None
        if self._ai_progress_dialog is not None:
            self._ai_progress_dialog.close()
            self._ai_progress_dialog = None
        current_seg = self._find_segment(self._selected_segment_id)
        self.btn_ai_description.setEnabled(bool(current_seg))
        self.btn_ai_segment.setEnabled(bool(self._video_path))
        self._ai_segmentation_replace = False
        self._ai_segmentation_target_segment_id = ""
        self._ai_segmentation_offset_ms = 0
        self._ai_segmentation_max_ms = 0
        self._refresh_status_bar(extra=self.tr("AI generation cancelled."))

    def _on_ai_segmentation_finished(self, text, success, error_message):
        worker = self.sender()
        if worker is not self._segmentation_worker:
            if worker:
                if worker in self._cancelled_segmentation_workers:
                    self._cancelled_segmentation_workers.remove(worker)
                worker.deleteLater()
            return
        self._segmentation_worker = None
        if worker:
            worker.deleteLater()

        if self._ai_progress_dialog is not None:
            self._ai_progress_dialog.close()
            self._ai_progress_dialog = None

        self.btn_ai_segment.setEnabled(bool(self._video_path))
        current_seg = self._find_segment(self._selected_segment_id)
        self.btn_ai_description.setEnabled(bool(current_seg))
        if not success:
            if error_message == AI_CANCELLED:
                self._reset_ai_segmentation_state()
                self._refresh_status_bar()
                return
            QMessageBox.warning(
                self,
                self.tr("AI Segmentation"),
                error_message or self.tr("Failed to generate segments."),
            )
            self._reset_ai_segmentation_state()
            self._refresh_status_bar()
            return

        segments = self._segments_from_ai_events(
            text,
            self._ai_segmentation_offset_ms,
            self._ai_segmentation_max_ms,
        )
        if not segments:
            QMessageBox.warning(
                self,
                self.tr("AI Segmentation"),
                self.tr(
                    "No valid segments found in the AI response.\n\n"
                    "Response preview:\n{preview}"
                ).format(preview=(text or "").strip()[:600]),
            )
            self._reset_ai_segmentation_state()
            self._refresh_status_bar()
            return

        self._push_undo_snapshot()
        target_id = self._ai_segmentation_target_segment_id
        if target_id:
            self._replace_segment_with_segments(target_id, segments)
        elif self._ai_segmentation_replace:
            self._sidecar.segments = segments
        else:
            self._sidecar.segments.extend(segments)
        self._reset_ai_segmentation_state()
        labels = self.label_panel.labels()
        for seg in segments:
            if seg.label in labels:
                color = self.label_panel.color_for(seg.label)
            else:
                color = color_for_label(
                    seg.label, index=len(self._sidecar.labels)
                )
                labels.append(seg.label)
            self._sidecar.upsert_label(seg.label, color)
        self.label_panel.set_labels(
            self._sidecar.labels, self._sidecar.label_colors
        )
        self._selected_segment_id = segments[0].id
        self._mark_dirty()
        self._refresh_timeline()
        self._refresh_segment_list()
        self.timeline.set_selected(segments[0].id)
        self.segment_list.select(segments[0].id)
        self._refresh_description_editor()
        self._refresh_status_bar(
            extra=self.tr("Generated {n} segments.").format(n=len(segments))
        )

    def _on_ai_description_finished(self, text, success, error_message):
        worker = self.sender()
        if worker is not self._description_worker:
            if worker:
                if worker in self._cancelled_description_workers:
                    self._cancelled_description_workers.remove(worker)
                worker.deleteLater()
            return
        self._description_worker = None
        if worker:
            worker.deleteLater()

        if self._ai_progress_dialog is not None:
            self._ai_progress_dialog.close()
            self._ai_progress_dialog = None

        current_seg = self._find_segment(self._selected_segment_id)
        self.btn_ai_description.setEnabled(bool(current_seg))
        if not success:
            if error_message == AI_CANCELLED:
                self._refresh_status_bar()
                return
            QMessageBox.warning(
                self,
                self.tr("AI Description"),
                error_message or self.tr("Failed to generate description."),
            )
            self._refresh_status_bar()
            return
        seg_id = worker.segment.id if worker and worker.segment else ""
        seg = self._find_segment(seg_id)
        if not seg:
            self._refresh_status_bar()
            return

        text = (text or "").strip()
        if text and text != seg.description:
            self._push_undo_snapshot()
            seg.description = text
            if seg.id == self._selected_segment_id:
                self._updating_description_editor = True
                self.segment_description_edit.setPlainText(text)
                self._updating_description_editor = False
            self._mark_dirty()
        self._refresh_status_bar(extra=self.tr("Description generated."))

    def _replace_segment_with_segments(self, target_id, segments):
        for index, seg in enumerate(self._sidecar.segments):
            if seg.id == target_id:
                self._sidecar.segments[index : index + 1] = segments
                return
        self._sidecar.segments.extend(segments)

    def _reset_ai_segmentation_state(self):
        self._ai_segmentation_replace = False
        self._ai_segmentation_target_segment_id = ""
        self._ai_segmentation_offset_ms = 0
        self._ai_segmentation_max_ms = 0

    def _segments_from_ai_events(self, text, offset_ms=0, max_ms=0):
        try:
            payload = json.loads(self._json_payload_text(text))
        except (TypeError, ValueError):
            return []

        if isinstance(payload, dict):
            events = payload.get("events") or payload.get("segments")
        elif isinstance(payload, list):
            events = payload
        else:
            events = None
        if not isinstance(events, list):
            return []

        active_label = self.label_panel.active_label()
        max_ms = int(max_ms or self._sidecar.duration_ms or 0)
        segments = []
        for event in events:
            if not isinstance(event, dict):
                continue
            start_ms = self._time_text_to_ms(
                event.get("start_time")
                or event.get("start")
                or event.get("begin_time")
            )
            end_ms = self._time_text_to_ms(
                event.get("end_time") or event.get("end")
            )
            description = str(
                event.get("event")
                or event.get("description")
                or event.get("action")
                or ""
            ).strip()
            if start_ms is None or end_ms is None or end_ms <= start_ms:
                continue
            start_ms += int(offset_ms or 0)
            end_ms += int(offset_ms or 0)
            if max_ms:
                start_ms = max(0, min(start_ms, max_ms))
                end_ms = max(0, min(end_ms, max_ms))
            if end_ms <= start_ms:
                continue
            label = active_label or self.tr("片段{n}").format(
                n=len(segments) + 1
            )
            segments.append(
                Segment.new(
                    label,
                    start_ms,
                    end_ms,
                    fps=self._sidecar.fps,
                    description=description,
                )
            )
        return segments

    def _json_payload_text(self, text):
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
            text = re.sub(r"\s*```$", "", text)
        object_start = text.find("{")
        object_end = text.rfind("}")
        array_start = text.find("[")
        array_end = text.rfind("]")
        if array_start >= 0 and (
            object_start < 0 or array_start < object_start
        ):
            start, end = array_start, array_end
        elif object_start >= 0 and object_end >= object_start:
            start, end = object_start, object_end
        else:
            start, end = -1, -1
        if start < 0 or end < start:
            raise ValueError("No JSON payload found")
        return text[start : end + 1]

    def _time_text_to_ms(self, text):
        if not isinstance(text, str):
            return None
        match = re.fullmatch(r"\s*(\d{1,2})[:：](\d{2})[:：](\d{2})\s*", text)
        if not match:
            return None
        hours, minutes, seconds = [int(part) for part in match.groups()]
        return ((hours * 60 + minutes) * 60 + seconds) * 1000

    def _find_segment(self, seg_id):
        for s in self._sidecar.segments:
            if s.id == seg_id:
                return s
        return None

    def _has_segment_overlap(self, start_ms, end_ms, ignore_id=""):
        start_ms = int(start_ms)
        end_ms = int(end_ms)
        if end_ms <= start_ms:
            return True
        for seg in self._sidecar.segments:
            if ignore_id and seg.id == ignore_id:
                continue
            if max(start_ms, seg.start_ms) < min(end_ms, seg.end_ms):
                return True
        return False

    def _jump_prev_segment(self):
        if not self._sidecar.segments:
            return
        pos = self.player.position()
        candidates = sorted(self._sidecar.segments, key=lambda s: s.start_ms)
        prev = None
        for s in candidates:
            if s.start_ms < pos - 1:
                prev = s
        if prev:
            self.player.seek(prev.start_ms)
            self.timeline.set_selected(prev.id)
            self.segment_list.select(prev.id)

    def _jump_next_segment(self):
        if not self._sidecar.segments:
            return
        pos = self.player.position()
        candidates = sorted(self._sidecar.segments, key=lambda s: s.start_ms)
        for s in candidates:
            if s.start_ms > pos + 1:
                self.player.seek(s.start_ms)
                self.timeline.set_selected(s.id)
                self.segment_list.select(s.id)
                return

    # Mark in / out
    def _mark_in(self):
        if not self._video_path:
            return
        self._pending_in_ms = self.player.position()
        if (
            self._pending_out_ms is not None
            and self._pending_out_ms < self._pending_in_ms
        ):
            self._pending_out_ms = None
        self._refresh_status_bar()

    def _mark_out(self):
        if not self._video_path:
            return
        self._pending_out_ms = self.player.position()
        if (
            self._pending_in_ms is not None
            and self._pending_out_ms < self._pending_in_ms
        ):
            self._pending_in_ms, self._pending_out_ms = (
                self._pending_out_ms,
                self._pending_in_ms,
            )
        self._refresh_status_bar()

    def _commit_from_marks(self):
        if self._pending_in_ms is None or self._pending_out_ms is None:
            return
        start = min(self._pending_in_ms, self._pending_out_ms)
        end = max(self._pending_in_ms, self._pending_out_ms)
        if end - start < 50:
            QMessageBox.information(
                self,
                self.tr("Add segment"),
                self.tr("The marked range is too short."),
            )
            return
        label = self._ensure_active_label_or_prompt()
        if not label:
            return
        if self._has_segment_overlap(start, end):
            return
        self._push_undo_snapshot()
        seg = Segment.new(label, start, end, fps=self._sidecar.fps)
        self._sidecar.segments.append(seg)
        self._sidecar.upsert_label(label, self.label_panel.color_for(label))
        self._pending_in_ms = None
        self._pending_out_ms = None
        self._selected_segment_id = seg.id
        self._mark_dirty()
        self._refresh_timeline()
        self._refresh_segment_list()
        self.timeline.set_selected(seg.id)
        self.segment_list.select(seg.id)
        self._refresh_description_editor()
        self._refresh_status_bar()

    def _undo_last_segment(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self._segment_snapshot())
        self._restore_segment_snapshot(self._undo_stack.pop())
        self._mark_dirty()
        self._refresh_actions()
        self._refresh_status_bar()

    def _redo_last_segment(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._segment_snapshot())
        self._restore_segment_snapshot(self._redo_stack.pop())
        self._mark_dirty()
        self._refresh_actions()
        self._refresh_status_bar()

    def _split_selected_segment(self):
        pos = self.player.position()
        seg = self._split_target_segment(pos)
        if not seg:
            return
        first = Segment.new(
            seg.label,
            seg.start_ms,
            pos,
            fps=self._sidecar.fps,
            description=seg.description,
        )
        second = Segment.new(
            seg.label,
            pos,
            seg.end_ms,
            fps=self._sidecar.fps,
            description=seg.description,
        )
        index = self._sidecar.segments.index(seg)
        self._push_undo_snapshot(seg.id)
        self._sidecar.segments[index : index + 1] = [first, second]
        self._selected_segment_id = second.id
        self._mark_dirty()
        self._refresh_timeline()
        self._refresh_segment_list()
        self.timeline.set_selected(second.id)
        self.segment_list.select(second.id)
        self._refresh_description_editor()
        self._refresh_actions()

    def _can_split_at_playhead(self):
        if not self._video_path:
            return False
        return self._split_target_segment(self.player.position()) is not None

    def _split_target_segment(self, position_ms):
        selected = self._find_segment(self._selected_segment_id)
        if selected and selected.start_ms < position_ms < selected.end_ms:
            return selected
        return self._find_segment_at_position(position_ms)

    def _find_segment_at_position(self, position_ms):
        for seg in self._sidecar.segments:
            if seg.start_ms < position_ms < seg.end_ms:
                return seg
        return None

    def _export_selected_segment(self):
        seg = self._find_segment(self._selected_segment_id)
        if not self._video_path or not seg:
            return
        if not self._save_dirty_sidecar():
            return
        ffmpeg = detect_ffmpeg()
        if not ffmpeg:
            QMessageBox.warning(
                self,
                self.tr("Export segment"),
                self.tr(
                    "ffmpeg was not found. Install ffmpeg or "
                    "imageio-ffmpeg to export clips."
                ),
            )
            return
        save_path = self._selected_segment_export_path(seg)
        if not save_path:
            return
        ok, error = self._export_segment_mp4(ffmpeg, seg, save_path)
        if ok:
            QMessageBox.information(
                self,
                self.tr("Export complete"),
                self.tr("Exported segment to {path}").format(path=save_path),
            )
            return
        QMessageBox.warning(
            self,
            self.tr("Export failed"),
            error or self.tr("Failed to export the selected segment."),
        )

    def _selected_segment_export_path(self, seg):
        stem = os.path.splitext(os.path.basename(self._video_path))[0]
        default_name = f"{stem}_{seg.id or 'segment'}.mp4"
        default_path = os.path.join(
            os.path.dirname(self._video_path), default_name
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Export selected segment"),
            default_path,
            self.tr("MP4 Video (*.mp4)"),
        )
        if not path:
            return ""
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        return path

    def _export_segment_mp4(self, ffmpeg, seg, save_path):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        progress = QProgressDialog(
            self.tr("Exporting selected segment..."),
            "",
            0,
            0,
            self,
        )
        progress.setCancelButton(None)
        progress.setWindowTitle(self.tr("Export segment"))
        progress.setStyleSheet(
            get_progress_dialog_style(color=get_theme()["text"], height=20)
        )
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        QApplication.processEvents()
        try:
            ok, error = self._run_segment_ffmpeg(
                ffmpeg, seg, save_path, re_encode=False
            )
            if not ok:
                ok, error = self._run_segment_ffmpeg(
                    ffmpeg, seg, save_path, re_encode=True
                )
            return ok, error
        finally:
            progress.close()

    def _run_segment_ffmpeg(self, ffmpeg, seg, save_path, re_encode):
        start = ms_to_seconds(seg.start_ms)
        duration = ms_to_seconds(seg.end_ms) - start
        if duration <= 0:
            return False, self.tr("Selected segment duration is invalid.")
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            self._video_path,
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
        ]
        if re_encode:
            cmd += [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
            ]
        else:
            cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        cmd += ["-an", save_path]
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NO_WINDOW", 0
            )
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            **kwargs,
        )
        if result.returncode == 0:
            return True, ""
        error = (result.stderr or b"").decode("utf-8", errors="replace")
        return False, error.strip()[-400:]

    # Export
    def _on_export_clicked(self):
        if not self._video_path or not self._sidecar.segments:
            QMessageBox.information(
                self,
                self.tr("Export Dataset"),
                self.tr(
                    "Load a video and annotate at least one segment first."
                ),
            )
            return
        if not self._save_dirty_sidecar():
            return
        folder = os.path.dirname(self._video_path) if self._video_path else ""
        if not detect_ffmpeg():
            QMessageBox.warning(
                self,
                self.tr("Export Dataset"),
                self.tr(
                    "ffmpeg was not found. Install ffmpeg or "
                    "imageio-ffmpeg to export clips."
                ),
            )
            return
        dlg = ExportDialog(folder, self._video_path, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = dlg.to_config()
        if not cfg.output_dir:
            QMessageBox.warning(
                self,
                self.tr("Export Dataset"),
                self.tr("Output directory is required."),
            )
            return
        if not (cfg.include_video or cfg.include_rawframes):
            QMessageBox.warning(
                self,
                self.tr("Export Dataset"),
                self.tr("Pick at least one output format."),
            )
            return

        self._run_export(cfg)

    def _run_export(self, cfg):
        progress = QProgressDialog(
            self.tr("Exporting…"), self.tr("Cancel"), 0, 100, self
        )
        progress.setWindowTitle(self.tr("Export Dataset"))
        progress.setStyleSheet(
            get_progress_dialog_style(color=get_theme()["text"], height=20)
        )
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)

        controller = ExporterController(cfg, self)
        self._exporter = controller

        def on_progress(pct, msg):
            progress.setValue(int(pct))
            progress.setLabelText(self.tr("Exporting…\n{msg}").format(msg=msg))

        def on_log(line):
            logger.info(f"[video_classifier export] {line}")

        def on_finished(ok, msg):
            progress.setValue(100)
            progress.close()
            self._exporter = None
            if ok:
                QMessageBox.information(self, self.tr("Export complete"), msg)
            else:
                QMessageBox.warning(self, self.tr("Export failed"), msg)

        controller.progressChanged.connect(on_progress)
        controller.logged.connect(on_log)
        controller.finished.connect(on_finished)
        progress.canceled.connect(controller.cancel)
        controller.start()

    # Status bar
    def _refresh_status_bar(self, extra="", hover_ms=None):
        parts = []
        if self._video_path:
            sd = self._sidecar
            parts.append(
                self.tr("FPS: {f:.2f}  Size: {w}x{h}  Duration: {d}").format(
                    f=sd.fps or 0.0,
                    w=sd.width or "?",
                    h=sd.height or "?",
                    d=ms_to_timecode(sd.duration_ms or 0),
                )
            )
            if self._pending_in_ms is not None:
                parts.append(
                    self.tr("In: {t}").format(
                        t=ms_to_timecode(self._pending_in_ms)
                    )
                )
            if self._pending_out_ms is not None:
                parts.append(
                    self.tr("Out: {t}").format(
                        t=ms_to_timecode(self._pending_out_ms)
                    )
                )
            parts.append(
                self.tr("Segments: {n}").format(n=len(self._sidecar.segments))
            )
            if hover_ms is not None:
                parts.append(
                    self.tr("Cursor: {t}").format(t=ms_to_timecode(hover_ms))
                )
            if self._dirty:
                parts.append(self.tr("[modified]"))
        else:
            parts.append(self.tr("Drop or open a video to begin."))
        if extra:
            parts.append(extra)
        self.status_label.setText("    ".join(parts))

    def _refresh_actions(self):
        enabled = bool(self._video_path)
        edit_enabled = enabled and not self.player.is_playing()
        self.btn_export_frame.setEnabled(enabled)
        self.btn_zoom.setEnabled(enabled)
        self.btn_ai_segment.setEnabled(
            enabled
            and self._description_worker is None
            and self._segmentation_worker is None
        )
        for button in (
            self.btn_back_s,
            self.btn_prev_frame,
            self.btn_play,
            self.btn_next_frame,
            self.btn_forward_s,
        ):
            button.setEnabled(enabled)
        self.btn_export.setEnabled(
            enabled or os.path.isdir(self._guess_folder())
        )
        self.btn_undo.setEnabled(edit_enabled and bool(self._undo_stack))
        self.btn_reset.setEnabled(edit_enabled and bool(self._redo_stack))
        self._refresh_split_action()
        self.btn_export_selected.setEnabled(
            edit_enabled
            and bool(self._find_segment(self._selected_segment_id))
        )
        if detect_ffmpeg():
            self.btn_export.setToolTip(self.tr("Export dataset"))
            self.btn_export_selected.setToolTip(self.tr("Export segment"))
        else:
            self.btn_export.setToolTip(self.tr("ffmpeg required"))
            self.btn_export_selected.setToolTip(self.tr("ffmpeg required"))

    def _refresh_split_action(self):
        if not hasattr(self, "btn_split"):
            return
        enabled = bool(self._video_path) and not self.player.is_playing()
        enabled = enabled and self._can_split_at_playhead()
        if self.btn_split.isEnabled() != enabled:
            self.btn_split.setEnabled(enabled)

    def _guess_folder(self):
        return os.path.dirname(self._video_path) if self._video_path else ""

    # Drag & drop
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = url.toLocalFile()
                if p and p.lower().endswith(SUPPORTED_VIDEO_EXTS):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p and p.lower().endswith(SUPPORTED_VIDEO_EXTS):
                event.acceptProposedAction()
                self.load_video(p)
                return
        event.ignore()

    def release_resources(self):
        if self._released:
            return
        self._released = True
        if self._exporter is not None:
            try:
                self._exporter.cancel()
            except Exception:
                pass
        if self._description_worker is not None:
            try:
                self._description_worker.cancel()
                self._description_worker.wait(1000)
            except Exception:
                pass
            self._description_worker = None
        if self._segmentation_worker is not None:
            try:
                self._segmentation_worker.cancel()
                self._segmentation_worker.wait(1000)
            except Exception:
                pass
            self._segmentation_worker = None
        for worker in list(self._cancelled_description_workers):
            try:
                worker.cancel()
                worker.wait(1000)
            except Exception:
                pass
        self._cancelled_description_workers = []
        for worker in list(self._cancelled_segmentation_workers):
            try:
                worker.cancel()
                worker.wait(1000)
            except Exception:
                pass
        self._cancelled_segmentation_workers = []
        self._ai_segmentation_replace = False
        self._ai_segmentation_target_segment_id = ""
        self._ai_segmentation_offset_ms = 0
        self._ai_segmentation_max_ms = 0
        if self._ai_progress_dialog is not None:
            self._ai_progress_dialog.close()
            self._ai_progress_dialog = None
        try:
            self.player.release()
        except Exception:
            pass

    def closeEvent(self, event):
        if not self._save_dirty_sidecar():
            event.ignore()
            return
        self.release_resources()
        super().closeEvent(event)
