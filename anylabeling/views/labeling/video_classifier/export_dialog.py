import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .exporter import ExportConfig
from .icons import apply_button_icon, theme_icon_color
from .style import (
    get_export_dialog_style,
    get_secondary_button_style,
    get_toolbar_button_style,
)


class ExportDialog(QDialog):
    def __init__(self, source_folder, current_video_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export Dataset"))
        self.setModal(True)
        self.setStyleSheet(get_export_dialog_style())
        self.resize(620, 400)
        self.setMinimumWidth(580)

        self._source_folder = source_folder
        self._current_video_path = current_video_path

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 16)
        layout.setSpacing(10)

        # Output dir
        out_section, out_layout = self._section(self.tr("Output"))
        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        self.out_edit = QLineEdit()
        default_out = os.path.join(
            source_folder or os.getcwd(),
            self._default_output_name(),
        )
        self.out_edit.setText(default_out)
        out_row.addWidget(self.out_edit, 1)
        browse = QPushButton(self.tr("Browse…"))
        browse.setStyleSheet(get_secondary_button_style(compact=True))
        apply_button_icon(
            browse, "folder", "svg", 16, theme_icon_color("text_secondary")
        )
        browse.clicked.connect(self._browse_output)
        out_row.addWidget(browse)
        out_layout.addLayout(out_row)
        layout.addWidget(out_section)

        # Formats
        format_section, fg_layout = self._section(self.tr("Formats"))
        self.chk_video = QCheckBox(self.tr("Video clips  (clip mp4)"))
        self.chk_video.setChecked(True)
        self.chk_rawframes = QCheckBox(
            self.tr("Raw frame sequences  (img_00001.jpg)")
        )
        fg_layout.addWidget(self.chk_video)
        fg_layout.addWidget(self.chk_rawframes)
        rf_row = QHBoxLayout()
        rf_row.setSpacing(8)
        rf_row.addWidget(QLabel(self.tr("RawFrames FPS (0 = use source):")))
        self.spin_rf_fps = self._spin_box(0, 0, 240)
        rf_row.addWidget(self.spin_rf_fps)
        rf_row.addStretch(1)
        fg_layout.addLayout(rf_row)
        layout.addWidget(format_section)

        # Options
        opt_section, og_layout = self._section(self.tr("Options"))
        self.chk_reencode = QCheckBox(
            self.tr("Re-encode clips (slower, frame-accurate cuts, libx264)")
        )
        self.chk_reencode.setToolTip(
            self.tr(
                "When disabled, clips are copied quickly around keyframes. "
                "Enable this for more accurate start/end cuts."
            )
        )
        self.chk_zip = QCheckBox(self.tr("Pack output as .zip"))
        self.chk_zip.setChecked(True)
        og_layout.addWidget(self.chk_reencode)
        og_layout.addWidget(self.chk_zip)
        layout.addWidget(opt_section)
        layout.addStretch(1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel = QPushButton(self.tr("Cancel"))
        cancel.setStyleSheet(get_secondary_button_style(compact=True))
        cancel.clicked.connect(self.reject)
        ok = QPushButton(self.tr("Export"))
        ok.setStyleSheet(get_toolbar_button_style(compact=True))
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        btn_row.addWidget(cancel)
        btn_row.addWidget(ok)
        layout.addLayout(btn_row)

    def _default_output_name(self):
        if self._current_video_path:
            stem = os.path.splitext(
                os.path.basename(self._current_video_path)
            )[0]
            if stem:
                return stem
        return "x-anylabeling-action"

    def _section(self, title, hint=None):
        section = QFrame(self)
        section.setObjectName("XvaExportSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(7)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setObjectName("XvaExportSectionTitle")
        header.addWidget(title_label)
        if hint:
            hint_label = QLabel(hint)
            hint_label.setObjectName("XvaExportHint")
            header.addSpacing(6)
            header.addWidget(hint_label)
        header.addStretch(1)
        layout.addLayout(header)
        return section, layout

    def _spin_box(self, value, minimum, maximum, suffix=""):
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setFixedWidth(86)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if suffix:
            spin.setSuffix(f" {suffix}")
        return spin

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self,
            self.tr("Select output directory"),
            self.out_edit.text() or self._source_folder or os.getcwd(),
        )
        if path:
            self.out_edit.setText(path)

    def to_config(self) -> ExportConfig:
        return ExportConfig(
            source_folder=self._source_folder or "",
            output_dir=self.out_edit.text().strip(),
            include_video=self.chk_video.isChecked(),
            include_rawframes=self.chk_rawframes.isChecked(),
            re_encode=self.chk_reencode.isChecked(),
            rawframe_fps=self.spin_rf_fps.value(),
            zip_output=self.chk_zip.isChecked(),
            only_video_path=self._current_video_path,
        )
