"""Compact pixel-edge control strip shown above the labeling canvas."""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets


class _EdgeComputationSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(int, object, object)


class EdgeComputationTask(QtCore.QRunnable):
    """Run one cancellable-by-generation edge calculation off the UI thread."""

    def __init__(self, request_id, function, *args, **kwargs):
        super().__init__()
        self.setAutoDelete(False)
        self.request_id = int(request_id)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = _EdgeComputationSignals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            result = self.function(*self.args, **self.kwargs)
            error = None
        except Exception as exc:  # pragma: no cover - defensive worker path
            result = None
            error = exc
        self.signals.finished.emit(self.request_id, result, error)


class PixelEdgeWidget(QtWidgets.QWidget):
    """Controls box segmentation and optional annotation edge refinement."""

    box_requested = QtCore.pyqtSignal()
    rectangle_requested = QtCore.pyqtSignal()
    confirm_requested = QtCore.pyqtSignal()
    cancel_requested = QtCore.pyqtSignal()
    settings_changed = QtCore.pyqtSignal(dict)
    double_click_changed = QtCore.pyqtSignal(bool)
    model_refine_changed = QtCore.pyqtSignal(bool)
    close_requested = QtCore.pyqtSignal()
    candidate_changed = QtCore.pyqtSignal(int)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self._settings = dict(settings or {})
        self._initializing = True
        self.setObjectName("pixelEdgeWidget")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 2, 2)
        outer.setSpacing(2)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setFixedHeight(43)
        content = QtWidgets.QWidget(scroll)
        row = QtWidgets.QHBoxLayout(content)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.box_button = QtWidgets.QPushButton("框选边缘", content)
        self.box_button.setToolTip(
            self.tr(
                "Draw a rectangle and preview the strongest nearby boundary"
            )
        )
        self.box_button.clicked.connect(
            lambda _checked=False: self.box_requested.emit()
        )
        row.addWidget(self.box_button)

        self.rectangle_button = QtWidgets.QPushButton(
            "长方形四角引导贴边", content
        )
        self.rectangle_button.setToolTip(
            "依次点击四个角画出四条引导边，完成后自动追踪附近像素分界"
        )
        self.rectangle_button.clicked.connect(
            lambda _checked=False: self.rectangle_requested.emit()
        )
        row.addWidget(self.rectangle_button)

        self.annotate_all = QtWidgets.QCheckBox("区域内所有目标", content)
        self.annotate_all.setChecked(
            bool(self._settings.get("annotate_all_in_box", False))
        )
        self.annotate_all.setToolTip(
            "未开启已有多边形批量贴合时，一次预览并标注框内所有独立目标"
        )
        row.addWidget(self.annotate_all)

        self.continuous_box = QtWidgets.QCheckBox("连续框选", content)
        self.continuous_box.setChecked(
            bool(self._settings.get("continuous_box", False))
        )
        self.continuous_box.setToolTip(
            "开启后每次确认都继续等待下一个框；关闭时只执行一次"
        )
        row.addWidget(self.continuous_box)

        self.boundary_side = QtWidgets.QComboBox(content)
        for label, value in (
            ("亮区外沿（白灰）", "bright"),
            ("自动边缘侧", "auto"),
            ("暗区外沿", "dark"),
        ):
            self.boundary_side.addItem(label, value)
        self.boundary_side.setCurrentIndex(
            max(
                0,
                self.boundary_side.findData(
                    self._settings.get("boundary_side", "bright")
                ),
            )
        )
        row.addWidget(self.boundary_side)
        row.addWidget(QtWidgets.QLabel("灰度分界", content))
        self.threshold_mode = QtWidgets.QComboBox(content)
        self.threshold_mode.addItem("自动", "auto")
        self.threshold_mode.addItem("手动", "manual")
        row.addWidget(self.threshold_mode)

        self.threshold = QtWidgets.QSpinBox(content)
        self.threshold.setRange(0, 255)
        self.threshold.setValue(int(self._settings.get("threshold", 64)))
        self.threshold.setToolTip(
            self.tr("Minimum local edge strength in manual mode")
        )
        row.addWidget(self.threshold)

        row.addWidget(QtWidgets.QLabel("微调", content))
        self.adjustment = QtWidgets.QSpinBox(content)
        self.adjustment.setRange(-127, 127)
        self.adjustment.setValue(
            int(self._settings.get("threshold_adjustment", 0))
        )
        self.adjustment.setToolTip(
            self.tr("Fine tune the automatically calculated edge threshold")
        )
        row.addWidget(self.adjustment)

        row.addWidget(QtWidgets.QLabel("点间距(px)", content))
        self.point_spacing = QtWidgets.QDoubleSpinBox(content)
        self.point_spacing.setRange(0.25, 100.0)
        self.point_spacing.setDecimals(2)
        self.point_spacing.setSingleStep(0.25)
        self.point_spacing.setValue(
            float(self._settings.get("point_spacing", 2.0))
        )
        self.point_spacing.setToolTip(
            "原图像素；直线段最大点间距，所有直角拐点始终保留"
        )
        row.addWidget(self.point_spacing)

        row.addWidget(QtWidgets.QLabel("搜索半径(px)", content))
        self.search_radius = QtWidgets.QDoubleSpinBox(content)
        self.search_radius.setRange(0.25, 50.0)
        self.search_radius.setDecimals(2)
        self.search_radius.setSingleStep(0.25)
        self.search_radius.setValue(
            float(self._settings.get("search_radius", 3.0))
        )
        self.search_radius.setToolTip(
            "从原标注向内外搜索边缘的距离，单位为原图像素"
        )
        row.addWidget(self.search_radius)

        self.precision_label = QtWidgets.QLabel("像素格直角走线", content)
        self.precision_label.setToolTip(
            "路径严格沿像素单元公共边；请预览确认是否选中所需白灰分界"
        )
        row.addWidget(self.precision_label)

        row.addWidget(QtWidgets.QLabel("平滑", content))
        self.blur_radius = QtWidgets.QSpinBox(content)
        self.blur_radius.setRange(0, 15)
        self.blur_radius.setValue(int(self._settings.get("blur_radius", 0)))
        row.addWidget(self.blur_radius)
        self.gap_repair = QtWidgets.QCheckBox("短缺口修复", content)
        self.gap_repair.setChecked(
            bool(self._settings.get("gap_repair", True))
        )
        self.gap_repair.setToolTip(
            "在严格面积变化限制内连接小缺口两端；只有完整闭环评分更好时才作为待确认预览"
        )
        row.addWidget(self.gap_repair)

        row.addWidget(QtWidgets.QLabel("最大缺口(px)", content))
        self.gap_bridge_max = QtWidgets.QSpinBox(content)
        self.gap_bridge_max.setRange(1, 100)
        self.gap_bridge_max.setValue(
            int(self._settings.get("gap_bridge_max", 24))
        )
        self.gap_bridge_max.setToolTip(
            "大缺口也只生成待确认预览；单位为原图像素"
        )
        row.addWidget(self.gap_bridge_max)

        row.addWidget(QtWidgets.QLabel("修补面积(%)", content))
        self.gap_bridge_ratio = QtWidgets.QDoubleSpinBox(content)
        self.gap_bridge_ratio.setRange(0.1, 50.0)
        self.gap_bridge_ratio.setDecimals(1)
        self.gap_bridge_ratio.setSingleStep(1.0)
        self.gap_bridge_ratio.setValue(
            float(self._settings.get("gap_bridge_ratio", 0.20)) * 100.0
        )
        self.gap_bridge_ratio.setToolTip(
            "修补像素占原区域的最大比例；大缺口可适当提高，结果仍需确认"
        )
        row.addWidget(self.gap_bridge_ratio)

        self.live_preview = QtWidgets.QCheckBox("实时预览", content)
        self.live_preview.setChecked(
            bool(self._settings.get("live_preview", True))
        )
        row.addWidget(self.live_preview)

        self.double_click = QtWidgets.QCheckBox(
            "双击贴合已有多边形（先预览）", content
        )
        self.double_click.setChecked(
            bool(self._settings.get("double_click_enabled", False))
        )
        row.addWidget(self.double_click)

        self.model_refine = QtWidgets.QCheckBox(
            "模型结果像素直角贴边", content
        )
        self.model_refine.setChecked(
            bool(self._settings.get("auto_label_enabled", False))
        )
        self.model_refine.setToolTip(
            "开启：模型推理→原图像素直角边缘分析→原有结果接收流程→画布；关闭：直接走原有结果接收流程。失败保留原结果。"
        )
        row.addWidget(self.model_refine)

        self.previous_candidate = QtWidgets.QPushButton("‹", content)
        self.next_candidate = QtWidgets.QPushButton("›", content)
        self.candidate_label = QtWidgets.QLabel("候选 0/0", content)
        self.previous_candidate.clicked.connect(
            lambda: self.candidate_changed.emit(-1)
        )
        self.next_candidate.clicked.connect(
            lambda: self.candidate_changed.emit(1)
        )
        row.addWidget(self.previous_candidate)
        row.addWidget(self.candidate_label)
        row.addWidget(self.next_candidate)

        self.confirm_button = QtWidgets.QPushButton(
            "确认贴合 (Ctrl+Enter)", content
        )
        self.confirm_button.setEnabled(False)
        self.confirm_button.setToolTip(
            "确认当前预览；已有标注可先拖动顶点人工修正"
        )
        self.confirm_button.clicked.connect(
            lambda _checked=False: self.confirm_requested.emit()
        )
        row.addWidget(self.confirm_button)

        self.cancel_button = QtWidgets.QPushButton("取消预览", content)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(
            lambda _checked=False: self.cancel_requested.emit()
        )
        row.addWidget(self.cancel_button)

        self.close_button = QtWidgets.QToolButton(content)
        self.close_button.setText("×")
        self.close_button.setToolTip(self.tr("Close pixel-edge panel"))
        self.close_button.clicked.connect(self._close_panel)
        row.addWidget(self.close_button)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        self.status_label = QtWidgets.QLabel(
            self.tr("Ready — draw a box or enable model-result refinement"),
            self,
        )
        self.status_label.setMinimumHeight(18)
        outer.addWidget(self.status_label)

        # The main window owns the configurable QAction, avoiding duplicate
        # hard-coded QShortcuts and making Settings conflict detection effective.

        mode = str(self._settings.get("threshold_mode", "auto"))
        index = self.threshold_mode.findData(mode)
        self.threshold_mode.setCurrentIndex(max(0, index))
        self._connect_controls()
        self._update_threshold_controls()
        self._initializing = False
        self.set_candidates(0, 0)

    def _connect_controls(self):
        self.gap_repair.toggled.connect(self._on_settings_changed)
        self.annotate_all.toggled.connect(self._on_settings_changed)
        self.continuous_box.toggled.connect(self._on_settings_changed)
        self.boundary_side.currentIndexChanged.connect(
            self._on_settings_changed
        )
        self.threshold_mode.currentIndexChanged.connect(
            self._on_settings_changed
        )
        self.threshold_mode.currentIndexChanged.connect(
            self._update_threshold_controls
        )
        for editor in (
            self.threshold,
            self.adjustment,
            self.point_spacing,
            self.search_radius,
            self.blur_radius,
            self.gap_bridge_max,
            self.gap_bridge_ratio,
        ):
            editor.valueChanged.connect(self._on_settings_changed)
        self.live_preview.toggled.connect(self._on_settings_changed)
        self.double_click.toggled.connect(self._on_double_click_changed)
        self.model_refine.toggled.connect(self._on_model_refine_changed)

    def _update_threshold_controls(self):
        automatic = self.threshold_mode.currentData() == "auto"
        self.threshold.setEnabled(not automatic)
        self.adjustment.setEnabled(automatic)

    def settings(self):
        values = dict(self._settings)
        values.pop("polarity", None)
        values.update(
            {
                "threshold_mode": self.threshold_mode.currentData(),
                "boundary_side": self.boundary_side.currentData(),
                "gap_repair": self.gap_repair.isChecked(),
                "gap_bridge_max": self.gap_bridge_max.value(),
                "gap_bridge_ratio": self.gap_bridge_ratio.value() / 100.0,
                "annotate_all_in_box": self.annotate_all.isChecked(),
                "continuous_box": self.continuous_box.isChecked(),
                "threshold": self.threshold.value(),
                "threshold_adjustment": self.adjustment.value(),
                "point_spacing": self.point_spacing.value(),
                "search_radius": self.search_radius.value(),
                "blur_radius": self.blur_radius.value(),
                "live_preview": self.live_preview.isChecked(),
                "double_click_enabled": self.double_click.isChecked(),
                "auto_label_enabled": self.model_refine.isChecked(),
            }
        )
        return values

    def apply_settings(self, settings):
        """Refresh controls after settings are changed outside the panel."""
        self._settings.update(settings or {})
        controls = (
            self.gap_repair,
            self.annotate_all,
            self.continuous_box,
            self.boundary_side,
            self.threshold_mode,
            self.threshold,
            self.adjustment,
            self.point_spacing,
            self.search_radius,
            self.blur_radius,
            self.gap_bridge_max,
            self.gap_bridge_ratio,
            self.live_preview,
            self.double_click,
            self.model_refine,
        )
        blockers = [QtCore.QSignalBlocker(control) for control in controls]
        self.gap_repair.setChecked(
            bool(self._settings.get("gap_repair", True))
        )
        self.gap_bridge_max.setValue(
            int(self._settings.get("gap_bridge_max", 24))
        )
        self.gap_bridge_ratio.setValue(
            float(self._settings.get("gap_bridge_ratio", 0.20)) * 100.0
        )
        self.annotate_all.setChecked(
            bool(self._settings.get("annotate_all_in_box", False))
        )
        self.continuous_box.setChecked(
            bool(self._settings.get("continuous_box", False))
        )
        self.boundary_side.setCurrentIndex(
            max(
                0,
                self.boundary_side.findData(
                    self._settings.get("boundary_side", "bright")
                ),
            )
        )
        mode_index = self.threshold_mode.findData(
            self._settings.get("threshold_mode", "auto")
        )
        self.threshold_mode.setCurrentIndex(max(0, mode_index))
        self.threshold.setValue(int(self._settings.get("threshold", 64)))
        self.adjustment.setValue(
            int(self._settings.get("threshold_adjustment", 0))
        )
        self.point_spacing.setValue(
            float(self._settings.get("point_spacing", 2.0))
        )
        self.search_radius.setValue(
            float(self._settings.get("search_radius", 3.0))
        )
        self.blur_radius.setValue(int(self._settings.get("blur_radius", 0)))
        self.live_preview.setChecked(
            bool(self._settings.get("live_preview", True))
        )
        self.double_click.setChecked(
            bool(self._settings.get("double_click_enabled", False))
        )
        self.model_refine.setChecked(
            bool(self._settings.get("auto_label_enabled", False))
        )
        self._update_threshold_controls()
        del blockers

    def _on_settings_changed(self, *_args):
        if self._initializing:
            return
        self._settings = self.settings()
        self.settings_changed.emit(dict(self._settings))

    def _on_double_click_changed(self, checked):
        self._on_settings_changed()
        self.double_click_changed.emit(bool(checked))

    def _on_model_refine_changed(self, checked):
        self._on_settings_changed()
        self.model_refine_changed.emit(bool(checked))

    def _close_panel(self):
        self.cancel_requested.emit()
        self.hide()
        self.close_requested.emit()

    def set_image_available(self, available):
        self.box_button.setEnabled(bool(available))
        self.rectangle_button.setEnabled(bool(available))

    def set_pending(self, pending):
        self.confirm_button.setEnabled(bool(pending))
        self.cancel_button.setEnabled(bool(pending))

    def set_candidates(self, index, count):
        self.candidate_label.setText(
            f"候选 {index + 1 if count else 0}/{count}"
        )
        self.previous_candidate.setEnabled(count > 1)
        self.next_candidate.setEnabled(count > 1)

    def update_confirm_shortcut(self, value):
        text = QtGui.QKeySequence(value or "").toString(
            QtGui.QKeySequence.SequenceFormat.NativeText
        )
        self.confirm_button.setText(
            "应用贴合" + (f" ({text})" if text else "")
        )

    def update_box_shortcut(self, value):
        text = QtGui.QKeySequence(value or "").toString(
            QtGui.QKeySequence.SequenceFormat.NativeText
        )
        self.box_button.setText("框选边缘" + (f" ({text})" if text else ""))

    def set_result_status(
        self,
        message,
        threshold_used=None,
        point_count=None,
        fit_error=None,
    ):
        details = []
        if threshold_used is not None:
            details.append(
                self.tr("edge threshold {value}").format(
                    value=int(round(threshold_used))
                )
            )
        if point_count is not None:
            details.append(self.tr("{count} points").format(count=point_count))
        if fit_error is not None:
            details.append("像素格路径已验证")
        suffix = f" ({', '.join(details)})" if details else ""
        self.status_label.setText(f"{message}{suffix}")
