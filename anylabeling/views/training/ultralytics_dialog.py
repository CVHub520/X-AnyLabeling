import json
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QMessageBox,
    QScrollArea,
    QHeaderView,
    QGroupBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
)

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.qt import new_icon
from anylabeling.views.training.widgets.ultralytics_widgets import *
from anylabeling.services.auto_training.ultralytics._io import *
from anylabeling.services.auto_training.ultralytics.config import *
from anylabeling.services.auto_training.ultralytics.style import *
from anylabeling.services.auto_training.ultralytics.utils import *
from anylabeling.services.auto_training.ultralytics.trainer import (
    create_yolo_dataset,
    validate_basic_config,
)


class UltralyticsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.resize(*DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(*DEFAULT_WINDOW_SIZE)

        self.image_list = parent.image_list
        self.output_dir = parent.output_dir
        self.supported_shape = parent.supported_shape
        self.selected_task_type = None
        self.config_widgets = {}
        self.task_type_buttons = {}

        self.init_ui()
        self.refresh_dataset_summary()

    def init_ui(self):
        self.data_tab = QWidget()
        self.config_tab = QWidget()
        self.train_tab = QWidget()
        self.eval_tab = QWidget()
        self.export_tab = QWidget()

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.data_tab, self.tr("Data"))
        self.tab_widget.addTab(self.config_tab, self.tr("Config"))
        self.tab_widget.addTab(self.train_tab, self.tr("Train"))
        self.tab_widget.addTab(self.eval_tab, self.tr("Eval"))
        self.tab_widget.addTab(self.export_tab, self.tr("Export"))

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tab_widget)

        self.init_data_tab()
        self.init_config_tab()
        self.init_train_tab()

    # Data Tab
    def show_pose_config(self):
        """Show the pose config field"""
        if hasattr(self, 'pose_config_label'):
            self.pose_config_label.setVisible(True)
            self.config_widgets["pose_config"].setVisible(True)

            for i in range(self.pose_config_layout.count()):
                widget = self.pose_config_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(True)

    def hide_pose_config(self):
        """Hide the pose config field"""
        if hasattr(self, 'pose_config_label'):
            self.pose_config_label.setVisible(False)
            self.config_widgets["pose_config"].setVisible(False)

            for i in range(self.pose_config_layout.count()):
                widget = self.pose_config_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(False)

    def on_task_type_selected(self, task_type):
        if self.selected_task_type == task_type:
            self.selected_task_type = None
            self.task_type_buttons[task_type].set_selected(False)
            self.hide_pose_config()
        else:
            if self.selected_task_type:
                self.task_type_buttons[self.selected_task_type].set_selected(False)
            self.selected_task_type = task_type
            self.task_type_buttons[task_type].set_selected(True)

            if task_type.lower() == "pose":
                self.show_pose_config()
            else:
                self.hide_pose_config()

    def create_task_handler(self, task_type):
        def handler():
            self.on_task_type_selected(task_type)
        return handler

    def init_task_configuration(self, parent_layout):
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)

        task_type_layout = QHBoxLayout()
        task_type_layout.addWidget(QLabel(self.tr("Task Type:")))
        for task_type in TASK_TYPES:
            button = CustomQPushButton(task_type)
            button.clicked.connect(self.create_task_handler(task_type))
            task_type_layout.addWidget(button)
            self.task_type_buttons[task_type] = button

        task_type_layout.addStretch()
        config_layout.addLayout(task_type_layout)
        parent_layout.addWidget(config_widget)

    def refresh_dataset_summary(self):
        if not self.image_list or not self.supported_shape:
            self.summary_table.clear()
            self.summary_table.setRowCount(0)
            self.summary_table.setColumnCount(0)
            return

        table_data = get_statistics_table_data(self.image_list, self.supported_shape, self.output_dir)
        if not table_data:
            self.summary_table.clear()
            self.summary_table.setRowCount(0)
            self.summary_table.setColumnCount(0)
            return

        headers = table_data[0]
        data_rows = table_data[1:]
        self.summary_table.setRowCount(len(data_rows))
        self.summary_table.setColumnCount(len(headers))
        self.summary_table.setHorizontalHeaderLabels(headers)

        for row, row_data in enumerate(data_rows):
            for col, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                self.summary_table.setItem(row, col, item)

        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def load_images(self):
        self.parent().open_folder_dialog()
        self.image_list = self.parent().image_list
        self.refresh_dataset_summary()

    def init_dataset_summary(self, parent_layout):
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.addWidget(QLabel(self.tr("Dataset Summary:")))

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.summary_table = QTableWidget()
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        scroll_area.setWidget(self.summary_table)
        summary_layout.addWidget(scroll_area)
        parent_layout.addWidget(summary_widget)

    def proceed_to_config(self):
        is_valid, error_message = validate_task_requirements(
            self.selected_task_type, self.image_list, self.output_dir
        )

        if not is_valid:
            QMessageBox.warning(self, self.tr("Validation Error"), error_message)
            return

        self.tab_widget.setCurrentIndex(1)

    def init_actions(self, parent_layout):
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)

        self.load_images_button = SecondaryButton(self.tr("Load Images"))
        self.load_images_button.clicked.connect(self.load_images)
        actions_layout.addWidget(self.load_images_button)
        actions_layout.addStretch()

        self.next_button = PrimaryButton(self.tr("Next"))
        self.next_button.clicked.connect(self.proceed_to_config)
        actions_layout.addWidget(self.next_button)
        parent_layout.addWidget(actions_widget)

    def init_data_tab(self):
        layout = QVBoxLayout(self.data_tab)
        self.init_task_configuration(layout)
        self.init_dataset_summary(layout)
        self.init_actions(layout)

    # Config Tab
    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Select Model File"), "", "Model Files (*.pt);;All Files (*)"
        )
        if file_path:
            self.config_widgets["model"].setText(file_path)

    def browse_data_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Select Data File"), "", "Text Files (*.yaml);;All Files (*)"
        )
        if file_path:
            self.config_widgets["data"].setText(file_path)

    def browse_pose_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Select Pose Config File"), "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            self.config_widgets["pose_config"].setText(file_path)

    def setup_cuda_checkboxes(self, device_count):
        if not hasattr(self, '_cuda_layout') or not self._cuda_layout:
            self._cuda_layout = QHBoxLayout(self.device_checkboxes)
            self._cuda_layout.setContentsMargins(0, 0, 0, 0)
            self._cuda_layout.setSpacing(5)
        else:
            while self._cuda_layout.count():
                child = self._cuda_layout.takeAt(0)
                if child.widget():
                    child.widget().setParent(None)
        
        for i in range(device_count):
            checkbox = CustomCheckBox(f"GPU {i}")
            checkbox.setMaximumHeight(20)
            checkbox.setChecked(True)  # Default check all GPUs
            self._cuda_layout.addWidget(checkbox)

    def on_device_changed(self, device_text):
        if device_text == "cuda":
            try:
                import torch
                device_count = torch.cuda.device_count()
                self.setup_cuda_checkboxes(device_count)
                self.device_checkboxes.setVisible(True)
            except ImportError:
                self.device_checkboxes.setVisible(False)
        else:
            self.device_checkboxes.setVisible(False)

    def init_basic_settings(self, parent_layout):
        group = QGroupBox(self.tr("Basic Settings"))
        layout = QFormLayout(group)

        self.config_widgets["project"] = CustomLineEdit()
        selected_task_type = self.selected_task_type.lower() if self.selected_task_type else "detect"
        text_project = os.path.join(DEFAULT_PROJECT_DIR, selected_task_type)
        self.config_widgets["project"].setText(text_project)
        layout.addRow("Project:", self.config_widgets["project"])

        self.config_widgets["name"] = CustomLineEdit()
        self.config_widgets["name"].setText("exp")
        layout.addRow("Name:", self.config_widgets["name"])

        model_layout = QHBoxLayout()
        self.config_widgets["model"] = CustomLineEdit()
        model_browse_btn = SecondaryButton("Browse")
        model_browse_btn.clicked.connect(self.browse_model_file)
        model_layout.addWidget(self.config_widgets["model"])
        model_layout.addWidget(model_browse_btn)
        layout.addRow("Model:", model_layout)

        data_layout = QHBoxLayout()
        self.config_widgets["data"] = CustomLineEdit()
        data_browse_btn = SecondaryButton("Browse")
        data_browse_btn.clicked.connect(self.browse_data_file)
        data_layout.addWidget(self.config_widgets["data"])
        data_layout.addWidget(data_browse_btn)
        layout.addRow("Data:", data_layout)

        pose_config_layout = QHBoxLayout()
        self.config_widgets["pose_config"] = CustomLineEdit()
        pose_config_browse_btn = SecondaryButton("Browse")
        pose_config_browse_btn.clicked.connect(self.browse_pose_config_file)
        pose_config_layout.addWidget(self.config_widgets["pose_config"])
        pose_config_layout.addWidget(pose_config_browse_btn)

        self.pose_config_label = QLabel("Pose Config:")
        layout.addRow(self.pose_config_label, pose_config_layout)
        self.pose_config_layout = pose_config_layout

        self.pose_config_label.setVisible(False)
        self.config_widgets["pose_config"].setVisible(False)
        pose_config_browse_btn.setVisible(False)

        device_layout = QHBoxLayout()
        self.config_widgets["device"] = CustomComboBox()
        self.config_widgets["device"].addItems(DEVICE_OPTIONS)
        self.device_checkboxes = QWidget()
        self.device_checkboxes.setVisible(False)
        self.config_widgets["device"].currentTextChanged.connect(self.on_device_changed)
        device_layout.addWidget(self.config_widgets["device"])
        device_layout.addWidget(self.device_checkboxes)
        layout.addRow("Device:", device_layout)
        self.on_device_changed(self.config_widgets["device"].currentText())

        dataset_layout = QHBoxLayout()
        self.config_widgets["dataset_ratio"] = CustomSlider(Qt.Horizontal)
        self.config_widgets["dataset_ratio"].setRange(5, 95)
        self.config_widgets["dataset_ratio"].setValue(80)
        self.dataset_ratio_label = QLabel("0.8")
        self.config_widgets["dataset_ratio"].valueChanged.connect(
            lambda v: self.dataset_ratio_label.setText(str(v/100.0))
        )
        dataset_layout.addWidget(self.config_widgets["dataset_ratio"])
        dataset_layout.addWidget(self.dataset_ratio_label)
        layout.addRow("Dataset Ratio:", dataset_layout)

        parent_layout.addWidget(group)

    def toggle_advanced_settings(self):
        """Toggle the visibility of advanced settings"""
        if self.advanced_content_widget.isVisible():
            self.advanced_content_widget.setVisible(False)
            self.advanced_toggle_btn.setIcon(QIcon(new_icon("caret-down", "svg")))
        else:
            self.advanced_content_widget.setVisible(True)
            self.advanced_toggle_btn.setIcon(QIcon(new_icon("caret-up", "svg")))

    def init_train_settings(self, parent_layout):
        group = QGroupBox(self.tr("Train Settings"))
        layout = QVBoxLayout(group)

        # Basic settings
        basic_group = QGroupBox(self.tr("Basic"))
        basic_layout = QHBoxLayout(basic_group)
        basic_layout.addWidget(QLabel("Epochs:"))
        self.config_widgets["epochs"] = CustomSpinBox()
        self.config_widgets["epochs"].setRange(1, 10000)
        self.config_widgets["epochs"].setValue(DEFAULT_TRAINING_CONFIG["epochs"])
        basic_layout.addWidget(self.config_widgets["epochs"])

        basic_layout.addWidget(QLabel("Batch:"))
        self.config_widgets["batch"] = CustomSpinBox()
        self.config_widgets["batch"].setRange(-1, 8192)
        self.config_widgets["batch"].setValue(DEFAULT_TRAINING_CONFIG["batch"])
        basic_layout.addWidget(self.config_widgets["batch"])

        basic_layout.addWidget(QLabel("Image Size:"))
        self.config_widgets["imgsz"] = CustomSpinBox()
        self.config_widgets["imgsz"].setRange(32, 8192)
        self.config_widgets["imgsz"].setValue(DEFAULT_TRAINING_CONFIG["imgsz"])
        basic_layout.addWidget(self.config_widgets["imgsz"])

        basic_layout.addWidget(QLabel("Workers:"))
        self.config_widgets["workers"] = CustomSpinBox()
        self.config_widgets["workers"].setRange(0, NUM_WORKERS)
        self.config_widgets["workers"].setValue(DEFAULT_TRAINING_CONFIG["workers"])
        basic_layout.addWidget(self.config_widgets["workers"])

        basic_layout.addWidget(QLabel("Classes:"))
        self.config_widgets["classes"] = CustomLineEdit()
        self.config_widgets["classes"].setText(DEFAULT_TRAINING_CONFIG["classes"])
        self.config_widgets["classes"].setPlaceholderText(
            self.tr("Class indices (e.g., 0,1,2) or leave empty for all")
        )
        basic_layout.addWidget(self.config_widgets["classes"])

        self.config_widgets["single_cls"] = CustomCheckBox("Single Class")
        self.config_widgets["single_cls"].setChecked(DEFAULT_TRAINING_CONFIG["single_cls"])
        basic_layout.addWidget(self.config_widgets["single_cls"])

        basic_layout.addStretch()
        layout.addWidget(basic_group)

        # Advanced settings
        advanced_container = QWidget()
        advanced_container_layout = QVBoxLayout(advanced_container)
        advanced_container_layout.setContentsMargins(0, 0, 0, 0)

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)

        advanced_label = QLabel(self.tr("Advanced Settings"))
        advanced_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(advanced_label)

        # Collapse/Expand button
        self.advanced_toggle_btn = QPushButton()
        self.advanced_toggle_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.advanced_toggle_btn.setStyleSheet(get_advanced_toggle_btn_style())
        self.advanced_toggle_btn.setIcon(QIcon(new_icon("caret-down", "svg")))
        self.advanced_toggle_btn.clicked.connect(self.toggle_advanced_settings)
        header_layout.addWidget(self.advanced_toggle_btn)
        header_layout.addStretch()
        advanced_container_layout.addWidget(header_widget)

        self.advanced_content_widget = QWidget()
        self.advanced_content_widget.setVisible(False)
        advanced_layout = QVBoxLayout(self.advanced_content_widget)

        # 1. Training Strategy
        strategy_group = QGroupBox("Training Strategy")
        strat_layout = QHBoxLayout(strategy_group)
        strat_layout.addWidget(QLabel("Time (h):"))
        self.config_widgets["time"] = CustomDoubleSpinBox()
        self.config_widgets["time"].setValue(DEFAULT_TRAINING_CONFIG["time"])
        self.config_widgets["time"].setSpecialValueText("None")
        strat_layout.addWidget(self.config_widgets["time"])

        strat_layout.addWidget(QLabel("Patience:"))
        self.config_widgets["patience"] = CustomSpinBox()
        self.config_widgets["patience"].setRange(1, 10000)
        self.config_widgets["patience"].setValue(DEFAULT_TRAINING_CONFIG["patience"])
        strat_layout.addWidget(self.config_widgets["patience"])

        strat_layout.addWidget(QLabel("Close Mosaic:"))
        self.config_widgets["close_mosaic"] = CustomSpinBox()
        self.config_widgets["close_mosaic"].setRange(0, 1000)
        self.config_widgets["close_mosaic"].setValue(DEFAULT_TRAINING_CONFIG["close_mosaic"])
        strat_layout.addWidget(self.config_widgets["close_mosaic"])

        strat_layout.addWidget(QLabel("Optimizer:"))
        self.config_widgets["optimizer"] = CustomComboBox()
        self.config_widgets["optimizer"].addItems(OPTIMIZER_OPTIONS)
        strat_layout.addWidget(self.config_widgets["optimizer"])

        self.config_widgets["cos_lr"] = CustomCheckBox("Cosine LR")
        self.config_widgets["cos_lr"].setChecked(DEFAULT_TRAINING_CONFIG["cos_lr"])
        strat_layout.addWidget(self.config_widgets["cos_lr"])
        self.config_widgets["amp"] = CustomCheckBox("AMP")
        self.config_widgets["amp"].setChecked(DEFAULT_TRAINING_CONFIG["amp"])
        strat_layout.addWidget(self.config_widgets["amp"])
        self.config_widgets["multi_scale"] = CustomCheckBox("Multi Scale")
        self.config_widgets["multi_scale"].setChecked(DEFAULT_TRAINING_CONFIG["multi_scale"])
        strat_layout.addWidget(self.config_widgets["multi_scale"])
        strat_layout.addStretch()
        advanced_layout.addWidget(strategy_group)

        # 2. Learning Rate
        lr_group = QGroupBox("Learning Rate")
        lr_layout = QHBoxLayout(lr_group)
        lr_layout.addWidget(QLabel("LR0:"))
        self.config_widgets["lr0"] = CustomDoubleSpinBox()
        self.config_widgets["lr0"].setDecimals(6)
        self.config_widgets["lr0"].setValue(DEFAULT_TRAINING_CONFIG["lr0"])
        lr_layout.addWidget(self.config_widgets["lr0"])

        lr_layout.addWidget(QLabel("LRF:"))
        self.config_widgets["lrf"] = CustomDoubleSpinBox()
        self.config_widgets["lrf"].setDecimals(6)
        self.config_widgets["lrf"].setValue(DEFAULT_TRAINING_CONFIG["lrf"])
        lr_layout.addWidget(self.config_widgets["lrf"])

        lr_layout.addWidget(QLabel("Momentum:"))
        self.config_widgets["momentum"] = CustomDoubleSpinBox()
        self.config_widgets["momentum"].setDecimals(3)
        self.config_widgets["momentum"].setValue(DEFAULT_TRAINING_CONFIG["momentum"])
        lr_layout.addWidget(self.config_widgets["momentum"])

        lr_layout.addWidget(QLabel("Weight Decay:"))
        self.config_widgets["weight_decay"] = CustomDoubleSpinBox()
        self.config_widgets["weight_decay"].setDecimals(6)
        self.config_widgets["weight_decay"].setValue(DEFAULT_TRAINING_CONFIG["weight_decay"])
        lr_layout.addWidget(self.config_widgets["weight_decay"])
        lr_layout.addStretch()
        advanced_layout.addWidget(lr_group)

        # 3. Warmup Parameters
        warmup_group = QGroupBox("Warmup Parameters")
        warmup_layout = QHBoxLayout(warmup_group)
        warmup_layout.addWidget(QLabel("Warmup Epochs:"))
        self.config_widgets["warmup_epochs"] = CustomDoubleSpinBox()
        self.config_widgets["warmup_epochs"].setDecimals(1)
        self.config_widgets["warmup_epochs"].setValue(DEFAULT_TRAINING_CONFIG["warmup_epochs"])
        warmup_layout.addWidget(self.config_widgets["warmup_epochs"])

        warmup_layout.addWidget(QLabel("Warmup Momentum:"))
        self.config_widgets["warmup_momentum"] = CustomDoubleSpinBox()
        self.config_widgets["warmup_momentum"].setDecimals(3)
        self.config_widgets["warmup_momentum"].setValue(DEFAULT_TRAINING_CONFIG["warmup_momentum"])
        warmup_layout.addWidget(self.config_widgets["warmup_momentum"])

        warmup_layout.addWidget(QLabel("Warmup Bias LR:"))
        self.config_widgets["warmup_bias_lr"] = CustomDoubleSpinBox()
        self.config_widgets["warmup_bias_lr"].setDecimals(3)
        self.config_widgets["warmup_bias_lr"].setValue(DEFAULT_TRAINING_CONFIG["warmup_bias_lr"])
        warmup_layout.addWidget(self.config_widgets["warmup_bias_lr"])
        warmup_layout.addStretch()
        advanced_layout.addWidget(warmup_group)

        # 4. Augmentation Settings
        augment_group = QGroupBox("Augmentation Settings")
        augment_layout = QVBoxLayout(augment_group)
        augment_params = [
            ("hsv_h", "HSV Hue:", DEFAULT_TRAINING_CONFIG["hsv_h"], 0.0, 1.0, 3),
            ("hsv_s", "HSV Saturation:", DEFAULT_TRAINING_CONFIG["hsv_s"], 0.0, 1.0, 3),
            ("hsv_v", "HSV Value:", DEFAULT_TRAINING_CONFIG["hsv_v"], 0.0, 1.0, 3),
            ("degrees", "Rotation Degrees:", DEFAULT_TRAINING_CONFIG["degrees"], -180.0, 180.0, 1),
            ("translate", "Translate:", DEFAULT_TRAINING_CONFIG["translate"], 0.0, 1.0, 3),
            ("scale", "Scale:", DEFAULT_TRAINING_CONFIG["scale"], 0.0, 2.0, 3),
            ("shear", "Shear:", DEFAULT_TRAINING_CONFIG["shear"], -45.0, 45.0, 1),
            ("perspective", "Perspective:", DEFAULT_TRAINING_CONFIG["perspective"], 0.0, 0.001, 6),
        ]

        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(10)
        grid_layout.setVerticalSpacing(5)
        for i, (param, label, default, min_val, max_val, decimals) in enumerate(augment_params):
            row = i // 4
            col = (i % 4) * 2

            label_widget = QLabel(label)
            label_widget.setMinimumWidth(80)
            grid_layout.addWidget(label_widget, row, col)

            widget = CustomDoubleSpinBox()
            widget.setRange(min_val, max_val)
            widget.setDecimals(decimals)
            widget.setValue(default)
            widget.setMinimumWidth(80)
            self.config_widgets[param] = widget
            grid_layout.addWidget(widget, row, col + 1)

        for col in range(8, 10):
            grid_layout.setColumnStretch(col, 1)
        augment_layout.addLayout(grid_layout)
        advanced_layout.addWidget(augment_group)

        # 5. Regularization
        reg_group = QGroupBox("Regularization")
        reg_layout = QHBoxLayout(reg_group)
        reg_layout.addWidget(QLabel("Dropout:"))
        self.config_widgets["dropout"] = CustomDoubleSpinBox()
        self.config_widgets["dropout"].setDecimals(3)
        self.config_widgets["dropout"].setValue(DEFAULT_TRAINING_CONFIG["dropout"])
        reg_layout.addWidget(self.config_widgets["dropout"])

        reg_layout.addWidget(QLabel("Fraction:"))
        self.config_widgets["fraction"] = CustomDoubleSpinBox()
        self.config_widgets["fraction"].setDecimals(3)
        self.config_widgets["fraction"].setValue(DEFAULT_TRAINING_CONFIG["fraction"])
        reg_layout.addWidget(self.config_widgets["fraction"])

        self.config_widgets["rect"] = CustomCheckBox("Rectangular")
        self.config_widgets["rect"].setChecked(DEFAULT_TRAINING_CONFIG["rect"])
        reg_layout.addWidget(self.config_widgets["rect"])
        reg_layout.addStretch()
        advanced_layout.addWidget(reg_group)

        # 6. Loss Weights
        loss_group = QGroupBox("Loss Weights")
        loss_layout = QHBoxLayout(loss_group)
        loss_layout.addWidget(QLabel("Box:"))
        self.config_widgets["box"] = CustomDoubleSpinBox()
        self.config_widgets["box"].setDecimals(2)
        self.config_widgets["box"].setValue(DEFAULT_TRAINING_CONFIG["box"])
        loss_layout.addWidget(self.config_widgets["box"])

        loss_layout.addWidget(QLabel("Cls:"))
        self.config_widgets["cls"] = CustomDoubleSpinBox()
        self.config_widgets["cls"].setDecimals(2)
        self.config_widgets["cls"].setValue(DEFAULT_TRAINING_CONFIG["cls"])
        loss_layout.addWidget(self.config_widgets["cls"])

        loss_layout.addWidget(QLabel("DFL:"))
        self.config_widgets["dfl"] = CustomDoubleSpinBox()
        self.config_widgets["dfl"].setDecimals(2)
        self.config_widgets["dfl"].setValue(DEFAULT_TRAINING_CONFIG["dfl"])
        loss_layout.addWidget(self.config_widgets["dfl"])

        loss_layout.addWidget(QLabel("Pose:"))
        self.config_widgets["pose"] = CustomDoubleSpinBox()
        self.config_widgets["pose"].setDecimals(2)
        self.config_widgets["pose"].setValue(DEFAULT_TRAINING_CONFIG["pose"])
        loss_layout.addWidget(self.config_widgets["pose"])

        loss_layout.addWidget(QLabel("Kobj:"))
        self.config_widgets["kobj"] = CustomDoubleSpinBox()
        self.config_widgets["kobj"].setDecimals(2)
        self.config_widgets["kobj"].setValue(DEFAULT_TRAINING_CONFIG["kobj"])
        loss_layout.addWidget(self.config_widgets["kobj"])
        loss_layout.addStretch()
        advanced_layout.addWidget(loss_group)

        # 7. Checkpoint and Validation
        ckpt_group = QGroupBox("Checkpoint and Validation")
        ckpt_layout = QHBoxLayout(ckpt_group)
        ckpt_layout.addWidget(QLabel("Save Period:"))
        self.config_widgets["save_period"] = CustomSpinBox()
        self.config_widgets["save_period"].setRange(-1, 1000)
        self.config_widgets["save_period"].setValue(DEFAULT_TRAINING_CONFIG["save_period"])
        self.config_widgets["save_period"].setSpecialValueText("Disabled")
        ckpt_layout.addWidget(self.config_widgets["save_period"])

        self.config_widgets["val"] = CustomCheckBox("Validation")
        self.config_widgets["val"].setChecked(DEFAULT_TRAINING_CONFIG["val"])
        ckpt_layout.addWidget(self.config_widgets["val"])
        self.config_widgets["plots"] = CustomCheckBox("Plots")
        self.config_widgets["plots"].setChecked(DEFAULT_TRAINING_CONFIG["plots"])
        ckpt_layout.addWidget(self.config_widgets["plots"])
        self.config_widgets["save"] = CustomCheckBox("Save")
        self.config_widgets["save"].setChecked(DEFAULT_TRAINING_CONFIG["save"])
        ckpt_layout.addWidget(self.config_widgets["save"])
        self.config_widgets["resume"] = CustomCheckBox("Resume")
        self.config_widgets["resume"].setChecked(DEFAULT_TRAINING_CONFIG["resume"])
        ckpt_layout.addWidget(self.config_widgets["resume"])
        self.config_widgets["cache"] = CustomCheckBox("Cache")
        self.config_widgets["cache"].setChecked(DEFAULT_TRAINING_CONFIG["cache"])
        ckpt_layout.addWidget(self.config_widgets["cache"])
        ckpt_layout.addStretch()
        advanced_layout.addWidget(ckpt_group)

        advanced_container_layout.addWidget(self.advanced_content_widget)
        layout.addWidget(advanced_container)
        parent_layout.addWidget(group)

    def load_config_to_ui(self, config):
        def set_widget_value(key, value):
            if key not in self.config_widgets:
                return

            widget = self.config_widgets[key]
            widget_type = type(widget).__name__

            try:
                if widget_type == "CustomLineEdit":
                    widget.setText(str(value))
                elif widget_type in ["CustomSpinBox", "CustomDoubleSpinBox"]:
                    widget.setValue(value)
                elif widget_type == "CustomComboBox":
                    if isinstance(value, str):
                        index = widget.findText(value)
                        if index >= 0:
                            widget.setCurrentIndex(index)
                    else:
                        widget.setCurrentIndex(value)
                elif widget_type == "CustomCheckBox":
                    widget.setChecked(bool(value))
                elif widget_type == "CustomSlider":
                    widget.setValue(value)
            except Exception as e:
                logger.warning(f"Failed to set value for widget {key}: {e}")

        sections_to_process = ["basic", "train", "augment", "strategy", "learning_rate", 
                               "warmup", "regularization", "loss_weights", "checkpoint"]
        for section in sections_to_process:
            if section in config:
                for key, value in config[section].items():
                    if key == "dataset_ratio":
                        if 0 <= value <= 1:
                            self.config_widgets[key].setValue(int(value * 100))
                            self.dataset_ratio_label.setText(str(value))
                        else:
                            self.config_widgets[key].setValue(int(value))
                            self.dataset_ratio_label.setText(str(value / 100.0))
                    elif key == "device":
                        index = self.config_widgets[key].findText(str(value))
                        if index >= 0:
                            self.config_widgets[key].setCurrentIndex(index)
                            self.on_device_changed(str(value))
                    elif key == "optimizer":
                        index = self.config_widgets[key].findText(str(value))
                        if index >= 0:
                            self.config_widgets[key].setCurrentIndex(index)
                    elif key == "pose_config":
                        if value:
                            self.config_widgets[key].setText(value)
                            self.on_task_type_selected("pose")
                    else:
                        set_widget_value(key, value)

        for key, value in config.items():
            if key not in sections_to_process and key in self.config_widgets:
                set_widget_value(key, value)

    def import_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Import Config"), "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            config = load_config_from_file(file_path)
            if config:
                self.load_config_to_ui(config)
                QMessageBox.information(self, self.tr("Success"), self.tr("Config imported successfully"))
            else:
                QMessageBox.warning(self, self.tr("Error"), self.tr("Failed to import config"))

    def get_current_config(self):
        def get_widget_value(key):
            if key not in self.config_widgets:
                return None
            widget = self.config_widgets[key]
            widget_type = type(widget).__name__
            try:
                if widget_type == "CustomLineEdit":
                    return widget.text()
                elif widget_type in ["CustomSpinBox", "CustomDoubleSpinBox"]:
                    return widget.value()
                elif widget_type == "CustomComboBox":
                    return widget.currentText()
                elif widget_type == "CustomCheckBox":
                    return widget.isChecked()
                elif widget_type == "CustomSlider":
                    return widget.value()
            except Exception:
                return None
            return None

        config = {
            "basic": {
                "project": get_widget_value("project"),
                "name": get_widget_value("name"),
                "model": get_widget_value("model"),
                "data": get_widget_value("data"),
                "device": get_widget_value("device"),
                "dataset_ratio": get_widget_value("dataset_ratio") / 100.0 if get_widget_value("dataset_ratio") is not None else 0.8,
                "classes": get_widget_value("classes"),
                "pose_config": get_widget_value("pose_config"),
            },
            "train": {
                "epochs": get_widget_value("epochs"),
                "batch": get_widget_value("batch"),
                "imgsz": get_widget_value("imgsz"),
                "workers": get_widget_value("workers"),
                "single_cls": get_widget_value("single_cls"),
            },
            "strategy": {
                "time": get_widget_value("time"),
                "patience": get_widget_value("patience"),
                "close_mosaic": get_widget_value("close_mosaic"),
                "optimizer": get_widget_value("optimizer"),
                "cos_lr": get_widget_value("cos_lr"),
                "amp": get_widget_value("amp"),
                "multi_scale": get_widget_value("multi_scale"),
            },
            "learning_rate": {
                "lr0": get_widget_value("lr0"),
                "lrf": get_widget_value("lrf"),
                "momentum": get_widget_value("momentum"),
                "weight_decay": get_widget_value("weight_decay"),
            },
            "warmup": {
                "warmup_epochs": get_widget_value("warmup_epochs"),
                "warmup_momentum": get_widget_value("warmup_momentum"),
                "warmup_bias_lr": get_widget_value("warmup_bias_lr"),
            },
            "augment": {
                "hsv_h": get_widget_value("hsv_h"),
                "hsv_s": get_widget_value("hsv_s"),
                "hsv_v": get_widget_value("hsv_v"),
                "degrees": get_widget_value("degrees"),
                "translate": get_widget_value("translate"),
                "scale": get_widget_value("scale"),
                "shear": get_widget_value("shear"),
                "perspective": get_widget_value("perspective"),
            },
            "regularization": {
                "dropout": get_widget_value("dropout"),
                "fraction": get_widget_value("fraction"),
                "rect": get_widget_value("rect"),
            },
            "loss_weights": {
                "box": get_widget_value("box"),
                "cls": get_widget_value("cls"),
                "dfl": get_widget_value("dfl"),
                "pose": get_widget_value("pose"),
                "kobj": get_widget_value("kobj"),
            },
            "checkpoint": {
                "save_period": get_widget_value("save_period"),
                "val": get_widget_value("val"),
                "plots": get_widget_value("plots"),
                "save": get_widget_value("save"),
                "resume": get_widget_value("resume"),
                "cache": get_widget_value("cache"),
            }
        }

        return config

    def save_config(self):
        config = self.get_current_config()
        try:
            with open(SETTINGS_CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            template = self.tr("Configuration saved successfully to %s")
            msg_test = template % SETTINGS_CONFIG_PATH
            QMessageBox.information(self, self.tr("Success"), msg_test)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Error"), f"Failed to save config: {str(e)}")

    def start_training(self):
        config = self.get_current_config()
        is_valid, error_message = validate_basic_config(config)
        if not is_valid:
            QMessageBox.warning(self, self.tr("Validation Error"), error_message)
            return
        if not self.selected_task_type:
            QMessageBox.warning(self, self.tr("Error"), self.tr("Please select a task type first"))
            return

        if self.selected_task_type.lower() == "pose":
            pose_config = config["basic"].get("pose_config", "")
            if not pose_config or not os.path.exists(pose_config):
                QMessageBox.warning(self, self.tr("Error"), self.tr("Please select a valid pose configuration file for pose detection tasks"))
                return

        temp_dir = create_yolo_dataset(
            self.image_list,
            self.selected_task_type,
            config["basic"]["dataset_ratio"],
            config["basic"]["data"],
            self.output_dir,
            config["basic"].get("pose_config")
        )
        logger.info(f"Created YOLO dataset: {temp_dir}")

        train_args = {
            "data": os.path.join(temp_dir, "data.yaml"),
            "model": config["basic"]["model"],
            "project": config["basic"]["project"],
            "name": config["basic"]["name"],
            "device": config["basic"]["device"],
        }

        advanced_params = {}
        for section in ["train", "strategy", "learning_rate", "warmup", "augment", "regularization", "loss_weights", "checkpoint"]:
            advanced_params.update(config.get(section, {}))
        advanced_params = {
            key: value for key, value in advanced_params.items()
            if key in DEFAULT_TRAINING_CONFIG and value != DEFAULT_TRAINING_CONFIG[key]
        }
        train_args.update(advanced_params)

        cmd_parts = ["yolo", self.selected_task_type.lower(), "train"]
        for key, value in train_args.items():
            cmd_parts.append(f"{key}={value}")

        confirm_dialog = TrainingConfirmDialog(cmd_parts, self)
        if confirm_dialog.exec_() != QDialog.Accepted:
            return

        try:
            success, message = self.training_manager.start_training(train_args)

            if not success:
                QMessageBox.critical(self, self.tr("Training Error"), message)
                return

            save_config(config)
            self.tab_widget.setCurrentIndex(2)

            QMessageBox.information(
                self, 
                self.tr("Training Started"), 
                self.tr("Training process has been started successfully")
            )

        except Exception as e:
            QMessageBox.critical(
                self, 
                self.tr("Training Error"), 
                f"Failed to start training: {str(e)}"
            )

    def init_config_buttons(self, parent_layout):
        button_layout = QHBoxLayout()

        import_btn = SecondaryButton(self.tr("Import Config"))
        import_btn.clicked.connect(self.import_config)
        button_layout.addWidget(import_btn)

        save_btn = SecondaryButton(self.tr("Save Config"))
        save_btn.clicked.connect(self.save_config)
        button_layout.addWidget(save_btn)
        button_layout.addStretch()

        train_btn = PrimaryButton(self.tr("Next"))
        train_btn.clicked.connect(self.start_training)
        button_layout.addWidget(train_btn)

        parent_layout.addLayout(button_layout)

    def load_default_config(self):
        config = load_config()
        self.load_config_to_ui(config)

    def init_config_tab(self):
        layout = QVBoxLayout(self.config_tab)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.init_basic_settings(scroll_layout)
        self.init_train_settings(scroll_layout)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        self.init_config_buttons(layout)
        self.load_default_config()

    # Train config
    def init_train_tab(self):
        layout = QVBoxLayout(self.train_tab)


