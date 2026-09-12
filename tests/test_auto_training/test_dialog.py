import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt, QTranslator
from PyQt6.QtGui import QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QGroupBox,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import anylabeling.resources.resources  # noqa: F401
from anylabeling.views.labeling.utils.qt import new_icon
from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.training.ultralytics_dialog import UltralyticsDialog
from anylabeling.views.training.widgets.ultralytics_widgets import (
    ExportFormatDialog,
)


def test_compiled_chinese_training_translations_are_available():
    translator = QTranslator()
    assert translator.load("anylabeling/resources/translations/zh_CN.qm")

    assert (
        translator.translate("UltralyticsDialog", "Stop Export") == "停止导出"
    )


def test_redetect_resets_state_and_starts_new_probe():
    probe_calls = []
    dialog = SimpleNamespace(
        environment_result={"cpu_available": True},
        environment_error="old error",
        environment_error_type="process_exit",
        environment_ready=True,
        training_python="/training/python",
        auto_install_packages=False,
        external_environment=True,
        environment_manager=SimpleNamespace(
            probe=lambda *args, **kwargs: probe_calls.append((args, kwargs))
        ),
        apply_environment_result=lambda: None,
    )

    UltralyticsDialog.detect_environment(dialog)

    assert dialog.environment_result is None
    assert dialog.environment_error is None
    assert dialog.environment_error_type is None
    assert dialog.environment_ready is False
    assert probe_calls == [
        (
            ("/training/python", False),
            {"external_environment": True},
        )
    ]


def test_close_hides_running_training_and_preserves_task():
    actions = []
    event = SimpleNamespace(ignore=lambda: actions.append("ignored"))
    environment_manager = SimpleNamespace(
        cancel=lambda: actions.append("cancelled")
    )
    dialog = SimpleNamespace(
        training_status="training",
        training_manager=SimpleNamespace(is_training=True),
        export_manager=SimpleNamespace(is_exporting=False),
        _application_closing=False,
        tr=lambda text: text,
        hide=lambda: actions.append("hidden"),
        clear_cache=lambda: actions.append("cleared"),
        environment_manager=environment_manager,
        save_training_logs_to_file=lambda: actions.append("saved"),
    )

    assert not UltralyticsDialog._prepare_to_close(dialog, event)
    assert actions == ["hidden", "ignored"]

    dialog.training_manager.is_training = False
    dialog.training_status = "completed"
    assert UltralyticsDialog._prepare_to_close(dialog, event)
    assert actions == [
        "hidden",
        "ignored",
        "saved",
        "cleared",
        "cancelled",
    ]


def test_close_does_not_resave_loaded_existing_training_log():
    actions = []
    dialog = SimpleNamespace(
        training_status="completed",
        training_manager=SimpleNamespace(is_training=False),
        export_manager=SimpleNamespace(is_exporting=False),
        _application_closing=False,
        _loaded_existing_model=True,
        clear_cache=lambda: actions.append("cleared"),
        environment_manager=SimpleNamespace(
            cancel=lambda: actions.append("cancelled")
        ),
        save_training_logs_to_file=lambda: actions.append("saved"),
    )

    assert UltralyticsDialog._prepare_to_close(dialog, SimpleNamespace())
    assert actions == ["cleared", "cancelled"]


def test_application_close_stops_background_training(monkeypatch):
    actions = []
    dialog = SimpleNamespace(
        training_status="training",
        training_manager=SimpleNamespace(
            is_training=True,
            stop_training=lambda wait=False: actions.append(
                ("training_stopped", wait)
            ),
        ),
        export_manager=SimpleNamespace(is_exporting=False),
        _application_closing=False,
        parent=lambda: None,
        tr=lambda text: text,
    )
    monkeypatch.setattr(
        "anylabeling.views.training.ultralytics_dialog.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    assert UltralyticsDialog.prepare_for_application_close(dialog)
    assert actions == [("training_stopped", True)]
    assert dialog.training_status == "stop"
    assert dialog._application_closing is True


def test_training_window_is_modeless_and_reused(monkeypatch):
    actions = []

    class DestroyedSignal:
        def connect(self, callback):
            actions.append(("connected", callback))

    class TrainingDialog:
        def __init__(self, _parent):
            self.destroyed = DestroyedSignal()
            actions.append("created")

        def showNormal(self):
            actions.append("shown")

        def raise_(self):
            actions.append("raised")

        def activateWindow(self):
            actions.append("activated")

    monkeypatch.setattr(
        "anylabeling.views.labeling.label_widget.UltralyticsDialog",
        TrainingDialog,
    )
    widget = SimpleNamespace(
        training_dialog=None,
        on_training_dialog_destroyed=lambda: None,
        error_message=lambda *_args: None,
    )

    LabelingWidget.start_training(widget, "ultralytics")
    first_dialog = widget.training_dialog
    LabelingWidget.start_training(widget, "ultralytics")

    assert widget.training_dialog is first_dialog
    assert actions.count("created") == 1
    assert actions.count("shown") == 2
    assert "exec" not in actions


def test_load_latest_training_log_uses_newest_saved_file(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    old_log = log_dir / "training_log_completed_20260901_120000.txt"
    new_log = log_dir / "training_log_completed_20260902_120000.txt"
    old_log.write_text("original training log", encoding="utf-8")
    new_log.write_text(
        "Loaded existing model from: /runs/exp/weights/best.pt",
        encoding="utf-8",
    )
    os.utime(old_log, (1, 1))
    os.utime(new_log, (2, 2))
    displayed_logs = []
    dialog = SimpleNamespace(
        current_project_path=str(tmp_path),
        log_display=SimpleNamespace(setPlainText=displayed_logs.append),
    )

    assert UltralyticsDialog.load_latest_training_log(dialog)
    assert displayed_logs == [
        "Loaded existing model from: /runs/exp/weights/best.pt"
    ]


def test_new_training_reenables_log_saving_after_loading_existing_model():
    dialog = SimpleNamespace(
        _loaded_existing_model=True,
        training_status="completed",
        total_epochs=0,
        current_epochs=3,
        progress_bar=SimpleNamespace(
            setValue=lambda _value: None,
            setFormat=lambda _value: None,
        ),
        update_training_status_display=lambda: None,
        start_training_button=SimpleNamespace(setVisible=lambda _value: None),
        stop_training_button=SimpleNamespace(setVisible=lambda _value: None),
        export_button=SimpleNamespace(setVisible=lambda _value: None),
        previous_button=SimpleNamespace(setVisible=lambda _value: None),
        progress_timer=SimpleNamespace(start=lambda _value: None),
        image_timer=SimpleNamespace(start=lambda _value: None),
        append_training_log=lambda _value: None,
        tr=lambda text: text,
    )

    UltralyticsDialog.on_training_event(
        dialog, "training_started", {"total_epochs": 10}
    )

    assert dialog._loaded_existing_model is False


def test_redetect_falls_back_to_cpu_when_previous_gpu_is_unavailable():
    app = QApplication.instance() or QApplication([])
    device = QComboBox()
    configured_gpu_selections = []
    dialog = SimpleNamespace(
        config_widgets={"device": device},
        environment_ready=True,
        environment_result={
            "cuda_available": True,
            "mps_available": False,
            "cpu_available": True,
            "gpus": [
                {
                    "index": 0,
                    "name": "GPU 0",
                    "available": True,
                    "error": None,
                }
            ],
        },
        environment_error=None,
        _pending_device_selection=("cuda", (1,)),
        setup_cuda_checkboxes=lambda _gpus, selected: (
            configured_gpu_selections.append(selected)
        ),
        on_device_changed=lambda _text: None,
    )

    UltralyticsDialog.apply_environment_result(dialog)
    app.processEvents()

    assert device.currentText() == "cpu"
    assert configured_gpu_selections == [None]
    assert dialog._pending_device_selection is None


def test_external_export_reuses_export_button_for_stop_action():
    app = QApplication.instance() or QApplication([])
    export_button = QPushButton("Export")
    logs = []
    dialog = SimpleNamespace(
        external_environment=True,
        export_button=export_button,
        append_training_log=logs.append,
        tr=lambda text: text,
    )
    dialog.reset_export_button = lambda: (
        UltralyticsDialog.reset_export_button(dialog)
    )

    UltralyticsDialog.on_export_event(dialog, "export_started", {})
    app.processEvents()

    assert export_button.text() == "Stop Export"
    assert export_button.isEnabled()

    UltralyticsDialog.on_export_event(dialog, "export_stopped", {})
    app.processEvents()

    assert export_button.text() == "Export"
    assert export_button.isEnabled()
    assert logs == ["Export started...", "Export stopped by user"]


def test_export_dialog_reuses_primary_button_for_export_stop_and_apply():
    app = QApplication.instance() or QApplication([])
    dialog = ExportFormatDialog()
    actions = []
    dialog.export_requested.connect(
        lambda export_format: actions.append(("export", export_format))
    )
    dialog.stop_requested.connect(lambda: actions.append(("stop", None)))
    dialog.apply_requested.connect(lambda: actions.append(("apply", None)))

    dialog.ok_btn.click()
    dialog.set_exporting()
    dialog.ok_btn.click()
    dialog.set_apply_ready()
    dialog.ok_btn.click()
    app.processEvents()

    assert actions == [
        ("export", "onnx"),
        ("stop", None),
        ("apply", None),
    ]
    assert dialog.ok_btn.text() == "Apply"
    assert not dialog.format_combo.isEnabled()
    dialog.close()


def test_onnx_export_completion_keeps_dialog_open_for_apply():
    app = QApplication.instance() or QApplication([])
    export_button = QPushButton("Stop Export")
    export_dialog = ExportFormatDialog()
    export_dialog.set_exporting()
    export_dialog.show()
    logs = []
    dialog = SimpleNamespace(
        export_button=export_button,
        export_dialog=export_dialog,
        exported_onnx_path=None,
        append_training_log=logs.append,
        tr=lambda text: text,
    )
    dialog.reset_export_button = lambda: (
        UltralyticsDialog.reset_export_button(dialog)
    )

    UltralyticsDialog.on_export_event(
        dialog,
        "export_completed",
        {"exported_path": "/runs/exp/weights/best.onnx", "format": "onnx"},
    )
    app.processEvents()

    assert export_dialog.action_state == "apply"
    assert export_dialog.isVisible()
    assert dialog.exported_onnx_path == "/runs/exp/weights/best.onnx"
    assert export_button.text() == "Export"
    export_dialog.close()


def test_apply_exported_model_loads_config_and_returns_to_main_panel(
    monkeypatch,
):
    actions = []
    export_dialog = SimpleNamespace(
        accept=lambda: actions.append("export_dialog_closed")
    )
    auto_labeling_widget = SimpleNamespace(
        load_custom_model_config=lambda path: actions.append(("load", path))
        or True,
        isVisible=lambda: False,
    )
    parent = SimpleNamespace(
        auto_labeling_widget=auto_labeling_widget,
        toggle_auto_labeling_widget=lambda: actions.append("panel_shown"),
    )
    dialog = SimpleNamespace(
        exported_onnx_path="/runs/exp/weights/best.onnx",
        current_project_path="/runs/exp",
        export_dialog=export_dialog,
        get_current_config=lambda: {"basic": {"pose_config": ""}},
        parent=lambda: parent,
        append_training_log=lambda message: actions.append(("log", message)),
        close=lambda: actions.append("training_dialog_closed"),
        tr=lambda text: text,
    )
    monkeypatch.setattr(
        "anylabeling.views.training.ultralytics_dialog.create_auto_labeling_config",
        lambda *_args: "/runs/exp/exp.yaml",
    )

    UltralyticsDialog.apply_exported_model(dialog)

    assert actions == [
        ("load", "/runs/exp/exp.yaml"),
        ("log", "Applied model configuration: /runs/exp/exp.yaml"),
        "export_dialog_closed",
        "training_dialog_closed",
        "panel_shown",
    ]


def test_training_status_keeps_original_compact_layout():
    app = QApplication.instance() or QApplication([])
    host = QWidget()
    layout = QVBoxLayout(host)
    dialog = SimpleNamespace(tr=lambda text: text)

    UltralyticsDialog.init_training_status(dialog, layout)
    app.processEvents()

    status_group = host.findChild(QGroupBox)
    assert status_group is not None
    assert status_group.layout().count() == 2
    assert status_group.findChildren(QPushButton) == []
    host.close()


def test_train_tab_is_initialized_before_first_navigation():
    app = QApplication.instance() or QApplication([])
    dialog = QDialog()
    calls = []
    dialog.init_data_tab = lambda: calls.append("data")
    dialog.ensure_train_tab_initialized = lambda: calls.append("train")

    UltralyticsDialog.init_ui(dialog)
    app.processEvents()

    assert calls == ["data", "train"]
    dialog.close()


def test_training_logs_use_hover_icon_actions():
    app = QApplication.instance() or QApplication([])
    host = QWidget()
    layout = QVBoxLayout(host)
    dialog = SimpleNamespace(
        tr=lambda text: text,
        clear_training_logs=lambda: None,
        copy_training_logs=lambda: None,
    )

    UltralyticsDialog.init_training_logs(dialog, layout)
    app.processEvents()

    assert dialog.copy_logs_button.text() == ""
    assert dialog.copy_logs_button.toolTip() == "Copy"
    assert not dialog.copy_logs_button.icon().isNull()
    assert dialog.clear_logs_button.text() == ""
    assert dialog.clear_logs_button.toolTip() == "Clear"
    assert not dialog.clear_logs_button.icon().isNull()
    assert dialog.training_log_actions.isHidden()

    QApplication.sendEvent(
        dialog.training_log_container, QEvent(QEvent.Type.Enter)
    )
    assert not dialog.training_log_actions.isHidden()

    QApplication.sendEvent(
        dialog.training_log_container, QEvent(QEvent.Type.Leave)
    )
    assert dialog.training_log_actions.isHidden()
    host.close()


def test_copy_training_logs_shows_temporary_checkmark(monkeypatch):
    app = QApplication.instance() or QApplication([])
    callbacks = []
    log_display = QTextEdit()
    log_display.setPlainText("training output")
    copy_button = QPushButton()
    copy_button.setIcon(new_icon("copy", "svg"))
    copy_image = copy_button.icon().pixmap(16, 16).toImage()
    dialog = SimpleNamespace(
        log_display=log_display,
        copy_logs_button=copy_button,
    )
    dialog.reset_copy_logs_button = lambda: (
        UltralyticsDialog.reset_copy_logs_button(dialog)
    )
    monkeypatch.setattr(
        "anylabeling.views.training.ultralytics_dialog.QTimer.singleShot",
        lambda _delay, callback: callbacks.append(callback),
    )

    UltralyticsDialog.copy_training_logs(dialog)

    assert QApplication.clipboard().text() == "training output"
    assert copy_button.icon().pixmap(16, 16).toImage() != copy_image
    assert len(callbacks) == 1

    callbacks[0]()
    assert copy_button.icon().pixmap(16, 16).toImage() == copy_image


def test_training_images_keep_first_six_and_scroll_for_remaining(tmp_path):
    app = QApplication.instance() or QApplication([])
    filenames = [
        "train_batch0.jpg",
        "train_batch1.jpg",
        "train_batch2.jpg",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "results.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "confusion_matrix.png",
        "labels.jpg",
        "val_batch0_labels.jpg",
    ]
    for filename in filenames:
        pixmap = QPixmap(10, 10)
        pixmap.fill(Qt.GlobalColor.white)
        assert pixmap.save(str(tmp_path / filename))

    host = QWidget()
    host.resize(1100, 300)
    layout = QVBoxLayout(host)
    dialog = SimpleNamespace(
        tr=lambda text: text,
        selected_task_type="Detect",
        current_project_path=str(tmp_path),
        on_image_clicked=lambda _index: None,
    )
    dialog.create_training_image_slot = lambda: (
        UltralyticsDialog.create_training_image_slot(dialog)
    )
    dialog.set_training_image_slot_count = lambda count: (
        UltralyticsDialog.set_training_image_slot_count(dialog, count)
    )
    dialog.update_training_image_layout = lambda: (
        UltralyticsDialog.update_training_image_layout(dialog)
    )

    UltralyticsDialog.init_training_images(dialog, layout)
    host.show()
    app.processEvents()
    UltralyticsDialog.update_training_images(dialog)
    app.processEvents()

    assert [label.toolTip() for label in dialog.image_labels[:6]] == filenames[
        :6
    ]
    assert len(dialog.image_labels) == len(filenames)
    assert (
        dialog.training_images_scroll_area.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    )
    assert (
        dialog.training_images_scroll_area.horizontalScrollBar().maximum() > 0
    )
    assert dialog.training_images_scroll_area.height() == 162
    assert all(label.height() == 150 for label in dialog.image_labels)
    assert (
        dialog.training_images_scroll_area.frameShape() == QFrame.Shape.NoFrame
    )

    scroll_bar = dialog.training_images_scroll_area.horizontalScrollBar()
    scroll_bar.setValue(0)
    wheel_event = QWheelEvent(
        QPointF(20, 20),
        QPointF(20, 20),
        QPoint(),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    dialog.training_images_scroll_area.wheelEvent(wheel_event)
    assert scroll_bar.value() == 120
    host.close()


def test_training_actions_use_single_export_button():
    app = QApplication.instance() or QApplication([])
    host = QWidget()
    layout = QVBoxLayout(host)
    dialog = SimpleNamespace(
        tr=lambda text: text,
        environment_ready=True,
        open_training_directory=lambda: None,
        stop_training=lambda: None,
        go_to_specific_tab=lambda _index: None,
        start_training_from_train_tab=lambda: None,
        on_export_button_clicked=lambda: None,
    )

    UltralyticsDialog.init_training_actions(dialog, layout)
    app.processEvents()

    button_texts = [button.text() for button in host.findChildren(QPushButton)]
    assert button_texts.count("Export") == 1
    assert "Stop Export" not in button_texts
    host.close()


def test_basic_settings_adds_editable_project_and_environment():
    app = QApplication.instance() or QApplication([])
    host = QWidget()
    layout = QVBoxLayout(host)
    dialog = SimpleNamespace(
        tr=lambda text: text,
        selected_task_type="Detect",
        training_python_setting="/training/python",
        config_widgets={},
        browse_training_environment=lambda: None,
        configure_training_environment=lambda _value: None,
        browse_model_file=lambda: None,
        browse_data_file=lambda: None,
        browse_pose_config_file=lambda: None,
        on_device_changed=lambda _value: None,
        apply_environment_result=lambda: None,
    )

    UltralyticsDialog.init_basic_settings(dialog, layout)
    app.processEvents()

    assert not dialog.config_widgets["project"].isReadOnly()
    assert dialog.config_widgets["env"].text() == "/training/python"
    host.close()


def test_configuring_environment_updates_training_and_export_managers():
    calls = []
    dialog = SimpleNamespace(
        training_python_setting="/old/python",
        auto_install_packages=True,
        training_manager=SimpleNamespace(
            configure_environment=lambda *args: calls.append(
                ("training", args)
            )
        ),
        export_manager=SimpleNamespace(
            configure_environment=lambda *args: calls.append(("export", args))
        ),
        detect_environment=lambda: calls.append(("detect", ())),
    )

    assert UltralyticsDialog.configure_training_environment(
        dialog, " /new/python "
    )
    assert dialog.training_python_setting == "/new/python"
    assert dialog.external_environment is True
    assert calls == [
        ("training", ("/new/python", True)),
        ("export", ("/new/python", True)),
        ("detect", ()),
    ]
