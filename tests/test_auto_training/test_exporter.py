import threading
import json
import yaml

from anylabeling.services.auto_training.ultralytics import exporter
from anylabeling.services.auto_training.ultralytics.exporter import (
    ExportManager,
    create_auto_labeling_config,
)


def test_create_auto_labeling_config_next_to_experiment(tmp_path, monkeypatch):
    project_path = tmp_path / "expCustom"
    weights_path = project_path / "weights"
    weights_path.mkdir(parents=True)
    exported_path = weights_path / "best.onnx"
    exported_path.write_bytes(b"onnx")
    (project_path / "args.yaml").write_text(
        "task: detect\nmodel: /models/yolo26n.pt\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        exporter,
        "_load_onnx_metadata",
        lambda _path: {"names": "{0: 'cat', 1: 'dog'}"},
    )

    config_path = create_auto_labeling_config(
        str(project_path), str(exported_path)
    )

    assert config_path == str(project_path / "expCustom.yaml")
    config = yaml.safe_load((project_path / "expCustom.yaml").read_text())
    assert config == {
        "type": "yolo26",
        "name": "expCustom",
        "provider": "Ultralytics",
        "display_name": "expCustom",
        "model_path": "weights/best.onnx",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_det": 300,
        "classes": ["cat", "dog"],
    }


def test_create_auto_labeling_pose_config_uses_pose_classes(
    tmp_path, monkeypatch
):
    project_path = tmp_path / "pose-exp"
    weights_path = project_path / "weights"
    weights_path.mkdir(parents=True)
    exported_path = weights_path / "best.onnx"
    exported_path.write_bytes(b"onnx")
    (project_path / "args.yaml").write_text(
        "task: pose\nmodel: yolo11n-pose.pt\n",
        encoding="utf-8",
    )
    pose_config_path = tmp_path / "pose.yaml"
    pose_config_path.write_text(
        "has_visible: false\nclasses:\n  person:\n    - nose\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(exporter, "_load_onnx_metadata", lambda _path: {})

    config_path = create_auto_labeling_config(
        str(project_path), str(exported_path), str(pose_config_path)
    )

    assert config_path == str(project_path / "pose-exp.yaml")
    config = yaml.safe_load(
        (project_path / "pose-exp.yaml").read_text(encoding="utf-8")
    )
    assert config["type"] == "yolo11_pose"
    assert config["has_visible"] is False
    assert config["classes"] == {"person": ["nose"]}


def test_stop_export_notifies_immediately_and_terminates_process(monkeypatch):
    terminated = threading.Event()
    process = object()
    monkeypatch.setattr(
        exporter,
        "terminate_process_tree",
        lambda target: terminated.set() if target is process else None,
    )
    manager = ExportManager()
    manager.is_exporting = True
    manager.export_process = process
    events = []
    manager.callbacks = [lambda event, data: events.append((event, data))]

    assert manager.stop_export()

    assert events == [("export_stopped", {})]
    assert terminated.wait(timeout=1)
    assert manager.stop_event.is_set()


def test_stop_export_can_terminate_process_synchronously(monkeypatch):
    process = object()
    terminated = []
    monkeypatch.setattr(
        exporter,
        "terminate_process_tree",
        lambda target: terminated.append(target),
    )
    manager = ExportManager()
    manager.is_exporting = True
    manager.export_process = process

    assert manager.stop_export(wait=True)
    assert terminated == [process]
    assert manager.stop_event.is_set()


def test_stopped_export_ignores_late_worker_events():
    manager = ExportManager()
    events = []
    manager.callbacks = [lambda event, data: events.append((event, data))]
    manager.stop_event.set()

    manager._notify_external_export_event(
        "export_completed", {"exported_path": "model.onnx"}
    )

    assert events == []


def test_external_export_routes_plain_worker_output_to_export_log(
    monkeypatch, tmp_path
):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("{}", encoding="utf-8")

    class ExportProcess:
        stdout = iter(
            [
                "Ultralytics export output\n",
                "__XANYLABELING_WORKER_EVENT__="
                + json.dumps(
                    {
                        "version": 1,
                        "event": "export_completed",
                        "exported_path": "/model.onnx",
                    }
                )
                + "\n",
            ]
        )

        def wait(self):
            return 0

    monkeypatch.setattr(
        exporter,
        "create_worker_payload",
        lambda *_args, **_kwargs: str(payload_path),
    )
    monkeypatch.setattr(
        exporter.subprocess,
        "Popen",
        lambda *_args, **_kwargs: ExportProcess(),
    )
    manager = ExportManager()
    manager.configure_environment("/training/python")
    events = []
    manager.callbacks = [lambda event, data: events.append((event, data))]

    manager._external_export_worker("/model.pt", "onnx")

    assert events == [
        ("export_log", {"message": "Ultralytics export output"}),
        ("export_completed", {"exported_path": "/model.onnx"}),
    ]
    assert not payload_path.exists()


def test_external_export_reports_process_without_terminal_event(
    monkeypatch, tmp_path
):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("{}", encoding="utf-8")

    class ExportProcess:
        stdout = iter(())

        def wait(self):
            return 0

    monkeypatch.setattr(
        exporter,
        "create_worker_payload",
        lambda *_args, **_kwargs: str(payload_path),
    )
    monkeypatch.setattr(
        exporter.subprocess,
        "Popen",
        lambda *_args, **_kwargs: ExportProcess(),
    )
    manager = ExportManager()
    manager.configure_environment("/training/python")
    events = []
    manager.callbacks = [lambda event, data: events.append((event, data))]

    manager._external_export_worker("/model.pt", "onnx")

    assert events == [
        ("export_error", {"error": "Export process returned no result"})
    ]
