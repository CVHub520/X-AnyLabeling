import json
import os
import sys
import threading
import time

from anylabeling.services.auto_training.ultralytics import trainer
from anylabeling.services.auto_training.ultralytics.trainer import (
    TrainingManager,
    build_training_worker_command,
)
from anylabeling.services.auto_training.ultralytics.validators import (
    validate_basic_config,
)


def test_external_training_payload_preserves_bare_model_name():
    manager = TrainingManager()
    manager.configure_environment(sys.executable, auto_install_packages=False)
    payload_path = manager._create_payload(
        {
            "model": "yolo11n.pt",
            "data": "/data/data.yaml",
            "project": "/runs",
            "name": "exp",
        }
    )
    try:
        with open(payload_path, encoding="utf-8") as stream:
            payload = json.load(stream)
        assert payload["action"] == "train"
        assert payload["args"]["model"] == "yolo11n.pt"
        assert payload["options"]["auto_install_packages"] is False
    finally:
        os.remove(payload_path)


def test_external_training_command_uses_selected_python():
    command = build_training_worker_command(
        "/tmp/payload.json",
        python_executable=sys.executable,
        external=True,
    )
    assert command[0] == sys.executable
    assert command[1] == "-u"
    assert command[-1] == "/tmp/payload.json"


def test_default_training_command_preserves_application_worker_entrypoint():
    command = build_training_worker_command(
        "/tmp/payload.json",
        external=False,
    )

    assert command[:3] == [sys.executable, "-m", "anylabeling.app"]
    assert command[-2:] == ["--payload", "/tmp/payload.json"]


def test_external_validation_allows_only_bare_pt_model_name(tmp_path):
    data_path = tmp_path / "data.yaml"
    data_path.write_text("names: {}", encoding="utf-8")
    base = {
        "basic": {
            "project": str(tmp_path / "runs"),
            "name": "exp",
            "data": str(data_path),
        }
    }
    base["basic"]["model"] = "yolo11n.pt"
    assert validate_basic_config(base, allow_model_name=True) == (True, "")
    base["basic"]["model"] = "models/yolo11n.pt"
    assert validate_basic_config(base, allow_model_name=True)[0] is False


def test_stop_training_terminates_silent_process_immediately(
    monkeypatch, tmp_path
):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        trainer,
        "build_training_worker_command",
        lambda *_args, **_kwargs: [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
    )
    manager = TrainingManager()
    manager.is_training = True
    events = []
    manager.callbacks = [lambda event, data: events.append((event, data))]
    training_thread = threading.Thread(
        target=manager._run_training,
        args=(str(payload_path),),
    )
    training_thread.start()

    deadline = time.monotonic() + 3
    while manager.training_process is None and time.monotonic() < deadline:
        time.sleep(0.01)

    process = manager.training_process
    assert process is not None
    assert manager.stop_training()
    training_thread.join(timeout=3)

    assert not training_thread.is_alive()
    assert process.poll() is not None
    event_types = [event for event, _data in events]
    assert event_types.count("training_stopped") == 1
    assert "training_completed" not in event_types
    assert "training_error" not in event_types
    assert not payload_path.exists()


def test_stop_training_can_terminate_process_synchronously(monkeypatch):
    process = object()
    terminated = []
    monkeypatch.setattr(
        trainer,
        "terminate_process_tree",
        lambda target: terminated.append(target),
    )
    manager = TrainingManager()
    manager.is_training = True
    manager.training_process = process

    assert manager.stop_training(wait=True)
    assert terminated == [process]
    assert manager.stop_event.is_set()
