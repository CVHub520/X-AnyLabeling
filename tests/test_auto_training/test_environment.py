import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from anylabeling.services.auto_training.ultralytics.environment import (
    EnvironmentManager,
    WORKER_EVENT_PREFIX,
    build_worker_command,
    build_worker_environment,
    create_worker_payload,
    expand_training_path,
    get_default_training_python,
    get_worker_creation_flags,
    get_worker_script_path,
    parse_worker_output,
    prepare_training_data_directory,
    resolve_training_data_directory,
    resolve_training_python,
)


def test_default_training_python_uses_current_interpreter(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert get_default_training_python() == sys.executable


def test_default_training_python_is_empty_when_frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert get_default_training_python() is None


def test_expand_training_path_supports_environment_and_relative_paths(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("TRAINING_ROOT", "models")
    assert expand_training_path("$TRAINING_ROOT/python", str(tmp_path)) == str(
        tmp_path / "models" / "python"
    )
    assert expand_training_path(
        "%TRAINING_ROOT%/python", str(tmp_path)
    ) == str(tmp_path / "models" / "python")


def test_expand_training_path_supports_tilde_and_unicode(
    monkeypatch, tmp_path
):
    home_path = tmp_path / "用户目录"
    monkeypatch.setenv("HOME", str(home_path))
    monkeypatch.setenv("TRAINING_HOME", "~/训练环境")

    assert expand_training_path("$TRAINING_HOME/bin/python") == str(
        home_path / "训练环境" / "bin" / "python"
    )


def test_empty_python_and_data_paths_preserve_defaults(monkeypatch):
    monkeypatch.setattr(
        "anylabeling.services.auto_training.ultralytics.environment.get_work_directory",
        lambda: "/work",
    )

    assert resolve_training_python(None) == sys.executable
    assert resolve_training_data_directory(None) == os.path.join(
        "/work", "xanylabeling_data", "trainer", "ultralytics"
    )


def test_prepare_training_data_directory_creates_expected_unicode_paths(
    tmp_path,
):
    root = tmp_path / "训练数据"

    prepare_training_data_directory(str(root))

    assert {path.name for path in root.iterdir()} == {
        "datasets",
        "weights",
        "runs",
    }


def test_default_data_directory_preserves_existing_location(monkeypatch):
    monkeypatch.setattr(
        "anylabeling.services.auto_training.ultralytics.environment.get_work_directory",
        lambda: "/work",
    )
    assert resolve_training_data_directory(None) == os.path.join(
        "/work", "xanylabeling_data", "trainer", "ultralytics"
    )


def test_payload_is_versioned_and_worker_command_is_list_form():
    payload_path = create_worker_payload(
        "train",
        args={"model": "yolo.pt"},
        paths={"weights_directory": "/weights"},
        options={"auto_install_packages": False},
    )
    try:
        payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
        assert payload["version"] == 1
        assert payload["action"] == "train"
        command = build_worker_command(sys.executable, payload_path)
        assert command[0] == sys.executable
        assert command[-1] == payload_path
        assert isinstance(command, list)
    finally:
        os.remove(payload_path)


def test_worker_script_resolves_from_pyinstaller_bundle(monkeypatch, tmp_path):
    worker_path = (
        tmp_path
        / "anylabeling"
        / "services"
        / "auto_training"
        / "ultralytics"
        / "worker.py"
    )
    worker_path.parent.mkdir(parents=True)
    worker_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert get_worker_script_path() == str(worker_path)


def test_worker_environment_maps_auto_install_without_changing_cuda(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    enabled = build_worker_environment(True)
    disabled = build_worker_environment(False)
    assert enabled["YOLO_AUTOINSTALL"] == "true"
    assert disabled["YOLO_AUTOINSTALL"] == "false"
    assert enabled["CUDA_VISIBLE_DEVICES"] == "2"
    assert disabled["CUDA_VISIBLE_DEVICES"] == "2"


def test_windows_workers_hide_console_without_losing_process_group(
    monkeypatch,
):
    monkeypatch.setattr(
        "anylabeling.services.auto_training.ultralytics.environment.os",
        SimpleNamespace(name="nt"),
    )
    monkeypatch.setattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200, raising=False
    )
    monkeypatch.setattr(
        subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False
    )

    assert get_worker_creation_flags() == 0x08000200


def test_parse_worker_output_handles_events_and_plain_logs():
    events = []
    detected = WORKER_EVENT_PREFIX + json.dumps(
        {
            "version": 1,
            "event": "environment_detected",
            "cpu_available": True,
        }
    )
    assert parse_worker_output(
        detected, lambda event, data: events.append((event, data))
    )
    assert events == [("environment_detected", {"cpu_available": True})]

    assert not parse_worker_output(
        "plain output", lambda event, data: events.append((event, data))
    )
    assert events[-1] == ("training_log", {"message": "plain output"})


def test_parse_worker_output_routes_export_logs_to_export_event():
    events = []

    assert not parse_worker_output(
        "Ultralytics export output",
        lambda event, data: events.append((event, data)),
        plain_log_event="export_log",
    )
    assert events == [("export_log", {"message": "Ultralytics export output"})]


def test_parse_worker_output_classifies_unsupported_protocol():
    events = []
    output = WORKER_EVENT_PREFIX + json.dumps(
        {"version": 999, "event": "environment_detected"}
    )

    assert parse_worker_output(
        output, lambda event, data: events.append((event, data))
    )
    assert events == [
        (
            "environment_error",
            {
                "error": "Unsupported worker protocol",
                "error_type": "protocol_error",
            },
        )
    ]


def test_parse_worker_output_treats_invalid_json_as_log():
    events = []

    assert not parse_worker_output(
        f"{WORKER_EVENT_PREFIX}not-json",
        lambda event, data: events.append((event, data)),
    )
    assert events == [
        ("training_log", {"message": f"{WORKER_EVENT_PREFIX}not-json"})
    ]


def test_environment_probe_timeout_terminates_process_and_cleans_payload(
    monkeypatch, tmp_path
):
    payload_path = tmp_path / "probe.json"
    payload_path.write_text("{}", encoding="utf-8")
    terminated = threading.Event()

    class TimedOutProcess:
        def communicate(self, timeout):
            raise subprocess.TimeoutExpired([sys.executable], timeout)

    process = TimedOutProcess()
    monkeypatch.setattr(
        "anylabeling.services.auto_training.ultralytics.environment.create_worker_payload",
        lambda _action: str(payload_path),
    )
    monkeypatch.setattr(
        "anylabeling.services.auto_training.ultralytics.environment.build_probe_command",
        lambda *_args: [sys.executable],
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        "anylabeling.services.auto_training.ultralytics.environment.terminate_process_tree",
        lambda target: terminated.set() if target is process else None,
    )
    completed = threading.Event()
    events = []
    manager = EnvironmentManager()
    manager.callbacks = [
        lambda event, data: (events.append((event, data)), completed.set())
    ]

    manager.probe(sys.executable, timeout=0.01)

    assert completed.wait(timeout=1)
    assert terminated.is_set()
    assert events[0][0] == "environment_error"
    assert events[0][1]["error_type"] == "probe_timeout"
    deadline = time.monotonic() + 1
    while payload_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not payload_path.exists()


def test_environment_process_exit_preserves_diagnostic_output(
    monkeypatch, tmp_path
):
    payload_path = tmp_path / "probe.json"
    payload_path.write_text("{}", encoding="utf-8")

    class FailedProcess:
        returncode = 7

        def communicate(self, timeout):
            return "环境启动失败\n".encode(), None

    monkeypatch.setattr(
        "anylabeling.services.auto_training.ultralytics.environment.create_worker_payload",
        lambda _action: str(payload_path),
    )
    monkeypatch.setattr(
        "anylabeling.services.auto_training.ultralytics.environment.build_probe_command",
        lambda *_args: [sys.executable],
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: FailedProcess(),
    )
    completed = threading.Event()
    events = []
    manager = EnvironmentManager()
    manager.callbacks = [
        lambda event, data: (
            (events.append((event, data)), completed.set())
            if event == "environment_error"
            else None
        )
    ]

    manager.probe(sys.executable)

    assert completed.wait(timeout=1)
    assert events == [
        (
            "environment_error",
            {
                "error": "Environment process exited with code 7",
                "error_type": "process_exit",
                "details": "环境启动失败",
            },
        )
    ]


def test_environment_cancel_does_not_wait_for_process_cleanup(monkeypatch):
    cleanup_started = threading.Event()
    allow_cleanup = threading.Event()
    process = object()

    def delayed_cleanup(target):
        assert target is process
        cleanup_started.set()
        allow_cleanup.wait(timeout=1)

    monkeypatch.setattr(
        "anylabeling.services.auto_training.ultralytics.environment.terminate_process_tree",
        delayed_cleanup,
    )
    manager = EnvironmentManager()
    manager._process = process

    started_at = time.monotonic()
    manager.cancel()
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.2
    assert cleanup_started.wait(timeout=1)
    allow_cleanup.set()
