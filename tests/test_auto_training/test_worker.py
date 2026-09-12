import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from anylabeling.services.auto_training.ultralytics import worker
from anylabeling.services.auto_training.ultralytics.environment import (
    WORKER_EVENT_PREFIX,
    create_worker_payload,
)


def test_export_dependency_packages_are_internal_whitelist():
    assert "onnx" in worker.EXPORT_DEPENDENCIES
    assert "torchscript" in worker.EXPORT_DEPENDENCIES
    assert all(
        isinstance(module, str) and isinstance(requirement, str)
        for dependencies in worker.EXPORT_DEPENDENCIES.values()
        for module, requirement in dependencies
    )


def test_load_payload_rejects_unknown_action(tmp_path):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "version": 1,
                "action": "shell",
                "args": {},
                "paths": {},
                "options": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported worker action"):
        worker.load_payload(str(payload_path))


def test_training_rejects_unknown_arguments(monkeypatch):
    payload = {
        "args": {"model": "model.pt", "arbitrary_command": "value"},
        "paths": {"weights_directory": "/tmp"},
        "options": {},
    }
    with pytest.raises(ValueError, match="Unsupported training arguments"):
        worker.run_train(payload)


def test_standalone_probe_reports_structured_result_without_site_packages():
    payload_path = create_worker_payload("probe")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(worker.__file__).resolve()),
                payload_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        event_lines = [
            line
            for line in result.stdout.splitlines()
            if line.startswith(WORKER_EVENT_PREFIX)
        ]
        assert len(event_lines) == 1
        event = json.loads(event_lines[0][len(WORKER_EVENT_PREFIX) :])
        assert event["version"] == 1
        assert event["event"] == "environment_error"
        assert event["error_type"] == "torch_missing"
    finally:
        os.remove(payload_path)


class _Tensor:
    def __add__(self, _value):
        return self

    def item(self):
        return 2.0


def test_probe_requires_real_cpu_operation_and_reports_cuda_runtime(
    monkeypatch,
):
    fake_torch = SimpleNamespace(
        __version__="2.0.0",
        version=SimpleNamespace(cuda="12.1"),
        tensor=lambda *_args, **_kwargs: _Tensor(),
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False)
        ),
    )
    fake_ultralytics = SimpleNamespace(__version__="8.0.0")
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(
        worker, "emit", lambda event, **data: events.append((event, data))
    )

    worker.run_probe()

    assert events[0][0] == "environment_detected"
    assert events[0][1]["cpu_available"] is True
    assert events[0][1]["cuda_available"] is False
    assert events[0][1]["cuda_error"]


def test_probe_keeps_usable_gpu_when_another_gpu_fails(monkeypatch):
    def tensor(*_args, device=None, **_kwargs):
        if device == "cuda:1":
            raise RuntimeError("device failure")
        return _Tensor()

    fake_torch = SimpleNamespace(
        __version__="2.0.0",
        version=SimpleNamespace(cuda="12.1"),
        tensor=tensor,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_name=lambda index: f"GPU {index}",
            synchronize=lambda _index: None,
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False)
        ),
    )
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(__version__="8.0.0"),
    )
    monkeypatch.setattr(
        worker, "emit", lambda event, **data: events.append((event, data))
    )

    worker.run_probe()

    result = events[0][1]
    assert result["cuda_available"] is True
    assert result["gpus"][0]["available"] is True
    assert result["gpus"][1]["available"] is False
    assert result["gpus"][1]["error"] == "device failure"


def test_probe_rejects_failed_cpu_tensor(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.0.0",
        version=SimpleNamespace(cuda=None),
        tensor=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("CPU failure")
        ),
    )
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        worker, "emit", lambda event, **data: events.append((event, data))
    )

    worker.run_probe()

    assert events[0][0] == "environment_error"
    assert events[0][1]["error_type"] == "cpu_unavailable"
    assert events[0][1]["environment"]["cpu_available"] is False


def test_probe_reports_failed_mps_tensor_without_disabling_cpu(monkeypatch):
    def tensor(*_args, device=None, **_kwargs):
        if device == "mps":
            raise RuntimeError("MPS failure")
        return _Tensor()

    fake_torch = SimpleNamespace(
        __version__="2.0.0",
        version=SimpleNamespace(cuda=None),
        tensor=tensor,
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True)
        ),
    )
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(__version__="8.0.0"),
    )
    monkeypatch.setattr(
        worker, "emit", lambda event, **data: events.append((event, data))
    )

    worker.run_probe()

    result = events[0][1]
    assert result["cpu_available"] is True
    assert result["mps_available"] is False
    assert result["mps_error"] == "MPS failure"


def test_training_device_validation_rejects_cuda_that_became_unavailable():
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 1,
        ),
        tensor=lambda *_args, **_kwargs: _Tensor(),
    )

    with pytest.raises(RuntimeError, match="CUDA is no longer available"):
        worker.validate_selected_device(fake_torch, [0])


def test_export_does_not_install_when_auto_install_is_disabled(
    monkeypatch, tmp_path
):
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"weights")
    monkeypatch.setattr(
        worker,
        "missing_export_packages",
        lambda _format: ["onnx>=1.15.0"],
    )
    monkeypatch.setattr(
        worker,
        "install_export_packages",
        lambda _packages: pytest.fail("pip must not be called"),
    )
    payload = {
        "args": {"format": "onnx"},
        "paths": {"weights_path": str(weights_path)},
        "options": {"auto_install_packages": False},
    }

    with pytest.raises(RuntimeError, match="Missing required packages"):
        worker.run_export(payload)


def test_export_install_reports_missing_pip(monkeypatch):
    class FailedInstaller:
        stdout = iter(["/training/python: No module named pip\n"])
        returncode = 1

        def poll(self):
            return self.returncode

    commands = []

    def start_installer(command, **_kwargs):
        commands.append(command)
        return FailedInstaller()

    monkeypatch.setattr(
        worker.subprocess,
        "Popen",
        start_installer,
    )
    monkeypatch.setattr(worker, "emit", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="pip is not available"):
        worker.install_export_packages(["onnxslim>=0.1.59"])
    assert commands == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "onnxslim>=0.1.59",
        ]
    ]


def test_train_runs_with_only_worker_payload_and_framework_modules(
    monkeypatch, tmp_path
):
    calls = {}

    class FakeYOLO:
        def __init__(self, model):
            calls["model"] = model

        def train(self, **kwargs):
            calls["train"] = kwargs

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    fake_ultralytics.settings = {}
    fake_torch = SimpleNamespace(
        tensor=lambda *_args, **_kwargs: _Tensor(),
    )
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(
        worker, "emit", lambda event, **data: events.append((event, data))
    )
    monkeypatch.chdir(tmp_path)
    weights_directory = tmp_path / "weights"
    payload = {
        "args": {
            "model": "yolo11n.pt",
            "data": str(tmp_path / "data.yaml"),
            "project": str(tmp_path / "runs"),
            "name": "exp",
            "device": "cpu",
            "epochs": 1,
        },
        "paths": {"weights_directory": str(weights_directory)},
        "options": {"auto_install_packages": False},
    }

    worker.run_train(payload)

    assert calls["model"] == "yolo11n.pt"
    assert calls["train"]["device"] == "cpu"
    assert calls["train"]["epochs"] == 1
    assert [event for event, _data in events] == [
        "training_started",
        "training_completed",
    ]


def test_export_accepts_directory_output(monkeypatch, tmp_path):
    output_directory = tmp_path / "best_openvino_model"
    output_directory.mkdir()
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"weights")

    class FakeYOLO:
        def __init__(self, _model):
            pass

        def export(self, **_kwargs):
            return output_directory

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    events = []
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(worker, "missing_export_packages", lambda _format: [])
    monkeypatch.setattr(
        worker, "emit", lambda event, **data: events.append((event, data))
    )

    worker.run_export(
        {
            "args": {"format": "openvino"},
            "paths": {"weights_path": str(weights_path)},
            "options": {"auto_install_packages": False},
        }
    )

    completed = [data for event, data in events if event == "export_completed"]
    assert completed == [
        {
            "exported_path": str(output_directory),
            "format": "openvino",
        }
    ]
