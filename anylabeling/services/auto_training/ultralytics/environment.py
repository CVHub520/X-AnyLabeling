from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from typing import Any, Callable

from anylabeling.config import get_work_directory

WORKER_EVENT_PREFIX = "__XANYLABELING_WORKER_EVENT__="
WORKER_PROTOCOL_VERSION = 1


def get_default_training_python() -> str | None:
    return None if getattr(sys, "frozen", False) else sys.executable


def expand_training_path(value: str, base_directory: str | None = None) -> str:
    expanded = os.path.expandvars(value.strip())
    expanded = re.sub(
        r"%([^%]+)%",
        lambda match: os.environ.get(match.group(1), match.group(0)),
        expanded,
    )
    expanded = os.path.expanduser(expanded)
    if not os.path.isabs(expanded):
        expanded = os.path.join(
            base_directory or get_work_directory(), expanded
        )
    return os.path.abspath(expanded)


def resolve_training_python(value: str | None) -> str:
    if not value:
        return sys.executable
    return expand_training_path(value)


def resolve_training_data_directory(value: str | None) -> str:
    if value:
        return expand_training_path(value)
    return os.path.join(
        get_work_directory(), "xanylabeling_data", "trainer", "ultralytics"
    )


def prepare_training_data_directory(path: str) -> None:
    for directory in ("datasets", "weights", "runs"):
        os.makedirs(os.path.join(path, directory), exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path):
        pass


def get_worker_script_path() -> str:
    relative_path = Path(
        "anylabeling/services/auto_training/ultralytics/worker.py"
    )
    candidates = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / relative_path)
    candidates.append(Path(__file__).resolve().with_name("worker.py"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "The bundled Ultralytics worker script was not found"
    )


def create_worker_payload(
    action: str,
    args: dict[str, Any] | None = None,
    paths: dict[str, str] | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    payload = {
        "version": WORKER_PROTOCOL_VERSION,
        "action": action,
        "args": args or {},
        "paths": paths or {},
        "options": options or {},
    }
    fd, payload_path = tempfile.mkstemp(
        prefix="xanylabeling-worker-", suffix=".json"
    )
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)
    return payload_path


def build_worker_command(
    python_executable: str, payload_path: str
) -> list[str]:
    return [python_executable, "-u", get_worker_script_path(), payload_path]


def build_probe_command(
    python_executable: str,
    payload_path: str,
    external_environment: bool,
) -> list[str]:
    if not external_environment and getattr(sys, "frozen", False):
        return [
            sys.executable,
            "--work-dir",
            get_work_directory(),
            "training-worker",
            "--payload",
            payload_path,
        ]
    return build_worker_command(python_executable, payload_path)


def build_worker_environment(auto_install_packages: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["YOLO_AUTOINSTALL"] = "true" if auto_install_packages else "false"
    return env


def get_worker_creation_flags() -> int:
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "CREATE_NO_WINDOW", 0
    )


def parse_worker_output(
    output: str,
    notify_callbacks: Callable[[str, dict], None],
    plain_log_event: str = "training_log",
) -> bool:
    cleaned_output = output.strip()
    if not cleaned_output:
        return False
    if not cleaned_output.startswith(WORKER_EVENT_PREFIX):
        notify_callbacks(plain_log_event, {"message": cleaned_output})
        return False
    payload_text = cleaned_output[len(WORKER_EVENT_PREFIX) :]
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        notify_callbacks(plain_log_event, {"message": cleaned_output})
        return False
    if payload.get("version") != WORKER_PROTOCOL_VERSION:
        notify_callbacks(
            "environment_error",
            {
                "error": "Unsupported worker protocol",
                "error_type": "protocol_error",
            },
        )
        return True
    event_type = payload.pop("event", "")
    payload.pop("version", None)
    if not event_type:
        return False
    notify_callbacks(event_type, payload)
    return (
        event_type == "environment_detected"
        or event_type.endswith("_completed")
        or event_type.endswith("_error")
    )


def terminate_process_tree(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except OSError:
        process.kill()


class EnvironmentManager:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[str, dict], None]] = []
        self._process: subprocess.Popen | None = None
        self._request_id: str | None = None
        self._lock = threading.Lock()

    def notify_callbacks(self, event_type: str, data: dict) -> None:
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception:
                pass

    def probe(
        self,
        python_executable: str,
        auto_install_packages: bool = True,
        timeout: float = 30,
        external_environment: bool = False,
    ) -> str:
        request_id = uuid.uuid4().hex
        with self._lock:
            previous_process = self._process
            self._request_id = request_id
        if previous_process is not None:
            threading.Thread(
                target=terminate_process_tree,
                args=(previous_process,),
                daemon=True,
            ).start()
        payload_path = create_worker_payload("probe")

        def run_probe() -> None:
            try:
                command = build_probe_command(
                    python_executable,
                    payload_path,
                    external_environment,
                )
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=build_worker_environment(auto_install_packages),
                    start_new_session=os.name != "nt",
                    creationflags=get_worker_creation_flags(),
                )
                with self._lock:
                    if self._request_id != request_id:
                        terminate_process_tree(process)
                        return
                    self._process = process
                try:
                    output, _ = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    terminate_process_tree(process)
                    self._notify_current(
                        request_id,
                        "environment_error",
                        {
                            "error": f"Environment detection timed out after {timeout:g} seconds",
                            "error_type": "probe_timeout",
                        },
                    )
                    return
                terminal_event_seen = False
                diagnostic_lines = []
                for line in output.decode(
                    "utf-8", errors="replace"
                ).splitlines():
                    if line.strip() and not line.strip().startswith(
                        WORKER_EVENT_PREFIX
                    ):
                        diagnostic_lines.append(line.strip())
                        diagnostic_lines = diagnostic_lines[-20:]
                    terminal_event_seen = (
                        parse_worker_output(
                            line,
                            lambda event, data: self._notify_current(
                                request_id, event, data
                            ),
                        )
                        or terminal_event_seen
                    )
                if process.returncode and not terminal_event_seen:
                    self._notify_current(
                        request_id,
                        "environment_error",
                        {
                            "error": f"Environment process exited with code {process.returncode}",
                            "error_type": "process_exit",
                            "details": "\n".join(diagnostic_lines),
                        },
                    )
                elif not terminal_event_seen:
                    self._notify_current(
                        request_id,
                        "environment_error",
                        {
                            "error": "Environment process returned no result",
                            "error_type": "invalid_output",
                        },
                    )
            except Exception as exc:
                error_type = (
                    "worker_missing"
                    if "bundled Ultralytics worker" in str(exc)
                    else "python_start_failed"
                )
                self._notify_current(
                    request_id,
                    "environment_error",
                    {
                        "error": (
                            f"Failed to start training Python "
                            f"'{python_executable}': {exc}"
                        ),
                        "error_type": error_type,
                    },
                )
            finally:
                try:
                    os.remove(payload_path)
                except OSError:
                    pass
                with self._lock:
                    if self._request_id == request_id:
                        self._process = None

        threading.Thread(target=run_probe, daemon=True).start()
        return request_id

    def _notify_current(
        self, request_id: str, event_type: str, data: dict
    ) -> None:
        with self._lock:
            if self._request_id != request_id:
                return
        self.notify_callbacks(event_type, data)

    def cancel(self) -> None:
        with self._lock:
            process = self._process
            self._request_id = None
        if process is not None:
            threading.Thread(
                target=terminate_process_tree,
                args=(process,),
                daemon=True,
            ).start()


_environment_manager = EnvironmentManager()


def get_environment_manager() -> EnvironmentManager:
    return _environment_manager
