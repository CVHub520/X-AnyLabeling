import base64
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from anylabeling.services.updates import protocol

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "anylabeling/services/updates"))
import installer


@pytest.fixture
def signing():
    key = Ed25519PrivateKey.generate()
    public = base64.b64encode(
        key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode()
    info = {
        "schema": 1,
        "version": "1.0.0",
        "platform": "Linux",
        "arch": "x86_64",
        "variant": "CPU",
        "public_key": public,
        "feed_url": "https://example.com/feed.json",
        "managed_paths": ["X-AnyLabeling", "build-info.json"],
    }

    def sign(artifact, version="2.0.0", notes="New release"):
        payload = {
            "schema": 1,
            "version": version,
            "release_notes": notes,
            "artifacts": {f"{info['platform']}-x86_64-CPU": artifact},
        }
        return {
            "payload": payload,
            "signature": base64.b64encode(
                key.sign(protocol.canonical_json(payload))
            ).decode(),
        }

    return info, sign


def artifact(data=b"update"):
    return {
        "url": "https://example.com/update.zip",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "root": "X-AnyLabeling",
    }


def test_signed_selection_and_version_matching(signing):
    info, sign = signing
    assert (
        protocol.select_artifact(sign(artifact()), info)["version"] == "2.0.0"
    )
    assert protocol.select_artifact(sign(artifact(), "1.0.0"), info) is None
    assert (
        protocol.select_artifact(sign(artifact()), {**info, "arch": "arm64"})
        is None
    )
    assert (
        protocol.select_artifact(
            sign(artifact()), {**info, "variant": "CUDA12"}
        )
        is None
    )


def test_modified_signature_is_rejected(signing):
    info, sign = signing
    envelope = sign(artifact())
    envelope["payload"]["artifacts"]["Linux-x86_64-CPU"][
        "url"
    ] = "https://attacker.example/update.zip"
    with pytest.raises(ValueError, match="signature"):
        protocol.select_artifact(envelope, info)


def test_invalid_release_notes_are_rejected_before_reaching_qt(signing):
    info, sign = signing
    with pytest.raises(ValueError, match="content"):
        protocol.select_artifact(
            sign(artifact(), notes={"invalid": True}), info
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"url": "http://example.com/update.zip"},
        {"url": "https://user:pass@example.com/update.zip"},
        {"root": "../app"},
        {"sha256": "x" * 64},
        {"size": 0},
    ],
)
def test_invalid_signed_artifact_is_rejected(signing, changes):
    info, sign = signing
    with pytest.raises(ValueError):
        protocol.select_artifact(sign({**artifact(), **changes}), info)


class Response:
    def __init__(self, chunks, url="https://example.com/update.zip"):
        self.chunks = chunks
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield from self.chunks


def test_download_checks_digest_and_cancellation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        protocol.requests, "get", lambda *a, **k: Response([b"up", b"date"])
    )
    path = tmp_path / "update.zip"
    progress = []
    assert (
        protocol.download_update(
            artifact(), path, threading.Event(), progress.append
        )
        == path
    )
    assert path.read_bytes() == b"update" and progress[-1] == 100
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(protocol.UpdateCancelled):
        protocol.download_update(
            artifact(), tmp_path / "cancel.zip", cancel, lambda value: None
        )
    assert not (tmp_path / "cancel.partial").exists()
    assert not (tmp_path / "cancel.zip").exists()


@pytest.mark.parametrize("chunks", [[b"corrupt"], [b"up"], [b"update extra"]])
def test_damaged_download_never_replaces_existing_file(
    tmp_path, monkeypatch, chunks
):
    monkeypatch.setattr(
        protocol.requests, "get", lambda *a, **k: Response(chunks)
    )
    path = tmp_path / "update.zip"
    path.write_bytes(b"existing")
    with pytest.raises(ValueError):
        protocol.download_update(
            artifact(), path, threading.Event(), lambda value: None
        )
    assert path.read_bytes() == b"existing"
    assert not path.with_suffix(".partial").exists()


def test_https_redirect_downgrade_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        protocol.requests,
        "get",
        lambda *a, **k: Response([b"update"], "http://example.com/update.zip"),
    )
    with pytest.raises(ValueError, match="HTTPS"):
        protocol.download_update(
            artifact(),
            tmp_path / "update.zip",
            threading.Event(),
            lambda value: None,
        )


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "OtherApp/file",
        "X-AnyLabeling/../../escape",
        "X-AnyLabeling/file:stream",
        "X-AnyLabeling\\escape",
    ],
)
def test_archive_path_traversal_is_rejected(tmp_path, name):
    path = tmp_path / "update.zip"
    with zipfile.ZipFile(path, "w") as archive:
        entry = zipfile.ZipInfo(name)
        entry.filename = name
        archive.writestr(entry, "bad")
    with pytest.raises(ValueError, match="unsafe"):
        installer.extract_update(path, tmp_path / "stage", "X-AnyLabeling")
    assert not (tmp_path / "escape").exists()


@pytest.mark.skipif(
    sys.platform == "win32", reason="Requires POSIX permissions and symlinks"
)
def test_archive_preserves_executable_and_safe_macos_symlinks(tmp_path):
    path = tmp_path / "update.zip"
    with zipfile.ZipFile(path, "w") as archive:
        executable = zipfile.ZipInfo(
            "X-AnyLabeling.app/Contents/MacOS/X-AnyLabeling"
        )
        executable.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(executable, "executable")
        link = zipfile.ZipInfo("X-AnyLabeling.app/Contents/MacOS/current")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "X-AnyLabeling")
    root = installer.extract_update(
        path, tmp_path / "stage", "X-AnyLabeling.app"
    )
    assert (
        root / "Contents/MacOS/X-AnyLabeling"
    ).stat().st_mode & stat.S_IXUSR
    assert (
        root / "Contents/MacOS/current"
    ).resolve().read_text() == "executable"


def test_archive_rejects_external_symlink(tmp_path):
    path = tmp_path / "update.zip"
    with zipfile.ZipFile(path, "w") as archive:
        link = zipfile.ZipInfo("X-AnyLabeling/link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "../../escape")
    with pytest.raises(ValueError, match="unsafe"):
        installer.extract_update(path, tmp_path / "stage", "X-AnyLabeling")


def test_directory_replacement_keeps_uninstaller_and_rolls_back(
    tmp_path, signing
):
    info, _ = signing
    old = tmp_path / "installed"
    new = tmp_path / "stage/X-AnyLabeling"
    old.mkdir()
    new.mkdir(parents=True)
    (old / "build-info.json").write_text(json.dumps(info))
    (old / "X-AnyLabeling").write_text("old")
    (old / "unins000.dat").write_text("uninstaller")
    (old / "annotations").mkdir()
    (old / "annotations/image.json").write_text("labels")
    (new / "X-AnyLabeling").write_text("new")
    transaction = installer.Replacement(old, new, "test")
    transaction.apply()
    assert (old / "X-AnyLabeling").read_text() == "new"
    assert (old / "unins000.dat").read_text() == "uninstaller"
    assert (old / "annotations/image.json").read_text() == "labels"
    transaction.rollback()
    assert (old / "X-AnyLabeling").read_text() == "old"


@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux installation layout"
)
def test_linux_link_switch_and_rollback(tmp_path):
    old = tmp_path / "versions/old"
    old.mkdir(parents=True)
    target = tmp_path / "current"
    target.symlink_to(old)
    new = tmp_path / ".update-2.0-test/X-AnyLabeling"
    new.mkdir(parents=True)
    transaction = installer.Replacement(target, new, "test")
    transaction.apply()
    assert target.resolve() != old
    transaction.rollback()
    assert target.resolve() == old


def test_second_rename_failure_restores_original(
    tmp_path, signing, monkeypatch
):
    info, _ = signing
    old = tmp_path / "installed"
    new = tmp_path / "stage"
    old.mkdir()
    new.mkdir()
    (old / "build-info.json").write_text(json.dumps(info))
    transaction = installer.Replacement(old, new, "test")
    rename = Path.rename

    def fail_new(path, destination):
        if path == new:
            raise OSError("simulated disk failure")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_new)
    with pytest.raises(OSError):
        transaction.apply()
    transaction.rollback()
    assert (old / "build-info.json").exists()


@pytest.mark.parametrize("fail_startup", [False, True])
def test_real_helper_process_updates_or_restores_old_app(
    tmp_path, signing, fail_startup, monkeypatch
):
    info, sign = signing
    root = tmp_path / "installed app"
    root.mkdir()
    (root / "build-info.json").write_text(json.dumps(info))
    pid_file = tmp_path / "app.pid"
    script = f"""#!{sys.executable}
import os, time
from pathlib import Path
Path({str(pid_file)!r}).write_text(str(os.getpid()))
ready = os.environ.get('X_ANYLABELING_UPDATE_READY_FILE')
if ready:
    Path(ready).write_text(os.environ['X_ANYLABELING_UPDATE_TOKEN'])
while True:
    time.sleep(0.1)
"""
    arguments = []
    executable_name = "X-AnyLabeling"
    executable_data = script.encode()
    if sys.platform == "win32":
        info["platform"] = "Windows"
        executable_name += ".exe"
        executable_data = Path(sys.executable).read_bytes()
        shutil.copy2(Path(sys.base_prefix) / "python312.dll", root)
        monkeypatch.setenv("PYTHONHOME", sys.base_prefix)
        native_script = script.split("\n", 1)[1]
        if fail_startup:
            native_script = (
                f"import json\nfrom pathlib import Path\n"
                f"if json.loads(Path({str(root / 'build-info.json')!r}).read_text())['version'] == '2.0.0': raise SystemExit(1)\n"
                + native_script
            )
        arguments = ["-c", native_script]
        (root / "build-info.json").write_text(json.dumps(info))
    (root / executable_name).write_bytes(executable_data)
    (root / executable_name).chmod(0o755)
    archive = tmp_path / "update.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(
            "X-AnyLabeling/build-info.json",
            json.dumps({**info, "version": "2.0.0"}),
        )
        entry = zipfile.ZipInfo(f"X-AnyLabeling/{executable_name}")
        entry.external_attr = (stat.S_IFREG | 0o755) << 16
        package.writestr(
            entry,
            (
                f"#!{sys.executable}\nraise SystemExit(1)\n"
                if fail_startup and sys.platform != "win32"
                else executable_data
            ),
        )
    plan = {
        "root": str(root),
        "target": str(root),
        "manifest": sign(artifact(archive.read_bytes())),
        "archive": str(archive),
        "pid": 999999,
        "arguments": arguments,
        "token": "test-token",
        "ready": str(tmp_path / "ready"),
        "armed": str(tmp_path / "armed"),
        "cancelled": str(tmp_path / "cancelled"),
        "result": str(tmp_path / "result.json"),
    }
    (tmp_path / "armed").touch()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    try:
        with (tmp_path / "helper.log").open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "anylabeling/services/updates/installer.py"),
                    str(path),
                ],
                stdout=log,
                stderr=log,
                timeout=20,
            )
        assert result.returncode == (1 if fail_startup else 0), (
            tmp_path / "helper.log"
        ).read_text()
        assert json.loads((root / "build-info.json").read_text())[
            "version"
        ] == ("1.0.0" if fail_startup else "2.0.0")
        assert json.loads(Path(plan["result"]).read_text())["status"] == (
            "failed" if fail_startup else "success"
        )
        assert not (tmp_path / ".installed app-update.lock").exists()
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists()
    finally:
        if pid_file.exists():
            pid = int(pid_file.read_text())
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                process.terminate()
                process.wait(timeout=5)


def test_child_environment_does_not_reuse_old_runtime(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/old/_internal")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/system/lib")
    monkeypatch.setenv("_PYI_APPLICATION_HOME_DIR", "/old")
    env = protocol.child_environment()
    assert env["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    assert env["LD_LIBRARY_PATH"] == "/system/lib"
    assert "_PYI_APPLICATION_HOME_DIR" not in env


def test_interrupted_directory_update_can_be_recovered(tmp_path, signing):
    info, _ = signing
    target = tmp_path / "installed"
    target.mkdir()
    backup = tmp_path / ".installed-backup-test"
    backup.mkdir()
    (backup / "build-info.json").write_text(json.dumps(info))
    (backup / "X-AnyLabeling").write_text("old")
    (target / "X-AnyLabeling").write_text("new")
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "status": "installing",
                "target": str(target),
                "backup": str(backup),
                "previous_link": None,
            }
        )
    )
    installer.recover_interrupted(result)
    assert (target / "X-AnyLabeling").read_text() == "old"
    assert list(tmp_path.glob(".installed-interrupted-*"))
    assert json.loads(result.read_text())["status"] == "failed"


def test_recovery_refuses_unrelated_backup_directory(tmp_path):
    target = tmp_path / "installed"
    target.mkdir()
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "status": "installing",
                "target": str(target),
                "backup": str(tmp_path / "unrelated"),
                "previous_link": None,
            }
        )
    )
    with pytest.raises(ValueError, match="backup"):
        installer.recover_interrupted(result)
    assert target.exists()


def test_restart_preserves_relative_cli_paths(tmp_path, monkeypatch):
    directory = tmp_path / "user project"
    directory.mkdir()
    launches = []
    monkeypatch.setattr(
        installer.subprocess,
        "Popen",
        lambda args, **kwargs: launches.append((args, kwargs))
        or SimpleNamespace(pid=1),
    )
    installer.launch(
        tmp_path / "app",
        "Linux",
        ["--filename", "./image.png"],
        working_directory=str(directory),
    )
    assert launches[0][1]["cwd"] == str(directory)
    assert launches[0][0][-1] == "./image.png"


@pytest.mark.parametrize(
    "python_platform,expected",
    [("win-amd64", "x86_64"), ("win-arm64", "arm64")],
)
def test_windows_update_architecture_matches_interpreter_on_arm_hardware(
    monkeypatch, python_platform, expected
):
    monkeypatch.setattr(protocol.sys, "platform", "win32")
    monkeypatch.setattr(protocol.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(
        protocol.sysconfig, "get_platform", lambda: python_platform
    )
    assert protocol.architecture() == expected


def test_intel_macos_updates_do_not_select_apple_silicon_packages(signing):
    info, sign = signing
    info["platform"] = "Darwin"
    envelope = sign({**artifact(), "root": "X-AnyLabeling.app"})
    assert protocol.select_artifact(envelope, info)["version"] == "2.0.0"
    assert (
        protocol.select_artifact(envelope, {**info, "arch": "arm64"}) is None
    )
