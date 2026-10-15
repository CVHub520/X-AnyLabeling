import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
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
        "kind": "executable",
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_windows_installation_finds_versioned_and_legacy_helpers(
    tmp_path, signing, monkeypatch, legacy
):
    info, _ = signing
    info = {**info, "platform": "Windows"}
    resources = tmp_path if legacy else tmp_path / info["version"]
    resources.mkdir(exist_ok=True)
    (resources / "build-info.json").write_text(json.dumps(info))
    helper = resources / "xanylabeling-updater.exe"
    helper.touch()
    executable = tmp_path / "X-AnyLabeling.exe"
    executable.touch()
    monkeypatch.setattr(protocol.sys, "frozen", True, raising=False)
    monkeypatch.setattr(protocol.sys, "platform", "win32")
    monkeypatch.setattr(protocol.sys, "executable", str(executable))
    monkeypatch.setattr(protocol.platform, "system", lambda: "Windows")
    monkeypatch.setattr(protocol, "architecture", lambda: "x86_64")
    context = protocol.installation()
    assert context["helper"] == str(helper)
    assert context["info"] == info


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("rollback", [False, True])
def test_versioned_replacement_preserves_uninstaller_and_supports_rollback(
    tmp_path, signing, legacy, rollback
):
    info, _ = signing
    root = tmp_path / "installed"
    stage = tmp_path / "stage"
    root.mkdir()
    stage.mkdir()
    old_resources = root / ("_internal" if legacy else "1.0.0")
    old_resources.mkdir()
    old_metadata = root if legacy else old_resources
    (old_metadata / "build-info.json").write_text(
        json.dumps(
            {
                **info,
                "managed_paths": (
                    ["X-AnyLabeling.exe", "_internal", "build-info.json"]
                    if legacy
                    else ["X-AnyLabeling.exe", "1.0.0"]
                ),
            }
        )
    )
    (root / "X-AnyLabeling.exe").write_text("old")
    (root / "unins000.exe").write_text("uninstaller")
    (root / "unins000.dat").write_text("uninstall log")
    (stage / "X-AnyLabeling.exe").write_text("new")
    (stage / "2.0.0").mkdir()
    (stage / "2.0.0/build-info.json").write_text(
        json.dumps(
            {
                **info,
                "version": "2.0.0",
                "managed_paths": ["X-AnyLabeling.exe", "2.0.0"],
            }
        )
    )
    transaction = installer.Replacement(root, stage, "test")
    transaction.apply()
    assert protocol.read_build_info(root)["version"] == "2.0.0"
    assert not old_resources.exists()
    if rollback:
        transaction.rollback()
        assert protocol.read_build_info(root)["version"] == "1.0.0"
        assert (root / "X-AnyLabeling.exe").read_text() == "old"
    else:
        transaction.finish()
        assert not transaction.backup.exists()
    assert (root / "unins000.exe").read_text() == "uninstaller"
    assert (root / "unins000.dat").read_text() == "uninstall log"


@pytest.mark.parametrize("matching_installation", [False, True])
@pytest.mark.parametrize("identity", ["X-AnyLabeling", "X-AnyLabeling-CPU"])
def test_windows_version_updates_only_the_matching_installation(
    tmp_path, monkeypatch, matching_installation, identity
):
    from unittest.mock import MagicMock

    writes = []
    registered = tmp_path / "registered"
    target = registered if matching_installation else tmp_path / "portable"
    key = MagicMock()

    def open_key(hive, path, reserved, access):
        if not path.endswith(identity + "_is1"):
            raise FileNotFoundError(path)
        return key

    registry = SimpleNamespace(
        HKEY_CURRENT_USER=0,
        KEY_SET_VALUE=2,
        KEY_QUERY_VALUE=1,
        REG_SZ=1,
        OpenKey=open_key,
        QueryValueEx=lambda key, name: (str(registered), 1),
        SetValueEx=lambda key, name, reserved, kind, value: writes.append(
            (name, value)
        ),
    )
    monkeypatch.setitem(sys.modules, "winreg", registry)
    monkeypatch.setattr(installer.sys, "platform", "win32")
    installer.update_windows_version(
        {"variant": "CPU", "version": "4.1.1"}, target
    )
    assert writes == (
        [("DisplayVersion", "4.1.1")] if matching_installation else []
    )


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
        {"kind": "installer"},
        {"kind": "unknown"},
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


@pytest.mark.skipif(
    os.name == "nt", reason="Requires POSIX executable fixtures"
)
@pytest.mark.parametrize("fail_startup", [False, True])
@pytest.mark.parametrize("system", ["Linux", "Darwin"])
def test_real_helper_process_updates_or_restores_old_app(
    tmp_path, signing, fail_startup, system
):
    if system == "Darwin" and sys.platform != "darwin":
        pytest.skip("Requires native macOS disk image tools")
    info, sign = signing
    info["platform"] = system
    root = tmp_path / (
        "installed app" if system == "Linux" else "Installed.app"
    )
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
    new_script = (
        f"#!{sys.executable}\nraise SystemExit(1)\n"
        if fail_startup
        else script + "\nversion = '2.0.0'\n"
    )
    archive = tmp_path / "release-package"
    if system == "Linux":
        root.write_text(script)
        root.chmod(0o755)
        archive.write_text(new_script)
        package_info = artifact(archive.read_bytes())
        executable = root
    else:
        executable = root / "Contents/MacOS/X-AnyLabeling"
        executable.parent.mkdir(parents=True)
        executable.write_text(script)
        executable.chmod(0o755)
        metadata = root / "Contents/Resources/build-info.json"
        metadata.parent.mkdir()
        metadata.write_text(json.dumps(info))
        import shutil

        image_root = tmp_path / "image-root"
        bundle = image_root / "X-AnyLabeling.app"
        shutil.copytree(root, bundle)
        (bundle / "Contents/MacOS/X-AnyLabeling").write_text(new_script)
        (bundle / "Contents/Resources/build-info.json").write_text(
            json.dumps({**info, "version": "2.0.0"})
        )
        (bundle / "Contents/MacOS/current").symlink_to("X-AnyLabeling")
        archive = archive.with_suffix(".dmg")
        subprocess.run(
            [
                "/usr/bin/hdiutil",
                "create",
                "-ov",
                "-fs",
                "HFS+",
                "-srcfolder",
                str(image_root),
                "-format",
                "UDZO",
                str(archive),
            ],
            check=True,
            capture_output=True,
        )
        package_info = {
            **artifact(archive.read_bytes()),
            "root": "X-AnyLabeling.app",
            "kind": "dmg",
        }
    plan = {
        "root": str(root),
        "target": str(root),
        "info": info,
        "manifest": sign(package_info),
        "archive": str(archive),
        "pid": 999999,
        "arguments": [],
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
                timeout=120 if system == "Darwin" else 20,
            )
        assert result.returncode == (1 if fail_startup else 0), (
            tmp_path / "helper.log"
        ).read_text()
        if system == "Darwin" and not fail_startup:
            assert (executable.parent / "current").is_symlink()
        assert executable.read_text() == (
            script if fail_startup else new_script
        )
        assert json.loads(Path(plan["result"]).read_text())["status"] == (
            "failed" if fail_startup else "success"
        )
        assert not list(tmp_path.glob(".*-update.lock"))
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
    envelope = sign({**artifact(), "root": "X-AnyLabeling.app", "kind": "dmg"})
    assert protocol.select_artifact(envelope, info)["version"] == "2.0.0"
    assert (
        protocol.select_artifact(envelope, {**info, "arch": "arm64"}) is None
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Linux single-file layout")
def test_linux_single_file_uses_embedded_metadata_and_helper(
    tmp_path, signing, monkeypatch
):
    info, _ = signing
    resources = tmp_path / "_MEI123"
    resources.mkdir()
    (resources / "build-info.json").write_text(json.dumps(info))
    helper = resources / "xanylabeling-updater"
    helper.write_bytes(b"helper")
    executable = tmp_path / "custom application name"
    executable.touch()
    alias = tmp_path / "application-link"
    alias.symlink_to(executable)
    monkeypatch.setattr(protocol.sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        protocol.sys, "_MEIPASS", str(resources), raising=False
    )
    monkeypatch.setattr(protocol.sys, "executable", str(alias))
    monkeypatch.setattr(protocol, "architecture", lambda: "x86_64")
    context = protocol.installation()
    assert context["root"] == context["target"] == str(executable)
    assert context["helper"] == str(helper)
    assert context["info"] == info


def test_interrupted_single_file_update_can_be_recovered(tmp_path):
    target = tmp_path / "application"
    target.write_bytes(b"new")
    backup = tmp_path / ".application-backup-test"
    backup.write_bytes(b"old")
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
    assert target.read_bytes() == b"old"
    assert json.loads(result.read_text())["status"] == "failed"


@pytest.mark.skipif(os.name == "nt", reason="Requires POSIX symlinks")
@pytest.mark.parametrize(
    "failure", [None, "attach", "missing", "symlink", "copy", "detach"]
)
def test_dmg_staging_detaches_and_preserves_existing_installation(
    tmp_path, monkeypatch, failure
):
    import shutil

    mount = tmp_path / "mount"
    mount.mkdir()
    destination = tmp_path / "staging"
    destination.mkdir()
    installed = tmp_path / "installed.app"
    installed.mkdir()
    (installed / "user-data").write_text("untouched")
    image = tmp_path / "release.dmg"
    image.touch()
    calls = []
    monkeypatch.setattr(installer.sys, "platform", "darwin")
    monkeypatch.setattr(
        installer.tempfile, "mkdtemp", lambda **kwargs: str(mount)
    )

    def run(args, **kwargs):
        operation = "copy" if args[0].endswith("ditto") else args[1]
        calls.append(operation)
        assert kwargs["check"]
        if operation == failure:
            raise subprocess.CalledProcessError(1, args)
        if operation == "attach":
            assert "-readonly" in args and "-nobrowse" in args
            if failure == "missing":
                return
            bundle = mount / "X-AnyLabeling.app"
            if failure == "symlink":
                bundle.symlink_to(installed, target_is_directory=True)
                return
            bundle.mkdir()
            (bundle / "application").write_text("new")
            (bundle / "current").symlink_to("application")
        elif operation == "copy":
            shutil.copytree(args[1], args[2], symlinks=True)
        elif operation == "detach":
            for child in mount.iterdir():
                if child.is_symlink():
                    child.unlink()
                else:
                    shutil.rmtree(child)

    monkeypatch.setattr(installer.subprocess, "run", run)
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            installer.stage_update(
                image,
                destination,
                {"kind": "dmg", "root": "X-AnyLabeling.app"},
            )
    else:
        staged = installer.stage_update(
            image, destination, {"kind": "dmg", "root": "X-AnyLabeling.app"}
        )
        assert (staged / "application").read_text() == "new"
        assert (staged / "current").is_symlink()
    assert calls == (
        ["attach"]
        if failure == "attach"
        else (
            ["attach", "detach"]
            if failure in ("missing", "symlink")
            else ["attach", "copy", "detach"]
        )
    )
    assert mount.exists() == (failure == "detach")
    assert (installed / "user-data").read_text() == "untouched"
