import base64
import importlib.util
import json
import os
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)

from anylabeling.services.updates.protocol import select_artifact

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "package_updates", ROOT / "scripts/package_updates.py"
)
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


@pytest.mark.parametrize(
    "system,kind,name",
    [
        ("Windows", "installer", "X-AnyLabeling-Setup.exe"),
        ("Darwin", "dmg", "X-AnyLabeling-macOS-unsigned.dmg"),
        ("Linux", "executable", "X-AnyLabeling-Linux-CPU"),
    ],
)
def test_release_artifact_and_signed_manifest_round_trip(
    tmp_path, monkeypatch, system, kind, name
):
    monkeypatch.setattr(packaging.platform, "system", lambda: system)
    key = Ed25519PrivateKey.generate()
    public = base64.b64encode(
        key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode()
    private = base64.b64encode(
        key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    ).decode()
    monkeypatch.setenv("X_ANYLABELING_UPDATE_PUBLIC_KEY", public)
    monkeypatch.setenv("X_ANYLABELING_UPDATE_SIGNING_KEY", private)
    monkeypatch.setattr(packaging, "app_version", lambda: "2.0.0")
    is_macos = packaging.platform.system() == "Darwin"
    bundle = tmp_path / ("X-AnyLabeling.app" if is_macos else "X-AnyLabeling")
    bundle.mkdir()
    if is_macos:
        (bundle / "Contents/Resources").mkdir(parents=True)
        (bundle / "Contents/MacOS").mkdir()
    executable = bundle / (
        "Contents/MacOS/X-AnyLabeling" if is_macos else "X-AnyLabeling"
    )
    executable.write_text("application")
    helper = tmp_path / "xanylabeling-updater"
    helper.write_text("helper")
    packaging.prepare(bundle, "CPU", helper)
    output = tmp_path / "release"
    output.mkdir()
    archive = output / name
    archive.write_bytes(b"existing release package")
    packaging.package(bundle, output, archive)
    assert archive.read_bytes() == b"existing release package"
    assert not list(output.glob("*-Update.zip"))
    identity = f"{packaging.platform.system()}-{packaging.architecture()}-CPU"
    notes = tmp_path / "notes.md"
    notes.write_text("Release notes")
    packaging.manifest(output, "2.0.0", notes, [identity])
    info = json.loads(packaging.metadata_path(bundle).read_text())
    envelope = json.loads((output / "update-manifest.json").read_text())
    selected = select_artifact(envelope, {**info, "version": "1.0.0"})
    assert selected["kind"] == kind
    assert selected["url"].endswith("/" + name)
    assert selected["version"] == "2.0.0"
    assert selected["release_notes"] == "Release notes"
    with pytest.raises(ValueError, match="required"):
        packaging.manifest(
            output, "2.0.0", notes, [identity, "missing-platform"]
        )
    archive.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="changed"):
        packaging.manifest(output, "2.0.0", notes, [identity])


def test_key_generator_does_not_overwrite_or_print_private_key(tmp_path):
    import subprocess
    import sys

    key = tmp_path / "signing.key"
    command = [
        sys.executable,
        str(ROOT / "scripts/generate_update_key.py"),
        "--output",
        str(key),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=True
    )
    assert key.read_text() not in result.stdout + result.stderr
    if os.name != "nt":
        assert key.stat().st_mode & 0o777 == 0o600
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_windows_packaging_keeps_update_files_in_version_directory(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(packaging.platform, "system", lambda: "Windows")
    monkeypatch.setattr(packaging, "app_version", lambda: "4.1.0")
    bundle = tmp_path / "X-AnyLabeling"
    resources = bundle / "4.1.0"
    resources.mkdir(parents=True)
    (bundle / "X-AnyLabeling.exe").write_text("application")
    helper = tmp_path / "xanylabeling-updater.exe"
    helper.write_text("helper")
    packaging.prepare(bundle, "CPU", helper)
    assert {p.name for p in bundle.iterdir()} == {"X-AnyLabeling.exe", "4.1.0"}
    assert (resources / helper.name).read_text() == "helper"
    info = json.loads(packaging.metadata_path(bundle).read_text())
    assert info["managed_paths"] == ["4.1.0", "X-AnyLabeling.exe"]
    setup = tmp_path / "Setup.exe"
    setup.write_bytes(b"installer")
    packaging.package(bundle, tmp_path / "release", setup)
