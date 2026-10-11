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


def test_packaged_archive_and_signed_manifest_round_trip(
    tmp_path, monkeypatch
):
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
    bundle = tmp_path / "X-AnyLabeling"
    bundle.mkdir()
    (bundle / "X-AnyLabeling").write_text("application")
    helper = tmp_path / "xanylabeling-updater"
    helper.write_text("helper")
    packaging.prepare(bundle, "CPU", helper)
    output = tmp_path / "release"
    packaging.package(bundle, output)
    identity = f"{packaging.platform.system()}-{packaging.architecture()}-CPU"
    notes = tmp_path / "notes.md"
    notes.write_text("Release notes")
    packaging.manifest(output, "2.0.0", notes, [identity])
    info = json.loads((bundle / "build-info.json").read_text())
    envelope = json.loads((output / "update-manifest.json").read_text())
    selected = select_artifact(envelope, {**info, "version": "1.0.0"})
    assert selected["version"] == "2.0.0"
    assert selected["release_notes"] == "Release notes"
    with pytest.raises(ValueError, match="required"):
        packaging.manifest(
            output, "2.0.0", notes, [identity, "missing-platform"]
        )
    archive = next(output.glob("*-Update.zip"))
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
