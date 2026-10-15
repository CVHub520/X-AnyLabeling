import base64
import hashlib
import json
import os
import platform
import sys
import sysconfig
from pathlib import Path
from urllib.parse import urlsplit

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from packaging.version import Version

DEFAULT_FEED = (
    "https://github.com/CVHub520/X-AnyLabeling/releases/latest/download/"
    "update-manifest.json"
)
PROTOCOL_VERSION = 1


class UpdateCancelled(Exception):
    pass


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def architecture():
    name = (
        sysconfig.get_platform().removeprefix("win-")
        if sys.platform == "win32"
        else platform.machine()
    ).lower()
    return {"amd64": "x86_64", "aarch64": "arm64"}.get(name, name)


def require_https(url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username:
        raise ValueError(
            "Update addresses must use HTTPS without credentials."
        )
    return url


def build_info_path(root):
    root = Path(root)
    for path in (
        root / "Contents/Resources/build-info.json",
        root / "build-info.json",
    ):
        if path.is_file():
            return path
    paths = list(root.glob("[0-9]*/build-info.json"))
    if len(paths) > 1:
        raise ValueError("Multiple installed application versions were found.")
    return paths[0] if paths else root / "build-info.json"


def read_build_info(root):
    return json.loads(build_info_path(root).read_text(encoding="utf-8-sig"))


def installation():
    if not getattr(sys, "frozen", False):
        return None
    executable = Path(sys.executable).resolve()
    root = executable.parent
    resources = root
    single_file = sys.platform.startswith("linux") and hasattr(sys, "_MEIPASS")
    if single_file:
        resources = Path(sys._MEIPASS)
    if sys.platform == "darwin":
        root = next(
            (p for p in executable.parents if p.suffix == ".app"), None
        )
        if root is None:
            return None
    try:
        info = read_build_info(resources if single_file else root)
    except (OSError, ValueError):
        return None
    if info.get("schema") != PROTOCOL_VERSION or not info.get("public_key"):
        return None
    if (
        info.get("platform") != platform.system()
        or info.get("arch") != architecture()
    ):
        return None
    if single_file:
        root = executable
    target = root
    if sys.platform.startswith("linux") and root.parent.name == "versions":
        link = root.parent.parent / "current"
        if link.is_symlink() and link.resolve() == root:
            target = link
    helper = (
        root / "Contents/Helpers/xanylabeling-updater"
        if sys.platform == "darwin"
        else (
            build_info_path(root).parent
            if sys.platform == "win32"
            else resources
        )
        / (
            "xanylabeling-updater.exe"
            if sys.platform == "win32"
            else "xanylabeling-updater"
        )
    )
    if not helper.is_file():
        return None
    return {
        "root": str(root),
        "target": str(target),
        "helper": str(helper),
        "info": info,
    }


def cache_directory(context):
    if sys.platform == "win32":
        base = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")
        )
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    identity = hashlib.sha256(context["target"].encode()).hexdigest()[:16]
    path = base / "X-AnyLabeling/updates" / identity
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def verify_manifest(envelope, public_key):
    payload = envelope["payload"]
    key = Ed25519PublicKey.from_public_bytes(
        base64.b64decode(public_key, validate=True)
    )
    try:
        key.verify(
            base64.b64decode(envelope["signature"], validate=True),
            canonical_json(payload),
        )
    except InvalidSignature as error:
        raise ValueError(
            "The update manifest signature is invalid."
        ) from error
    if payload.get("schema") != PROTOCOL_VERSION:
        raise ValueError("This release requires a newer update protocol.")
    Version(payload["version"])
    if not isinstance(payload.get("release_notes", ""), str) or not isinstance(
        payload.get("artifacts"), dict
    ):
        raise ValueError("Invalid update manifest content.")
    return payload


def select_artifact(envelope, info):
    payload = verify_manifest(envelope, info["public_key"])
    if Version(payload["version"]) <= Version(info["version"]):
        return None
    identity = f"{info['platform']}-{info['arch']}-{info['variant']}"
    artifact = payload["artifacts"].get(identity)
    if artifact is None:
        return None
    require_https(artifact["url"])
    expected_root = (
        "X-AnyLabeling.app"
        if info["platform"] == "Darwin"
        else "X-AnyLabeling"
    )
    expected_kind = {
        "Windows": "installer",
        "Darwin": "dmg",
        "Linux": "executable",
    }[info["platform"]]
    if artifact.get("kind") != expected_kind:
        raise ValueError("Unexpected update package format.")
    if artifact["root"] != expected_root:
        raise ValueError("Unexpected update archive layout.")
    if type(artifact["size"]) is not int or artifact["size"] <= 0:
        raise ValueError("Invalid update download size.")
    digest = artifact["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Invalid update checksum.")
    return {
        **artifact,
        "version": payload["version"],
        "release_notes": payload.get("release_notes", ""),
    }


def fetch_manifest(info, timeout=10):
    url = require_https(info["feed_url"])
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        require_https(response.url)
        data = bytearray()
        for chunk in response.iter_content(64 * 1024):
            data.extend(chunk)
            if len(data) > 1024 * 1024:
                raise ValueError("The update manifest is too large.")
    envelope = json.loads(data)
    return envelope, select_artifact(envelope, info)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_update(artifact, destination, cancel, progress):
    destination = Path(destination)
    partial = destination.with_suffix(".partial")
    size = 0
    digest = hashlib.sha256()
    try:
        with requests.get(
            require_https(artifact["url"]), stream=True, timeout=(10, 30)
        ) as response:
            response.raise_for_status()
            require_https(response.url)
            with partial.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if cancel.is_set():
                        raise UpdateCancelled()
                    size += len(chunk)
                    if size > artifact["size"]:
                        raise ValueError(
                            "The update download exceeds its signed size."
                        )
                    stream.write(chunk)
                    digest.update(chunk)
                    progress(int(size * 100 / artifact["size"]))
        if cancel.is_set():
            raise UpdateCancelled()
        if (
            size != artifact["size"]
            or digest.hexdigest() != artifact["sha256"]
        ):
            raise ValueError(
                "The update download failed integrity verification."
            )
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def child_environment():
    env = dict(os.environ)
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        original = env.pop(name + "_ORIG", None)
        env.pop(name, None)
        if original:
            env[name] = original
    for name in list(env):
        if name.startswith("_PYI_"):
            env.pop(name)
    env.pop("X_ANYLABELING_UPDATE_READY_FILE", None)
    env.pop("X_ANYLABELING_UPDATE_TOKEN", None)
    env.pop("X_ANYLABELING_UPDATE_VERSION", None)
    return env
