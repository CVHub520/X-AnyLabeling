import argparse
import ast
import base64
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "anylabeling/services/updates"))
from protocol import (
    DEFAULT_FEED,
    architecture,
    build_info_path,
    canonical_json,
    file_digest,
    require_https,
)


def app_version():
    tree = ast.parse((ROOT / "anylabeling/app_info.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "__version__" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Application version is missing.")


def metadata_path(bundle):
    return build_info_path(bundle)


def prepare(bundle, variant, helper):
    system = platform.system()
    info = {
        "schema": 1,
        "version": app_version(),
        "platform": system,
        "arch": architecture(),
        "variant": variant,
        "public_key": os.environ.get("X_ANYLABELING_UPDATE_PUBLIC_KEY", ""),
        "feed_url": os.environ.get("X_ANYLABELING_UPDATE_FEED_URL")
        or DEFAULT_FEED,
    }
    require_https(info["feed_url"])
    if (
        info["public_key"]
        and len(base64.b64decode(info["public_key"], validate=True)) != 32
    ):
        raise ValueError(
            "The update public key must be a base64-encoded Ed25519 key."
        )
    resources = bundle / info["version"] if system == "Windows" else bundle
    target = (
        bundle / "Contents/Helpers" / helper.name
        if system == "Darwin"
        else resources / helper.name
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(helper, target)
    info["managed_paths"] = sorted(
        {p.name for p in bundle.iterdir()}
        | ({"build-info.json"} if system != "Windows" else set())
    )
    path = (
        resources / "build-info.json"
        if system == "Windows"
        else (
            bundle / "Contents/Resources/build-info.json"
            if system == "Darwin"
            else bundle / "build-info.json"
        )
    )
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def package(bundle, output, path):
    info = json.loads(metadata_path(bundle).read_text())
    identity = f"{info['platform']}-{info['arch']}-{info['variant']}"
    kind = {"Windows": "installer", "Darwin": "dmg", "Linux": "executable"}[
        info["platform"]
    ]
    output.mkdir(parents=True, exist_ok=True)
    name = path.name
    repository = os.environ.get("GITHUB_REPOSITORY", "CVHub520/X-AnyLabeling")
    base = (
        os.environ.get("X_ANYLABELING_RELEASE_BASE_URL")
        or f"https://github.com/{repository}/releases/download/v{info['version']}/"
    )
    require_https(base)
    digest = file_digest(path)
    descriptor = {
        "identity": identity,
        "version": info["version"],
        "public_key": info["public_key"],
        "artifact": {
            "url": base.rstrip("/") + "/" + name,
            "sha256": digest,
            "size": path.stat().st_size,
            "root": (
                "X-AnyLabeling.app"
                if info["platform"] == "Darwin"
                else "X-AnyLabeling"
            ),
            "kind": kind,
        },
    }
    (output / f"update-artifact-{identity}.json").write_text(
        json.dumps(descriptor, indent=2)
    )
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {name}\n")
    print(f"Registered release artifact: {path}")


def manifest(directory, version, notes, expected):
    private_key = Ed25519PrivateKey.from_private_bytes(
        base64.b64decode(
            os.environ["X_ANYLABELING_UPDATE_SIGNING_KEY"], validate=True
        )
    )
    public_key = base64.b64encode(
        private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode()
    artifacts = {}
    for path in sorted(directory.glob("update-artifact-*.json")):
        descriptor = json.loads(path.read_text())
        if (
            descriptor["version"] != version
            or descriptor["public_key"] != public_key
        ):
            raise ValueError(
                "The release artifact version or signing key does not match this release."
            )
        identity = descriptor["identity"]
        if identity in artifacts:
            raise ValueError("Duplicate release artifact identity.")
        artifact = descriptor["artifact"]
        archive = directory / artifact["url"].rsplit("/", 1)[-1]
        if (
            archive.stat().st_size != artifact["size"]
            or file_digest(archive) != artifact["sha256"]
        ):
            raise ValueError(
                "A release artifact changed during release assembly."
            )
        artifacts[identity] = artifact
    if not artifacts or (expected and set(artifacts) != set(expected)):
        raise ValueError(
            "The release does not contain all required release artifacts."
        )
    payload = {
        "schema": 1,
        "version": version,
        "release_notes": notes.read_text(),
        "artifacts": artifacts,
    }
    envelope = {
        "payload": payload,
        "signature": base64.b64encode(
            private_key.sign(canonical_json(payload))
        ).decode(),
    }
    (directory / "update-manifest.json").write_text(
        json.dumps(envelope, indent=2)
    )


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--bundle", type=Path, required=True)
    preparation.add_argument("--helper", type=Path, required=True)
    preparation.add_argument(
        "--variant",
        choices=["CPU", "CUDA11", "CUDA12", "CUDA13"],
        required=True,
    )
    archive = commands.add_parser("package")
    archive.add_argument("--bundle", type=Path, required=True)
    archive.add_argument("--artifact", type=Path, required=True)
    archive.add_argument("--output", type=Path, default=ROOT / "dist")
    signing = commands.add_parser("manifest")
    signing.add_argument("--directory", type=Path, required=True)
    signing.add_argument("--version", required=True)
    signing.add_argument("--notes", type=Path, required=True)
    signing.add_argument("--expected", nargs="*", default=[])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.bundle, args.variant, args.helper)
    elif args.command == "package":
        package(args.bundle, args.output, args.artifact)
    else:
        manifest(args.directory, args.version, args.notes, args.expected)


if __name__ == "__main__":
    main()
