import argparse
import base64
import datetime
import functools
import http.server
import ipaddress
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from cryptography.x509.oid import NameOID


def gui_environment(directory, interactive=False):
    env = dict(
        os.environ,
        HOME=str(directory / "home"),
        XDG_CACHE_HOME=str(directory / "cache"),
        XDG_CONFIG_HOME=str(directory / "config"),
    )
    Path(env["HOME"]).mkdir()
    env["QT_QPA_PLATFORM"] = {"win32": "windows", "darwin": "cocoa"}.get(
        sys.platform, "xcb"
    )
    if not interactive:
        env["QT_QPA_PLATFORM"] = "offscreen"
    if sys.platform == "win32":
        env.update(
            USERPROFILE=env["HOME"],
            APPDATA=str(directory / "roaming"),
            LOCALAPPDATA=str(directory / "local"),
        )
        Path(env["APPDATA"]).mkdir()
        Path(env["LOCALAPPDATA"]).mkdir()
    return env


def interactive_update(root, directory, version):
    from protocol import read_build_info

    env = gui_environment(directory, interactive=True)
    executable = root / (
        "Contents/MacOS/X-AnyLabeling"
        if sys.platform == "darwin"
        else (
            "X-AnyLabeling.exe" if sys.platform == "win32" else "X-AnyLabeling"
        )
    )
    cache = (
        Path(
            env["LOCALAPPDATA"]
            if sys.platform == "win32"
            else env["XDG_CACHE_HOME"]
        )
        / "X-AnyLabeling/updates"
    )
    with (directory / "gui.log").open("w") as log:
        subprocess.Popen(
            [str(executable), "--work-dir", str(directory / "user-data")],
            env=env,
            cwd=directory,
            stdout=log,
            stderr=log,
        )
        print(
            "Click the update icon beside Keyboard shortcuts, then Update and Restart. This test uses temporary user data.",
            flush=True,
        )
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            for path in cache.glob("*/result.json"):
                result = json.loads(path.read_text())
                if result["status"] == "failed":
                    raise RuntimeError(result["error"])
                if result["status"] == "success":
                    assert read_build_info(root)["version"] == version
                    print(
                        "PASS: The GUI downloaded, installed, and restarted the updated application.",
                        flush=True,
                    )
                    input(
                        "Inspect the restarted application, then press Enter to close and remove the temporary installation: "
                    )
                    return
            time.sleep(0.1)
        raise TimeoutError(
            "No successful interactive update within ten minutes."
        )


def certificate(directory):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = directory / "tls.pem"
    key_path = directory / "tls-key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    with os.fdopen(
        os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
    ) as stream:
        stream.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    return cert_path, key_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.bundle = args.bundle or args.repo / "dist/X-AnyLabeling"
    sys.path.insert(0, str(args.repo / "anylabeling/services/updates"))
    from protocol import (
        canonical_json,
        download_update,
        fetch_manifest,
        file_digest,
        read_build_info,
    )

    if sys.platform not in ("linux", "win32", "darwin"):
        parser.error(
            "This isolated GUI smoke test requires Linux, Windows, or macOS."
        )
    info = read_build_info(args.bundle)
    key = ed25519.Ed25519PrivateKey.from_private_bytes(
        base64.b64decode(args.signing_key.read_bytes(), validate=True)
    )
    public = base64.b64encode(
        key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    ).decode()
    if public != info["public_key"]:
        raise ValueError(
            "The signing key does not match the built application."
        )
    with tempfile.TemporaryDirectory(
        prefix="xanylabeling-update-smoke-"
    ) as temporary:
        directory = Path(temporary)
        root = directory / (
            "installed app.app"
            if sys.platform == "darwin"
            else "installed app"
        )
        shutil.copytree(args.bundle, root, symlinks=True)
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            functools.partial(
                http.server.SimpleHTTPRequestHandler, directory=str(directory)
            ),
        )
        cert_path, key_path = certificate(directory)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"https://127.0.0.1:{server.server_port}"
        archive_link = directory / "package.zip"
        try:
            os.link(args.archive.resolve(), archive_link)
        except OSError:
            shutil.copy2(args.archive, archive_link)
        identity = f"{info['platform']}-{info['arch']}-{info['variant']}"
        payload = {
            "schema": 1,
            "version": info["version"],
            "release_notes": "Local smoke test",
            "artifacts": {
                identity: {
                    "url": url + "/package.zip",
                    "size": args.archive.stat().st_size,
                    "sha256": file_digest(args.archive),
                    "root": (
                        "X-AnyLabeling.app"
                        if sys.platform == "darwin"
                        else "X-AnyLabeling"
                    ),
                }
            },
        }
        envelope = {
            "payload": payload,
            "signature": base64.b64encode(
                key.sign(canonical_json(payload))
            ).decode(),
        }
        (directory / "manifest.json").write_text(json.dumps(envelope))
        old_info = {
            **info,
            "version": "0.0.0",
            "feed_url": url + "/manifest.json",
        }
        metadata = root / (
            "Contents/Resources/build-info.json"
            if sys.platform == "darwin"
            else "build-info.json"
        )
        metadata.write_text(json.dumps(old_info))
        if sys.platform == "darwin":
            subprocess.run(
                ["codesign", "--force", "--deep", "--sign", "-", str(root)],
                check=True,
            )

        old_ca = os.environ.get("REQUESTS_CA_BUNDLE")
        os.environ["REQUESTS_CA_BUNDLE"] = str(cert_path)
        try:
            if args.interactive:
                print(
                    "The old version is simulated by lowering installed metadata to 0.0.0.",
                    flush=True,
                )
                interactive_update(root, directory, info["version"])
                return
            manifest, selected = fetch_manifest(old_info)
            downloaded = download_update(
                selected,
                directory / "download.zip",
                threading.Event(),
                lambda value: None,
            )
            token = "local-smoke-test"
            plan = {
                "root": str(root),
                "target": str(root),
                "manifest": manifest,
                "archive": str(downloaded),
                "pid": 0,
                "arguments": [
                    "--work-dir",
                    str(directory / "user-data"),
                    "--no-auto-update-check",
                ],
                "token": token,
                "ready": str(directory / "ready"),
                "helper_ready": str(directory / "helper-ready"),
                "armed": str(directory / "armed"),
                "cancelled": str(directory / "cancelled"),
                "result": str(directory / "result.json"),
            }
            helper_name = (
                "xanylabeling-updater.exe"
                if sys.platform == "win32"
                else "xanylabeling-updater"
            )
            helper = directory / helper_name
            helper_source = (
                root
                / ("Contents/Helpers" if sys.platform == "darwin" else "")
                / helper_name
            )
            shutil.copy2(helper_source, helper)
            path = directory / "plan.json"
            path.write_text(json.dumps(plan))
            (directory / "armed").touch()
            env = gui_environment(directory)
            with (directory / "updater.log").open("w") as log:
                process = subprocess.run(
                    [str(helper), str(path)],
                    env=env,
                    stdout=log,
                    stderr=log,
                    timeout=180,
                )
            if process.returncode != 0:
                raise RuntimeError((directory / "updater.log").read_text())
            assert (
                json.loads(Path(plan["result"]).read_text())["status"]
                == "success"
            )
            assert read_build_info(root)["version"] == info["version"]
            assert Path(plan["ready"]).read_text() == token
            assert Path(plan["helper_ready"]).read_text() == token
            print(
                "PASS: HTTPS discovery, signed manifest, verified download, bundled updater, application replacement, and real GUI startup acknowledgement."
            )
            print(
                "The old version was simulated by lowering installed metadata; this does not replace testing two real release versions."
            )
        finally:
            for process in psutil.process_iter(["exe"]):
                try:
                    if process.info["exe"] and Path(
                        process.info["exe"]
                    ).is_relative_to(directory):
                        children = process.children(recursive=True)
                        process.terminate()
                        for child in children:
                            child.terminate()
                        _, alive = psutil.wait_procs(
                            [process, *children], timeout=5
                        )
                        for child in alive:
                            child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            server.shutdown()
            server.server_close()
            if old_ca is None:
                os.environ.pop("REQUESTS_CA_BUNDLE", None)
            else:
                os.environ["REQUESTS_CA_BUNDLE"] = old_ca


if __name__ == "__main__":
    main()
