import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import psutil

from protocol import (
    child_environment,
    file_digest,
    read_build_info,
    select_artifact,
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def stage_update(archive, destination, artifact):
    destination = Path(destination)
    staged = destination / artifact["root"]
    if artifact["kind"] == "dmg":
        if sys.platform != "darwin":
            raise ValueError("DMG updates require macOS.")
        mount = Path(tempfile.mkdtemp(prefix="xanylabeling-volume-"))
        attached = False
        try:
            subprocess.run(
                [
                    "/usr/bin/hdiutil",
                    "attach",
                    str(archive),
                    "-readonly",
                    "-nobrowse",
                    "-noautoopen",
                    "-mountpoint",
                    str(mount),
                ],
                check=True,
                env=child_environment(),
            )
            attached = True
            source = mount / artifact["root"]
            if source.is_symlink() or not source.is_dir():
                raise ValueError(
                    "The disk image does not contain an application bundle."
                )
            subprocess.run(
                ["/usr/bin/ditto", str(source), str(staged)],
                check=True,
                env=child_environment(),
            )
        finally:
            if attached or os.path.ismount(mount):
                subprocess.run(
                    ["/usr/bin/hdiutil", "detach", str(mount)],
                    check=True,
                    env=child_environment(),
                )
            if mount.exists():
                mount.rmdir()
        return staged
    if artifact["kind"] == "executable":
        shutil.copy2(archive, staged)
        staged.chmod(0o755)
        return staged
    if artifact["kind"] != "installer" or sys.platform != "win32":
        raise ValueError("Unsupported update package format.")
    import ctypes

    ctypes.windll.kernel32.SetDllDirectoryW(None)
    subprocess.run(
        [
            str(archive),
            "/UPDATESTAGE=1",
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/NOCANCEL",
            "/SP-",
            "/NOCLOSEAPPLICATIONS",
            "/NORESTARTAPPLICATIONS",
            f"/DIR={staged}",
            f"/LOG={archive.parent / 'setup.log'}",
        ],
        check=True,
        env=child_environment(),
    )
    return staged


def executable_in(root, system):
    root = Path(root)
    if root.is_file():
        return root
    if system == "Darwin":
        return root / "Contents/MacOS/X-AnyLabeling"
    return root / (
        "X-AnyLabeling.exe" if system == "Windows" else "X-AnyLabeling"
    )


def wait_for_exit(pid, root, timeout=120):
    if pid > 0:
        try:
            psutil.Process(pid).wait(timeout=timeout)
        except psutil.NoSuchProcess:
            pass
    deadline = time.monotonic() + timeout
    root = Path(root).resolve()
    while time.monotonic() < deadline:
        running = False
        for process in psutil.process_iter(["pid", "exe"]):
            if process.pid == os.getpid():
                continue
            try:
                if process.info["exe"] and Path(
                    process.info["exe"]
                ).resolve().is_relative_to(root):
                    running = True
                    break
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                continue
        if not running:
            return
        time.sleep(0.2)
    raise TimeoutError("Close other X-AnyLabeling instances before updating.")


def launch(
    root,
    system,
    arguments,
    ready_file=None,
    token=None,
    working_directory=None,
    expected_version=None,
):
    env = child_environment()
    if ready_file:
        env["X_ANYLABELING_UPDATE_READY_FILE"] = str(ready_file)
        env["X_ANYLABELING_UPDATE_TOKEN"] = token
        if expected_version:
            env["X_ANYLABELING_UPDATE_VERSION"] = expected_version
    directory = Path(working_directory) if working_directory else Path.home()
    options = {
        "env": env,
        "cwd": str(directory if directory.is_dir() else Path.home()),
    }
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetDllDirectoryW(None)
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    return subprocess.Popen(
        [str(executable_in(root, system)), *arguments], **options
    )


def wait_for_ready(process, ready_file, token, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if Path(ready_file).exists() and Path(ready_file).read_text() == token:
            return
        if process.poll() is not None:
            raise RuntimeError(
                "The new application exited before startup completed."
            )
        time.sleep(0.2)
    raise TimeoutError(
        "The new application did not confirm a successful startup."
    )


def stop_application(process):
    if process is None or process.poll() is not None:
        return
    children = []
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        pass
    process.terminate()
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    _, alive = psutil.wait_procs(children, timeout=10)
    for child in alive:
        child.kill()
    psutil.wait_procs(alive, timeout=10)


class Replacement:
    def __init__(self, target, staged, token):
        self.target = Path(target)
        self.staged = Path(staged)
        self.backup = self.target.with_name(
            f".{self.target.name}-backup-{token}"
        )
        self.previous_link = None
        self.moved_old = False
        self.switched = False

    def apply(self):
        if self.target.is_symlink():
            self.previous_link = os.readlink(self.target)
            versions = self.target.parent / "versions"
            versions.mkdir(exist_ok=True)
            final = versions / self.staged.parent.name
            self.staged.rename(final)
            temporary = self.target.with_name(f".current-{uuid.uuid4().hex}")
            temporary.symlink_to(final)
            temporary.replace(self.target)
            self.switched = True
            return
        if self.target.is_dir():
            managed = set(read_build_info(self.target)["managed_paths"])
            for item in self.target.iterdir():
                if (
                    item.name not in managed
                    and not (self.staged / item.name).exists()
                ):
                    if item.is_dir() and not item.is_symlink():
                        shutil.copytree(
                            item, self.staged / item.name, symlinks=True
                        )
                    elif item.is_symlink():
                        (self.staged / item.name).symlink_to(os.readlink(item))
                    else:
                        shutil.copy2(item, self.staged / item.name)
        deadline = time.monotonic() + 10
        while True:
            try:
                self.target.rename(self.backup)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
        self.moved_old = True
        self.staged.rename(self.target)
        self.switched = True

    def rollback(self):
        if self.previous_link is not None and self.switched:
            temporary = self.target.with_name(f".rollback-{uuid.uuid4().hex}")
            temporary.symlink_to(self.previous_link)
            temporary.replace(self.target)
        elif self.moved_old:
            if self.switched:
                failed = self.target.with_name(
                    f".{self.target.name}-failed-{uuid.uuid4().hex}"
                )
                self.target.rename(failed)
            self.backup.rename(self.target)

    def finish(self):
        if self.backup.exists():
            if self.backup.is_dir():
                shutil.rmtree(self.backup)
            else:
                self.backup.unlink()


def recover_interrupted(result_path):
    journal = json.loads(Path(result_path).read_text())
    if journal.get("status") != "installing":
        raise ValueError("There is no interrupted installation to recover.")
    target = Path(journal["target"])
    wait_for_exit(0, target)
    previous = journal.get("previous_link")
    if previous:
        old = Path(previous).resolve()
        if not old.is_relative_to(target.parent / "versions"):
            raise ValueError(
                "The previous installation is outside the version directory."
            )
        read_build_info(old)
        temporary = target.with_name(f".recover-{uuid.uuid4().hex}")
        temporary.symlink_to(previous)
        temporary.replace(target)
    else:
        backup = Path(journal["backup"])
        if (
            backup.parent != target.parent
            or not backup.name.startswith(f".{target.name}-backup-")
            or backup.is_symlink()
        ):
            raise ValueError("The recovery backup path is invalid.")
        if backup.is_dir():
            read_build_info(backup)
        if target.exists():
            target.rename(
                target.with_name(
                    f".{target.name}-interrupted-{uuid.uuid4().hex}"
                )
            )
        backup.rename(target)
    write_json(
        result_path,
        {
            "status": "failed",
            "error": "An interrupted update was recovered. The previous version is restored.",
        },
    )


def update_windows_version(info, root):
    if sys.platform != "win32":
        return
    import winreg

    base = "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\"
    for identity in ("X-AnyLabeling", f"X-AnyLabeling-{info['variant']}"):
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                base + identity + "_is1",
                0,
                winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE,
            ) as key:
                location = winreg.QueryValueEx(key, "InstallLocation")[0]
                if Path(location).resolve() != Path(root).resolve():
                    continue
                winreg.SetValueEx(
                    key, "DisplayVersion", 0, winreg.REG_SZ, info["version"]
                )
                return
        except OSError:
            continue


def run_update(plan):
    root = Path(plan["root"])
    target = Path(plan["target"])
    info = plan["info"] if root.is_file() else read_build_info(root)
    artifact = select_artifact(plan["manifest"], info)
    if artifact is None:
        raise ValueError(
            "The update is no longer newer than the installed version."
        )
    archive = Path(plan["archive"])
    if (
        archive.stat().st_size != artifact["size"]
        or file_digest(archive) != artifact["sha256"]
    ):
        raise ValueError("The downloaded update changed after verification.")
    if target.resolve() != root.resolve():
        raise ValueError("The installed application changed before updating.")
    wait_for_exit(plan["pid"], root)
    lock_path = target.parent / f".{target.name}-update.lock"
    if lock_path.exists():
        owner = int(lock_path.read_text())
        if psutil.pid_exists(owner):
            raise RuntimeError("Another update is already in progress.")
        lock_path.unlink()
    with lock_path.open("x") as lock:
        lock.write(str(os.getpid()))
    transaction = None
    process = None
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".update-{artifact['version']}-", dir=target.parent
        ) as stage:
            staged = stage_update(archive, stage, artifact)
            new_info = (
                {**info, "version": artifact["version"]}
                if artifact["kind"] == "executable"
                else read_build_info(staged)
            )
            for key in ("platform", "arch", "variant", "public_key"):
                if new_info[key] != info[key]:
                    raise ValueError(
                        f"The update does not match the installed {key}."
                    )
            if new_info["version"] != artifact["version"]:
                raise ValueError(
                    "The update version does not match its signed manifest."
                )
            if not executable_in(staged, info["platform"]).is_file():
                raise ValueError(
                    "The update does not contain an application executable."
                )
            transaction = Replacement(target, staged, plan["token"])
            journal = {
                "target": str(target),
                "backup": str(transaction.backup),
                "previous_link": (
                    os.readlink(target) if target.is_symlink() else None
                ),
                "version": artifact["version"],
                "status": "installing",
            }
            write_json(plan["result"], journal)
            transaction.apply()
            process = launch(
                target,
                info["platform"],
                plan["arguments"],
                plan["ready"],
                plan["token"],
                working_directory=plan.get("cwd"),
                expected_version=artifact["version"],
            )
            wait_for_ready(process, plan["ready"], plan["token"])
            write_json(
                plan["result"],
                {
                    **journal,
                    "status": "success",
                    "directory": str(archive.parent),
                },
            )
            update_windows_version(new_info, target)
            try:
                transaction.finish()
            except OSError:
                pass
            try:
                archive.unlink(missing_ok=True)
            except OSError:
                pass
    except Exception as error:
        stop_application(process)
        if transaction is not None:
            transaction.rollback()
        write_json(
            plan["result"],
            {
                "status": "failed",
                "error": str(error),
                "version": artifact["version"],
                "restarted": True,
            },
        )
        launch(
            target,
            info["platform"],
            plan["arguments"],
            working_directory=plan.get("cwd"),
        )
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path, nargs="?")
    parser.add_argument("--recover", type=Path)
    args = parser.parse_args()
    if args.recover:
        recover_interrupted(args.recover)
        return
    if args.plan is None:
        parser.error("Specify an update plan or --recover.")
    plan = json.loads(args.plan.read_text())
    if plan.get("helper_ready"):
        ready = Path(plan["helper_ready"])
        temporary = ready.with_suffix(".tmp")
        temporary.write_text(plan["token"])
        temporary.replace(ready)
    try:
        parent = psutil.Process(plan["pid"]) if plan["pid"] > 0 else None
    except psutil.NoSuchProcess:
        parent = None
    while not Path(plan["armed"]).exists():
        if (
            Path(plan["cancelled"]).exists()
            or parent is None
            or not parent.is_running()
        ):
            return
        time.sleep(0.1)
    try:
        run_update(plan)
    except Exception as error:
        result = Path(plan["result"])
        previous = json.loads(result.read_text()) if result.exists() else {}
        if previous.get("status") == "installing":
            write_json(result, {**previous, "error": str(error)})
            sys.exit(1)
        if previous.get("status") != "failed" or not previous.get("restarted"):
            write_json(
                result, {**previous, "status": "failed", "error": str(error)}
            )
            target = Path(plan["target"])
            info = (
                plan["info"] if target.is_file() else read_build_info(target)
            )
            if executable_in(target, info["platform"]).is_file():
                launch(
                    target,
                    info["platform"],
                    plan["arguments"],
                    working_directory=plan.get("cwd"),
                )
        sys.exit(1)


if __name__ == "__main__":
    main()
