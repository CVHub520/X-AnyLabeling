import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from PyQt6.QtCore import (
    QCoreApplication,
    QEventLoop,
    QObject,
    QTimer,
    pyqtSignal,
)

from anylabeling.app_info import __version__
from anylabeling.services.updates.protocol import (
    UpdateCancelled,
    cache_directory,
    child_environment,
    download_update,
    fetch_manifest,
    installation,
)
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.widgets.update_panel import UpdatePanel


class UpdateController(QObject):
    checked = pyqtSignal(object, object, bool)
    downloaded = pyqtSignal(str)
    failed = pyqtSignal(str, bool)
    progress = pyqtSignal(int)

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.context = installation()
        self.enabled = self.context is not None
        self.busy = False
        self.closing = False
        self.cancel = threading.Event()
        self.dialog = None
        self.manifest = None
        self.artifact = None
        wrapper = getattr(window, "labeling_widget", None)
        self.button = getattr(
            getattr(wrapper, "view", None), "update_button", None
        )
        self.anchor = getattr(
            getattr(wrapper, "view", None), "shortcuts_button", None
        )
        if self.button is not None:
            self.button.clicked.connect(self.show_panel)
        self.checked.connect(self._on_checked)
        self.downloaded.connect(self._on_downloaded)
        self.failed.connect(self._on_failed)
        self.progress.connect(self._on_progress)
        if self.enabled:
            self.cache = cache_directory(self.context)
            QTimer.singleShot(1000, self._show_previous_result)

    def check(self, manual=False):
        if not self.enabled or self.busy or self.closing:
            return
        self.busy = True

        def work():
            try:
                envelope, artifact = fetch_manifest(self.context["info"])
                if not self.closing:
                    self.checked.emit(envelope, artifact, manual)
            except Exception as error:
                if not self.closing:
                    self.failed.emit(str(error), manual)

        threading.Thread(target=work, daemon=True).start()

    def _on_checked(self, envelope, artifact, manual):
        self.busy = False
        if self.closing:
            return
        self.manifest = envelope
        self.artifact = artifact
        if self.button is not None:
            self.button.setVisible(artifact is not None)
            if artifact is not None:
                self.button.setToolTip(
                    self.tr("Update to {version}").format(
                        version=artifact["version"]
                    )
                )
        if artifact is None:
            if manual:
                self._panel().set_state("current")
                self.show_panel()
            return
        if self.dialog is not None:
            self.dialog.set_state("available", artifact["version"])
        if manual:
            self.show_panel()

    def _panel(self):
        if self.dialog is None:
            self.dialog = UpdatePanel(self.window)
            self.dialog.update_requested.connect(self._start_update)
            self.dialog.cancel_requested.connect(self.cancel.set)
        return self.dialog

    def show_panel(self):
        if self.closing:
            return
        panel = self._panel()
        if (
            self.artifact is not None
            and not self.busy
            and panel.mode not in ("error", "downloading", "installing")
        ):
            panel.set_state("available", self.artifact["version"])
        anchor = (
            self.button
            if self.button is not None and self.button.isVisible()
            else self.anchor
        )
        panel.reveal(anchor)

    def _start_update(self):
        if self.busy:
            return
        if self.artifact is not None:
            self._download()
        else:
            self.check(manual=True)

    def _download(self):
        target = Path(self.context["target"])
        if not os.access(target.parent, os.W_OK):
            self._on_failed(
                self.tr(
                    "The installation directory is not writable. Install in a user-owned directory to enable automatic updates."
                ),
                True,
            )
            return
        self.busy = True
        self.cancel.clear()
        self._panel().set_state("downloading")
        self.show_panel()
        directory = Path(tempfile.mkdtemp(prefix="download-", dir=self.cache))

        def work():
            try:
                archive = download_update(
                    self.artifact,
                    directory
                    / {
                        "installer": "setup.exe",
                        "dmg": "update.dmg",
                        "executable": "update.bin",
                    }[self.artifact["kind"]],
                    self.cancel,
                    self.progress.emit,
                )
                if not self.closing:
                    self.downloaded.emit(str(archive))
            except UpdateCancelled:
                shutil.rmtree(directory, ignore_errors=True)
                if not self.closing:
                    self.failed.emit("", False)
            except Exception as error:
                shutil.rmtree(directory, ignore_errors=True)
                if not self.closing:
                    self.failed.emit(str(error), True)

        threading.Thread(target=work, daemon=True).start()

    def _on_progress(self, value):
        if self.dialog is not None:
            self.dialog.set_progress(value)

    def _on_downloaded(self, archive):
        self.busy = False
        if self.closing or self.cancel.is_set():
            shutil.rmtree(Path(archive).parent, ignore_errors=True)
            if self.dialog is not None and not self.closing:
                self.dialog.set_state(
                    "available",
                    self.artifact["version"] if self.artifact else "",
                )
            return
        if self.dialog is not None:
            self.dialog.set_state("installing")
        directory = Path(archive).parent
        token = uuid.uuid4().hex
        helper = directory / Path(self.context["helper"]).name
        plan = {
            "root": self.context["root"],
            "target": self.context["target"],
            "manifest": self.manifest,
            "info": self.context["info"],
            "archive": archive,
            "pid": os.getpid(),
            "arguments": sys.argv[1:],
            "cwd": os.getcwd(),
            "token": token,
            "ready": str(directory / "ready"),
            "helper_ready": str(directory / "helper-ready"),
            "armed": str(directory / "armed"),
            "cancelled": str(directory / "cancelled"),
            "result": str(self.cache / "result.json"),
        }
        app = self.window.app
        quit_on_close = app.quitOnLastWindowClosed()
        approved = False
        self.busy = True
        try:
            shutil.copy2(self.context["helper"], helper)
            if sys.platform != "win32":
                helper.chmod(0o700)
            path = directory / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            prepared = directory / "prepared"
            prepared.write_text(token)
            options = {"env": child_environment(), "cwd": str(directory)}
            if sys.platform == "win32":
                options["creationflags"] = subprocess.CREATE_NO_WINDOW
                ctypes.windll.kernel32.SetDllDirectoryW(None)
            else:
                options["start_new_session"] = True
            with (directory / "updater.log").open("w") as log:
                process = subprocess.Popen(
                    [str(helper), str(path)], stdout=log, stderr=log, **options
                )
            if process.poll() is not None:
                raise RuntimeError(
                    self.tr("The update installer could not start.")
                )
            deadline = time.monotonic() + 15
            helper_ready = Path(plan["helper_ready"])
            while not helper_ready.exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(
                        self.tr(
                            "The update installer did not become ready. Your current application is still running."
                        )
                    )
                if self.closing:
                    Path(plan["cancelled"]).touch()
                    return
                QCoreApplication.processEvents(
                    QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
                )
                time.sleep(0.05)
            if helper_ready.read_text() != token:
                raise RuntimeError(
                    self.tr(
                        "The update installer returned an invalid acknowledgement."
                    )
                )
            app.setQuitOnLastWindowClosed(False)
            if self.window.close():
                approved = True
                prepared.replace(plan["armed"])
                app.quit()
            else:
                Path(plan["cancelled"]).touch()
                self._panel().set_state(
                    "available",
                    self.artifact["version"] if self.artifact else "",
                )
                self.dialog.description.setText(
                    self.tr(
                        "Finish saving your work, then try updating again."
                    )
                )
                self.show_panel()
        except Exception as error:
            Path(plan["cancelled"]).touch()
            if approved:
                self.closing = False
                self.window.show()
            self._on_failed(str(error), True)
        finally:
            self.busy = False
            app.setQuitOnLastWindowClosed(quit_on_close)
            if sys.platform == "win32" and getattr(sys, "frozen", False):
                ctypes.windll.kernel32.SetDllDirectoryW(sys._MEIPASS)

    def _on_failed(self, message, visible):
        self.busy = False
        if self.closing:
            return
        if not message:
            if self.dialog is not None:
                self.dialog.set_state(
                    "available",
                    self.artifact["version"] if self.artifact else "",
                )
            return
        logger.warning(f"Update failed: {message}")
        if visible:
            if "SSL" in message or "CERTIFICATE_VERIFY_FAILED" in message:
                message = self.tr(
                    "Couldn't verify the update server. Check your network or proxy settings, then try again."
                )
            elif "ConnectionPool" in message or "ConnectionError" in message:
                message = self.tr(
                    "Couldn't connect to the update server. Check your connection and try again."
                )
            self._panel().set_state("error", message=message)
            self.show_panel()

    def _show_previous_result(self):
        path = self.cache / "result.json"
        if not path.exists():
            return
        try:
            result = json.loads(path.read_text())
            if result.get("status") == "success" and result.get("directory"):
                directory = Path(result["directory"])
                if (
                    directory.parent == self.cache
                    and directory.name.startswith("download-")
                ):
                    shutil.rmtree(directory, ignore_errors=True)
                path.unlink()
            if result.get("status") == "failed":
                self._on_failed(
                    self.tr(
                        "The update failed. Your previous version has been restored.\n\n"
                    )
                    + result["error"],
                    True,
                )
                path.unlink()
        except (OSError, ValueError, KeyError) as error:
            logger.warning(f"Unable to read update result: {error}")

    def prepare_for_close(self):
        self.closing = True
        self.cancel.set()


def acknowledge_update():
    path = os.environ.pop("X_ANYLABELING_UPDATE_READY_FILE", None)
    token = os.environ.pop("X_ANYLABELING_UPDATE_TOKEN", None)
    expected = os.environ.pop("X_ANYLABELING_UPDATE_VERSION", None)
    if path and token:
        context = installation()
        if (
            context is None
            or context["info"]["version"] != __version__
            or (expected is not None and expected != __version__)
        ):
            logger.error(
                "The installed update does not match the running application version."
            )
            return
        try:
            Path(path).write_text(token, encoding="utf-8")
        except OSError as error:
            logger.warning(f"Unable to acknowledge update startup: {error}")
