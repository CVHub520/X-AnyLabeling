from __future__ import annotations

import json
import os
import re
import sys
import threading
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PyQt6 import QtCore

from anylabeling.config import get_work_directory
from anylabeling.views.labeling.utils.general import collect_system_info


def get_log_dir(work_dir: str | None = None) -> Path:
    base_dir = Path(work_dir or get_work_directory()).expanduser().resolve()
    return base_dir / ".xanylabeling" / "logs"


def ensure_log_dir(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _iter_log_files(log_dir: Path) -> Iterable[Path]:
    if not log_dir.exists():
        return []
    candidates = list(log_dir.glob("*.log*")) + list(log_dir.glob("*.txt*"))
    candidates = [p for p in candidates if p.is_file()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def _redact_text(text: str) -> str:
    patterns: list[tuple[re.Pattern[str], str]] = [
        (
            re.compile(r"(?i)(authorization\s*:\s*bearer\s+)([^\s]+)"),
            r"\1***",
        ),
        (
            re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([^\s'\"]+)"),
            r"\1***",
        ),
        (
            re.compile(r"(?i)(openai[_-]?api[_-]?key\s*[:=]\s*)([^\s'\"]+)"),
            r"\1***",
        ),
        (re.compile(r"sk-[A-Za-z0-9]{16,}"), "sk-***"),
    ]
    redacted = text
    for pat, repl in patterns:
        redacted = pat.sub(repl, redacted)
    return redacted


def install_qt_message_handler(py_logger) -> None:
    def handler(msg_type, context, message):
        try:
            location = ""
            if context and getattr(context, "file", None):
                location = f"{context.file}:{context.line}"
            text = f"{location} {message}".strip()
            if msg_type == QtCore.QtMsgType.QtDebugMsg:
                py_logger.debug(text)
            elif msg_type == QtCore.QtMsgType.QtInfoMsg:
                py_logger.info(text)
            elif msg_type == QtCore.QtMsgType.QtWarningMsg:
                py_logger.warning(text)
            elif msg_type == QtCore.QtMsgType.QtCriticalMsg:
                py_logger.error(text)
            elif msg_type == QtCore.QtMsgType.QtFatalMsg:
                py_logger.critical(text)
            else:
                py_logger.info(text)
        except Exception:
            pass

    QtCore.qInstallMessageHandler(handler)


def install_exception_hooks(py_logger, *, log_dir: Path) -> None:
    log_dir = ensure_log_dir(log_dir)

    def _show_dialog(message: str) -> None:
        try:
            from PyQt6 import QtWidgets

            app = QtWidgets.QApplication.instance()
            if app is None:
                return
            QtWidgets.QMessageBox.critical(
                None,
                "X-AnyLabeling",
                message,
            )
        except Exception:
            return

    def excepthook(exc_type, exc_value, exc_tb):
        try:
            tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            py_logger.critical(tb)
            _show_dialog(
                "An unexpected error occurred.\n"
                f"Logs: {log_dir}\n\n"
                "Please export diagnostics and attach them to your issue report."
            )
        finally:
            sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = excepthook

    if hasattr(threading, "excepthook"):
        prev = threading.excepthook

        def thread_excepthook(args):
            try:
                tb = "".join(
                    traceback.format_exception(
                        args.exc_type, args.exc_value, args.exc_traceback
                    )
                )
                py_logger.critical(tb)
            finally:
                prev(args)

        threading.excepthook = thread_excepthook


def enable_faulthandler(*, log_dir: Path) -> None:
    try:
        import faulthandler

        log_dir = ensure_log_dir(log_dir)
        fp = open(log_dir / "faulthandler.log", "a", encoding="utf-8")
        faulthandler.enable(file=fp, all_threads=True)
    except Exception:
        return


def export_diagnostics_zip(
    output_zip: str | os.PathLike[str],
    *,
    log_dir: Path | None = None,
    config_file: str | None = None,
    max_log_files: int = 10,
) -> Path:
    log_dir = log_dir or get_log_dir()
    ensure_log_dir(log_dir)
    output_path = Path(output_zip).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    system_info, pkg_info = collect_system_info()
    diagnostics = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "system": system_info,
        "packages": pkg_info,
    }

    config_payload = None
    if config_file:
        config_path = Path(config_file).expanduser()
        if config_path.exists() and config_path.is_file():
            try:
                raw = config_path.read_text(encoding="utf-8")
            except Exception:
                raw = config_path.read_text(errors="ignore")
            config_payload = {
                "path": str(config_path.resolve()),
                "content": _redact_text(raw),
            }

    log_files = list(_iter_log_files(log_dir))[:max_log_files]

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("system_info.json", json.dumps(diagnostics, indent=2))
        if config_payload is not None:
            zf.writestr("config.json", json.dumps(config_payload, indent=2))
        for lf in log_files:
            try:
                try:
                    raw = lf.read_text(encoding="utf-8")
                except Exception:
                    raw = lf.read_text(errors="ignore")
                zf.writestr(f"logs/{lf.name}", _redact_text(raw))
            except Exception:
                continue

    return output_path
