from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _default_log_dir() -> Path:
    base = Path(os.path.expanduser("~")).resolve()
    return base / ".xanylabeling" / "logs" / "pytest"


def _format_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def run_pytest(
    pytest_args: list[str],
    *,
    timeout_seconds: int,
    log_dir: Path,
) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pytest_{_format_ts()}.log"

    env = os.environ.copy()
    env.setdefault("PYTHONFAULTHANDLER", "1")

    cmd = [sys.executable, "-m", "pytest", *pytest_args]
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    ) as proc, open(log_path, "w", encoding="utf-8") as fp:
        start = time.monotonic()
        last_output = time.monotonic()

        def _write(line: str) -> None:
            nonlocal last_output
            last_output = time.monotonic()
            sys.stdout.write(line)
            fp.write(line)
            fp.flush()

        while True:
            if proc.poll() is not None:
                break

            now = time.monotonic()
            if now - start > timeout_seconds:
                _write(
                    f"\n[run_tests] Timeout after {timeout_seconds}s. "
                    "Requesting traceback dump (SIGUSR1) and terminating.\n"
                )
                try:
                    proc.send_signal(signal.SIGUSR1)
                except Exception:
                    pass
                time.sleep(2.0)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break

            try:
                line = proc.stdout.readline() if proc.stdout else ""
            except Exception:
                line = ""

            if line:
                _write(line)
            else:
                time.sleep(0.05)

        exit_code = proc.returncode if proc.returncode is not None else 1

    print(f"[run_tests] pytest log saved to: {log_path}")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="overall test session timeout in seconds",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="directory to write pytest output log",
    )
    parser.add_argument("pytest_args", nargs="*", help="arguments passed to pytest")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else _default_log_dir()
    return run_pytest(
        args.pytest_args,
        timeout_seconds=args.timeout,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())

