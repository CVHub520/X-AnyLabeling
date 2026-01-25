import os
import subprocess
import sys
import zipfile
from pathlib import Path

from tests.fixtures.sample_data import create_sample_workspace


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str], *, cwd: Path, env: dict, timeout: int = 120):
    return subprocess.run(
        [sys.executable, "-m", "anylabeling.app", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_version_config_and_diagnostics(tmp_path):
    env = os.environ.copy()

    r = _run(["--work-dir", str(tmp_path), "version"], cwd=REPO_ROOT, env=env)
    assert r.returncode == 0
    assert "X-AnyLabeling" not in r.stdout
    assert r.stdout.strip()

    r = _run(["--work-dir", str(tmp_path), "config"], cwd=REPO_ROOT, env=env)
    assert r.returncode == 0
    assert r.stdout.strip() == str(tmp_path / ".xanylabelingrc")

    out_zip = tmp_path / "diagnostics.zip"
    r = _run(
        ["--work-dir", str(tmp_path), "diagnostics", "--output-zip", str(out_zip)],
        cwd=REPO_ROOT,
        env=env,
    )
    assert r.returncode == 0
    assert out_zip.exists()
    with zipfile.ZipFile(out_zip, "r") as zf:
        assert "system_info.json" in zf.namelist()


def test_cli_convert_xlabel_to_voc(tmp_path):
    ws = create_sample_workspace(tmp_path)
    env = os.environ.copy()

    out_dir = tmp_path / "voc"
    out_dir.mkdir()

    r = _run(
        [
            "--work-dir",
            str(tmp_path),
            "convert",
            "--task",
            "xlabel2voc",
            "--mode",
            "detect",
            "--images",
            str(ws["images_dir"]),
            "--labels",
            str(ws["labels_dir"]),
            "--output",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        timeout=120,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert (out_dir / "sample.xml").exists()
