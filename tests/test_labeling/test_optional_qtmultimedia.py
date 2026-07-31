import os
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT_DIR = Path(__file__).resolve().parents[2]


def run_isolated_import(script):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )


def test_widgets_import_without_qtmultimedia():
    result = run_isolated_import("""
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "PyQt6.QtMultimedia":
                error = ModuleNotFoundError(f"No module named {name}")
                error.name = name
                raise error
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked_import

        from anylabeling.views.labeling import widgets

        assert widgets.VideoClassifierDialog is None
        """)

    assert result.returncode == 0, result.stderr


def test_unrelated_video_classifier_import_error_is_not_suppressed():
    result = run_isolated_import("""
        import builtins

        target = "anylabeling.views.labeling.video_classifier.player"
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == target:
                error = ModuleNotFoundError(f"No module named {name}")
                error.name = name
                raise error
            return real_import(name, *args, **kwargs)

        builtins.__import__ = blocked_import

        try:
            import anylabeling.views.labeling.widgets
        except ModuleNotFoundError as error:
            if error.name != target:
                raise
        else:
            raise AssertionError("unrelated import error was suppressed")
        """)

    assert result.returncode == 0, result.stderr
