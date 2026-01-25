import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

import pytest

from anylabeling.qt_compat import apply_qt5_compat

apply_qt5_compat()


@pytest.fixture
def xanylabeling_workdir(tmp_path):
    import anylabeling.config as config_module
    from anylabeling.config import get_default_config, set_work_directory

    set_work_directory(str(tmp_path))
    config_module.current_config_file = str(tmp_path / ".xanylabelingrc")
    return get_default_config()


@pytest.fixture(autouse=True)
def _no_blocking_message_boxes(monkeypatch):
    from PyQt6 import QtWidgets

    mb = QtWidgets.QMessageBox
    ok = getattr(mb, "Ok", mb.StandardButton.Ok)
    discard = getattr(mb, "Discard", mb.StandardButton.Discard)

    monkeypatch.setattr(
        mb, "question", staticmethod(lambda *args, **kwargs: discard), raising=False
    )
    monkeypatch.setattr(
        mb, "critical", staticmethod(lambda *args, **kwargs: ok), raising=False
    )
    monkeypatch.setattr(
        mb, "warning", staticmethod(lambda *args, **kwargs: ok), raising=False
    )
    monkeypatch.setattr(
        mb, "information", staticmethod(lambda *args, **kwargs: ok), raising=False
    )
    return None


@pytest.fixture(autouse=True)
def _qt_cleanup(qapp):
    yield
    try:
        from PyQt6.QtWidgets import QApplication

        for w in QApplication.topLevelWidgets():
            try:
                view = getattr(getattr(w, "labeling_widget", None), "view", None)
                if view is not None and hasattr(view, "async_exif_scanner"):
                    view.async_exif_scanner.stop_scan()
                if view is not None and hasattr(view, "dirty"):
                    view.dirty = False
            except Exception:
                pass
            try:
                w.close()
            except Exception:
                pass
        qapp.processEvents()
    except Exception:
        pass
