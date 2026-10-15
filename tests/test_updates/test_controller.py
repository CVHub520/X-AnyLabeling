from pathlib import Path
import json
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication, QToolButton, QWidget
from PyQt6.QtCore import QTimer

from anylabeling.views.labeling.utils import update_controller


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(tmp_path, monkeypatch, app):
    helper = tmp_path / "installed/xanylabeling-updater"
    helper.parent.mkdir()
    helper.write_text("helper")
    context = {
        "target": str(helper.parent),
        "root": str(helper.parent),
        "helper": str(helper),
        "info": {},
    }
    monkeypatch.setattr(update_controller, "installation", lambda: context)
    monkeypatch.setattr(
        update_controller, "cache_directory", lambda context: tmp_path
    )
    messages = []
    state = {"quit_on_close": True, "quit": False, "accept_close": True}

    class Window(QWidget):
        def close(self):
            if state["accept_close"]:
                self.update_controller.prepare_for_close()
            return state["accept_close"]

    window = Window()
    window.labeling_widget = SimpleNamespace(
        view=SimpleNamespace(update_button=QToolButton(window))
    )
    window.app = SimpleNamespace(
        quitOnLastWindowClosed=lambda: state["quit_on_close"],
        setQuitOnLastWindowClosed=lambda value: state.update(
            quit_on_close=value
        ),
        quit=lambda: state.update(quit=True),
    )
    instance = update_controller.UpdateController(window)
    window.update_controller = instance
    instance.manifest = {"signed": "manifest"}
    download = tmp_path / "download"
    download.mkdir()
    archive = download / "update.zip"
    archive.write_text("verified archive")
    launches = []

    def start(arguments, **kwargs):
        launches.append(arguments)
        plan = json.loads(Path(arguments[1]).read_text())
        Path(plan["helper_ready"]).write_text(plan["token"])
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(update_controller.subprocess, "Popen", start)
    yield instance, state, archive, launches, messages
    window.deleteLater()


def test_verified_download_closes_and_arms_only_after_approval(controller):
    instance, state, archive, launches, messages = controller
    instance._panel().set_state("downloading")
    instance._on_downloaded(str(archive))
    assert len(launches) == 1
    assert (archive.parent / "armed").exists()
    assert not (archive.parent / "cancelled").exists()
    assert state["quit"] and state["quit_on_close"]
    assert not messages


def test_unsaved_work_refusal_does_not_arm_installer(controller):
    instance, state, archive, launches, messages = controller
    state["accept_close"] = False
    instance._on_downloaded(str(archive))
    assert len(launches) == 1
    assert not (archive.parent / "armed").exists()
    assert (archive.parent / "cancelled").exists()
    assert not state["quit"] and not instance.closing
    assert instance.dialog.mode in ("available", "error")


def test_cancelled_download_does_not_start_installer(controller):
    instance, state, archive, launches, messages = controller
    instance.cancel.set()
    instance._on_downloaded(str(archive))
    assert not launches and not state["quit"]
    assert not archive.parent.exists()


def test_missing_helper_leaves_application_running(controller):
    instance, state, archive, launches, messages = controller
    Path(instance.context["helper"]).unlink()
    instance._on_downloaded(str(archive))
    assert not launches and not state["quit"]
    assert not (archive.parent / "armed").exists()
    assert instance.dialog.mode in ("available", "error")


def test_installer_startup_failure_does_not_close_application(
    controller, monkeypatch
):
    instance, state, archive, launches, messages = controller
    monkeypatch.setattr(
        update_controller.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(poll=lambda: 1),
    )
    instance._on_downloaded(str(archive))
    assert not state["quit"] and not instance.closing
    assert not (archive.parent / "armed").exists()
    assert instance.dialog.mode in ("available", "error")


def test_delayed_installer_acknowledgement_keeps_qt_responsive(
    controller, monkeypatch
):
    instance, state, archive, launches, messages = controller

    def start(arguments, **kwargs):
        plan = json.loads(Path(arguments[1]).read_text())
        QTimer.singleShot(
            10, lambda: Path(plan["helper_ready"]).write_text(plan["token"])
        )
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(update_controller.subprocess, "Popen", start)
    instance._on_downloaded(str(archive))
    assert state["quit"] and not messages


def test_startup_acknowledgement_requires_matching_application_version(
    tmp_path, monkeypatch
):
    path = tmp_path / "ready"
    monkeypatch.setenv("X_ANYLABELING_UPDATE_READY_FILE", str(path))
    monkeypatch.setenv("X_ANYLABELING_UPDATE_TOKEN", "token")
    monkeypatch.setattr(
        update_controller,
        "installation",
        lambda: {"info": {"version": "0.0.0"}},
    )
    update_controller.acknowledge_update()
    assert not path.exists()
    monkeypatch.setenv("X_ANYLABELING_UPDATE_READY_FILE", str(path))
    monkeypatch.setenv("X_ANYLABELING_UPDATE_TOKEN", "token")
    monkeypatch.setattr(
        update_controller,
        "installation",
        lambda: {"info": {"version": update_controller.__version__}},
    )
    update_controller.acknowledge_update()
    assert path.read_text() == "token"


def test_background_check_only_reveals_toolbar_indicator(controller):
    instance, state, archive, launches, messages = controller
    artifact = {"version": "9.0.0", "release_notes": "Release notes"}
    instance._on_checked({"payload": "signed"}, artifact, False)
    assert not instance.button.isHidden()
    assert instance.dialog is None
    assert not launches
    instance.button.click()
    assert instance.dialog.isVisible()
    assert instance.dialog.mode == "available"
    assert "Release notes" not in instance.dialog.description.text()


def test_hiding_download_panel_keeps_download_running(controller):
    instance, state, archive, launches, messages = controller
    instance.busy = True
    instance._panel().set_state("downloading")
    instance.show_panel()
    instance.dialog.close_button.click()
    assert not instance.dialog.isVisible()
    assert not instance.cancel.is_set()
    instance.button.click()
    assert instance.dialog.isVisible()
    assert instance.dialog.mode == "downloading"
    instance.dialog.secondary.click()
    assert instance.cancel.is_set()


def test_about_check_releases_modal_before_showing_error(
    controller, app, monkeypatch
):
    from anylabeling.views.labeling.widgets.about_dialog import AboutDialog

    instance, state, archive, launches, messages = controller
    instance.window.show()
    about = AboutDialog(instance.window)
    called = []

    def check(manual=False):
        called.append(manual)
        assert QApplication.activeModalWidget() is None
        instance._on_failed("SSLError: CERTIFICATE_VERIFY_FAILED", True)

    monkeypatch.setattr(instance, "check", check)
    QTimer.singleShot(10, about.check_for_updates)
    about.exec()
    app.processEvents()
    assert called == [True]
    assert instance.dialog.isVisible()
    assert "SSLError" not in instance.dialog.description.text()
    instance.dialog.close_button.click()
    assert not instance.dialog.isVisible()
    instance.show_panel()
    instance.dialog.secondary.click()
    assert not instance.dialog.isVisible()
    instance.window.hide()


def test_hidden_update_indicator_uses_visible_header_anchor(controller):
    instance, state, archive, launches, messages = controller
    instance.window.resize(900, 600)
    anchor = QToolButton(instance.window)
    anchor.move(820, 20)
    instance.anchor = anchor
    instance.button.hide()
    instance.window.show()
    anchor.show()
    instance._on_failed("SSLError: CERTIFICATE_VERIFY_FAILED", True)
    assert instance.dialog.x() > instance.window.x() + 400
    assert instance.dialog.y() < instance.window.y() + 150
    instance.dialog.close_button.click()
    instance.window.hide()


def test_about_has_changelog_and_update_entry(controller):
    from PyQt6.QtWidgets import QPushButton
    from anylabeling.views.labeling.widgets.about_dialog import AboutDialog

    instance, state, archive, launches, messages = controller
    about = AboutDialog(instance.window)
    labels = [button.text() for button in about.findChildren(QPushButton)]
    assert "Changelog" in labels
    assert "Check for Updates" in labels
    about.close()


def test_startup_acknowledgement_rejects_wrong_signed_target_version(
    tmp_path, monkeypatch
):
    path = tmp_path / "ready"
    monkeypatch.setenv("X_ANYLABELING_UPDATE_READY_FILE", str(path))
    monkeypatch.setenv("X_ANYLABELING_UPDATE_TOKEN", "token")
    monkeypatch.setenv("X_ANYLABELING_UPDATE_VERSION", "99.0.0")
    monkeypatch.setattr(
        update_controller,
        "installation",
        lambda: {"info": {"version": update_controller.__version__}},
    )
    update_controller.acknowledge_update()
    assert not path.exists()
