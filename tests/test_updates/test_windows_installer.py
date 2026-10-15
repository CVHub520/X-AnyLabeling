import os
from pathlib import Path
import subprocess
import uuid

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Requires Windows")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def installers(tmp_path_factory):
    compiler = os.environ.get("INNO_SETUP_COMPILER")
    if not compiler or not Path(compiler).is_file():
        pytest.skip("Set INNO_SETUP_COMPILER to Inno Setup 6.7.3+ ISCC.exe")
    directory = tmp_path_factory.mktemp("setup-rollback")
    script = (ROOT / "packaging/windows/installer.iss").read_text()
    identity = "X-AnyLabeling-Test-" + uuid.uuid4().hex
    script = script.replace("AppId=X-AnyLabeling", f"AppId={identity}")
    script = script.replace(
        "UsePreviousGroup=no", "UsePreviousGroup=no\nUsePreviousAppDir=no"
    )
    images = str(ROOT / "anylabeling/resources/images").replace("/", "\\")
    script = script.replace("..\\..\\anylabeling\\resources\\images", images)
    for constant in ("group", "userdesktop", "userprograms"):
        script = script.replace(
            "{" + constant + "}", "{app}\\shortcuts\\" + constant
        )
    outputs = {}
    for version, variant, failure in (
        ("4.1.0", "CPU", ""),
        ("4.1.1", "CPU", ""),
        ("4.1.0", "CUDA12", ""),
        ("4.1.0", "CUDA12", "error"),
        ("4.1.1", "CPU", "error"),
        ("4.1.1", "CPU", "cancel"),
        ("4.1.1", "CPU", "backup"),
    ):
        name = f"{version}-{variant}-{failure or 'success'}"
        source = directory / name / "source"
        runtime = source / version
        runtime.mkdir(parents=True)
        (source / "X-AnyLabeling.exe").write_bytes(name.encode())
        (runtime / "build-info.json").write_text(
            '{"schema": 1, "version": "'
            + version
            + '", "platform": "Windows"}'
        )
        (runtime / "payload.bin").write_bytes(name.encode() * 100)
        content = script
        if failure == "error":
            content = content.replace(
                "[Messages]",
                'Source: "{tmp}\\missing-payload.bin"; DestDir: "{app}"; '
                "Flags: external ignoreversion\n\n[Messages]",
            )
        elif failure == "backup":
            content = content.replace(
                "BackupInstalledPath('{#AppVersion}');",
                "RaiseException('Simulated backup failure');",
            )
        elif failure == "cancel":
            content = content.replace(
                "[Code]",
                "[Code]\nprocedure CancelButtonClick(CurPageID: Integer; "
                "var Cancel, Confirm: Boolean);\nbegin\n"
                "  Confirm := False;\nend;\n"
                "procedure FailInstallation;\nbegin\n"
                "  WizardForm.CancelButton.OnClick(WizardForm.CancelButton);\n"
                "end;\n",
            ).replace(
                "createallsubdirs solidbreak",
                "createallsubdirs solidbreak; AfterInstall: FailInstallation",
            )
        path = source.parent / "test.iss"
        path.write_text(content, encoding="utf-8-sig")
        subprocess.run(
            [
                compiler,
                "/Q",
                f"/DAppVersion={version}",
                f"/DVariant={variant}",
                f"/DSourceDir={source}",
                f"/DOutputDir={source.parent}",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        outputs[version, variant, failure] = next(
            source.parent.glob("*-Setup.exe")
        )
    yield outputs
    import winreg

    key = (
        "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\"
        + identity
        + "_is1"
    )
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)
    except FileNotFoundError:
        pass


def run_setup(installer, target):
    result = subprocess.run(
        [
            str(installer),
            (
                "/SILENT"
                if installer.parent.name.endswith("-cancel")
                else "/VERYSILENT"
            ),
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            f"/DIR={target}",
            f"/LOG={target.parent / (installer.parent.name + '.log')}",
        ],
        timeout=60,
    )
    return result.returncode


def snapshot(target):
    return {
        str(path.relative_to(target)): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file() and "shortcuts" not in path.parts
    }


@pytest.mark.parametrize(
    "version,variant,failure",
    [
        ("4.1.0", "CUDA12", "error"),
        ("4.1.1", "CPU", "error"),
        ("4.1.1", "CPU", "cancel"),
        ("4.1.1", "CPU", "backup"),
    ],
)
def test_failed_setup_restores_previous_installation(
    installers, tmp_path, version, variant, failure
):
    target = tmp_path / "installed app"
    assert run_setup(installers["4.1.0", "CPU", ""], target) == 0
    (target / "annotations.json").write_text("user data")
    previous = snapshot(target)
    assert run_setup(installers[version, variant, failure], target) != 0
    assert snapshot(target) == previous
    assert not (target / ".setup-backup").exists()


@pytest.mark.parametrize(
    "version,variant",
    [
        ("4.1.1", "CPU"),
        ("4.1.0", "CUDA12"),
    ],
)
def test_successful_setup_replaces_runtime_and_preserves_user_files(
    installers, tmp_path, version, variant
):
    target = tmp_path / "installed app"
    assert run_setup(installers["4.1.0", "CPU", ""], target) == 0
    (target / "4.1.0/obsolete.dll").write_text("obsolete dependency")
    (target / "annotations.json").write_text("user data")
    assert run_setup(installers[version, variant, ""], target) == 0
    assert (
        target / "X-AnyLabeling.exe"
    ).read_text() == f"{version}-{variant}-success"
    assert (target / "annotations.json").read_text() == "user data"
    assert (target / version / "payload.bin").is_file()
    assert not (target / "4.1.0/obsolete.dll").exists()
    assert not (target / ".setup-backup").exists()
    if version != "4.1.0":
        assert not (target / "4.1.0").exists()


def test_setup_preserves_backup_from_interrupted_install(installers, tmp_path):
    target = tmp_path / "installed app"
    backup = target / ".setup-backup"
    backup.mkdir(parents=True)
    (backup / "X-AnyLabeling.exe").write_text("recoverable application")
    previous = snapshot(target)
    assert run_setup(installers["4.1.1", "CPU", ""], target) != 0
    assert snapshot(target) == previous


@pytest.mark.parametrize("failure", ["", "error"])
def test_update_staging_preserves_installed_application_and_registration(
    installers, tmp_path, monkeypatch, failure
):
    import winreg

    monkeypatch.syspath_prepend(str(ROOT / "anylabeling/services/updates"))
    import installer

    target = tmp_path / "installed app"
    assert run_setup(installers["4.1.0", "CPU", ""], target) == 0
    previous = snapshot(target)
    uninstall = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"

    def registrations():
        values = {}
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, uninstall) as parent:
            for index in range(winreg.QueryInfoKey(parent)[0]):
                name = winreg.EnumKey(parent, index)
                if not name.startswith("X-AnyLabeling-Test-"):
                    continue
                with winreg.OpenKey(parent, name) as key:
                    values[name] = [
                        winreg.EnumValue(key, i)
                        for i in range(winreg.QueryInfoKey(key)[1])
                    ]
        return values

    before = registrations()
    stage = tmp_path / "staged update"
    stage.mkdir()
    setup = installers["4.1.1", "CPU", failure]
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            installer.stage_update(
                setup, stage, {"kind": "installer", "root": "X-AnyLabeling"}
            )
    else:
        staged = installer.stage_update(
            setup, stage, {"kind": "installer", "root": "X-AnyLabeling"}
        )
        assert (staged / "4.1.1/payload.bin").is_file()
        assert not list(staged.glob("unins*"))
        assert not (staged / "shortcuts").exists()
    assert snapshot(target) == previous
    assert registrations() == before
