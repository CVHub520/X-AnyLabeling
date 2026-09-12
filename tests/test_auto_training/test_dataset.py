from types import SimpleNamespace

import pytest

from anylabeling.services.auto_training.ultralytics import general
from anylabeling.services.auto_training.ultralytics._io import load_yaml_config
from anylabeling.services.auto_training.ultralytics.general import (
    create_yolo_dataset,
    is_prepared_dataset,
    resolve_prepared_dataset,
)
from anylabeling.services.auto_training.ultralytics.validators import (
    validate_data_file,
    validate_task_requirements,
)
from anylabeling.views.training.ultralytics_dialog import UltralyticsDialog


def test_windows_image_staging_prefers_symlink(monkeypatch):
    calls = []
    monkeypatch.setattr(
        general.os,
        "symlink",
        lambda source, destination: calls.append(
            ("symlink", source, destination)
        ),
    )
    monkeypatch.setattr(
        general.shutil,
        "copy2",
        lambda *_args: pytest.fail("image was copied"),
    )

    general._link_or_copy_image("source.jpg", "destination.jpg")

    assert calls == [("symlink", "source.jpg", "destination.jpg")]


def test_windows_image_staging_falls_back_to_copy(monkeypatch):
    calls = []

    def raise_symlink_error(_source, _destination):
        raise OSError

    monkeypatch.setattr(general.os, "symlink", raise_symlink_error)
    monkeypatch.setattr(
        general.shutil,
        "copy2",
        lambda source, destination: calls.append(
            ("copy", source, destination)
        ),
    )

    general._link_or_copy_image("source.jpg", "destination.jpg")

    assert calls == [("copy", "source.jpg", "destination.jpg")]


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-8", "gbk"])
def test_load_yaml_config_supports_dataset_encodings(tmp_path, encoding):
    data_path = tmp_path / f"data-{encoding}.yaml"
    data_path.write_bytes("names:\n  0: 中文标签\n".encode(encoding))

    assert load_yaml_config(str(data_path)) == {"names": {0: "中文标签"}}


def test_validate_data_file_reports_path_for_invalid_yaml(tmp_path):
    data_path = tmp_path / "data.yaml"
    data_path.write_text("", encoding="utf-8")

    is_valid, message = validate_data_file(str(data_path))

    assert not is_valid
    assert str(data_path) in message


def test_validate_data_file_accepts_list_names(tmp_path):
    data_path = tmp_path / "data.yaml"
    data_path.write_text("names: [cat, dog]\n", encoding="utf-8")

    assert validate_data_file(str(data_path)) == (True, ["cat", "dog"])


def test_dataset_preparation_reports_invalid_yaml_path(tmp_path):
    data_path = tmp_path / "data.yaml"
    data_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match=str(data_path)):
        create_yolo_dataset([], "Detect", 0.8, str(data_path))


def test_is_prepared_dataset_resolves_relative_split_directories(tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "images" / "val").mkdir(parents=True)
    data_path = tmp_path / "data.yaml"
    data_path.write_text(
        "path: dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "names: [cat]\n",
        encoding="utf-8",
    )

    assert is_prepared_dataset(str(data_path))

    (dataset_root / "images" / "val").rmdir()
    assert not is_prepared_dataset(str(data_path))


def test_data_yaml_without_dataset_root_uses_automatic_preparation(tmp_path):
    data_path = tmp_path / "data.yaml"
    data_path.write_text("names: [cat]\n", encoding="utf-8")

    assert not is_prepared_dataset(str(data_path))


def test_wsl_dataset_paths_are_normalized_for_windows(monkeypatch, tmp_path):
    data_file = "//wsl.localhost/Ubuntu-24.04/home/user/project/data.yaml"
    mapped_root = "//wsl.localhost/Ubuntu-24.04/home/user/datasets/coco"
    monkeypatch.setattr(
        general,
        "load_yaml_config",
        lambda _path: {
            "path": "/home/user/datasets/coco",
            "train": "images/train",
            "val": "images/val",
            "names": ["cat"],
        },
    )
    monkeypatch.setattr(
        general.os.path,
        "isdir",
        lambda path: path.replace("\\", "/")
        in {
            mapped_root,
            f"{mapped_root}/images/train",
            f"{mapped_root}/images/val",
        },
    )
    monkeypatch.setattr(general, "get_dataset_path", lambda: str(tmp_path))

    resolved_path, error = resolve_prepared_dataset(data_file)

    assert error == ""
    assert resolved_path is not None
    resolved_data = load_yaml_config(resolved_path)
    assert resolved_data["path"] == mapped_root
    assert resolved_data["train"] == "images/train"
    assert resolved_data["val"] == "images/val"


def test_wsl_dataset_reports_checked_mapped_path(monkeypatch):
    data_file = "//wsl.localhost/Ubuntu-24.04/home/user/project/data.yaml"
    monkeypatch.setattr(
        general,
        "load_yaml_config",
        lambda _path: {
            "path": "/home/user/missing",
            "train": "images/train",
            "val": "images/val",
        },
    )
    monkeypatch.setattr(general.os.path, "isdir", lambda _path: False)

    resolved_path, error = resolve_prepared_dataset(data_file)

    assert resolved_path is None
    assert "/home/user/missing" in error
    assert "//wsl.localhost/Ubuntu-24.04/home/user/missing" in error


def test_task_validation_allows_existing_dataset_flow_without_images(
    monkeypatch,
):
    assert validate_task_requirements("Detect", []) == (True, "")
    assert not validate_task_requirements(None, ["image.jpg"])[0]

    monkeypatch.setattr(
        "anylabeling.services.auto_training.ultralytics.validators.MIN_LABELED_IMAGES_THRESHOLD",
        20,
    )
    monkeypatch.setattr(
        "anylabeling.services.auto_training.ultralytics.validators.get_task_valid_images",
        lambda *_args: 19,
    )
    assert not validate_task_requirements("Detect", ["image.jpg"])[0]


def test_empty_dataset_summary_displays_existing_dataset_guidance():
    visibility = []
    dialog = SimpleNamespace(
        image_list=[],
        summary_table=SimpleNamespace(
            clear=lambda: visibility.append("cleared"),
            setVisible=lambda visible: visibility.append(("table", visible)),
        ),
        empty_dataset_hint=SimpleNamespace(
            setVisible=lambda visible: visibility.append(("hint", visible))
        ),
        _summary_view_mode="detect",
    )

    UltralyticsDialog.refresh_dataset_summary(dialog)

    assert visibility == ["cleared", ("table", False), ("hint", True)]
    assert dialog._summary_view_mode is None


def test_config_next_rejects_unprepared_yaml_without_loaded_images(
    monkeypatch, tmp_path
):
    data_path = tmp_path / "data.yaml"
    data_path.write_text("names: [cat]\n", encoding="utf-8")
    warnings = []
    dialog = SimpleNamespace(
        training_status="idle",
        get_current_config=lambda: {
            "basic": {"env": "", "data": str(data_path)}
        },
        configure_training_environment=lambda _value: None,
        environment_ready=True,
        environment_error=None,
        selected_task_type="Detect",
        _uses_existing_dataset=lambda path: is_prepared_dataset(path),
        image_list=[],
        names=[],
        tr=lambda text: text,
        append_training_log=lambda _message: None,
    )
    monkeypatch.setattr(
        "anylabeling.views.training.ultralytics_dialog.QMessageBox.warning",
        lambda _parent, _title, message: warnings.append(message),
    )

    UltralyticsDialog.start_training(dialog)

    assert len(warnings) == 1
    assert warnings[0].startswith(
        "When no images are loaded, the data file must reference existing train and val directories."
    )
    assert "Dataset YAML must define 'path'" in warnings[0]


def test_get_training_args_reuses_prepared_dataset(monkeypatch, tmp_path):
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "images" / "val").mkdir(parents=True)
    data_path = tmp_path / "data.yaml"
    data_path.write_text(
        f"path: {dataset_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names: [cat]\n",
        encoding="utf-8",
    )
    logs = []
    dialog = SimpleNamespace(
        selected_task_type="Detect",
        image_list=[],
        output_dir="",
        append_training_log=logs.append,
        _uses_existing_dataset=lambda path: is_prepared_dataset(path),
    )
    config = {
        "basic": {
            "data": str(data_path),
            "model": "yolo11n.pt",
            "project": str(tmp_path / "runs"),
            "name": "exp",
            "device": "cpu",
            "dataset_ratio": 0.8,
            "pose_config": "",
        },
        "checkpoint": {},
    }
    monkeypatch.setattr(
        "anylabeling.views.training.ultralytics_dialog.create_yolo_dataset",
        lambda *_args, **_kwargs: pytest.fail("dataset was copied"),
    )

    train_args = UltralyticsDialog.get_training_args(dialog, config)

    assert train_args["data"] == str(data_path)
    assert not any("Using existing dataset" in log for log in logs)


def test_get_training_args_uses_resolved_wsl_dataset(monkeypatch):
    dialog = SimpleNamespace(
        selected_task_type="Detect",
        append_training_log=lambda _message: None,
        _uses_existing_dataset=lambda _path: True,
    )
    config = {
        "basic": {
            "data": "//wsl.localhost/Ubuntu-24.04/data.yaml",
            "model": "yolo11n.pt",
            "project": "/runs",
            "name": "exp",
            "device": "cpu",
        },
        "checkpoint": {},
    }
    monkeypatch.setattr(
        "anylabeling.views.training.ultralytics_dialog.resolve_prepared_dataset",
        lambda _path: ("C:/resolved/data.yaml", ""),
    )

    train_args = UltralyticsDialog.get_training_args(dialog, config)

    assert train_args["data"] == "C:/resolved/data.yaml"


@pytest.mark.parametrize(
    ("existing_dataset", "expected_logs"),
    [
        (
            True,
            ["Using custom dataset: /data/data.yaml", "Preparing training..."],
        ),
        (
            False,
            [
                "Using workspace snapshot: 0 images",
                "Changes made during training will be used in the next run.",
                "Preparing training...",
            ],
        ),
    ],
)
def test_training_logs_data_source_before_preparation(
    existing_dataset, expected_logs, tmp_path
):
    logs = []
    dialog = SimpleNamespace(
        environment_ready=True,
        environment_error=None,
        image_list=[],
        get_current_config=lambda: {
            "basic": {
                "project": str(tmp_path),
                "name": "exp",
                "data": "/data/data.yaml",
            }
        },
        _uses_existing_dataset=lambda _path: existing_dataset,
        append_training_log=logs.append,
        get_training_args=lambda _config: {},
        training_manager=SimpleNamespace(
            start_training=lambda _args: (True, "")
        ),
        tr=lambda text: text,
    )

    UltralyticsDialog.start_training_from_train_tab(dialog)

    assert logs[: len(expected_logs)] == expected_logs
