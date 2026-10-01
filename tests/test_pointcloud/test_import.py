import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest

from anylabeling.views.labeling.pointcloud import import_dataset as module
from anylabeling.views.labeling.pointcloud.cuboid import Cuboid
from anylabeling.views.labeling.pointcloud.export import (
    EXPORT_FORMATS,
    ExportCancelled,
    export_dataset,
)
from anylabeling.views.labeling.pointcloud.import_dataset import (
    ImportTarget,
    apply_import,
    prepare_import,
)
from anylabeling.views.labeling.pointcloud.importers import read_dataset
from anylabeling.views.labeling.pointcloud.io import (
    load_classes,
    load_cuboids,
    save_classes,
    save_cuboids,
)
from anylabeling.views.labeling.pointcloud.model import ClassDefinition

from . import test_export

dataset = test_export.dataset


def targets_for(frames):
    return [
        ImportTarget(
            frame.path, frame.path.with_suffix(".label"), frame.cuboid_path
        )
        for frame in frames
    ]


@pytest.mark.parametrize("format_name", EXPORT_FORMATS)
@pytest.mark.parametrize("media", (False, True))
@pytest.mark.parametrize("wrapped", (False, True))
def test_export_import_roundtrip(
    tmp_path, dataset, format_name, media, wrapped
):
    frames, classes = dataset
    archive = tmp_path / "annotations.zip"
    export_dataset(archive, format_name, frames, classes, media)
    if wrapped:
        wrapped_path = tmp_path / "wrapped.zip"
        with (
            ZipFile(archive) as source,
            ZipFile(wrapped_path, "w") as destination,
        ):
            for name in source.namelist():
                destination.writestr("dataset/" + name, source.read(name))
        archive = wrapped_path
    current = [classes[0], replace(classes[1], id=25)]
    config = tmp_path / "classes.json"
    save_classes(config, {"detection": current[1:], "segmentation": current})
    targets = targets_for(frames)
    for target in targets:
        np.array([10, (5 << 16) | 25], dtype="<u4").tofile(target.label_path)
    labels_before = [target.label_path.read_bytes() for target in targets]
    plan = prepare_import(archive, format_name, targets, current, config)
    assert plan.object_count == 2
    assert plan.new_classes == 0
    assert plan.previous_objects == 2
    apply_import(plan)
    for index, target in enumerate(plan.targets):
        boxes = load_cuboids(target.cuboid_path, target.path)
        assert boxes == target.cuboids
        if index == 2:
            assert boxes == ()
            continue
        assert boxes[0].center == ((8, 9, 10) if index == 0 else (5, 6, 7))
        assert boxes[0].size == (2, 3, 4)
        assert boxes[0].rotation == (0.1, 0.2, -0.3)
        assert boxes[0].class_id == 25
        assert boxes[0].occluded
        assert boxes[0].locked == (format_name != "kitti_raw")
    assert [
        target.label_path.read_bytes() for target in targets
    ] == labels_before
    assert load_classes(config) == {
        "detection": current[1:],
        "segmentation": current,
    }


@pytest.mark.parametrize("format_name", EXPORT_FORMATS)
def test_import_without_class_file_preserves_existing_config(
    tmp_path, dataset, format_name
):
    frames, classes = dataset
    archive = tmp_path / "annotations.zip"
    export_dataset(archive, format_name, frames, classes)
    config = tmp_path / "pointcloud_classes.json"
    config.write_bytes(b"Existing user configuration")
    plan = prepare_import(
        archive, format_name, targets_for(frames), classes, None
    )
    assert all(change.path != config for change in plan.changes)
    apply_import(plan)
    assert config.read_bytes() == b"Existing user configuration"
    for target in plan.targets:
        assert load_cuboids(target.cuboid_path, target.path) == target.cuboids


def test_new_classes_do_not_reuse_reserved_detection_ids(tmp_path, dataset):
    frames, classes = dataset
    archive = tmp_path / "annotations.zip"
    export_dataset(archive, "datumaro", frames, classes)
    targets = targets_for(frames)
    np.array([40, 100], dtype="<u4").tofile(targets[2].label_path)
    config = tmp_path / "classes.json"
    plan = prepare_import(
        archive, "datumaro", targets, classes[:1], config, reserved_ids=(200,)
    )
    assert plan.new_classes == 1
    assert plan.classes[0].id == 201
    assert plan.targets[0].cuboids[0].class_id == 201
    assert not config.exists()


@pytest.mark.parametrize("format_name", EXPORT_FORMATS)
def test_import_keeps_segmentation_names_and_ids_independent(
    tmp_path, dataset, format_name
):
    frames, classes = dataset
    archive = tmp_path / "annotations.zip"
    export_dataset(archive, format_name, frames, classes)
    targets = targets_for(frames)
    segmentation = [
        classes[0],
        replace(classes[1], id=11),
        ClassDefinition(12, "Road", "#AABBCC"),
    ]
    config = tmp_path / "classes.json"
    save_classes(config, {"detection": [], "segmentation": segmentation})
    np.array([1000, 2000], dtype="<u4").tofile(targets[0].label_path)
    before = targets[0].label_path.read_bytes()
    plan = prepare_import(archive, format_name, targets, [], config)
    assert plan.new_classes == 1
    assert plan.classes == (replace(classes[1], id=11),)
    apply_import(plan)
    assert load_classes(config) == {
        "detection": list(plan.classes),
        "segmentation": segmentation,
    }
    assert targets[0].label_path.read_bytes() == before


def test_subset_empty_frames_replace_only_matching_targets(tmp_path, dataset):
    frames, classes = dataset
    archive = tmp_path / "annotations.zip"
    export_dataset(
        archive, "sly_pointcloud", [replace(frames[0], cuboids=())], classes
    )
    targets = targets_for(frames)
    before = targets[1].cuboid_path.read_bytes()
    plan = prepare_import(
        archive, "sly_pointcloud", targets, classes, tmp_path / "classes.json"
    )
    assert len(plan.targets) == 1 and plan.object_count == 0
    apply_import(plan)
    assert load_cuboids(targets[0].cuboid_path, targets[0].path) == ()
    assert targets[1].cuboid_path.read_bytes() == before
    assert not targets[2].cuboid_path.exists()


@pytest.mark.parametrize("cancel", (False, True))
def test_failed_or_cancelled_import_rolls_back_all_files(
    tmp_path, dataset, cancel
):
    frames, classes = dataset
    archive = tmp_path / "annotations.zip"
    export_dataset(archive, "datumaro", frames, classes)
    targets = targets_for(frames)
    config = tmp_path / "classes.json"
    save_classes(config, {"detection": classes[1:], "segmentation": classes})
    plan = prepare_import(archive, "datumaro", targets, classes, config)
    before = {change.path: change.before for change in plan.changes}
    cancelled = False
    original_replace = module.os.replace

    def replace_file(source, destination):
        if (
            Path(source).suffix == ".import"
            and Path(destination) == config
            and not cancel
        ):
            raise OSError("Disk error")
        return original_replace(source, destination)

    def progress(current, total):
        nonlocal cancelled
        if current == total:
            cancelled = cancel

    with patch.object(module.os, "replace", replace_file):
        with pytest.raises(ExportCancelled if cancel else OSError):
            apply_import(plan, progress=progress, cancelled=lambda: cancelled)
    for path, contents in before.items():
        assert (path.read_bytes() if path.exists() else None) == contents
    assert not list(tmp_path.glob(".*.import"))
    assert not list(tmp_path.glob(".*.backup"))


def test_changed_destination_aborts_before_replacing_files(tmp_path, dataset):
    frames, classes = dataset
    archive = tmp_path / "annotations.zip"
    export_dataset(archive, "datumaro", frames, classes)
    plan = prepare_import(
        archive,
        "datumaro",
        targets_for(frames),
        classes,
        tmp_path / "classes.json",
    )
    changed = plan.changes[1].path
    changed.write_bytes(b"external update")
    with pytest.raises(ValueError, match="changed"):
        apply_import(plan)
    assert plan.changes[0].path.read_bytes() == plan.changes[0].before
    assert changed.read_bytes() == b"external update"


@pytest.mark.parametrize(
    "invalid", ("frame", "duplicate", "size", "class", "shape", "nan")
)
def test_invalid_datumaro_archive_does_not_write(tmp_path, dataset, invalid):
    frames, classes = dataset
    archive = tmp_path / "valid.zip"
    export_dataset(archive, "datumaro", frames, classes)
    with ZipFile(archive) as source:
        data = json.loads(source.read("annotations/default.json"))
    annotation = data["items"][0]["annotations"][0]
    if invalid == "frame":
        data["items"][0]["id"] = "unknown"
    elif invalid == "duplicate":
        data["items"][1]["id"] = data["items"][0]["id"]
    elif invalid == "size":
        annotation["scale"] = [1, -1, 2]
    elif invalid == "class":
        annotation["label_id"] = 20
    elif invalid == "shape":
        annotation["type"] = "polygon"
    else:
        annotation["position"][0] = float("nan")
    bad = tmp_path / "bad.zip"
    with ZipFile(bad, "w") as destination:
        destination.writestr("annotations/default.json", json.dumps(data))
    config = tmp_path / "classes.json"
    original = frames[0].cuboid_path.read_bytes()
    with pytest.raises(ValueError):
        prepare_import(bad, "datumaro", targets_for(frames), classes, config)
    assert frames[0].cuboid_path.read_bytes() == original
    assert not config.exists()


def test_kitti_tracks_keep_original_frame_indices(tmp_path, dataset):
    import xml.etree.ElementTree as ET

    frames, classes = dataset
    archive = tmp_path / "valid.zip"
    export_dataset(archive, "kitti_raw", frames[:1], classes)
    with ZipFile(archive) as source:
        tree = ET.fromstring(source.read("tracklet_labels.xml"))
    track = tree.find("tracklets/item")
    poses = track.find("poses")
    pose = poses.find("item")
    poses.find("count").text = "3"
    for _ in range(2):
        poses.append(ET.fromstring(ET.tostring(pose)))
    poses.findall("item")[1].find("truncation").text = "2"
    poses.findall("item")[2].find("tx").text = "123"
    custom = tmp_path / "tracks.zip"
    with ZipFile(custom, "w") as destination:
        destination.writestr("tracklet_labels.xml", ET.tostring(tree))
    plan = prepare_import(
        custom,
        "kitti_raw",
        targets_for(frames),
        classes,
        tmp_path / "classes.json",
    )
    assert [len(target.cuboids) for target in plan.targets] == [1, 0, 1]
    assert plan.targets[2].cuboids[0].center[0] == 123


def test_duplicate_source_stems_roundtrip_and_reject_ambiguous_subset(
    tmp_path, dataset
):
    frames, classes = dataset
    nested = tmp_path / "nested" / frames[0].path.name
    nested.parent.mkdir()
    nested.write_bytes(frames[0].path.read_bytes())
    frames.append(
        replace(
            frames[0],
            path=nested,
            cuboid_path=nested.with_suffix(".cuboids.json"),
        )
    )
    archive = tmp_path / "export.zip"
    export_dataset(archive, "datumaro", frames, classes)
    plan = prepare_import(
        archive,
        "datumaro",
        targets_for(frames),
        classes,
        tmp_path / "classes.json",
    )
    assert len(plan.targets) == 4
    subset = tmp_path / "subset.zip"
    export_dataset(subset, "datumaro", frames[:1], classes)
    with pytest.raises(ValueError, match="uniquely match"):
        prepare_import(
            subset,
            "datumaro",
            targets_for(frames),
            classes,
            tmp_path / "classes.json",
        )


def test_unsafe_archive_and_xml_entities_are_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as destination:
        destination.writestr("../meta.json", "{}")
    with pytest.raises(ValueError, match="Unsafe"):
        read_dataset(archive, "sly_pointcloud")
    with ZipFile(archive, "w") as destination:
        destination.writestr(
            "tracklet_labels.xml", '<!DOCTYPE x [<!ENTITY a "bad">]><x>&a;</x>'
        )
    with pytest.raises(ValueError, match="entity"):
        read_dataset(archive, "kitti_raw")
