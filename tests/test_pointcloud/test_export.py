import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from zipfile import ZipFile

import numpy as np
import pytest

from anylabeling.views.labeling.pointcloud.cuboid import Cuboid
from anylabeling.views.labeling.pointcloud.export import (
    EXPORT_FORMATS,
    ExportCancelled,
    ExportFrame,
    export_dataset,
)
from anylabeling.views.labeling.pointcloud.io import (
    cuboid_source_path,
    save_cuboids,
)
from anylabeling.views.labeling.pointcloud.model import ClassDefinition


@pytest.fixture
def dataset(tmp_path):
    classes = [
        ClassDefinition(0, "Unlabeled", "#808080"),
        ClassDefinition(10, "Vehicle", "#AAAAFF"),
    ]
    points = np.array([[1, 2, 3, 0.25], [4, 5, 6, 0.5]], dtype="<f4")
    frames = []
    for index in range(3):
        path = tmp_path / f"{index:06d}.bin"
        points.tofile(path)
        frames.append(ExportFrame(path, path.with_suffix(".cuboids.json")))
    box = Cuboid(1, 10, (5, 6, 7), (2, 3, 4), (0.1, 0.2, -0.3), True, True)
    save_cuboids(frames[0].cuboid_path, (box,), frames[0].path)
    save_cuboids(frames[1].cuboid_path, (box,), frames[1].path)
    frames[0] = replace(frames[0], cuboids=(replace(box, center=(8, 9, 10)),))
    return frames, classes


@pytest.mark.parametrize("format_name", EXPORT_FORMATS)
@pytest.mark.parametrize("save_images", (False, True))
def test_formats_preserve_boxes_empty_frames_and_optional_media(
    tmp_path, dataset, format_name, save_images
):
    frames, classes = dataset
    path = tmp_path / "export.zip"
    progress = []
    assert export_dataset(
        path,
        format_name,
        frames,
        classes,
        save_images,
        progress=lambda current, total: progress.append((current, total)),
    ) == (3, 2)
    assert progress == [(1, 3), (2, 3), (3, 3)]
    with ZipFile(path) as archive:
        clouds = [name for name in archive.namelist() if name.endswith(".pcd")]
        assert len(clouds) == (3 if save_images else 0)
        assert not any(name.endswith(".label") for name in archive.namelist())
        if save_images:
            header, data = archive.read(clouds[0]).split(b"DATA binary\n", 1)
            assert b"FIELDS x y z intensity" in header
            np.testing.assert_array_equal(
                np.frombuffer(data, dtype="<f4"),
                np.fromfile(frames[0].path, dtype="<f4"),
            )
        if format_name == "datumaro":
            data = json.loads(archive.read("annotations/default.json"))
            assert [item["id"] for item in data["items"]] == [
                frame.path.stem for frame in frames
            ]
            assert [len(item["annotations"]) for item in data["items"]] == [
                1,
                1,
                0,
            ]
            annotation = data["items"][0]["annotations"][0]
            assert annotation["position"] == [8, 9, 10]
            assert annotation["scale"] == [2, 3, 4]
            assert annotation["rotation"] == [0.1, 0.2, -0.3]
            assert annotation["attributes"] == {
                "occluded": True,
                "locked": True,
            }
            assert annotation["label_id"] == 0
        elif format_name == "kitti_raw":
            xml = ET.fromstring(archive.read("tracklet_labels.xml"))
            tracks = xml.findall("tracklets/item")
            assert len(tracks) == 2
            assert [track.findtext("first_frame") for track in tracks] == [
                "0",
                "1",
            ]
            assert [float(tracks[0].findtext(axis)) for axis in "hwl"] == [
                2,
                3,
                4,
            ]
            pose = tracks[0].find("poses/item")
            assert [
                float(pose.findtext(axis)) for axis in ("tx", "ty", "tz")
            ] == [8, 9, 10]
            assert [
                float(pose.findtext(axis)) for axis in ("rx", "ry", "rz")
            ] == [0.1, 0.2, -0.3]
            assert pose.findtext("occlusion") == "1"
            assert len(archive.read("frame_list.txt").splitlines()) == 3
            assert json.loads(archive.read("dataset_meta.json"))["labels"] == [
                "Vehicle"
            ]
        else:
            first = json.loads(archive.read("ds0/ann/000000.pcd.json"))
            geometry = first["figures"][0]["geometry"]
            assert geometry["position"] == dict(x=8, y=9, z=10)
            assert geometry["dimensions"] == dict(x=2, y=3, z=4)
            assert geometry["rotation"] == dict(x=0.1, y=0.2, z=-0.3)
            empty = json.loads(archive.read("ds0/ann/000002.pcd.json"))
            assert not empty["figures"] and not empty["objects"]
            mapping = json.loads(archive.read("key_id_map.json"))
            assert len(mapping["videos"]) == 3
            assert len(set(mapping["objects"].values())) == 2


@pytest.mark.parametrize("format_name", EXPORT_FORMATS)
def test_deleted_current_boxes_do_not_reappear(tmp_path, dataset, format_name):
    frames, classes = dataset
    frames[0] = replace(frames[0], cuboids=())
    assert export_dataset(
        tmp_path / "export.zip", format_name, frames, classes
    ) == (3, 1)


@pytest.mark.parametrize("format_name", EXPORT_FORMATS)
def test_images_and_ply_colors_are_preserved(tmp_path, format_name):
    from PIL import Image

    source = tmp_path / "cloud.ply"
    source.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n1 2 3 255 0 20\n4 5 6 0 255 40\n"
    )
    image = tmp_path / "cloud.png"
    Image.new("RGB", (16, 8), "red").save(image)
    frame = ExportFrame(source, source.with_suffix(".cuboids.json"), (), image)
    path = tmp_path / "export.zip"
    export_dataset(path, format_name, [frame], [], True)
    with ZipFile(path) as archive:
        cloud = next(
            name for name in archive.namelist() if name.endswith(".pcd")
        )
        header, binary = archive.read(cloud).split(b"DATA binary\n", 1)
        assert b"FIELDS x y z rgb" in header
        data = np.frombuffer(binary, dtype=[("xyz", "<f4", 3), ("rgb", "<u4")])
        np.testing.assert_array_equal(data["xyz"], [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(data["rgb"], [0xFF0014, 0x00FF28])
        media = next(
            name for name in archive.namelist() if name.endswith(".png")
        )
        assert archive.read(media) == image.read_bytes()


def test_cancellation_and_failure_preserve_existing_zip(tmp_path, dataset):
    frames, classes = dataset
    target = tmp_path / "export.zip"
    target.write_bytes(b"existing export")
    cancelled = False

    def progress(current, total):
        nonlocal cancelled
        cancelled = True

    with pytest.raises(ExportCancelled):
        export_dataset(
            target,
            "datumaro",
            frames,
            classes,
            True,
            progress=progress,
            cancelled=lambda: cancelled,
        )
    assert target.read_bytes() == b"existing export"
    assert not list(tmp_path.glob(".export.zip.*.tmp"))
    frames[1].path.write_bytes(b"bad input")
    with pytest.raises(ValueError, match="BIN"):
        export_dataset(target, "datumaro", frames, classes, True)
    assert target.read_bytes() == b"existing export"
    assert not list(tmp_path.glob(".export.zip.*.tmp"))


def test_cuboid_paths_match_loader_and_export_handles_unknown_classes(
    tmp_path, dataset
):
    frames, classes = dataset
    output = tmp_path / "output"
    output.mkdir()
    source = frames[0].path
    label = output / source.with_suffix(".label").name
    assert cuboid_source_path(source, label) == frames[0].cuboid_path
    preferred = label.with_suffix(".cuboids.json")
    save_cuboids(preferred, (), source)
    assert cuboid_source_path(source, label) == preferred
    path = tmp_path / "export.zip"
    export_dataset(path, "datumaro", frames, [])
    with ZipFile(path) as archive:
        labels = json.loads(archive.read("annotations/default.json"))[
            "categories"
        ]["label"]["labels"]
        assert labels[0]["name"] == "Class 10"


def test_duplicate_names_and_source_protection(tmp_path, dataset):
    frames, classes = dataset
    duplicate = tmp_path / "nested" / frames[0].path.name
    duplicate.parent.mkdir()
    duplicate.write_bytes(frames[0].path.read_bytes())
    frames.append(replace(frames[0], path=duplicate))
    target = tmp_path / "export.zip"
    export_dataset(target, "datumaro", frames, classes)
    with ZipFile(target) as archive:
        data = json.loads(archive.read("annotations/default.json"))
        assert len({item["id"] for item in data["items"]}) == 4
    target.unlink()
    target.symlink_to(frames[0].path)
    original = frames[0].path.read_bytes()
    with pytest.raises(ValueError, match="source file"):
        export_dataset(target, "datumaro", frames, classes)
    assert frames[0].path.read_bytes() == original


def test_invalid_annotations_do_not_create_partial_export(tmp_path, dataset):
    frames, classes = dataset
    frames[1].cuboid_path.write_text("{}")
    path = tmp_path / "export.zip"
    with pytest.raises(ValueError, match="cuboids"):
        export_dataset(path, "sly_pointcloud", frames, classes)
    assert not path.exists()
