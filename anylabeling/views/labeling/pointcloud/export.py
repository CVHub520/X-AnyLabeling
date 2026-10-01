import json
import os
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .cuboid import Cuboid
from .io import _check_unchanged, _read_ply, _signature, load_cuboids
from .model import ClassDefinition

EXPORT_FORMATS = {
    "datumaro": "Datumaro 3D",
    "kitti_raw": "Kitti Raw Format",
    "sly_pointcloud": "Sly Point Cloud Format",
}


@dataclass(frozen=True)
class ExportFrame:
    path: Path
    cuboid_path: Path
    cuboids: tuple[Cuboid, ...] | None = None
    image_path: Path | None = None


class ExportCancelled(Exception):
    pass


def _json(archive, path, value):
    archive.writestr(
        path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    )


def _write_pcd(archive, name, source, check_cancelled):
    check_cancelled()
    path_before = _signature(source.stat())
    with source.open("rb") as stream:
        before = _signature(os.fstat(stream.fileno()))
        if source.suffix.lower() == ".bin":
            if not before[2] or before[2] % 16:
                raise ValueError(f"{source}: invalid BIN point records.")
            points = np.fromfile(stream, dtype="<f4").reshape(-1, 4)
            if points.nbytes != before[2]:
                raise ValueError(f"{source}: incomplete BIN point records.")
            intensity, rgb = True, None
        elif source.suffix.lower() == ".ply":
            points, intensity, rgb = _read_ply(stream)
        else:
            raise ValueError(f"{source}: supported formats are BIN and PLY.")
        _check_unchanged(stream, source, before, path_before)
    fields = [(axis, "<f4") for axis in "xyz"]
    if intensity:
        fields.append(("intensity", "<f4"))
    if rgb is not None:
        fields.append(("rgb", "<u4"))
    header = "\n".join(
        (
            "VERSION .7",
            "FIELDS " + " ".join(key for key, _ in fields),
            "SIZE " + " ".join("4" for _ in fields),
            "TYPE "
            + " ".join("U" if key == "rgb" else "F" for key, _ in fields),
            "COUNT " + " ".join("1" for _ in fields),
            f"WIDTH {len(points)}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {len(points)}",
            "DATA binary\n",
        )
    )
    with archive.open(name, "w", force_zip64=True) as output:
        output.write(header.encode("ascii"))
        for offset in range(0, len(points), 65536):
            check_cancelled()
            block = points[offset : offset + 65536]
            if not np.isfinite(block[:, :3]).all():
                raise ValueError(
                    f"{source}: point coordinates must be finite."
                )
            records = np.empty(len(block), dtype=fields)
            for column, axis in enumerate("xyz"):
                records[axis] = block[:, column]
            if intensity:
                records["intensity"] = block[:, 3]
            if rgb is not None:
                colors = rgb[offset : offset + len(block)].astype(np.uint32)
                records["rgb"] = (
                    colors[:, 0] << 16 | colors[:, 1] << 8 | colors[:, 2]
                )
            output.write(records.tobytes())


def _write_image(archive, name, source, check_cancelled):
    path_before = _signature(source.stat())
    with (
        source.open("rb") as stream,
        archive.open(name, "w", force_zip64=True) as output,
    ):
        before = _signature(os.fstat(stream.fileno()))
        while block := stream.read(1024 * 1024):
            check_cancelled()
            output.write(block)
        _check_unchanged(stream, source, before, path_before)


def _datumaro(archive, entries, classes, check_cancelled):
    label_ids = {item.id: index for index, item in enumerate(classes)}
    items = []
    annotation_id = 0
    for index, (frame, name, cuboids) in enumerate(entries):
        check_cancelled()
        item = {
            "id": name,
            "attr": {"frame": index},
            "point_cloud": {"path": f"{name}.pcd"},
            "annotations": [],
        }
        if frame.image_path is not None:
            item["related_images"] = [
                {
                    "path": f"{name}/extra_image_0{frame.image_path.suffix.lower()}"
                }
            ]
        for cuboid in cuboids:
            annotation_id += 1
            item["annotations"].append(
                {
                    "id": annotation_id,
                    "type": "cuboid_3d",
                    "attributes": {
                        "occluded": cuboid.occluded,
                        "locked": cuboid.locked,
                    },
                    "group": 0,
                    "label_id": label_ids[cuboid.class_id],
                    "position": cuboid.center,
                    "rotation": cuboid.rotation,
                    "scale": cuboid.size,
                }
            )
        items.append(item)
    _json(
        archive,
        "annotations/default.json",
        {
            "dm_format_version": "1.0",
            "media_type": 6,
            "infos": {},
            "categories": {
                "label": {
                    "labels": [
                        {"name": item.name, "parent": "", "attributes": []}
                        for item in classes
                    ],
                    "label_groups": [],
                    "attributes": ["occluded", "locked"],
                }
            },
            "items": items,
        },
    )


def _kitti_raw(archive, entries, classes, check_cancelled):
    names = {item.id: item.name for item in classes}
    root = ET.Element(
        "boost_serialization", version="9", signature="serialization::archive"
    )
    tracklets = ET.SubElement(
        root, "tracklets", version="0", tracking_level="0", class_id="0"
    )
    ET.SubElement(tracklets, "count").text = str(
        sum(len(boxes) for _, _, boxes in entries)
    )
    ET.SubElement(tracklets, "item_version").text = "1"
    first = True
    for index, (_, _, cuboids) in enumerate(entries):
        check_cancelled()
        for cuboid in cuboids:
            track = ET.SubElement(tracklets, "item")
            if first:
                track.attrib.update(
                    version="1", tracking_level="0", class_id="1"
                )
            ET.SubElement(track, "objectType").text = names[cuboid.class_id]
            for axis, value in zip("hwl", cuboid.size):
                ET.SubElement(track, axis).text = str(value)
            ET.SubElement(track, "first_frame").text = str(index)
            poses = ET.SubElement(track, "poses")
            if first:
                poses.attrib.update(
                    version="0", tracking_level="0", class_id="2"
                )
            ET.SubElement(poses, "count").text = "1"
            ET.SubElement(poses, "item_version").text = "0"
            pose = ET.SubElement(poses, "item")
            if first:
                pose.attrib.update(
                    version="1", tracking_level="0", class_id="3"
                )
            values = dict(
                zip(
                    ("tx", "ty", "tz", "rx", "ry", "rz"),
                    cuboid.center + cuboid.rotation,
                )
            )
            values.update(
                state=2,
                occlusion=int(cuboid.occluded),
                occlusion_kf=0,
                truncation=0,
                amt_occlusion=-1,
                amt_border_l=-1,
                amt_border_r=-1,
                amt_occlusion_kf=-1,
                amt_border_kf=-1,
            )
            for key, value in values.items():
                ET.SubElement(pose, key).text = str(value)
            ET.SubElement(track, "finished").text = "1"
            first = False
    ET.indent(root)
    archive.writestr(
        "tracklet_labels.xml",
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        b"<!DOCTYPE boost_serialization>\n"
        + ET.tostring(root, encoding="utf-8"),
    )
    archive.writestr(
        "frame_list.txt",
        "".join(
            f"{index} {name}\n" for index, (_, name, _) in enumerate(entries)
        ),
    )
    _json(
        archive,
        "dataset_meta.json",
        {"labels": [item.name for item in classes]},
    )


def _sly_pointcloud(archive, entries, classes, check_cancelled):
    names = {item.id: item.name for item in classes}
    _json(
        archive,
        "meta.json",
        {
            "classes": [
                {
                    "id": index + 1,
                    "title": item.name,
                    "color": item.color,
                    "shape": "cuboid_3d",
                    "geometry_config": {},
                }
                for index, item in enumerate(classes)
            ],
            "tags": [
                {
                    "id": index + 1,
                    "name": name,
                    "value_type": "any_string",
                    "color": "",
                    "hotkey": "",
                    "applicable_type": "objectsOnly",
                    "classes": [],
                }
                for index, name in enumerate(("occluded", "locked"))
            ],
            "projectType": "point_clouds",
        },
    )
    key_map = {key: {} for key in ("tags", "objects", "figures", "videos")}
    for index, (_, name, cuboids) in enumerate(entries):
        check_cancelled()
        frame_key = uuid.uuid4().hex
        key_map["videos"][frame_key] = index
        annotation = {
            "description": "",
            "key": frame_key,
            "tags": [],
            "objects": [],
            "figures": [],
        }
        for cuboid in cuboids:
            object_key, figure_key = uuid.uuid4().hex, uuid.uuid4().hex
            object_id = len(key_map["objects"]) + 1
            key_map["objects"][object_key] = object_id
            key_map["figures"][figure_key] = object_id
            annotation["objects"].append(
                {
                    "key": object_key,
                    "classTitle": names[cuboid.class_id],
                    "tags": [
                        {
                            "key": uuid.uuid4().hex,
                            "name": key,
                            "value": str(getattr(cuboid, key)).lower(),
                        }
                        for key in ("occluded", "locked")
                    ],
                }
            )
            annotation["figures"].append(
                {
                    "key": figure_key,
                    "objectKey": object_key,
                    "geometryType": "cuboid_3d",
                    "geometry": {
                        key: dict(zip("xyz", value))
                        for key, value in (
                            ("position", cuboid.center),
                            ("rotation", cuboid.rotation),
                            ("dimensions", cuboid.size),
                        )
                    },
                }
            )
        _json(archive, f"ds0/ann/{name}.pcd.json", annotation)
    _json(archive, "key_id_map.json", key_map)


def _export_entries(path, frames, classes, check_cancelled):
    entries = []
    definitions = {item.id: item for item in classes if item.id != 0}
    for frame, name in zip(frames, export_frame_names(frames)):
        check_cancelled()
        for source in (frame.path, frame.cuboid_path, frame.image_path):
            if source is not None and (
                path.resolve() == source.resolve()
                or (
                    path.exists() and source.exists() and path.samefile(source)
                )
            ):
                raise ValueError(
                    "The export must not overwrite a source file."
                )
        cuboids = frame.cuboids
        if cuboids is None:
            cuboids = (
                load_cuboids(frame.cuboid_path, frame.path)
                if frame.cuboid_path.exists() or frame.cuboid_path.is_symlink()
                else ()
            )
        for cuboid in cuboids:
            definitions.setdefault(
                cuboid.class_id,
                ClassDefinition(
                    cuboid.class_id, f"Class {cuboid.class_id}", "#808080"
                ),
            )
        entries.append((frame, name, cuboids))
    classes = list(definitions.values())
    if len({item.name for item in classes}) != len(classes):
        raise ValueError("Class names must be unique for dataset export.")
    return entries, classes


def export_frame_names(frames):
    used_names = set()
    for index, frame in enumerate(frames):
        name = (
            frame.path.stem.replace("\\", "_")
            .replace("\n", "_")
            .replace("\r", "_")
            .strip()
        )
        if name in ("", ".", ".."):
            name = str(index)
        if name in used_names:
            name = f"{index}_{name}"
        while name in used_names:
            name = "_" + name
        used_names.add(name)
        yield name


def export_dataset(
    path,
    format_name,
    frames,
    classes,
    save_images=False,
    *,
    calibration=None,
    progress=None,
    cancelled=None,
):
    if format_name not in EXPORT_FORMATS:
        raise ValueError(f"Unknown export format: {format_name}")
    if not frames:
        raise ValueError("No point cloud frames to export.")
    path = Path(path)
    if path.suffix.lower() != ".zip":
        raise ValueError("Select a ZIP file for the export.")

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise ExportCancelled()

    entries, classes = _export_entries(path, frames, classes, check_cancelled)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=6) as archive:
            writers = {
                "datumaro": _datumaro,
                "kitti_raw": _kitti_raw,
                "sly_pointcloud": _sly_pointcloud,
            }
            writers[format_name](archive, entries, classes, check_cancelled)
            for index, (frame, name, _) in enumerate(entries):
                check_cancelled()
                if save_images:
                    cloud_name = {
                        "datumaro": f"point_clouds/default/{name}.pcd",
                        "kitti_raw": f"velodyne_points/data/{name}.pcd",
                        "sly_pointcloud": f"ds0/pointcloud/{name}.pcd",
                    }[format_name]
                    _write_pcd(
                        archive, cloud_name, frame.path, check_cancelled
                    )
                    if frame.image_path is not None:
                        suffix = frame.image_path.suffix.lower()
                        image_name = {
                            "datumaro": f"images/default/{name}/extra_image_0{suffix}",
                            "kitti_raw": f"image_00/data/{name}{suffix}",
                            "sly_pointcloud": f"ds0/related_images/{name}_pcd/extra_image_0{suffix}",
                        }[format_name]
                        _write_image(
                            archive,
                            image_name,
                            frame.image_path,
                            check_cancelled,
                        )
                        if format_name == "sly_pointcloud":
                            meta = {}
                            if calibration is not None:
                                meta["sensorsData"] = {
                                    "extrinsicMatrix": np.asarray(
                                        calibration["T_pointcloud_to_camera"]
                                    )[:3]
                                    .ravel()
                                    .tolist(),
                                    "intrinsicMatrix": np.asarray(
                                        calibration["camera_matrix"]
                                    )
                                    .ravel()
                                    .tolist(),
                                }
                            _json(
                                archive,
                                image_name + ".json",
                                {"name": Path(image_name).name, "meta": meta},
                            )
                if progress is not None:
                    progress(index + 1, len(entries))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        check_cancelled()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(entries), sum(len(boxes) for _, _, boxes in entries)
