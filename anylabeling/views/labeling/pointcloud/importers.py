import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import PurePosixPath
from zipfile import ZipFile

from .cuboid import Cuboid
from .export import EXPORT_FORMATS, ExportCancelled
from .model import ClassDefinition


@dataclass(frozen=True)
class ImportedFrame:
    name: str | None
    index: int | None
    cuboids: tuple[Cuboid, ...]


class AnnotationArchive:
    def __init__(self, archive, cancelled):
        self.archive = archive
        self.cancelled = cancelled
        self.members = {}
        self.bytes_read = 0
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {name}")
            if info.is_dir():
                continue
            name = str(path)
            if name in self.members:
                raise ValueError(f"Duplicate archive member: {name}")
            self.members[name] = info

    def check_cancelled(self):
        if self.cancelled is not None and self.cancelled():
            raise ExportCancelled()

    def read(self, name):
        self.check_cancelled()
        info = self.members[name]
        self.bytes_read += info.file_size
        if info.file_size > 128 * 1024**2 or self.bytes_read > 512 * 1024**2:
            raise ValueError("Annotation data exceeds the import size limit.")
        return self.archive.read(info).decode("utf-8-sig")

    def json(self, name):
        return json.loads(self.read(name))

    def root(self, marker):
        matches = [
            name
            for name in self.members
            if name == marker or name.endswith("/" + marker)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {marker} in the selected archive.")
        return matches[0][: -len(marker)]


def _boolean(value):
    if value in (True, 1, "true", "True", "1"):
        return True
    if value in (False, 0, "false", "False", "0", None, ""):
        return False
    raise ValueError(f"Invalid boolean attribute: {value}")


def _classes(entries):
    classes = []
    names = set()
    for entry in entries:
        name = entry["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError(
                "Imported class names must be nonempty and unique."
            )
        color = entry.get("color", "#AAAAFF")
        if not isinstance(color, str) or not re.fullmatch(
            r"#[0-9a-fA-F]{6}", color
        ):
            color = "#AAAAFF"
        classes.append(ClassDefinition(len(classes) + 1, name, color))
        names.add(name)
    if len(classes) > 65535:
        raise ValueError("Too many imported classes.")
    return tuple(classes)


def _box(index, class_id, position, scale, rotation, attributes):
    return Cuboid(
        index + 1,
        class_id,
        position,
        scale,
        rotation,
        _boolean(attributes.get("occluded", False)),
        _boolean(attributes.get("locked", False)),
    )


def _datumaro(source):
    annotation_files = [
        name
        for name in source.members
        if name.endswith(".json")
        and PurePosixPath(name).parent.name == "annotations"
    ]
    if not annotation_files:
        raise ValueError("No Datumaro annotations/*.json files found.")
    classes = None
    frames = []
    for name in sorted(annotation_files):
        data = source.json(name)
        if (
            data.get("dm_format_version", "1.0") != "1.0"
            or data.get("media_type", 6) != 6
        ):
            raise ValueError("Expected a Datumaro 3D 1.0 point cloud dataset.")
        definitions = _classes(
            data.get("categories", {}).get("label", {}).get("labels", [])
        )
        if classes is not None and definitions != classes:
            raise ValueError(
                "Datumaro subsets must use the same class definitions."
            )
        classes = definitions
        for item in data["items"]:
            source.check_cancelled()
            if "point_cloud" not in item:
                raise ValueError("Datumaro items must reference point clouds.")
            boxes = []
            for annotation in item.get("annotations", []):
                if annotation["type"] != "cuboid_3d":
                    raise ValueError(
                        "Only cuboid_3d annotations can be imported."
                    )
                label = annotation["label_id"]
                if type(label) is not int or not 0 <= label < len(classes):
                    raise ValueError("Invalid Datumaro class reference.")
                boxes.append(
                    _box(
                        len(boxes),
                        label + 1,
                        annotation["position"],
                        annotation["scale"],
                        annotation.get("rotation", [0, 0, 0]),
                        annotation.get("attributes", {}),
                    )
                )
            frames.append(ImportedFrame(str(item["id"]), None, tuple(boxes)))
    return classes, tuple(frames)


def _kitti_names(source, root):
    mapping = {}
    name = root + "frame_list.txt"
    if name in source.members:
        for line in source.read(name).splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            index, frame = line.split(maxsplit=1)
            index = int(index)
            if index < 0 or index in mapping:
                raise ValueError("Invalid or duplicate KITTI frame index.")
            mapping[index] = frame.strip()
    return mapping


def _kitti_track(track, classes, frames, source):
    class_id = classes[track.findtext("objectType")]
    size = [float(track.findtext(axis)) for axis in "hwl"]
    start = int(track.findtext("first_frame"))
    poses = track.find("poses")
    entries = poses.findall("item")
    if start < 0 or int(poses.findtext("count")) != len(entries):
        raise ValueError("Invalid KITTI tracklet pose count or first frame.")
    previous_occluded = False
    for offset, pose in enumerate(entries):
        source.check_cancelled()
        boxes = frames.setdefault(start + offset, [])
        occlusion = int(pose.findtext("occlusion", "-1"))
        if occlusion not in (-1, 0, 1, 2):
            raise ValueError("Invalid KITTI occlusion value.")
        occluded = (
            previous_occluded if occlusion == -1 else occlusion in (1, 2)
        )
        if pose.findtext("occlusion_kf", "0") == "1":
            previous_occluded = occluded
        truncation = int(pose.findtext("truncation", "-1"))
        if truncation in (2, 99):
            continue
        if truncation not in (-1, 0, 1):
            raise ValueError("Invalid KITTI truncation value.")
        boxes.append(
            _box(
                len(boxes),
                class_id,
                [float(pose.findtext(axis)) for axis in ("tx", "ty", "tz")],
                size,
                [
                    float(pose.findtext(axis, "0"))
                    for axis in ("rx", "ry", "rz")
                ],
                {"occluded": occluded},
            )
        )


def _kitti_raw(source):
    root = source.root("tracklet_labels.xml")
    text = source.read(root + "tracklet_labels.xml")
    if "<!ENTITY" in text:
        raise ValueError("XML entity declarations are not supported.")
    tree = ET.fromstring(text)
    if tree.tag != "boost_serialization" or tree.find("tracklets") is None:
        raise ValueError("Expected KITTI Raw tracklet_labels.xml.")
    tracks = tree.findall("tracklets/item")
    if int(tree.findtext("tracklets/count")) != len(tracks):
        raise ValueError("Invalid KITTI tracklet count.")
    labels = []
    metadata = root + "dataset_meta.json"
    if metadata in source.members:
        labels = source.json(metadata).get("labels", [])
    for track in tracks:
        name = track.findtext("objectType")
        if name not in labels:
            labels.append(name)
    definitions = _classes([{"name": name} for name in labels])
    classes = {item.name: item.id for item in definitions}
    mapping = _kitti_names(source, root)
    frames = {index: [] for index in mapping}
    for track in tracks:
        _kitti_track(track, classes, frames, source)
    if mapping and frames.keys() - mapping.keys():
        raise ValueError(
            "KITTI tracklets reference frames absent from frame_list.txt."
        )
    return definitions, tuple(
        ImportedFrame(mapping.get(index), index, tuple(boxes))
        for index, boxes in sorted(frames.items())
    )


def _sly_pointcloud(source):
    root = source.root("meta.json")
    meta = source.json(root + "meta.json")
    if meta.get("projectType") != "point_clouds":
        raise ValueError("Expected a Supervisely point_clouds project.")
    definitions = _classes(
        [
            {"name": item["title"], "color": item.get("color")}
            for item in meta["classes"]
        ]
    )
    classes = {item.name: item.id for item in definitions}
    names = [
        name
        for name in source.members
        if name.startswith(root)
        and "/ann/" in name[len(root) :]
        and name.endswith(".pcd.json")
    ]
    if not names:
        raise ValueError("No Supervisely point cloud annotations found.")
    frames = []
    for name in sorted(names):
        data = source.json(name)
        objects = {}
        for item in data["objects"]:
            key = item["key"]
            if key in objects or item["classTitle"] not in classes:
                raise ValueError(
                    "Duplicate Supervisely object key or unknown class."
                )
            objects[key] = item
        boxes = []
        keys = set()
        for figure in data["figures"]:
            source.check_cancelled()
            if figure["geometryType"] != "cuboid_3d":
                raise ValueError("Only cuboid_3d figures can be imported.")
            key = figure["key"]
            if key in keys:
                raise ValueError("Duplicate Supervisely figure key.")
            keys.add(key)
            item = objects[figure["objectKey"]]
            attributes = {
                tag["name"]: tag.get("value") for tag in item.get("tags", [])
            }
            geometry = figure["geometry"]
            boxes.append(
                _box(
                    len(boxes),
                    classes[item["classTitle"]],
                    [geometry["position"][axis] for axis in "xyz"],
                    [geometry["dimensions"][axis] for axis in "xyz"],
                    [
                        geometry.get("rotation", {}).get(axis, 0)
                        for axis in "xyz"
                    ],
                    attributes,
                )
            )
        frame_name = name.split("/ann/", 1)[1][: -len(".pcd.json")]
        frames.append(ImportedFrame(frame_name, None, tuple(boxes)))
    return definitions, tuple(frames)


def read_dataset(path, format_name, cancelled=None):
    if format_name not in EXPORT_FORMATS:
        raise ValueError(f"Unknown import format: {format_name}")
    readers = {
        "datumaro": _datumaro,
        "kitti_raw": _kitti_raw,
        "sly_pointcloud": _sly_pointcloud,
    }
    try:
        with ZipFile(path) as archive:
            source = AnnotationArchive(archive, cancelled)
            classes, frames = readers[format_name](source)
            source.check_cancelled()
    except (
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        ET.ParseError,
    ) as error:
        raise ValueError(
            f"Invalid {EXPORT_FORMATS[format_name]} annotations: {error}"
        ) from error
    if not frames:
        raise ValueError(
            "No point cloud frames found in the annotation archive."
        )
    return classes, frames
