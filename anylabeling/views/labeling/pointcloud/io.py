import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .model import ClassDefinition, Frame, _validate_id, validate_labels

_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _natural_key(path):
    parts = re.split(r"(\d+)", path.name.casefold())
    return (
        tuple(
            (1, int(part)) if part.isdigit() else (0, part) for part in parts
        ),
        path.name,
    )


def discover_frames(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory}: select a point cloud directory.")
    if (directory / "velodyne").is_dir():
        directory = directory / "velodyne"
    frames = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in (".bin", ".ply")
        ),
        key=_natural_key,
    )
    if not frames:
        raise ValueError(
            f"{directory}: no BIN or PLY point cloud files found."
        )
    return frames


def default_label_path(path):
    return Path(path).with_suffix(".label")


def _path_present(path):
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def label_candidates(path, label_directory=None):
    path = Path(path)
    if label_directory is not None:
        candidates = [Path(label_directory) / f"{path.stem}.label"]
    else:
        directory = path.parent
        if directory.name == "velodyne":
            directory = directory.parent
        candidates = [
            default_label_path(path),
            directory / "labels" / f"{path.stem}.label",
        ]
    result = []
    seen = set()
    for candidate in candidates:
        if _path_present(candidate) and candidate.resolve() not in seen:
            result.append(candidate)
            seen.add(candidate.resolve())
    return result


def _signature(stat):
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _check_unchanged(stream, path, before, path_before):
    if (
        _signature(os.fstat(stream.fileno())) != before
        or _signature(path.stat()) != path_before
    ):
        raise ValueError(
            f"{path}: file changed during loading; retry loading."
        )


def _ply_property(fields):
    if len(fields) == 3 and fields[1] in _PLY_TYPES:
        return fields[2], _PLY_TYPES[fields[1]], None
    if (
        len(fields) == 5
        and fields[1] == "list"
        and fields[2] in _PLY_TYPES
        and fields[3] in _PLY_TYPES
        and _PLY_TYPES[fields[2]][0] in "iu"
    ):
        return fields[4], _PLY_TYPES[fields[3]], _PLY_TYPES[fields[2]]
    raise ValueError("Unsupported PLY property declaration.")


def _ply_header(stream):
    if stream.readline(256).strip() != b"ply":
        raise ValueError("Missing PLY magic header.")
    encoding = None
    elements = []
    names = set()
    while stream.tell() < 1024 * 1024:
        line = stream.readline(65536)
        if not line:
            raise ValueError("Incomplete PLY header.")
        fields = line.decode("ascii").split()
        if not fields or fields[0] in ("comment", "obj_info"):
            continue
        if fields[0] == "end_header" and len(fields) == 1:
            break
        if fields[0] == "format" and len(fields) == 3:
            if encoding is not None or fields[2] != "1.0":
                raise ValueError(
                    "Unsupported or repeated PLY format declaration."
                )
            encoding = fields[1]
        elif fields[0] == "element" and len(fields) == 3:
            count = int(fields[2])
            if count < 0 or fields[1] in names:
                raise ValueError("Invalid or duplicate PLY element.")
            elements.append((fields[1], count, []))
            names.add(fields[1])
        elif fields[0] == "property" and elements:
            prop = _ply_property(fields)
            if prop[0] in {item[0] for item in elements[-1][2]}:
                raise ValueError("Duplicate PLY property name.")
            elements[-1][2].append(prop)
        else:
            raise ValueError("Invalid PLY header declaration.")
    else:
        raise ValueError("PLY header exceeds 1 MiB.")
    if encoding not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError("Unsupported or missing PLY encoding.")
    vertices = [item for item in elements if item[0] == "vertex"]
    if not vertices or vertices[0][1] == 0:
        raise ValueError("PLY requires a nonempty vertex element.")
    properties = vertices[0][2]
    if not {"x", "y", "z"}.issubset({item[0] for item in properties}):
        raise ValueError("PLY vertices require x, y and z properties.")
    if any(item[2] is not None for item in properties):
        raise ValueError("List properties on PLY vertices are not supported.")
    if any(count and not properties for _, count, properties in elements):
        raise ValueError("PLY elements with records require properties.")
    return encoding, elements


def _read_exact(stream, count):
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("Truncated PLY data records.")
    return data


def _ascii_scalar(token, dtype):
    if dtype[0] == "f":
        return float(token)
    value = int(token)
    limits = np.iinfo(dtype)
    if not limits.min <= value <= limits.max:
        raise ValueError("PLY integer value exceeds its declared type.")
    return value


def _skip_ply_records(stream, count, properties, encoding):
    endian = "<" if encoding == "binary_little_endian" else ">"
    if encoding != "ascii" and all(prop[2] is None for prop in properties):
        size = sum(np.dtype(prop[1]).itemsize for prop in properties) * count
        if size > os.fstat(stream.fileno()).st_size - stream.tell():
            raise ValueError("Truncated PLY data records.")
        stream.seek(size, os.SEEK_CUR)
        return
    for _ in range(count):
        tokens = stream.readline().split() if encoding == "ascii" else None
        position = 0
        for _, dtype, count_dtype in properties:
            size = 1
            if count_dtype:
                if tokens is not None:
                    if position >= len(tokens):
                        raise ValueError("Truncated PLY list record.")
                    size = _ascii_scalar(tokens[position], count_dtype)
                    position += 1
                else:
                    size = int(
                        np.frombuffer(
                            _read_exact(
                                stream, np.dtype(count_dtype).itemsize
                            ),
                            dtype=endian + count_dtype,
                        )[0]
                    )
                if size < 0:
                    raise ValueError("Negative PLY list length.")
            if tokens is None:
                byte_count = np.dtype(dtype).itemsize * size
                if (
                    byte_count
                    > os.fstat(stream.fileno()).st_size - stream.tell()
                ):
                    raise ValueError("Truncated PLY list data.")
                stream.seek(byte_count, os.SEEK_CUR)
            else:
                if position + size > len(tokens):
                    raise ValueError("Truncated PLY data record.")
                for token in tokens[position : position + size]:
                    _ascii_scalar(token, dtype)
                position += size
        if tokens is not None and position != len(tokens):
            raise ValueError("Unexpected values in PLY data record.")


def _read_ply(stream):
    encoding, elements = _ply_header(stream)
    points = None
    has_intensity = False
    rgb = None
    for name, count, properties in elements:
        if name != "vertex":
            _skip_ply_records(stream, count, properties, encoding)
            continue
        endian = "<" if encoding == "binary_little_endian" else ">"
        dtype = np.dtype([(prop[0], endian + prop[1]) for prop in properties])
        if encoding == "ascii":
            values = np.loadtxt(
                stream,
                dtype=np.float64,
                max_rows=count,
                ndmin=2,
                comments=None,
            )
            if values.shape != (count, len(properties)):
                raise ValueError(
                    "PLY vertex count or property count mismatch."
                )
            for column, (_, scalar, _) in enumerate(properties):
                if scalar[0] in "iu":
                    limits = np.iinfo(scalar)
                    data = values[:, column]
                    if not (
                        np.isfinite(data).all()
                        and np.all(data == np.floor(data))
                        and np.all((data >= limits.min) & (data <= limits.max))
                    ):
                        raise ValueError(
                            "Invalid integer in PLY vertex property."
                        )
            columns = {
                prop[0]: values[:, i] for i, prop in enumerate(properties)
            }
        else:
            if (
                count * dtype.itemsize
                > os.fstat(stream.fileno()).st_size - stream.tell()
            ):
                raise ValueError("Truncated PLY vertex records.")
            values = np.fromfile(stream, dtype=dtype, count=count)
            if len(values) != count:
                raise ValueError("Truncated PLY vertex records.")
            columns = {prop[0]: values[prop[0]] for prop in properties}
        points = np.zeros((count, 4), dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            for column, key in enumerate(("x", "y", "z", "intensity")):
                if key in columns:
                    points[:, column] = columns[key]
        has_intensity = "intensity" in columns
        if all(key in columns for key in ("red", "green", "blue")):
            color_types = {prop[0]: prop[1] for prop in properties}
            if all(
                color_types[key] == "u1" for key in ("red", "green", "blue")
            ):
                rgb = np.column_stack(
                    [columns[key] for key in ("red", "green", "blue")]
                ).astype(np.uint8)

    if encoding == "ascii":
        if stream.read().strip():
            raise ValueError("Unexpected data after declared PLY elements.")
    elif stream.read(1):
        raise ValueError("Unexpected data after declared PLY elements.")
    return points, has_intensity, rgb


def load_frame(path, label_path=None):
    path = Path(path)
    if path.suffix.lower() not in (".bin", ".ply"):
        raise ValueError(
            f"{path}: supported point cloud formats are BIN and PLY."
        )
    warnings = []
    has_intensity = True
    rgb = None
    try:
        path_before = _signature(path.stat())
        with path.open("rb") as stream:
            before = _signature(os.fstat(stream.fileno()))
            if path.suffix.lower() == ".bin":
                size = before[2]
                if not size or size % 16:
                    raise ValueError(
                        "BIN must contain nonempty 16-byte XYZI records."
                    )
                values = np.fromfile(stream, dtype="<f4")
                if values.size * 4 != size:
                    raise ValueError("Incomplete BIN point records.")
                points = values.reshape(-1, 4)
            else:
                points, has_intensity, rgb = _read_ply(stream)
                if not has_intensity:
                    warnings.append(
                        "PLY has no intensity; intensity coloring is unavailable."
                    )
            _check_unchanged(stream, path, before, path_before)
        invalid = ~np.isfinite(points[:, :3]).all(axis=1)
        if invalid.any():
            indices = np.flatnonzero(invalid)
            raise ValueError(
                f"{indices.size} points have invalid coordinates; "
                f"first point indices: {indices[:10].tolist()}."
            )
        invalid_intensity = np.count_nonzero(~np.isfinite(points[:, 3]))
        if invalid_intensity:
            warnings.append(
                f"{invalid_intensity} points have nonfinite intensity; "
                "only display colors use a fallback."
            )
        candidates = (
            label_candidates(path)
            if label_path is None
            else [Path(label_path)]
        )
        label_path = next(iter(candidates), default_label_path(path))
        labels = np.zeros(len(points), dtype=np.uint32)
        label_exists = _path_present(label_path)
        if label_exists:
            path_before = _signature(label_path.stat())
            with label_path.open("rb") as stream:
                before = _signature(os.fstat(stream.fileno()))
                if before[2] != len(points) * 4:
                    raise ValueError(
                        f"{label_path}: expected {len(points)} labels "
                        f"({len(points) * 4} bytes), found {before[2]} bytes."
                    )
                labels = np.fromfile(stream, dtype="<u4")
                _check_unchanged(stream, label_path, before, path_before)
        special = np.count_nonzero(
            (labels & 0xFFFF == 0) & (labels >> 16 != 0)
        )
        if special:
            warnings.append(
                f"{special} unlabeled points have nonzero instance IDs; "
                "original labels are preserved."
            )
        return Frame(
            path,
            points,
            labels,
            label_path,
            tuple(warnings),
            label_exists,
            has_intensity=has_intensity,
            rgb=rgb,
        )
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{path}: {error}") from error


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_labels(path, labels, source_path):
    path, source_path = Path(path), Path(source_path)
    if path.suffix.lower() != ".label":
        raise ValueError(
            f"{path}: label output must use the .label extension."
        )
    if path.resolve() == source_path.resolve() or (
        path.exists() and source_path.exists() and path.samefile(source_path)
    ):
        raise ValueError(
            f"{path}: labels cannot overwrite the source point cloud."
        )
    labels = validate_labels(labels)
    with source_path.open("rb") as stream:
        if source_path.suffix.lower() == ".bin":
            size = os.fstat(stream.fileno()).st_size
            if not size or size % 16:
                raise ValueError(f"{source_path}: invalid BIN point records.")
            count = size // 16
        elif source_path.suffix.lower() == ".ply":
            _, elements = _ply_header(stream)
            count = next(
                count for name, count, _ in elements if name == "vertex"
            )
        else:
            raise ValueError(f"{source_path}: unsupported point cloud format.")
    if len(labels) != count:
        raise ValueError(
            f"{path}: {len(labels)} labels do not match {count} points."
        )
    encoded = labels.astype("<u4", copy=False)
    _atomic_write(path, lambda stream: encoded.tofile(stream))


def _validate_classes(classes):
    result = []
    seen = set()
    for entry in classes:
        if not isinstance(entry, ClassDefinition):
            raise ValueError(
                "Class entries require id, name and color fields."
            )
        class_id = _validate_id(entry.id)
        if class_id in seen:
            raise ValueError(f"Duplicate class ID: {class_id}.")
        if not isinstance(entry.name, str) or not entry.name.strip():
            raise ValueError(f"Class {class_id}: name must not be empty.")
        if not isinstance(entry.color, str) or not re.fullmatch(
            r"#[0-9a-fA-F]{6}", entry.color
        ):
            raise ValueError(f"Class {class_id}: color must be #RRGGBB.")
        seen.add(class_id)
        result.append(
            ClassDefinition(class_id, entry.name.strip(), entry.color.upper())
        )
    if 0 not in seen:
        raise ValueError("Class configuration must include ID 0 (unlabeled).")
    return result


def load_classes(path):
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if (
            not isinstance(data, dict)
            or type(data.get("version")) is not int
            or data["version"] != 1
        ):
            raise ValueError("Class configuration requires version 1.")
        if not isinstance(data.get("classes"), list):
            raise ValueError("Class configuration requires a classes array.")
        classes = []
        for entry in data["classes"]:
            if not isinstance(entry, dict) or not {
                "id",
                "name",
                "color",
            }.issubset(entry):
                raise ValueError(
                    "Class entries require id, name and color fields."
                )
            classes.append(
                ClassDefinition(entry["id"], entry["name"], entry["color"])
            )
        return _validate_classes(classes)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{path}: {error}") from error


def save_classes(path, classes):
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(
            f"{path}: class configuration must use the .json extension."
        )
    classes = _validate_classes(classes)
    data = {"version": 1, "classes": [asdict(entry) for entry in classes]}
    encoded = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    _atomic_write(path, lambda stream: stream.write(encoded))
