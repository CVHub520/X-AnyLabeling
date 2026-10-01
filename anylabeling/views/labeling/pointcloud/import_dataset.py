import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath

from .export import ExportCancelled, export_frame_names
from .importers import read_dataset
from .io import (
    class_config_data,
    cuboid_source_path,
    load_classes,
    load_cuboids,
)
from .model import ClassDefinition, DEFAULT_CLASSES


@dataclass(frozen=True)
class ImportTarget:
    path: Path
    label_path: Path
    cuboid_path: Path
    cuboids: tuple | None = None


@dataclass(frozen=True)
class FileChange:
    path: Path
    before: bytes | None
    after: bytes


@dataclass(frozen=True)
class ImportPlan:
    targets: tuple[ImportTarget, ...]
    classes: tuple[ClassDefinition, ...]
    changes: tuple[FileChange, ...]
    previous_objects: int
    new_classes: int

    @property
    def object_count(self):
        return sum(len(target.cuboids) for target in self.targets)


def _frame_key(name):
    name = PurePosixPath(name.replace("\\", "/")).name
    if PurePosixPath(name).suffix.lower() in (".bin", ".ply", ".pcd"):
        return str(PurePosixPath(name).with_suffix(""))
    return name


def _match_frames(imported, targets):
    exported = dict(zip(export_frame_names(targets), range(len(targets))))
    exact = {item.name for item in imported} == set(exported)
    by_name = {}
    for index, target in enumerate(targets):
        by_name.setdefault(target.path.stem, []).append(index)
    matched, seen = [], set()
    for item in imported:
        if item.name is None:
            index = item.index
        elif exact:
            index = exported[item.name]
        else:
            candidates = by_name.get(_frame_key(item.name), [])
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot uniquely match imported frame: {item.name}"
                )
            index = candidates[0]
        if type(index) is not int or not 0 <= index < len(targets):
            raise ValueError(
                f"Imported frame index is outside the current sequence: {index}"
            )
        if index in seen:
            raise ValueError(
                f"Multiple imported frames match {targets[index].path.name}."
            )
        seen.add(index)
        matched.append((targets[index], item.cuboids))
    return matched


def _reserved_ids(targets, check):
    used = set()
    for target in targets:
        check()
        source = cuboid_source_path(
            target.path, target.label_path, target.cuboid_path
        )
        boxes = target.cuboids
        if boxes is None and source.exists():
            boxes = load_cuboids(source, target.path)
        used.update(box.class_id for box in boxes or ())
    return used


def _merge_classes(imported, classes, targets, reserved_ids, check):
    result = list(classes)
    by_name = {item.name: item for item in classes if item.id != 0}
    if len(by_name) != sum(item.id != 0 for item in classes):
        raise ValueError(
            "Current class names must be unique before importing."
        )
    used = set(reserved_ids) | {item.id for item in classes}
    if any(item.name not in by_name for item in imported):
        used.update(_reserved_ids(targets, check))
    mapping = {}
    for item in imported:
        if item.name not in by_name:
            value = max(used, default=0) + 1
            if value > 65535:
                value = next(
                    (index for index in range(1, 65536) if index not in used),
                    None,
                )
            if value is None:
                raise ValueError(
                    "No class IDs remain for the imported classes."
                )
            entry = ClassDefinition(value, item.name, item.color)
            result.append(entry)
            by_name[item.name] = entry
            used.add(value)
        mapping[item.id] = by_name[item.name].id
    return tuple(result), mapping


def _read_existing(path):
    if path.is_symlink():
        raise ValueError(
            f"Choose a regular file as the import destination: {path}"
        )
    return path.read_bytes() if path.exists() else None


def prepare_import(
    path,
    format_name,
    targets,
    classes,
    config_path,
    *,
    reserved_ids=(),
    segmentation_classes=None,
    cancelled=None,
):
    def check():
        if cancelled is not None and cancelled():
            raise ExportCancelled()

    classes = tuple(entry for entry in classes if entry.id)
    definitions, imported = read_dataset(path, format_name, cancelled)
    matched = _match_frames(imported, targets)
    merged, mapping = _merge_classes(
        definitions, classes, targets, reserved_ids, check
    )
    changes, result = [], []
    previous_objects = 0
    protected = {Path(path).resolve()}
    for target in targets:
        protected.update((target.path.resolve(), target.label_path.resolve()))
    destinations = set()

    def add_change(destination, data):
        resolved = destination.resolve()
        if resolved in protected or resolved in destinations:
            raise ValueError(f"Conflicting import destination: {destination}")
        destinations.add(resolved)
        encoded = (
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        changes.append(
            FileChange(destination, _read_existing(destination), encoded)
        )

    for target, boxes in matched:
        check()
        if not target.cuboid_path.name.endswith(".cuboids.json"):
            raise ValueError(
                "Cuboid output must use the .cuboids.json extension."
            )
        before = target.cuboids
        source = cuboid_source_path(
            target.path, target.label_path, target.cuboid_path
        )
        if before is None:
            before = (
                load_cuboids(source, target.path) if source.exists() else ()
            )
        previous_objects += len(before)
        converted = tuple(
            replace(box, class_id=mapping[box.class_id]) for box in boxes
        )
        result.append(replace(target, cuboids=converted))
        add_change(
            target.cuboid_path,
            {
                "version": 1,
                "type": "pointcloud-cuboids",
                "point_cloud": target.path.name,
                "cuboids": [asdict(box) for box in converted],
            },
        )
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.suffix.lower() != ".json":
            raise ValueError(
                "Class configuration must use the .json extension."
            )
        if segmentation_classes is None:
            existing = (
                load_classes(config_path) if config_path.exists() else {}
            )
            segmentation_classes = existing.get(
                "segmentation", DEFAULT_CLASSES
            )
        add_change(
            config_path,
            class_config_data(
                {"detection": merged, "segmentation": segmentation_classes}
            ),
        )
    check()
    return ImportPlan(
        tuple(result),
        merged,
        tuple(changes),
        previous_objects,
        len(merged) - len(classes),
    )


def apply_import(plan, *, cancelled=None, progress=None):
    def check():
        if cancelled is not None and cancelled():
            raise ExportCancelled()

    staged, committed = [], []
    backups = {}
    try:
        for change in plan.changes:
            check()
            if _read_existing(change.path) != change.before:
                raise ValueError(
                    f"File changed after import preview: {change.path}"
                )
            change.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=f".{change.path.name}.",
                suffix=".import",
                dir=change.path.parent,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                staged.append(temporary)
                stream.write(change.after)
                stream.flush()
                os.fsync(stream.fileno())
        for index, (change, temporary) in enumerate(zip(plan.changes, staged)):
            check()
            if _read_existing(change.path) != change.before:
                raise ValueError(f"File changed during import: {change.path}")
            if change.before is not None:
                descriptor, name = tempfile.mkstemp(
                    prefix=f".{change.path.name}.",
                    suffix=".backup",
                    dir=change.path.parent,
                )
                os.close(descriptor)
                backup = Path(name)
                try:
                    os.replace(change.path, backup)
                except OSError:
                    backup.unlink(missing_ok=True)
                    raise
                backups[change.path] = backup
            committed.append(change.path)
            os.replace(temporary, change.path)
            if progress is not None:
                progress(index + 1, len(plan.changes))
        check()
    except BaseException:
        _rollback_import(committed, backups)
        raise
    finally:
        for temporary in staged:
            _remove_temporary(temporary)
    for backup in backups.values():
        _remove_temporary(backup)


def _remove_temporary(path):
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logging.getLogger(__name__).warning(
            "Could not remove import temporary file: %s", path
        )


def _rollback_import(committed, backups):
    failures = []
    for destination in reversed(committed):
        try:
            if destination in backups:
                os.replace(backups[destination], destination)
                del backups[destination]
            else:
                destination.unlink(missing_ok=True)
        except OSError as recovery_error:
            failures.append(f"{destination}: {recovery_error}")
    if failures:
        recovery = ", ".join(str(path) for path in backups.values())
        raise OSError(
            "Import rollback could not finish: "
            + "; ".join(failures)
            + f". Original files are retained at: {recovery}"
        )
