from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_ID = 65535


def _validate_id(value, allow_zero=True):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("IDs must be integers.")
    if not (0 if allow_zero else 1) <= value <= MAX_ID:
        raise ValueError("ID is outside the supported 16-bit range.")
    return int(value)


def validate_labels(labels, count=None):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("Labels must be a one-dimensional integer array.")
    if count is not None and labels.size != count:
        raise ValueError(
            f"Point count {count} does not match label count {labels.size}."
        )
    if labels.size and (labels.min() < 0 or labels.max() > 0xFFFFFFFF):
        raise ValueError("Labels must fit unsigned 32-bit integers.")
    return np.ascontiguousarray(labels, dtype=np.uint32)


@dataclass
class Frame:
    path: Path
    points: np.ndarray
    labels: np.ndarray
    label_path: Path | None = None
    warnings: tuple[str, ...] = ()
    label_exists: bool = False
    has_intensity: bool = True
    rgb: np.ndarray | None = None

    def __post_init__(self):
        self.path = Path(self.path)
        if self.label_path is not None:
            self.label_path = Path(self.label_path)
        self.points = np.ascontiguousarray(self.points, dtype=np.float32)
        if (
            self.points.ndim != 2
            or self.points.shape[1] != 4
            or not len(self.points)
        ):
            raise ValueError("A frame requires a nonempty N x 4 point array.")
        if not np.isfinite(self.points[:, :3]).all():
            raise ValueError(f"{self.path}: point coordinates must be finite.")
        if self.rgb is not None:
            rgb = np.asarray(self.rgb)
            if rgb.shape != (len(self.points), 3) or rgb.dtype != np.uint8:
                raise ValueError("RGB colors must be an N x 3 uint8 array.")
            self.rgb = np.ascontiguousarray(rgb)
            self.rgb.flags.writeable = False
        self.points.flags.writeable = False
        self.labels = validate_labels(self.labels, len(self.points))


@dataclass(frozen=True)
class ClassDefinition:
    id: int
    name: str
    color: str


DEFAULT_CLASSES = (ClassDefinition(0, "Unlabeled", "#808080"),)


@dataclass
class _Edit:
    indices: np.ndarray
    before: np.ndarray
    after: np.ndarray


class AnnotationDocument:
    def __init__(self, frame, history_limit=50):
        if history_limit < 20:
            raise ValueError("History must retain at least 20 operations.")
        self.frame = frame
        self._labels = frame.labels.copy()
        self.frame.labels = self.labels
        self._baseline = self._labels.copy()
        self._dirty_count = 0
        self._undo = []
        self._redo = []
        self._history_limit = history_limit
        halves = self._labels.view(np.uint16).reshape(-1, 2)
        halves.flags.writeable = False
        self._semantic_view = halves[:, 0 if np.little_endian else 1]
        self._instance_view = halves[:, 1 if np.little_endian else 0]
        self._semantic_counts = np.zeros(MAX_ID + 1, dtype=np.int64)
        self._instance_counts = {}
        self._update_counts(self._labels, 1)
        self._revision = 0
        self._last_changed_indices = np.empty(0, dtype=np.int32)

    @property
    def labels(self):
        values = self._labels.view()
        values.flags.writeable = False
        return values

    @property
    def semantic(self):
        return self._labels & np.uint32(MAX_ID)

    @property
    def instance(self):
        return self._labels >> np.uint32(16)

    @property
    def semantic_view(self):
        return self._semantic_view.view()

    @property
    def instance_view(self):
        return self._instance_view.view()

    @property
    def revision(self):
        return self._revision

    @property
    def last_changed_indices(self):
        values = self._last_changed_indices.view()
        values.flags.writeable = False
        return values

    def semantic_counts(self):
        values = self._semantic_counts.view()
        values.flags.writeable = False
        return values

    @property
    def dirty(self):
        return self._dirty_count != 0

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    def _indices(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1:
            raise ValueError("Point indices must be a one-dimensional array.")
        if indices.dtype.kind == "b":
            if len(indices) != len(self._labels):
                raise ValueError(
                    "Point mask must match the frame point count."
                )
            return np.flatnonzero(indices)
        if not indices.size:
            return np.empty(0, dtype=np.int32)
        if indices.dtype.kind not in "iu":
            raise ValueError("Point indices must be integers.")
        if indices.min() < 0 or indices.max() >= len(self._labels):
            raise ValueError("Point index is outside the current frame.")
        dtype = np.int32 if len(self._labels) <= 0x7FFFFFFF else np.int64
        return np.unique(indices).astype(dtype, copy=False)

    def _update_counts(self, labels, direction):
        values, counts = np.unique(labels, return_counts=True)
        np.add.at(
            self._semantic_counts,
            values & np.uint32(MAX_ID),
            direction * counts,
        )
        instances = values >> np.uint32(16)
        members = instances != 0
        for value, count in zip(
            values[members].tolist(), counts[members].tolist()
        ):
            key = value & MAX_ID, value >> 16
            count = self._instance_counts.get(key, 0) + direction * count
            if count:
                self._instance_counts[key] = count
            else:
                del self._instance_counts[key]

    def _write(self, indices, values):
        baseline = self._baseline[indices]
        previous = self._labels[indices]
        before = np.count_nonzero(previous != baseline)
        after = np.count_nonzero(values != baseline)
        self._dirty_count += int(after) - int(before)
        self._update_counts(previous, -1)
        self._update_counts(values, 1)
        self._labels[indices] = values
        self._revision += 1
        self._last_changed_indices = indices

    def _commit(self, indices, values):
        values = np.broadcast_to(
            np.asarray(values, dtype=np.uint32), indices.shape
        )
        changed = self._labels[indices] != values
        indices = indices[changed]
        if not indices.size:
            return 0
        values = values[changed].copy()
        edit = _Edit(indices, self._labels[indices].copy(), values)
        self._write(indices, values)
        self._undo.append(edit)
        del self._undo[: max(0, len(self._undo) - self._history_limit)]
        self._redo.clear()
        return len(indices)

    def assign_semantic(self, indices, semantic_id, overwrite=False):
        semantic_id = _validate_id(semantic_id)
        indices = self._indices(indices)
        semantic = self._labels[indices] & np.uint32(MAX_ID)
        if not overwrite:
            indices = indices[semantic == 0]
            semantic = semantic[semantic == 0]
        changed_class = semantic != semantic_id
        return self._commit(indices[changed_class], semantic_id)

    def clear(self, indices):
        return self._commit(self._indices(indices), 0)

    def _key(self, key, require_existing=True):
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            raise ValueError(
                "An instance requires a semantic and instance ID."
            )
        semantic_id = _validate_id(key[0])
        instance_id = _validate_id(key[1], allow_zero=False)
        packed = np.uint32((instance_id << 16) | semantic_id)
        if (
            require_existing
            and (semantic_id, instance_id) not in self._instance_counts
        ):
            raise ValueError("The selected instance no longer exists.")
        return semantic_id, instance_id, packed

    def _allocate_instance(self, semantic_id):
        semantic_id = _validate_id(semantic_id, allow_zero=False)
        used = np.zeros(MAX_ID + 1, dtype=bool)
        for member_semantic, instance_id in self._instance_counts:
            if member_semantic == semantic_id:
                used[instance_id] = True
        used[0] = True
        free = np.flatnonzero(~used)
        if not free.size:
            raise ValueError(
                "No instance IDs remain for this class (range 1-65535)."
            )
        return int(free[0])

    def create_instance(self, indices, semantic_id):
        semantic_id = _validate_id(semantic_id, allow_zero=False)
        indices = self._indices(indices)
        indices = indices[
            (self._labels[indices] & np.uint32(MAX_ID)) == semantic_id
        ]
        if not indices.size:
            raise ValueError("Select at least one point of the target class.")
        instance_id = self._allocate_instance(semantic_id)
        self._commit(indices, (instance_id << 16) | semantic_id)
        return semantic_id, instance_id

    def add_to_instance(self, indices, key):
        semantic_id, _, packed = self._key(key)
        if semantic_id == 0:
            raise ValueError("Unlabeled points cannot acquire an instance.")
        indices = self._indices(indices)
        indices = indices[
            (self._labels[indices] & np.uint32(MAX_ID)) == semantic_id
        ]
        return self._commit(indices, packed)

    def remove_from_instance(self, indices, key):
        semantic_id, _, packed = self._key(key)
        indices = self._indices(indices)
        indices = indices[self._labels[indices] == packed]
        return self._commit(indices, semantic_id)

    def split_instance(self, indices, key):
        semantic_id, instance_id, packed = self._key(key)
        indices = self._indices(indices)
        indices = indices[self._labels[indices] == packed]
        count = self._instance_counts[(semantic_id, instance_id)]
        if not indices.size or indices.size == count:
            raise ValueError(
                "Split requires a nonempty proper subset of an instance."
            )
        instance_id = self._allocate_instance(semantic_id)
        self._commit(indices, (instance_id << 16) | semantic_id)
        return semantic_id, instance_id

    def merge_instances(self, target_key, source_keys):
        semantic_id, _, target = self._key(target_key)
        if semantic_id == 0:
            raise ValueError(
                "Unlabeled instances must be cleared or assigned a class."
            )
        sources = set()
        for key in source_keys:
            source_semantic, _, packed = self._key(key)
            if source_semantic != semantic_id:
                raise ValueError(
                    "Only instances of the same class can be merged."
                )
            if packed != target:
                sources.add(int(packed))
        if not sources:
            raise ValueError(
                "Select at least two distinct instances to merge."
            )
        indices = np.flatnonzero(np.isin(self._labels, list(sources)))
        return self._commit(indices, target)

    def delete_instance(self, key):
        semantic_id, _, packed = self._key(key)
        return self._commit(
            np.flatnonzero(self._labels == packed), semantic_id
        )

    def instance_counts(self):
        return self._instance_counts.copy()

    def undo(self):
        if not self._undo:
            return False
        edit = self._undo.pop()
        self._write(edit.indices, edit.before)
        self._redo.append(edit)
        return True

    def redo(self):
        if not self._redo:
            return False
        edit = self._redo.pop()
        self._write(edit.indices, edit.after)
        self._undo.append(edit)
        return True

    def mark_saved(self, path=None):
        self.frame.label_exists = True
        self._baseline[:] = self._labels
        self._dirty_count = 0
        if path is not None:
            self.frame.label_path = Path(path)
