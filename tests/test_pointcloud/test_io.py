import json
import os
import struct

import numpy as np
import pytest

from anylabeling.views.labeling.pointcloud import io
from anylabeling.views.labeling.pointcloud.model import (
    DEFAULT_CLASSES,
    AnnotationDocument,
    ClassDefinition,
)


def bin_file(path, points=None):
    points = np.asarray(
        points if points is not None else [[1, 2, 3, 0.5], [4, 5, 6, 0.8]],
        dtype="<f4",
    )
    points.tofile(path)
    return path


def ply_file(path, encoding, intensity=True, mesh=False):
    properties = "property float z\nproperty double x\nproperty float y\nproperty uchar red\n"
    if intensity:
        properties += "property float intensity\n"
    header = f"ply\nformat {encoding} 1.0\ncomment test\nelement vertex 2\n{properties}"
    if mesh:
        header += "element face 1\nproperty list uchar int vertex_indices\n"
    header += "end_header\n"
    rows = [(3, 1, 2, 255), (6, 4, 5, 0)]
    rows = [
        row + (value,) if intensity else row
        for row, value in zip(rows, [0.5, 0.8])
    ]
    if encoding == "ascii":
        payload = "".join(
            " ".join(map(str, row)) + "\n" for row in rows
        ).encode("ascii")
        if mesh:
            payload += b"3 0 1 0\n"
    else:
        endian = "<" if encoding == "binary_little_endian" else ">"
        format_ = endian + "fdfB" + ("f" if intensity else "")
        payload = b"".join(struct.pack(format_, *row) for row in rows)
        if mesh:
            payload += struct.pack(endian + "Biii", 3, 0, 1, 0)
    path.write_bytes(header.encode("ascii") + payload)
    return path


def test_bin_without_labels_and_unknown_label_exact_roundtrip(tmp_path):
    source = bin_file(tmp_path / "scan.bin")
    original = source.read_bytes()
    frame = io.load_frame(source)
    np.testing.assert_array_equal(frame.labels, [0, 0])
    assert frame.label_path == tmp_path / "scan.label"
    assert not frame.label_path.exists()
    assert not (tmp_path / "labels").exists()
    labels = np.array([0xFFFFEA60, 0x00070000], dtype=np.uint32)
    io.save_labels(frame.label_path, labels, source)
    assert frame.label_path.read_bytes() == struct.pack("<II", *labels)
    restored = io.load_frame(source)
    np.testing.assert_array_equal(restored.labels, labels)
    assert any("nonzero instance" in warning for warning in restored.warnings)
    assert source.read_bytes() == original
    assert not (tmp_path / "labels").exists()


@pytest.mark.parametrize(
    "encoding", ["ascii", "binary_little_endian", "binary_big_endian"]
)
@pytest.mark.parametrize("intensity", [False, True])
@pytest.mark.parametrize("mesh", [False, True])
def test_ply_encodings_preserve_order_and_optional_attributes(
    tmp_path, encoding, intensity, mesh
):
    source = ply_file(tmp_path / "scan.ply", encoding, intensity, mesh)
    original = source.read_bytes()
    frame = io.load_frame(source)
    assert frame.label_path == source.with_suffix(".label")
    np.testing.assert_allclose(frame.points[:, :3], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(
        frame.points[:, 3], [0.5, 0.8] if intensity else [0, 0]
    )
    assert bool(frame.warnings) == (not intensity)
    labels = np.array([0x0007000A, 0x0007001E], dtype=np.uint32)
    io.save_labels(frame.label_path, labels, source)
    np.testing.assert_array_equal(io.load_frame(source).labels, labels)
    assert source.read_bytes() == original


@pytest.mark.parametrize("payload", [b"", b"\x00", b"\x00" * 15, b"\x00" * 17])
def test_incomplete_bin_rejected(tmp_path, payload):
    source = tmp_path / "broken.bin"
    source.write_bytes(payload)
    with pytest.raises(ValueError, match="broken.bin"):
        io.load_frame(source)


@pytest.mark.parametrize("coordinate", [np.nan, np.inf, -np.inf])
def test_invalid_coordinates_rejected_without_removing_points(
    tmp_path, coordinate
):
    source = bin_file(
        tmp_path / "scan.bin", [[coordinate, 0, 0, 0], [1, 2, 3, 4]]
    )
    with pytest.raises(ValueError, match="1 points.*indices: \\[0\\]"):
        io.load_frame(source)


def test_nonfinite_intensity_is_preserved_with_warning(tmp_path):
    source = bin_file(tmp_path / "scan.bin", [[1, 2, 3, np.nan]])
    frame = io.load_frame(source)
    assert np.isnan(frame.points[0, 3])
    assert any("nonfinite intensity" in warning for warning in frame.warnings)


@pytest.mark.parametrize("payload", [b"", b"\x00", b"\x00" * 4, b"\x00" * 12])
def test_corrupt_labels_never_treated_as_missing(tmp_path, payload):
    source = bin_file(tmp_path / "scan.bin")
    label = tmp_path / "scan.label"
    label.write_bytes(payload)
    with pytest.raises(ValueError, match="scan.label.*expected 2 labels"):
        io.load_frame(source, label)


def test_missing_explicit_directory_does_not_read_other_candidate(tmp_path):
    source = bin_file(tmp_path / "scan.bin")
    source.with_suffix(".label").write_bytes(struct.pack("<II", 10, 30))
    legacy = tmp_path / "labels/scan.label"
    legacy.parent.mkdir()
    legacy.write_bytes(struct.pack("<II", 40, 70))
    target = tmp_path / "explicit/scan.label"
    assert io.label_candidates(source, target.parent) == []
    frame = io.load_frame(source, target)
    np.testing.assert_array_equal(frame.labels, [0, 0])
    assert frame.label_path == target
    assert not target.parent.exists()


def test_natural_discovery_is_stable_direct_only_and_supports_sequence(
    tmp_path,
):
    point_dir = tmp_path / "velodyne"
    point_dir.mkdir()
    for name in ["10.bin", "2.bin", "02.bin", "1.PLY", "ignore.pcd"]:
        (point_dir / name).touch()
    (point_dir / "nested").mkdir()
    (point_dir / "nested/0.bin").touch()
    expected = ["1.PLY", "02.bin", "2.bin", "10.bin"]
    assert [path.name for path in io.discover_frames(tmp_path)] == expected
    assert [path.name for path in io.discover_frames(point_dir)] == expected
    assert (
        io.default_label_path(point_dir / "02.bin") == point_dir / "02.label"
    )
    with pytest.raises(ValueError):
        io.discover_frames(tmp_path / "absent")


@pytest.mark.parametrize("point_directory", ["clouds", "velodyne"])
def test_label_candidates_find_conflicts_and_match_complete_stem(
    tmp_path, point_directory
):
    point_dir = tmp_path / point_directory
    point_dir.mkdir()
    source = bin_file(point_dir / "000001.bin")
    adjacent = source.with_suffix(".label")
    adjacent.touch()
    legacy_directory = tmp_path if point_directory == "velodyne" else point_dir
    legacy = legacy_directory / "labels/000001.label"
    legacy.parent.mkdir()
    legacy.touch()
    (point_dir / "1.label").touch()
    assert io.label_candidates(source) == [adjacent, legacy]
    assert io.label_candidates(source, point_dir) == [adjacent]
    assert io.label_candidates(source, legacy.parent) == [legacy]
    assert io.label_candidates(source, tmp_path / "absent") == []


@pytest.mark.parametrize("point_directory", ["clouds", "velodyne"])
@pytest.mark.parametrize("extension", ["bin", "ply"])
def test_legacy_labels_are_read_without_creating_a_new_default(
    tmp_path, point_directory, extension
):
    point_dir = tmp_path / point_directory
    point_dir.mkdir()
    source = point_dir / f"scan.{extension}"
    if extension == "bin":
        bin_file(source)
    else:
        ply_file(source, "ascii")
    legacy_directory = tmp_path if point_directory == "velodyne" else point_dir
    legacy = legacy_directory / "labels/scan.label"
    legacy.parent.mkdir()
    labels = np.array([0x0007000A, 0xFFFFEA60], dtype="<u4")
    labels.tofile(legacy)
    adjacent = source.with_suffix(".label")
    assert io.default_label_path(source) == adjacent
    assert io.label_candidates(source) == [legacy]

    frame = io.load_frame(source)

    assert frame.label_path == legacy
    assert frame.label_exists
    np.testing.assert_array_equal(frame.labels, labels)
    io.save_labels(frame.label_path, frame.labels, source)
    assert legacy.read_bytes() == labels.tobytes()
    assert not adjacent.exists()


def test_adjacent_labels_take_priority_and_explicit_source_stays_isolated(
    tmp_path, monkeypatch
):
    source = bin_file(tmp_path / "scan.bin")
    adjacent = source.with_suffix(".label")
    adjacent.write_bytes(struct.pack("<II", 10, 30))
    legacy = tmp_path / "labels/scan.label"
    legacy.parent.mkdir()
    legacy.write_bytes(struct.pack("<II", 40, 70))

    frame = io.load_frame(source)

    assert frame.label_path == adjacent
    np.testing.assert_array_equal(frame.labels, [10, 30])
    original = io._path_present
    checked = []

    def record_path(path):
        checked.append(path)
        return original(path)

    monkeypatch.setattr(io, "_path_present", record_path)
    assert io.label_candidates(source, legacy.parent) == [legacy]
    frame = io.load_frame(source, legacy)
    assert frame.label_path == legacy
    np.testing.assert_array_equal(frame.labels, [40, 70])
    assert checked == [legacy, legacy]


def test_legacy_alias_does_not_duplicate_adjacent_candidate(tmp_path):
    source = bin_file(tmp_path / "scan.bin")
    adjacent = source.with_suffix(".label")
    adjacent.write_bytes(struct.pack("<II", 10, 30))
    legacy = tmp_path / "labels/scan.label"
    legacy.parent.mkdir()
    legacy.symlink_to(adjacent)

    assert io.label_candidates(source) == [adjacent]
    assert io.load_frame(source).label_path == adjacent


def test_broken_label_symlink_reports_missing_file(tmp_path):
    source = bin_file(tmp_path / "scan.bin")
    label = source.with_suffix(".label")
    label.symlink_to(tmp_path / "gone.label")
    assert io.label_candidates(source) == [label]
    with pytest.raises(OSError):
        io.load_frame(source, label)


def test_label_permission_failure_is_not_treated_as_missing(
    tmp_path, monkeypatch
):
    source = bin_file(tmp_path / "scan.bin")
    label = io.default_label_path(source)
    original_lstat = type(label).lstat

    def denied_lstat(path):
        if path == label:
            raise PermissionError(f"{path}: permission denied")
        return original_lstat(path)

    monkeypatch.setattr(type(label), "lstat", denied_lstat)
    with pytest.raises(PermissionError, match="permission denied"):
        io.label_candidates(source)
    with pytest.raises(PermissionError, match="permission denied"):
        io.load_frame(source, label)


def test_atomic_save_failure_preserves_previous_result_and_dirty_state(
    tmp_path, monkeypatch
):
    source = bin_file(tmp_path / "scan.bin")
    target = source.with_suffix(".label")
    target.write_bytes(struct.pack("<II", 30, 40))
    doc = AnnotationDocument(io.load_frame(source, target))
    doc.assign_semantic([0], 10, overwrite=True)
    previous = target.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        io.save_labels(target, doc.labels, source)
    assert target.read_bytes() == previous
    assert doc.dirty
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_fsync_failure_leaves_existing_target_complete(
    tmp_path, monkeypatch
):
    source = bin_file(tmp_path / "scan.bin")
    target = source.with_suffix(".label")
    target.write_bytes(struct.pack("<II", 30, 40))

    def fail_fsync(*args):
        raise OSError("simulated disk full")

    monkeypatch.setattr(io.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="disk full"):
        io.save_labels(target, np.array([0, 0]), source)
    assert target.read_bytes() == struct.pack("<II", 30, 40)
    assert not list(tmp_path.glob(".*.tmp"))


def test_save_rejects_original_alias_and_count_mismatch(tmp_path):
    source = bin_file(tmp_path / "scan.bin")
    original = source.read_bytes()
    alias = tmp_path / "alias.label"
    os.link(source, alias)
    for target in [source, alias]:
        with pytest.raises(ValueError):
            io.save_labels(target, np.array([0, 0]), source)
    with pytest.raises(ValueError, match="do not match"):
        io.save_labels(tmp_path / "bad.label", np.array([0]), source)
    assert source.read_bytes() == original
    assert not (tmp_path / "bad.label").exists()


def test_source_changed_during_read_is_rejected(tmp_path, monkeypatch):
    source = bin_file(tmp_path / "scan.bin")
    original_read = io.np.fromfile

    def changed_read(*args, **kwargs):
        result = original_read(*args, **kwargs)
        with source.open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(io.np, "fromfile", changed_read)
    with pytest.raises(ValueError, match="changed during loading"):
        io.load_frame(source)


@pytest.mark.parametrize(
    "encoding", ["ascii", "binary_little_endian", "binary_big_endian"]
)
def test_truncated_ply_and_extra_records_rejected(tmp_path, encoding):
    source = ply_file(tmp_path / "scan.ply", encoding, mesh=True)
    original = source.read_bytes()
    source.write_bytes(original[:-3])
    with pytest.raises(ValueError):
        io.load_frame(source)
    source.write_bytes(original + b"unexpected")
    with pytest.raises(ValueError, match="Unexpected data"):
        io.load_frame(source)


@pytest.mark.parametrize(
    "header",
    [
        "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n",
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nend_header\n1 2\n",
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float x\nend_header\n1 2\n",
        "ply\nformat ascii 2.0\nend_header\n",
        "not a ply file\n",
        "ply\nformat ascii 1.0\n",
    ],
)
def test_invalid_ply_headers_report_path(tmp_path, header):
    source = tmp_path / "broken.ply"
    source.write_text(header)
    with pytest.raises(ValueError, match="broken.ply"):
        io.load_frame(source)


def test_ply_invalid_integer_attribute_rejected(tmp_path):
    source = ply_file(tmp_path / "scan.ply", "ascii")
    source.write_bytes(source.read_bytes().replace(b"255", b"256"))
    with pytest.raises(ValueError, match="Invalid integer"):
        io.load_frame(source)


def test_class_configuration_roundtrip_and_atomic_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "classes.json"
    classes = list(DEFAULT_CLASSES) + [
        ClassDefinition(65535, "Custom name", "#AABBCC")
    ]
    io.save_classes(target, classes)
    assert io.load_classes(target) == classes
    assert json.loads(target.read_text())["version"] == 1
    original = target.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated failure")

    monkeypatch.setattr(io.os, "replace", fail_replace)
    with pytest.raises(OSError):
        io.save_classes(target, DEFAULT_CLASSES)
    assert target.read_bytes() == original


@pytest.mark.parametrize(
    "data",
    [
        {"version": 2, "classes": []},
        {"version": True, "classes": []},
        {"version": 1, "classes": []},
        {"version": 1, "classes": [{"id": 0, "name": "", "color": "#123456"}]},
        {
            "version": 1,
            "classes": [{"id": 0, "name": "Zero", "color": "blue"}],
        },
        {
            "version": 1,
            "classes": [{"id": 65536, "name": "Bad", "color": "#123456"}],
        },
        {
            "version": 1,
            "classes": [{"id": 0.0, "name": "Bad", "color": "#123456"}],
        },
        {"version": 1, "classes": [{"id": 0, "name": "Missing color"}]},
        {
            "version": 1,
            "classes": [{"id": 0, "name": "Zero", "color": "#123456"}] * 2,
        },
    ],
)
def test_invalid_class_config_never_partially_applies(tmp_path, data):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="invalid.json"):
        io.load_classes(path)


@pytest.mark.skipif(
    os.environ.get("POINTCLOUD_BENCHMARK") != "1",
    reason="Set POINTCLOUD_BENCHMARK=1 to run core performance measurements.",
)
def test_core_benchmark_and_fifty_work_cycles(tmp_path):
    import gc
    import hashlib
    import platform
    import statistics
    import time
    from pathlib import Path

    import psutil

    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    result = {
        "scope": "core-only synthetic data; warm OS cache, no rendering/UI",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu": cpu,
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "seed": 20260906,
        "repetitions": 5,
        "points": {},
    }
    for count in [100_000, 1_000_000]:
        rng = np.random.default_rng(result["seed"])
        points = rng.normal(size=(count, 4)).astype("<f4")
        source = tmp_path / f"{count}.bin"
        points.tofile(source)
        original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        metrics = {}
        for encoding in [
            "bin",
            "binary_little_endian",
            "binary_big_endian",
            "ascii",
        ]:
            path = source
            if encoding != "bin":
                path = tmp_path / f"{count}_{encoding}.ply"
                header = f"ply\nformat {encoding} 1.0\nelement vertex {count}\nproperty float x\nproperty float y\nproperty float z\nproperty float intensity\nend_header\n"
                with path.open("wb") as stream:
                    stream.write(header.encode("ascii"))
                    if encoding == "ascii":
                        np.savetxt(stream, points, fmt="%.9g")
                    else:
                        points.astype(
                            "<f4"
                            if encoding == "binary_little_endian"
                            else ">f4"
                        ).tofile(stream)
            timings = []
            for _ in range(result["repetitions"]):
                start = time.perf_counter()
                frame = io.load_frame(path)
                timings.append((time.perf_counter() - start) * 1000)
            np.testing.assert_allclose(frame.points, points, rtol=1e-6)
            metrics[f"{encoding}_load_median_ms"] = round(
                statistics.median(timings), 3
            )
        doc = AnnotationDocument(io.load_frame(source))
        selected = rng.choice(count, 1000, replace=False)
        timings = []
        save_times = []
        target = tmp_path / f"{count}.label"
        for _ in range(result["repetitions"]):
            start = time.perf_counter()
            doc.assign_semantic(selected, 10)
            timings.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            io.save_labels(target, doc.labels, source)
            save_times.append((time.perf_counter() - start) * 1000)
            doc.undo()
        metrics["edit_1000_points_median_ms"] = round(
            statistics.median(timings), 3
        )
        metrics["atomic_save_median_ms"] = round(
            statistics.median(save_times), 3
        )
        rss = []
        for cycle in range(50):
            doc = AnnotationDocument(io.load_frame(source, target))
            doc.assign_semantic(selected, cycle + 1, overwrite=True)
            identity = doc.create_instance(selected, cycle + 1)
            assert doc.instance_counts()[identity] == 1000
            io.save_labels(target, doc.labels, source)
            restored = io.load_frame(source, target)
            np.testing.assert_array_equal(restored.labels, doc.labels)
            doc.undo()
            doc.redo()
            np.testing.assert_array_equal(restored.labels, doc.labels)
            rss.append(psutil.Process().memory_info().rss / 2**20)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
        metrics["core_cycles_passed"] = 50
        metrics["cycle_rss_mib_first_last_max"] = [
            round(rss[0], 1),
            round(rss[-1], 1),
            round(max(rss), 1),
        ]
        result["points"][str(count)] = metrics
        del doc, frame, points, restored
        gc.collect()
    print(json.dumps(result, indent=2))


@pytest.mark.parametrize(
    "encoding", ["ascii", "binary_little_endian", "binary_big_endian"]
)
def test_ply_rgb_preserves_colors_and_reports_missing_intensity(
    tmp_path, encoding
):
    path = tmp_path / "rgb.ply"
    header = (
        f"ply\nformat {encoding} 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    rows = [(1, 2, 3, 255, 0, 17), (4, 5, 6, 0, 128, 255)]
    if encoding == "ascii":
        body = "".join(" ".join(map(str, row)) + "\n" for row in rows).encode()
    else:
        endian = "<" if encoding == "binary_little_endian" else ">"
        body = b"".join(struct.pack(endian + "fffBBB", *row) for row in rows)
    path.write_bytes(header.encode() + body)
    frame = io.load_frame(path)
    assert not frame.has_intensity
    assert frame.rgb.dtype == np.uint8 and not frame.rgb.flags.writeable
    np.testing.assert_array_equal(frame.rgb, [[255, 0, 17], [0, 128, 255]])
    np.testing.assert_array_equal(frame.points[:, :3], [[1, 2, 3], [4, 5, 6]])
