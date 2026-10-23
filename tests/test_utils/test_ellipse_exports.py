"""Mixed-object formats, legacy semantics and per-file failure protection."""

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

from anylabeling.views.labeling.label_converter import LabelConverter
from anylabeling.views.labeling.utils import ellipse_io
from anylabeling.views.labeling.utils.ellipse import (
    ellipse_frame_points,
    segmentation_points,
)
from anylabeling.views.labeling.utils.shape import (
    convert_ellipse_files,
    shape_to_mask,
    shapes_to_label,
)


def sample(
    tmp_path: Path, count: int = 128
) -> tuple[LabelConverter, Path, Path, dict]:
    """Create a mixed native document and configured converter.

    Args:
        tmp_path: Isolated test directory.
        count: Requested sampling count.

    Returns:
        Converter, annotation path, image path and source data.
    """
    image = tmp_path / "frame_1.png"
    Image.new("RGB", (100, 80), "white").save(image)
    shapes = []
    for kind, points in [
        ("polygon", [(5, 5), (15, 5), (10, 15)]),
        ("quadrilateral", [(-10, 40), (20, 40), (20, 70), (-10, 70)]),
        ("ellipse", ellipse_frame_points(80, 40, 70, 30, 0.37)),
    ]:
        shapes.append(
            {
                "shape_type": kind,
                "points": points,
                "label": "object",
                "group_id": len(shapes) + 1,
                "description": "preserve me",
                "flags": {"reviewed": True},
                "attributes": {"material": "generic"},
                "difficult": False,
            }
        )
    data = {
        "version": "4.2.0",
        "shapes": shapes,
        "imageWidth": 100,
        "imageHeight": 80,
        "imagePath": image.name,
        "flags": {},
    }
    annotation = tmp_path / "frame_1.json"
    annotation.write_text(json.dumps(data), encoding="utf-8")
    converter = LabelConverter(ellipse_polygon_points=count)
    converter.classes = ["object"]
    return converter, annotation, image, data


@pytest.mark.parametrize("count", [12, 64, 128, 512])
def test_all_segmentation_formats_share_contours(
    tmp_path: Path, count: int
) -> None:
    """Check YOLO, VOC, COCO, MASK and custom MOTS against one contour."""
    converter, native, image, data = sample(tmp_path, count)
    before = native.read_bytes()
    expected = [segmentation_points(s, 100, 80, count) for s in data["shapes"]]
    yolo = tmp_path / "out.txt"
    converter.custom_to_yolo(str(native), str(yolo), "seg")
    rows = [
        list(map(float, row.split())) for row in yolo.read_text().splitlines()
    ]
    assert len(rows) == 3
    for row, contour in zip(rows, expected):
        assert row[0] == 0
        assert np.array(row[1:]).reshape(-1, 2) * [100, 80] == pytest.approx(
            np.array(contour)
        )
        assert all(0 <= coordinate <= 1 for coordinate in row[1:])
    assert len(rows[1]) == 9

    voc = tmp_path / "out.xml"
    converter.custom_to_voc(str(image), str(native), str(voc), "polygon")
    objects = ET.parse(voc).findall("object")
    assert len(objects) == 3
    for obj, contour in zip(objects, expected):
        vertices = list(obj.find("polygon"))
        assert [float(v.text) for v in vertices] == pytest.approx(
            [x for p in contour for x in p]
        )
    output = tmp_path / "coco"
    output.mkdir()
    converter.custom_to_coco(
        [str(image)], str(tmp_path), str(output), "polygon"
    )
    coco = json.loads(next(output.glob("*.json")).read_text(encoding="utf-8"))
    assert len(coco["annotations"]) == 3
    for index, (annotation, contour) in enumerate(
        zip(coco["annotations"], expected)
    ):
        assert annotation["segmentation"][0] == pytest.approx(
            [x for p in contour for x in p]
        )
        if index == 0:
            assert annotation["area"] == converter.calculate_polygon_area(
                annotation["segmentation"]
            )
            assert annotation["bbox"] == converter.get_min_enclosing_bbox(
                annotation["segmentation"]
            )
            continue
        assert annotation["area"] == pytest.approx(
            Polygon(contour).area, rel=1e-5
        )
        x0, y0, x1, y1 = Polygon(contour).bounds
        assert annotation["bbox"] == pytest.approx([x0, y0, x1 - x0, y1 - y0])

    # COCO modifies its own class list by inserting background, as upstream does.
    converter.classes = ["object"]
    mots = tmp_path / "mots"
    mots.mkdir()
    converter.custom_to_mots(str(tmp_path), str(mots))
    rows = (mots / "custom_gt.txt").read_text().splitlines()
    assert len(rows) == 3
    for row, contour in zip(rows, expected):
        points = ast.literal_eval(row.split(" ", 5)[5])
        assert np.array(points) == pytest.approx(np.array(contour))
    assert native.read_bytes() == before


@pytest.mark.parametrize(
    "format_name, colors",
    [("grayscale", {"object": 1}), ("rgb", {"object": [255, 0, 0]})],
)
def test_mask_corners_remain_background(
    tmp_path: Path, format_name: str, colors: dict
) -> None:
    """Catch the former rectangular control-frame fill in every mask path."""
    converter, native, _, data = sample(tmp_path)
    data["shapes"] = [data["shapes"][2]]
    data["shapes"][0]["points"] = ellipse_frame_points(50, 40, 60, 30, 0)
    native.write_text(json.dumps(data), encoding="utf-8")
    mask = tmp_path / "mask.png"
    converter.custom_to_mask(
        str(native), str(mask), {"type": format_name, "colors": colors}
    )
    pixels = np.array(Image.open(mask))
    assert not pixels[26, 21].any()
    assert pixels[40, 50].any()
    raw = shape_to_mask((80, 100), data["shapes"][0]["points"], "ellipse")
    assert not raw[26, 21] and raw[40, 50]
    cls, ins = shapes_to_label(
        (80, 100), data["shapes"], {"object": 7}, ellipse_polygon_points=64
    )
    assert cls[40, 50] == 7 and ins[40, 50] == 1
    assert cls[26, 21] == ins[26, 21] == 0


@pytest.mark.parametrize("mode", ["hbb", "obb", "pose"])
def test_incompatible_yolo_preserves_output(tmp_path: Path, mode: str) -> None:
    """Unsupported ellipse modes fail before truncating an old label."""
    converter, native, _, _ = sample(tmp_path)
    output = tmp_path / "old.txt"
    output.write_bytes(b"existing valid output\n")
    with pytest.raises(
        ValueError, match=r"frame_1.json.*object 2.*incompatible"
    ):
        converter.custom_to_yolo(str(native), str(output), mode)
    assert output.read_bytes() == b"existing valid output\n"


def test_batch_preflight_and_source_change(tmp_path: Path) -> None:
    """Whole-batch validation and fingerprint checks precede replacement."""
    converter, native, _, data = sample(tmp_path)
    other = tmp_path / "invalid.json"
    data["shapes"][2]["direction"] = True
    other.write_text(json.dumps(data), encoding="utf-8")
    old = tmp_path / "old.txt"
    old.write_bytes(b"keep")
    with pytest.raises(ValueError, match="invalid.json"):
        converter.prepare_export(
            [str(native), str(other)], "seg", use_classes=True
        )
    assert old.read_bytes() == b"keep"
    converter = LabelConverter()
    converter.classes = ["object"]
    converter.prepare_export([str(native)], "seg", use_classes=True)
    native.write_bytes(native.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed after preflight"):
        converter.custom_to_yolo(str(native), str(old), "seg")
    assert old.read_bytes() == b"keep"


def test_replace_failure_preserves_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replace failure cannot damage an existing label or leave temp files."""
    converter, native, _, _ = sample(tmp_path)
    output = tmp_path / "old.txt"
    output.write_bytes(b"keep")

    def fail_replace(source: str, destination: str) -> None:
        """Simulate Windows denying replacement of a locked destination."""
        raise PermissionError("locked destination")

    monkeypatch.setattr(ellipse_io.os, "replace", fail_replace)
    with pytest.raises(PermissionError):
        converter.custom_to_yolo(str(native), str(output), "seg")
    assert output.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".xal_ellipse_*"))


def test_missing_mapping_and_class_are_explicit(tmp_path: Path) -> None:
    """New objects cannot disappear because a class or mask color is absent."""
    converter, native, _, _ = sample(tmp_path)
    output = tmp_path / "old.png"
    output.write_bytes(b"keep")
    with pytest.raises(ValueError, match="mapping"):
        converter.custom_to_mask(
            str(native), str(output), {"type": "grayscale", "colors": {}}
        )
    assert output.read_bytes() == b"keep"
    converter.classes = []
    with pytest.raises(ValueError, match="classes"):
        converter.custom_to_yolo(str(native), str(tmp_path / "out.txt"), "seg")


def test_conversion_retains_metadata_and_cancel(tmp_path: Path) -> None:
    """Explicit conversion removes direction and leaves unprocessed files intact."""
    _, native, _, data = sample(tmp_path)
    second = tmp_path / "second.json"
    second.write_bytes(native.read_bytes())
    before = second.read_bytes()
    results = convert_ellipse_files(
        [str(native), str(second)],
        64,
        cancelled=lambda: native.read_bytes() != before,
    )
    assert results[str(native)][0] == "success"
    assert results[str(second)][0] == "unprocessed"
    converted = json.loads(native.read_text())["shapes"][2]
    assert (
        converted["shape_type"] == "polygon" and len(converted["points"]) == 64
    )
    for key in ("label", "group_id", "description", "flags", "attributes"):
        assert converted[key] == data["shapes"][2][key]
    assert "direction" not in converted and second.read_bytes() == before


def test_legacy_polygon_filtering_and_skip_preserve_output(
    tmp_path: Path,
) -> None:
    """Existing unknown-class polygons still skip, preserving old empty outputs."""
    converter, native, _, data = sample(tmp_path)
    data["shapes"] = [data["shapes"][0]]
    data["shapes"][0]["label"] = "unknown"
    native.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "old.txt"
    output.write_bytes(b"keep")
    assert converter.custom_to_yolo(
        str(native), str(output), "seg", skip_empty_files=True
    )
    assert output.read_bytes() == b"keep"


@pytest.mark.parametrize(
    "override, expected", [(None, 64), (12, 12), (512, 512)]
)
def test_real_cli_config_before_gui(
    tmp_path: Path, override: int | None, expected: int
) -> None:
    """Exercise the real early convert subcommand and its selected YAML."""
    source = tmp_path / "source"
    source.mkdir()
    _, native, _, data = sample(source)
    data["shapes"] = [data["shapes"][2]]
    data["shapes"][0]["points"] = ellipse_frame_points(50, 40, 40, 20, 0.3)
    native.write_text(json.dumps(data), encoding="utf-8")
    classes = tmp_path / "classes.txt"
    classes.write_text("object\n", encoding="utf-8")
    config = tmp_path / "selected.yaml"
    config.write_text(
        "canvas:\n  ellipse:\n    polygon_points: 64\n", encoding="utf-8"
    )
    output = tmp_path / "output"
    command = [
        sys.executable,
        "-m",
        "anylabeling.app",
        "convert",
        "--task",
        "xlabel2yolo",
        "--mode",
        "segment",
        "--images",
        str(source),
        "--classes",
        str(classes),
        "--output",
        str(output),
        "--config",
        str(config),
        "--work-dir",
        str(tmp_path / "home"),
    ]
    if override is not None:
        command.extend(["--ellipse-polygon-points", str(override)])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=dict(os.environ, PYTHONUTF8="1"),
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        len((output / "frame_1.txt").read_text().split()) == 1 + 2 * expected
    )
    assert "success=1" in result.stdout


def test_segment_dataset_uses_current_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Data preparation includes native ellipses instead of marking background."""
    from anylabeling.services.auto_training.ultralytics import general

    source = tmp_path / "source"
    source.mkdir()
    _, native, image, data = sample(source)
    data["shapes"] = [data["shapes"][2]]
    data["shapes"][0]["points"] = ellipse_frame_points(50, 40, 40, 20, 0.3)
    native.write_text(json.dumps(data), encoding="utf-8")
    config = tmp_path / "data.yaml"
    config.write_text("names: [object]\n", encoding="utf-8")
    monkeypatch.setattr(
        general, "get_dataset_path", lambda: str(tmp_path / "datasets")
    )
    output = general.create_yolo_dataset(
        [str(image)], "Segment", 1.0, str(config), ellipse_polygon_points=64
    )
    label = next(Path(output).glob("labels/train/*.txt"))
    assert len(label.read_text().split()) == 129
    assert (
        "Background images: 0"
        in (Path(output) / "dataset_info.txt").read_text()
    )


def test_coco_grouping_and_ignore_policy(tmp_path: Path) -> None:
    """Mixed instances preserve grouping and the existing COCO ignore label."""
    from shapely.ops import unary_union

    converter, native, image, data = sample(tmp_path, 64)
    for shape in data["shapes"]:
        shape["group_id"] = 7
    ignored = copy.deepcopy(data["shapes"][2])
    ignored["label"] = "__ignore__"
    data["shapes"].append(ignored)
    native.write_text(json.dumps(data), encoding="utf-8")
    before = native.read_bytes()
    output = tmp_path / "coco"
    output.mkdir()
    converter.custom_to_coco(
        [str(image)], str(tmp_path), str(output), "polygon"
    )
    exported = json.loads(next(output.glob("*.json")).read_text())
    assert len(exported["annotations"]) == 1
    annotation = exported["annotations"][0]
    assert len(annotation["segmentation"]) == 3
    geometry = unary_union(
        [
            Polygon(segmentation_points(s, 100, 80, 64))
            for s in data["shapes"][:3]
        ]
    )
    assert annotation["area"] == pytest.approx(geometry.area)
    assert native.read_bytes() == before


def test_output_cannot_replace_native_source(tmp_path: Path) -> None:
    """Only explicit native conversion may replace an annotation input."""
    _, native, _, _ = sample(tmp_path)
    before = native.read_bytes()
    fingerprints = ellipse_io.preflight_files([str(native)], "seg")
    with pytest.raises(ValueError, match="overlaps"):
        with ellipse_io.atomic_open(
            str(native), fingerprints=fingerprints
        ) as stream:
            stream.write("{}")
    assert native.read_bytes() == before


def test_gui_preflight_preserves_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad object is rejected before output confirmation or directory edits."""
    from types import SimpleNamespace
    from PyQt6 import QtWidgets
    from anylabeling.views.labeling.utils.export import (
        _prepare_export_directory,
    )

    converter, native, image, data = sample(tmp_path)
    data["shapes"][2]["direction"] = None
    native.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_bytes(b"protected")
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args: warnings.append(args[-1]),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: pytest.fail("confirmation preceded preflight"),
    )
    widget = SimpleNamespace(
        image_list=[str(image)],
        filename=str(image),
        output_dir=None,
        tr=lambda text: text,
    )
    assert not _prepare_export_directory(
        widget, converter, str(output), "seg", use_classes=True
    )
    assert sentinel.read_bytes() == b"protected"
    assert "object 2" in warnings[0]


def test_unrelated_changed_input_does_not_block_completed_file(
    tmp_path: Path,
) -> None:
    """Per-image commits recheck their own input instead of the whole batch."""
    converter, native, _, _ = sample(tmp_path)
    other = tmp_path / "other.json"
    other.write_bytes(native.read_bytes())
    converter.prepare_export(
        [str(native), str(other)], "seg", use_classes=True
    )
    other.write_bytes(other.read_bytes() + b"\n")
    first_output = tmp_path / "first.txt"
    converter.custom_to_yolo(str(native), str(first_output), "seg")
    assert first_output.is_file()
    old_output = tmp_path / "old.txt"
    old_output.write_bytes(b"keep")
    with pytest.raises(ValueError, match="changed after preflight"):
        converter.custom_to_yolo(str(other), str(old_output), "seg")
    assert old_output.read_bytes() == b"keep"
    assert converter.export_results[str(native)][0] == "success"
    assert converter.export_results[str(other)][0] == "failed"
