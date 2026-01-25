import json
from pathlib import Path

from anylabeling.views.common.converter import run_conversion

from tests.fixtures.sample_data import create_sample_workspace, load_json


def _write_classes(path: Path) -> None:
    path.write_text("cat\npoly\n", encoding="utf-8")


def _bbox_from_rectangle(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _assert_xlabel_schema(payload: dict) -> None:
    for k in ["version", "flags", "shapes", "imagePath", "imageData", "imageHeight", "imageWidth"]:
        assert k in payload
    assert isinstance(payload["shapes"], list)


def test_roundtrip_xlabel_yolo_detect(tmp_path):
    ws = create_sample_workspace(tmp_path)
    classes = tmp_path / "classes.txt"
    _write_classes(classes)

    yolo_out = tmp_path / "yolo"
    yolo_out.mkdir()

    run_conversion(
        "xlabel2yolo",
        images=str(ws["images_dir"]),
        labels=str(ws["labels_dir"]),
        output=str(yolo_out),
        classes_file=str(classes),
        mode="detect",
    )

    assert (yolo_out / "sample.txt").exists()

    back_out = tmp_path / "xlabel_from_yolo"
    back_out.mkdir()
    run_conversion(
        "yolo2xlabel",
        images=str(ws["images_dir"]),
        labels=str(yolo_out),
        output=str(back_out),
        classes_file=str(classes),
        mode="detect",
    )

    back = load_json(back_out / "sample.json")
    orig = load_json(ws["label"])
    _assert_xlabel_schema(back)
    _assert_xlabel_schema(orig)

    assert back["imageWidth"] == orig["imageWidth"]
    assert back["imageHeight"] == orig["imageHeight"]

    orig_rect = [s for s in orig["shapes"] if s["shape_type"] == "rectangle"][0]
    back_rect = [s for s in back["shapes"] if s["shape_type"] == "rectangle"][0]
    assert back_rect["label"] == orig_rect["label"]

    ob = _bbox_from_rectangle(orig_rect["points"])
    bb = _bbox_from_rectangle(back_rect["points"])
    for a, b in zip(ob, bb):
        assert abs(a - b) <= 1.0


def test_roundtrip_xlabel_voc_detect(tmp_path):
    ws = create_sample_workspace(tmp_path)

    voc_out = tmp_path / "voc"
    voc_out.mkdir()
    run_conversion(
        "xlabel2voc",
        images=str(ws["images_dir"]),
        labels=str(ws["labels_dir"]),
        output=str(voc_out),
        mode="detect",
    )
    assert (voc_out / "sample.xml").exists()

    back_out = tmp_path / "xlabel_from_voc"
    back_out.mkdir()
    run_conversion(
        "voc2xlabel",
        labels=str(voc_out),
        output=str(back_out),
        mode="detect",
    )
    back = load_json(back_out / "sample.json")
    _assert_xlabel_schema(back)
    assert any(s["label"] == "cat" for s in back["shapes"])


def test_roundtrip_xlabel_coco_detect(tmp_path):
    ws = create_sample_workspace(tmp_path)
    classes = tmp_path / "classes.txt"
    _write_classes(classes)

    coco_out = tmp_path / "coco"
    coco_out.mkdir()
    run_conversion(
        "xlabel2coco",
        images=str(ws["images_dir"]),
        labels=str(ws["labels_dir"]),
        output=str(coco_out),
        classes_file=str(classes),
        mode="detect",
    )

    coco_json = coco_out / "coco_detection.json"
    assert coco_json.exists()
    payload = json.loads(coco_json.read_text(encoding="utf-8"))
    assert "images" in payload and "annotations" in payload and "categories" in payload

    back_out = tmp_path / "xlabel_from_coco"
    back_out.mkdir()
    run_conversion(
        "coco2xlabel",
        labels=str(coco_json),
        output=str(back_out),
        classes_file=str(classes),
        mode="detect",
    )
    back = load_json(back_out / "sample.json")
    _assert_xlabel_schema(back)
    assert any(s["label"] == "cat" for s in back["shapes"])
