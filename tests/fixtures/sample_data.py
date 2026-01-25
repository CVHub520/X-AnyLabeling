import json
from pathlib import Path

from PyQt6 import QtGui


def write_png(path: Path, *, width: int = 32, height: int = 32) -> None:
    image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("white"))
    if not image.save(str(path)):
        raise RuntimeError(f"Failed to save image: {path}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def create_sample_xlabel(image_path: str, *, width: int, height: int) -> dict:
    from anylabeling.views.labeling.schema import create_xlabel_template

    shapes = [
        {
            "label": "cat",
            "points": [[2, 2], [20, 2], [20, 20], [2, 20]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
            "description": "",
            "difficult": False,
            "kie_linking": [],
        },
        {
            "label": "poly",
            "points": [[5, 25], [10, 28], [18, 26], [15, 22]],
            "group_id": 1,
            "shape_type": "polygon",
            "flags": {"occluded": False},
            "description": "p",
            "difficult": False,
            "kie_linking": [],
        },
    ]
    return create_xlabel_template(
        image_path=image_path,
        image_data=None,
        image_height=height,
        image_width=width,
        shapes=shapes,
        flags={"verified": True},
    )


def create_sample_workspace(tmp_path: Path) -> dict:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    img = images_dir / "sample.png"
    write_png(img, width=32, height=32)
    xlabel = create_sample_xlabel("sample.png", width=32, height=32)
    lbl = labels_dir / "sample.json"
    write_json(lbl, xlabel)

    return {"images_dir": images_dir, "labels_dir": labels_dir, "image": img, "label": lbl}

