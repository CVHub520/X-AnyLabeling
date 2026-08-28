"""Real X-AnyLabeling BatchProcessingThread test with one cached YOLOE VPE."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets

from anylabeling import config as anylabeling_config
from anylabeling.services.auto_labeling.model_manager import ModelManager
from anylabeling.services.auto_labeling.yoloe import YOLOE
from anylabeling.views.labeling.utils.batch import BatchProcessingThread

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / ".cache" / "models"
ASSET_DIR = PROJECT_ROOT / ".cache" / "yoloe" / "ultralytics" / "assets"
OUTPUT_ROOT = PROJECT_ROOT / "tests" / "output" / "phase4_batch"
INPUT_DIR = OUTPUT_ROOT / "images"
LABEL_DIR = OUTPUT_ROOT / "labels"
REFERENCE = ASSET_DIR / "bus.jpg"
REFERENCE_BOXES = [
    [53.44, 400.42, 244.41, 900.67],
    [221.52, 405.8, 344.98, 857.54],
]


def make_model() -> YOLOE:
    config = {
        "type": "yoloe",
        "name": "yoloe_11s-phase4-batch-integration",
        "display_name": "YOLOE-11-S Phase 4 Batch Integration",
        "config_file": str(Path(__file__).resolve()),
        "model_path": str(MODEL_DIR / "yoloe-11s-seg.pt"),
        "model_pf_path": str(MODEL_DIR / "yoloe-11s-seg-pf.pt"),
        "embedding_model_path": str(MODEL_DIR / "mobileclip_blt.pt"),
        "input_height": 640,
        "input_width": 640,
        "iou_threshold": 0.70,
        "conf_threshold": 0.10,
        "max_det": 1000,
        "with_mask": False,
        "classes": ["person", "car"],
    }
    return YOLOE(config, on_message=print)


def prepare_batch_files():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    sources = [
        (ASSET_DIR / "bus.jpg", INPUT_DIR / "01_bus.jpg"),
        (ASSET_DIR / "zidane.jpg", INPUT_DIR / "02_zidane.jpg"),
        (
            ASSET_DIR / "bus_composite.jpg",
            INPUT_DIR / "03_bus_composite.jpg",
        ),
    ]
    for source, target in sources:
        shutil.copy2(source, target)
    broken = INPUT_DIR / "04_broken.jpg"
    broken.write_bytes(b"not a valid image")

    for label_file in LABEL_DIR.glob("*.json"):
        label_file.unlink()

    manual_shape = {
        "label": "manual_keep",
        "score": None,
        "points": [[1, 1], [5, 1], [5, 5], [1, 5]],
        "group_id": None,
        "description": "existing annotation",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
        "kie_linking": [],
    }
    existing = {
        "version": "4.0.3",
        "flags": {},
        "shapes": [manual_shape],
        "imagePath": "02_zidane.jpg",
        "imageData": None,
        "imageHeight": 720,
        "imageWidth": 1280,
        "description": "",
    }
    (LABEL_DIR / "02_zidane.json").write_text(
        json.dumps(existing, indent=2), encoding="utf-8"
    )
    (LABEL_DIR / "04_broken.json").write_text(
        json.dumps({"sentinel": "keep"}, indent=2), encoding="utf-8"
    )
    return [str(target) for _source, target in sources] + [str(broken)]


def main() -> int:
    for path in (
        REFERENCE,
        ASSET_DIR / "zidane.jpg",
        ASSET_DIR / "bus_composite.jpg",
        MODEL_DIR / "yoloe-11s-seg.pt",
        MODEL_DIR / "yoloe-11s-seg-pf.pt",
        MODEL_DIR / "mobileclip_blt.pt",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    anylabeling_config.current_config_file = "{}"
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        sys.argv
    )
    assert qt_app is not None
    image_list = prepare_batch_files()
    model = make_model()

    counts = {"build_visual_prompt": 0, "set_visual_prompt": 0}
    original_build = model.build_visual_prompt
    original_set = model.set_visual_prompt

    def counted_build(*args, **kwargs):
        counts["build_visual_prompt"] += 1
        return original_build(*args, **kwargs)

    def counted_set(*args, **kwargs):
        counts["set_visual_prompt"] += 1
        return original_set(*args, **kwargs)

    model.build_visual_prompt = counted_build
    model.set_visual_prompt = counted_set
    model.set_auto_labeling_marks(
        [
            {"type": "rectangle", "data": box, "label": 1}
            for box in REFERENCE_BOXES
        ]
    )
    model.build_visual_prompt(str(REFERENCE))
    model.set_auto_labeling_preserve_existing_annotations_state(True)

    cached_vpe = model.visual_prompt_vpe
    vpe_pointer = cached_vpe.data_ptr()
    cross_model_id = id(model._cross_image_visual_model)

    manager = ModelManager()
    manager.loaded_model_config = {"type": "yoloe", "model": model}
    app = SimpleNamespace(
        auto_labeling_widget=SimpleNamespace(model_manager=manager),
        image=True,
        output_dir=str(LABEL_DIR),
        _config={"store_data": False},
        cancel_processing=False,
        image_index=0,
    )

    progress = []
    failed = []
    finished = []
    fatal_errors = []
    worker = BatchProcessingThread(
        app,
        image_list,
        0,
        "yoloe",
        "",
        False,
        False,
        visual_prompt=True,
    )
    worker.progress_updated.connect(
        lambda value, label: progress.append([value, label])
    )
    worker.image_failed.connect(
        lambda filename, message: failed.append(
            {"image": filename, "message": message}
        )
    )
    worker.processing_finished.connect(
        lambda count, cancelled: finished.append(
            {"failed_count": count, "cancelled": cancelled}
        )
    )
    worker.error_occurred.connect(fatal_errors.append)

    loop = QtCore.QEventLoop()
    timeout = QtCore.QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    worker.finished.connect(loop.quit)
    timeout.start(120_000)
    started = time.perf_counter()
    worker.start()
    loop.exec()
    elapsed = time.perf_counter() - started
    timed_out = timeout.remainingTime() == -1 and worker.isRunning()
    timeout.stop()
    worker.wait(5_000)

    label_files = sorted(LABEL_DIR.glob("*.json"))
    valid_labels = {}
    for name in ("01_bus.json", "02_zidane.json", "03_bus_composite.json"):
        path = LABEL_DIR / name
        if path.is_file():
            valid_labels[name] = json.loads(path.read_text(encoding="utf-8"))
    broken_data = json.loads(
        (LABEL_DIR / "04_broken.json").read_text(encoding="utf-8")
    )
    detection_counts = {
        name: len(
            [
                shape
                for shape in data["shapes"]
                if shape.get("label") == "object"
            ]
        )
        for name, data in valid_labels.items()
    }
    existing_labels = [
        shape.get("label")
        for shape in valid_labels["02_zidane.json"]["shapes"]
    ]

    checks = {
        "three_valid_annotations_saved": len(valid_labels) == 3,
        "three_different_images_detected": all(
            count > 0 for count in detection_counts.values()
        ),
        "vpe_generated_once": counts["build_visual_prompt"] == 1,
        "vpe_set_once": counts["set_visual_prompt"] == 1,
        "same_vpe_reused": (
            model.visual_prompt_vpe is cached_vpe
            and model.visual_prompt_vpe.data_ptr() == vpe_pointer
        ),
        "same_model_reused": id(model._cross_image_visual_model)
        == cross_model_id,
        "standard_target_predictor": type(
            model._cross_image_visual_model.predictor
        ).__name__
        != "YOLOEVPSegPredictor",
        "progress_reached_all_inputs": [item[0] for item in progress]
        == [1, 2, 3, 4],
        "broken_image_reported_once": len(failed) == 1,
        "broken_annotation_preserved": broken_data == {"sentinel": "keep"},
        "existing_annotation_preserved": "manual_keep" in existing_labels,
        "batch_finished_without_fatal_error": (
            finished == [{"failed_count": 1, "cancelled": False}]
            and not fatal_errors
            and not timed_out
        ),
    }
    report = {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reference": str(REFERENCE.resolve()),
        "reference_boxes": REFERENCE_BOXES,
        "batch_images": image_list,
        "elapsed_seconds": elapsed,
        "vpe_shape": list(cached_vpe.shape),
        "vpe_dtype": str(cached_vpe.dtype),
        "vpe_device": str(cached_vpe.device),
        "vpe_calls": counts,
        "detection_counts": detection_counts,
        "progress": progress,
        "failed_images": failed,
        "saved_label_files": [str(path.resolve()) for path in label_files],
        "output_directory": str(LABEL_DIR.resolve()),
    }
    (OUTPUT_ROOT / "phase4_batch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    model.unload()
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
