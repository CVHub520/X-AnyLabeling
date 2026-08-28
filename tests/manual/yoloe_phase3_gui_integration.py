"""Real Qt-panel integration test for Phase 3 cross-image YOLOE."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets

import anylabeling.resources.resources  # noqa: F401
from anylabeling import config as anylabeling_config
from anylabeling.services.auto_labeling.yoloe import YOLOE
from anylabeling.views.labeling.widgets.auto_labeling.auto_labeling import (
    AutoLabelingWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / ".cache" / "models"
ASSET_DIR = PROJECT_ROOT / ".cache" / "yoloe" / "ultralytics" / "assets"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output" / "phase3"
REFERENCE = ASSET_DIR / "bus.jpg"
TARGET_B = ASSET_DIR / "zidane.jpg"
TARGET_C = ASSET_DIR / "bus_composite.jpg"
REFERENCE_BOXES = [
    [53.44, 400.42, 244.41, 900.67],
    [221.52, 405.8, 344.98, 857.54],
]


class PanelParent:
    """Small LabelWidget stand-in; the tested panel itself is production UI."""

    def __init__(self):
        self._config = anylabeling_config.get_config()
        self.image = True
        self.filename = str(REFERENCE)
        self.results = []

    def new_shapes_from_auto_labeling(self, result):
        self.results.append(result)


def make_model() -> YOLOE:
    config = {
        "type": "yoloe",
        "name": "yoloe_11s-phase3-gui-integration",
        "display_name": "YOLOE-11-S Phase 3 GUI Integration",
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


def wait_for_prediction(manager, action, timeout_ms=120_000):
    loop = QtCore.QEventLoop()
    timed_out = False

    def on_timeout():
        nonlocal timed_out
        timed_out = True
        loop.quit()

    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(on_timeout)
    manager.prediction_finished.connect(loop.quit)
    timer.start(timeout_ms)
    action()
    loop.exec()
    timer.stop()
    manager.prediction_finished.disconnect(loop.quit)
    if timed_out:
        raise TimeoutError("GUI model worker did not finish in time.")


def summarize(result):
    return {
        "count": len(result.shapes),
        "labels": [shape.label for shape in result.shapes],
        "scores": [shape.score for shape in result.shapes],
    }


def main() -> int:
    for path in (
        REFERENCE,
        TARGET_B,
        TARGET_C,
        MODEL_DIR / "yoloe-11s-seg.pt",
        MODEL_DIR / "yoloe-11s-seg-pf.pt",
        MODEL_DIR / "mobileclip_blt.pt",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    anylabeling_config.current_config_file = "{}"
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    parent = PanelParent()
    panel = AutoLabelingWidget(parent)
    panel.resize(1600, panel.sizeHint().height())
    panel.show()
    app.processEvents()

    model = make_model()
    model_config = {
        "type": "yoloe",
        "name": model.config["name"],
        "display_name": model.config["display_name"],
        "iou_threshold": model.iou_thres,
        "conf_threshold": model.conf_thres,
        "model": model,
    }
    panel.model_manager.loaded_model_config = model_config
    panel.model_manager.model_loaded.emit(model_config)
    app.processEvents()

    marks = [
        {"type": "rectangle", "data": box, "label": 1}
        for box in REFERENCE_BOXES
    ]
    cleared_marks = []
    panel.clear_auto_labeling_action_requested.connect(
        lambda: cleared_marks.append(True)
    )
    panel.on_new_marks(marks)
    app.processEvents()
    marks_status = panel.visual_prompt_status_label.text()

    generation_started = time.perf_counter()
    wait_for_prediction(
        panel.model_manager,
        panel.button_generate_visual_prompt.click,
    )
    generation_seconds = time.perf_counter() - generation_started
    app.processEvents()
    ready_status = panel.visual_prompt_status_label.text()
    vpe = model.visual_prompt_vpe
    vpe_pointer = vpe.data_ptr()
    cross_model_id = id(model._cross_image_visual_model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.grab().save(str(OUTPUT_DIR / "auto_labeling_panel.png"))

    parent.filename = str(TARGET_B)
    target_b_started = time.perf_counter()
    wait_for_prediction(panel.model_manager, panel.button_send.click)
    target_b_seconds = time.perf_counter() - target_b_started
    app.processEvents()
    target_b = parent.results[-1]

    parent.filename = str(TARGET_C)
    target_c_started = time.perf_counter()
    wait_for_prediction(panel.model_manager, panel.button_send.click)
    target_c_seconds = time.perf_counter() - target_c_started
    app.processEvents()
    target_c = parent.results[-1]

    cache_survived = (
        model.visual_prompt_vpe is vpe
        and model.visual_prompt_vpe.data_ptr() == vpe_pointer
        and id(model._cross_image_visual_model) == cross_model_id
        and model.has_visual_prompt()
    )
    target_predictor = type(model._cross_image_visual_model.predictor).__name__

    panel.button_clear_visual_prompt.click()
    app.processEvents()
    cleared_status = panel.visual_prompt_status_label.text()

    checks = {
        "controls_visible_for_yoloe": all(
            widget.isVisible()
            for widget in (
                panel.button_generate_visual_prompt,
                panel.visual_prompt_status_label,
                panel.button_clear_visual_prompt,
            )
        ),
        "marks_status_displayed": "Reference boxes ready (2)" in marks_status,
        "vpe_ready_status_displayed": "Ready" in ready_status,
        "temporary_marks_clear_requested": bool(cleared_marks),
        "vpe_generated_on_cuda": str(vpe.device).startswith("cuda"),
        "target_b_detected": bool(target_b.shapes),
        "target_c_detected": bool(target_c.shapes),
        "cache_survived_targets": cache_survived,
        "target_uses_standard_predictor": (
            target_predictor != "YOLOEVPSegPredictor"
        ),
        "clear_removed_vpe": not model.has_visual_prompt(),
        "clear_status_displayed": "Not generated" in cleared_status,
        "clear_button_disabled": not panel.button_clear_visual_prompt.isEnabled(),
    }
    report = {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reference": str(REFERENCE.resolve()),
        "reference_boxes": REFERENCE_BOXES,
        "target_b": str(TARGET_B.resolve()),
        "target_c": str(TARGET_C.resolve()),
        "marks_status": marks_status,
        "ready_status": ready_status,
        "cleared_status": cleared_status,
        "vpe_shape": list(vpe.shape),
        "vpe_dtype": str(vpe.dtype),
        "vpe_device": str(vpe.device),
        "vpe_generation_seconds": generation_seconds,
        "target_b_seconds": target_b_seconds,
        "target_c_seconds": target_c_seconds,
        "target_predictor": target_predictor,
        "target_b_result": summarize(target_b),
        "target_c_result": summarize(target_c),
        "panel_screenshot": str(
            (OUTPUT_DIR / "auto_labeling_panel.png").resolve()
        ),
    }
    (OUTPUT_DIR / "phase3_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    panel.close()
    model.unload()
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
