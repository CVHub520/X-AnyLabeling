"""Real-model integration test for the Phase 2 YOLOE backend."""

from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from anylabeling import config as anylabeling_config
from anylabeling.services.auto_labeling.yoloe import YOLOE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / ".cache" / "models"
ASSET_DIR = PROJECT_ROOT / ".cache" / "yoloe" / "ultralytics" / "assets"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output" / "phase2"
REFERENCE = ASSET_DIR / "bus.jpg"
TARGET_B = ASSET_DIR / "zidane.jpg"
TARGET_C = ASSET_DIR / "bus_composite.jpg"
REFERENCE_BOXES = [
    [53.44, 400.42, 244.41, 900.67],
    [221.52, 405.8, 344.98, 857.54],
]


def make_model() -> YOLOE:
    anylabeling_config.current_config_file = "{}"
    config = {
        "type": "yoloe",
        "name": "yoloe_11s-phase2-integration",
        "display_name": "YOLOE-11-S Phase 2 Integration",
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


def summarize(result) -> dict:
    return {
        "count": len(result.shapes),
        "labels": [shape.label for shape in result.shapes],
        "scores": [shape.score for shape in result.shapes],
        "boxes": [
            [
                shape.points[0].x(),
                shape.points[0].y(),
                shape.points[2].x(),
                shape.points[2].y(),
            ]
            for shape in result.shapes
        ],
    }


def render_shapes(image_path: Path, result, output_path: Path) -> None:
    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    for shape in result.shapes:
        box = [
            shape.points[0].x(),
            shape.points[0].y(),
            shape.points[2].x(),
            shape.points[2].y(),
        ]
        draw.rectangle(box, outline="blue", width=4)
        draw.text(
            (box[0], max(0, box[1] - 14)),
            f"{shape.label} {shape.score:.2f}",
            fill="blue",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


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

    model = make_model()

    # Text -> Cross-Image -> Text. The same text model must survive unchanged.
    text_before = model.predict_shapes(
        image=True,
        image_path=str(REFERENCE),
        text_prompt="person.car",
    )
    text_model_id = id(model._text_model)

    model.set_auto_labeling_marks(
        [
            {
                "type": "rectangle",
                "data": box,
                "label": 1,
            }
            for box in REFERENCE_BOXES
        ]
    )
    vpe_started = time.perf_counter()
    model.build_visual_prompt(str(REFERENCE))
    vpe_seconds = time.perf_counter() - vpe_started
    cached_vpe = model.visual_prompt_vpe
    cached_vpe_pointer = cached_vpe.data_ptr()
    cross_model_id = id(model._cross_image_visual_model)

    target_b_started = time.perf_counter()
    target_b = model.predict_shapes(
        image=True,
        image_path=str(TARGET_B),
        text_prompt="",
    )
    target_b_seconds = time.perf_counter() - target_b_started

    target_c_started = time.perf_counter()
    target_c = model.predict_shapes(
        image=True,
        image_path=str(TARGET_C),
        text_prompt="",
    )
    target_c_seconds = time.perf_counter() - target_c_started

    cache_survived_targets = (
        model.visual_prompt_vpe is cached_vpe
        and model.visual_prompt_vpe.data_ptr() == cached_vpe_pointer
        and id(model._cross_image_visual_model) == cross_model_id
        and model.has_visual_prompt()
    )
    target_predictor = type(model._cross_image_visual_model.predictor).__name__

    text_after = model.predict_shapes(
        image=True,
        image_path=str(REFERENCE),
        text_prompt="person.car",
    )
    text_model_reused = id(model._text_model) == text_model_id

    # Prompt-Free -> Cross-Image -> Prompt-Free. Clearing only the cross-image
    # state must leave the prompt-free model alive and reusable.
    model.clear_visual_prompt()
    prompt_free_before = model.predict_shapes(
        image=True,
        image_path=str(REFERENCE),
        text_prompt="",
    )
    prompt_free_model_id = id(model._prompt_free_model)

    model.build_visual_prompt(
        str(REFERENCE),
        reference_boxes=REFERENCE_BOXES,
    )
    prompt_free_sequence_visual = model.predict_shapes(
        image=True,
        image_path=str(TARGET_B),
        text_prompt="",
    )
    model.clear_visual_prompt()
    prompt_free_after = model.predict_shapes(
        image=True,
        image_path=str(REFERENCE),
        text_prompt="",
    )
    prompt_free_model_reused = (
        id(model._prompt_free_model) == prompt_free_model_id
    )

    target_b_output = OUTPUT_DIR / "target_b_result.jpg"
    target_c_output = OUTPUT_DIR / "target_c_result.jpg"
    render_shapes(TARGET_B, target_b, target_b_output)
    render_shapes(TARGET_C, target_c, target_c_output)

    checks = {
        "text_before_detected": bool(text_before.shapes),
        "target_b_detected": bool(target_b.shapes),
        "target_c_detected": bool(target_c.shapes),
        "cache_survived_targets": cache_survived_targets,
        "target_uses_standard_predictor": (
            target_predictor != "YOLOEVPSegPredictor"
        ),
        "text_after_detected": bool(text_after.shapes),
        "text_model_reused": text_model_reused,
        "prompt_free_before_detected": bool(prompt_free_before.shapes),
        "prompt_free_sequence_visual_detected": bool(
            prompt_free_sequence_visual.shapes
        ),
        "prompt_free_after_detected": bool(prompt_free_after.shapes),
        "prompt_free_model_reused": prompt_free_model_reused,
    }
    report = {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reference": str(REFERENCE.resolve()),
        "reference_boxes": REFERENCE_BOXES,
        "target_b": str(TARGET_B.resolve()),
        "target_c": str(TARGET_C.resolve()),
        "vpe_shape": list(cached_vpe.shape),
        "vpe_dtype": str(cached_vpe.dtype),
        "vpe_device": str(cached_vpe.device),
        "vpe_generation_seconds": vpe_seconds,
        "target_b_seconds": target_b_seconds,
        "target_c_seconds": target_c_seconds,
        "target_predictor": target_predictor,
        "text_before": summarize(text_before),
        "target_b_result": summarize(target_b),
        "target_c_result": summarize(target_c),
        "text_after": summarize(text_after),
        "prompt_free_before": summarize(prompt_free_before),
        "prompt_free_visual": summarize(prompt_free_sequence_visual),
        "prompt_free_after": summarize(prompt_free_after),
        "target_b_output": str(target_b_output.resolve()),
        "target_c_output": str(target_c_output.resolve()),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "phase2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    model.unload()
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
