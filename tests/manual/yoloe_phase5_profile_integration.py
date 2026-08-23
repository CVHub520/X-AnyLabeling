"""Real CUDA save/clear/load test for VisualPromptProfile."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from anylabeling import config as anylabeling_config
from anylabeling.services.auto_labeling.yoloe import YOLOE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / ".cache" / "models"
ASSET_DIR = PROJECT_ROOT / ".cache" / "yoloe" / "ultralytics" / "assets"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output" / "phase5_profile"
PROFILE_DIR = OUTPUT_DIR / "visual_prompts" / "product_A"
REFERENCE = ASSET_DIR / "bus.jpg"
TARGET_B = ASSET_DIR / "zidane.jpg"
TARGET_C = ASSET_DIR / "bus_composite.jpg"
REFERENCE_BOXES = [
    [53.44, 400.42, 244.41, 900.67],
    [221.52, 405.8, 344.98, 857.54],
]


def make_model() -> YOLOE:
    config = {
        "type": "yoloe",
        "name": "yoloe_11s-phase5-profile-integration",
        "display_name": "YOLOE-11-S Phase 5 Profile Integration",
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


def main() -> int:
    required = (
        REFERENCE,
        TARGET_B,
        TARGET_C,
        MODEL_DIR / "yoloe-11s-seg.pt",
        MODEL_DIR / "yoloe-11s-seg-pf.pt",
        MODEL_DIR / "mobileclip_blt.pt",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    anylabeling_config.current_config_file = "{}"
    if PROFILE_DIR.exists():
        shutil.rmtree(PROFILE_DIR)

    model = make_model()
    started = time.perf_counter()
    model.build_visual_prompt(str(REFERENCE), reference_boxes=REFERENCE_BOXES)
    generated_vpe = model.visual_prompt_vpe.detach().cpu().clone()
    generation_seconds = time.perf_counter() - started
    metadata_path = Path(
        model.save_visual_prompt_profile("product_A", PROFILE_DIR)
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    model.clear_visual_prompt()
    cleared = not model.has_visual_prompt()
    started = time.perf_counter()
    profile = model.load_visual_prompt_profile(metadata_path)
    load_seconds = time.perf_counter() - started
    restored_vpe = model.visual_prompt_vpe.detach().cpu()

    target_b = model.predict_with_visual_prompt(str(TARGET_B))
    target_c = model.predict_with_visual_prompt(str(TARGET_C))
    predictor_name = type(model._cross_image_visual_model.predictor).__name__
    checks = {
        "metadata_json_exists": metadata_path.is_file(),
        "embedding_npz_exists": (PROFILE_DIR / "embedding.npz").is_file(),
        "profile_version_is_one": metadata["version"] == 1,
        "reference_metadata_preserved": (
            profile.reference["image_path"] == str(REFERENCE.resolve())
            and np.allclose(profile.reference["boxes"], REFERENCE_BOXES)
        ),
        "cache_cleared_before_load": cleared,
        "embedding_round_trip_exact": generated_vpe.equal(restored_vpe),
        "restored_to_cuda": model.visual_prompt_vpe.device.type == "cuda",
        "restored_float32": str(model.visual_prompt_vpe.dtype)
        == "torch.float32",
        "target_b_detected": len(target_b.shapes) > 0,
        "target_c_detected": len(target_c.shapes) > 0,
        "standard_target_predictor": predictor_name != "YOLOEVPSegPredictor",
        "profile_stays_ready": model.has_visual_prompt(),
    }
    report = {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "profile": profile.name,
        "metadata_path": str(metadata_path.resolve()),
        "embedding_path": str((PROFILE_DIR / "embedding.npz").resolve()),
        "vpe_shape": list(model.visual_prompt_vpe.shape),
        "vpe_dtype": str(model.visual_prompt_vpe.dtype),
        "vpe_device": str(model.visual_prompt_vpe.device),
        "target_b_detection_count": len(target_b.shapes),
        "target_c_detection_count": len(target_c.shapes),
        "target_predictor": predictor_name,
        "generation_seconds": generation_seconds,
        "profile_load_seconds": load_seconds,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "phase5_profile_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    model.unload()
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
