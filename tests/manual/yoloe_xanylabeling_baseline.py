"""Exercise X-AnyLabeling's existing YOLOE text and visual modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anylabeling import config as anylabeling_config
from anylabeling.services.auto_labeling.yoloe import YOLOE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = PROJECT_ROOT / ".cache" / "models"
DEFAULT_REFERENCE = (
    PROJECT_ROOT / ".cache" / "yoloe" / "ultralytics" / "assets" / "bus.jpg"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODELS / "yoloe-11s-seg.pt",
    )
    parser.add_argument(
        "--prompt-free-model",
        type=Path,
        default=DEFAULT_MODELS / "yoloe-11s-seg-pf.pt",
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=DEFAULT_MODELS / "mobileclip_blt.pt",
    )
    return parser.parse_args()


def summarize(result) -> dict:
    return {
        "shape_count": len(result.shapes),
        "labels": [shape.label for shape in result.shapes],
        "scores": [shape.score for shape in result.shapes],
        "shape_types": [shape.shape_type for shape in result.shapes],
    }


def main() -> int:
    args = parse_args()
    for path in (
        args.image,
        args.model,
        args.prompt_free_model,
        args.embedding_model,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    # The desktop entry point normally initializes this before Model creation.
    anylabeling_config.current_config_file = "{}"
    config = {
        "type": "yoloe",
        "name": "yoloe_11s-baseline",
        "display_name": "YOLOE-11-S Baseline",
        "config_file": str(PROJECT_ROOT / "tests" / "manual"),
        "model_path": str(args.model.resolve()),
        "model_pf_path": str(args.prompt_free_model.resolve()),
        "embedding_model_path": str(args.embedding_model.resolve()),
        "input_height": 640,
        "input_width": 640,
        "iou_threshold": 0.70,
        "conf_threshold": 0.25,
        "max_det": 1000,
        "with_mask": False,
    }
    model = YOLOE(config, on_message=lambda message: print(message))
    image_path = str(args.image.resolve())

    text_result = model.predict_shapes(
        image=True,
        image_path=image_path,
        text_prompt="person.car",
    )

    model.set_auto_labeling_marks(
        [
            {
                "type": "rectangle",
                "data": [221, 405, 345, 858],
                "label": 1,
            }
        ]
    )
    visual_result = model.predict_shapes(image=True, image_path=image_path)
    predictor_after_visual = type(model._visual_model.predictor).__name__

    report = {
        "text_prompt": summarize(text_result),
        "intra_image_visual_prompt": summarize(visual_result),
        "visual_marks_remaining": len(model.marks),
        "visual_predictor_after_inference": predictor_after_visual,
        "result": (
            "PASS"
            if text_result.shapes and visual_result.shapes and not model.marks
            else "FAIL"
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    model.unload()
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
