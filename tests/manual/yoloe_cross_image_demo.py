"""Validate YOLOE cross-image visual prompting with two distinct images.

This is a manual integration test. It requires the THU-MIG/yoloe fork of
Ultralytics and a YOLOE segmentation checkpoint; it never downloads or mocks
either input images or detections.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLOE_SOURCE = PROJECT_ROOT / ".cache" / "yoloe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a VPE on one image and detect on another."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "models" / "yoloe-11s-seg.pt",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=YOLOE_SOURCE / "ultralytics" / "assets" / "bus.jpg",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=YOLOE_SOURCE / "ultralytics" / "assets" / "zidane.jpg",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        action="append",
        metavar=("X1", "Y1", "X2", "Y2"),
        help=(
            "Reference box in xyxy pixels; repeat for multiple instances. "
            "Defaults to the person box from the official YOLOE demo."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "tests" / "output" / "cross_image_result.jpg",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> np.ndarray:
    for name in ("model", "reference", "target"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    reference = args.reference.resolve()
    target = args.target.resolve()
    if reference == target:
        raise ValueError("Reference and target images must be different files")

    boxes = np.asarray(
        args.bbox or [[221.52, 405.8, 344.98, 857.54]],
        dtype=np.float32,
    )
    width, height = Image.open(reference).size
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Bounding boxes must have shape [N, 4]")
    if not np.isfinite(boxes).all():
        raise ValueError("Bounding boxes must contain finite values")
    if (
        (boxes[:, 0] < 0).any()
        or (boxes[:, 1] < 0).any()
        or (boxes[:, 2] > width).any()
        or (boxes[:, 3] > height).any()
        or (boxes[:, 2] <= boxes[:, 0]).any()
        or (boxes[:, 3] <= boxes[:, 1]).any()
    ):
        raise ValueError(
            f"Bounding boxes must be valid within reference size {width}x{height}"
        )
    return boxes


def main() -> int:
    args = parse_args()
    boxes = validate_inputs(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_started = time.perf_counter()
    model = YOLOE(str(args.model.resolve()))
    model.eval()
    model.to(device)
    model_seconds = time.perf_counter() - model_started

    prompts = {
        "bboxes": boxes,
        "cls": np.zeros(len(boxes), dtype=np.int64),
    }
    vpe_started = time.perf_counter()
    model.predict(
        source=str(args.reference.resolve()),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
        prompts=prompts,
        predictor=YOLOEVPSegPredictor,
        return_vpe=True,
    )
    vpe = model.predictor.vpe.detach()
    vpe_seconds = time.perf_counter() - vpe_started

    model.set_classes(["object"], vpe)
    model.predictor = None

    target_started = time.perf_counter()
    results = model.predict(
        source=str(args.target.resolve()),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )
    target_seconds = time.perf_counter() - target_started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered_bgr = results[0].plot()
    Image.fromarray(rendered_bgr[..., ::-1]).save(args.output)

    boxes_result = results[0].boxes
    confidences = (
        boxes_result.conf.detach().cpu().tolist()
        if boxes_result is not None
        else []
    )
    xyxy = (
        boxes_result.xyxy.detach().cpu().tolist()
        if boxes_result is not None
        else []
    )
    metrics = {
        "cross_image_visual_prompt": "PASS" if confidences else "FAIL",
        "model": str(args.model.resolve()),
        "reference": str(args.reference.resolve()),
        "reference_boxes": boxes.tolist(),
        "target": str(args.target.resolve()),
        "reference_target_are_distinct": True,
        "device": str(vpe.device),
        "vpe_shape": list(vpe.shape),
        "vpe_dtype": str(vpe.dtype),
        "detection_count": len(confidences),
        "confidences": confidences,
        "detections_xyxy": xyxy,
        "model_load_seconds": model_seconds,
        "vpe_generation_seconds": vpe_seconds,
        "target_inference_seconds": target_seconds,
        "output": str(args.output.resolve()),
    }
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if confidences else 1


if __name__ == "__main__":
    raise SystemExit(main())
