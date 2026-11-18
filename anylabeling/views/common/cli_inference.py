"""Command handlers for headless auto-labeling inference."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PyQt6 import QtGui
from PyQt6.QtCore import QCoreApplication

from anylabeling.app_info import __appname__, __version__
from anylabeling.services.auto_labeling.inference import (
    InferenceError,
    find_model,
    get_model_catalog,
    inspect_model,
    load_model,
    model_capabilities,
    model_weight_paths,
    predict_shapes,
    supported_image,
)


class CLIError(RuntimeError):
    """Raised for user-facing CLI input or output errors."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _json_print(value: dict[str, Any]) -> None:
    """Write one machine-readable JSON document to stdout."""
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _human_or_json(value: dict[str, Any], as_json: bool) -> None:
    """Write either JSON or a compact text representation."""
    if as_json:
        _json_print(value)
        return
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _work_dir(args: Any) -> Path:
    """Get the X-AnyLabeling working directory from parsed arguments."""
    return Path(args.work_dir).expanduser().resolve()


def _catalog_for_args(args: Any) -> list[dict[str, Any]]:
    """Read configured model metadata without loading weights."""
    return get_model_catalog(_work_dir(args), getattr(args, "config", None))


def list_models(args: Any) -> int:
    """List registry-backed models and their CLI capabilities."""
    try:
        models = _catalog_for_args(args)
        if getattr(args, "task", None) == "segmentation":
            models = [
                model
                for model in models
                if model_capabilities(model)["segmentation"]
            ]
        result = {
            "schema": "xanylabeling.cli-models.v1",
            "models": [
                {
                    "id": model.get("model_id"),
                    "name": model.get("display_name"),
                    "type": model.get("type"),
                    "config": model.get("config_file"),
                    "custom": bool(model.get("is_custom_model", False)),
                    "capabilities": model_capabilities(model),
                }
                for model in models
            ],
        }
        if args.json:
            _json_print(result)
        else:
            for model in result["models"]:
                caps = ", ".join(
                    name
                    for name, enabled in model["capabilities"].items()
                    if enabled
                ) or "no segmentation modes"
                print(f"{model['id']}\t{model['name']}\t{caps}")
            print(f"Total: {len(models)} model(s)")
        return 0
    except (InferenceError, OSError, ValueError) as error:
        _human_or_json(
            {"schema": "xanylabeling.cli-models.v1", "status": "failed", "error": str(error)},
            args.json,
        )
        return 2


def model_info(args: Any) -> int:
    """Show registry metadata and CLI capabilities for one model."""
    try:
        model = find_model(args.model_id, _work_dir(args), args.config)
        result = {
            "schema": "xanylabeling.cli-model-info.v1",
            "model": {
                "id": model.get("model_id"),
                "name": model.get("display_name"),
                "type": model.get("type"),
                "config": model.get("config_file"),
                "custom": bool(model.get("is_custom_model", False)),
                "capabilities": model_capabilities(model),
                "parameters": {
                    key: value
                    for key, value in model.items()
                    if key.endswith("threshold")
                    or key in {"epsilon", "input_width", "input_height"}
                },
            },
        }
        _human_or_json(result, args.json)
        return 0
    except (InferenceError, OSError, ValueError) as error:
        _human_or_json(
            {"schema": "xanylabeling.cli-model-info.v1", "status": "failed", "error": str(error)},
            args.json,
        )
        return 2


def check_model(args: Any) -> int:
    """Check model dependencies, weights, and selected device without loading it."""
    try:
        model = find_model(args.model_id, _work_dir(args), args.config)
        result = inspect_model(model, _work_dir(args), args.device)
        result = {"schema": "xanylabeling.cli-model-check.v1", **result}
        _human_or_json(result, args.json)
        return 0 if result["ready"] else 3
    except (InferenceError, OSError, ValueError) as error:
        _human_or_json(
            {"schema": "xanylabeling.cli-model-check.v1", "ready": False, "error": str(error)},
            args.json,
        )
        return 3


def _initialize_qt() -> QCoreApplication:
    """Create the minimum Qt application required by XAL model classes."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([sys.argv[0]])
    return app


def _collect_images(input_path: Path, recursive: bool) -> list[Path]:
    """Collect a single image or image files in stable path order."""
    if input_path.is_file():
        if not supported_image(input_path):
            raise CLIError(2, f"Unsupported image format: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise CLIError(2, f"Input path does not exist: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    images = [path for path in iterator if path.is_file() and supported_image(path)]
    images.sort(key=lambda path: str(path.relative_to(input_path)).casefold())
    if not images:
        raise CLIError(2, f"No supported images found in {input_path}")
    return images


def _relative_reference(target: Path, relative_to: Path) -> str:
    """Return a relative path when possible, otherwise an absolute reference."""
    try:
        return Path(os.path.relpath(target, relative_to)).as_posix()
    except ValueError:
        return str(target.resolve())


def _sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a source image."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_dict(shape: Any) -> dict[str, Any]:
    """Convert an XAL shape object into JSON-safe native geometry."""
    shape_data = shape.to_dict() if hasattr(shape, "to_dict") else dict(shape)
    shape_data["points"] = [
        [float(point[0]), float(point[1])] for point in shape_data.get("points", [])
    ]
    score = shape_data.get("score")
    if score is not None:
        shape_data["score"] = float(score)
    return shape_data


def _model_providers(model: Any) -> list[str] | None:
    """Read active ONNX providers from the loaded model when exposed."""
    providers: set[str] = set()
    candidates = [model]
    for attribute in ("model", "net"):
        nested = getattr(model, attribute, None)
        if nested is not None:
            candidates.append(nested)
    for candidate in candidates:
        session = getattr(candidate, "ort_session", None)
        if session is not None and hasattr(session, "get_providers"):
            providers.update(session.get_providers())
        providers.update(getattr(candidate, "providers", []) or [])
    return sorted(providers) or None


def _outputs_for_image(
    output_root: Path, input_root: Path | None, image_path: Path
) -> dict[str, Path]:
    """Build stable output paths while preserving batch-relative directories."""
    relative = image_path.name if input_root is None else str(image_path.relative_to(input_root))
    relative_path = Path(relative)
    annotation = output_root / "annotations" / f"{relative}.json"
    metadata = output_root / "candidates" / f"{relative}.candidates.json"
    masks = output_root / "masks" / relative_path
    return {"annotation": annotation, "metadata": metadata, "masks": masks}


def _assert_contained(path: Path, root: Path) -> None:
    """Reject output paths that escape the requested root."""
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise CLIError(2, f"Output path escapes the requested directory: {path}")


def _save_image_result(
    image_path: Path,
    output_root: Path,
    input_root: Path | None,
    model_config: dict[str, Any],
    prompts: list[str],
    marks: list[dict[str, Any]] | None,
    device: str,
    providers: list[str] | None,
    confidence: float | None,
    shapes: list[Any],
    width: int,
    height: int,
    overwrite: bool,
    weights: dict[str, dict[str, str | None]],
) -> dict[str, Any]:
    """Atomically stage XLABEL, per-candidate masks, and provenance metadata."""
    from anylabeling.views.labeling.label_file import LabelFile
    from anylabeling.views.labeling.utils.shape import shape_to_mask

    targets = _outputs_for_image(output_root, input_root, image_path)
    for target in targets.values():
        _assert_contained(target, output_root)
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise CLIError(2, f"Output already exists: {existing[0]}; use --overwrite")

    stage_root = Path(tempfile.mkdtemp(prefix=".xal-segment-", dir=output_root))
    stage_targets = _outputs_for_image(stage_root, input_root, image_path)
    for target in stage_targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
    stage_targets["masks"].mkdir(parents=True, exist_ok=True)

    try:
        source_hash = _sha256(image_path)
        annotation_shapes = []
        candidates = []
        mask_outputs = []
        for index, shape in enumerate(shapes, start=1):
            shape_data = _shape_dict(shape)
            mask_name = f"candidate-{index:04d}.png"
            mask_stage_path = stage_targets["masks"] / mask_name
            points = shape_data.get("points", [])
            shape_type = shape_data.get("shape_type", "polygon")
            mask = shape_to_mask((height, width), points, shape_type)
            binary_mask = np.where(mask, 255, 0).astype(np.uint8)
            Image.fromarray(binary_mask, mode="L").save(mask_stage_path, format="PNG")

            candidate_id = f"candidate-{index:04d}"
            final_mask_path = targets["masks"] / mask_name
            original_description = str(shape_data.get("description") or "").strip()
            candidate_note = f"X-AnyLabeling CLI candidate ({candidate_id}); unreviewed"
            shape_data["description"] = (
                f"{original_description} | {candidate_note}"
                if original_description
                else candidate_note
            )
            annotation_shapes.append(shape_data)
            score = shape_data.get("score")
            candidates.append(
                {
                    "id": candidate_id,
                    "label": shape_data.get("label"),
                    "prompt": getattr(shape, "_xanylabeling_cli_prompt", None),
                    "score": score,
                    "shape_type": shape_type,
                    "points": shape_data.get("points", []),
                    "mask": _relative_reference(
                        final_mask_path, targets["metadata"].parent
                    ),
                    "status": "unreviewed",
                }
            )
            mask_outputs.append(str(final_mask_path.resolve()))

        source_reference = _relative_reference(
            image_path.resolve(), targets["annotation"].parent
        )
        LabelFile().save(
            filename=str(stage_targets["annotation"]),
            shapes=annotation_shapes,
            image_path=source_reference,
            image_height=height,
            image_width=width,
            image_data=None,
            other_data={
                "description": "X-AnyLabeling CLI inference candidates; unreviewed"
            },
            flags={},
        )

        config_parameters = {
            key: value
            for key, value in model_config.items()
            if key.endswith("threshold")
            or key in {"epsilon", "input_width", "input_height"}
        }
        metadata = {
            "schema": "xanylabeling.candidates.v1",
            "status": "unreviewed_candidates",
            "source": {
                "path": _relative_reference(
                    image_path.resolve(), targets["metadata"].parent
                ),
                "sha256": source_hash,
                "width": width,
                "height": height,
            },
            "application": {"name": __appname__, "version": __version__},
            "inference": {
                "model_id": model_config.get("model_id"),
                "model_type": model_config.get("type"),
                "config": model_config.get("config_file"),
                "device": device,
                "provider": providers,
                "provider_reason": (
                    None
                    if providers
                    else "loaded model does not expose its active runtime provider"
                ),
                "prompts": prompts,
                "marks": marks,
                "parameters": {
                    **config_parameters,
                    "confidence_override": confidence,
                },
                "weights": weights,
            },
            "candidates": candidates,
        }
        stage_targets["metadata"].write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        annotation_stage = stage_targets["annotation"]
        metadata_stage = stage_targets["metadata"]
        mask_stage = stage_targets["masks"]
        for target in existing:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        for staged, target in (
            (annotation_stage, targets["annotation"]),
            (metadata_stage, targets["metadata"]),
            (mask_stage, targets["masks"]),
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, target)

        return {
            "input": str(image_path.resolve()),
            "status": "success",
            "candidate_count": len(candidates),
            "outputs": {
                "annotation": str(targets["annotation"].resolve()),
                "metadata": str(targets["metadata"].resolve()),
                "masks": mask_outputs,
            },
        }
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _read_marks(path: str | None) -> list[dict[str, Any]] | None:
    """Load and validate point/rectangle marks from a JSON file."""
    if path is None:
        return None
    marks_path = Path(path).expanduser().resolve()
    if not marks_path.is_file():
        raise CLIError(2, f"Marks file not found: {marks_path}")
    try:
        data = json.loads(marks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CLIError(2, f"Could not read marks JSON: {error}") from error
    if not isinstance(data, list):
        raise CLIError(2, "Marks JSON must contain a list of marks")
    normalized = []
    for index, mark in enumerate(data):
        if not isinstance(mark, dict) or mark.get("type") not in {"point", "rectangle"}:
            raise CLIError(2, f"Mark {index} must have type point or rectangle")
        coordinates = mark.get("data")
        expected = 2 if mark["type"] == "point" else 4
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != expected
            or any(not isinstance(value, (int, float)) for value in coordinates)
        ):
            raise CLIError(2, f"Mark {index} has invalid coordinates")
        label = mark.get("label", 1)
        if label not in {0, 1}:
            raise CLIError(2, f"Mark {index} label must be 0 or 1")
        normalized.append(
            {"type": mark["type"], "data": coordinates, "label": label}
        )
    return normalized


def segment(args: Any) -> int:
    """Generate native XAL proposals, per-shape masks, and run metadata."""
    result: dict[str, Any]
    status_code = 0
    try:
        input_path = Path(args.input).expanduser().resolve()
        output_root = Path(args.output).expanduser().resolve()
        if input_path.is_dir():
            try:
                output_root.relative_to(input_path)
            except ValueError:
                pass
            else:
                raise CLIError(
                    2,
                    "Output directory cannot be inside the input directory; choose a separate output path",
                )
            input_root = input_path
        else:
            input_root = None
        images = _collect_images(input_path, args.recursive)
        if bool(args.model) == bool(args.model_config):
            raise CLIError(2, "Specify exactly one of --model or --model-config")
        model_config = find_model(
            args.model or "",
            _work_dir(args),
            args.config,
            Path(args.model_config) if args.model_config else None,
        )
        capabilities = model_capabilities(model_config)
        if not capabilities["segmentation"]:
            raise CLIError(
                2,
                f"Model type {model_config.get('type')!r} is not registered as a segmentation model",
            )
        prompts = args.prompt or []
        if prompts and not capabilities["text_prompt"]:
            raise CLIError(
                2,
                f"Model type {model_config.get('type')!r} does not support text prompts",
            )
        if capabilities["text_prompt_required"] and not prompts:
            raise CLIError(
                2,
                f"Model {args.model!r} requires at least one --prompt",
            )
        marks = _read_marks(args.marks_json)
        if marks and not capabilities["marks"]:
            raise CLIError(
                2,
                f"Model type {model_config.get('type')!r} does not support point/rectangle marks",
            )
        if args.confidence is not None and not capabilities["confidence"]:
            raise CLIError(
                2,
                f"Model type {model_config.get('type')!r} does not expose a confidence threshold",
            )
        if args.confidence is not None and not 0.0 <= args.confidence <= 1.0:
            raise CLIError(2, "--confidence must be between 0 and 1")

        output_root.mkdir(parents=True, exist_ok=True)
        for image_path in images:
            for target in _outputs_for_image(output_root, input_root, image_path).values():
                _assert_contained(target, output_root)
        if not args.overwrite:
            existing = [
                target
                for image_path in images
                for target in _outputs_for_image(
                    output_root, input_root, image_path
                ).values()
                if target.exists()
            ]
            if existing:
                raise CLIError(
                    2,
                    f"Output already exists: {existing[0]}; use --overwrite",
                )
        app = _initialize_qt()
        with redirect_stdout(sys.stderr):
            model = load_model(
                model_config,
                args.device,
                on_message=lambda message: print(message, file=sys.stderr),
            )
        from anylabeling.views.common.device_manager import device_manager

        selected_device = device_manager.get_device().lower()
        providers = _model_providers(model)
        weights = model_weight_paths(model_config, _work_dir(args))
        results = []
        try:
            for image_index, image_path in enumerate(images):
                try:
                    reader = QtGui.QImageReader(str(image_path))
                    image = reader.read()
                    if image.isNull():
                        raise CLIError(
                            2,
                            f"Could not decode image {image_path}: {reader.errorString()}",
                        )
                    width, height = image.width(), image.height()
                    with redirect_stdout(sys.stderr):
                        shapes = predict_shapes(
                            model,
                            image,
                            image_path,
                            prompts,
                            marks,
                            args.confidence,
                        )
                    result_item = _save_image_result(
                        image_path,
                        output_root,
                        input_root,
                        model_config,
                        prompts,
                        marks,
                        selected_device,
                        providers,
                        args.confidence,
                        shapes,
                        width,
                        height,
                        args.overwrite,
                        weights,
                    )
                except Exception as error:
                    if isinstance(error, CLIError):
                        code = error.code
                    elif isinstance(error, InferenceError):
                        code = 3
                    else:
                        code = 3
                    result_item = {
                        "input": str(image_path.resolve()),
                        "status": "failed",
                        "error": {"code": code, "message": str(error)},
                    }
                    status_code = max(status_code, 1 if code == 2 else code)
                results.append(result_item)
                if result_item["status"] == "failed" and args.fail_fast:
                    for pending_path in images[image_index + 1 :]:
                        results.append(
                            {
                                "input": str(pending_path.resolve()),
                                "status": "skipped",
                                "error": {
                                    "code": 1,
                                    "message": "Skipped because --fail-fast stopped the batch",
                                },
                            }
                        )
                    break
        finally:
            try:
                model.unload()
            except Exception:
                pass
            del app

        failed = [item for item in results if item["status"] == "failed"]
        result = {
            "schema": "xanylabeling.cli-result.v1",
            "status": "failed" if failed and len(failed) == len(results) else "partial" if failed else "success",
            "model_id": model_config.get("model_id"),
            "input_count": len(images),
            "completed_count": sum(item["status"] == "success" for item in results),
            "failed_count": len(failed),
            "results": results,
        }
    except Exception as error:
        code = error.code if isinstance(error, CLIError) else 3 if isinstance(error, InferenceError) else 2
        status_code = code
        result = {
            "schema": "xanylabeling.cli-result.v1",
            "status": "failed",
            "error": {"code": code, "message": str(error)},
        }

    if args.json:
        _json_print(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return status_code
