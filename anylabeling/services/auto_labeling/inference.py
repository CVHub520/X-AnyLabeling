"""Headless access to X-AnyLabeling auto-labeling models."""

from __future__ import annotations

import copy
import importlib.resources as resources
import importlib.util
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

import anylabeling.configs as model_configs_package
from anylabeling.services.auto_labeling import (
    _AUTO_LABELING_CONF_MODELS,
    _AUTO_LABELING_MARKS_MODELS,
    _BATCH_PROCESSING_TEXT_PROMPT_MODELS,
)


SEGMENTATION_MODEL_TYPES = {
    "edge_sam",
    "efficientvit_sam",
    "geco",
    "grounding_sam",
    "grounding_sam2",
    "open_vision",
    "sam_hq",
    "sam_med2d",
    "segment_anything",
    "segment_anything_2",
    "segment_anything_2_video",
    "segment_anything_3",
    "yolov5_sam",
    "yolov8_sam2",
}
SEGMENTATION_MODEL_TYPES.update(
    {
        "dfine_seg",
        "rfdetr_seg",
        "yolov5_seg",
        "yolov8_seg",
        "yolo11_seg",
        "yolo12_seg",
        "yolo26_seg",
        "yolov8_seg_track",
        "yolo11_seg_track",
        "yolo26_seg_track",
    }
)
TEXT_PROMPT_MODEL_TYPES = set(_BATCH_PROCESSING_TEXT_PROMPT_MODELS) | {
    "florence2",
    "grounding_sam",
    "grounding_sam2",
    "grounding_dino",
    "segment_anything_3",
    "yoloe",
}
REQUIRED_TEXT_PROMPT_MODEL_TYPES = {
    "grounding_sam",
    "grounding_sam2",
    "segment_anything_3",
}
MARK_MODEL_TYPES = set(_AUTO_LABELING_MARKS_MODELS) - {
    # The current SAM3 implementation only exposes text grounding.
    "segment_anything_3",
}

_MODEL_FILE_FIELDS = (
    "model_path",
    "encoder_model_path",
    "decoder_model_path",
    "language_encoder_path",
    "encoder_model_data_path",
    "decoder_model_data_path",
    "language_encoder_data_path",
)
_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class InferenceError(RuntimeError):
    """Raised when model inference cannot be prepared or executed."""


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping and reject other YAML root values."""
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise InferenceError(f"Expected a YAML mapping in {path}")
    return value


def _read_user_config(work_dir: Path, config_arg: str | None) -> dict[str, Any]:
    """Read custom-model settings without creating or changing user config."""
    if config_arg:
        candidate = Path(config_arg).expanduser()
        if candidate.is_file():
            return _read_yaml(candidate)
        try:
            value = yaml.safe_load(config_arg)
        except yaml.YAMLError:
            value = None
        if isinstance(value, dict):
            return value
        raise InferenceError(
            "--config must be an existing YAML file or a YAML mapping"
        )

    config_path = work_dir / ".xanylabelingrc"
    if config_path.is_file():
        return _read_yaml(config_path)
    return {}


def _built_in_entries() -> list[dict[str, Any]]:
    """Load built-in model configs using the same registry resources as XAL."""
    models_path = resources.files(model_configs_package).joinpath("models.yaml")
    with models_path.open("r", encoding="utf-8") as stream:
        entries = yaml.safe_load(stream) or []

    result = []
    for entry in entries:
        config_ref = entry.get("config_file", "")
        if not config_ref.startswith(":/"):
            continue
        config_path = resources.files(model_configs_package).joinpath(
            "auto_labeling", config_ref[2:]
        )
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        config["config_file"] = config_ref
        config["model_id"] = entry.get("model_name", config.get("name"))
        config["is_custom_model"] = False
        result.append(config)
    return result


def get_model_catalog(
    work_dir: Path, config_arg: str | None = None
) -> list[dict[str, Any]]:
    """Return built-in and valid custom models without loading model weights."""
    models = _built_in_entries()
    user_config = _read_user_config(work_dir, config_arg)
    for custom in user_config.get("custom_models", []):
        if not isinstance(custom, dict):
            continue
        config_path = Path(str(custom.get("config_file", ""))).expanduser()
        if not config_path.is_file():
            continue
        try:
            config = _read_yaml(config_path)
        except (OSError, InferenceError, yaml.YAMLError):
            continue
        if not config.get("name") or not config.get("type"):
            continue
        config["config_file"] = str(config_path.resolve())
        config["model_id"] = str(config["name"])
        config["is_custom_model"] = True
        models.append(config)
    return models


def model_capabilities(model_config: dict[str, Any]) -> dict[str, Any]:
    """Describe CLI prompt modes supported by the model implementation."""
    model_type = str(model_config.get("type", ""))
    return {
        "segmentation": model_type in SEGMENTATION_MODEL_TYPES
        or model_type.endswith("_seg"),
        "text_prompt": model_type in TEXT_PROMPT_MODEL_TYPES,
        "text_prompt_required": model_type
        in REQUIRED_TEXT_PROMPT_MODEL_TYPES,
        "marks": model_type in MARK_MODEL_TYPES,
        "confidence": model_type in _AUTO_LABELING_CONF_MODELS,
    }


def find_model(
    model_id: str,
    work_dir: Path,
    config_arg: str | None = None,
    custom_config: Path | None = None,
) -> dict[str, Any]:
    """Resolve a model by registry ID or a caller-provided YAML file."""
    if custom_config is not None:
        path = custom_config.expanduser().resolve()
        if not path.is_file():
            raise InferenceError(f"Model config not found: {path}")
        config = _read_yaml(path)
        if not config.get("type") or not config.get("name"):
            raise InferenceError(
                "Custom model YAML must define both 'type' and 'name'"
            )
        config["config_file"] = str(path)
        config["model_id"] = str(config["name"])
        config["is_custom_model"] = True
        return config

    matches = [
        model
        for model in get_model_catalog(work_dir, config_arg)
        if model.get("model_id") == model_id
    ]
    if not matches:
        raise InferenceError(f"Unknown model ID: {model_id}")
    return matches[0]


def _local_model_path(
    field_value: str, model_config: dict[str, Any], work_dir: Path
) -> Path | None:
    """Resolve a model asset path without triggering a download."""
    parsed = urlparse(field_value)
    if parsed.scheme in {"http", "https"}:
        filename = Path(parsed.path).name
        model_name = str(model_config.get("name", ""))
        for data_dir in ("xanylabeling_data", "anylabeling_data"):
            candidate = work_dir / data_dir / "models" / model_name / filename
            if candidate.is_file():
                return candidate.resolve()
        return None

    candidate = Path(field_value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    config_ref = str(model_config.get("config_file", ""))
    if config_ref.startswith(":/"):
        return None
    config_parent = Path(config_ref).parent
    relative_candidate = (config_parent / candidate).resolve()
    return relative_candidate if relative_candidate.is_file() else None


def inspect_model(
    model_config: dict[str, Any], work_dir: Path, device: str = "auto"
) -> dict[str, Any]:
    """Check model runtime dependencies, local assets, and device availability."""
    from anylabeling.views.common.device_manager import device_manager

    model_type = str(model_config.get("type", ""))
    dependency_names = ["PyQt6", "PIL", "cv2", "numpy"]
    module_names = ["PyQt6", "PIL", "cv2", "numpy"]
    if model_type not in {"remote_server", "grounding_dino_api"}:
        dependency_names.append("onnxruntime")
        module_names.append("onnxruntime")
    dependencies = {
        name: importlib.util.find_spec(module) is not None
        for name, module in zip(dependency_names, module_names)
    }

    model_assets = {}
    for field in _MODEL_FILE_FIELDS:
        value = model_config.get(field)
        if not isinstance(value, str):
            continue
        path = _local_model_path(value, model_config, work_dir)
        model_assets[field] = {
            "configured": value,
            "local_path": str(path) if path else None,
            "available": path is not None,
            "requires_download": value.startswith(("http://", "https://"))
            and path is None,
        }

    requested_device = device.lower()
    if requested_device not in {"auto", "cpu", "gpu"}:
        raise InferenceError("Device must be one of: auto, cpu, gpu")
    available_devices = [item.lower() for item in device_manager.get_available_devices()]
    selected_device = (
        device_manager.get_device()
        if requested_device == "auto"
        else requested_device.upper()
    )
    device_available = selected_device.lower() in available_devices
    return {
        "model_id": model_config.get("model_id"),
        "model_type": model_type,
        "display_name": model_config.get("display_name"),
        "config_file": model_config.get("config_file"),
        "is_custom_model": bool(model_config.get("is_custom_model", False)),
        "capabilities": model_capabilities(model_config),
        "dependencies": dependencies,
        "assets": model_assets,
        "device": {
            "requested": requested_device,
            "selected": selected_device.lower(),
            "available": device_available,
            "available_devices": available_devices,
        },
        "ready": all(dependencies.values())
        and device_available
        and all(asset["available"] for asset in model_assets.values()),
    }


def load_model(
    model_config: dict[str, Any], device: str = "auto", on_message=None
):
    """Instantiate an XAL model implementation without creating its GUI manager."""
    from PyQt6.QtCore import QCoreApplication

    from anylabeling.services.auto_labeling.model_manager import _get_model_class
    from anylabeling.views.common.device_manager import device_manager

    if QCoreApplication.instance() is None:
        raise InferenceError("Headless Qt core application is not initialized")
    requested_device = device.lower()
    if requested_device not in {"auto", "cpu", "gpu"}:
        raise InferenceError("Device must be one of: auto, cpu, gpu")
    if requested_device == "auto":
        device_manager.reset_device_preference()
    elif requested_device == "gpu":
        if "GPU" not in device_manager.get_available_devices():
            raise InferenceError(
                "GPU was requested but X-AnyLabeling reports no available GPU provider"
            )
        device_manager.set_device("GPU")
    else:
        device_manager.set_device("CPU")

    config = copy.deepcopy(model_config)
    model_type = str(config.get("type", ""))
    try:
        model_class = _get_model_class(model_type)
        callback = on_message or (lambda message: None)
        return model_class(config, on_message=callback)
    except Exception as error:
        raise InferenceError(
            f"Could not load model {model_config.get('model_id')}: {error}"
        ) from error


def predict_shapes(
    model: Any,
    image: Any,
    image_path: Path,
    prompts: list[str],
    marks: list[dict[str, Any]] | None,
    confidence: float | None,
) -> list[Any]:
    """Run XAL model prediction and return the native XAL shape objects."""
    model_type = str(getattr(model, "config", {}).get("type", ""))
    capabilities = model_capabilities({"type": model_type})
    if prompts and not capabilities["text_prompt"]:
        raise InferenceError(
            f"Model type {model_type!r} does not support text prompts"
        )
    if marks and not capabilities["marks"]:
        raise InferenceError(f"Model type {model_type!r} does not support marks")
    if confidence is not None:
        if not capabilities["confidence"]:
            raise InferenceError(
                f"Model type {model_type!r} does not expose a confidence threshold"
            )
        if not 0.0 <= confidence <= 1.0:
            raise InferenceError("Confidence must be between 0 and 1")
        model.set_auto_labeling_conf(confidence)
    if marks:
        model.set_auto_labeling_marks(marks)

    import inspect

    try:
        accepts_text_prompt = "text_prompt" in inspect.signature(
            model.predict_shapes
        ).parameters
    except (TypeError, ValueError):
        accepts_text_prompt = False
    if prompts and not accepts_text_prompt:
        raise InferenceError(
            f"Model type {model_type!r} does not accept a text_prompt argument"
        )

    call_prompts: list[str | None] = prompts or [None]
    output_shapes = []
    for prompt in call_prompts:
        if accepts_text_prompt:
            result = model.predict_shapes(
                image, str(image_path), text_prompt=prompt
            )
        else:
            result = model.predict_shapes(image, str(image_path))
        shapes = getattr(result, "shapes", result)
        if shapes is None:
            raise InferenceError(
                f"Model {model_type!r} returned no prediction result"
            )
        for shape in shapes:
            try:
                shape._xanylabeling_cli_prompt = prompt
            except Exception:
                pass
            output_shapes.append(shape)
    return output_shapes


def supported_image(path: Path) -> bool:
    """Return whether a path uses an image format supported by the CLI."""
    return path.suffix.lower() in _IMAGE_EXTENSIONS


def model_weight_paths(
    model_config: dict[str, Any], work_dir: Path
) -> dict[str, dict[str, str | None]]:
    """Return local model-weight paths and hashes for run provenance."""
    import hashlib

    result: dict[str, dict[str, str | None]] = {}
    max_hash_bytes = 512 * 1024 * 1024
    for field in _MODEL_FILE_FIELDS:
        value = model_config.get(field)
        if not isinstance(value, str):
            continue
        path = _local_model_path(value, model_config, work_dir)
        digest = None
        reason = "not available locally"
        if path is not None:
            if path.stat().st_size > max_hash_bytes:
                reason = f"file exceeds {max_hash_bytes} byte hashing limit"
            else:
                hasher = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
                reason = None
        result[field] = {
            "path": str(path) if path else None,
            "sha256": digest,
            "reason": reason,
        }
    return result
