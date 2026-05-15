import json
import os
import threading
import time
import urllib.error
import urllib.request

from openai import OpenAI

from anylabeling.views.labeling.chatbot.config import *
from anylabeling.views.labeling.chatbot.utils import (
    EventTracker,
    load_json,
    save_json,
)
from anylabeling.views.labeling.logger import logger

api_call_tracker = EventTracker()

VISION_INPUT_TOKENS = {
    "image",
    "images",
    "vision",
    "visual",
    "video",
}

VISION_NAME_HINTS = (
    "bakllava",
    "claude-3",
    "claude-haiku-4",
    "claude-opus-4",
    "claude-sonnet-4",
    "gemini",
    "gemma3",
    "gpt-4.1",
    "gpt-4o",
    "granite-vision",
    "llama3.2-vision",
    "llava",
    "minicpm-v",
    "mistral-small-3.2",
    "mistral-medium-3.1",
    "mistral-medium-2508",
    "mistral-small-2506",
    "mistral-small-2503",
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-small",
    "moondream",
    "multimodal",
    "nemotron-nano-12b-v2-vl",
    "pixtral",
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "vision",
)

NON_VISION_INPUT_HINTS = (
    "asr",
    "audio-preview",
    "dall-e",
    "embedding",
    "gpt-image",
    "image-edit",
    "image-generation",
    "rerank",
    "speech",
    "text-embedding",
    "tts",
    "whisper",
)


def init_model_config():
    """Initialize the model config"""
    models_config_path = get_models_config_path()
    if not os.path.exists(models_config_path):
        model_config = dict(
            settings=DEFAULT_SETTINGS,
            models_data={},
            supported_vision_models=SUPPORTED_VISION_MODELS,
        )
        save_json(model_config, models_config_path)

    model_config = load_json(models_config_path)
    return model_config["settings"]


def get_models_data(provider: str, base_url: str, api_key: str) -> dict:
    """Get models data from the API

    Args:
        provider: Provider name (custom, deepseek, ollama, qwen, etc)
        base_url: Base URL for the API
        api_key: API key for authentication

    Returns:
        dict: Models data
    """
    config_path = get_models_config_path()
    total_data = load_json(config_path)

    api_call_tracker.increment(provider)
    if time.time() - api_call_tracker.timer[provider] > REFRESH_INTERVAL:
        api_call_tracker.reset(provider)
        api_call_tracker.increment(provider)

    call_times = api_call_tracker.get_count(provider)
    if (
        call_times > 1
        and provider.lower() != "ollama"
        and provider in total_data["models_data"]
    ):
        return total_data["models_data"]

    if provider not in total_data["models_data"]:
        total_data["models_data"][provider] = {}

    if not total_data["models_data"][provider]:
        try:
            total_data = _refresh_models_data(
                provider, base_url, api_key, config_path
            )
        except Exception as e:
            logger.debug(f"Error updating models: {e}")
        if provider not in total_data["models_data"]:
            total_data["models_data"][provider] = {}
        return total_data["models_data"]

    thread = threading.Thread(
        target=fetch_models_async,
        args=(provider, base_url, api_key, config_path),
    )
    thread.daemon = True
    thread.start()

    return total_data["models_data"]


def fetch_models_async(provider_display_name, base_url, api_key, config_path):
    """Fetch models data asynchronously"""
    try:
        _refresh_models_data(
            provider_display_name, base_url, api_key, config_path
        )
    except Exception as e:
        logger.debug(f"Error updating models: {e}")


def _refresh_models_data(
    provider_display_name, base_url, api_key, config_path
):
    total_data = load_json(config_path)
    supported_vision_models = _get_supported_vision_models(total_data)
    raw_models = get_models_raw_data(base_url, api_key, timeout=5)
    models_by_id = {
        model["id"]: model for model in raw_models if model.get("id")
    }
    models_id_list = list(models_by_id)

    if provider_display_name not in total_data["models_data"]:
        total_data["models_data"][provider_display_name] = {}
    models_data = total_data["models_data"][provider_display_name]

    models_to_remove = [
        model_id for model_id in models_data if model_id not in models_id_list
    ]
    for model_id in models_to_remove:
        del models_data[model_id]

    for model_id in models_id_list:
        is_vision = model_supports_vision(
            model_id, models_by_id.get(model_id, {}), supported_vision_models
        )
        if model_id not in models_data:
            models_data[model_id] = dict(
                vision=is_vision, selected=False, favorite=False
            )
        else:
            models_data[model_id]["vision"] = is_vision
            models_data[model_id].setdefault("selected", False)
            models_data[model_id].setdefault("favorite", False)
    total_data["models_data"][provider_display_name] = models_data

    save_json(total_data, config_path)
    return total_data


def _get_supported_vision_models(total_data: dict) -> list:
    supported_vision_models = []
    for vision_model in total_data.get("supported_vision_models", []):
        if vision_model not in supported_vision_models:
            supported_vision_models.append(vision_model)
    for vision_model in SUPPORTED_VISION_MODELS:
        if vision_model not in supported_vision_models:
            supported_vision_models.append(vision_model)
    total_data["supported_vision_models"] = supported_vision_models
    return supported_vision_models


def _model_to_dict(model) -> dict:
    if isinstance(model, dict):
        return model
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(getattr(model, "__dict__", {}))


def get_models_raw_data(base_url: str, api_key: str, timeout: int = 5) -> list:
    """Get raw model data from the API."""
    if "anthropic" in base_url:
        endpoint = base_url.rstrip("/") + "/models"
        request = urllib.request.Request(
            endpoint,
            headers={
                "x-api-key": (api_key or "").strip(),
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("data", [])
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            logger.error(
                f"Anthropic models fetch failed: HTTP {e.code} {body}"
            )
            return []
        except Exception as e:
            logger.error(
                f"Anthropic models fetch failed: {type(e).__name__}: {e}"
            )
            return []

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    return [_model_to_dict(model) for model in client.models.list()]


def get_models_id_list(base_url: str, api_key: str, timeout: int = 5) -> list:
    """Get models id list from the API"""
    return [
        model["id"]
        for model in get_models_raw_data(base_url, api_key, timeout)
        if model.get("id")
    ]


def model_supports_vision(
    model_id: str, model_metadata: dict, supported_vision_models: list
) -> bool:
    """Determine whether a model supports vision input."""
    metadata_result = _metadata_supports_vision(model_metadata)
    if metadata_result is not None:
        return metadata_result

    model_id_lower = model_id.lower()
    if any(
        vision_model.lower() in model_id_lower
        for vision_model in supported_vision_models
    ):
        return True

    return _model_name_suggests_vision(model_id_lower)


def _metadata_supports_vision(model_metadata: dict):
    if not model_metadata:
        return None

    explicit = _explicit_vision_value(model_metadata)
    if explicit is not None:
        return explicit

    architecture = model_metadata.get("architecture")
    if isinstance(architecture, dict):
        result = _metadata_supports_vision(architecture)
        if result is not None:
            return result

    for key in ("input_modalities", "input_types", "inputs"):
        result = _modalities_support_vision(model_metadata.get(key))
        if result is not None:
            return result

    modalities = model_metadata.get("modalities")
    if isinstance(modalities, dict):
        for key in ("input", "inputs", "input_modalities"):
            result = _modalities_support_vision(modalities.get(key))
            if result is not None:
                return result
    else:
        result = _modalities_support_vision(modalities)
        if result is not None:
            return result

    modality = model_metadata.get("modality")
    if isinstance(modality, str):
        return _modality_string_supports_vision(modality)

    return None


def _explicit_vision_value(model_metadata: dict):
    for key in ("vision", "supports_vision", "multimodal"):
        value = model_metadata.get(key)
        if isinstance(value, bool):
            return value

    capabilities = model_metadata.get("capabilities")
    if isinstance(capabilities, dict):
        for key in ("vision", "image", "images", "video", "multimodal"):
            value = capabilities.get(key)
            if isinstance(value, bool):
                return value

    return None


def _modalities_support_vision(modalities):
    if modalities is None:
        return None

    if isinstance(modalities, str):
        modality_tokens = {modalities.lower()}
    else:
        try:
            modality_tokens = {str(item).lower() for item in modalities}
        except TypeError:
            return None

    return bool(VISION_INPUT_TOKENS.intersection(modality_tokens))


def _modality_string_supports_vision(modality: str):
    normalized = modality.lower()
    if "->" in normalized:
        input_part = normalized.split("->", 1)[0]
        return any(token in input_part for token in VISION_INPUT_TOKENS)
    return None


def _model_name_suggests_vision(model_id_lower: str) -> bool:
    if any(token in model_id_lower for token in NON_VISION_INPUT_HINTS):
        return False
    return any(token in model_id_lower for token in VISION_NAME_HINTS)


def get_default_model_id(provider: str) -> str:
    """Get the default model id"""
    default_model_id = "Select Model"

    models_config_path = get_models_config_path()
    if not os.path.exists(models_config_path):
        return default_model_id

    model_config = load_json(models_config_path)

    if model_config["settings"]["model_id"]:
        return model_config["settings"]["model_id"] + f" ({provider})"

    return default_model_id


def get_providers_data() -> dict:
    """Get the providers configs"""
    default_providers_data = DEFAULT_PROVIDERS_DATA
    providers_config_path = get_providers_config_path()

    if not os.path.exists(providers_config_path):
        save_json(default_providers_data, providers_config_path)
        return default_providers_data

    custom_providers_data = load_json(providers_config_path)
    for provider, provider_data in default_providers_data.items():
        if provider not in custom_providers_data:
            custom_providers_data[provider] = provider_data

    return custom_providers_data
