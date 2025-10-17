import json
import os
import subprocess
import threading
import time

from openai import OpenAI

from anylabeling.views.labeling.chatbot.config import *
from anylabeling.views.labeling.chatbot.utils import (
    EventTracker,
    load_json,
    save_json,
)
from anylabeling.views.labeling.logger import logger

api_call_tracker = EventTracker()


def init_model_config():
    """Initialize the model config"""
    if not os.path.exists(MODELS_CONFIG_PATH):
        model_config = dict(
            settings=DEFAULT_SETTINGS,
            models_data={},
            supported_vision_models=SUPPORTED_VISION_MODELS,
        )
        save_json(model_config, MODELS_CONFIG_PATH)

    model_config = load_json(MODELS_CONFIG_PATH)
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
    config_path = MODELS_CONFIG_PATH
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

    thread = threading.Thread(
        target=fetch_models_async,
        args=(provider, base_url, api_key, total_data, config_path),
    )
    thread.daemon = True
    thread.start()

    if provider not in total_data["models_data"]:
        total_data["models_data"][provider] = {}

    return total_data["models_data"]


def fetch_models_async(
    provider_display_name, base_url, api_key, total_data, config_path
):
    """Fetch models data asynchronously"""
    try:
        supported_vision_models = total_data["supported_vision_models"]
        models_id_list = get_models_id_list(base_url, api_key, timeout=5)

        if provider_display_name not in total_data["models_data"]:
            total_data["models_data"][provider_display_name] = {}
        models_data = total_data["models_data"][provider_display_name]

        models_to_remove = [
            model_id
            for model_id in models_data
            if model_id not in models_id_list
        ]
        for model_id in models_to_remove:
            del models_data[model_id]

        for model_id in models_id_list:
            if model_id not in models_data:
                is_vision = any(
                    vision_model in model_id.lower()
                    for vision_model in supported_vision_models
                )
                models_data[model_id] = dict(
                    vision=is_vision, selected=False, favorite=False
                )
        total_data["models_data"][provider_display_name] = models_data

        save_json(total_data, config_path)

    except Exception as e:
        logger.debug(f"Error updating models: {e}")


def get_models_id_list(base_url: str, api_key: str, timeout: int = 5) -> list:
    """Get models id list from the API"""
    if "anthropic" in base_url:
        url = f'curl https://api.anthropic.com/v1/models --header "x-api-key:{api_key}" --header "anthropic-version: 2023-06-01"'
        response = subprocess.run(
            url, shell=True, capture_output=True, text=True
        )
        try:
            response_data = json.loads(response.stdout)
            return [model["id"] for model in response_data.get("data", [])]
        except Exception as e:
            logger.debug(f"Response: {response.stdout}")
            return []
    else:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        return [model.id for model in client.models.list()]


def get_default_model_id(provider: str) -> str:
    """Get the default model id"""
    default_model_id = "Select Model"

    if not os.path.exists(MODELS_CONFIG_PATH):
        return default_model_id

    model_config = load_json(MODELS_CONFIG_PATH)

    if model_config["settings"]["model_id"]:
        return model_config["settings"]["model_id"] + f" ({provider})"

    return default_model_id


def get_providers_data() -> dict:
    """Get the providers configs"""
    default_providers_data = DEFAULT_PROVIDERS_DATA

    if not os.path.exists(PROVIDERS_CONFIG_PATH):
        save_json(default_providers_data, PROVIDERS_CONFIG_PATH)
        return default_providers_data

    custom_providers_data = load_json(PROVIDERS_CONFIG_PATH)
    for provider, provider_data in custom_providers_data.items():
        if provider not in custom_providers_data:
            custom_providers_data[provider] = provider_data

    for provider, provider_data in default_providers_data.items():
        if provider not in custom_providers_data:
            custom_providers_data[provider] = provider_data

    return custom_providers_data
