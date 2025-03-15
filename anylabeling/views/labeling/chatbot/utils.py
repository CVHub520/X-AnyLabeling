import json
import os
from openai import OpenAI

from anylabeling.views.labeling.chatbot.config import MODELS_CONFIG_PATH, PROVIDERS_CONFIG_PATH
from anylabeling.views.labeling.logger import logger


def get_models_data(provider: str, base_url: str, api_key: str) -> dict:
    """Get models data from the API

    Args:
        provider: Provider name (custom, deepseek, ollama, qwen, etc)
        base_url: Base URL for the API
        api_key: API key for authentication

    Returns:
        dict: Models data
    """
    provider_display_name_map = {
        "deepseek": "DeepSeek",
        "ollama": "Ollama",
        "qwen": "Qwen",
    }

    supported_vision_models = [
        "bakllava",
        "granite3.2-vision",
        "minicpm-v",
        "moondream",
        "llava",
        "llava-llama3",
        "llava-phi3",
        "llama3.2-vision",
    ]

    config_path = MODELS_CONFIG_PATH
    if not os.path.exists(config_path):
        total_data = {"models_data": {}, "supported_vision_models": supported_vision_models}
    else:
        with open(config_path, "r") as f:
            total_data = json.load(f)

    provider_display_name = provider_display_name_map.get(provider, provider)
    if provider_display_name in total_data["models_data"]:
        return total_data["models_data"]

    try:
        models_id_list = get_models_id_list(base_url, api_key, timeout=5)

        if provider_display_name not in total_data["models_data"]:
            total_data["models_data"][provider_display_name] = {}
        models_data = total_data["models_data"][provider_display_name]

        for model_id in models_id_list: 
            if model_id not in models_data:
                is_vision = any(vision_model in model_id.lower() 
                                for vision_model in supported_vision_models)
                models_data[model_id] = dict(vision=is_vision, selected=False, favorite=False)
        total_data["models_data"][provider_display_name] = models_data

    except Exception as e:
        logger.debug(f"Error updating models: {e}")

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(total_data, f, indent=4)

    return total_data["models_data"]


def get_models_id_list(base_url: str, api_key: str, **kwargs) -> list:
    """Get models id list from the API"""
    client = OpenAI(base_url=base_url, api_key=api_key, **kwargs)
    return [model.id for model in client.models.list()]


def get_selected_model_id() -> str:
    """Get the selected model id"""
    selected_model_id = "Select Model"

    if not os.path.exists(MODELS_CONFIG_PATH):
        return selected_model_id

    with open(MODELS_CONFIG_PATH, "r") as f:
        total_data = json.load(f)

    for model_items in total_data["models_data"].values():
        for model_id, model_data in model_items.items():
            if model_data["selected"]:
                selected_model_id = model_id
                break

    return selected_model_id


def get_providers_data() -> dict:
    """Get the providers configs"""
    default_providers_data = {
        "deepseek": {
            "api_address": "https://api.deepseek.com/v1",
            "api_key" : None,
            "api_key_url": "https://platform.deepseek.com/api_keys",
            "api_docs_url": "https://platform.deepseek.com/docs",
            "model_docs_url": "https://platform.deepseek.com/models"
        },
        "ollama": {
            "api_address": "http://localhost:11434/v1",
            "api_key": "ollama",
            "api_key_url": None,
            "api_docs_url": "https://github.com/ollama/ollama/blob/main/docs/api.md",
            "model_docs_url": "https://ollama.com/search"
        },
        "qwen": {
            "api_address": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": None,
            "api_key_url": "https://bailian.console.aliyun.com/?apiKey=1#/api-key",
            "api_docs_url": "https://help.aliyun.com/document_detail/2590237.html",
            "model_docs_url": "https://help.aliyun.com/document_detail/2590257.html"
        },
    }

    if not os.path.exists(PROVIDERS_CONFIG_PATH):
        save_providers_data(default_providers_data)
        return default_providers_data

    with open(PROVIDERS_CONFIG_PATH, "r") as f:
        custom_providers_data = json.load(f)

    for provider, provider_data in custom_providers_data.items():
        if provider not in custom_providers_data:
            custom_providers_data[provider] = provider_data

    return custom_providers_data


def save_providers_data(providers_data: dict):
    """Save the providers data"""
    with open(PROVIDERS_CONFIG_PATH, "w") as f:
        json.dump(providers_data, f, indent=4)


def set_icon_path(icon_name: str) -> str:
    """Set the path to the icon"""
    return f"anylabeling/resources/icons/{icon_name}.svg"
