import json
import os
import threading
import time
from openai import OpenAI

from anylabeling.views.labeling.chatbot.config import (
    MODELS_CONFIG_PATH,
    PROVIDERS_CONFIG_PATH,
    REFRESH_INTERVAL,
    DEFAULT_PROVIDERS_DATA,
    SUPPORTED_VISION_MODELS,
)
from anylabeling.views.labeling.logger import logger


class ApiCallTracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ApiCallTracker, cls).__new__(cls)
            cls._instance.counters = {}
            cls._instance.timer = {}
        return cls._instance

    def increment(self, counter_name):
        if counter_name not in self.counters:
            self.counters[counter_name] = 0
        self.counters[counter_name] += 1
        if counter_name not in self.timer:
            self.timer[counter_name] = time.time()
        return self.counters[counter_name]

    def get_count(self, counter_name):
        return self.counters.get(counter_name, 0)

    def get_all_counts(self):
        return self.counters.copy()

    def reset(self, counter_name=None):
        if counter_name is None:
            self.counters = {}
            self.timer = {}
        elif counter_name in self.counters:
            self.counters[counter_name] = 0
            self.timer[counter_name] = 0


api_call_tracker = ApiCallTracker()


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
    if not os.path.exists(config_path):
        total_data = {"models_data": {}, "supported_vision_models": SUPPORTED_VISION_MODELS}
    else:
        with open(config_path, "r") as f:
            total_data = json.load(f)

    api_call_tracker.increment(provider)
    if time.time() - api_call_tracker.timer[provider] > REFRESH_INTERVAL:
        api_call_tracker.reset(provider)
        api_call_tracker.increment(provider)

    call_times = api_call_tracker.get_count(provider)
    if call_times > 1 and \
        provider != "Ollama" and \
        provider in total_data["models_data"]:
        return total_data["models_data"]

    thread = threading.Thread(
        target=fetch_models_async,
        args=(provider, base_url, api_key, total_data, config_path)
    )
    thread.daemon = True
    thread.start()

    if provider not in total_data["models_data"]:
        total_data["models_data"][provider] = {}

    return total_data["models_data"]


def fetch_models_async(provider_display_name, base_url, api_key, total_data, config_path):
    """Fetch models data asynchronously"""
    try:
        supported_vision_models = total_data["supported_vision_models"]
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

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(total_data, f, indent=4)
            
    except Exception as e:
        logger.debug(f"Error updating models: {e}")


def get_models_id_list(base_url: str, api_key: str, timeout: int = 5) -> list:
    """Get models id list from the API"""
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
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
    default_providers_data = DEFAULT_PROVIDERS_DATA

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
