import os

from anylabeling.config import get_work_directory
from anylabeling.views.labeling.utils.theme import get_theme


def get_chatbot_root_dir():
    return os.path.join(get_work_directory(), "xanylabeling_data", "chatbot")


def get_models_config_path():
    return os.path.join(get_chatbot_root_dir(), "models.json")


def get_settings_config_path():
    return os.path.join(get_chatbot_root_dir(), "settings.json")


def get_providers_config_path():
    return os.path.join(get_chatbot_root_dir(), "providers.json")


# Global design system
ANIMATION_DURATION = "200ms"
BORDER_RADIUS = "8px"
FONT_SIZE_TINY = "9px"
FONT_SIZE_SMALL = "11px"
FONT_SIZE_NORMAL = "13px"
FONT_SIZE_LARGE = "16px"
ICON_SIZE_NORMAL = (32, 32)
ICON_SIZE_SMALL = (16, 16)
MESSAGE_ACTION_BUTTON_SIZE = (20, 20)

# Initialization parameters
DEFAULT_WINDOW_TITLE = "Chatbot"
DEFAULT_WINDOW_SIZE = (1200, 700)  # (w, h)
DEFAULT_FIXED_HEIGHT = 32
CHAT_PANEL_PERCENTAGE = 88
INPUT_PANEL_PERCENTAGE = 12
MIN_MSG_INPUT_HEIGHT = 20
MAX_MSG_INPUT_HEIGHT = 300
USER_MESSAGE_MAX_WIDTH_PERCENT = 70
REFRESH_INTERVAL = 300  # seconds

THEME = get_theme()


# Providers config
DEFAULT_SETTINGS = {
    "provider": "ollama",
    "model_id": None,
    "temperature": 10,
    "max_length": None,
    "system_prompt": None,
}

DEFAULT_PROVIDERS_DATA = {
    "custom": {
        "api_address": "",
        "api_key": "",
        "api_key_url": None,
        "api_docs_url": None,
        "model_docs_url": None,
    },
    "anthropic": {
        "api_address": "https://api.anthropic.com/v1/",
        "api_key": None,
        "api_key_url": "https://console.anthropic.com/settings/keys",
        "api_docs_url": "https://docs.anthropic.com/en/docs",
        "model_docs_url": "https://docs.anthropic.com/en/docs/about-claude/models/all-models",
    },
    "deepseek": {
        "api_address": "https://api.deepseek.com/v1",
        "api_key": None,
        "api_key_url": "https://platform.deepseek.com/api_keys",
        "api_docs_url": "https://platform.deepseek.com/docs",
        "model_docs_url": "https://platform.deepseek.com/models",
    },
    "google": {
        "api_address": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": None,
        "api_key_url": "https://aistudio.google.com/app/apikey",
        "api_docs_url": "https://ai.google.dev/gemini-api/docs",
        "model_docs_url": "https://ai.google.dev/gemini-api/docs/models",
    },
    "ollama": {
        "api_address": "http://localhost:11434/v1",
        "api_key": "ollama",
        "api_key_url": None,
        "api_docs_url": "https://github.com/ollama/ollama/blob/main/docs/api.md",
        "model_docs_url": "https://ollama.com/search",
    },
    "openai": {
        "api_address": "https://api.openai.com/v1",
        "api_key": None,
        "api_key_url": "https://platform.openai.com/api-keys",
        "api_docs_url": "https://platform.openai.com/docs",
        "model_docs_url": "https://platform.openai.com/docs/models",
    },
    "openrouter": {
        "api_address": "https://openrouter.ai/api/v1",
        "api_key": None,
        "api_key_url": "https://openrouter.ai/settings/keys",
        "api_docs_url": "https://openrouter.ai/docs/quick-start",
        "model_docs_url": "https://openrouter.ai/models",
    },
    "qwen": {
        "api_address": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": None,
        "api_key_url": "https://bailian.console.aliyun.com/?apiKey=1#/api-key",
        "api_docs_url": "https://help.aliyun.com/document_detail/2590237.html",
        "model_docs_url": "https://help.aliyun.com/zh/model-studio/developer-reference/what-is-qwen-llm",
    },
}

SUPPORTED_VISION_MODELS = [
    # Anthropic
    "claude-3",
    "claude-haiku-4",
    "claude-opus-4",
    "claude-sonnet-4",
    # Google Gemini
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash",
    "gemini-3-pro",
    "gemini-3.1-flash",
    "gemini-3.1-pro",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    # Ollama
    "bakllava",
    "gemma3",
    "granite3.2-vision",
    "llama3.2-vision",
    "llama4",
    "llava",
    "minicpm-v",
    "moondream",
    "qwen2.5-vl",
    "qwen2.5vl",
    # OpenAI
    "gpt-4-turbo",
    "gpt-4.1",
    "gpt-4o",
    "gpt-5",
    "o1",
    "o3",
    "o4-mini",
    # DashScope
    "qvq",
    "qwen-vl",
    "qwen3-omni",
    "qwen3-vl",
    "qwen3.5-",
    "qwen3.6-",
    # OpenRouter
    "ernie-4.5-vl",
    "glm-4.5v",
    "glm-4.6v",
    "glm-5v",
    "grok-4",
    "kimi-k2",
    "llama-3.2-11b-vision",
    "llama-4-maverick",
    "llama-4-scout",
    "mistral-medium-3",
    "mistral-small-3.1",
    "mistral-small-3.2",
    "nemotron-nano-12b-v2-vl",
    "nova-2-lite",
    "nova-lite",
    "nova-premier",
    "nova-pro",
    "pixtral",
    "qianfan-ocr",
    "seed-1.6",
    "seed-2.0",
    "ui-tars",
]
