import os

# Chatbot config path
home_dir = os.path.expanduser("~")
root_dir = os.path.join(home_dir, "xanylabeling_data/chatbot")
MODELS_CONFIG_PATH = os.path.join(root_dir, "models.json")
SETTINGS_CONFIG_PATH = os.path.join(root_dir, "settings.json")
PROVIDERS_CONFIG_PATH = os.path.join(root_dir, "providers.json")

# Global design system
ANIMATION_DURATION = "200ms"
BORDER_RADIUS = "8px"
FONT_SIZE_TINY = "9px"
FONT_SIZE_SMALL = "11px"
FONT_SIZE_NORMAL = "13px"
FONT_SIZE_LARGE = "16px"
ICON_SIZE_NORMAL = (32, 32)
ICON_SIZE_SMALL = (16, 16)

# Initialization parameters
DEFAULT_WINDOW_TITLE = "Chatbot (Beta)"
DEFAULT_WINDOW_SIZE = (1200, 700)  # (w, h)
DEFAULT_FIXED_HEIGHT = 32
CHAT_PANEL_PERCENTAGE = 88
INPUT_PANEL_PERCENTAGE = 12
MIN_MSG_INPUT_HEIGHT = 20
MAX_MSG_INPUT_HEIGHT = 300
USER_MESSAGE_MAX_WIDTH_PERCENT = 70
REFRESH_INTERVAL = 300  # seconds

# Theme configuration
THEME = {
    "primary": "#60A5FA",  # Tailwind CSS blue-500
    "background": "#FFFFFF",  # Clean white background
    "background_secondary": "#F9F9F9",  # Light gray background
    "background_hover": "#DBDBDB",  # Light gray for hover
    "border": "#E5E5E5",  # Subtle border color
    "text": "#2C2C2E",  # Dark gray for better readability
    "highlight_text": "#2196F3",  # Highlight text color
    "success": "#30D158",  # Softer green
    "warning": "#FF9F0A",  # Warm orange
    "error": "#FF453A",  # Refined red
}


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
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20240620",
    # Google AI
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-exp",
    "models/gemini-2.0-pro-exp",
    "models/gemini-2.0-pro-exp-02-05",
    "models/gemini-2.0-flash-thinking-exp",
    "models/gemini-2.0-flash-thinking-exp-1219",
    "models/gemini-2.0-flash-thinking-exp-01-21",
    # Ollama
    "gemma3",
    "gemma3:4b",
    "gemma3:12b",
    "gemma3:27b",
    "bakllava",
    "granite3.2-vision",
    "minicpm-v",
    "moondream",
    "llava",
    "llava-llama3",
    "llava-phi3",
    "llama3.2-vision",
    # Qwen
    "qwen-vl-ocr-latest",
    "qwen-vl-ocr",
    "qwen-vl-max",
    "qwen-vl-plus",
    # OpenAI
    "gpt-4.5-preview",
    "gpt-4.5-preview-2025-02-27",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview-2024-12-17",
]
