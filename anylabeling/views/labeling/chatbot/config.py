# Global design system
ANIMATION_DURATION = "200ms"
BORDER_RADIUS = "16px"
FONT_FAMILY = "SF Pro Text, -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif"
FONT_SIZE_SMALL = "12px"
FONT_SIZE_NORMAL = "13px"
FONT_SIZE_LARGE = "15px"
ICON_SIZE_NORMAL = (32, 32)
ICON_SIZE_SMALL = (16, 16)


# Initialization parameters
DEFAULT_WINDOW_TITLE = "Chatbot"
DEFAULT_WINDOW_SIZE = (1200, 700)  # Width and height
DEFAULT_PROVIDER = "ollama"
CHAT_PANEL_PERCENTAGE = 88
INPUT_PANEL_PERCENTAGE = 12
MIN_MSG_INPUT_HEIGHT = 20
MAX_MSG_INPUT_HEIGHT = 300
MAX_USER_MSG_WIDTH = 70 # 70% of the layout width


# Theme configuration
THEME = {
    "primary": "#007AFF",           # Modern iOS blue
    "primary_hover": "#2B8FFF",     # Slightly lighter primary
    "background": "#FFFFFF",        # Clean white background
    "sidebar": "#F8F8FA",           # Light gray for sidebar
    "border": "#E5E5EA",            # Subtle border color
    "text": "#2C2C2E",              # Dark gray for better readability
    "text_secondary": "#8E8E93",    # Balanced gray for secondary text
    "success": "#30D158",           # Softer green
    "warning": "#FF9F0A",           # Warm orange
    "error": "#FF453A",             # Refined red
    "user_bubble": "#f6f2f2",       # Lighter, softer blue for user messages
    "input_bg": "#F4F4F5",          # Light gray for input background
    "hover": "#F2F2F7",             # Subtle hover state
}

# Provider configurations
PROVIDER_CONFIGS = {
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
