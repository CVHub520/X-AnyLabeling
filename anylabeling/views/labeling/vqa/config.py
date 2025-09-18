import os


# VQA config path
home_dir = os.path.expanduser("~")
root_dir = os.path.join(home_dir, "xanylabeling_data/vqa")
PROMPTS_CONFIG_PATH = os.path.join(root_dir, "prompts.json")
COMPONENTS_CONFIG_PATH = os.path.join(root_dir, "components.json")

# Global design system
BORDER_RADIUS = "8px"
FONT_SIZE_SMALL = "11px"
FONT_SIZE_NORMAL = "13px"
FONT_SIZE_LARGE = "16px"
DEFAULT_COMPONENT_HEIGHT = 32
PANEL_SIZE = 600
ICON_SIZE_NORMAL = (32, 32)
ICON_SIZE_SMALL = (16, 16)

# Initialization parameters
DEFAULT_WINDOW_TITLE = "VQA"
DEFAULT_WINDOW_SIZE = (1200, 700)  # (w, h)
DEFAULT_COMPONENT_WINDOW_SIZE = (600, 350)

# Theme configuration
THEME = {
    "primary": "#60A5FA",  # Tailwind CSS blue-500
    "background": "#FFFFFF",  # Clean white background
    "background_secondary": "#F9F9F9",  # Light gray background
    "background_hover": "#DBDBDB",  # Light gray for hover
    "border": "#E5E5E5",  # Subtle border color
    "text": "#718096",  # Dark gray for better readability
    "highlight_text": "#2196F3",  # Highlight text color
    "success": "#30D158",  # Softer green
    "warning": "#FF9F0A",  # Warm orange
    "error": "#FF453A",  # Refined red
}

# Button color schemes
BUTTON_COLORS = {
    "primary": {
        "background": "#0077ed",
        "hover": "#0066cc",
        "pressed": "#005bb5",
        "text": "white",
    },
    "secondary": {
        "background": "#f5f5f7",
        "hover": "#e5e5e5",
        "pressed": "#d6d6d6",
        "text": "#1d1d1f",
        "border": "#d2d2d7",
    },
    "success": {
        "background": "#10b981",
        "hover": "#059669",
        "pressed": "#047857",
        "text": "white",
    },
    "danger": {
        "background": "#dc2626",
        "hover": "#b91c1c",
        "pressed": "#991b1b",
        "text": "white",
    },
}

SUPPORTED_WIDGETS = [
    "QLineEdit",
    "QRadioButton",
    "QComboBox",
    "QCheckBox",
]

# AI Assistant
REQUEST_TIMEOUT = 120
AI_PROMPT_PLACEHOLDER = (
    "Examples:\n"
    "   1. @image Describe this image\n"
    "   2. Translate to English, return translated text only: @text\n"
    "   3. @widget.title - Reference widget values\n"
    "   4. @label.shapes - Reference annotation data"
)
DEFAULT_TEMPLATES = {
    # @text
    "Condense text": "Please make this text more concise while keeping the key points. Output the condensed version only:\n@text",
    "Academic style": "Please rewrite this text in a formal academic style. Return only the reformatted text:\n@text",
    "Clarify meaning": "Please break down and explain the meaning of this text. Provide only the explanation:\n@text",
    "Grammar check": "Please review and correct any grammatical errors in this text. Return only the corrected version:\n@text",
    "Enhance writing": "Please refine and polish this text to make it more effective. Return only the enhanced version:\n@text",
    "Brief summary": "Please provide a concise summary of the key points in this text. Return only the summary:\n@text",
    "Chinese translation": "Please convert this text to Chinese language. Return only the translated version:\n@text",
    # @image
    "Image Description": "@image Describe this image in detail.",
    "OCR": "@image Read all texts in the image, output in lines.",
    "Text Recognition": "@image Recognise all texts in the image, output in lines.",
    "Text Spotting": "@image Spotting all the text in the image with line-level, and output in JSON format.",
}
