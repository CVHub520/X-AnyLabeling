import os


# VQA config path
home_dir = os.path.expanduser("~")
root_dir = os.path.join(home_dir, "xanylabeling_data/vqa")
COMPONENTS_CONFIG_PATH = os.path.join(root_dir, "components.json")

# Global design system
BORDER_RADIUS = "8px"
FONT_SIZE_SMALL = "11px"
FONT_SIZE_NORMAL = "13px"
FONT_SIZE_LARGE = "16px"
DEFAULT_COMPONENT_HEIGHT = 32
ICON_SIZE_NORMAL = (32, 32)
ICON_SIZE_SMALL = (16, 16)

# Initialization parameters
DEFAULT_WINDOW_TITLE = "VQA (Beta)"
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

SUPPORTED_WIDGETS = [
    "QLineEdit",
    "QRadioButton",
    "QComboBox",
    "QCheckBox",
]
