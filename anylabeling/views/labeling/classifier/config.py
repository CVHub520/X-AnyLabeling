from anylabeling.views.labeling.utils.theme import get_theme

BORDER_RADIUS = "8px"
FONT_SIZE_SMALL = "11px"
FONT_SIZE_NORMAL = "13px"
FONT_SIZE_LARGE = "16px"
DEFAULT_COMPONENT_HEIGHT = 32
PANEL_SIZE = 600
ICON_SIZE_NORMAL = (32, 32)
ICON_SIZE_SMALL = (16, 16)
IMAGE_DISPLAY_MAX_SIZE = (PANEL_SIZE, PANEL_SIZE)

DEFAULT_WINDOW_TITLE = "Image Classifier"
DEFAULT_WINDOW_SIZE = (1200, 700)

THEME = get_theme()

BUTTON_COLORS = {
    "primary": {
        "background": THEME["primary"],
        "hover": THEME["primary_hover"],
        "pressed": THEME["primary_pressed"],
        "text": "white",
    },
    "secondary": {
        "background": THEME["surface"],
        "hover": THEME["surface_hover"],
        "pressed": THEME["surface_pressed"],
        "text": THEME["text"],
        "border": THEME["border_light"],
    },
    "outline": {
        "background": "transparent",
        "hover": THEME["background_secondary"],
        "pressed": THEME["surface"],
        "text": THEME["text_secondary"],
        "border": THEME["border"],
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
    "light_green": {
        "background": "#90ee90",
        "hover": "#7fdd7f",
        "pressed": "#6fcd6f",
        "text": "#1d1d1f",
        "border": "#90ee90",
    },
}

DEFAULT_OUTPUT_DIR = "classified"
REQUEST_TIMEOUT = 60
