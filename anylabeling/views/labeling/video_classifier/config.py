from anylabeling.views.labeling.utils.theme import get_theme

DEFAULT_WINDOW_TITLE = "Video Classifier"
DEFAULT_WINDOW_SIZE = (1280, 800)
DEFAULT_COMPONENT_HEIGHT = 32
BORDER_RADIUS = "8px"
FONT_SIZE_SMALL = "11px"
FONT_SIZE_NORMAL = "13px"
FONT_SIZE_LARGE = "16px"

SCHEMA_VERSION = "1.0.0"
SIDECAR_TYPE = "video_classification"

SUPPORTED_VIDEO_EXTS = (
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".flv",
    ".wmv",
)
SUPPORTED_VIDEO_FILTER = (
    "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.flv *.wmv);;"
    "All Files (*)"
)

LABEL_PALETTE = [
    "#EF476F",
    "#FFD166",
    "#06D6A0",
    "#118AB2",
    "#8338EC",
    "#FF7F50",
    "#3A86FF",
    "#FB5607",
    "#2EC4B6",
    "#9D4EDD",
    "#FF006E",
    "#43AA8B",
]

DEFAULT_LABEL_NAMES = []

TIMELINE_HEIGHT = 84
TIMELINE_PAD_X = 12
TIMELINE_RULER_HEIGHT = 22
TIMELINE_TRACK_GAP = 8
TIMELINE_TRACK_BOTTOM_PAD = 4
TIMELINE_HANDLE_WIDTH = 8
MIN_SEGMENT_MS = 50

DEFAULT_PLAYBACK_RATES = [0.25, 0.5, 1.0, 1.5, 2.0]
DEFAULT_SPLIT = (80, 10, 10)
DEFAULT_RANDOM_SEED = 2026


def theme():
    return get_theme()
