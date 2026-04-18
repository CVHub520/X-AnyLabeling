from __future__ import annotations

from dataclasses import dataclass

DEFAULT_WINDOW_TITLE = "PaddleOCR"
DEFAULT_WINDOW_SIZE = (1560, 920)
LEFT_PANEL_WIDTH = 240
MIN_DIALOG_WIDTH = 1200
MIN_DIALOG_HEIGHT = 760

PPOCR_ROOT_DIRNAME = "xanylabeling_data/paddleocr"
PPOCR_FILES_DIRNAME = "files"
PPOCR_JSONS_DIRNAME = "jsons"
PPOCR_PDF_DIR_PREFIX = "__PDF_"
PPOCR_BLOCK_IMAGES_DIR_PREFIX = "__BLOCK_IMAGES_"

PPOCR_STATUS_PENDING = "pending"
PPOCR_STATUS_PARSED = "parsed"
PPOCR_STATUS_ERROR = "error"
PPOCR_RUNTIME_STATUS_PARSING = "parsing"

PPOCR_FILE_TYPE_IMAGE = "image"
PPOCR_FILE_TYPE_PDF = "pdf"
PPOCR_FILE_TYPE_ALL = "all"

PPOCR_SORT_NEWEST = "newest"
PPOCR_SORT_OLDEST = "oldest"

PPOCR_OFFLINE_MODEL_LABEL = "Offline Mode"
PPOCR_API_MODEL_ID = "__ppocr_api__"
PPOCR_API_MODEL_LABEL = "PPOCR (API)"
PPOCR_API_MODEL_SERVER_ID = "ppocr_api"
PPOCR_PIPELINE_CAPABILITY_KEY = "ppocr_pipeline"

PPOCR_SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".cif",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
PPOCR_SUPPORTED_PDF_SUFFIXES = {".pdf"}
PPOCR_SUPPORTED_SUFFIXES = (
    PPOCR_SUPPORTED_IMAGE_SUFFIXES | PPOCR_SUPPORTED_PDF_SUFFIXES
)

PPOCR_BLOCK_CARD_MAX_HEIGHT_PX = 640
PPOCR_ZOOM_STEP = 0.1
PPOCR_MIN_ZOOM = 0.2
PPOCR_MAX_ZOOM = 5.0

PPOCR_COLOR_TEXT = "rgb(70, 88, 255)"
PPOCR_COLOR_TABLE = "rgb(47, 189, 113)"
PPOCR_COLOR_IMAGE = "rgb(189, 76, 255)"
PPOCR_COLOR_HEADER = "rgb(182, 178, 241)"
PPOCR_COLOR_FORMULA = "rgb(250, 219, 20)"
PPOCR_COLOR_EDITED = "rgb(255, 156, 40)"
PPOCR_COLOR_OVERLAY = "rgb(133, 144, 255)"


@dataclass(frozen=True)
class PPOCRPipelineModel:
    model_id: str
    display_name: str


@dataclass(frozen=True)
class PPOCRServiceProbe:
    is_online: bool
    server_url: str
    pipeline_model: str
    pipeline_models: tuple[PPOCRPipelineModel, ...]
    error_message: str = ""
