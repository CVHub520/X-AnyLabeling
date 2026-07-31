# flake8: noqa

from .about_dialog import AboutDialog
from .auto_labeling import AutoLabelingWidget
from .brightness_contrast_dialog import BrightnessContrastDialog
from .canvas import Canvas
from .canvas_adjustment import CanvasAdjustmentWidget
from .compare_view import CompareViewManager, CompareViewSlider
from .chatbot_dialog import ChatbotDialog
from .classifier_dialog import ClassifierDialog
from .crosshair_settings_dialog import CrosshairSettingsDialog
from .file_dialog_preview import FileDialogPreview
from .filter_label_widget import GroupIDFilterComboBox, LabelFilterComboBox
from .shape_dialog import ShapeModifyDialog
from .label_dialog import (
    DigitShortcutDialog,
    GroupIDModifyDialog,
    LabelDialog,
    LabelModifyDialog,
    LabelQLineEdit,
)
from .label_list_widget import LabelListWidget, LabelListWidgetItem
from .model_dropdown_widget import SearchBar
from .navigator_widget import NavigatorDialog
from .overview_dialog import OverviewDialog
from .polygon_sides_dialog import PolygonSidesDialog
from .ppocr_dialog import PPOCRDialog
from .popup import Popup
from .toolbar import ToolBar
from .unique_label_qlist_widget import UniqueLabelQListWidget

try:
    from .video_classifier_dialog import VideoClassifierDialog
except ImportError:
    # QtMultimedia is optional in PyQt6 packaging and is absent from some
    # builds (e.g. the conda `pyqt` packages for linux-aarch64). Only the
    # Video Classifier needs it, so degrade gracefully rather than taking
    # the whole application down at import time.
    VideoClassifierDialog = None
from .vqa_dialog import VQADialog
from .zoom_widget import ZoomWidget
