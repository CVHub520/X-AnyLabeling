import os
import copy
import re
import time
import importlib.resources as pkg_resources
from threading import Lock, Event

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

import anylabeling.configs as auto_labeling_configs
from anylabeling.services.auto_labeling.worker import GenericWorker
from anylabeling.services.auto_labeling.model import load_model_config
from anylabeling.views.labeling.logger import logger
from anylabeling.config import get_config, save_config
from anylabeling.services.auto_labeling.types import (
    AutoLabelingResult,
    DownloadCancelledError,
)
from anylabeling.services.auto_labeling.utils import TimeoutContext
from anylabeling.services.auto_labeling import (
    _CUSTOM_MODELS,
    _CACHED_AUTO_LABELING_MODELS,
    _AUTO_LABELING_MARKS_MODELS,
    _AUTO_LABELING_API_TOKEN_MODELS,
    _AUTO_LABELING_RESET_TRACKER_MODELS,
    _AUTO_LABELING_CONF_MODELS,
    _AUTO_LABELING_IOU_MODELS,
    _AUTO_LABELING_MASK_FINENESS_MODELS,
    _AUTO_LABELING_CROPPING_MODE_MODELS,
    _AUTO_LABELING_PRESERVE_EXISTING_ANNOTATIONS_STATE_MODELS,
    _AUTO_LABELING_PROMPT_MODELS,
    _ON_NEXT_FILES_CHANGED_MODELS,
)

_AUTO_SEGMENTATION_MODELS = {
    "yolov5_sam",
    "yolov8_sam2",
    "grounding_sam",
    "grounding_sam2",
    "open_vision",
    "segment_anything",
    "segment_anything_2",
    "segment_anything_3",
    "segment_anything_2_video",
    "efficientvit_sam",
    "sam_med2d",
    "edge_sam",
    "sam_hq",
}
_TIMEOUT_MODELS = {"florence2", "geco"}


def _get_model_class(model_type):  # noqa: C901
    if model_type == "yolov5":
        from .yolov5 import YOLOv5

        return YOLOv5
    elif model_type == "yolov6":
        from .yolov6 import YOLOv6

        return YOLOv6
    elif model_type == "yolov7":
        from .yolov7 import YOLOv7

        return YOLOv7
    elif model_type == "yolov5_sahi":
        from .yolov5_sahi import YOLOv5_SAHI

        return YOLOv5_SAHI
    elif model_type == "yolov8_sahi":
        from .yolov8_sahi import YOLOv8_SAHI

        return YOLOv8_SAHI
    elif model_type == "yolo26_sahi":
        from .yolo26_sahi import YOLO26_SAHI

        return YOLO26_SAHI
    elif model_type == "yolo11_sahi":
        from .yolo11_sahi import YOLO11_SAHI

        return YOLO11_SAHI
    elif model_type == "yolov8":
        from .yolov8 import YOLOv8

        return YOLOv8
    elif model_type == "yolov9":
        from .yolov9 import YOLOv9

        return YOLOv9
    elif model_type == "yolov10":
        from .yolov10 import YOLOv10

        return YOLOv10
    elif model_type == "yolo11":
        from .yolo11 import YOLO11

        return YOLO11
    elif model_type == "yolow":
        from .yolow import YOLOW

        return YOLOW
    elif model_type == "yolov5_seg":
        from .yolov5_seg import YOLOv5_Seg

        return YOLOv5_Seg
    elif model_type == "yolov5_ram":
        from .yolov5_ram import YOLOv5_RAM

        return YOLOv5_RAM
    elif model_type == "yolow_ram":
        from .yolow_ram import YOLOW_RAM

        return YOLOW_RAM
    elif model_type == "yolov8_seg":
        from .yolov8_seg import YOLOv8_Seg

        return YOLOv8_Seg
    elif model_type == "yolo11_seg":
        from .yolo11_seg import YOLO11_Seg

        return YOLO11_Seg
    elif model_type == "yolov8_obb":
        from .yolov8_obb import YOLOv8_OBB

        return YOLOv8_OBB
    elif model_type == "yolo11_obb":
        from .yolo11_obb import YOLO11_OBB

        return YOLO11_OBB
    elif model_type == "yolov8_pose":
        from .yolov8_pose import YOLOv8_Pose

        return YOLOv8_Pose
    elif model_type == "yolo11_pose":
        from .yolo11_pose import YOLO11_Pose

        return YOLO11_Pose
    elif model_type == "yolox":
        from .yolox import YOLOX

        return YOLOX
    elif model_type == "yolo_nas":
        from .yolo_nas import YOLO_NAS

        return YOLO_NAS
    elif model_type == "damo_yolo":
        from .damo_yolo import DAMO_YOLO

        return DAMO_YOLO
    elif model_type == "gold_yolo":
        from .gold_yolo import Gold_YOLO

        return Gold_YOLO
    elif model_type == "grounding_dino":
        from .grounding_dino import Grounding_DINO

        return Grounding_DINO
    elif model_type == "grounding_dino_api":
        from .grounding_dino_api import Grounding_DINO_API

        return Grounding_DINO_API
    elif model_type == "ram":
        from .ram import RAM

        return RAM
    elif model_type == "internimage_cls":
        from .internimage_cls import InternImage_CLS

        return InternImage_CLS
    elif model_type == "pulc_attribute":
        from .pulc_attribute import PULC_Attribute

        return PULC_Attribute
    elif model_type == "yolov5_sam":
        from .yolov5_sam import YOLOv5SegmentAnything

        return YOLOv5SegmentAnything
    elif model_type == "yolov8_sam2":
        from .yolov8_sam2 import YOLOv8SegmentAnything2

        return YOLOv8SegmentAnything2
    elif model_type == "grounding_sam":
        from .grounding_sam import GroundingSAM

        return GroundingSAM
    elif model_type == "grounding_sam2":
        from .grounding_sam2 import GroundingSAM2

        return GroundingSAM2
    elif model_type == "open_vision":
        from .open_vision import OpenVision

        return OpenVision
    elif model_type == "doclayout_yolo":
        from .doclayout_yolo import DocLayoutYOLO

        return DocLayoutYOLO
    elif model_type == "yolov5_obb":
        from .yolov5_obb import YOLOv5OBB

        return YOLOv5OBB
    elif model_type == "segment_anything":
        from .segment_anything import SegmentAnything

        return SegmentAnything
    elif model_type == "segment_anything_2":
        from .segment_anything_2 import SegmentAnything2

        return SegmentAnything2
    elif model_type == "segment_anything_3":
        from .segment_anything_3 import SegmentAnything3

        return SegmentAnything3
    elif model_type == "segment_anything_2_video":
        from .segment_anything_2_video import SegmentAnything2Video

        return SegmentAnything2Video
    elif model_type == "efficientvit_sam":
        from .efficientvit_sam import EfficientViT_SAM

        return EfficientViT_SAM
    elif model_type == "sam_med2d":
        from .sam_med2d import SAM_Med2D

        return SAM_Med2D
    elif model_type == "edge_sam":
        from .edge_sam import EdgeSAM

        return EdgeSAM
    elif model_type == "sam_hq":
        from .sam_hq import SAM_HQ

        return SAM_HQ
    elif model_type == "yolov5_resnet":
        from .yolov5_resnet import YOLOv5_ResNet

        return YOLOv5_ResNet
    elif model_type == "rtdetr":
        from .rtdetr import RTDETR

        return RTDETR
    elif model_type == "rtdetrv2":
        from .rtdetrv2 import RTDETRv2

        return RTDETRv2
    elif model_type == "rio_detr":
        from .rio_detr import RiODETR

        return RiODETR
    elif model_type == "deimv2":
        from .deimv2 import DEIMv2

        return DEIMv2
    elif model_type == "yolov6_face":
        from .yolov6_face import YOLOv6Face

        return YOLOv6Face
    elif model_type == "scrfd":
        from .scrfd import SCRFD

        return SCRFD
    elif model_type == "yolox_dwpose":
        from .yolox_dwpose import YOLOX_DWPose

        return YOLOX_DWPose
    elif model_type == "rtmdet_pose":
        from .rtmdet_pose import RTMDet_Pose

        return RTMDet_Pose
    elif model_type == "clrnet":
        from .clrnet import CLRNet

        return CLRNet
    elif model_type == "ppocr_v4":
        from .ppocr_v4 import PPOCRv4

        return PPOCRv4
    elif model_type == "ppocr_v5":
        from .ppocr_v5 import PPOCRv5

        return PPOCRv5
    elif model_type == "ppocr_v6":
        from .ppocr_v6 import PPOCRv6

        return PPOCRv6
    elif model_type == "yolov5_cls":
        from .yolov5_cls import YOLOv5_CLS

        return YOLOv5_CLS
    elif model_type == "yolov5_car_plate":
        from .yolov5_car_plate import YOLOv5CarPlateDetRec

        return YOLOv5CarPlateDetRec
    elif model_type == "yolov8_cls":
        from .yolov8_cls import YOLOv8_CLS

        return YOLOv8_CLS
    elif model_type == "yolo11_cls":
        from .yolo11_cls import YOLO11_CLS

        return YOLO11_CLS
    elif model_type == "yolov5_det_track":
        from .yolov5_det_track import YOLOv5_Det_Tracker

        return YOLOv5_Det_Tracker
    elif model_type == "yolov8_det_track":
        from .yolov8_det_track import YOLOv8_Det_Tracker

        return YOLOv8_Det_Tracker
    elif model_type == "yolo11_det_track":
        from .yolo11_det_track import YOLO11_Det_Tracker

        return YOLO11_Det_Tracker
    elif model_type == "yolov8_seg_track":
        from .yolov8_seg_track import YOLOv8_Seg_Tracker

        return YOLOv8_Seg_Tracker
    elif model_type == "yolo11_seg_track":
        from .yolo11_seg_track import YOLO11_Seg_Tracker

        return YOLO11_Seg_Tracker
    elif model_type == "yolov8_obb_track":
        from .yolov8_obb_track import YOLOv8_Obb_Tracker

        return YOLOv8_Obb_Tracker
    elif model_type == "yolo11_obb_track":
        from .yolo11_obb_track import YOLO11_Obb_Tracker

        return YOLO11_Obb_Tracker
    elif model_type == "yolov8_pose_track":
        from .yolov8_pose_track import YOLOv8_Pose_Tracker

        return YOLOv8_Pose_Tracker
    elif model_type == "yolo11_pose_track":
        from .yolo11_pose_track import YOLO11_Pose_Tracker

        return YOLO11_Pose_Tracker
    elif model_type == "yolo26_det_track":
        from .yolo26_det_track import YOLO26_Det_Tracker

        return YOLO26_Det_Tracker
    elif model_type == "yolo26_seg_track":
        from .yolo26_seg_track import YOLO26_Seg_Tracker

        return YOLO26_Seg_Tracker
    elif model_type == "yolo26_obb_track":
        from .yolo26_obb_track import YOLO26_Obb_Tracker

        return YOLO26_Obb_Tracker
    elif model_type == "yolo26_pose_track":
        from .yolo26_pose_track import YOLO26_Pose_Tracker

        return YOLO26_Pose_Tracker
    elif model_type == "rmbg":
        from .rmbg import RMBG

        return RMBG
    elif model_type == "depth_anything":
        from .depth_anything import DepthAnything

        return DepthAnything
    elif model_type == "depth_anything_v2":
        from .depth_anything_v2 import DepthAnythingV2

        return DepthAnythingV2
    elif model_type == "upn":
        from .upn import UPN

        return UPN
    elif model_type == "remote_server":
        from .remote_server import RemoteServer

        return RemoteServer
    elif model_type == "florence2":
        from .florence2 import Florence2

        return Florence2
    elif model_type == "geco":
        from .geco import GeCo

        return GeCo
    elif model_type == "rfdetr":
        from .rfdetr import RFDETR

        return RFDETR
    elif model_type == "rfdetr_seg":
        from .rfdetr_seg import RFDETR_Seg

        return RFDETR_Seg
    elif model_type == "dfine":
        from .dfine import DFINE

        return DFINE
    elif model_type == "dfine_seg":
        from .dfine_seg import DFINESeg

        return DFINESeg
    elif model_type == "yolo12":
        from .yolo12 import YOLO12

        return YOLO12
    elif model_type == "yolo26":
        from .yolo26 import YOLO26

        return YOLO26
    elif model_type == "yolo26_seg":
        from .yolo26_seg import YOLO26_Seg

        return YOLO26_Seg
    elif model_type == "yolo26_obb":
        from .yolo26_obb import YOLO26_OBB

        return YOLO26_OBB
    elif model_type == "yolo26_pose":
        from .yolo26_pose import YOLO26_Pose

        return YOLO26_Pose
    elif model_type == "yoloe":
        from .yoloe import YOLOE

        return YOLOE
    elif model_type == "u_rtdetr":
        from .u_rtdetr import U_RTDETR

        return U_RTDETR
    raise ValueError(f"Unknown model type: {model_type}")


class ModelManager(QObject):
    """Model manager"""

    MAX_NUM_CUSTOM_MODELS = 5
    CUSTOM_MODEL_NAME_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
    model_configs_changed = pyqtSignal(list)
    new_model_status = pyqtSignal(str)
    model_loaded = pyqtSignal(dict)
    new_auto_labeling_result = pyqtSignal(AutoLabelingResult)
    auto_segmentation_model_selected = pyqtSignal()
    auto_segmentation_model_unselected = pyqtSignal()
    prediction_started = pyqtSignal()
    prediction_finished = pyqtSignal()
    request_next_files_requested = pyqtSignal()
    output_modes_changed = pyqtSignal(dict, str)
    download_progress = pyqtSignal(int, int)
    download_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.model_configs = []

        self.loaded_model_config = None
        self.loaded_model_config_lock = Lock()

        self.model_download_worker = None
        self.model_download_thread = None
        self.model_execution_thread = None
        self.model_execution_thread_lock = Lock()
        self.model_execution_worker = None
        self._cancel_event = Event()

        self.load_model_configs()

    def load_model_configs(self):
        """Load model configs"""
        # Load list of default models
        with (
            pkg_resources.files(auto_labeling_configs)
            .joinpath("models.yaml")
            .open(encoding="utf-8") as f
        ):
            model_list = load_model_config(f)

        # Load list of custom models
        custom_models = get_config().get("custom_models", [])
        for custom_model in custom_models:
            custom_model["is_custom_model"] = True

        # Remove invalid/not found custom models
        custom_models = [
            custom_model
            for custom_model in custom_models
            if os.path.isfile(custom_model.get("config_file", ""))
        ]
        config = get_config()
        config["custom_models"] = custom_models
        save_config(config)

        model_list += custom_models

        # Load model configs
        model_configs = []
        for model in model_list:
            model_config = {}
            config_file = model["config_file"]
            if config_file.startswith(":/"):  # Config file is in resources
                config_file_name = config_file[2:]
                resource_path = pkg_resources.files(
                    auto_labeling_configs
                ).joinpath("auto_labeling", config_file_name)
                config_content = resource_path.read_text(encoding="utf-8")
                model_config = load_model_config(config_content)
                model_config["config_file"] = str(config_file)
            else:  # Config file is in local file system
                with open(config_file, "r", encoding="utf-8") as f:
                    model_config = load_model_config(f)
                    model_config["config_file"] = os.path.normpath(
                        os.path.abspath(config_file)
                    )
            is_custom = model.get("is_custom_model", False)
            model_config["is_custom_model"] = is_custom
            if is_custom and not self.is_valid_custom_model_name(
                model_config.get("name")
            ):
                logger.error(
                    "Skipping custom model with an invalid 'name' field."
                )
                continue
            if is_custom and not model_config["name"].startswith("_custom_"):
                model_config["name"] = f"_custom_{model_config['name']}"

            model_configs.append(model_config)

        # Sort by last used
        for i, model_config in enumerate(model_configs):
            # Keep order for integrated models
            if not model_config.get("is_custom_model", False):
                model_config["last_used"] = -i
            else:
                model_config["last_used"] = model_config.get(
                    "last_used", time.time()
                )
        model_configs.sort(key=lambda x: x.get("last_used", 0), reverse=True)

        self.model_configs = model_configs
        self.model_configs_changed.emit(model_configs)

    @classmethod
    def is_valid_custom_model_name(cls, name):
        return (
            isinstance(name, str)
            and name not in (".", "..")
            and cls.CUSTOM_MODEL_NAME_PATTERN.fullmatch(name) is not None
        )

    def update_model_config(self, config_file, key, value):
        """Update a specific key in a model's configuration."""
        for config in self.model_configs:
            if config.get("config_file") == config_file:
                config[key] = value
                if (
                    self.loaded_model_config
                    and self.loaded_model_config.get("config_file")
                    == config_file
                ):
                    self.loaded_model_config[key] = value
                break

        if config_file and config_file.startswith(":/"):
            user_config = get_config()
            if "remote_server_settings" not in user_config:
                user_config["remote_server_settings"] = {}
            user_config["remote_server_settings"][key] = value
            save_config(user_config)

    def get_model_configs(self):
        """Return model infos"""
        return self.model_configs

    def set_output_mode(self, mode):
        """Set output mode"""
        if self.loaded_model_config and self.loaded_model_config["model"]:
            self.loaded_model_config["model"].set_output_mode(mode)

    def cancel_download(self):
        """Cancel the current model download."""
        self._cancel_event.set()

    @pyqtSlot()
    def on_model_download_finished(self):
        """Handle model download thread finished"""
        self.download_finished.emit()
        if self._cancel_event.is_set():
            self._cancel_event.clear()
            self.new_model_status.emit(self.tr("Download cancelled."))
            self.model_loaded.emit({})
            return
        if self.loaded_model_config and self.loaded_model_config["model"]:
            self.new_model_status.emit(
                self.tr("Model loaded. Ready for labeling.")
            )
            self.model_loaded.emit(self.loaded_model_config)
            self.output_modes_changed.emit(
                self.loaded_model_config["model"].Meta.output_modes,
                self.loaded_model_config["model"].Meta.default_output_mode,
            )
        else:
            self.model_loaded.emit({})

    def load_custom_model(self, config_file):
        """Run custom model loading in a thread"""
        config_file = os.path.normpath(os.path.abspath(config_file))
        if (
            self.model_download_thread is not None
            and self.is_model_download_running()
        ):
            logger.info(
                "Another model is being loaded. Please wait for it to finish."
            )
            return False

        # Check config file path
        if not config_file or not os.path.isfile(config_file):
            logger.error(
                "An error occurred while loading the custom model: "
                "The model path is invalid."
            )
            self.new_model_status.emit(
                self.tr("Error in loading custom model: Invalid path.")
            )
            return False

        # Check config file content
        model_config = {}
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                model_config = load_model_config(f)
                model_config["config_file"] = os.path.abspath(config_file)
        except Exception as e:
            logger.error(
                "An error occurred while loading the custom model: "
                "The config file is invalid."
            )
            self.new_model_status.emit(
                self.tr("Error in loading custom model: Invalid config file.")
            )
            return False

        if (
            "type" not in model_config
            or "display_name" not in model_config
            or "name" not in model_config
            or model_config["type"] not in _CUSTOM_MODELS
        ):
            if "type" not in model_config:
                logger.error(
                    "An error occurred while loading the custom model: "
                    "The 'type' field is missing in the model configuration file."
                )
            elif "display_name" not in model_config:
                logger.error(
                    "An error occurred while loading the custom model: "
                    "The 'display_name' field is missing in the model configuration file."
                )
            elif "name" not in model_config:
                logger.error(
                    "An error occurred while loading the custom model: "
                    "The 'name' field is missing in the model configuration file."
                )
            else:
                logger.error(
                    "An error occurred while loading the custom model: "
                    "The model type {model_config['type']} is not supported."
                )
            self.new_model_status.emit(
                self.tr(
                    "Error in loading custom model: Invalid config file format."
                )
            )
            self.model_loaded.emit({})
            return False

        if not self.is_valid_custom_model_name(model_config["name"]):
            logger.error(
                "An error occurred while loading the custom model: "
                "The 'name' field must be a single path segment containing "
                "only letters, numbers, dots, underscores, and hyphens."
            )
            self.new_model_status.emit(
                self.tr("Error in loading custom model: Invalid model name.")
            )
            self.model_loaded.emit({})
            return False

        # Add or replace custom model
        custom_models = get_config().get("custom_models", [])
        matched_index = None
        for i, model in enumerate(custom_models):
            if os.path.normpath(model["config_file"]) == os.path.normpath(
                config_file
            ):
                matched_index = i
                break
        if matched_index is not None:
            model_config["last_used"] = time.time()
            custom_models[matched_index] = model_config
        else:
            if len(custom_models) >= self.MAX_NUM_CUSTOM_MODELS:
                custom_models.sort(
                    key=lambda x: x.get("last_used", 0), reverse=True
                )
                custom_models.pop()
            custom_models = [model_config] + custom_models

        # Save config
        config = get_config()
        config["custom_models"] = custom_models
        save_config(config)

        # Reload model configs
        self.load_model_configs()

        # Load model
        self.load_model(model_config["config_file"])

        return True

    def remove_custom_model(self, model_name):
        """Forget a custom model without deleting its config or weights."""
        if not model_name.startswith("_custom_"):
            return False
        if self.is_model_download_running() or (
            self.model_execution_thread is not None
            and self.model_execution_thread.isRunning()
        ):
            return False

        config = get_config()
        config["custom_models"] = [
            model
            for model in config.get("custom_models", [])
            if (
                model.get("name", "")
                if model.get("name", "").startswith("_custom_")
                else f"_custom_{model.get('name', '')}"
            )
            != model_name
        ]
        save_config(config)

        if (
            self.loaded_model_config is not None
            and self.loaded_model_config.get("name") == model_name
        ):
            self.unload_model()
            self.model_loaded.emit({})
            self.new_model_status.emit(self.tr("No model selected."))

        self.model_configs = [
            model
            for model in self.model_configs
            if not (
                model.get("is_custom_model", False)
                and model.get("name") == model_name
            )
        ]
        self.model_configs_changed.emit(self.model_configs)
        return True

    def load_model(self, config_file):
        """Run model loading in a thread"""
        if self.is_model_download_running():
            logger.info(
                "Another model is being loaded. Please wait for it to finish."
            )
            return
        if not config_file:
            if self.model_download_worker is not None:
                try:
                    self.model_download_worker.finished.disconnect(
                        self.on_model_download_finished
                    )
                except TypeError:
                    pass
            self.unload_model()
            self.new_model_status.emit(self.tr("No model selected."))
            return

        # Check and get model id
        model_id = None
        for i, model_config in enumerate(self.model_configs):
            if model_config["config_file"] == config_file:
                model_id = i
                break
        if model_id is None:
            logger.error(
                "An error occurred while loading the model: "
                "The model name is invalid."
            )
            self.new_model_status.emit(
                self.tr("Error in loading model: Invalid model name.")
            )
            return

        self._cancel_event.clear()
        self.model_download_thread = QThread()
        template = "Loading model: {model_name}. Please wait..."
        translated_template = self.tr(template)
        message = translated_template.format(
            model_name=self.model_configs[model_id]["display_name"]
        )
        self.new_model_status.emit(message)

        self.model_download_worker = GenericWorker(self._load_model, model_id)
        self.model_download_worker.finished.connect(
            self.on_model_download_finished
        )
        self.model_download_worker.finished.connect(
            self.model_download_thread.quit
        )
        self.model_download_thread.finished.connect(
            self.on_model_download_thread_finished
        )
        self.model_download_thread.finished.connect(
            self.model_download_thread.deleteLater
        )
        self.model_download_worker.moveToThread(self.model_download_thread)
        self.model_download_thread.started.connect(
            self.model_download_worker.run
        )
        self.model_download_thread.start()

    def is_model_download_running(self):
        """Return whether the model download thread is still running."""
        try:
            return (
                self.model_download_thread is not None
                and self.model_download_thread.isRunning()
            )
        except RuntimeError:
            self.model_download_thread = None
            self.model_download_worker = None
            return False

    @pyqtSlot()
    def on_model_download_thread_finished(self):
        """Clear finished model download thread references."""
        self.model_download_thread = None
        self.model_download_worker = None

    def _load_model(self, model_id):  # noqa: C901
        """Load and return model info"""
        with self.loaded_model_config_lock:
            old_config = self.loaded_model_config
            if old_config is not None:
                self.loaded_model_config = None
        if old_config is not None:
            old_config["model"].unload()
            self.auto_segmentation_model_unselected.emit()

        model_config = copy.deepcopy(self.model_configs[model_id])
        model_config["_cancel_event"] = self._cancel_event
        model_config["_on_progress"] = (
            lambda downloaded, total: self.download_progress.emit(
                downloaded, total
            )
        )
        model_type = model_config["type"]

        def _construct():
            if model_type in _TIMEOUT_MODELS or model_type == "remote_server":
                logger.info(f"⌛ Loading model: {model_type}")
            model_config["model"] = model_cls(
                model_config, on_message=self.new_model_status.emit
            )
            if model_type in _AUTO_SEGMENTATION_MODELS:
                self.auto_segmentation_model_selected.emit()
            else:
                self.auto_segmentation_model_unselected.emit()
            logger.info(f"✅ Model loaded successfully: {model_type}")

        try:
            model_cls = _get_model_class(model_type)
            if model_type in _TIMEOUT_MODELS:
                with TimeoutContext(
                    timeout=300,
                    timeout_message="""Model loading timeout! Please check your network connection.
                                    Alternatively, you can try to load the model from local directory.""",
                ) as ctx:
                    _ = ctx.run(_construct)
            else:
                _construct()
        except Exception as e:  # noqa
            template = "Error in loading model: {error_message}"
            translated_template = self.tr(template)
            error_text = translated_template.format(error_message=str(e))
            self.new_model_status.emit(error_text)
            error_model_type = (
                f" `{model_type}`"
                if model_type in _TIMEOUT_MODELS
                else f": {model_type}"
            )
            logger.error(
                f"❌ Error in loading model{error_model_type} with error: {str(e)}"
            )
            return

        if (
            model_type in _AUTO_SEGMENTATION_MODELS
            and model_type != "segment_anything_3"
        ):
            # Request next files for prediction
            self.request_next_files_requested.emit()

        with self.loaded_model_config_lock:
            self.loaded_model_config = model_config
        return model_config

    def set_cache_auto_label(self, text, gid):
        """Set cache auto label"""
        if (
            self.loaded_model_config is not None
            and self.loaded_model_config["type"]
            in _CACHED_AUTO_LABELING_MODELS
        ):
            self.loaded_model_config["model"].set_cache_auto_label(text, gid)

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks
        (For example, for segment_anything model, it is the marks for)
        """
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"]
            not in _AUTO_LABELING_MARKS_MODELS
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_marks(marks)

    def set_auto_labeling_api_token(self, token):
        """Set the API token for the model"""
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"]
            not in _AUTO_LABELING_API_TOKEN_MODELS
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_api_token(token)

    def set_auto_labeling_reset_tracker(self):
        """Resets the tracker to its initial state,
        clearing all tracked objects and internal states.
        """
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"]
            not in _AUTO_LABELING_RESET_TRACKER_MODELS
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_reset_tracker()

    def set_auto_labeling_conf(self, value):
        """Set auto labeling confidences"""
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"]
            not in _AUTO_LABELING_CONF_MODELS
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_conf(value)

    def set_auto_labeling_iou(self, value):
        """Set auto labeling iou"""
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"]
            not in _AUTO_LABELING_IOU_MODELS
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_iou(value)

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        if (
            self.loaded_model_config is not None
            and self.loaded_model_config["type"]
            in _AUTO_LABELING_PRESERVE_EXISTING_ANNOTATIONS_STATE_MODELS
        ):
            self.loaded_model_config[
                "model"
            ].set_auto_labeling_preserve_existing_annotations_state(state)

    def set_auto_labeling_prompt(self):
        if (
            self.loaded_model_config is not None
            and self.loaded_model_config["type"]
            in _AUTO_LABELING_PROMPT_MODELS
        ):
            self.loaded_model_config["model"].set_auto_labeling_prompt()

    def set_auto_labeling_filter_classes(self, class_names):
        """Set the active class filter by name on the loaded model."""
        if self.loaded_model_config is None:
            return
        model = self.loaded_model_config.get("model")
        if model and hasattr(model, "set_auto_labeling_filter_classes"):
            model.set_auto_labeling_filter_classes(class_names)

    def unload_model(self):
        """Unload model"""
        if self.loaded_model_config is not None:
            self.loaded_model_config["model"].unload()
            self.loaded_model_config = None

    def predict_shapes(
        self,
        image,
        filename=None,
        text_prompt=None,
        run_tracker=False,
        batch=False,
        existing_shapes=None,
    ):
        """Predict shapes.
        NOTE: This function is blocking. The model can take a long time to
        predict. So it is recommended to use predict_shapes_threading instead.
        """
        with self.loaded_model_config_lock:
            model_config = self.loaded_model_config
        if model_config is None:
            self.new_model_status.emit(
                self.tr("Model is not loaded. Choose a mode to continue.")
            )
            self.prediction_finished.emit()
            return

        try:
            if text_prompt is not None:
                auto_labeling_result = model_config["model"].predict_shapes(
                    image, filename, text_prompt=text_prompt
                )
            elif run_tracker is True:
                auto_labeling_result = model_config["model"].predict_shapes(
                    image, filename, run_tracker=run_tracker
                )
            elif existing_shapes is not None:
                auto_labeling_result = model_config["model"].predict_shapes(
                    image, filename, existing_shapes=existing_shapes
                )
            else:
                auto_labeling_result = model_config["model"].predict_shapes(
                    image, filename
                )

            if isinstance(auto_labeling_result, AutoLabelingResult):
                auto_labeling_result.image_path = filename

            if batch:
                return auto_labeling_result
            else:
                self.new_auto_labeling_result.emit(auto_labeling_result)
                self.new_model_status.emit(
                    self.tr("Finished inferencing AI model. Check the result.")
                )

        except Exception as e:  # noqa
            logger.error(f"Error in predict_shapes: {e}")
            template = "Error in model prediction: {error_message}"
            translated_template = self.tr(template)
            error_text = translated_template.format(error_message=str(e))
            self.new_model_status.emit(error_text)

        self.prediction_finished.emit()

    @pyqtSlot()
    def predict_shapes_threading(
        self,
        image,
        filename=None,
        text_prompt=None,
        run_tracker=False,
        existing_shapes=None,
    ):
        """Predict shapes.
        This function starts a thread to run the prediction.
        """
        with self.loaded_model_config_lock:
            _config_snapshot = self.loaded_model_config
        if _config_snapshot is None:
            self.new_model_status.emit(
                self.tr("Model is not loaded. Choose a mode to continue.")
            )
            return
        self.new_model_status.emit(
            self.tr("Inferencing AI model. Please wait...")
        )
        self.prediction_started.emit()

        with self.model_execution_thread_lock:
            try:
                execution_running = (
                    self.model_execution_thread is not None
                    and self.model_execution_thread.isRunning()
                )
            except RuntimeError:
                self.model_execution_thread = None
                self.model_execution_worker = None
                execution_running = False

            if execution_running:
                self.new_model_status.emit(
                    self.tr(
                        "Another model is being executed."
                        " Please wait for it to finish."
                    )
                )
                self.prediction_finished.emit()
                return

            self.model_execution_thread = QThread()
            if text_prompt is not None:
                self.model_execution_worker = GenericWorker(
                    self.predict_shapes,
                    image,
                    filename,
                    text_prompt=text_prompt,
                )
            elif run_tracker is True:
                self.model_execution_worker = GenericWorker(
                    self.predict_shapes,
                    image,
                    filename,
                    run_tracker=run_tracker,
                )
            elif existing_shapes is not None:
                self.model_execution_worker = GenericWorker(
                    self.predict_shapes,
                    image,
                    filename,
                    existing_shapes=existing_shapes,
                )
            else:
                self.model_execution_worker = GenericWorker(
                    self.predict_shapes, image, filename
                )
            self.model_execution_worker.finished.connect(
                self.model_execution_thread.quit
            )
            self.model_execution_thread.finished.connect(
                self.on_model_execution_finished
            )
            self.model_execution_thread.finished.connect(
                self.model_execution_thread.deleteLater
            )
            self.model_execution_worker.moveToThread(
                self.model_execution_thread
            )
            self.model_execution_thread.started.connect(
                self.model_execution_worker.run
            )
            self.model_execution_thread.start()

    @pyqtSlot()
    def on_model_execution_finished(self):
        """Clear finished model execution thread references."""
        with self.model_execution_thread_lock:
            self.model_execution_thread = None
            self.model_execution_worker = None

    def on_next_files_changed(self, next_files):
        """Run prediction on next files in advance to save inference time later"""
        if self.loaded_model_config is None:
            return

        # Currently only segment_anything-like model supports this feature
        if (
            self.loaded_model_config["type"]
            not in _ON_NEXT_FILES_CHANGED_MODELS
        ):
            return

        self.loaded_model_config["model"].on_next_files_changed(next_files)

    # Specific model setters
    def set_upn_mode(self, mode):
        """Set UPN mode"""
        if self.loaded_model_config is None:
            return

        if self.loaded_model_config["type"] == "upn":
            self.loaded_model_config["model"].set_upn_mode(mode)

    def set_groundingdino_mode(self, mode):
        """Set GroundingDino (API) mode"""
        if self.loaded_model_config is None:
            return

        if self.loaded_model_config["type"] == "grounding_dino_api":
            self.loaded_model_config["model"].set_groundingdino_mode(mode)

    def set_florence2_mode(self, mode):
        """Set Florence2 mode"""
        if self.loaded_model_config is None:
            return

        if self.loaded_model_config["type"] == "florence2":
            self.loaded_model_config["model"].set_florence2_mode(mode)

    def set_remote_server_model(self, model_id):
        """Set remote server model ID"""
        if self.loaded_model_config is None:
            return

        if self.loaded_model_config["type"] == "remote_server":
            self.loaded_model_config["model"].set_model_id(model_id)

    def get_remote_server_available_models(self):
        """Get available models from remote server"""
        if self.loaded_model_config is None:
            return {}

        if self.loaded_model_config["type"] == "remote_server":
            return self.loaded_model_config["model"].get_available_models()
        return {}

    def get_remote_server_current_model_id(self):
        """Get current remote server model ID"""
        if self.loaded_model_config is None:
            return None

        if self.loaded_model_config["type"] == "remote_server":
            return self.loaded_model_config["model"].current_model_id
        return None

    def set_task(self, task_id):
        """Set task ID for the current model"""
        if self.loaded_model_config is None:
            return

        if self.loaded_model_config["type"] == "remote_server":
            self.loaded_model_config["model"].set_task(task_id)

    def set_mask_fineness(self, epsilon):
        """Set mask fineness (epsilon value for Douglas-Peucker algorithm)"""
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"]
            not in _AUTO_LABELING_MASK_FINENESS_MODELS
        ):
            return
        self.loaded_model_config["model"].set_mask_fineness(epsilon)

    def set_cropping_mode(self, enabled: bool):
        """Set cropping mode for small object detection"""
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"]
            not in _AUTO_LABELING_CROPPING_MODE_MODELS
        ):
            return
        self.loaded_model_config["model"].set_cropping_mode(enabled)
