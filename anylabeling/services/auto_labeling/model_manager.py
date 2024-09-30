import os
import copy
import time
import yaml
import importlib.resources as pkg_resources
from threading import Lock


from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from anylabeling.utils import GenericWorker
from anylabeling.views.labeling.logger import logger
from anylabeling.config import get_config, save_config
from anylabeling.configs import auto_labeling as auto_labeling_configs
from anylabeling.services.auto_labeling.types import AutoLabelingResult


class ModelManager(QObject):
    """Model manager"""

    MAX_NUM_CUSTOM_MODELS = 5
    CUSTOM_MODELS = [
        "segment_anything",
        "segment_anything_2",
        "segment_anything_2_video",
        "sam_med2d",
        "sam_hq",
        "yolov5",
        "yolov6",
        "yolov7",
        "yolov8",
        "yolov8_seg",
        "yolox",
        "yolov5_resnet",
        "yolov6_face",
        "rtdetr",
        "yolo_nas",
        "yolox_dwpose",
        "clrnet",
        "ppocr_v4",
        "yolov5_sam",
        "efficientvit_sam",
        "yolov5_track",
        "damo_yolo",
        "yolov8_sahi",
        "grounding_sam",
        "grounding_sam2",
        "grounding_dino",
        "yolov5_obb",
        "gold_yolo",
        "yolov8_efficientvit_sam",
        "ram",
        "yolov5_seg",
        "yolov5_ram",
        "yolov8_pose",
        "pulc_attribute",
        "internimage_cls",
        "edge_sam",
        "yolov5_cls",
        "yolov8_cls",
        "yolov8_obb",
        "yolov5_car_plate",
        "rtmdet_pose",
        "yolov9",
        "yolow",
        "yolov10",
        "rmbg",
        "depth_anything",
        "depth_anything_v2",
        "yolow_ram",
        "rtdetrv2",
        "yolov8_det_track",
        "yolov8_seg_track",
        "yolov8_obb_track",
        "yolov8_pose_track",
        "yolo11",
        "yolo11_cls",
        "yolo11_obb",
        "yolo11_seg",
        "yolo11_pose",
        "yolo11_det_track",
        "yolo11_seg_track",
        "yolo11_obb_track",
        "yolo11_pose_track",
    ]

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

    def __init__(self):
        super().__init__()
        self.model_configs = []

        self.loaded_model_config = None
        self.loaded_model_config_lock = Lock()

        self.model_download_worker = None
        self.model_download_thread = None
        self.model_execution_thread = None
        self.model_execution_thread_lock = Lock()

        self.load_model_configs()

    def load_model_configs(self):
        """Load model configs"""
        # Load list of default models
        with pkg_resources.open_text(
            auto_labeling_configs, "models.yaml"
        ) as f:
            model_list = yaml.safe_load(f)

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
                with pkg_resources.open_text(
                    auto_labeling_configs, config_file_name
                ) as f:
                    model_config = yaml.safe_load(f)
                    model_config["config_file"] = config_file
            else:  # Config file is in local file system
                with open(config_file, "r", encoding="utf-8") as f:
                    model_config = yaml.safe_load(f)
                    model_config["config_file"] = os.path.normpath(
                        os.path.abspath(config_file)
                    )
            model_config["is_custom_model"] = model.get(
                "is_custom_model", False
            )
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

    def get_model_configs(self):
        """Return model infos"""
        return self.model_configs

    def set_output_mode(self, mode):
        """Set output mode"""
        if self.loaded_model_config and self.loaded_model_config["model"]:
            self.loaded_model_config["model"].set_output_mode(mode)

    @pyqtSlot()
    def on_model_download_finished(self):
        """Handle model download thread finished"""
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
            and self.model_download_thread.isRunning()
        ):
            logger.info(
                "Another model is being loaded. Please wait for it to finish."
            )
            return

        # Check config file path
        if not config_file or not os.path.isfile(config_file):
            logger.error(
                "An error occurred while loading the custom model: "
                "The model path is invalid."
            )
            self.new_model_status.emit(
                self.tr("Error in loading custom model: Invalid path.")
            )
            return

        # Check config file content
        model_config = {}
        with open(config_file, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
            model_config["config_file"] = os.path.abspath(config_file)
        if not model_config:
            logger.error(
                "An error occurred while loading the custom model: "
                "The config file is invalid."
            )
            self.new_model_status.emit(
                self.tr("Error in loading custom model: Invalid config file.")
            )
            return
        if (
            "type" not in model_config
            or "display_name" not in model_config
            or "name" not in model_config
            or model_config["type"] not in self.CUSTOM_MODELS
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
            return

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

    def load_model(self, config_file):
        """Run model loading in a thread"""
        if (
            self.model_download_thread is not None
            and self.model_download_thread.isRunning()
        ):
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

        self.model_download_thread = QThread()
        self.new_model_status.emit(
            self.tr("Loading model: {model_name}. Please wait...").format(
                model_name=self.model_configs[model_id]["display_name"]
            )
        )
        self.model_download_worker = GenericWorker(self._load_model, model_id)
        self.model_download_worker.finished.connect(
            self.on_model_download_finished
        )
        self.model_download_worker.finished.connect(
            self.model_download_thread.quit
        )
        self.model_download_worker.moveToThread(self.model_download_thread)
        self.model_download_thread.started.connect(
            self.model_download_worker.run
        )
        self.model_download_thread.start()

    def _load_model(self, model_id):  # noqa: C901
        """Load and return model info"""
        if self.loaded_model_config is not None:
            self.loaded_model_config["model"].unload()
            self.loaded_model_config = None
            self.auto_segmentation_model_unselected.emit()

        model_config = copy.deepcopy(self.model_configs[model_id])
        if model_config["type"] == "yolov5":
            from .yolov5 import YOLOv5

            try:
                model_config["model"] = YOLOv5(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov6":
            from .yolov6 import YOLOv6

            try:
                model_config["model"] = YOLOv6(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov7":
            from .yolov7 import YOLOv7

            try:
                model_config["model"] = YOLOv7(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov8_sahi":
            from .yolov8_sahi import YOLOv8_SAHI

            try:
                model_config["model"] = YOLOv8_SAHI(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov8":
            from .yolov8 import YOLOv8

            try:
                model_config["model"] = YOLOv8(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov9":
            from .yolov9 import YOLOv9

            try:
                model_config["model"] = YOLOv9(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov10":
            from .yolov10 import YOLOv10

            try:
                model_config["model"] = YOLOv10(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolo11":
            from .yolo11 import YOLO11

            try:
                model_config["model"] = YOLO11(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolow":
            from .yolow import YOLOW

            try:
                model_config["model"] = YOLOW(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov5_seg":
            from .yolov5_seg import YOLOv5_Seg

            try:
                model_config["model"] = YOLOv5_Seg(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov5_ram":
            from .yolov5_ram import YOLOv5_RAM

            try:
                model_config["model"] = YOLOv5_RAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolow_ram":
            from .yolow_ram import YOLOW_RAM

            try:
                model_config["model"] = YOLOW_RAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov8_seg":
            from .yolov8_seg import YOLOv8_Seg

            try:
                model_config["model"] = YOLOv8_Seg(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolo11_seg":
            from .yolo11_seg import YOLO11_Seg

            try:
                model_config["model"] = YOLO11_Seg(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov8_obb":
            from .yolov8_obb import YOLOv8_OBB

            try:
                model_config["model"] = YOLOv8_OBB(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolo11_obb":
            from .yolo11_obb import YOLO11_OBB

            try:
                model_config["model"] = YOLO11_OBB(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov8_pose":
            from .yolov8_pose import YOLOv8_Pose

            try:
                model_config["model"] = YOLOv8_Pose(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolo11_pose":
            from .yolo11_pose import YOLO11_Pose

            try:
                model_config["model"] = YOLO11_Pose(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolox":
            from .yolox import YOLOX

            try:
                model_config["model"] = YOLOX(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolo_nas":
            from .yolo_nas import YOLO_NAS

            try:
                model_config["model"] = YOLO_NAS(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "damo_yolo":
            from .damo_yolo import DAMO_YOLO

            try:
                model_config["model"] = DAMO_YOLO(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "gold_yolo":
            from .gold_yolo import Gold_YOLO

            try:
                model_config["model"] = Gold_YOLO(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "grounding_dino":
            from .grounding_dino import Grounding_DINO

            try:
                model_config["model"] = Grounding_DINO(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "ram":
            from .ram import RAM

            try:
                model_config["model"] = RAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "internimage_cls":
            from .internimage_cls import InternImage_CLS

            try:
                model_config["model"] = InternImage_CLS(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "pulc_attribute":
            from .pulc_attribute import PULC_Attribute

            try:
                model_config["model"] = PULC_Attribute(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov5_sam":
            from .yolov5_sam import YOLOv5SegmentAnything

            try:
                model_config["model"] = YOLOv5SegmentAnything(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "yolov8_efficientvit_sam":
            from .yolov8_efficientvit_sam import YOLOv8_EfficientViT_SAM

            try:
                model_config["model"] = YOLOv8_EfficientViT_SAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "grounding_sam":
            from .grounding_sam import GroundingSAM

            try:
                model_config["model"] = GroundingSAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "grounding_sam2":
            from .grounding_sam2 import GroundingSAM2

            try:
                model_config["model"] = GroundingSAM2(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "yolov5_obb":
            from .yolov5_obb import YOLOv5OBB

            try:
                model_config["model"] = YOLOv5OBB(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "segment_anything":
            from .segment_anything import SegmentAnything

            try:
                model_config["model"] = SegmentAnything(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "segment_anything_2":
            from .segment_anything_2 import SegmentAnything2

            try:
                model_config["model"] = SegmentAnything2(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "segment_anything_2_video":
            try:
                from .segment_anything_2_video import SegmentAnything2Video

                model_config["model"] = SegmentAnything2Video(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "efficientvit_sam":
            from .efficientvit_sam import EfficientViT_SAM

            try:
                model_config["model"] = EfficientViT_SAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "sam_med2d":
            from .sam_med2d import SAM_Med2D

            try:
                model_config["model"] = SAM_Med2D(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "edge_sam":
            from .edge_sam import EdgeSAM

            try:
                model_config["model"] = EdgeSAM(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "sam_hq":
            from .sam_hq import SAM_HQ

            try:
                model_config["model"] = SAM_HQ(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_selected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                return
            # Request next files for prediction
            self.request_next_files_requested.emit()
        elif model_config["type"] == "yolov5_resnet":
            from .yolov5_resnet import YOLOv5_ResNet

            try:
                model_config["model"] = YOLOv5_ResNet(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "rtdetr":
            from .rtdetr import RTDETR

            try:
                model_config["model"] = RTDETR(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "rtdetrv2":
            from .rtdetrv2 import RTDETRv2

            try:
                model_config["model"] = RTDETRv2(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov6_face":
            from .yolov6_face import YOLOv6Face

            try:
                model_config["model"] = YOLOv6Face(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolox_dwpose":
            from .yolox_dwpose import YOLOX_DWPose

            try:
                model_config["model"] = YOLOX_DWPose(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "rtmdet_pose":
            from .rtmdet_pose import RTMDet_Pose

            try:
                model_config["model"] = RTMDet_Pose(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "clrnet":
            from .clrnet import CLRNet

            try:
                model_config["model"] = CLRNet(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "ppocr_v4":
            from .ppocr_v4 import PPOCRv4

            try:
                model_config["model"] = PPOCRv4(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov5_cls":
            from .yolov5_cls import YOLOv5_CLS

            try:
                model_config["model"] = YOLOv5_CLS(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov5_car_plate":
            from .yolov5_car_plate import YOLOv5CarPlateDetRec

            try:
                model_config["model"] = YOLOv5CarPlateDetRec(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov8_cls":
            from .yolov8_cls import YOLOv8_CLS

            try:
                model_config["model"] = YOLOv8_CLS(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolo11_cls":
            from .yolo11_cls import YOLO11_CLS

            try:
                model_config["model"] = YOLO11_CLS(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov5_det_track":
            from .yolov5_det_track import YOLOv5_Det_Tracker

            try:
                model_config["model"] = YOLOv5_Det_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov8_det_track":
            from .yolov8_det_track import YOLOv8_Det_Tracker

            try:
                model_config["model"] = YOLOv8_Det_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolo11_det_track":
            from .yolo11_det_track import YOLO11_Det_Tracker

            try:
                model_config["model"] = YOLO11_Det_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov8_seg_track":
            from .yolov8_seg_track import YOLOv8_Seg_Tracker

            try:
                model_config["model"] = YOLOv8_Seg_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolo11_seg_track":
            from .yolo11_seg_track import YOLO11_Seg_Tracker

            try:
                model_config["model"] = YOLO11_Seg_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov8_obb_track":
            from .yolov8_obb_track import YOLOv8_Obb_Tracker

            try:
                model_config["model"] = YOLOv8_Obb_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolo11_obb_track":
            from .yolo11_obb_track import YOLO11_Obb_Tracker

            try:
                model_config["model"] = YOLO11_Obb_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolov8_pose_track":
            from .yolov8_pose_track import YOLOv8_Pose_Tracker

            try:
                model_config["model"] = YOLOv8_Pose_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "yolo11_pose_track":
            from .yolo11_pose_track import YOLO11_Pose_Tracker

            try:
                model_config["model"] = YOLO11_Pose_Tracker(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "rmbg":
            from .rmbg import RMBG

            try:
                model_config["model"] = RMBG(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "depth_anything":
            from .depth_anything import DepthAnything

            try:
                model_config["model"] = DepthAnything(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        elif model_config["type"] == "depth_anything_v2":
            from .depth_anything_v2 import DepthAnythingV2

            try:
                model_config["model"] = DepthAnythingV2(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                self.new_model_status.emit(
                    self.tr(
                        "Error in loading model: {error_message}".format(
                            error_message=str(e)
                        )
                    )
                )
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
        else:
            raise Exception(f"Unknown model type: {model_config['type']}")

        self.loaded_model_config = model_config
        return self.loaded_model_config

    def set_cache_auto_label(self, text, gid):
        """Set cache auto label"""
        valid_models = [
            "segment_anything_2_video",
        ]
        if (
            self.loaded_model_config is not None
            and self.loaded_model_config["type"] in valid_models
        ):
            self.loaded_model_config["model"].set_cache_auto_label(text, gid)

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks
        (For example, for segment_anything model, it is the marks for)
        """
        marks_model_list = [
            "segment_anything",
            "segment_anything_2",
            "segment_anything_2_video",
            "sam_med2d",
            "sam_hq",
            "yolov5_sam",
            "efficientvit_sam",
            "yolov8_efficientvit_sam",
            "grounding_sam",
            "grounding_sam2",
            "edge_sam",
        ]
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"] not in marks_model_list
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_marks(marks)

    def set_auto_labeling_reset_tracker(self):
        """Resets the tracker to its initial state,
        clearing all tracked objects and internal states.
        """
        model_list = [
            "yolov5_det_track",
            "yolov8_det_track",
            "yolov8_obb_track",
            "yolov8_seg_track",
            "yolov8_pose_track",
            "segment_anything_2_video",
            "yolo11_det_track",
            "yolo11_seg_track",
            "yolo11_obb_track",
            "yolo11_pose_track",
        ]
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"] not in model_list
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_reset_tracker()

    def set_auto_labeling_conf(self, value):
        """Set auto labeling confidences"""
        model_list = [
            "damo_yolo",
            "gold_yolo",
            "grounding_dino",
            "rtdetr",
            "rtdetrv2",
            "yolo_nas",
            "yolov5_obb",
            "yolov5_seg",
            "yolov5_det_track",
            "yolov5",
            "yolov6",
            "yolov6_face",
            "yolov7",
            "yolov8_obb",
            "yolov8_pose",
            "yolov8_seg",
            "yolov8_det_track",
            "yolov8_seg_track",
            "yolov8_obb_track",
            "yolov8_pose_track",
            "yolov8",
            "yolov9",
            "yolov10",
            "yolo11",
            "yolo11_obb",
            "yolo11_seg",
            "yolo11_pose",
            "yolo11_det_track",
            "yolo11_seg_track",
            "yolo11_obb_track",
            "yolo11_pose_track",
            "yolow",
            "yolox",
        ]
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"] not in model_list
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_conf(value)

    def set_auto_labeling_iou(self, value):
        """Set auto labeling iou"""
        model_list = [
            "damo_yolo",
            "gold_yolo",
            "yolo_nas",
            "yolov5_obb",
            "yolov5_seg",
            "yolov5_det_track",
            "yolov5",
            "yolov6",
            "yolov7",
            "yolov8_obb",
            "yolov8_pose",
            "yolov8_seg",
            "yolov8_det_track",
            "yolov8_seg_track",
            "yolov8_obb_track",
            "yolov8_pose_track",
            "yolov8",
            "yolov9",
            "yolo11",
            "yolo11_obb",
            "yolo11_seg",
            "yolo11_pose",
            "yolo11_det_track",
            "yolo11_seg_track",
            "yolo11_obb_track",
            "yolo11_pose_track",
            "yolox",
        ]
        if (
            self.loaded_model_config is None
            or self.loaded_model_config["type"] not in model_list
        ):
            return
        self.loaded_model_config["model"].set_auto_labeling_iou(value)

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        model_list = [
            "damo_yolo",
            "gold_yolo",
            "grounding_dino",
            "rtdetr",
            "rtdetrv2",
            "yolo_nas",
            "yolov5_obb",
            "yolov5_seg",
            "yolov5_det_track",
            "yolov5",
            "yolov6",
            "yolov7",
            "yolov8_obb",
            "yolov8_pose",
            "yolov8_seg",
            "yolov8_det_track",
            "yolov8_seg_track",
            "yolov8_obb_track",
            "yolov8_pose_track",
            "yolov8",
            "yolov9",
            "yolov10",
            "yolo11",
            "yolo11_obb",
            "yolo11_seg",
            "yolo11_pose",
            "yolo11_det_track",
            "yolo11_seg_track",
            "yolo11_obb_track",
            "yolo11_pose_track",
            "yolow",
            "yolox",
        ]
        if (
            self.loaded_model_config is not None
            and self.loaded_model_config["type"] in model_list
        ):
            self.loaded_model_config[
                "model"
            ].set_auto_labeling_preserve_existing_annotations_state(state)

    def set_auto_labeling_prompt(self):
        model_list = ["segment_anything_2_video"]
        if (
            self.loaded_model_config is not None
            and self.loaded_model_config["type"] in model_list
        ):
            self.loaded_model_config["model"].set_auto_labeling_prompt()

    def unload_model(self):
        """Unload model"""
        if self.loaded_model_config is not None:
            self.loaded_model_config["model"].unload()
            self.loaded_model_config = None

    def predict_shapes(
        self, image, filename=None, text_prompt=None, run_tracker=False
    ):
        """Predict shapes.
        NOTE: This function is blocking. The model can take a long time to
        predict. So it is recommended to use predict_shapes_threading instead.
        """
        if self.loaded_model_config is None:
            self.new_model_status.emit(
                self.tr("Model is not loaded. Choose a mode to continue.")
            )
            self.prediction_finished.emit()
            return
        try:
            if text_prompt is not None:
                auto_labeling_result = self.loaded_model_config[
                    "model"
                ].predict_shapes(image, filename, text_prompt=text_prompt)
            elif run_tracker is True:
                auto_labeling_result = self.loaded_model_config[
                    "model"
                ].predict_shapes(image, filename, run_tracker=run_tracker)
            else:
                auto_labeling_result = self.loaded_model_config[
                    "model"
                ].predict_shapes(image, filename)
            self.new_auto_labeling_result.emit(auto_labeling_result)
            self.new_model_status.emit(
                self.tr("Finished inferencing AI model. Check the result.")
            )
        except Exception as e:  # noqa
            logger.error(f"Error in predict_shapes: {e}")
            self.new_model_status.emit(
                self.tr(
                    f"Error in model prediction: {e}. Please check the model."
                )
            )
        self.prediction_finished.emit()

    @pyqtSlot()
    def predict_shapes_threading(
        self, image, filename=None, text_prompt=None, run_tracker=False
    ):
        """Predict shapes.
        This function starts a thread to run the prediction.
        """
        if self.loaded_model_config is None:
            self.new_model_status.emit(
                self.tr("Model is not loaded. Choose a mode to continue.")
            )
            return
        self.new_model_status.emit(
            self.tr("Inferencing AI model. Please wait...")
        )
        self.prediction_started.emit()

        with self.model_execution_thread_lock:
            if (
                self.model_execution_thread is not None
                and self.model_execution_thread.isRunning()
            ):
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
            else:
                self.model_execution_worker = GenericWorker(
                    self.predict_shapes, image, filename
                )
            self.model_execution_worker.finished.connect(
                self.model_execution_thread.quit
            )
            self.model_execution_worker.moveToThread(
                self.model_execution_thread
            )
            self.model_execution_thread.started.connect(
                self.model_execution_worker.run
            )
            self.model_execution_thread.start()

    def on_next_files_changed(self, next_files):
        """Run prediction on next files in advance to save inference time later"""
        if self.loaded_model_config is None:
            return

        # Currently only segment_anything-like model supports this feature
        if self.loaded_model_config["type"] not in [
            "segment_anything",
            "segment_anything_2",
            "sam_med2d",
            "sam_hq",
            "yolov5_sam",
            "efficientvit_sam",
            "yolov8_efficientvit_sam",
            "grounding_sam",
            "grounding_sam2",
            "edge_sam",
        ]:
            return

        self.loaded_model_config["model"].on_next_files_changed(next_files)
