import os
import multiprocessing

from anylabeling.config import get_work_directory

_trainer_root_dir = None


def configure_trainer_root_dir(path=None):
    global _trainer_root_dir
    _trainer_root_dir = path


def get_trainer_root_dir():
    if _trainer_root_dir:
        return _trainer_root_dir
    return os.path.join(
        get_work_directory(), "xanylabeling_data", "trainer", "ultralytics"
    )


def get_data_path():
    return os.path.join(get_trainer_root_dir(), "data.yaml")


def get_dataset_path():
    return os.path.join(get_trainer_root_dir(), "datasets")


def get_settings_config_path():
    return os.path.join(get_trainer_root_dir(), "settings.json")


def get_default_project_dir():
    return os.path.join(get_trainer_root_dir(), "runs")


# UI configuration
DEFAULT_WINDOW_TITLE = "Ultralytics Training Platforms 🚀"
DEFAULT_WINDOW_SIZE = (1200, 800)  # (w, h)
ICON_SIZE_NORMAL = (32, 32)
ICON_SIZE_SMALL = (16, 16)

# Task configuration
TASK_TYPES = ["Classify", "Detect", "OBB", "Segment", "Pose"]
TASK_SHAPE_MAPPINGS = {
    "Classify": ["flags"],
    "Detect": ["rectangle"],
    "OBB": ["rotation"],
    "Segment": ["polygon"],
    "Pose": ["point"],
}
TASK_LABEL_MAPPINGS = {
    "Classify": "classify",
    "Detect": "hbb",
    "OBB": "obb",
    "Segment": "seg",
    "Pose": "pose",
}

# Training configuration
MIN_LABELED_IMAGES_THRESHOLD = 0
NUM_WORKERS = multiprocessing.cpu_count()
DEFAULT_TRAINING_CONFIG = {
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "workers": 8,
    "classes": "",
    "single_cls": False,
    "time": 0,
    "patience": 100,
    "close_mosaic": 10,
    "optimizer": "auto",
    "cos_lr": False,
    "amp": True,
    "multi_scale": False,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "dropout": 0.0,
    "fraction": 1.0,
    "rect": False,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "pose": 12.0,
    "kobj": 2.0,
    "save_period": -1,
    "val": True,
    "plots": True,
    "save": True,
    "resume": False,
    "cache": False,
}
OPTIMIZER_OPTIONS = [
    "auto",
    "SGD",
    "Adam",
    "AdamW",
    "NAdam",
    "RAdam",
    "RMSProp",
]
TRAINING_STATUS_COLORS = {
    "idle": "#6c757d",
    "training": "#6f42c1",
    "completed": "#28a745",
    "error": "#ffc107",
}
TRAINING_STATUS_TEXTS = {
    "idle": "Ready to train",
    "training": "Training in progress",
    "completed": "Training completed",
    "error": "Training error",
}


# Env Check
DEVICE_OPTIONS = ["cpu"]
