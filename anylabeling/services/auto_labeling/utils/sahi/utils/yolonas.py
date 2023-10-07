import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class YoloNasTestConstants:
    YOLONAS_S_MODEL_URL = "https://sghub.deci.ai/models/yolo_nas_s_coco.pth"
    YOLONAS_S_MODEL_PATH = "tests/data/models/yolonas/yolo_nas_s_coco.pt"

    YOLONAS_M_MODEL_URL = "https://sghub.deci.ai/models/yolo_nas_m_coco.pth"
    YOLONAS_M_MODEL_PATH = "tests/data/models/yolonas/yolo_nas_m_coco.pt"

    YOLONAS_L_MODEL_URL = "https://sghub.deci.ai/models/yolo_nas_l_coco.pth"
    YOLONAS_L_MODEL_PATH = "tests/data/models/yolonas/yolo_nas_l_coco.pt"


def download_yolonas_s_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = YoloNasTestConstants.YOLONAS_S_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            YoloNasTestConstants.YOLONAS_S_MODEL_URL,
            destination_path,
        )


def download_yolonas_m_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = YoloNasTestConstants.YOLONAS_M_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            YoloNasTestConstants.YOLONAS_M_MODEL_URL,
            destination_path,
        )


def download_yolonas_l_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = YoloNasTestConstants.YOLONAS_L_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            YoloNasTestConstants.YOLONAS_L_MODEL_URL,
            destination_path,
        )
