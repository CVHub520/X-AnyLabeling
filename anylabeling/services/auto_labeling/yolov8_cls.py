import logging
import numpy as np

from PyQt5 import QtCore

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .types import AutoLabelingResult
from .yolov5_cls import YOLOv5_CLS
from .utils import softmax


class YOLOv8_CLS(YOLOv5_CLS):
    pass
