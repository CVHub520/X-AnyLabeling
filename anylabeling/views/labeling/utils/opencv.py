import os.path

import cv2
import numpy as np
import qimage2ndarray
from PyQt5 import QtGui
from PyQt5.QtGui import QImage


def qt_img_to_rgb_cv_img(qt_img, img_path=None):
    """
    Convert 8bit/16bit RGB image or 8bit/16bit Gray image to 8bit RGB image
    """
    if img_path is not None and os.path.exists(img_path):
        # Load Image From Path Directly
        cv_image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    else:
        if (
            qt_img.format() == QImage.Format_RGB32
            or qt_img.format() == QImage.Format_ARGB32
            or qt_img.format() == QImage.Format_ARGB32_Premultiplied
        ):
            cv_image = qimage2ndarray.rgb_view(qt_img)
        else:
            cv_image = qimage2ndarray.raw_view(qt_img)
    # To uint8
    if cv_image.dtype != np.uint8:
        cv2.normalize(cv_image, cv_image, 0, 255, cv2.NORM_MINMAX)
        cv_image = np.array(cv_image, dtype=np.uint8)
    # To RGB
    if len(cv_image.shape) == 2 or cv_image.shape[2] == 1:
        cv_image = cv2.merge([cv_image, cv_image, cv_image])
    return cv_image


def qt_img_to_cv_img(in_image):
    return qimage2ndarray.rgb_view(in_image)


def cv_img_to_qt_img(in_mat):
    return QtGui.QImage(qimage2ndarray.array2qimage(in_mat))
