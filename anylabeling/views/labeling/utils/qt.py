import natsort
import os
import os.path as osp
from math import sqrt

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.logger import logger


def scan_all_images(folder_path):
    try:
        extensions = [
            f".{fmt.data().decode().lower()}"
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]

        images = []
        folder_path = osp.normpath(osp.abspath(folder_path))

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relative_path = osp.normpath(osp.join(root, file))
                    relative_path = str(relative_path)
                    images.append(relative_path)

        try:
            return natsort.natsorted(images)
        except (OSError, ValueError) as e:
            logger.warning(
                f"Warning: Natural sort failed, falling back to regular sort: {e}"
            )
            return sorted(images)
    except Exception as e:
        logger.error(f"Error scanning images: {e}")
        return []


def new_icon(icon, ext="png"):
    return QtGui.QIcon(osp.join(f":/images/images/{icon}.{ext}"))


def new_icon_path(icon, ext="png"):
    """Returns the resource path string for an icon."""
    return f":/images/images/{icon}.{ext}"


def new_button(text, icon=None, slot=None):
    b = QtWidgets.QPushButton(text)
    if icon is not None:
        b.setIcon(new_icon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def new_action(
    parent,
    text,
    slot=None,
    shortcut=None,
    icon=None,
    tip=None,
    checkable=False,
    enabled=True,
    checked=False,
    auto_trigger=False,
):
    """Create a new action and assign callbacks, shortcuts, etc."""
    action = QtWidgets.QAction(text, parent)
    if icon is not None:
        action.setIconText(text.replace(" ", "\n"))
        action.setIcon(new_icon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            action.setShortcuts(shortcut)
        else:
            action.setShortcut(shortcut)
    if tip is not None:
        action.setToolTip(tip)
        action.setStatusTip(tip)
    if slot is not None:
        action.triggered.connect(slot)
    if checkable:
        action.setCheckable(True)
    action.setEnabled(enabled)
    action.setChecked(checked)
    if auto_trigger:
        action.triggered.emit(checked)
    return action


def add_actions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QtWidgets.QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def label_validator():
    return QtGui.QRegularExpressionValidator(
        QtCore.QRegularExpression(r"^[^ \t].+"), None
    )


class Struct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def distance_to_line(point, line):
    p1, p2 = line
    p1 = np.array([p1.x(), p1.y()])
    p2 = np.array([p2.x(), p2.y()])
    p3 = np.array([point.x(), point.y()])
    if np.dot((p3 - p1), (p2 - p1)) < 0:
        return np.linalg.norm(p3 - p1)
    if np.dot((p3 - p2), (p1 - p2)) < 0:
        return np.linalg.norm(p3 - p2)
    if np.linalg.norm(p2 - p1) == 0:
        return 0
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fmt_shortcut(text):
    mod, key = text.split("+", 1)
    return f"<b>{mod}</b>+<b>{key}</b>"
