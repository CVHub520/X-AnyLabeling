import numpy as np
import PIL.Image

from PyQt6.QtGui import QImage

from anylabeling.views.labeling.widgets.brightness_contrast_dialog import (
    BrightnessContrastDialog,
)


def _run_dialog_on_image(qtbot, img):
    received = []

    def callback(qimage):
        received.append(qimage)

    dialog = BrightnessContrastDialog(callback)
    qtbot.addWidget(dialog)
    dialog.update_image(img)
    dialog.slider_brightness.setValue(60)
    dialog.slider_contrast.setValue(60)
    dialog.on_new_value()
    assert received
    assert isinstance(received[-1], QImage)
    assert received[-1].width() == img.size[0]
    assert received[-1].height() == img.size[1]


def test_brightness_contrast_rgb(qtbot):
    img = PIL.Image.new("RGB", (32, 16), color=(10, 20, 30))
    _run_dialog_on_image(qtbot, img)


def test_brightness_contrast_rgba(qtbot):
    img = PIL.Image.new("RGBA", (32, 16), color=(10, 20, 30, 200))
    _run_dialog_on_image(qtbot, img)


def test_brightness_contrast_l(qtbot):
    img = PIL.Image.new("L", (32, 16), color=128)
    _run_dialog_on_image(qtbot, img)


def test_brightness_contrast_palette(qtbot):
    img = PIL.Image.new("RGB", (32, 16), color=(10, 20, 30)).convert("P")
    _run_dialog_on_image(qtbot, img)


def test_brightness_contrast_16bit(qtbot):
    arr = (np.arange(32 * 16, dtype=np.uint16).reshape((16, 32)) * 4) % 65535
    img = PIL.Image.fromarray(arr)
    _run_dialog_on_image(qtbot, img)
