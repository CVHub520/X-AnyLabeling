from PyQt6.QtGui import QColor, QImage

from anylabeling.views.labeling.utils.visualization import (
    _qimage_to_bgr_array,
)


def test_qimage_to_bgr_array_drops_alpha_and_preserves_channels():
    image = QImage(3, 1, QImage.Format.Format_ARGB32)
    image.setPixelColor(0, 0, QColor(255, 0, 0, 64))
    image.setPixelColor(1, 0, QColor(0, 255, 0, 128))
    image.setPixelColor(2, 0, QColor(0, 0, 255, 255))

    array = _qimage_to_bgr_array(image)

    assert array.shape == (1, 3, 3)
    assert array[0, 0].tolist() == [0, 0, 255]
    assert array[0, 1].tolist() == [0, 255, 0]
    assert array[0, 2].tolist() == [255, 0, 0]
