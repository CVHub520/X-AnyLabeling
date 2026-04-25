import unittest
from unittest import mock

from PIL import Image
from PyQt6 import QtGui

from anylabeling.views.labeling.utils.image import (
    get_supported_image_extensions,
    img_data_to_qimage,
)


class TestImageUtils(unittest.TestCase):

    def test_supported_image_extensions_include_heif_variants(self):
        extensions = get_supported_image_extensions()

        self.assertIn(".heic", extensions)
        self.assertIn(".heif", extensions)

    def test_img_data_to_qimage_falls_back_to_pil(self):
        with mock.patch.object(
            QtGui.QImage,
            "fromData",
            return_value=QtGui.QImage(),
        ):
            with mock.patch(
                "anylabeling.views.labeling.utils.image.img_data_to_pil",
                return_value=Image.new("RGB", (2, 3), "white"),
            ):
                image = img_data_to_qimage(
                    b"not-a-qt-image", "sample.heic"
                )

        self.assertFalse(image.isNull())
        self.assertEqual((image.width(), image.height()), (2, 3))

    def test_img_data_to_qimage_skips_pil_for_non_heif(self):
        with mock.patch.object(
            QtGui.QImage,
            "fromData",
            return_value=QtGui.QImage(),
        ):
            with mock.patch(
                "anylabeling.views.labeling.utils.image.img_data_to_pil"
            ) as mocked_img_data_to_pil:
                image = img_data_to_qimage(
                    b"not-a-qt-image", "sample.jpg"
                )

        self.assertTrue(image.isNull())
        mocked_img_data_to_pil.assert_not_called()
