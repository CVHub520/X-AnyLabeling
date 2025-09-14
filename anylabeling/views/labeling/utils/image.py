import os
import os.path as osp
import base64
import io
import shutil

import numpy as np
import PIL.Image
import PIL.ImageOps

from PyQt5 import QtGui

from ...labeling.logger import logger


def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = PIL.Image.open(f)
    return img_pil


def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr


def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


def img_pil_to_data(img_pil):
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_data = f.getvalue()
    return img_data


def pil_to_qimage(img):
    """Convert PIL Image to QImage."""
    img = img.convert("RGBA")  # Ensure image is in RGBA format
    data = np.array(img)
    height, width, channel = data.shape
    bytes_per_line = 4 * width
    qimage = QtGui.QImage(
        data, width, height, bytes_per_line, QtGui.QImage.Format_RGBA8888
    )
    return qimage


def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_bin = f.getvalue()
    if hasattr(base64, "encodebytes"):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64


def img_data_to_png_data(img_data):
    with io.BytesIO() as f:
        f.write(img_data)
        img = PIL.Image.open(f)

        with io.BytesIO() as f:
            img.save(f, "PNG")
            f.seek(0)
            return f.read()


def get_pil_img_dim(img_path):
    """
    Get the dimensions of a PIL image.

    Args:
        img_path (str or bytes or PIL.Image.Image): The path to the image file or the image data.

    Returns:
        tuple: The dimensions of the image (width, height).
    """
    try:
        if isinstance(img_path, str):
            with PIL.Image.open(img_path) as img:
                return img.size[0], img.size[1]
        elif isinstance(img_path, bytes):
            with PIL.Image.open(io.BytesIO(img_path)) as img:
                return img.size[0], img.size[1]
        elif isinstance(img_path, PIL.Image.Image):
            return img_path.size[0], img_path.size[1]
        else:
            raise ValueError(f"Invalid image path type: {type(img_path)}")

    except Exception as e:
        logger.error(
            f"Error reading image dimensions from {img_path}: {str(e)}"
        )
        raise


def check_img_exif(filename):
    """Check if image needs EXIF orientation correction"""
    try:
        with PIL.Image.open(filename) as img:
            exif = img.getexif()
            orientation = exif.get(0x0112, 1)
            return orientation not in (1, None)

    except Exception:
        return False


def process_image_exif(filename):
    """Process image EXIF orientation."""
    try:
        with PIL.Image.open(filename) as img:
            exif = img.getexif()
            orientation = exif.get(0x0112, 1)
            if orientation in (1, None):
                return

            corrected_img = PIL.ImageOps.exif_transpose(img)

            backup_dir = osp.join(
                osp.dirname(osp.dirname(filename)),
                "x-anylabeling-exif-backup",
            )
            os.makedirs(backup_dir, exist_ok=True)
            backup_filename = osp.join(backup_dir, osp.basename(filename))
            shutil.copy2(filename, backup_filename)
            corrected_img.save(filename)

    except Exception as e:
        logger.error(f"Error processing EXIF orientation for {filename}: {e}")
