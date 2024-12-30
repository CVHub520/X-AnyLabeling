import os
import os.path as osp
import base64
import io
import shutil

import numpy as np
import PIL.ExifTags
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
        logger.error(f"Error reading image dimensions from {img_path}: {str(e)}")
        raise


def process_image_exif(filename):
    """Process image EXIF orientation and save if necessary."""
    with PIL.Image.open(filename) as img:
        exif_data = None
        if hasattr(img, "_getexif"):
            exif_data = img._getexif()
        if exif_data is not None:
            for tag, value in exif_data.items():
                tag_name = PIL.ExifTags.TAGS.get(tag, tag)
                if tag_name != "Orientation":
                    continue
                if value == 3:
                    img = img.rotate(180, expand=True)
                    rotation = "180 degrees"
                elif value == 6:
                    img = img.rotate(270, expand=True)
                    rotation = "270 degrees"
                elif value == 8:
                    img = img.rotate(90, expand=True)
                    rotation = "90 degrees"
                else:
                    return  # No rotation needed
                backup_dir = osp.join(
                    osp.dirname(osp.dirname(filename)),
                    "x-anylabeling-exif-backup",
                )
                os.makedirs(backup_dir, exist_ok=True)
                backup_filename = osp.join(backup_dir, osp.basename(filename))
                shutil.copy2(filename, backup_filename)
                img.save(filename)
                logger.info(
                    f"Rotated {filename} by {rotation}, saving backup to {backup_filename}"
                )
                break


def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        return image

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif.items()
        if k in PIL.ExifTags.TAGS
    }

    orientation = exif.get("Orientation", None)

    if orientation == 1:
        # do nothing
        return image
    if orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    if orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    if orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    if orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    if orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    if orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    if orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    return image
