import base64
import json
import os.path as osp

import PIL.Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from . import utils
from .label_converter import LabelConverter
from .logger import logger
from .schema import XLABEL_BASIC_FIELDS, create_xlabel_template
from .shape import Shape

PIL.Image.MAX_IMAGE_PIXELS = None


class LabelFileError(Exception):
    pass


class LabelFile:
    suffix = ".json"

    def __init__(self, filename=None, image_dir=None):
        self.shapes = []
        self.image_path = None
        self.image_data = None
        self.image_dir = image_dir
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def _check_image_height_and_width(image_data, image_height, image_width):
        img_arr = utils.img_b64_to_arr(image_data)
        if image_height is not None and img_arr.shape[0] != image_height:
            logger.error(
                "image_height does not match with image_data or image_path, "
                "so getting image_height from actual image."
            )
            image_height = img_arr.shape[0]
        if image_width is not None and img_arr.shape[1] != image_width:
            logger.error(
                "image_width does not match with image_data or image_path, "
                "so getting image_width from actual image."
            )
            image_width = img_arr.shape[1]
        return image_height, image_width

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix

    @staticmethod
    def load_image_file(filename, default=None):
        try:
            with open(filename, "rb") as f:
                return f.read()
        except Exception:
            logger.error(f"Failed opening image file: {filename}")
            return default

    def load(self, filename):
        try:
            with utils.io_open(filename, "r") as f:
                data = json.load(f)

            if data.get("version") is None:
                logger.warning(
                    f"Loading JSON file ({filename}) of unknown version"
                )

            if data["shapes"]:
                for i in range(len(data["shapes"])):
                    shape_points = data["shapes"][i]["points"]
                    if (
                        data["shapes"][i]["shape_type"] == "rectangle"
                        and len(shape_points) == 2
                    ):
                        logger.warning(
                            "UserWarning: Diagonal vertex mode is deprecated in X-AnyLabeling release v2.2.0 or later.\n"
                            "Please update your code to accommodate the new four-point mode."
                        )
                        data["shapes"][i]["points"] = (
                            utils.rectangle_from_diagonal(shape_points)
                        )

            data["imagePath"] = osp.basename(data["imagePath"])
            if data["imageData"] is not None:
                image_data = base64.b64decode(data["imageData"])
            else:
                # relative path from label file to relative path from cwd
                if self.image_dir:
                    image_path = osp.join(self.image_dir, data["imagePath"])
                else:
                    image_path = osp.join(
                        osp.dirname(filename), data["imagePath"]
                    )
                image_data = self.load_image_file(image_path)

            flags = data.get("flags", {})
            image_path = data["imagePath"]

            self._check_image_height_and_width(
                base64.b64encode(image_data).decode("utf-8"),
                data.get("imageHeight"),
                data.get("imageWidth"),
            )

            shapes = [Shape().load_from_dict(s) for s in data["shapes"]]

        except Exception as e:  # noqa
            raise LabelFileError(e) from e

        other_data = {}
        for key, value in data.items():
            if key not in XLABEL_BASIC_FIELDS:
                other_data[key] = value

        # Add new fields if not available
        other_data["description"] = other_data.get("description", "")

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.image_path = image_path
        self.image_data = image_data
        self.filename = filename
        self.other_data = other_data

    def save(
        self,
        filename=None,
        shapes=None,
        image_path=None,
        image_height=None,
        image_width=None,
        image_data=None,
        other_data=None,
        flags=None,
    ):
        if image_data is not None:
            image_data = base64.b64encode(image_data).decode("utf-8")
            image_height, image_width = self._check_image_height_and_width(
                image_data, image_height, image_width
            )

        if other_data is None:
            other_data = {}
        if flags is None:
            flags = {}
        for i, shape in enumerate(shapes):
            if shape["shape_type"] == "rectangle":
                sorted_box = LabelConverter.calculate_bounding_box(
                    shape["points"]
                )
                xmin, ymin, xmax, ymax = sorted_box
                shape["points"] = [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                ]
                shapes[i] = shape

        data = create_xlabel_template(
            flags=flags,
            shapes=shapes,
            image_path=image_path,
            image_data=image_data,
            image_height=image_height,
            image_width=image_width,
        )

        for key, value in other_data.items():
            assert key not in data
            data[key] = value
        try:
            with utils.io_open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:  # noqa
            raise LabelFileError(e) from e
