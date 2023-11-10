import os
import base64
import contextlib
import io
import json
import os.path as osp

import PIL.Image

from ...app_info import __version__
from . import utils
from .logger import logger
from .label_converter import LabelConverter

PIL.Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def io_open(name, mode):
    assert mode in ["r", "w"]
    encoding = "utf-8"
    yield io.open(name, mode, encoding=encoding)


class LabelFileError(Exception):
    pass


class LabelFile:
    suffix = ".json"

    def __init__(self, filename=None):
        self.shapes = []
        self.image_path = None
        self.image_data = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error("Failed opening image file: %s", filename)
            return None

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                image_pil = image_pil.convert("RGB")
                img_format = "JPEG"
            else:
                img_format = "PNG"
            image_pil.save(f, format=img_format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        shape_keys = [
            "label",
            "text",
            "points",
            "group_id",
            "shape_type",
            "flags",
            "attributes",
        ]
        try:
            with io_open(filename, "r") as f:
                data = json.load(f)
            version = data.get("version")
            if version is None:
                logger.warning(
                    "Loading JSON file (%s) of unknown version", filename
                )

            if data["imageData"] is not None:
                image_data = base64.b64decode(data["imageData"])
            else:
                # relative path from label file to relative path from cwd
                image_path = osp.join(osp.dirname(filename), data["imagePath"])
                image_data = self.load_image_file(image_path)
            flags = data.get("flags") or {}
            image_path = data["imagePath"]
            self._check_image_height_and_width(
                base64.b64encode(image_data).decode("utf-8"),
                data.get("imageHeight"),
                data.get("imageWidth"),
            )
            shapes = [
                {
                    "label": s["label"],
                    "text": s.get("text", ""),
                    "points": s["points"],
                    "shape_type": s.get("shape_type", "polygon"),
                    "flags": s.get("flags", {}),
                    "group_id": s.get("group_id"),
                    "attributes": s.get("attributes", {}),
                    "other_data": {
                        k: v for k, v in s.items() if k not in shape_keys
                    },
                }
                for s in data["shapes"]
            ]
            for i, s in enumerate(data["shapes"]):
                if s.get("shape_type", "polygon") == "rotation":
                    shapes[i]["direction"] = s.get("direction", 0)
        except Exception as e:  # noqa
            raise LabelFileError(e) from e

        other_data = {}
        for key, value in data.items():
            if key not in keys:
                other_data[key] = value

        # Add new fields if not available
        other_data["text"] = other_data.get("text", "")

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.image_path = image_path
        self.image_data = image_data
        self.filename = filename
        self.other_data = other_data

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
        output_format='defalut',
        classes_file=None,
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
        data = {
            "version": __version__,
            "flags": flags,
            "shapes": shapes,
            "imagePath": image_path,
            "imageData": image_data,
            "imageHeight": image_height,
            "imageWidth": image_width,
        }

        for key, value in other_data.items():
            assert key not in data
            data[key] = value
        try:
            with io_open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
            _ = self.save_other_mode(data, output_format, classes_file)
        except Exception as e:  # noqa
            raise LabelFileError(e) from e

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix

    def save_other_mode(self, data, mode, classes_file=None):
        target_formats = ["polygon", "rectangle", "rotation"]
        shape_type = self.get_shape_type(data, target_formats)
        if mode == "default" or not shape_type:
            return False

        root_path, file_name = osp.split(self.filename)
        base_name = osp.splitext(file_name)[0]

        if mode == "yolo":
            save_path = root_path + '/labels'
            dst_file = save_path + '/' + base_name+'.txt'
            os.makedirs(save_path, exist_ok=True)
        elif mode == "coco":
            pass
        elif mode == "voc":
            save_path = root_path + '/Annotations'
            dst_file = save_path + '/' + base_name+'.xml'
            os.makedirs(save_path, exist_ok=True)
        elif mode == "dota":
            save_path = root_path + '/labelTxt'
            dst_file = save_path + '/' + base_name+'.txt'
            os.makedirs(save_path, exist_ok=True)
        elif mode == "mot":
            dst_file = root_path + '/' + base_name.rsplit("_", 1)[0]+'.csv'

        converter = LabelConverter(classes_file=classes_file)
        if mode == "yolo" and shape_type == "rectangle":
            converter.custom_to_yolo_rectangle(data, dst_file)
            return True
        elif mode == "yolo" and shape_type == "polygon":
            converter.custom_to_yolo_polygon(data, dst_file)
            return True
        if mode == "coco" and shape_type in ["rectangle", "polygon"]:
            pass
            return True
        elif mode == "voc" and shape_type == "rectangle":
            converter.custom_to_voc_rectangle(data, dst_file)
            return True
        elif mode == "dota" and shape_type == "rotation":
            converter.custom_to_dota(data, dst_file)
            return True
        elif mode == "mot" and shape_type == "rectangle":
            converter.custom_to_mot_rectangle(data, dst_file, base_name)
            return True
        else:
            return False

    @staticmethod
    def get_shape_type(data, target_formats):
        for d in data["shapes"]:
            if d["shape_type"] in target_formats:
                return d["shape_type"]
        return ""