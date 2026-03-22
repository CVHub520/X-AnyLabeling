import argparse
import json
import os
import os.path as osp
import cv2
import time
import math

from PIL import Image, ImageDraw
from tqdm import tqdm
from datetime import date

import numpy as np
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

import sys

sys.path.append(".")
from anylabeling.app_info import __version__  # noqa: E402

VERSION = __version__


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


class BaseLabelConverter:
    def __init__(self, classes_file=None):
        if classes_file:
            with open(classes_file, "r", encoding="utf-8") as f:
                self.classes = f.read().splitlines()
        else:
            self.classes = []
        print(f"import classes is: {self.classes}")

    def reset(self):
        self.custom_data = dict(
            version=VERSION,
            flags={},
            shapes=[],
            imagePath="",
            imageData=None,
            imageHeight=-1,
            imageWidth=-1,
        )

    def get_image_size(self, image_file):
        with Image.open(image_file) as img:
            width, height = img.size
            return width, height

    def get_minimal_enclosing_rectangle(self, poly):
        assert len(poly) == 8, "Input rectangle must contain exactly 8 values."

        x_coords = [poly[i] for i in range(0, 8, 2)]
        y_coords = [poly[i] for i in range(1, 8, 2)]

        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        x = min_x
        y = min_y
        width = max_x - min_x
        height = max_y - min_y

        return [x, y, width, height]

    def get_poly_area(self, poly):
        # Ensure that poly contains exactly 8 values
        assert len(poly) == 8, "Input polygon must contain exactly 8 values."

        # Extract x and y coordinates
        x_coords = [poly[i] for i in range(0, 8, 2)]
        y_coords = [poly[i] for i in range(1, 8, 2)]

        # Calculate the area using the Shoelace formula
        area = 0.5 * abs(
            sum(
                x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i]
                for i in range(3)
            )
            + x_coords[3] * y_coords[0]
            - x_coords[0] * y_coords[3]
        )

        return area

    def get_coco_data(self):
        coco_data = {
            "info": {
                "year": 2023,
                "version": VERSION,
                "description": "COCO Label Conversion",
                "contributor": "CVHub",
                "url": "https://github.com/CVHub520/X-AnyLabeling",
                "date_created": str(date.today()),
            },
            "licenses": [
                {
                    "id": 1,
                    "url": "https://www.gnu.org/licenses/gpl-3.0.html",
                    "name": "GNU GENERAL PUBLIC LICENSE Version 3",
                }
            ],
            "categories": [],
            "images": [],
            "annotations": [],
        }
        return coco_data

    def ensure_output_path(self, output_path, ext=None):
        if osp.isfile(output_path):
            # Check if the file has the expected extension
            if not output_path.lower().endswith(ext.lower()):
                raise ValueError(
                    f"The specified file '{output_path}' \
                        does not have the expected '{ext}' extension."
                )
        else:
            # Check if the folder exists, and create it if it doesn't
            if not osp.exists(output_path):
                os.makedirs(output_path, exist_ok=True)


class RectLabelConverter(BaseLabelConverter):
    def custom_to_voc2017(self, input_file, output_dir):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = data["imagePath"]
        image_width = data["imageWidth"]
        image_height = data["imageHeight"]

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = osp.dirname(output_dir)
        ET.SubElement(root, "filename").text = osp.basename(image_path)
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(image_width)
        ET.SubElement(size, "height").text = str(image_height)
        ET.SubElement(size, "depth").text = "3"

        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]
            difficult = shape.get("difficult", False)

            xmin = str(points[0][0])
            ymin = str(points[0][1])
            xmax = str(points[2][0])
            ymax = str(points[2][1])

            object_elem = ET.SubElement(root, "object")
            ET.SubElement(object_elem, "name").text = label
            ET.SubElement(object_elem, "pose").text = "Unspecified"
            ET.SubElement(object_elem, "truncated").text = "0"
            ET.SubElement(object_elem, "difficult").text = str(int(difficult))
            bndbox = ET.SubElement(object_elem, "bndbox")
            ET.SubElement(bndbox, "xmin").text = xmin
            ET.SubElement(bndbox, "ymin").text = ymin
            ET.SubElement(bndbox, "xmax").text = xmax
            ET.SubElement(bndbox, "ymax").text = ymax

        xml_string = ET.tostring(root, encoding="utf-8")
        dom = minidom.parseString(xml_string)
        formatted_xml = dom.toprettyxml(indent="  ")

        with open(output_dir, "w") as f:
            f.write(formatted_xml)

    def voc2017_to_custom(self, input_file, output_file):
        self.reset()

        tree = ET.parse(input_file)
        root = tree.getroot()

        image_path = root.find("filename").text
        image_width = int(root.find("size/width").text)
        image_height = int(root.find("size/height").text)

        self.custom_data["imagePath"] = image_path
        self.custom_data["imageHeight"] = image_height
        self.custom_data["imageWidth"] = image_width

        for obj in root.findall("object"):
            label = obj.find("name").text
            difficult = "0"
            if obj.find("difficult") is not None:
                difficult = str(obj.find("difficult").text)
            xmin = float(obj.find("bndbox/xmin").text)
            ymin = float(obj.find("bndbox/ymin").text)
            xmax = float(obj.find("bndbox/xmax").text)
            ymax = float(obj.find("bndbox/ymax").text)

            shape = {
                "label": label,
                "description": "",
                "points": [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                ],
                "group_id": None,
                "difficult": bool(int(difficult)),
                "shape_type": "rectangle",
                "flags": {},
            }

            self.custom_data["shapes"].append(shape)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def custom_to_yolov5(self, input_file, output_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_width = data["imageWidth"]
        image_height = data["imageHeight"]

        with open(output_file, "w", encoding="utf-8") as f:
            for shape in data["shapes"]:
                label = shape["label"]
                points = shape["points"]

                class_index = self.classes.index(label)

                x_center = (points[0][0] + points[2][0]) / (2 * image_width)
                y_center = (points[0][1] + points[2][1]) / (2 * image_height)
                width = abs(points[2][0] - points[0][0]) / image_width
                height = abs(points[2][1] - points[0][1]) / image_height

                f.write(
                    f"{class_index} {x_center} {y_center} {width} {height}\n"
                )

    def yolov5_to_custom(self, input_file, output_file, image_file):
        self.reset()
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        img_w, img_h = self.get_image_size(image_file)

        for line in lines:
            line = line.strip().split(" ")
            class_index = int(line[0])
            cx = float(line[1])
            cy = float(line[2])
            nw = float(line[3])
            nh = float(line[4])
            xmin = int((cx - nw / 2) * img_w)
            ymin = int((cy - nh / 2) * img_h)
            xmax = int((cx + nw / 2) * img_w)
            ymax = int((cy + nh / 2) * img_h)

            shape_type = "rectangle"
            label = self.classes[class_index]
            points = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ]
            shape = {
                "label": label,
                "shape_type": shape_type,
                "flags": {},
                "points": points,
                "group_id": None,
                "description": None,
                "difficult": False,
                "attributes": {},
            }
            self.custom_data["shapes"].append(shape)
        self.custom_data["imagePath"] = os.path.basename(image_file)
        self.custom_data["imageHeight"] = img_h
        self.custom_data["imageWidth"] = img_w
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def custom_to_coco(self, input_path, output_path):
        raise DeprecationWarning(
            "This function is deprecated. Please use the GUI for COCO export."
        )

        coco_data = self.get_coco_data()

        for i, class_name in enumerate(self.classes):
            coco_data["categories"].append(
                {"id": i + 1, "name": class_name, "supercategory": ""}
            )

        image_id = 0
        annotation_id = 0

        file_list = os.listdir(input_path)
        for file_name in tqdm(
            file_list, desc="Converting files", unit="file", colour="green"
        ):
            if not file_name.endswith(".json"):
                continue
            image_id += 1

            input_file = osp.join(input_path, file_name)
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_path = data["imagePath"]
            image_name = osp.splitext(osp.basename(image_path))[0]

            coco_data["images"].append(
                {
                    "id": image_id,
                    "file_name": image_name,
                    "width": data["imageWidth"],
                    "height": data["imageHeight"],
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "",
                }
            )

            for shape in data["shapes"]:
                annotation_id += 1
                label = shape["label"]
                points = shape["points"]
                difficult = shape.get("difficult", False)
                class_id = self.classes.index(label)
                x_min = min(points[0][0], points[2][0])
                y_min = min(points[0][1], points[2][1])
                x_max = max(points[0][0], points[2][0])
                y_max = max(points[0][1], points[2][1])
                width = x_max - x_min
                height = y_max - y_min

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "ignore": int(difficult),
                    "segmentation": [],
                }

                coco_data["annotations"].append(annotation)

        output_file = osp.join(output_path, "instances_default.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=4, ensure_ascii=False)

    def coco_to_custom(self, input_file, image_path, output_path):
        raise DeprecationWarning(
            "This function is deprecated. Please use the GUI for COCO upload."
        )

        img_dic = {}
        for file in os.listdir(image_path):
            img_dic[file] = file

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not self.classes:
            for cat in data["categories"]:
                self.classes.append(cat["name"])

        total_info, label_info = {}, {}

        # map category_id to name
        for dic_info in data["categories"]:
            label_info[dic_info["id"]] = dic_info["name"]

        # map image_id to info
        for dic_info in data["images"]:
            total_info[dic_info["id"]] = {
                "imageWidth": dic_info["width"],
                "imageHeight": dic_info["height"],
                "imagePath": img_dic[dic_info["file_name"]],
                "shapes": [],
            }

        for dic_info in data["annotations"]:
            bbox = dic_info["bbox"]
            xmin = bbox[0]
            ymin = bbox[1]
            width = bbox[2]
            height = bbox[3]
            xmax = xmin + width
            ymax = ymin + height

            shape_type = "rectangle"
            difficult = bool(int(str(dic_info.get("ignore", "0"))))
            label = self.classes[dic_info["category_id"] - 1]
            points = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ]
            shape = {
                "label": label,
                "shape_type": shape_type,
                "flags": {},
                "points": points,
                "group_id": None,
                "description": None,
                "difficult": difficult,
                "attributes": {},
            }

            total_info[dic_info["image_id"]]["shapes"].append(shape)

        for dic_info in tqdm(
            total_info.values(),
            desc="Converting files",
            unit="file",
            colour="green",
        ):
            self.reset()
            self.custom_data["shapes"] = dic_info["shapes"]
            self.custom_data["imagePath"] = dic_info["imagePath"]
            self.custom_data["imageHeight"] = dic_info["imageHeight"]
            self.custom_data["imageWidth"] = dic_info["imageWidth"]

            output_file = osp.join(
                output_path, osp.splitext(dic_info["imagePath"])[0] + ".json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_data, f, indent=2, ensure_ascii=False)


class PolyLabelConvert(BaseLabelConverter):
    def mask2box(self, mask):
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return (
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        )

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def custom_to_coco(self, input_path, output_path):
        raise DeprecationWarning(
            "This function is deprecated. Please use the GUI for COCO export."
        )

        coco_data = self.get_coco_data()

        for i, class_name in enumerate(self.classes):
            coco_data["categories"].append(
                {"id": i + 1, "name": class_name, "supercategory": ""}
            )

        image_id = 0
        annotation_id = 0

        file_list = os.listdir(input_path)
        for file_name in tqdm(
            file_list, desc="Converting files", unit="file", colour="green"
        ):
            if not file_name.endswith(".json"):
                continue
            image_id += 1
            input_file = osp.join(input_path, file_name)
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_path = data["imagePath"]
            image_name = osp.splitext(osp.basename(image_path))[0] + ".jpg"

            coco_data["images"].append(
                {
                    "id": image_id,
                    "file_name": image_name,
                    "width": data["imageWidth"],
                    "height": data["imageHeight"],
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "",
                }
            )

            for shape in data["shapes"]:
                annotation_id += 1
                label = shape["label"]
                points = shape["points"]
                difficult = shape.get("difficult", False)
                class_id = self.classes.index(label)
                mask = self.polygons_to_mask(
                    [data["imageHeight"], data["imageWidth"]], points
                )
                x_min, y_min, width, height = self.mask2box(mask)

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x_min, y_min, width, height],
                    "segmentation": [list(np.asarray(points).flatten())],
                    "area": width * height,
                    "iscrowd": 0,
                    "ignore": int(difficult),
                }

                coco_data["annotations"].append(annotation)

        output_file = osp.join(output_path, "instances_default.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                coco_data, f, indent=4, ensure_ascii=False, cls=JsonEncoder
            )

    def custom_to_yolov5(self, input_file, output_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_width = data["imageWidth"]
        image_height = data["imageHeight"]
        image_size = np.array([[image_width, image_height]])

        with open(output_file, "w", encoding="utf-8") as f:
            for shape in data["shapes"]:
                label = shape["label"]
                points = np.array(shape["points"])
                class_index = self.classes.index(label)
                norm_points = points / image_size
                f.write(
                    f"{class_index} "
                    + " ".join(
                        [
                            " ".join([str(cell[0]), str(cell[1])])
                            for cell in norm_points.tolist()
                        ]
                    )
                    + "\n"
                )

    def yolov5_to_custom(self, input_file, output_file, image_file):
        self.reset()

        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        image_width, image_height = self.get_image_size(image_file)
        image_size = np.array([image_width, image_height], np.float64)

        for line in lines:
            line = line.strip().split(" ")
            class_index = int(line[0])
            label = self.classes[class_index]
            masks = line[1:]
            shape = {
                "label": label,
                "points": [],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "difficult": False,
                "description": None,
            }
            for x, y in zip(masks[0::2], masks[1::2]):
                point = [np.float64(x), np.float64(y)]
                point = np.array(point, np.float64) * image_size
                shape["points"].append(point.tolist())
            self.custom_data["shapes"].append(shape)

        self.custom_data["imagePath"] = osp.basename(image_file)
        self.custom_data["imageHeight"] = image_height
        self.custom_data["imageWidth"] = image_width

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def coco_to_custom(self, input_file, image_path, output_path):
        img_dic = {}
        for file in os.listdir(image_path):
            img_dic[file] = file

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not self.classes:
            for cat in data["categories"]:
                self.classes.append(cat["name"])

        total_info, label_info = {}, {}

        # map category_id to name
        for dic_info in data["categories"]:
            label_info[dic_info["id"]] = dic_info["name"]

        # map image_id to info
        for dic_info in data["images"]:
            total_info[dic_info["id"]] = {
                "imageWidth": dic_info["width"],
                "imageHeight": dic_info["height"],
                "imagePath": img_dic[dic_info["file_name"]],
                "shapes": [],
            }

        for dic_info in data["annotations"]:
            points = []
            segmentation = dic_info["segmentation"][0]
            for i in range(0, len(segmentation), 2):
                x, y = segmentation[i : i + 2]
                point = [float(x), float(y)]
                points.append(point)
            difficult = str(dic_info.get("ignore", "0"))
            shape_info = {
                "label": self.classes[dic_info["category_id"] - 1],
                "description": None,
                "points": points,
                "group_id": None,
                "difficult": bool(int(difficult)),
                "shape_type": "polygon",
                "flags": {},
            }

            total_info[dic_info["image_id"]]["shapes"].append(shape_info)

        for dic_info in tqdm(
            total_info.values(),
            desc="Converting files",
            unit="file",
            colour="green",
        ):
            self.reset()
            self.custom_data["shapes"] = dic_info["shapes"]
            self.custom_data["imagePath"] = dic_info["imagePath"]
            self.custom_data["imageHeight"] = dic_info["imageHeight"]
            self.custom_data["imageWidth"] = dic_info["imageWidth"]

            output_file = osp.join(
                output_path, osp.splitext(dic_info["imagePath"])[0] + ".json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_data, f, indent=2, ensure_ascii=False)


class RotateLabelConverter(BaseLabelConverter):
    def custom_to_dota(self, input_file, output_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(output_file, "w", encoding="utf-8") as f:
            for shape in data["shapes"]:
                label = shape["label"]
                points = shape["points"]
                difficult = shape.get("difficult", False)
                x0 = points[0][0]
                y0 = points[0][1]
                x1 = points[1][0]
                y1 = points[1][1]
                x2 = points[2][0]
                y2 = points[2][1]
                x3 = points[3][0]
                y3 = points[3][1]
                f.write(
                    f"{x0} {y0} {x1} {y1} {x2} {y2} {x3} {y3} {label} {int(difficult)}\n"
                )

    def dota_to_custom(self, input_file, output_file, image_file):
        self.reset()

        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        image_width, image_height = self.get_image_size(image_file)

        for line in lines:
            line = line.strip().split(" ")
            x0, y0, x1, y1, x2, y2, x3, y3 = [float(i) for i in line[:8]]
            difficult = line[-1]
            shape = {
                "label": line[8],
                "description": None,
                "points": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]],
                "group_id": None,
                "difficult": bool(int(difficult)),
                "direction": 0,
                "shape_type": "rotation",
                "flags": {},
            }
            self.custom_data["shapes"].append(shape)

        self.custom_data["imagePath"] = osp.basename(image_file)
        self.custom_data["imageHeight"] = image_height
        self.custom_data["imageWidth"] = image_width

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def dota_to_dcoco(self, input_path, output_path, image_path):
        self.ensure_output_path(output_path, "json")
        coco_data = self.get_coco_data()

        for i, class_name in enumerate(self.classes):
            coco_data["categories"].append(
                {"id": i + 1, "name": class_name, "supercategory": ""}
            )

        image_id = 0
        annotation_id = 0

        file_list = os.listdir(image_path)
        for image_file in tqdm(
            file_list, desc="Converting files", unit="file", colour="green"
        ):
            label_file = osp.join(
                input_path, osp.splitext(image_file)[0] + ".txt"
            )

            image_width, image_height = self.get_image_size(
                osp.join(image_path, image_file)
            )

            image_id += 1

            coco_data["images"].append(
                {
                    "id": image_id,
                    "file_name": image_file,
                    "width": image_width,
                    "height": image_height,
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "",
                }
            )

            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip().split(" ")
                *poly, label, difficult = line
                poly = list(map(float, poly))
                area = self.get_poly_area(poly)
                rect = self.get_minimal_enclosing_rectangle(poly)
                annotation_id += 1
                class_id = self.classes.index(label)
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": rect,
                    "segmentation": [poly],
                    "area": area,
                    "iscrowd": 0,
                    "ignore": difficult,
                }

                coco_data["annotations"].append(annotation)

        if osp.isdir(output_path):
            output_path = osp.join(output_path, "x_anylabeling_coco.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=4, ensure_ascii=False)

    def dcoco_to_dota(self, input_file, output_path):
        self.ensure_output_path(output_path)
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        label_info = {}
        # map category_id to label
        for dic_info in data["categories"]:
            label_info[dic_info["id"]] = dic_info["name"]

        name_info = {}
        # map image_id to file_naame
        for dic_info in data["images"]:
            name_info[dic_info["id"]] = dic_info["file_name"]

        total_info = {}
        for dic_info in data["annotations"]:
            poly = dic_info["segmentation"][0]
            image_id = dic_info["image_id"]
            category_id = dic_info["category_id"]
            label = label_info[category_id]
            difficult = dic_info.get("ignore", 0)
            if image_id not in total_info:
                total_info[image_id] = [[*poly, label, difficult]]
            else:
                total_info[image_id].append([*poly, label, difficult])

        for image_id, label_info in total_info.items():
            label_file = osp.basename(name_info[image_id]) + ".txt"
            output_file = osp.join(output_path, label_file)
            with open(output_file, "w", encoding="utf-8") as f:
                for info in label_info:
                    x0, y0, x1, y1, x2, y2, x3, y3, label, difficult = info
                    f.write(
                        f"{x0} {y0} {x1} {y1} {x2} {y2} {x3} {y3} {label} {int(difficult)}\n"
                    )

    def dxml_to_dota(self, input_file, output_file):
        tree = ET.parse(input_file)
        root = tree.getroot()
        with open(output_file, "w", encoding="utf-8") as f:
            for obj in root.findall("object"):
                obj_type = obj.find("type").text
                difficult = 0
                if obj.find("difficult") is not None:
                    difficult = obj.find("difficult").text
                label = obj.find("name").text
                if obj_type == "bndbox":
                    hbndbox = obj.find("bndbox")
                    points = self.hbndbox_to_dota(hbndbox)
                elif obj_type == "robndbox":
                    rbndbox = obj.find("robndbox")
                    points = self.rbndbox_to_dota(rbndbox)
                p0, p1, p2, p3 = points
                x0, y0, x1, y1, x2, y2, x3, y3 = *p0, *p1, *p2, *p3
                f.write(
                    f"{x0} {y0} {x1} {y1} {x2} {y2} {x3} {y3} {label} {difficult}\n"
                )

    @staticmethod
    def rotatePoint(xc, yc, xp, yp, theta):
        xoff = xp - xc
        yoff = yp - yc
        cosTheta = math.cos(theta)
        sinTheta = math.sin(theta)
        pResx = cosTheta * xoff + sinTheta * yoff
        pResy = -sinTheta * xoff + cosTheta * yoff
        return xc + pResx, yc + pResy

    def rbndbox_to_dota(self, box):
        cx = float(box.find("cx").text)
        cy = float(box.find("cy").text)
        w = float(box.find("w").text)
        h = float(box.find("h").text)
        angle = float(box.find("angle").text)

        x0, y0 = self.rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
        x1, y1 = self.rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
        x2, y2 = self.rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
        x3, y3 = self.rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)
        points = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        return points

    @staticmethod
    def hbndbox_to_dota(box):
        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        return points


class MOTSConverter(BaseLabelConverter):
    def custom_to_gt(self, gt_file, output_file):
        import pycocotools.mask as coco_mask

        with open(gt_file, "r") as f:
            lines = f.readlines()
        results = []
        for line in lines:
            label = line.strip().split(" ", maxsplit=5)
            height, width = int(label[3]), int(label[4])
            polygon = [np.array(eval(label[-1])).flatten()]
            rle = self.polygon_to_rle(polygon, height, width)
            label[-1] = rle["counts"]
            results.append(label)
        save_path = osp.dirname(output_file)
        os.makedirs(save_path, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for row in results:
                f.write(" ".join(map(str, row)) + "\n")

    @staticmethod
    def polygon_to_rle(polygon, height, width):
        import pycocotools.mask as coco_mask

        rles = coco_mask.frPyObjects(polygon, height, width)
        rle = coco_mask.merge(rles)
        return rle

    @staticmethod
    def rle_to_polygon(rle):
        import pycocotools.mask as coco_mask

        mask = coco_mask.decode(rle)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons = [contour.flatten().tolist() for contour in contours]
        return polygons

    @staticmethod
    def draw_rle_to_image(image_file, rle):
        import pycocotools.mask as coco_mask

        """
        Draw the RLE encoded mask onto the given image and save the images with contours and masked image.

        Parameters:
        - image_file: str, the file path of the original image.
        - rle: dict, a dictionary containing the RLE encoding, typically with 'size' and 'counts' keys.

        Results:
        - Two image files are saved in the same directory as the original image file:
            - 'contorus.jpg': The original image with contours drawn on it.
            - 'masked_image.jpg': The image created using the mask.
        """
        image = cv2.imread(image_file)
        mask = coco_mask.decode(rle)
        mask = mask.astype(np.uint8)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
        image_path = osp.dirname(image_file)
        cv2.imwrite(osp.join(image_path, "contorus.jpg"), image)
        cv2.imwrite(osp.join(image_path, "masked_image.jpg"), masked_image)


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description="Label Converter")
    parser.add_argument(
        "--task",
        default="rectangle",
        choices=["rectangle", "polygon", "rotation", "mots"],
        help="Choose the type of task to perform",
    )
    parser.add_argument("--src_path", help="Path to input file or directory")
    parser.add_argument("--dst_path", help="Path to output file or directory")
    parser.add_argument("--img_path", help="Path to image file or directory")
    parser.add_argument(
        "--classes",
        default=None,
        help="Path to classes.txt file, \
                            where each line represent a specific class",
    )
    parser.add_argument(
        "--mode",
        help="Choose the conversion mode what you need",
        choices=[
            "custom2voc",
            "voc2custom",
            "custom2yolo",
            "yolo2custom",
            "custom2coco",
            "coco2custom",
            "custom2dota",
            "dota2custom",
            "dota2dcoco",
            "dcoco2dota",
            "dxml2dota",
            "custom_to_gt",
        ],
    )
    args = parser.parse_args()

    print(f"Starting conversion to {args.mode} format of {args.task}...")
    start_time = time.time()

    if args.task == "rectangle":
        converter = RectLabelConverter(args.classes)
        valid_modes = [
            "custom2voc",
            "voc2custom",
            "custom2yolo",
            "yolo2custom",
            "custom2coco",
            "coco2custom",
        ]
        assert (
            args.mode in valid_modes
        ), f"Rectangle tasks are only supported in {valid_modes} now!"
    elif args.task == "polygon":
        converter = PolyLabelConvert(args.classes)
        valid_modes = [
            "custom2yolo",
            "yolo2custom",
            "coco2custom",
            "custom2coco",
        ]
        assert (
            args.mode in valid_modes
        ), f"Polygon tasks are only supported in {valid_modes} now!"
    elif args.task == "rotation":
        converter = RotateLabelConverter(args.classes)
        valid_modes = [
            "custom2dota",
            "dota2custom",
            "dota2dcoco",
            "dcoco2dota",
            "dxml2dota",
        ]
        assert (
            args.mode in valid_modes
        ), f"Rotation tasks are only supported in {valid_modes} now!"
    elif args.task == "mots":
        converter = MOTSConverter()
        valid_modes = [
            "custom_to_gt",
        ]
        assert (
            args.mode in valid_modes
        ), f"MOTS tasks are only supported in {valid_modes} now!"

    if args.mode == "custom2voc":
        file_list = os.listdir(args.src_path)
        os.makedirs(args.dst_path, exist_ok=True)
        for file_name in tqdm(
            file_list, desc="Converting files", unit="file", colour="green"
        ):
            if not file_name.endswith(".json"):
                continue
            src_file = osp.join(args.src_path, file_name)
            dst_file = osp.join(
                args.dst_path, osp.splitext(file_name)[0] + ".xml"
            )
            converter.custom_to_voc2017(src_file, dst_file)
    elif args.mode == "voc2custom":
        file_list = os.listdir(args.src_path)
        os.makedirs(args.dst_path, exist_ok=True)
        for file_name in tqdm(
            file_list, desc="Converting files", unit="file", colour="green"
        ):
            src_file = osp.join(args.src_path, file_name)
            dst_file = osp.join(
                args.dst_path, osp.splitext(file_name)[0] + ".json"
            )
            converter.voc2017_to_custom(src_file, dst_file)
    elif args.mode == "custom2yolo":
        file_list = os.listdir(args.src_path)
        os.makedirs(args.dst_path, exist_ok=True)
        for file_name in tqdm(
            file_list, desc="Converting files", unit="file", colour="green"
        ):
            if not file_name.endswith(".json"):
                continue
            src_file = osp.join(args.src_path, file_name)
            dst_file = osp.join(
                args.dst_path, osp.splitext(file_name)[0] + ".txt"
            )
            converter.custom_to_yolov5(src_file, dst_file)
    elif args.mode == "yolo2custom":
        img_dic = {}
        os.makedirs(args.dst_path, exist_ok=True)
        for file in os.listdir(args.img_path):
            prefix = file.rsplit(".", 1)[0]
            img_dic[prefix] = file
        file_list = os.listdir(args.src_path)
        for file_name in tqdm(
            file_list, desc="Converting files", unit="file", colour="green"
        ):
            src_file = osp.join(args.src_path, file_name)
            dst_file = osp.join(
                args.dst_path, osp.splitext(file_name)[0] + ".json"
            )
            img_file = osp.join(
                args.img_path, img_dic[osp.splitext(file_name)[0]]
            )
            converter.yolov5_to_custom(src_file, dst_file, img_file)
    elif args.mode == "custom2coco":
        os.makedirs(args.dst_path, exist_ok=True)
        converter.custom_to_coco(args.src_path, args.dst_path)
    elif args.mode == "coco2custom":
        os.makedirs(args.dst_path, exist_ok=True)
        converter.coco_to_custom(args.src_path, args.img_path, args.dst_path)
    elif args.mode == "custom2dota":
        file_list = os.listdir(args.src_path)
        os.makedirs(args.dst_path, exist_ok=True)
        for file_name in tqdm(
            file_list, desc="Converting files", unit="file", colour="green"
        ):
            if not file_name.endswith(".json"):
                continue
            src_file = osp.join(args.src_path, file_name)
            dst_file = osp.join(
                args.dst_path, osp.splitext(file_name)[0] + ".txt"
            )
            converter.custom_to_dota(src_file, dst_file)
    elif args.mode == "dota2custom":
        img_dic = {}
        for file in os.listdir(args.img_path):
            prefix = file.rsplit(".", 1)[0]
            img_dic[prefix] = file
        file_list = os.listdir(args.src_path)
        for file_name in tqdm(
            file_list, desc="Converting files", unit="file", colour="green"
        ):
            src_file = osp.join(args.src_path, file_name)
            dst_file = osp.join(
                args.dst_path, osp.splitext(file_name)[0] + ".json"
            )
            img_file = osp.join(
                args.img_path, img_dic[osp.splitext(file_name)[0]]
            )
            converter.dota_to_custom(src_file, dst_file, img_file)
    elif args.mode == "dota2dcoco":
        converter.dota_to_dcoco(args.src_path, args.dst_path, args.img_path)
    elif args.mode == "dcoco2dota":
        converter.dcoco_to_dota(args.src_path, args.dst_path)
    elif args.mode == "dxml2dota":
        file_list = os.listdir(args.src_path)
        os.makedirs(args.dst_path, exist_ok=True)
        for file_name in tqdm(
            file_list, desc="Converting files", unit="file", colour="green"
        ):
            src_file = osp.join(args.src_path, file_name)
            dst_file = osp.join(
                args.dst_path, osp.splitext(file_name)[0] + ".txt"
            )
            converter.dxml_to_dota(src_file, dst_file)
    elif args.mode == "custom_to_gt":
        converter.custom_to_gt(args.src_path, args.dst_path)
    end_time = time.time()
    print(f"Conversion completed successfully: {args.dst_path}")
    print(f"Conversion time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
