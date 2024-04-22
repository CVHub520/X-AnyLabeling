import os
import os.path as osp
import cv2
import csv
import json
import math
import numpy as np
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

from PIL import Image
from datetime import date
from itertools import chain

from anylabeling.app_info import __version__
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.shape import rectangle_from_diagonal


class LabelConverter:
    def __init__(self, classes_file=None):
        self.classes = []
        if classes_file:
            with open(classes_file, "r", encoding="utf-8") as f:
                self.classes = f.read().splitlines()
            logger.info(f"Loading classes: {self.classes}")

    def reset(self):
        self.custom_data = dict(
            version=__version__,
            flags={},
            shapes=[],
            imagePath="",
            imageData=None,
            imageHeight=-1,
            imageWidth=-1,
        )

    @staticmethod
    def calculate_rotation_theta(points):
        x1, y1 = points[0]
        x2, y2 = points[1]

        # Calculate one of the diagonal vectors (after rotation)
        diagonal_vector_x = x2 - x1
        diagonal_vector_y = y2 - y1

        # Calculate the rotation angle in radians
        rotation_angle = math.atan2(diagonal_vector_y, diagonal_vector_x)

        # Convert radians to degrees
        rotation_angle_degrees = math.degrees(rotation_angle)

        if rotation_angle_degrees < 0:
            rotation_angle_degrees += 360

        return rotation_angle_degrees / 360 * (2 * math.pi)

    @staticmethod
    def calculate_polygon_area(segmentation):
        x, y = [], []
        for i in range(len(segmentation)):
            if i % 2 == 0:
                x.append(segmentation[i])
            else:
                y.append(segmentation[i])
        area = 0.5 * np.abs(
            np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
        )
        return float(area)

    @staticmethod
    def get_image_size(image_file):
        with Image.open(image_file) as img:
            width, height = img.size
            return width, height

    @staticmethod
    def get_min_enclosing_bbox(segmentation):
        if not segmentation:
            return []
        polygon_points = [
            (segmentation[i], segmentation[i + 1])
            for i in range(0, len(segmentation), 2)
        ]
        x_coords, y_coords = zip(*polygon_points)
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        return [x_min, y_min, bbox_width, bbox_height]

    @staticmethod
    def get_contours_and_labels(mask, mapping_table, epsilon_factor=0.001):
        results = []
        input_type = mapping_table["type"]
        mapping_color = mapping_table["colors"]

        if input_type == "grayscale":
            color_to_label = {v: k for k, v in mapping_color.items()}
            binaray_img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            # use the different color_value to find the sub-region for each class
            for color_value in np.unique(binaray_img):
                class_name = color_to_label.get(color_value, "Unknown")
                label_map = (binaray_img == color_value).astype(np.uint8)

                contours, _ = cv2.findContours(
                    label_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    epsilon = epsilon_factor * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) < 5:
                        continue

                    points = []
                    for point in approx:
                        x, y = point[0].tolist()
                        points.append([x, y])
                    result_item = {"points": points, "label": class_name}
                    results.append(result_item)
        elif input_type == "rgb":
            color_to_label = {
                tuple(color): label for label, color in mapping_color.items()
            }
            rgb_img = cv2.imread(mask)
            hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)

            _, binary_img = cv2.threshold(
                hsv_img[:, :, 1], 0, 255, cv2.THRESH_BINARY
            )
            contours, _ = cv2.findContours(
                binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) < 5:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                center = (int(x + w / 2), int(y + h / 2))
                rgb_color = rgb_img[center[1], center[0]].tolist()
                label = color_to_label.get(tuple(rgb_color[::-1]), "Unknown")

                points = []
                for point in approx:
                    x, y = point[0].tolist()
                    points.append([x, y])

                result_item = {"points": points, "label": label}
                results.append(result_item)
        return results

    def get_coco_data(self):
        coco_data = {
            "info": {
                "year": 2023,
                "version": __version__,
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

    def calculate_normalized_bbox(self, poly, img_w, img_h):
        """
        Calculate the minimum bounding box for a set of four points and return the YOLO format rectangle representation (normalized).

        Args:
        - poly (list): List of four points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
        - img_w (int): Width of the corresponding image.
        - img_h (int): Height of the corresponding image.

        Returns:
        - tuple: Tuple representing the YOLO format rectangle in xywh_center form (all normalized).
        """
        xmin, ymin, xmax, ymax = self.calculate_bounding_box(poly)
        x_center = (xmin + xmax) / (2 * img_w)
        y_center = (ymin + ymax) / (2 * img_h)
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        return x_center, y_center, width, height

    @staticmethod
    def calculate_bounding_box(poly):
        """
        Calculate the minimum bounding box for a set of four points.

        Args:
        - poly (list): List of four points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].

        Returns:
        - tuple: Tuple representing the bounding box (xmin, ymin, xmax, ymax).
        """
        x_vals, y_vals = zip(*poly)
        return min(x_vals), min(y_vals), max(x_vals), max(y_vals)

    def yolo_obb_to_custom(self, input_file, output_file, image_file):
        self.reset()
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        img_w, img_h = self.get_image_size(image_file)
        for line in lines:
            line = line.strip().split(" ")
            class_index = int(line[0])
            label = self.classes[class_index]
            shape_type = "rotation"
            # Extracting coordinates from YOLO format
            x0, y0, x1, y1, x2, y2, x3, y3 = map(float, line[1:])
            # Rescaling coordinates to image size
            x0, y0, x1, y1, x2, y2, x3, y3 = (
                x0 * img_w,
                y0 * img_h,
                x1 * img_w,
                y1 * img_h,
                x2 * img_w,
                y2 * img_h,
                x3 * img_w,
                y3 * img_h,
            )
            # Creating points in the custom format
            points = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
            shape = {
                "label": label,
                "shape_type": shape_type,
                "flags": {},
                "points": points,
                "group_id": None,
                "description": None,
                "difficult": False,
                "direction": self.calculate_rotation_theta(points),
                "attributes": {},
            }
            self.custom_data["shapes"].append(shape)
        self.custom_data["imagePath"] = osp.basename(image_file)
        self.custom_data["imageHeight"] = img_h
        self.custom_data["imageWidth"] = img_w
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def yolo_to_custom(self, input_file, output_file, image_file):
        self.reset()
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        img_w, img_h = self.get_image_size(image_file)
        image_size = np.array([img_w, img_h], np.float64)
        for line in lines:
            line = line.strip().split(" ")
            class_index = int(line[0])
            label = self.classes[class_index]
            if len(line) == 5:
                shape_type = "rectangle"
                cx = float(line[1])
                cy = float(line[2])
                nw = float(line[3])
                nh = float(line[4])
                xmin = int((cx - nw / 2) * img_w)
                ymin = int((cy - nh / 2) * img_h)
                xmax = int((cx + nw / 2) * img_w)
                ymax = int((cy + nh / 2) * img_h)
                points = [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                ]
            else:
                shape_type = "polygon"
                points, masks = [], line[1:]
                for x, y in zip(masks[0::2], masks[1::2]):
                    point = [np.float64(x), np.float64(y)]
                    point = np.array(point, np.float64) * image_size
                    points.append(point.tolist())
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
        self.custom_data["imagePath"] = osp.basename(image_file)
        self.custom_data["imageHeight"] = img_h
        self.custom_data["imageWidth"] = img_w
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def voc_to_custom(self, input_file, output_file, image_filename):
        self.reset()

        tree = ET.parse(input_file)
        root = tree.getroot()

        image_width = int(root.find("size/width").text)
        image_height = int(root.find("size/height").text)

        self.custom_data["imagePath"] = image_filename
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

    def coco_to_custom(self, input_file, image_path):
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
                "imagePath": osp.basename(dic_info["file_name"]),
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

        for dic_info in total_info.values():
            self.reset()
            self.custom_data["shapes"] = dic_info["shapes"]
            self.custom_data["imagePath"] = dic_info["imagePath"]
            self.custom_data["imageHeight"] = dic_info["imageHeight"]
            self.custom_data["imageWidth"] = dic_info["imageWidth"]

            output_file = osp.join(
                image_path, osp.splitext(dic_info["imagePath"])[0] + ".json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def dota_to_custom(self, input_file, output_file, image_file):
        self.reset()

        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        image_width, image_height = self.get_image_size(image_file)

        for line in lines:
            line = line.strip().split(" ")
            x0, y0, x1, y1, x2, y2, x3, y3 = [float(i) for i in line[:8]]
            difficult = line[-1]
            points = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
            shape = {
                "label": line[8],
                "description": None,
                "points": points,
                "group_id": None,
                "difficult": bool(int(difficult)),
                "direction": self.calculate_rotation_theta(points),
                "shape_type": "rotation",
                "flags": {},
            }
            self.custom_data["shapes"].append(shape)

        self.custom_data["imagePath"] = osp.basename(image_file)
        self.custom_data["imageHeight"] = image_height
        self.custom_data["imageWidth"] = image_width

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def mask_to_custom(
        self, input_file, output_file, image_file, mapping_table
    ):
        self.reset()

        results = self.get_contours_and_labels(input_file, mapping_table)
        for result in results:
            shape = {
                "label": result["label"],
                "text": "",
                "points": result["points"],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
            self.custom_data["shapes"].append(shape)

        image_width, image_height = self.get_image_size(image_file)
        self.custom_data["imagePath"] = os.path.basename(image_file)
        self.custom_data["imageHeight"] = image_height
        self.custom_data["imageWidth"] = image_width

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def mot_to_custom(self, input_file, output_path, image_path):
        with open(input_file, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            mot_data = [row for row in reader]

        data_to_shape = {}
        for data in mot_data:
            frame_id = int(data[0])
            group_id = int(data[1])
            xmin = int(data[2])
            ymin = int(data[3])
            xmax = int(data[4]) + xmin
            ymax = int(data[5]) + ymin
            label = self.classes[int(data[7])]
            info = [label, xmin, ymin, xmax, ymax, group_id]
            if frame_id not in data_to_shape:
                data_to_shape[frame_id] = [info]
            else:
                data_to_shape[frame_id].append(info)

        file_list = os.listdir(image_path)
        for file_name in file_list:
            if file_name.endswith(".json"):
                continue

            self.reset()
            frame_id = int(osp.splitext(file_name.split("_")[-1])[0])
            data = data_to_shape[frame_id]
            image_file = osp.join(image_path, file_name)
            imageWidth, imageHeight = self.get_image_size(image_file)

            shapes = []
            for d in data:
                label, xmin, ymin, xmax, ymax, group_id = d
                points = [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                ]
                shape = {
                    "label": label,
                    "description": None,
                    "points": points,
                    "group_id": group_id,
                    "difficult": False,
                    "direction": 0,
                    "shape_type": "rectangle",
                    "flags": {},
                }
                shapes.append(shape)

            imagePath = file_name
            if output_path != image_path:
                imagePath = osp.join(output_path, file_name)
            self.custom_data["imagePath"] = imagePath
            self.custom_data["imageWidth"] = imageWidth
            self.custom_data["imageHeight"] = imageHeight
            self.custom_data["shapes"] = shapes

            output_file = osp.join(
                output_path, osp.splitext(file_name)[0] + ".json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def custom_to_yolo(self, input_file, output_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_width = data["imageWidth"]
        image_height = data["imageHeight"]
        image_size = np.array([[image_width, image_height]])

        with open(output_file, "w", encoding="utf-8") as f:
            for shape in data["shapes"]:
                shape_type = shape["shape_type"]
                if shape_type == "rectangle":
                    label = shape["label"]
                    points = shape["points"]
                    if len(points) == 2:
                        logger.warning(
                            "UserWarning: Diagonal vertex mode is deprecated in X-AnyLabeling release v2.2.0 or later.\n"
                            "Please update your code to accommodate the new four-point mode."
                        )
                        points = rectangle_from_diagonal(points)

                    class_index = self.classes.index(label)

                    x_center = (points[0][0] + points[2][0]) / (
                        2 * image_width
                    )
                    y_center = (points[0][1] + points[2][1]) / (
                        2 * image_height
                    )
                    width = abs(points[2][0] - points[0][0]) / image_width
                    height = abs(points[2][1] - points[0][1]) / image_height

                    f.write(
                        f"{class_index} {x_center} {y_center} {width} {height}\n"
                    )
                elif shape_type == "polygon":
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
                elif shape_type == "rotation":
                    label = shape["label"]
                    points = list(chain.from_iterable(shape["points"]))
                    normalized_coords = [
                        points[i] / image_width
                        if i % 2 == 0
                        else points[i] / image_height
                        for i in range(8)
                    ]
                    x0, y0, x1, y1, x2, y2, x3, y3 = normalized_coords
                    class_index = self.classes.index(label)
                    f.write(
                        f"{class_index} {x0} {y0} {x1} {y1} {x2} {y2} {x3} {y3}\n"
                    )

    def custom_to_voc(self, input_file, output_dir):
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
            if shape["shape_type"] != "rectangle":
                continue
            label = shape["label"]
            points = shape["points"]
            difficult = shape.get("difficult", False)
            if len(points) == 2:
                logger.warning(
                    "UserWarning: Diagonal vertex mode is deprecated in X-AnyLabeling release v2.2.0 or later.\n"
                    "Please update your code to accommodate the new four-point mode."
                )
                points = rectangle_from_diagonal(points)
            xmin, ymin, xmax, ymax = self.calculate_bounding_box(points)

            object_elem = ET.SubElement(root, "object")
            ET.SubElement(object_elem, "name").text = label
            ET.SubElement(object_elem, "pose").text = "Unspecified"
            ET.SubElement(object_elem, "truncated").text = "0"
            ET.SubElement(object_elem, "difficult").text = str(int(difficult))
            bndbox = ET.SubElement(object_elem, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(xmin))
            ET.SubElement(bndbox, "ymin").text = str(int(ymin))
            ET.SubElement(bndbox, "xmax").text = str(int(xmax))
            ET.SubElement(bndbox, "ymax").text = str(int(ymax))

        xml_string = ET.tostring(root, encoding="utf-8")
        dom = minidom.parseString(xml_string)
        formatted_xml = dom.toprettyxml(indent="  ")

        with open(output_dir, "w", encoding="utf-8") as f:
            f.write(formatted_xml)

    def custom_to_coco(self, input_path, output_path):
        coco_data = self.get_coco_data()

        for i, class_name in enumerate(self.classes):
            coco_data["categories"].append(
                {"id": i + 1, "name": class_name, "supercategory": ""}
            )

        image_id = 0
        annotation_id = 0

        label_file_list = os.listdir(input_path)
        for file_name in label_file_list:
            if not file_name.endswith(".json"):
                continue
            image_id += 1
            input_file = osp.join(input_path, file_name)
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            coco_data["images"].append(
                {
                    "id": image_id,
                    "file_name": data["imagePath"],
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
                bbox, segmentation, area = [], [], 0
                shape_type = shape["shape_type"]
                if shape_type == "rectangle":
                    if len(points) == 2:
                        logger.warning(
                            "UserWarning: Diagonal vertex mode is deprecated in X-AnyLabeling release v2.2.0 or later.\n"
                            "Please update your code to accommodate the new four-point mode."
                        )
                        points = rectangle_from_diagonal(points)
                    x_min = min(points[0][0], points[2][0])
                    y_min = min(points[0][1], points[2][1])
                    x_max = max(points[0][0], points[2][0])
                    y_max = max(points[0][1], points[2][1])
                    width = x_max - x_min
                    height = y_max - y_min
                    bbox = [x_min, y_min, width, height]
                    area = width * height
                elif shape_type == "polygon":
                    for point in points:
                        segmentation += point
                    bbox = self.get_min_enclosing_bbox(segmentation)
                    area = self.calculate_polygon_area(segmentation)
                    segmentation = [segmentation]

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "ignore": int(difficult),
                    "segmentation": segmentation,
                }

                coco_data["annotations"].append(annotation)

        output_file = osp.join(output_path, "instances_default.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=4, ensure_ascii=False)

    def custom_to_dota(self, input_file, output_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(output_file, "w", encoding="utf-8") as f:
            for shape in data["shapes"]:
                points = shape["points"]
                shape_type = shape["shape_type"]
                if shape_type != "rotation" or len(points) != 4:
                    continue
                label = shape["label"]
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

    def custom_to_mask(self, input_file, output_file, mapping_table):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_width = data["imageWidth"]
        image_height = data["imageHeight"]
        image_shape = (image_height, image_width)

        polygons = []
        for shape in data["shapes"]:
            shape_type = shape["shape_type"]
            if shape_type != "polygon":
                continue
            points = shape["points"]
            polygon = []
            for point in points:
                x, y = point
                polygon.append((int(x), int(y)))
            polygons.append(
                {
                    "label": shape["label"],
                    "polygon": polygon,
                }
            )

        output_format = mapping_table["type"]
        if output_format not in ["grayscale", "rgb"]:
            raise ValueError("Invalid output format specified")
        mapping_color = mapping_table["colors"]
        if output_format == "grayscale" and polygons:
            # Initialize binary_mask
            binary_mask = np.zeros(image_shape, dtype=np.uint8)
            for item in polygons:
                label, polygon = item["label"], item["polygon"]
                mask = np.zeros(image_shape, dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
                if label in mapping_color:
                    mask_mapped = mask * mapping_color[label]
                else:
                    mask_mapped = mask
                binary_mask += mask_mapped
            cv2.imencode(".png", binary_mask)[1].tofile(output_file)
        elif output_format == "rgb" and polygons:
            # Initialize rgb_mask
            color_mask = np.zeros(
                (image_height, image_width, 3), dtype=np.uint8
            )
            for item in polygons:
                label, polygon = item["label"], item["polygon"]
                # Create a mask for each polygon
                mask = np.zeros(image_shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
                # Initialize mask_mapped with a default value
                mask_mapped = mask
                # Map the mask values using the provided mapping table
                if label in mapping_color:
                    color = mapping_color[label]
                    mask_mapped = np.zeros_like(color_mask)
                    cv2.fillPoly(
                        mask_mapped, [np.array(polygon, dtype=np.int32)], color
                    )
                    color_mask = cv2.addWeighted(
                        color_mask, 1, mask_mapped, 1, 0
                    )
            cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))[
                1
            ].tofile(output_file)

    def custom_to_mot(self, input_path, output_file):
        mot_data = []
        label_file_list = os.listdir(input_path)
        label_file_list.sort(
            key=lambda x: int(osp.splitext(x.split("_")[-1])[0])
        )

        for label_file_name in label_file_list:
            if not label_file_name.endswith("json"):
                continue
            label_file = os.path.join(input_path, label_file_name)
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            frame_id = int(osp.splitext(label_file_name.split("_")[-1])[0])
            for shape in data["shapes"]:
                if shape["shape_type"] != "rectangle":
                    continue
                class_id = self.classes.index(shape["label"])
                track_id = int(shape["group_id"]) if shape["group_id"] else -1
                points = shape["points"]
                if len(points) == 2:
                    logger.warning(
                        "UserWarning: Diagonal vertex mode is deprecated in X-AnyLabeling release v2.2.0 or later.\n"
                        "Please update your code to accommodate the new four-point mode."
                    )
                    points = rectangle_from_diagonal(points)
                xmin = int(points[0][0])
                ymin = int(points[0][1])
                xmax = int(points[2][0])
                ymax = int(points[2][1])
                data = [
                    frame_id,
                    track_id,
                    xmin,
                    ymin,
                    xmax - xmin,
                    ymax - ymin,
                    0,
                    class_id,
                    1,
                ]
                mot_data.append(data)

        # Save updated_data to output_file
        with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerows(mot_data)
