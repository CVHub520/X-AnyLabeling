import os
import os.path as osp
import cv2
import json
import jsonlines
import json_repair
import math
import re
import uuid
import yaml
import pathlib
import configparser
import numpy as np
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

from PIL import Image
from datetime import date
from itertools import chain

from anylabeling.app_info import __version__
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.shape import rectangle_from_diagonal
from anylabeling.views.labeling.utils.general import is_possible_rectangle


class LabelConverter:
    def __init__(self, classes_file=None, pose_cfg_file=None):
        self.classes = []
        if classes_file:
            with open(classes_file, "r", encoding="utf-8") as f:
                self.classes = f.read().splitlines()
            logger.info(f"Loading classes: {self.classes}")
        self.pose_classes = {}
        if pose_cfg_file:
            with open(pose_cfg_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.has_vasiable = data["has_visible"]
                for class_name, keypoint_name in data["classes"].items():
                    self.pose_classes[class_name] = keypoint_name
                self.classes = list(self.pose_classes.keys())
            logger.info(f"Loading pose classes: {self.pose_classes}")

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
    def calculate_polygon_area(segmentations):
        """Calculates the total area of polygons by rasterizing them onto a mask
        and counting the pixels, aligning with pycocotools.mask.area.

        Args:
            segmentations: A list where each element is a list of coordinates
                        representing a polygon (e.g., [x1, y1, x2, y2, ...]).

        Returns:
            The total area (pixel count) of all polygons in the list as a float.
            Returns 0.0 if the input is empty or contains only invalid polygons.
        """
        if not segmentations:
            return 0.0

        all_points = []
        valid_segmentations = []
        for seg in segmentations:
            if isinstance(seg, list) and len(seg) >= 6 and len(seg) % 2 == 0:
                points = np.array(seg).reshape(-1, 2)
                all_points.extend(points.tolist())
                valid_segmentations.append(points)

        if not all_points:
            return 0.0

        all_points_np = np.array(all_points)
        min_x, min_y = np.min(all_points_np, axis=0)
        max_x, max_y = np.max(all_points_np, axis=0)

        # Determine mask dimensions and offset
        # Use ceil/floor to ensure the mask covers the entire
        # integer pixel grid spanned by the points
        offset_x = -math.floor(min_x)
        offset_y = -math.floor(min_y)
        height = int(math.ceil(max_y) - math.floor(min_y))
        width = int(math.ceil(max_x) - math.floor(min_x))

        # Ensure width and height are at least 1
        height = max(1, height)
        width = max(1, width)

        mask = np.zeros((height, width), dtype=np.uint8)

        for points in valid_segmentations:
            shifted_points = np.copy(points)
            shifted_points[:, 0] += offset_x
            shifted_points[:, 1] += offset_y
            # Use rounding instead of truncation for potentially better alignment
            points_int = np.round(shifted_points).astype(np.int32)

            cv2.fillPoly(mask, [points_int], 1)

        # Area is the sum of pixels in the mask
        total_area = np.sum(mask)

        return float(total_area)

    @staticmethod
    def get_image_size(image_file):
        with Image.open(image_file) as img:
            width, height = img.size
            return width, height

    @staticmethod
    def get_min_enclosing_bbox(segmentations):
        """
        Calculates the minimum enclosing bounding box for a list of segmentations,
        matching the logic typically used by pycocotools (integer pixel grid).

        Args:
            segmentations: A list where each element is a list of coordinates
                        representing a polygon (e.g., [x1, y1, x2, y2, ...]).

        Returns:
            A list [x_min, y_min, width, height] representing the bounding box,
            or an empty list if the input is empty or contains only empty segmentations.
            Coordinates and dimensions are returned as floats for compatibility,
            but represent integer pixel boundaries.
        """
        all_polygon_points = []
        if not segmentations:
            return []

        for segmentation in segmentations:
            if not segmentation:
                continue
            polygon_points = [
                (segmentation[i], segmentation[i + 1])
                for i in range(0, len(segmentation), 2)
            ]
            all_polygon_points.extend(polygon_points)

        if not all_polygon_points:
            return []

        x_coords, y_coords = zip(*all_polygon_points)
        x_min_fp = min(x_coords)
        y_min_fp = min(y_coords)
        x_max_fp = max(x_coords)
        y_max_fp = max(y_coords)

        # Calculate bbox based on integer pixel indices covered
        x_min_int = math.floor(x_min_fp)
        y_min_int = math.floor(y_min_fp)
        x_max_int = math.floor(x_max_fp)
        y_max_int = math.floor(y_max_fp)

        bbox_width = float(x_max_int - x_min_int + 1)
        bbox_height = float(y_max_int - y_min_int + 1)

        return [float(x_min_int), float(y_min_int), bbox_width, bbox_height]

    @staticmethod
    def get_contours_and_labels(mask, mapping_table, epsilon_factor=0.001):
        results = []
        input_type = mapping_table["type"]
        mapping_color_data = mapping_table[
            "colors"
        ]  # {"label_name": [R, G, B], ...}

        if input_type == "grayscale":
            color_to_label = {v: k for k, v in mapping_color_data.items()}
            binary_img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            if binary_img is None:
                logger.error(f"Failed to read grayscale mask: {mask}")
                return results

            unique_colors = np.unique(binary_img)
            for color_value in unique_colors:
                if color_value == 0 and 0 not in color_to_label:
                    continue
                if color_value not in color_to_label:
                    continue

                class_name = color_to_label.get(color_value)

                # Create a binary map for the current color_value
                label_map = (binary_img == color_value).astype(np.uint8)

                contours, _ = cv2.findContours(
                    label_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    epsilon = epsilon_factor * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) < 3:
                        continue

                    points = [p[0].tolist() for p in approx]
                    result_item = {"points": points, "label": class_name}
                    results.append(result_item)

        elif input_type == "rgb":
            rgb_img_bgr = cv2.imread(mask)
            if rgb_img_bgr is None:
                logger.error(f"Failed to read RGB mask: {mask}")
                return results

            for class_name, color_rgb in mapping_color_data.items():
                if (
                    not isinstance(color_rgb, (list, tuple, np.ndarray))
                    or len(color_rgb) != 3
                ):
                    logger.warning(
                        f"Invalid color format for label {class_name}: {color_rgb}. Skipping."
                    )
                    continue

                r, g, b = color_rgb
                lower_bound_bgr = np.array([b, g, r], dtype=np.uint8)
                upper_bound_bgr = np.array([b, g, r], dtype=np.uint8)

                specific_color_mask = cv2.inRange(
                    rgb_img_bgr, lower_bound_bgr, upper_bound_bgr
                )

                contours, _ = cv2.findContours(
                    specific_color_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                for contour in contours:
                    if len(contour) < 3:
                        continue
                    epsilon = epsilon_factor * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) < 3:
                        continue

                    points = [p[0].tolist() for p in approx]
                    result_item = {"points": points, "label": class_name}
                    results.append(result_item)

        return results

    @staticmethod
    def clamp_points(points, image_width, image_height):
        """Clamp points to ensure they are within image boundaries.

        Args:
            points (list): List of points to be clamped.
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            list: Clamped points within the image boundaries.
        """
        return [
            [
                max(0, min(p[0], image_width - 1)),
                max(0, min(p[1], image_height - 1)),
            ]
            for p in points
        ]

    @staticmethod
    def _extract_bbox_answer(content):
        answer_matches = re.findall(
            r"<answer>(.*?)</answer>", content, re.DOTALL
        )
        if answer_matches:
            text = answer_matches[-1]
        else:
            text = content

        try:
            data = json_repair.loads(text)
            if isinstance(data, list) and len(data) > 0:
                return data
            else:
                return []
        except Exception as e:
            logger.error(f"Error while parsing JSON: {e}")
            return []

    def get_coco_data(self, mode):
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

        if mode == "polygon":
            coco_data["type"] = "instances"

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

    @staticmethod
    def gen_quad_from_poly(poly):
        """
        Generate min area quad from poly.
        """
        point_num = poly.shape[0]
        min_area_quad = np.zeros((4, 2), dtype=np.float32)
        rect = cv2.minAreaRect(poly.astype(np.int32))
        box = np.array(cv2.boxPoints(rect))

        first_point_idx = 0
        min_dist = 1e4
        for i in range(4):
            dist = (
                np.linalg.norm(box[(i + 0) % 4] - poly[0])
                + np.linalg.norm(box[(i + 1) % 4] - poly[point_num // 2 - 1])
                + np.linalg.norm(box[(i + 2) % 4] - poly[point_num // 2])
                + np.linalg.norm(box[(i + 3) % 4] - poly[-1])
            )
            if dist < min_dist:
                min_dist = dist
                first_point_idx = i
        for i in range(4):
            min_area_quad[i] = box[(first_point_idx + i) % 4]

        bbox_new = min_area_quad.tolist()
        bbox = []

        for box in bbox_new:
            box = list(map(int, box))
            bbox.append(box)

        return bbox

    @staticmethod
    def get_rotate_crop_image(img, points):
        # Use Green's theory to judge clockwise or counterclockwise
        # author: biyanhua
        d = 0.0
        for index in range(-1, 3):
            d += (
                -0.5
                * (points[index + 1][1] + points[index][1])
                * (points[index + 1][0] - points[index][0])
            )
        if d < 0:  # counterclockwise
            tmp = np.array(points)
            points[1], points[3] = tmp[3], tmp[1]

        try:
            img_crop_width = int(
                max(
                    np.linalg.norm(points[0] - points[1]),
                    np.linalg.norm(points[2] - points[3]),
                )
            )
            img_crop_height = int(
                max(
                    np.linalg.norm(points[0] - points[3]),
                    np.linalg.norm(points[1] - points[2]),
                )
            )
            pts_std = np.float32(
                [
                    [0, 0],
                    [img_crop_width, 0],
                    [img_crop_width, img_crop_height],
                    [0, img_crop_height],
                ]
            )
            M = cv2.getPerspectiveTransform(points, pts_std)
            dst_img = cv2.warpPerspective(
                img,
                M,
                (img_crop_width, img_crop_height),
                borderMode=cv2.BORDER_REPLICATE,
                flags=cv2.INTER_CUBIC,
            )
            dst_img_height, dst_img_width = dst_img.shape[0:2]
            if dst_img_height * 1.0 / dst_img_width >= 1.5:
                dst_img = np.rot90(dst_img)
            return dst_img
        except Exception as e:
            logger.error(e)

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

    def yolo_pose_to_custom(self, input_file, output_file, image_file):
        self.reset()
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        img_w, img_h = self.get_image_size(image_file)
        classes = list(self.pose_classes.keys())
        for i, line in enumerate(lines):
            line = line.strip().split(" ")
            class_index = int(line[0])
            label = classes[class_index]
            # Add rectangle info
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
            shape = {
                "label": label,
                "shape_type": "rectangle",
                "flags": {},
                "points": points,
                "group_id": i,
                "description": None,
                "difficult": False,
                "attributes": {},
            }
            self.custom_data["shapes"].append(shape)
            # Add keypoints info
            keypoint_name = self.pose_classes[label]
            keypoints = line[5:]
            interval = 3 if self.has_vasiable else 2
            for j in range(0, len(keypoints), interval):
                x = float(keypoints[j]) * img_w
                y = float(keypoints[j + 1]) * img_h
                flag = int(keypoints[j + 2]) if self.has_vasiable else 0
                if (x == 0 and y == 0) or (flag == 0 and self.has_vasiable):
                    continue
                if interval == 3 and flag == 1:
                    difficult = True
                else:
                    difficult = False
                shape = {
                    "label": keypoint_name[j // interval],
                    "shape_type": "point",
                    "flags": {},
                    "points": [[x, y]],
                    "group_id": i,
                    "description": None,
                    "difficult": difficult,
                    "attributes": {},
                }
                self.custom_data["shapes"].append(shape)

        self.custom_data["imagePath"] = osp.basename(image_file)
        self.custom_data["imageHeight"] = img_h
        self.custom_data["imageWidth"] = img_w
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def yolo_to_custom(self, input_file, output_file, image_file, mode):
        self.reset()
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        img_w, img_h = self.get_image_size(image_file)
        image_size = np.array([img_w, img_h], np.float64)
        for line in lines:
            line = line.strip().split(" ")
            class_index = int(line[0])
            label = self.classes[class_index]
            if mode == "hbb":
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
            elif mode == "seg":
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

    def voc_to_custom(self, input_file, output_file, image_filename, mode):
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
            points = []
            if obj.find("polygon") is not None and mode == "polygon":
                num_points = len(obj.find("polygon")) // 2
                for i in range(1, num_points + 1):
                    x_tag = f"polygon/x{i}"
                    y_tag = f"polygon/y{i}"
                    x = float(obj.find(x_tag).text)
                    y = float(obj.find(y_tag).text)
                    points.append([x, y])
                shape_type = "polygon"
            elif obj.find("bndbox") is not None and mode in [
                "rectangle",
                "polygon",
            ]:
                xmin = float(obj.find("bndbox/xmin").text)
                ymin = float(obj.find("bndbox/ymin").text)
                xmax = float(obj.find("bndbox/xmax").text)
                ymax = float(obj.find("bndbox/ymax").text)
                points = [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                ]
                shape_type = "rectangle"
            shape = {
                "label": label,
                "description": "",
                "points": points,
                "group_id": None,
                "difficult": bool(int(difficult)),
                "shape_type": shape_type,
                "flags": {},
            }

            self.custom_data["shapes"].append(shape)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def coco_to_custom(self, input_file, output_dir_path, mode):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if mode in ["rectangle", "polygon"]:
            if not self.classes:
                for cat in data["categories"]:
                    self.classes.append(cat["name"])
        elif mode == "pose":
            if not self.pose_classes:
                for cat in data["categories"]:
                    self.pose_classes[cat["name"]] = cat["keypoints"]
                self.classes = list(self.pose_classes.keys())

        total_info, label_info, image_info = {}, {}, {}

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
        image_ids = {}
        for dic_info in data["annotations"]:
            difficult = bool(int(str(dic_info.get("ignore", "0"))))
            label = label_info[dic_info["category_id"]]
            image_id = dic_info["image_id"]

            if mode == "rectangle":
                shape_type = "rectangle"
                bbox = dic_info["bbox"]
                xmin = bbox[0]
                ymin = bbox[1]
                width = bbox[2]
                height = bbox[3]
                xmax = xmin + width
                ymax = ymin + height
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

            elif mode == "polygon":
                shape_type = "polygon"
                segmentations_list = []

                for segmentation in dic_info["segmentation"]:
                    if isinstance(segmentation, dict) and "counts" in segmentation:
                        # TODO: Handle RLE format segmentation
                        continue

                    if not isinstance(segmentation, list):
                        continue

                    if len(segmentation) < 6 or len(segmentation) % 2 != 0:
                        continue
                    segmentations_list.append(segmentation)

                if len(segmentations_list) == 0:
                    continue

                group_id = None
                if len(segmentations_list) > 1:
                    if image_id not in image_info:
                        image_info[image_id] = 1
                    else:
                        image_info[image_id] += 1
                    group_id = image_info[image_id]

                for segmentation in segmentations_list:
                    points = []
                    seen_points = set()
                    for i in range(0, len(segmentation), 2):
                        point = (segmentation[i], segmentation[i + 1])
                        if point not in seen_points:
                            points.append([point[0], point[1]])
                            seen_points.add(point)

                    if not points:
                        continue

                    shape = {
                        "label": label,
                        "shape_type": shape_type,
                        "flags": {},
                        "points": points,
                        "group_id": group_id,
                        "description": None,
                        "difficult": difficult,
                        "attributes": {},
                    }
                    total_info[dic_info["image_id"]]["shapes"].append(shape)

            elif mode == "pose":
                if image_id not in image_ids:
                    image_ids[image_id] = 0
                else:
                    image_ids[image_id] += 1
                # bbox
                shape_type = "rectangle"
                bbox = dic_info["bbox"]
                xmin = bbox[0]
                ymin = bbox[1]
                width = bbox[2]
                height = bbox[3]
                xmax = xmin + width
                ymax = ymin + height
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
                    "group_id": image_ids[image_id],
                    "description": None,
                    "difficult": difficult,
                    "attributes": {},
                }
                total_info[dic_info["image_id"]]["shapes"].append(shape)
                # keypoints
                keypoints = dic_info["keypoints"]
                kpt_names = self.pose_classes[label]
                if len(kpt_names) * 3 == len(keypoints):
                    has_vasiable = True
                else:
                    has_vasiable = False
                interval = 3 if has_vasiable else 2
                for i in range(0, len(keypoints), interval):
                    x = keypoints[i]
                    y = keypoints[i + 1]
                    flag = keypoints[i + 2] if has_vasiable else 0
                    if x == 0 and y == 0 and flag == 0:
                        continue
                    shape = {
                        "label": kpt_names[i // interval],
                        "shape_type": "point",
                        "flags": {},
                        "points": [[x, y]],
                        "group_id": image_ids[image_id],
                        "description": None,
                        "difficult": flag == 1,
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
                output_dir_path,
                osp.splitext(dic_info["imagePath"])[0] + ".json",
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
        with open(input_file, "r", encoding="utf-8") as f:
            mot_data = [line.strip().split(",") for line in f]

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
            frame_id = osp.splitext(file_name.rsplit("_")[-1])[0]
            if frame_id.isdigit():
                frame_id = int(frame_id)
            else:
                match = re.search(r"\d+", frame_id)
                frame_id = int(match.group()) if match else 0

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

    def odvg_to_custom(self, input_file, output_path):
        # Load od.json or od.jsonl
        with jsonlines.open(input_file, "r") as reader:
            od_data = list(reader)
        # Save custom info
        for data in od_data:
            self.reset()
            shapes = []
            for instances in data["detection"]["instances"]:
                xmin, ymin, xmax, ymax = instances["bbox"]
                points = [
                    [xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax],
                ]
                shape = {
                    "label": instances["category"],
                    "description": None,
                    "points": points,
                    "group_id": None,
                    "difficult": False,
                    "direction": 0,
                    "shape_type": "rectangle",
                    "flags": {},
                }
                shapes.append(shape)
            self.custom_data["imagePath"] = data["filename"]
            self.custom_data["imageHeight"] = data["height"]
            self.custom_data["imageWidth"] = data["width"]
            self.custom_data["shapes"] = shapes
            output_file = osp.join(
                output_path, osp.splitext(data["filename"])[0] + ".json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def mmgd_to_custom(
        self, input_file, output_file, image_file, labels, thresholds
    ):
        self.reset()

        with open(input_file, "r", encoding="utf-8") as f:
            mmgd_data = json.load(f)

        required_keys = ["labels", "scores", "bboxes"]
        missing_keys = [key for key in required_keys if key not in mmgd_data]
        if missing_keys:
            logger.error(
                f"Missing required keys in uploaded file: {missing_keys}"
            )
            raise ValueError(
                f"Missing required keys in uploaded file: {missing_keys}"
            )
        indexes = mmgd_data["labels"]
        scores = mmgd_data["scores"]
        bboxes = mmgd_data["bboxes"]

        image_width, image_height = self.get_image_size(image_file)
        self.custom_data["imagePath"] = osp.basename(image_file)
        self.custom_data["imageHeight"] = image_height
        self.custom_data["imageWidth"] = image_width

        valid_label_set = set(labels)
        filtered = [
            (self.classes[index], score, bbox)
            for index, score, bbox in zip(indexes, scores, bboxes)
            if (
                index < len(self.classes)
                and self.classes[index] in valid_label_set
                and score >= thresholds[self.classes[index]]
            )
        ]

        for label, score, bbox in filtered:
            xmin, ymin, xmax, ymax = bbox
            points = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ]
            shape_type = "rectangle"
            shape = {
                "label": label,
                "score": round(score, 2),
                "description": "",
                "points": points,
                "group_id": None,
                "difficult": False,
                "shape_type": shape_type,
                "flags": {},
            }
            self.custom_data["shapes"].append(shape)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def ppocr_to_custom(self, input_file, output_path, image_path, mode):
        if mode in ["rec", "kie"]:
            with open(input_file, "r", encoding="utf-8") as f:
                ocr_data = [line.strip().split("\t", 1) for line in f]

        for data in ocr_data:
            # init
            self.reset()

            # image
            filename = osp.basename(data[0])
            image_file = osp.join(image_path, filename)
            imageWidth, imageHeight = self.get_image_size(image_file)
            self.custom_data["imageHeight"] = imageHeight
            self.custom_data["imageWidth"] = imageWidth
            self.custom_data["imagePath"] = filename

            # label
            shapes = []
            annotations = json.loads(data[1])
            for annotation in annotations:
                points = annotation["points"]
                shape_type = (
                    "rectangle" if is_possible_rectangle(points) else "polygon"
                )
                shape = {
                    "label": annotation.get("label", "text"),
                    "description": annotation["transcription"],
                    "points": points,
                    "group_id": annotation.get("id", None),
                    "difficult": annotation.get("difficult", False),
                    "kie_linking": annotation.get("linking", []),
                    "shape_type": shape_type,
                    "flags": {},
                }
                shapes.append(shape)
            self.custom_data["shapes"] = shapes
            output_file = osp.join(
                output_path, osp.splitext(filename)[0] + ".json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    def vlm_r1_ovd_to_custom(self, input_data, output_file, image_file):
        self.reset()

        image_width, image_height = self.get_image_size(image_file)

        bbox_data = self._extract_bbox_answer(input_data)
        for bbox in bbox_data:
            label = bbox["label"]
            xmin, ymin, xmax, ymax = bbox["bbox_2d"]
            points = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            shape = {
                "label": label,
                "points": points,
                "group_id": None,
                "description": None,
                "shape_type": "rectangle",
            }
            self.custom_data["shapes"].append(shape)

        self.custom_data["imagePath"] = osp.basename(image_file)
        self.custom_data["imageHeight"] = image_height
        self.custom_data["imageWidth"] = image_width

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.custom_data, f, indent=2, ensure_ascii=False)

    # Export functions
    def custom_to_yolo(  # noqa: C901
        self, input_file, output_file, mode, skip_empty_files=False
    ):
        is_empty_file = True
        if osp.exists(input_file):
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            if not skip_empty_files:
                pathlib.Path(output_file).touch()
            return is_empty_file

        image_width = data["imageWidth"]
        image_height = data["imageHeight"]
        image_size = np.array([[image_width, image_height]])
        if mode == "pose":
            pose_data = {}
        with open(output_file, "w", encoding="utf-8") as f:
            for shape in data["shapes"]:
                shape_type = shape["shape_type"]
                if mode == "hbb" and shape_type == "rectangle":
                    label = shape["label"]
                    points = self.clamp_points(
                        shape["points"], image_width, image_height
                    )
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
                        f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    )

                    is_empty_file = False
                elif mode == "seg" and shape_type == "polygon":
                    label = shape["label"]
                    points = np.array(
                        self.clamp_points(
                            shape["points"], image_width, image_height
                        )
                    )
                    if len(points) < 3:
                        continue
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
                    is_empty_file = False
                elif mode == "obb" and shape_type == "rotation":
                    label = shape["label"]
                    points = shape["points"]
                    if not any(
                        0 <= p[0] < image_width and 0 <= p[1] < image_height
                        for p in points
                    ):
                        logger.warning(
                            f"{data['imagePath']}: Skip out of bounds coordinates of {points}!"
                        )
                        continue
                    points = list(chain.from_iterable(points))
                    normalized_coords = [
                        (
                            points[i] / image_width
                            if i % 2 == 0
                            else points[i] / image_height
                        )
                        for i in range(8)
                    ]
                    x0, y0, x1, y1, x2, y2, x3, y3 = normalized_coords
                    class_index = self.classes.index(label)
                    f.write(
                        f"{class_index} {x0} {y0} {x1} {y1} {x2} {y2} {x3} {y3}\n"
                    )
                    is_empty_file = False
                elif mode == "pose":
                    if shape_type not in ["rectangle", "point"]:
                        continue
                    if shape["group_id"] is None:
                        logger.error(
                            f"group_id is None for {shape} in {input_file}."
                        )
                        raise ValueError(
                            f"group_id is None for {shape} in {input_file}."
                        )
                    label = shape["label"]
                    points = self.clamp_points(
                        shape["points"], image_width, image_height
                    )
                    group_id = int(shape["group_id"])
                    if group_id not in pose_data:
                        pose_data[group_id] = {
                            "rectangle": [],
                            "keypoints": {},
                        }
                    if shape_type == "rectangle":
                        if len(points) == 2:
                            points = rectangle_from_diagonal(points)
                        pose_data[group_id]["rectangle"] = points
                        pose_data[group_id]["box_label"] = label
                    else:
                        x, y = points[0]
                        difficult = shape.get("difficult", False)
                        visible = 1 if difficult is True else 2
                        pose_data[group_id]["keypoints"][label] = [
                            x,
                            y,
                            visible,
                        ]
                    is_empty_file = False
            if mode == "pose":
                classes = list(self.pose_classes.keys())
                max_keypoints = max(
                    [len(kpts) for kpts in self.pose_classes.values()]
                )
                for data in pose_data.values():
                    box_label = data["box_label"]
                    box_index = classes.index(box_label)
                    kpt_names = self.pose_classes[box_label]
                    rectangle = data["rectangle"]
                    x_center = (rectangle[0][0] + rectangle[2][0]) / (
                        2 * image_width
                    )
                    y_center = (rectangle[0][1] + rectangle[2][1]) / (
                        2 * image_height
                    )
                    width = (
                        abs(rectangle[2][0] - rectangle[0][0]) / image_width
                    )
                    height = (
                        abs(rectangle[2][1] - rectangle[0][1]) / image_height
                    )
                    x = round(x_center, 6)
                    y = round(y_center, 6)
                    w = round(width, 6)
                    h = round(height, 6)
                    label = f"{box_index} {x} {y} {w} {h}"
                    keypoints = data["keypoints"]
                    for name in kpt_names:
                        # 0: Invisible, 1: Occluded, 2: Visible
                        if name not in keypoints:
                            if self.has_vasiable:
                                label += " 0 0 0"
                            else:
                                label += " 0 0"
                        else:
                            x, y, visible = keypoints[name]
                            x = round((int(x) / image_width), 6)
                            y = round((int(y) / image_height), 6)
                            if self.has_vasiable:
                                label += f" {x} {y} {visible}"
                            else:
                                label += f" {x} {y}"

                    # Pad the label with zeros to meet
                    # the yolov8-pose model's training data format requirements
                    for _ in range(max_keypoints - len(kpt_names)):
                        if self.has_vasiable:
                            label += " 0 0 0"
                        else:
                            label += " 0 0"
                    f.write(f"{label}\n")
        return is_empty_file

    def custom_to_voc(
        self, image_file, input_file, output_dir, mode, skip_empty_files=False
    ):
        is_emtpy_file = True
        image = cv2.imread(image_file)
        image_height, image_width, image_depth = image.shape
        if osp.exists(input_file):
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            shapes = data["shapes"]
        else:
            if not skip_empty_files:
                shapes = []
            else:
                return is_emtpy_file

        image_path = osp.basename(image_file)
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = osp.dirname(output_dir)
        ET.SubElement(root, "filename").text = osp.basename(image_path)
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(image_width)
        ET.SubElement(size, "height").text = str(image_height)
        ET.SubElement(size, "depth").text = str(image_depth)
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = (
            "https://github.com/CVHub520/X-AnyLabeling"
        )
        for shape in shapes:
            label = shape["label"]
            points = self.clamp_points(
                shape["points"], image_width, image_height
            )
            difficult = shape.get("difficult", False)
            object_elem = ET.SubElement(root, "object")
            ET.SubElement(object_elem, "name").text = label
            ET.SubElement(object_elem, "pose").text = "Unspecified"
            ET.SubElement(object_elem, "truncated").text = "0"
            ET.SubElement(object_elem, "occluded").text = "0"
            ET.SubElement(object_elem, "difficult").text = str(int(difficult))
            if shape["shape_type"] == "rectangle" and mode in [
                "rectangle",
                "polygon",
            ]:
                is_emtpy_file = False
                if len(points) == 2:
                    logger.warning(
                        "UserWarning: Diagonal vertex mode is deprecated in X-AnyLabeling release v2.2.0 or later.\n"
                        "Please update your code to accommodate the new four-point mode."
                    )
                    points = rectangle_from_diagonal(points)
                xmin, ymin, xmax, ymax = self.calculate_bounding_box(points)
                bndbox = ET.SubElement(object_elem, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(xmin))
                ET.SubElement(bndbox, "ymin").text = str(int(ymin))
                ET.SubElement(bndbox, "xmax").text = str(int(xmax))
                ET.SubElement(bndbox, "ymax").text = str(int(ymax))
            elif shape["shape_type"] == "polygon" and mode == "polygon":
                if len(points) < 3:
                    continue
                is_emtpy_file = False
                xmin, ymin, xmax, ymax = self.calculate_bounding_box(points)
                bndbox = ET.SubElement(object_elem, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(xmin))
                ET.SubElement(bndbox, "ymin").text = str(int(ymin))
                ET.SubElement(bndbox, "xmax").text = str(int(xmax))
                ET.SubElement(bndbox, "ymax").text = str(int(ymax))
                polygon = ET.SubElement(object_elem, "polygon")
                for i, point in enumerate(points):
                    x_tag = ET.SubElement(polygon, f"x{i+1}")
                    y_tag = ET.SubElement(polygon, f"y{i+1}")
                    x_tag.text = str(point[0])
                    y_tag.text = str(point[1])

        xml_string = ET.tostring(root, encoding="utf-8")
        dom = minidom.parseString(xml_string)
        formatted_xml = dom.toprettyxml(indent="  ")

        with open(output_dir, "w", encoding="utf-8") as f:
            f.write(formatted_xml)

        return is_emtpy_file

    def custom_to_coco(self, image_list, input_path, output_path, mode):
        coco_data = self.get_coco_data(mode)

        if mode == "rectangle":
            for i, class_name in enumerate(self.classes):
                coco_data["categories"].append(
                    {"id": i + 1, "name": class_name, "supercategory": ""}
                )
        elif mode == "polygon":
            class_name_to_id = {}
            if self.classes[0] == "__ignore__":
                self.classes = self.classes[1:]
            if self.classes[0] != "_background_":
                self.classes = ["_background_"] + self.classes
            for i, class_name in enumerate(self.classes):
                class_name_to_id[class_name] = i
                coco_data["categories"].append(
                    {"id": i, "name": class_name, "supercategory": None}
                )
        elif mode == "pose":
            for i, (name, keypoints) in enumerate(self.pose_classes.items()):
                coco_data["categories"].append(
                    {
                        "id": i + 1,
                        "name": name,
                        "supercategory": "",
                        "keypoints": keypoints,
                        "skeleton": [],
                    }
                )

        image_id = 0
        annotation_id = 0

        for image_file in image_list:
            # Reset pose_data for each new image when in pose mode
            if mode == "pose":
                pose_data = {}
            elif mode == "polygon":
                polygon_data = {}

            image_name = osp.basename(image_file)
            label_name = osp.splitext(image_name)[0] + ".json"
            label_file = osp.join(input_path, label_name)
            if not osp.exists(label_file):
                label_file = osp.join(osp.dirname(image_file), label_name)
                if not osp.exists(label_file):
                    continue

            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            image_width = data["imageWidth"]
            image_height = data["imageHeight"]
            coco_data["images"].append(
                {
                    "license": 0,
                    "url": None,
                    "file_name": image_name,
                    "height": image_height,
                    "width": image_width,
                    "date_captured": None,
                    "id": image_id,
                }
            )
            for shape in data["shapes"]:
                label = shape["label"]
                points = self.clamp_points(
                    shape["points"], image_width, image_height
                )

                group_id = shape.get("group_id", None)
                if group_id is not None:
                    group_id = int(group_id)
                else:
                    group_id = hash(uuid.uuid1())

                difficult = shape.get("difficult", False)
                bbox, area = [], 0
                shape_type = shape["shape_type"]

                if mode == "pose":
                    if shape_type in ["point", "rectangle"]:
                        group_id = int(shape["group_id"])
                        if group_id not in pose_data:
                            pose_data[group_id] = {
                                "rectangle": [],
                                "keypoints": {},
                            }
                        if shape_type == "rectangle":
                            if len(points) == 2:
                                points = rectangle_from_diagonal(points)
                            pose_data[group_id]["rectangle"] = points
                            pose_data[group_id]["box_label"] = label
                        else:
                            x, y = points[0]
                            difficult = shape.get("difficult", False)
                            visible = 1 if difficult is True else 2
                            pose_data[group_id]["keypoints"][label] = [
                                x,
                                y,
                                visible,
                            ]

                elif mode == "rectangle":
                    if shape_type != "rectangle":
                        continue

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
                    class_id = self.classes.index(label)

                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "ignore": int(difficult),
                        "segmentation": [],
                    }
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1

                elif mode == "polygon":
                    if shape_type != "polygon":
                        continue

                    if label == "__ignore__" or label not in class_name_to_id:
                        continue

                    instance = (label, group_id)

                    if instance not in polygon_data:
                        polygon_data[instance] = {
                            "label": label,
                            "difficult": difficult,
                            "segmentation": [],
                        }
                    flattened_points = [
                        coord for point in points for coord in point
                    ]
                    polygon_data[instance]["segmentation"].append(
                        flattened_points
                    )

            if mode == "pose":
                for data in pose_data.values():
                    points = data["rectangle"]
                    box_label = data["box_label"]
                    class_id = self.classes.index(box_label)
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

                    keypoints = []
                    kpt_names = self.pose_classes[box_label]
                    num_keypoints = 0
                    for name in kpt_names:
                        # 0: Invisible, 1: Occluded, 2: Visible
                        if name not in data["keypoints"]:
                            if self.has_vasiable:
                                keypoints += [0, 0, 0]
                            else:
                                keypoints += [0, 0]
                        else:
                            num_keypoints += 1
                            x, y, visible = data["keypoints"][name]
                            x = int(x)
                            y = int(y)
                            if self.has_vasiable:
                                keypoints += [x, y, visible]
                            else:
                                keypoints += [x, y]

                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "keypoints": keypoints,
                        "num_keypoints": num_keypoints,
                        "ignore": int(difficult),
                        "segmentation": [],
                    }
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1

            elif mode == "polygon":
                for _, data in polygon_data.items():
                    area = self.calculate_polygon_area(data["segmentation"])
                    bbox = self.get_min_enclosing_bbox(data["segmentation"])

                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_name_to_id[data["label"]],
                        "segmentation": data["segmentation"],
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0,
                        "ignore": int(data["difficult"]),
                    }
                    coco_data["annotations"].append(annotation)

                    annotation_id += 1

            image_id += 1

        if mode == "rectangle":
            output_file = osp.join(output_path, "coco_detection.json")
        elif mode == "polygon":
            output_file = osp.join(
                output_path, "coco_instance_segmentation.json"
            )
        elif mode == "pose":
            output_file = osp.join(output_path, "coco_keypoints.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=4, ensure_ascii=False)

    def custom_to_dota(self, input_file, output_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        w, h = data["imageWidth"], data["imageHeight"]
        with open(output_file, "w", encoding="utf-8") as f:
            for shape in data["shapes"]:
                points = shape["points"]
                shape_type = shape["shape_type"]
                if shape_type != "rotation" or len(points) != 4:
                    continue
                if not any(0 <= p[0] < w and 0 <= p[1] < h for p in points):
                    logger.warning(
                        f"{data['imagePath']}: Skip out of bounds coordinates of {points}!"
                    )
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
            points = self.clamp_points(
                shape["points"], image_width, image_height
            )
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
            # Sort polygons by area to handle overlapping (larger areas first)
            polygons.sort(
                key=lambda x: cv2.contourArea(np.array(x["polygon"])),
                reverse=True,
            )

            for item in polygons:
                label, polygon = item["label"], item["polygon"]
                if label in mapping_color:
                    mask = np.zeros(image_shape, dtype=np.uint8)
                    cv2.fillPoly(
                        mask,
                        [np.array(polygon, dtype=np.int32)],
                        mapping_color[label],
                    )
                    # Only update unassigned pixels (where binary_mask is still 0)
                    binary_mask = np.where(binary_mask == 0, mask, binary_mask)

            cv2.imencode(".png", binary_mask)[1].tofile(output_file)

        elif output_format == "rgb" and polygons:
            # Initialize rgb_mask
            color_mask = np.zeros(
                (image_height, image_width, 3), dtype=np.uint8
            )
            polygons.sort(
                key=lambda x: cv2.contourArea(np.array(x["polygon"])),
                reverse=True,
            )

            for item in polygons:
                label, polygon = item["label"], item["polygon"]
                if label in mapping_color:
                    color = mapping_color[label]
                    # Create mask for current polygon
                    curr_mask = np.zeros(image_shape[:2], dtype=np.uint8)
                    cv2.fillPoly(
                        curr_mask, [np.array(polygon, dtype=np.int32)], 1
                    )
                    # Only update pixels that haven't been assigned yet
                    unassigned = np.all(color_mask == 0, axis=2)
                    color_mask[curr_mask.astype(bool) & unassigned] = color

            cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))[
                1
            ].tofile(output_file)

    def custom_to_mot(self, input_path, save_path):
        mot_structure = {
            "sequence": dict(
                name="MOT",
                imDir=osp.basename(save_path),
                frameRate=30,
                seqLength=None,
                imWidth=None,
                imHeight=None,
                imExt=None,
            ),
            "det": [],
            "gt": [],
        }
        seg_len, im_widht, im_height, im_ext = 0, None, None, None

        label_file_list = os.listdir(input_path)
        label_file_list.sort(
            key=lambda x: (
                int(osp.splitext(x.rsplit("-", 1)[-1])[0])
                if osp.splitext(x.rsplit("-", 1)[-1])[0].isdigit()
                else 0
            )
        )

        for label_file_name in label_file_list:
            if not label_file_name.endswith("json"):
                continue
            label_file = os.path.join(input_path, label_file_name)
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            seg_len += 1
            if im_widht is None:
                im_widht = data["imageWidth"]
            if im_height is None:
                im_height = data["imageHeight"]
            if im_ext is None:
                im_ext = osp.splitext(osp.basename(data["imagePath"]))[-1]

            frame_id = osp.splitext(label_file_name.rsplit("_")[-1])[0]
            if frame_id.isdigit():
                frame_id = int(frame_id)
            else:
                match = re.search(r"\d+", frame_id)
                frame_id = int(match.group()) if match else 0

            for shape in data["shapes"]:
                if shape["shape_type"] != "rectangle":
                    continue
                diccicult = shape.get("diccicult", False)
                class_id = int(self.classes.index(shape["label"]))
                track_id = int(shape["group_id"]) if shape["group_id"] else -1
                points = self.clamp_points(
                    shape["points"], im_widht, im_height
                )
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
                boxw = xmax - xmin
                boxh = ymax - ymin
                det = [frame_id, -1, xmin, ymin, boxw, boxh, 1, -1, -1, -1]
                gt = [
                    frame_id,
                    track_id,
                    xmin,
                    ymin,
                    boxw,
                    boxh,
                    int(not diccicult),
                    class_id,
                    1,
                ]
                mot_structure["det"].append(det)
                mot_structure["gt"].append(gt)

        # Save seqinfo.ini
        mot_structure["sequence"]["seqLength"] = seg_len
        mot_structure["sequence"]["imWidth"] = im_widht
        mot_structure["sequence"]["imHeight"] = im_height
        mot_structure["sequence"]["imExt"] = im_ext
        config = configparser.ConfigParser()
        config.add_section("Sequence")
        for key, value in mot_structure["sequence"].items():
            config["Sequence"][key] = str(value)
        with open(osp.join(save_path, "seqinfo.ini"), "w") as f:
            config.write(f)
        # Save det.txt
        with open(osp.join(save_path, "det.txt"), "w", encoding="utf-8") as f:
            for row in mot_structure["det"]:
                f.write(",".join(map(str, row)) + "\n")
        # Save gt.txt
        with open(osp.join(save_path, "gt.txt"), "w", encoding="utf-8") as f:
            for row in mot_structure["gt"]:
                f.write(",".join(map(str, row)) + "\n")

    def custom_to_mots(self, input_path, save_path):
        mots_structure = {
            "sequence": dict(
                name="MOTS",
                imDir=osp.basename(save_path),
                frameRate=30,
                seqLength=None,
                imWidth=None,
                imHeight=None,
                imExt=None,
            ),
            "gt": [],
        }
        seg_len, im_widht, im_height, im_ext = 0, None, None, None

        label_file_list = os.listdir(input_path)
        label_file_list.sort(
            key=lambda x: (
                int(osp.splitext(x.rsplit("-", 1)[-1])[0])
                if osp.splitext(x.rsplit("-", 1)[-1])[0].isdigit()
                else 0
            )
        )

        for label_file_name in label_file_list:
            if not label_file_name.endswith("json"):
                continue
            label_file = os.path.join(input_path, label_file_name)
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            seg_len += 1
            if im_widht is None:
                im_widht = data["imageWidth"]
            if im_height is None:
                im_height = data["imageHeight"]
            if im_ext is None:
                im_ext = osp.splitext(osp.basename(data["imagePath"]))[-1]

            frame_id = osp.splitext(label_file_name.split("_")[-1])[0]
            if frame_id.isdigit():
                frame_id = int(frame_id)
            else:
                match = re.search(r"\d+", frame_id)
                frame_id = int(match.group()) if match else 0

            for shape in data["shapes"]:
                if shape["shape_type"] != "polygon":
                    continue
                class_id = int(self.classes.index(shape["label"]))
                track_id = int(shape["group_id"]) if shape["group_id"] else -1
                points = self.clamp_points(
                    shape["points"], im_widht, im_height
                )
                gt = [
                    frame_id,
                    track_id,
                    class_id,
                    im_height,
                    im_widht,
                    points,
                ]
                mots_structure["gt"].append(gt)

        # Save seqinfo.ini
        mots_structure["sequence"]["seqLength"] = seg_len
        mots_structure["sequence"]["imWidth"] = im_widht
        mots_structure["sequence"]["imHeight"] = im_height
        mots_structure["sequence"]["imExt"] = im_ext
        config = configparser.ConfigParser()
        config.add_section("Sequence")
        for key, value in mots_structure["sequence"].items():
            config["Sequence"][key] = str(value)
        with open(osp.join(save_path, "seqinfo.ini"), "w") as f:
            config.write(f)
        # Save gt.txt
        with open(
            osp.join(save_path, "custom_gt.txt"), "w", encoding="utf-8"
        ) as f:
            for row in mots_structure["gt"]:
                f.write(" ".join(map(str, row)) + "\n")

    def custom_to_odvg(self, image_list, label_path, save_path):
        # Save label_map.json
        label_map = {}
        for i, c in enumerate(self.classes):
            label_map[i] = c
        label_map_file = osp.join(save_path, "label_map.json")
        with open(label_map_file, "w") as f:
            json.dump(label_map, f)
        # Save od.json
        od_data = []
        for image_file in image_list:
            image_name = osp.basename(image_file)
            label_name = osp.splitext(image_name)[0] + ".json"
            label_file = osp.join(label_path, label_name)
            if not osp.exists(label_file):
                label_file = osp.join(osp.dirname(image_file), label_name)
            img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), 1)
            height, width = img.shape[:2]
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            instances = []
            for shape in data["shapes"]:
                if (
                    shape["shape_type"] != "rectangle"
                    or shape["label"] not in self.classes
                ):
                    continue
                points = self.clamp_points(shape["points"], width, height)
                xmin = float(points[0][0])
                ymin = float(points[0][1])
                xmax = float(points[2][0])
                ymax = float(points[2][1])
                bbox = [xmin, ymin, xmax, ymax]
                label = self.classes.index(shape["label"])
                category = shape["label"]
                instances.append(
                    {"bbox": bbox, "label": label, "category": category}
                )
            od_data.append(
                {
                    "filename": image_name,
                    "height": height,
                    "width": width,
                    "detection": {"instances": instances},
                }
            )
        od_file = osp.join(save_path, "od.json")
        with jsonlines.open(od_file, mode="w") as writer:
            writer.write_all(od_data)

    def custom_to_vlm_r1_ovd(
        self, image_list, label_path, save_path, prefix=""
    ):
        with jsonlines.open(save_path, mode="w") as writer:
            for i, image_file in enumerate(image_list):
                image_name = osp.basename(image_file)
                label_name = osp.splitext(image_name)[0] + ".json"
                label_file = osp.join(label_path, label_name)
                if not osp.exists(label_file):
                    label_file = osp.join(osp.dirname(image_file), label_name)

                img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), 1)
                height, width = img.shape[:2]

                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                box_labels, unique_labels = [], set()
                for shape in data["shapes"]:
                    if shape["shape_type"] != "rectangle":
                        continue

                    points = self.clamp_points(shape["points"], width, height)
                    xmin = int(points[0][0])
                    ymin = int(points[0][1])
                    xmax = int(points[2][0])
                    ymax = int(points[2][1])

                    label = shape["label"]
                    if not self.classes:
                        box_labels.append(
                            {
                                "bbox_2d": [xmin, ymin, xmax, ymax],
                                "label": label,
                            }
                        )
                        unique_labels.add(label)
                    else:
                        if label in self.classes:
                            box_labels.append(
                                {
                                    "bbox_2d": [xmin, ymin, xmax, ymax],
                                    "label": label,
                                }
                            )
                            unique_labels.add(label)

                if not box_labels and not self.classes:
                    continue

                if self.classes and not box_labels:
                    answer = "None"
                else:
                    box_strings = [
                        f'{{"bbox_2d": {b["bbox_2d"]}, "label": "{b["label"]}"}}'
                        for b in box_labels
                    ]
                    boxes_str = ",\n ".join(box_strings)
                    answer = f"""```json\n[\n{boxes_str}\n]\n```"""

                if self.classes:
                    labels = self.classes
                else:
                    labels = list(unique_labels)

                question = f"Please carefully check the image and detect the following objects: {labels}. "
                question = (
                    question
                    + 'Output the bbox coordinates of detected objects in JSON format in <answer></answer>. The bbox coordinate format should be: \n```json\n[{"bbox_2d": [x1, y1, x2, y2], "label": "object name"},\n{"bbox_2d": [x1, y1, x2, y2], "label": "object name"},\n...\n]\n```\n. If no targets are detected in the image, simply respond with "None".'
                )

                item = {
                    "id": i + 1,
                    "image": prefix + image_name,
                    "conversations": [
                        {"from": "human", "value": question},
                        {"from": "assistant", "value": answer},
                    ],
                }

                writer.write(item)

    def custom_to_ppocr(self, image_file, label_file, save_path, mode):
        if not osp.exists(label_file):
            return
        image_name = osp.basename(image_file)
        prefix = osp.splitext(image_name)[0]
        dir_name = osp.basename(osp.dirname(image_file))

        avaliable_shape_types = ["rectangle", "rotation", "polygon"]
        img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), 1)
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        image_width = data["imageWidth"]
        image_height = data["imageHeight"]

        if mode == "rec":
            crop_img_count, rec_gt, annotations = 0, [], []
            Label_file = osp.join(save_path, "Label.txt")
            rec_gt_file = osp.join(save_path, "rec_gt.txt")
            save_crop_img_path = osp.join(save_path, "crop_img")

            for shape in data["shapes"]:
                shape_type = shape["shape_type"]
                if shape_type not in avaliable_shape_types:
                    continue
                transcription = shape["description"]
                difficult = shape.get("difficult", False)
                points = [
                    list(map(int, p))
                    for p in self.clamp_points(
                        shape["points"], image_width, image_height
                    )
                ]
                annotations.append(
                    dict(
                        transcription=transcription,
                        points=points,
                        difficult=difficult,
                    )
                )
                if len(points) > 4:
                    points = self.gen_quad_from_poly(np.array(points))
                assert len(points) == 4
                img_crop = self.get_rotate_crop_image(
                    img, np.array(points, np.float32)
                )
                if img_crop is None:
                    logger.warning(
                        f"Can not recognise the detection box in {image_file}. Please change manually"
                    )
                    continue
                crop_img_filenmame = f"{prefix}_crop_{crop_img_count}.jpg"
                crop_img_file = osp.join(
                    save_crop_img_path, crop_img_filenmame
                )
                cv2.imwrite(crop_img_file, img_crop)
                rec_gt.append(
                    f"crop_img/{crop_img_filenmame}\t{transcription}\n"
                )
                crop_img_count += 1
            if annotations:
                Label = f"{dir_name}/{image_name}\t{json.dumps(annotations, ensure_ascii=False)}\n"
                with open(Label_file, "a", encoding="utf-8") as f:
                    f.write(Label)
                with open(rec_gt_file, "a", encoding="utf-8") as f:
                    for item in rec_gt:
                        f.write(item)
        elif mode == "kie":
            annotations, class_set = [], set()
            ppocr_kie_file = osp.join(save_path, "ppocr_kie.json")
            for shape in data["shapes"]:
                shape_type = shape["shape_type"]
                if shape_type not in avaliable_shape_types:
                    continue
                label = shape["label"]
                class_set.add(label)
                transcription = shape["description"]
                group_id = shape.get("group_id", 0)
                kie_linking = shape.get("kie_linking", [])
                difficult = shape.get("difficult", False)
                points = [list(map(int, p)) for p in shape["points"]]
                annotations.append(
                    dict(
                        transcription=transcription,
                        label=label,
                        points=points,
                        difficult=difficult,
                        id=group_id,
                        linking=kie_linking,
                    )
                )
            if annotations:
                item = f"{dir_name}/{image_name}\t{json.dumps(annotations, ensure_ascii=False)}\n"
                with open(ppocr_kie_file, "a", encoding="utf-8") as f:
                    f.write(item)
            return class_set
