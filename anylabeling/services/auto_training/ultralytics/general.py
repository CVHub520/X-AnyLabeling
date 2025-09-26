import json
import os
import re
import shutil
import random
from datetime import datetime
from typing import List

from ._io import load_yaml_config, save_yaml_config
from .config import DATASET_PATH, TASK_LABEL_MAPPINGS, TASK_SHAPE_MAPPINGS


def create_yolo_dataset(
    image_list: List[str],
    task_type: str,
    dataset_ratio: float,
    data_file: str,
    output_dir: str = None,
    pose_cfg_file: str = None,
    skip_empty_files: bool = False,
) -> str:
    """Create YOLO dataset from image list and annotations.

    Args:
        image_list: List of image paths
        task_type: Type of detection task
        dataset_ratio: Ratio to split train/val data
        data_file: Path to data config file
        output_dir: Optional output directory for labels
        pose_cfg_file: Optional pose config file for pose detection
        skip_empty_files: Whether to skip empty label files

    Returns:
        Path to created dataset directory
    """
    from anylabeling.views.labeling.label_converter import LabelConverter

    def _process_images_batch(
        image_label_pairs, images_dir, labels_dir, converter, mode, skip_empty
    ):
        for image_file, label_file in image_label_pairs:
            filename = os.path.basename(image_file)
            dst_image_path = os.path.join(images_dir, filename)

            if os.name == "nt":  # Windows
                shutil.copy2(image_file, dst_image_path)
            else:
                os.symlink(image_file, dst_image_path)

            if label_file and os.path.exists(label_file):
                dst_label_path = os.path.join(
                    labels_dir, os.path.splitext(filename)[0] + ".txt"
                )
                converter.custom_to_yolo(
                    label_file,
                    dst_label_path,
                    mode,
                    skip_empty_files=skip_empty,
                )

    def _process_classify_images_batch(image_label_pairs, base_dir):
        for image_file, label_file in image_label_pairs:
            filename = os.path.basename(image_file)

            if not label_file or not os.path.exists(label_file):
                continue

            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                flags = data.get("flags", {})

                for flag_name, flag_value in flags.items():
                    if flag_value:
                        class_dir = os.path.join(base_dir, flag_name)
                        os.makedirs(class_dir, exist_ok=True)
                        dst_image_path = os.path.join(class_dir, filename)

                        if os.name == "nt":  # Windows
                            shutil.copy2(image_file, dst_image_path)
                        else:
                            os.symlink(image_file, dst_image_path)
                        break
            except (json.JSONDecodeError, IOError):
                continue

    if task_type == "Classify":
        data = {"names": {}, "nc": 0}
        converter = None
        data_file_name = "classification"
    else:
        data = load_yaml_config(data_file)
        if task_type.lower() == "pose":
            if not pose_cfg_file:
                return (
                    None,
                    "Pose configuration file is required for pose detection tasks",
                )
            converter = LabelConverter(pose_cfg_file=pose_cfg_file)
        else:
            converter = LabelConverter()
        converter.classes = [
            data["names"][i] for i in sorted(data["names"].keys())
        ]
        data_file_name = os.path.splitext(os.path.basename(data_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.join(
        DATASET_PATH, task_type.lower(), f"{data_file_name}_{timestamp}"
    )

    if task_type == "Classify":
        train_dir = os.path.join(temp_dir, "train")
        val_dir = os.path.join(temp_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
    else:
        train_images_dir = os.path.join(temp_dir, "images", "train")
        val_images_dir = os.path.join(temp_dir, "images", "val")
        train_labels_dir = os.path.join(temp_dir, "labels", "train")
        val_labels_dir = os.path.join(temp_dir, "labels", "val")
        for dir_path in [
            train_images_dir,
            val_images_dir,
            train_labels_dir,
            val_labels_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

    background_images = []
    valid_images = []
    valid_shapes = TASK_SHAPE_MAPPINGS.get(task_type, [])

    for image_file in image_list:
        label_dir, filename = os.path.split(image_file)
        if output_dir:
            label_dir = output_dir
        label_file = os.path.join(
            label_dir, os.path.splitext(filename)[0] + ".json"
        )

        if not os.path.exists(label_file):
            background_images.append(image_file)
            continue

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                label_info = json.load(f)

            if task_type == "Classify":
                flags = label_info.get("flags", {})
                has_valid_flag = any(
                    flag_value for flag_value in flags.values()
                )
                if has_valid_flag:
                    valid_images.append((image_file, label_file))
                else:
                    background_images.append(image_file)
            else:
                shapes = label_info.get("shapes", [])
                has_valid_shape = any(
                    shape.get("shape_type") in valid_shapes
                    for shape in shapes
                    if "shape_type" in shape
                )
                if has_valid_shape:
                    valid_images.append((image_file, label_file))
                else:
                    background_images.append(image_file)
        except Exception:
            background_images.append(image_file)
            continue

    # ensure train/val split is randomized
    valid_images = random.sample(valid_images, k=len(valid_images))

    train_count = int(len(valid_images) * dataset_ratio)
    train_valid_images = valid_images[:train_count]
    val_valid_images = valid_images[train_count:]

    if task_type == "Classify":
        _process_classify_images_batch(train_valid_images, train_dir)
        _process_classify_images_batch(val_valid_images, val_dir)
    else:
        if skip_empty_files:
            all_train_images = train_valid_images
        else:
            all_train_images = [
                (img, None) for img in background_images
            ] + train_valid_images

        mode = TASK_LABEL_MAPPINGS.get(task_type, "hbb")
        _process_images_batch(
            all_train_images,
            train_images_dir,
            train_labels_dir,
            converter,
            mode,
            skip_empty_files,
        )
        _process_images_batch(
            val_valid_images,
            val_images_dir,
            val_labels_dir,
            converter,
            mode,
            skip_empty_files,
        )

    info_file = os.path.join(temp_dir, "dataset_info.txt")
    with open(info_file, "w", encoding="utf-8") as f:
        f.write(
            f"Dataset created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Task type: {task_type}\n")
        f.write(f"Total images: {len(image_list)}\n")
        if task_type == "Classify":
            f.write(f"Train images: {len(train_valid_images)}\n")
            f.write(f"Val images: {len(val_valid_images)}\n")
        else:
            f.write(f"Train images: {len(all_train_images)}\n")
            f.write(f"Val images: {len(val_valid_images)}\n")
            f.write(f"Background images: {len(background_images)}\n")
            f.write(f"Skip empty files: {skip_empty_files}\n")
        f.write(f"Valid labeled images: {len(valid_images)}\n")
        f.write(f"Dataset ratio: {dataset_ratio}\n")

    yaml_file = os.path.join(temp_dir, "data.yaml")

    if task_type == "Classify":
        class_names = {}
        train_dir = os.path.join(temp_dir, "train")
        if os.path.exists(train_dir):
            class_dirs = [
                d
                for d in os.listdir(train_dir)
                if os.path.isdir(os.path.join(train_dir, d))
            ]
            for i, class_name in enumerate(sorted(class_dirs)):
                class_names[i] = class_name

        data = {
            "path": temp_dir,
            "train": "train",
            "val": "val",
            "names": class_names,
            "nc": len(class_names),
        }
    else:
        data["path"] = temp_dir
        data["train"] = "images/train"
        data["val"] = "images/val"

    save_yaml_config(data, yaml_file)

    return temp_dir


def format_classes_display(classes_value) -> str:
    """Formats class values for display.

    This function takes a classes value and formats it into a string representation.
    It handles None values, empty values, lists, and single values.

    Args:
        classes_value: The value to format. Can be None, a list, or a single value.

    Returns:
        A string representation of the classes value:
        - Empty string if input is None or empty
        - Comma-separated string if input is a list
        - String conversion of the input value otherwise
    """
    if classes_value is None or not classes_value:
        return ""
    if isinstance(classes_value, list):
        return ",".join(map(str, classes_value))
    return str(classes_value) if classes_value else ""


def parse_string_to_digit_list(input_string: str) -> List[int]:
    """Parses a string containing numbers into a list of integers.

    This function uses regular expressions to find all numerical digits
    in the input string, treating any non-digit characters as delimiters.
    It then converts the found sequences of digits into integers.

    Args:
        input_string: The string to parse. It can contain numbers
            separated by commas, spaces, or any other non-digit symbols.
            Example: "1, 2 3-4".

    Returns:
        A list of integers found in the string. For example, for the input
        "1, 2 3-4", the output would be [1, 2, 3, 4]. Returns None if
        no numbers are found, input is empty, or parsing fails.
    """
    try:
        if not input_string:
            return None

        numbers_as_strings = re.findall(r"\d+", input_string)
        if not numbers_as_strings:
            return None

        return [int(num) for num in numbers_as_strings]

    except Exception:
        return None
