import hashlib
import json
import os
import re
import shutil
import random
from datetime import datetime
from typing import List

from ._io import load_yaml_config, save_yaml_config
from .config import (
    get_dataset_path,
    TASK_LABEL_MAPPINGS,
    TASK_SHAPE_MAPPINGS,
)


def _link_or_copy_image(source_path: str, destination_path: str) -> None:
    try:
        os.symlink(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)


def _get_wsl_unc_root(data_file: str) -> str | None:
    normalized_path = data_file.replace("\\", "/")
    match = re.match(
        r"^//(wsl(?:\.localhost|\$))/([^/]+)(?:/|$)",
        normalized_path,
        re.IGNORECASE,
    )
    if not match:
        return None
    return f"//{match.group(1)}/{match.group(2)}"


def _map_wsl_absolute_path(path: str, wsl_root: str | None) -> str | None:
    if not wsl_root or not path.startswith("/") or path.startswith("//"):
        return None
    return f"{wsl_root}{path}"


def resolve_prepared_dataset(data_file: str) -> tuple[str | None, str]:
    data = load_yaml_config(data_file)
    if not isinstance(data, dict):
        return None, f"Failed to parse dataset YAML: {data_file}"

    dataset_root = data.get("path")
    if not isinstance(dataset_root, str) or not dataset_root.strip():
        return None, f"Dataset YAML must define 'path': {data_file}"
    dataset_root = os.path.expanduser(os.path.expandvars(dataset_root))
    wsl_root = _get_wsl_unc_root(data_file)
    normalized_data = dict(data)
    normalized = False
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.join(
            os.path.dirname(os.path.abspath(data_file)), dataset_root
        )
    if not os.path.isdir(dataset_root):
        mapped_root = _map_wsl_absolute_path(dataset_root, wsl_root)
        if not mapped_root or not os.path.isdir(mapped_root):
            checked_path = (
                f"; also checked: {mapped_root}" if mapped_root else ""
            )
            return (
                None,
                f"Dataset path does not exist: {dataset_root}{checked_path}",
            )
        dataset_root = mapped_root
        normalized_data["path"] = mapped_root
        normalized = True

    for split in ("train", "val"):
        split_path = data.get(split)
        if not isinstance(split_path, str) or not split_path.strip():
            return None, f"Dataset YAML must define '{split}': {data_file}"
        split_path = os.path.expanduser(os.path.expandvars(split_path))
        if not os.path.isabs(split_path):
            split_path = os.path.join(dataset_root, split_path)
        if not os.path.isdir(split_path):
            mapped_split = _map_wsl_absolute_path(split_path, wsl_root)
            if not mapped_split or not os.path.isdir(mapped_split):
                checked_path = (
                    f"; also checked: {mapped_split}" if mapped_split else ""
                )
                return (
                    None,
                    f"Dataset split '{split}' does not exist: "
                    f"{split_path}{checked_path}",
                )
            split_path = mapped_split
            normalized_data[split] = mapped_split
            normalized = True

    if not normalized:
        return data_file, ""

    test_path = normalized_data.get("test")
    mapped_test = (
        _map_wsl_absolute_path(test_path, wsl_root)
        if isinstance(test_path, str)
        else None
    )
    if mapped_test:
        normalized_data["test"] = mapped_test

    source_path = data_file.replace("\\", "/")
    source_name = os.path.splitext(os.path.basename(source_path))[0]
    source_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_name)
    fingerprint = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:12]
    resolved_dir = os.path.join(get_dataset_path(), "resolved")
    resolved_path = os.path.join(
        resolved_dir,
        f"{source_name or 'dataset'}_{fingerprint}.yaml",
    )
    try:
        os.makedirs(resolved_dir, exist_ok=True)
    except OSError as error:
        return None, f"Failed to prepare resolved dataset YAML: {error}"
    if not save_yaml_config(normalized_data, resolved_path):
        return None, f"Failed to save resolved dataset YAML: {resolved_path}"
    return resolved_path, ""


def is_prepared_dataset(data_file: str) -> bool:
    resolved_path, _ = resolve_prepared_dataset(data_file)
    return resolved_path is not None


def create_yolo_dataset(
    image_list: List[str],
    task_type: str,
    dataset_ratio: float,
    data_file: str,
    output_dir: str = None,
    pose_cfg_file: str = None,
    skip_empty_files: bool = False,
    only_checked_files: bool = False,
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
        only_checked_files: Whether to use only checked files

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
                _link_or_copy_image(image_file, dst_image_path)
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
                            _link_or_copy_image(image_file, dst_image_path)
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
        if not isinstance(data, dict):
            raise ValueError(f"Failed to parse dataset YAML: {data_file}")
        names = data.get("names")
        if not isinstance(names, (dict, list)) or not names:
            raise ValueError(
                f"Dataset YAML must contain a non-empty 'names' field: {data_file}"
            )
        if task_type.lower() == "pose":
            if not pose_cfg_file:
                return (
                    None,
                    "Pose configuration file is required for pose detection tasks",
                )
            converter = LabelConverter(pose_cfg_file=pose_cfg_file)
        else:
            converter = LabelConverter()
        converter.classes = (
            [names[index] for index in sorted(names)]
            if isinstance(names, dict)
            else names
        )
        data_file_name = os.path.splitext(os.path.basename(data_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.join(
        get_dataset_path(), task_type.lower(), f"{data_file_name}_{timestamp}"
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
            if only_checked_files:
                continue
            background_images.append(image_file)
            continue

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                label_info = json.load(f)

            if (
                only_checked_files
                and label_info.get("checked", False) is not True
            ):
                continue

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
            if only_checked_files:
                continue
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
            f.write(f"Only checked files: {only_checked_files}\n")
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
