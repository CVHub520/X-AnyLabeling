import os
import subprocess
import sys
import yaml
from typing import Dict, List, Tuple, Union

from .utils import get_task_valid_images
from .config import MIN_LABELED_IMAGES_THRESHOLD


def validate_basic_config(config: Dict) -> Tuple[Union[bool, str], str]:
    """Validate basic training configuration

    Args:
        config: Training configuration dictionary

    Returns:
        Tuple of (is_valid_or_status, error_message_or_path)
        - (True, "") - validation passed
        - (False, error_message) - validation failed
        - ("directory_exists", directory_path) - directory exists, needs user confirmation
    """
    basic = config.get("basic", {})

    if not basic.get("project", "").strip():
        return False, "Project field is required"

    if not basic.get("name", "").strip():
        return False, "Name field is required"

    save_dir = os.path.join(basic["project"], basic["name"])
    if os.path.exists(save_dir):
        return "directory_exists", save_dir

    model_path = basic.get("model", "").strip()
    if not model_path or not os.path.exists(model_path):
        return False, "Valid model file is required"

    data_path = basic.get("data", "").strip()
    if not data_path or not os.path.exists(data_path):
        return False, "Valid data file is required"

    return True, ""


def validate_classes(classes_str: str, names: List[str]) -> Tuple[bool, str]:
    """Validate class indices against available class names.

    Args:
        classes_str: String containing class indices to validate, can be empty
        names: List of available class names to validate against

    Returns:
        Tuple containing:
            - bool: True if validation passes, False if validation fails
            - str: Empty string if validation passes, error message if validation fails
    """
    if not classes_str.strip():
        return True, ""

    from .general import parse_string_to_digit_list

    classes = parse_string_to_digit_list(classes_str)

    if classes is None:
        return False, "Invalid classes format"

    max_index = len(names) - 1
    for cls_idx in classes:
        if cls_idx < 0 or cls_idx > max_index:
            return False, f"Class index {cls_idx} out of range (0-{max_index})"

    return True, ""


def validate_data_file(file_path: str) -> Tuple[bool, Union[str, List[str]]]:
    """Validate data YAML file.

    Args:
        file_path (str): Path to the data file.

    Returns:
        Tuple[bool, Union[str, List[str]]]: (is_valid, error_message_or_names_list)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "names" not in data:
            return False, "Data file must contain 'names' field"

        return True, list(data["names"].values())

    except Exception as e:
        return False, f"Failed to read file: {e}"


def install_packages_with_timeout(packages, timeout=30):
    cmd = [sys.executable, "-m", "pip", "install"] + packages

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode == 0, stdout, stderr

    except subprocess.TimeoutExpired:
        process.kill()
        return False, "", "Installation timed out"
    except Exception as e:
        return False, "", str(e)


def validate_task_requirements(
    task_type: str, image_list: List[str], output_dir: str = None
) -> Tuple[bool, str]:
    if not task_type:
        return False, "Please select a task type"

    if not image_list:
        return False, "Please load images first"

    valid_images = get_task_valid_images(image_list, task_type, output_dir)

    if valid_images < MIN_LABELED_IMAGES_THRESHOLD:
        return (
            False,
            f"Need at least {MIN_LABELED_IMAGES_THRESHOLD} labeled images for {task_type} task. Found: {valid_images}",
        )

    return True, ""
