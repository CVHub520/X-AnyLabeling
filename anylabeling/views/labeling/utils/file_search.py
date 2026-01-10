import json
import os.path as osp
import re
from typing import Dict, Optional


class SearchPattern:
    """Represents a parsed search pattern."""

    def __init__(
        self,
        mode: str,
        pattern: Optional[str] = None,
        regex: Optional[re.Pattern] = None,
        attribute_filter: Optional[Dict] = None,
    ):
        self.mode = mode
        self.pattern = pattern
        self.regex = regex
        self.attribute_filter = attribute_filter


def parse_search_pattern(search_text: str) -> SearchPattern:
    """
    Parse search text into a SearchPattern object.

    Args:
        search_text: The search text input by user.

    Returns:
        SearchPattern object containing parsed search information.

    Examples:
        - "test" -> normal text search
        - "<\.png$>" -> regex search
        - "difficult::1" -> attribute search for difficult objects
        - "gid::0" -> attribute search for files with group_id 0
        - "gid::1" -> attribute search for files with group_id 1
        - "shape::1" -> attribute search for files with any shapes
        - "label::person" -> attribute search for files with label "person"
        - "type::rectangle" -> attribute search for files with shape_type "rectangle"
        - "score::[0,0.5]" -> attribute search for files with score in [0, 0.5]
        - "score::(0,0.6]" -> attribute search for files with score in (0, 0.6]
        - "description::1" -> attribute search for files with non-empty description
        - "description::true" -> attribute search for files with non-empty description
    """
    if not search_text:
        return SearchPattern(mode="normal", pattern=None)

    search_text = search_text.strip()
    if not search_text:
        return SearchPattern(mode="normal", pattern=None)

    if (
        search_text.startswith("<")
        and search_text.endswith(">")
        and len(search_text) > 2
    ):
        try:
            regex_pattern = re.compile(search_text[1:-1], re.IGNORECASE)
            return SearchPattern(mode="regex", regex=regex_pattern)
        except re.error:
            return SearchPattern(mode="normal", pattern=search_text)

    if "::" in search_text:
        parts = search_text.split("::", 1)
        if len(parts) == 2:
            attr_name = parts[0].strip().lower()
            attr_value = parts[1].strip()

            if attr_name == "difficult":
                attribute_filter = {
                    "type": "difficult",
                    "value": attr_value.lower() in ("true", "1", "yes"),
                }
                return SearchPattern(
                    mode="attribute", attribute_filter=attribute_filter
                )
            elif attr_name == "gid":
                try:
                    gid_value = int(attr_value.strip())
                    attribute_filter = {
                        "type": "gid",
                        "value": gid_value,
                    }
                    return SearchPattern(
                        mode="attribute", attribute_filter=attribute_filter
                    )
                except ValueError:
                    pass
            elif attr_name == "shape":
                attribute_filter = {
                    "type": "shape",
                    "value": attr_value.lower() in ("true", "1", "yes"),
                }
                return SearchPattern(
                    mode="attribute", attribute_filter=attribute_filter
                )
            elif attr_name == "label":
                attribute_filter = {
                    "type": "label",
                    "value": attr_value,
                }
                return SearchPattern(
                    mode="attribute", attribute_filter=attribute_filter
                )
            elif attr_name == "type":
                attribute_filter = {
                    "type": "type",
                    "value": attr_value,
                }
                return SearchPattern(
                    mode="attribute", attribute_filter=attribute_filter
                )
            elif attr_name == "score":
                score_range = _parse_score_range(attr_value)
                if score_range:
                    attribute_filter = {
                        "type": "score",
                        "value": score_range,
                    }
                    return SearchPattern(
                        mode="attribute", attribute_filter=attribute_filter
                    )
            elif attr_name == "description":
                attribute_filter = {
                    "type": "description",
                    "value": attr_value.lower() in ("true", "1", "yes"),
                }
                return SearchPattern(
                    mode="attribute", attribute_filter=attribute_filter
                )

    return SearchPattern(mode="normal", pattern=search_text)


def _parse_score_range(range_str: str) -> Optional[Dict]:
    """
    Parse score range string into a dictionary.

    Args:
        range_str: Range string like "[0,0.5]", "(0,0.6]", "[0,0.6)", "(0,0.6)"

    Returns:
        Dictionary with 'min', 'max', 'min_inclusive', 'max_inclusive' keys,
        or None if parsing fails.
    """
    range_str = range_str.strip()
    if not range_str:
        return None

    if len(range_str) < 5:
        return None

    left_bracket = range_str[0]
    right_bracket = range_str[-1]

    if left_bracket not in ("[", "(") or right_bracket not in ("]", ")"):
        return None

    min_inclusive = left_bracket == "["
    max_inclusive = right_bracket == "]"

    content = range_str[1:-1]
    parts = content.split(",")
    if len(parts) != 2:
        return None

    try:
        min_val = float(parts[0].strip())
        max_val = float(parts[1].strip())
        if min_val > max_val:
            return None
        return {
            "min": min_val,
            "max": max_val,
            "min_inclusive": min_inclusive,
            "max_inclusive": max_inclusive,
        }
    except (ValueError, AttributeError):
        return None


def matches_filename(filename: str, search_pattern: SearchPattern) -> bool:
    """
    Check if filename matches the search pattern.

    Args:
        filename: The filename to check.
        search_pattern: The parsed search pattern.

    Returns:
        True if filename matches, False otherwise.
    """
    if search_pattern.mode == "regex":
        if search_pattern.regex:
            return bool(search_pattern.regex.search(filename))
        return False
    elif search_pattern.mode == "normal":
        if search_pattern.pattern:
            return search_pattern.pattern in filename
        return True
    return True


def matches_label_attribute(
    image_file: str,
    label_file: str,
    search_pattern: SearchPattern,
) -> bool:
    """
    Check if label file contains shapes matching the attribute filter.

    Args:
        image_file: Path to the image file.
        label_file: Path to the label JSON file.
        search_pattern: The parsed search pattern.

    Returns:
        True if label file matches the attribute filter, False otherwise.
    """
    if (
        search_pattern.mode != "attribute"
        or not search_pattern.attribute_filter
    ):
        return True

    if not osp.exists(label_file):
        return False

    try:
        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        shapes = data.get("shapes", [])
        filter_type = search_pattern.attribute_filter["type"]
        filter_value = search_pattern.attribute_filter["value"]

        if filter_type == "shape":
            has_shapes = len(shapes) > 0
            return has_shapes == filter_value

        if filter_type == "gid":
            target_gid = filter_value
            for shape in shapes:
                group_id = shape.get("group_id", -1)
                if group_id is not None and group_id == target_gid:
                    return True
            return False

        if filter_type == "difficult":
            for shape in shapes:
                if "difficult" not in shape:
                    continue
                shape_value = shape["difficult"]
                if isinstance(shape_value, bool):
                    if shape_value == filter_value:
                        return True
                elif isinstance(shape_value, str):
                    str_value = shape_value.lower() in ("true", "1", "yes")
                    if str_value == filter_value:
                        return True
                elif isinstance(shape_value, (int, float)):
                    bool_value = bool(shape_value)
                    if bool_value == filter_value:
                        return True
            return False

        if filter_type == "label":
            search_label = filter_value.lower()
            for shape in shapes:
                label = shape.get("label", "")
                if label and label.lower() == search_label:
                    return True
            return False

        if filter_type == "type":
            search_type = filter_value.lower()
            for shape in shapes:
                shape_type = shape.get("shape_type", "")
                if shape_type and shape_type.lower() == search_type:
                    return True
            return False

        if filter_type == "score":
            score_range = filter_value
            for shape in shapes:
                score = shape.get("score", 0.0)
                if not isinstance(score, (int, float)):
                    continue
                score_float = float(score)
                min_val = score_range["min"]
                max_val = score_range["max"]
                min_inclusive = score_range["min_inclusive"]
                max_inclusive = score_range["max_inclusive"]

                min_match = (
                    score_float >= min_val
                    if min_inclusive
                    else score_float > min_val
                )
                max_match = (
                    score_float <= max_val
                    if max_inclusive
                    else score_float < max_val
                )
                if min_match and max_match:
                    return True
            return False

        if filter_type == "description":
            is_nonempty = filter_value
            for shape in shapes:
                description = shape.get("description", "")
                has_description = bool(description and description.strip())
                if has_description == is_nonempty:
                    return True
            return False

        return False
    except (json.JSONDecodeError, IOError, KeyError):
        return False


def filter_image_files(
    image_files: list,
    search_pattern: SearchPattern,
    output_dir: Optional[str] = None,
) -> list:
    """
    Filter image files based on search pattern.

    Args:
        image_files: List of image file paths.
        search_pattern: The parsed search pattern.
        output_dir: Optional output directory for label files.

    Returns:
        Filtered list of image files.
    """
    filtered_files = []

    for filename in image_files:
        if not matches_filename(filename, search_pattern):
            continue

        if search_pattern.mode == "attribute":
            label_file = osp.splitext(filename)[0] + ".json"
            if output_dir:
                label_file_without_path = osp.basename(label_file)
                label_file = osp.join(output_dir, label_file_without_path)

            if not matches_label_attribute(
                filename, label_file, search_pattern
            ):
                continue

        filtered_files.append(filename)

    return filtered_files
