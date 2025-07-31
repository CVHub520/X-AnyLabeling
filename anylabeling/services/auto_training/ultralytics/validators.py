import os
from typing import Dict, Tuple


def validate_basic_config(config: Dict) -> Tuple[bool, str]:
    """Validate basic training configuration

    Args:
        config: Training configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    basic = config.get("basic", {})

    if not basic.get("project", "").strip():
        return False, "Project field is required"

    if not basic.get("name", "").strip():
        return False, "Name field is required"

    save_dir = os.path.join(basic["project"], basic["name"])
    if os.path.exists(save_dir):
        return False, f"Project directory already exists: {save_dir}"

    model_path = basic.get("model", "").strip()
    if not model_path or not os.path.exists(model_path):
        return False, "Valid model file is required"

    data_path = basic.get("data", "").strip()
    if not data_path or not os.path.exists(data_path):
        return False, "Valid data file is required"

    return True, ""
