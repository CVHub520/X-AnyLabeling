import json
import os
import yaml
from typing import Dict, Any

from .config import SETTINGS_CONFIG_PATH


def ensure_config_dir():
    """
    Ensure the configuration directory exists, create it if it doesn't.
    """
    config_dir = os.path.dirname(SETTINGS_CONFIG_PATH)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)


def save_config(config: Dict[str, Any]) -> bool:
    """
    Save configuration dictionary to JSON file.

    Args:
        config: Dictionary containing configuration data

    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        ensure_config_dir()
        with open(SETTINGS_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False


def save_yaml_config(config: Dict[str, Any], file_path: str) -> bool:
    """
    Save configuration dictionary to YAML file.

    Args:
        config: Dictionary containing configuration data
        file_path: Path to save the YAML file

    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception:
        return False


def load_config() -> Dict[str, Any]:
    """
    Load configuration from default JSON config file.

    Returns:
        Dict: Configuration dictionary if successful, empty dict otherwise
    """
    try:
        if os.path.exists(SETTINGS_CONFIG_PATH):
            with open(SETTINGS_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from specified JSON file.

    Args:
        file_path: Path to the JSON config file

    Returns:
        Dict: Configuration dictionary if successful, None otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        file_path: Path to the YAML config file

    Returns:
        Dict: Configuration dictionary if successful, None otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None
