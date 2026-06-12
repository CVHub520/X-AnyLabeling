import os
import json
import logging
import pathlib
import subprocess
import sys
from typing import Literal, Optional

logger = logging.getLogger(__name__)

DeviceType = Literal["CPU", "GPU", "AUTO"]
_PROVIDER_PROBE_TIMEOUT = 5


def _build_probe_command() -> list[str]:
    """Build the hidden CLI command used for safe device probing."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "probe-device"]
    app_path = pathlib.Path(__file__).resolve().parents[2] / "app.py"
    return [sys.executable, str(app_path), "probe-device"]


def _subprocess_kwargs() -> dict:
    """Extra kwargs for subprocess calls to avoid console windows."""
    kwargs = {}
    if sys.platform.startswith("win") and hasattr(
        subprocess, "CREATE_NO_WINDOW"
    ):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def _probe_onnx_providers(timeout: int = _PROVIDER_PROBE_TIMEOUT) -> list[str]:
    """Safely probe available ONNX Runtime providers in a child process."""
    command = _build_probe_command()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            **_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "ONNX provider probe timed out. Falling back to CPU mode."
        )
        return []
    except OSError as e:
        logger.warning(
            "ONNX provider probe failed to start: %s. Falling back to CPU mode.",
            e,
        )
        return []

    if completed.returncode != 0:
        logger.warning(
            "ONNX provider probe crashed with exit code %s. "
            "Falling back to CPU mode.",
            completed.returncode,
        )
        return []

    output = completed.stdout.strip()
    if not output:
        logger.debug("ONNX provider probe exited cleanly without payload.")
        return []

    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        logger.warning(
            "ONNX provider probe returned invalid JSON. Falling back to CPU mode."
        )
        return []

    providers = payload.get("providers", [])
    return providers if isinstance(providers, list) else []


class DeviceManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._preferred_device = None
        self._available_providers = None

    def get_device(self) -> str:
        if self._preferred_device:
            return self._preferred_device

        env_device = os.getenv("X_ANYLABELING_DEVICE", "").upper()
        if env_device in ["CPU", "GPU"]:
            logger.info(f"Using device from environment: {env_device}")
            return self._validate_and_set(env_device)

        config_device = self._load_from_config()
        if config_device:
            logger.info(f"Using device from config: {config_device}")
            return self._validate_and_set(config_device)

        return self._auto_detect()

    def _validate_and_set(self, device: str) -> str:
        if device == "GPU" and not self._is_gpu_available():
            logger.warning(
                "GPU requested but not available. Falling back to CPU."
            )
            device = "CPU"

        self._preferred_device = device
        return device

    def _auto_detect(self) -> str:
        if self._is_gpu_available():
            logger.info("GPU detected and available, using GPU")
            self._preferred_device = "GPU"
            return "GPU"

        logger.info("GPU not available, using CPU")
        self._preferred_device = "CPU"
        return "CPU"

    def _is_gpu_available(self) -> bool:
        if self._available_providers is None:
            try:
                self._available_providers = _probe_onnx_providers()
            except Exception as e:
                logger.debug(f"Failed to get ONNX providers: {e}")
                self._available_providers = []

        return "CUDAExecutionProvider" in self._available_providers

    def _load_from_config(self) -> Optional[str]:
        try:
            import yaml
            from anylabeling.config import get_work_directory

            config_path = os.path.join(get_work_directory(), ".xanylabelingrc")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    if config:
                        device = str(config.get("device", "")).upper()
                        if device in ["CPU", "GPU"]:
                            return device
        except Exception as e:
            logger.debug(f"Failed to load config: {e}")

        return None

    def set_device(self, device: str):
        device = device.upper()
        if device not in ["CPU", "GPU"]:
            raise ValueError(f"Invalid device: {device}")

        self._preferred_device = device
        logger.info(f"Device manually set to: {device}")

    def reset_device_preference(self):
        self._preferred_device = None
        logger.info("Device preference reset to auto")

    def get_available_devices(self) -> list:
        devices = ["CPU"]
        if self._is_gpu_available():
            devices.append("GPU")
        return devices


device_manager = DeviceManager()


def get_preferred_device() -> str:
    return device_manager.get_device()
