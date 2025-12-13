import os
import pathlib
import yaml
import urllib.request
import time
import multiprocessing
from urllib.parse import urlparse
from urllib.error import URLError

import ssl

ssl._create_default_https_context = (
    ssl._create_unverified_context
)  # Prevent issue when downloading models behind a proxy

import socket

socket.setdefaulttimeout(240)  # Prevent timeout when downloading models

from abc import abstractmethod

from PyQt5.QtCore import QCoreApplication, QFile, QObject
from PyQt5.QtGui import QImage

from .types import AutoLabelingResult
from anylabeling.config import get_config
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.label_file import LabelFile, LabelFileError


def _check_model_worker(model_path):
    """Worker function to validate model in subprocess."""
    try:
        file_extension = os.path.splitext(model_path)[1].lower()
        if file_extension == ".onnx":
            import onnx

            onnx.checker.check_model(model_path)
        elif file_extension in [".pth", ".pt"]:
            import torch

            torch.load(model_path, map_location="cpu")
        else:
            raise ValueError(f"Unsupported model format: {file_extension}")
    except Exception as e:
        import sys

        print(f"Model check failed: {e}", file=sys.stderr)
        sys.exit(1)


def safe_check_model(model_path, timeout=60):
    """Safely check model integrity in subprocess to prevent crashes."""
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_check_model_worker, args=(model_path,))
    p.start()
    p.join(timeout)

    if p.exitcode == 0:
        return True
    elif p.exitcode is None:
        logger.warning(
            f"Model check timeout after {timeout}s for {model_path}"
        )
        p.terminate()
        p.join(1)
        if p.is_alive():
            p.kill()
            p.join()
        return False
    else:
        logger.warning(
            f"Model check failed with exit code {p.exitcode} for {model_path}"
        )
        return False


class Model(QObject):
    BASE_DOWNLOAD_URL = (
        "https://github.com/CVHub520/X-AnyLabeling/releases/tag"
    )

    # Add retry settings
    MAX_RETRIES = 2
    RETRY_DELAY = 3  # seconds

    class Meta(QObject):
        required_config_names = []
        widgets = ["button_run"]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__()
        self.on_message = on_message
        # Load and check config
        if isinstance(model_config, str):
            if not os.path.isfile(model_config):
                raise FileNotFoundError(
                    QCoreApplication.translate(
                        "Model", "Config file not found: {model_config}"
                    ).format(model_config=model_config)
                )
            with open(model_config, "r") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(model_config, dict):
            self.config = model_config
        else:
            raise ValueError(
                QCoreApplication.translate(
                    "Model", "Unknown config type: {type}"
                ).format(type=type(model_config))
            )
        self.check_missing_config(
            config_names=self.Meta.required_config_names,
            config=self.config,
        )
        self.output_mode = self.Meta.default_output_mode
        self._config = get_config()

    def get_required_widgets(self):
        """
        Get required widgets for showing in UI
        """
        return self.Meta.widgets

    @staticmethod
    def allow_migrate_data():
        """Check if the current env have write permissions"""
        home_dir = os.path.expanduser("~")
        old_model_path = os.path.join(home_dir, "anylabeling_data")
        new_model_path = os.path.join(home_dir, "xanylabeling_data")

        if os.path.exists(new_model_path) or not os.path.exists(
            old_model_path
        ):
            return True

        if not os.access(home_dir, os.W_OK):
            return False

        try:
            os.rename(old_model_path, new_model_path)
            return True
        except Exception as e:
            logger.error(f"An error occurred during data migration: {str(e)}")
            return False

    def download_with_retry(self, url, dest_path, progress_callback):
        """Download file with retry mechanism"""
        for attempt in range(self.MAX_RETRIES):
            try:
                if attempt > 0:
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.MAX_RETRIES}"
                    )
                urllib.request.urlretrieve(url, dest_path, progress_callback)
                return True
            except URLError as e:
                delay = self.RETRY_DELAY * (attempt + 1)
                if attempt < self.MAX_RETRIES - 1:
                    error_msg = f"Connection failed, retrying in {delay}s... (Attempt {attempt + 1}/{self.MAX_RETRIES} failed)"
                    logger.warning(error_msg)
                    self.on_message(error_msg)
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"All download attempts failed ({self.MAX_RETRIES} tries)"
                    )
                    raise e

    def get_model_abs_path(self, model_config, model_path_field_name):
        """
        Get model absolute path from config path or download from url
        """
        # Try getting model path from config folder
        model_path = model_config[model_path_field_name]

        # Model path is a local path
        if not model_path.startswith(("http://", "https://")):
            # Relative path to executable or absolute path?
            model_abs_path = os.path.abspath(model_path)
            if os.path.exists(model_abs_path):
                return model_abs_path

            # Relative path to config file?
            config_file_path = model_config["config_file"]
            config_folder = os.path.dirname(config_file_path)
            model_abs_path = os.path.abspath(
                os.path.join(config_folder, model_path)
            )
            if os.path.exists(model_abs_path):
                return model_abs_path

            raise QCoreApplication.translate(
                "Model", "Model path not found: {model_path}"
            ).format(model_path=model_path)

        # Download model from url
        self.on_message(
            QCoreApplication.translate(
                "Model", "Downloading model from registry..."
            )
        )

        # Build download url
        def get_filename_from_url(url):
            a = urlparse(url)
            return os.path.basename(a.path)

        filename = get_filename_from_url(model_path)
        download_url = model_path

        # Continue with the rest of your function logic
        migrate_flag = self.allow_migrate_data()
        home_dir = os.path.expanduser("~")
        data_dir = "xanylabeling_data" if migrate_flag else "anylabeling_data"

        # Create model folder
        home_dir = os.path.expanduser("~")
        model_path = os.path.abspath(os.path.join(home_dir, data_dir))
        model_abs_path = os.path.abspath(
            os.path.join(
                model_path,
                "models",
                model_config["name"],
                filename,
            )
        )
        if os.path.exists(model_abs_path):
            file_extension = os.path.splitext(model_abs_path)[1].lower()
            is_known_type = file_extension in (".onnx", ".pth", ".pt")
            is_valid = False
            # file_not_empty = os.path.getsize(model_abs_path) > 0

            if is_known_type:
                logger.info(f"Validating model integrity: {filename}")
                is_valid = safe_check_model(model_abs_path)
            elif os.path.getsize(model_abs_path) > 0:
                logger.info(
                    f"Model file exists and is not empty: {model_abs_path}"
                )
                is_valid = True

            if is_valid:
                logger.info(f"Model file is valid: {model_abs_path}")
                return model_abs_path
            else:
                logger.warning(
                    f"Model validation failed or file is empty: {model_abs_path}. Deleting and redownloading..."
                )
                try:
                    os.remove(model_abs_path)
                    logger.info(
                        f"Model file {model_abs_path} deleted successfully"
                    )
                except Exception as e2:  # noqa
                    logger.error(f"Could not delete corrupted file: {str(e2)}")
        pathlib.Path(model_abs_path).parent.mkdir(parents=True, exist_ok=True)

        # Download url
        use_modelscope = False
        env_model_hub = os.getenv("XANYLABELING_MODEL_HUB")
        if env_model_hub == "modelscope":
            use_modelscope = True
        elif (
            env_model_hub is None or env_model_hub == ""
        ):  # Only check config if env var is not set or empty
            if self._config.get("model_hub") == "modelscope":
                use_modelscope = True
            # Fallback to language check only if model_hub is not 'modelscope'
            elif (
                self._config.get("model_hub") is None
                or self._config.get("model_hub") == ""
            ):
                if self._config.get("language") == "zh_CN":
                    use_modelscope = True

        if use_modelscope:
            model_type = model_config["name"].split("-")[0]
            model_name = os.path.basename(download_url)
            download_url = f"https://www.modelscope.cn/models/CVHub520/{model_type}/resolve/master/{model_name}"

        ellipsis_download_url = download_url
        if len(download_url) > 40:
            ellipsis_download_url = (
                download_url[:20] + "..." + download_url[-20:]
            )

        logger.info(f"Downloading {download_url} to {model_abs_path}")
        try:

            def _progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                self.on_message(
                    QCoreApplication.translate(
                        "Model", "Downloading {download_url}: {percent}%"
                    ).format(
                        download_url=ellipsis_download_url, percent=percent
                    )
                )

            self.download_with_retry(download_url, model_abs_path, _progress)

        except Exception as e:  # noqa
            logger.error(
                f"Could not download {download_url}: {e}, you can try to download it manually."
            )
            self.on_message(f"Download failed! Please try again later.")
            time.sleep(1)
            return None

        return model_abs_path

    def check_missing_config(self, config_names, config):
        """
        Check if config has all required config names
        """
        for name in config_names:
            if name not in config:
                raise Exception(f"Missing config: {name}")

    @abstractmethod
    def predict_shapes(self, image, filename=None) -> AutoLabelingResult:
        """
        Predict image and return AnyLabeling shapes
        """
        raise NotImplementedError

    @abstractmethod
    def unload(self):
        """
        Unload memory
        """
        raise NotImplementedError

    @staticmethod
    def load_image_from_filename(filename):
        """Load image from labeling file and return image data and image path."""
        label_file = os.path.splitext(filename)[0] + ".json"
        if QFile.exists(label_file) and LabelFile.is_label_file(label_file):
            try:
                label_file = LabelFile(label_file)
            except LabelFileError as e:
                logger.error("Error reading {}: {}".format(label_file, e))
                return None, None
            image_data = label_file.image_data
        else:
            image_data = LabelFile.load_image_file(filename)
        image = QImage.fromData(image_data)
        if image.isNull():
            logger.error("Error reading {}".format(filename))
        return image

    def on_next_files_changed(self, next_files):
        """
        Handle next files changed. This function can preload next files
        and run inference to save time for user.
        """
        pass

    def set_output_mode(self, mode):
        """
        Set output mode
        """
        self.output_mode = mode
