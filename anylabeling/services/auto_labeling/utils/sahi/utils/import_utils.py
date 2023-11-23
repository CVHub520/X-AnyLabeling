import importlib.util
import logging
import os

# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)


def get_package_info(package_name: str, verbose: bool = True):
    """
    Returns the package version as a string and the package name as a string.
    """
    _is_available = is_available(package_name)

    if _is_available:
        try:
            import importlib.metadata as _importlib_metadata

            _version = _importlib_metadata.version(package_name)
        except (ModuleNotFoundError, AttributeError):
            try:
                _version = importlib.import_module(package_name).__version__
            except AttributeError:
                _version = "unknown"
        if verbose:
            logger.info(f"{package_name} version {_version} is available.")
    else:
        _version = "N/A"

    return _is_available, _version


def print_enviroment_info():
    _tensorflow_available, _tensorflow_version = get_package_info("tensorflow")
    _tensorflow_hub_available, _tensorflow_hub_version = get_package_info(
        "tensorflow-hub"
    )
    _yolov5_available, _yolov5_version = get_package_info("yolov5")
    _mmdet_available, _mmdet_version = get_package_info("mmdet")
    _mmcv_available, _mmcv_version = get_package_info("mmcv")
    _detectron2_available, _detectron2_version = get_package_info("detectron2")
    _transformers_available, _transformers_version = get_package_info(
        "transformers"
    )
    _timm_available, _timm_version = get_package_info("timm")
    _fiftyone_available, _fiftyone_version = get_package_info("fiftyone")


def is_available(module_name: str):
    return importlib.util.find_spec(module_name) is not None


def check_requirements(package_names):
    """
    Raise error if module is not installed.
    """
    missing_packages = []
    for package_name in package_names:
        if importlib.util.find_spec(package_name) is None:
            missing_packages.append(package_name)
    if missing_packages:
        raise ImportError(
            f"The following packages are required to use this module: {missing_packages}"
        )
    yield


def check_package_minimum_version(
    package_name: str, minimum_version: str, verbose=False
):
    """
    Raise error if module version is not compatible.
    """
    from packaging import version

    _is_available, _version = get_package_info(package_name, verbose=verbose)
    if _is_available:
        if _version == "unknown":
            logger.warning(
                f"Could not determine version of {package_name}. Assuming version {minimum_version} is compatible."
            )
        else:
            if version.parse(_version) < version.parse(minimum_version):
                return False
    return True


def ensure_package_minimum_version(
    package_name: str, minimum_version: str, verbose=False
):
    """
    Raise error if module version is not compatible.
    """
    from packaging import version

    _is_available, _version = get_package_info(package_name, verbose=verbose)
    if _is_available:
        if _version == "unknown":
            logger.warning(
                f"Could not determine version of {package_name}. Assuming version {minimum_version} is compatible."
            )
        else:
            if version.parse(_version) < version.parse(minimum_version):
                raise ImportError(
                    f"Please upgrade {package_name} to version {minimum_version} or higher to use this module."
                )
    yield
