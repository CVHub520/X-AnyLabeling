import ast
import os
import re
import subprocess
import sys
import threading
from io import StringIO
from typing import Tuple

from PyQt6.QtCore import QObject, pyqtSignal


from .utils import check_package_installed
from .validators import install_packages_with_timeout
from ._io import load_yaml_config, save_yaml_config
from .config import get_trainer_root_dir
from .environment import (
    build_worker_command,
    build_worker_environment,
    create_worker_payload,
    get_worker_creation_flags,
    parse_worker_output,
    resolve_training_python,
    terminate_process_tree,
)

_MODEL_FAMILIES = (
    ("yolo26", "yolo26"),
    ("yolo12", "yolo12"),
    ("yolo11", "yolo11"),
    ("yolov10", "yolov10"),
    ("yolov9", "yolov9"),
    ("yolov8", "yolov8"),
    ("yolov7", "yolov7"),
    ("yolov6", "yolov6"),
    ("yolov5", "yolov5"),
)
_TASK_SUFFIXES = {
    "detect": "",
    "segment": "_seg",
    "pose": "_pose",
    "obb": "_obb",
    "classify": "_cls",
}


def _load_onnx_metadata(model_path):
    try:
        import onnx

        model = onnx.load(model_path, load_external_data=False)
        return {item.key: item.value for item in model.metadata_props}
    except Exception:
        return {}


def _ordered_names(names):
    if isinstance(names, list):
        return [str(name) for name in names]
    if not isinstance(names, dict) or not names:
        return []

    def sort_key(value):
        try:
            return 0, int(value)
        except (TypeError, ValueError):
            return 1, str(value)

    return [str(names[key]) for key in sorted(names, key=sort_key)]


def _resolve_model_type(model_hint, task):
    family = next(
        (
            model_type
            for marker, model_type in _MODEL_FAMILIES
            if marker in model_hint.lower()
        ),
        None,
    )
    if family is None:
        raise ValueError(
            "Unable to determine the Ultralytics model family from the "
            "training experiment"
        )

    suffix = _TASK_SUFFIXES.get(task)
    if suffix is None:
        raise ValueError(f"Unsupported training task: {task}")
    model_type = f"{family}{suffix}"

    from anylabeling.services.auto_labeling import _CUSTOM_MODELS

    if model_type not in _CUSTOM_MODELS:
        raise ValueError(
            f"X-AnyLabeling does not support loading {family} {task} models"
        )
    return model_type


def create_auto_labeling_config(
    project_path, exported_path, pose_config_path=None
):
    project_path = os.path.abspath(project_path)
    exported_path = os.path.abspath(exported_path)
    if not os.path.isfile(exported_path):
        raise ValueError(f"Exported ONNX model not found: {exported_path}")

    training_args = load_yaml_config(os.path.join(project_path, "args.yaml"))
    if not isinstance(training_args, dict):
        training_args = {}
    metadata = _load_onnx_metadata(exported_path)
    task = str(metadata.get("task") or training_args.get("task") or "").lower()
    model_hint = " ".join(
        (
            str(training_args.get("model", "")),
            str(metadata.get("description", "")),
        )
    )
    model_type = _resolve_model_type(model_hint, task)

    metadata_names = metadata.get("names")
    try:
        names = ast.literal_eval(metadata_names) if metadata_names else None
    except (SyntaxError, ValueError):
        names = None
    if not names:
        data_path = str(training_args.get("data", ""))
        if os.path.isdir(data_path):
            classes_path = os.path.join(data_path, "train")
            classes_path = (
                classes_path if os.path.isdir(classes_path) else data_path
            )
            names = sorted(
                entry.name
                for entry in os.scandir(classes_path)
                if entry.is_dir()
            )
        else:
            data_config = load_yaml_config(data_path)
            names = (
                data_config.get("names")
                if isinstance(data_config, dict)
                else None
            )

    experiment_name = os.path.basename(project_path)
    config_name = re.sub(r"[^A-Za-z0-9._-]+", "_", experiment_name).strip("._")
    config = {
        "type": model_type,
        "name": config_name or "trained-model",
        "provider": "Ultralytics",
        "display_name": experiment_name,
        "model_path": os.path.relpath(exported_path, project_path).replace(
            os.sep, "/"
        ),
    }

    if task == "pose":
        pose_config = load_yaml_config(pose_config_path or "")
        pose_classes = (
            pose_config.get("classes")
            if isinstance(pose_config, dict)
            else None
        )
        if not isinstance(pose_classes, dict) or not pose_classes:
            raise ValueError(
                "A valid pose configuration is required to load this model"
            )
        config.update(
            {
                "conf_threshold": 0.5,
                "iou_threshold": 0.6,
                "kpt_threshold": 0.25,
                "has_visible": bool(pose_config.get("has_visible", True)),
                "classes": pose_classes,
            }
        )
    elif task == "classify":
        class_names = _ordered_names(names)
        if not class_names:
            raise ValueError("Unable to determine the model classes")
        config["classes"] = dict(enumerate(class_names))
    else:
        class_names = _ordered_names(names)
        if not class_names:
            raise ValueError("Unable to determine the model classes")
        config.update(
            {
                "conf_threshold": 0.25,
                "iou_threshold": 0.45,
                "max_det": 300,
                "classes": class_names,
            }
        )

    config_path = os.path.join(project_path, f"{experiment_name}.yaml")
    if not save_yaml_config(config, config_path):
        raise OSError(f"Failed to save model configuration: {config_path}")
    return config_path


class ExportEventRedirector(QObject):
    """Thread-safe export event redirector"""

    export_event_signal = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()

    def emit_export_event(self, event_type, data):
        """Safely emit export events from child thread to main thread"""
        self.export_event_signal.emit(event_type, data)


class ExportLogRedirector(QObject):
    """Thread-safe export log redirector"""

    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.log_stream = StringIO()

    def write(self, text):
        """Write text to log stream and emit signal if not empty"""
        if text.strip():
            self.log_signal.emit(text)

    def flush(self):
        """Flush the log stream"""
        pass


def validate_onnx_export_environment():
    required_packages = ["onnx", "onnxslim", "onnxruntime"]
    missing_packages = []
    package_mapping = {
        "onnx": "onnx>=1.15.0",
        "onnxslim": "onnxslim>=0.1.59",
        "onnxruntime": "onnxruntime",
    }

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    try:
        import onnx

        if hasattr(onnx, "__version__"):
            onnx_version = onnx.__version__
            from packaging import version

            if version.parse(onnx_version) < version.parse("1.15.0"):
                missing_packages.append("onnx>=1.15.0")
    except:
        pass

    return missing_packages


def validate_openvino_export_environment():
    required_packages = ["openvino"]
    missing_packages = []
    package_mapping = {"openvino": "openvino>=2024.0.0"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_tensorrt_export_environment():
    required_packages = ["tensorrt"]
    missing_packages = []
    package_mapping = {"tensorrt": "tensorrt>7.0.0,!=10.1.0"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_coreml_export_environment():
    required_packages = ["coremltools"]
    missing_packages = []
    package_mapping = {"coremltools": "coremltools>=8.0"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_tensorflow_export_environment():
    required_packages = ["tensorflow"]
    missing_packages = []
    package_mapping = {"tensorflow": "tensorflow>=2.0.0"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_paddle_export_environment():
    required_packages = ["paddlepaddle", "x2paddle"]
    missing_packages = []
    package_mapping = {
        "paddlepaddle": "paddlepaddle-gpu",
        "x2paddle": "x2paddle",
    }

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_mnn_export_environment():
    required_packages = ["MNN"]
    missing_packages = []
    package_mapping = {"MNN": "MNN>=2.9.6"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_ncnn_export_environment():
    required_packages = ["ncnn"]
    missing_packages = []
    package_mapping = {"ncnn": "ncnn"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_imx500_export_environment():
    required_packages = ["imx500-converter", "mct-quantizers"]
    missing_packages = []
    package_mapping = {
        "imx500-converter": "imx500-converter[pt]>=3.16.1",
        "mct-quantizers": "mct-quantizers>=1.6.0",
    }

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def validate_rknn_export_environment():
    required_packages = ["rknn-toolkit2"]
    missing_packages = []
    package_mapping = {"rknn-toolkit2": "rknn-toolkit2"}

    for package in required_packages:
        if not check_package_installed(package):
            missing_packages.append(package_mapping[package])

    return missing_packages


def get_export_validator(export_format):
    validators = {
        "onnx": validate_onnx_export_environment,
        "openvino": validate_openvino_export_environment,
        "engine": validate_tensorrt_export_environment,
        "coreml": validate_coreml_export_environment,
        "saved_model": validate_tensorflow_export_environment,
        "pb": validate_tensorflow_export_environment,
        "tflite": validate_tensorflow_export_environment,
        "edgetpu": validate_tensorflow_export_environment,
        "tfjs": validate_tensorflow_export_environment,
        "paddle": validate_paddle_export_environment,
        "mnn": validate_mnn_export_environment,
        "ncnn": validate_ncnn_export_environment,
        "imx": validate_imx500_export_environment,
        "rknn": validate_rknn_export_environment,
        "torchscript": lambda: [],
    }
    return validators.get(export_format, lambda: [])


class ExportManager:
    def __init__(self):
        self.is_exporting = False
        self.callbacks = []
        self.export_thread = None
        self.export_process = None
        self.python_executable = None
        self.auto_install_packages = True
        self.external_environment = False
        self.stop_event = threading.Event()

    def configure_environment(
        self,
        python_executable=None,
        auto_install_packages=True,
    ):
        self.python_executable = python_executable
        self.auto_install_packages = bool(auto_install_packages)
        self.external_environment = bool(python_executable)

    def notify_callbacks(self, event_type: str, data: dict):
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                print(f"Error in export callback: {e}")

    def start_export(
        self, project_path: str, export_format: str = "onnx"
    ) -> Tuple[bool, str]:
        if self.is_exporting:
            return False, "Export already in progress"

        weights_path = os.path.join(project_path, "weights", "best.pt")
        if not self.external_environment and not os.path.exists(weights_path):
            return False, f"Model weights not found at: {weights_path}"

        self.is_exporting = True
        self.stop_event.clear()
        target = (
            self._external_export_worker
            if self.external_environment
            else self._export_worker
        )
        self.export_thread = threading.Thread(
            target=target, args=(weights_path, export_format), daemon=True
        )
        self.export_thread.start()
        return True, "Export started successfully"

    def _external_export_worker(self, weights_path: str, export_format: str):
        payload_path = create_worker_payload(
            "export",
            args={"format": export_format},
            paths={
                "data_directory": get_trainer_root_dir(),
                "weights_path": weights_path,
            },
            options={"auto_install_packages": self.auto_install_packages},
        )
        try:
            if self.stop_event.is_set():
                return
            self.export_process = subprocess.Popen(
                build_worker_command(
                    resolve_training_python(self.python_executable),
                    payload_path,
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=build_worker_environment(self.auto_install_packages),
                start_new_session=os.name != "nt",
                creationflags=get_worker_creation_flags(),
            )
            terminal_event_seen = False
            if self.export_process.stdout is None:
                raise RuntimeError("Export process output is unavailable")
            for output in self.export_process.stdout:
                terminal_event_seen = (
                    parse_worker_output(
                        output,
                        self._notify_external_export_event,
                        plain_log_event="export_log",
                    )
                    or terminal_event_seen
                )
            return_code = self.export_process.wait()
            if not terminal_event_seen and not self.stop_event.is_set():
                error = (
                    f"Export process exited with code {return_code}"
                    if return_code
                    else "Export process returned no result"
                )
                self.notify_callbacks(
                    "export_error",
                    {"error": error},
                )
        except Exception as exc:
            if not self.stop_event.is_set():
                self.notify_callbacks("export_error", {"error": str(exc)})
        finally:
            self.is_exporting = False
            self.export_process = None
            try:
                os.remove(payload_path)
            except OSError:
                pass

    def _notify_external_export_event(
        self, event_type: str, data: dict
    ) -> None:
        if self.stop_event.is_set():
            return
        self.notify_callbacks(event_type, data)

    def _export_worker(self, weights_path: str, export_format: str):
        try:
            if self.stop_event.is_set():
                return
            self.notify_callbacks(
                "export_started",
                {"weights_path": weights_path, "format": export_format},
            )
            self.notify_callbacks(
                "export_log", {"message": "Checking export environment..."}
            )
            missing_packages = get_export_validator(export_format)()
            if missing_packages:
                self.notify_callbacks(
                    "export_log",
                    {
                        "message": f"Missing required packages: {', '.join(missing_packages)}"
                    },
                )
                if not self.auto_install_packages:
                    command = f'"{sys.executable}" -m pip install ' + " ".join(
                        missing_packages
                    )
                    self.notify_callbacks(
                        "export_error",
                        {
                            "error": (
                                "Missing required packages: "
                                f"{', '.join(missing_packages)}. "
                                f"Install them with: {command}"
                            )
                        },
                    )
                    return
                self.notify_callbacks(
                    "export_log",
                    {"message": "Attempting to install missing packages..."},
                )
                success, stdout, stderr = install_packages_with_timeout(
                    missing_packages, timeout=1800
                )
                if not success:
                    error_msg = f"Failed to install required packages: {', '.join(missing_packages)}. Please manually install these packages and restart the application."
                    self.notify_callbacks("export_error", {"error": error_msg})
                    return
                self.notify_callbacks(
                    "export_log",
                    {"message": "Required packages installed successfully"},
                )
            else:
                self.notify_callbacks(
                    "export_log",
                    {"message": "All required packages are available"},
                )

            original_stdout = sys.stdout
            original_stderr = sys.stderr

            log_redirector = ExportLogRedirector()
            sys.stdout = log_redirector
            sys.stderr = log_redirector

            try:
                from ultralytics import YOLO

                self.notify_callbacks(
                    "export_log",
                    {"message": f"Loading model from {weights_path}"},
                )
                model = YOLO(weights_path)

                self.notify_callbacks(
                    "export_log",
                    {
                        "message": f"Starting export to {export_format} format..."
                    },
                )
                results = model.export(format=export_format)

                exported_path = (
                    results if isinstance(results, str) else str(results)
                )
                if not exported_path:
                    weights_dir = os.path.dirname(weights_path)
                    model_name = os.path.splitext(
                        os.path.basename(weights_path)
                    )[0]
                    exported_path = os.path.join(
                        weights_dir, f"{model_name}.{export_format}"
                    )

                if os.path.exists(exported_path):
                    self.notify_callbacks(
                        "export_completed",
                        {
                            "exported_path": exported_path,
                            "format": export_format,
                        },
                    )
                else:
                    possible_path = weights_path.replace(
                        ".pt", f".{export_format}"
                    )
                    if os.path.exists(possible_path):
                        self.notify_callbacks(
                            "export_completed",
                            {
                                "exported_path": possible_path,
                                "format": export_format,
                            },
                        )
                    else:
                        self.notify_callbacks(
                            "export_error",
                            {
                                "error": "Export completed but output file not found"
                            },
                        )

            except ImportError as e:
                self.notify_callbacks(
                    "export_error",
                    {"error": f"Failed to import ultralytics: {str(e)}"},
                )
            except Exception as e:
                self.notify_callbacks(
                    "export_error", {"error": f"Export failed: {str(e)}"}
                )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

                if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
                    del os.environ["CUDA_VISIBLE_DEVICES"]

        except Exception as e:
            self.notify_callbacks(
                "export_error",
                {"error": f"Unexpected error during export: {str(e)}"},
            )
        finally:
            self.is_exporting = False

    def stop_export(self, wait: bool = False) -> bool:
        if not self.is_exporting:
            return False

        self.is_exporting = False
        self.stop_event.set()
        self.notify_callbacks("export_stopped", {})
        if wait:
            terminate_process_tree(self.export_process)
            return True
        threading.Thread(
            target=terminate_process_tree,
            args=(self.export_process,),
            daemon=True,
        ).start()
        return True


_export_manager = None


def get_export_manager() -> ExportManager:
    global _export_manager
    if _export_manager is None:
        _export_manager = ExportManager()
    return _export_manager


def export_model(
    project_path: str, export_format: str = "onnx"
) -> Tuple[bool, str]:
    manager = get_export_manager()
    return manager.start_export(project_path, export_format)
