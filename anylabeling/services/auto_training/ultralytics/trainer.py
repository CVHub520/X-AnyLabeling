import os
import signal
import shutil
import subprocess
import time
import threading
from io import StringIO
from typing import Dict, Tuple

from PyQt5.QtCore import QObject, pyqtSignal

from .config import SETTINGS_CONFIG_PATH


class TrainingEventRedirector(QObject):
    """Thread-safe training event redirector"""

    training_event_signal = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()

    def emit_training_event(self, event_type, data):
        """Safely emit training events from child thread to main thread"""
        self.training_event_signal.emit(event_type, data)


class TrainingLogRedirector(QObject):
    """Thread-safe training log redirector"""

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


class TrainingManager:
    def __init__(self):
        self.training_process = None
        self.is_training = False
        self.callbacks = []
        self.total_epochs = 100
        self.stop_event = threading.Event()

    def notify_callbacks(self, event_type: str, data: dict):
        for callback in self.callbacks:
            try:
                callback(event_type, data)
            except Exception:
                pass

    def start_training(self, train_args: Dict) -> Tuple[bool, str]:
        if self.is_training:
            return False, "Training is already in progress"

        try:
            import sys

            self.total_epochs = train_args.get("epochs", 100)
            self.stop_event.clear()

            script_content = f"""# -*- coding: utf-8 -*-
import io
import os
import signal
import sys
import multiprocessing

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')

from ultralytics import YOLO

def signal_handler(signum, frame):
    print("Training interrupted by signal", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        multiprocessing.set_start_method('spawn', force=True)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        model = YOLO({repr(train_args.pop("model"))})
        train_args = {repr(train_args)}
        train_args['verbose'] = False
        train_args['show'] = False
        results = model.train(**train_args)
    except KeyboardInterrupt:
        print("Training interrupted by user", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"Training error: {{e}}", flush=True)
        sys.exit(1)
"""

            script_path = os.path.join(
                train_args.get("project", "/tmp"), "train_script.py"
            )
            os.makedirs(os.path.dirname(script_path), exist_ok=True)

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            def run_training():
                try:
                    self.is_training = True
                    self.notify_callbacks(
                        "training_started", {"total_epochs": self.total_epochs}
                    )

                    self.training_process = subprocess.Popen(
                        [sys.executable, script_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        bufsize=1,
                        preexec_fn=os.setsid if os.name != "nt" else None,
                    )

                    while True:
                        if self.stop_event.is_set():
                            self.training_process.terminate()
                            try:
                                self.training_process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                if os.name == "nt":
                                    self.training_process.kill()
                                else:
                                    os.killpg(
                                        os.getpgid(self.training_process.pid),
                                        signal.SIGKILL,
                                    )
                            self.is_training = False
                            self.notify_callbacks("training_stopped", {})
                            return

                        output = self.training_process.stdout.readline()
                        if (
                            output == ""
                            and self.training_process.poll() is not None
                        ):
                            break
                        if output:
                            cleaned_output = output.strip()
                            if cleaned_output:
                                self.notify_callbacks(
                                    "training_log", {"message": cleaned_output}
                                )

                    return_code = self.training_process.poll()
                    self.is_training = False

                    if return_code == 0:
                        self.notify_callbacks(
                            "training_completed",
                            {"results": "Training completed successfully"},
                        )
                    else:
                        self.notify_callbacks(
                            "training_error",
                            {
                                "error": f"Training process exited with code {return_code}"
                            },
                        )

                except Exception as e:
                    self.is_training = False
                    self.notify_callbacks("training_error", {"error": str(e)})
                finally:
                    try:
                        os.remove(script_path)
                    except:
                        pass

            def save_settings_config():
                save_path = os.path.join(
                    train_args["project"], train_args["name"]
                )
                save_file = os.path.join(save_path, "settings.json")

                while not os.path.exists(save_path):
                    time.sleep(1)

                shutil.copy2(SETTINGS_CONFIG_PATH, save_file)

            training_thread = threading.Thread(target=run_training)
            training_thread.daemon = True
            training_thread.start()

            config_thread = threading.Thread(target=save_settings_config)
            config_thread.daemon = True
            config_thread.start()

            return True, "Training started successfully"

        except ImportError:
            return (
                False,
                "Ultralytics is not installed. Please install it with: pip install ultralytics",
            )
        except Exception as e:
            return False, f"Failed to start training: {str(e)}"

    def stop_training(self) -> bool:
        if not self.is_training:
            return False

        try:
            self.stop_event.set()
            return True
        except Exception:
            return False


_training_manager = TrainingManager()


def get_training_manager() -> TrainingManager:
    return _training_manager
