import os
import signal
import shutil
import subprocess
import time
import threading
import sys
import multiprocessing
from multiprocessing import Process, Queue
from io import StringIO
from typing import Dict, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

from .config import get_settings_config_path


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

                shutil.copy2(get_settings_config_path(), save_file)

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

    def start_training_mp(self, train_args: Dict) -> Tuple[bool, str]:
        """replaced python train_script.py with multiprocessing.process"""
        if self.is_training:
            return False, "Training is already in progress"

        try:
            import sys
            self.total_epochs = train_args.get("epochs", 100)
            self.stop_event.clear()

            def run_training():
                try:
                    self.is_training = True
                    self.notify_callbacks(
                        "training_started", {"total_epochs": self.total_epochs}
                    )

                    logstr_queue = Queue()
                    train_is_finished = multiprocessing.Event()
                    train_is_failed = multiprocessing.Event()
                    self.training_process = Process(
                        name="train_process",
                        target=train_process_func, 
                        args=(logstr_queue, train_args, train_is_finished, train_is_failed))
                    self.training_process.start()

                    while True:
                        if self.stop_event.is_set():
                            self.training_process.terminate()
                            self.training_process.join(timeout=5)
                            if self.training_process.is_alive():
                                self.training_process.kill()
                            self.notify_callbacks("training_stopped", {})
                            self.is_training = False
                            return

                        if train_is_finished.is_set() and logstr_queue.empty():
                            self.notify_callbacks("training_completed",{"results": "Training completed successfully"})
                            self.is_training = False
                            break
                        
                        if train_is_failed.is_set() and logstr_queue.empty():
                            self.notify_callbacks("training_error", {"error": str(e)})
                            self.stop_training()
                            return

                        if not logstr_queue.empty():
                            output = logstr_queue.get_nowait()
                            if output:
                                cleaned_output = output.strip()
                                if cleaned_output:
                                    self.notify_callbacks("training_log", {"message": cleaned_output})

                    self.training_process.join()
                    if self.training_process.is_alive():
                        self.training_process.kill()
                    logstr_queue.close()
                    
                    self.is_training = False
                    self.notify_callbacks(
                            "training_completed",
                            {"results": "Training completed successfully"},
                        )

                except Exception as e:
                    self.is_training = False
                    self.notify_callbacks("training_error", {"error": str(e)})
                finally:
                    pass

            def save_settings_config():
                save_path = os.path.join(
                    train_args["project"], train_args["name"]
                )
                save_file = os.path.join(save_path, "settings.json")

                while not os.path.exists(save_path):
                    time.sleep(1)

                shutil.copy2(get_settings_config_path(), save_file)

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

def train_process_func(str_queue, train_args, train_is_finished, train_is_failed):
    "contents of train_script.py"

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    class Logger:
        def write(self, data):
            str_queue.put(data)
        def flush(self):
            pass

    # must set stdout and stderr before import ultralytics
    sys.stdout = Logger()
    sys.stderr = Logger()
    
    from ultralytics import YOLO
    
    try:
        task =  str(train_args['model']) 
        model = YOLO(train_args['model'])

        train_args['verbose'] = False
        train_args['show'] = False

        results = model.train(**train_args)
    except Exception as e:
        train_is_failed.set()
        return
        
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    train_is_finished.set()


_training_manager = TrainingManager()


def get_training_manager() -> TrainingManager:
    return _training_manager
