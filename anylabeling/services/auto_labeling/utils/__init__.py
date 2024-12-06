from .box import *
from .general import *
from .points_conversion import *

import queue
import threading
import time


class TimeoutContext:
    """A context manager for handling timeout operations"""

    def __init__(self, timeout=300, timeout_message=None):
        self.timeout = timeout
        self.timeout_message = timeout_message
        self.thread = None
        self.result_queue = None
        self.stop_event = threading.Event()

    def __enter__(self):
        import queue

        self.result_queue = queue.Queue()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(0)
            if self.timeout_message:
                raise TimeoutError(self.timeout_message)
            else:
                raise TimeoutError(
                    f"Operation timeout after {self.timeout} seconds"
                )
        return False

    def run(self, target_func, *args, **kwargs):
        def wrapper():
            try:
                while not self.stop_event.is_set():
                    result = target_func(*args, **kwargs)
                    self.result_queue.put(("success", result))
                    return
            except Exception as e:
                self.result_queue.put(("error", e))

        start_time = time.time()
        self.thread = threading.Thread(target=wrapper)
        self.thread.daemon = True
        self.thread.start()

        while time.time() - start_time < self.timeout:
            self.thread.join(0.1)
            try:
                status, result = self.result_queue.get_nowait()
                if status == "error":
                    raise result
                return result
            except queue.Empty:
                continue

        if self.timeout_message:
            raise TimeoutError(self.timeout_message)
        else:
            raise TimeoutError(
                f"Operation timeout after {self.timeout} seconds"
            )
