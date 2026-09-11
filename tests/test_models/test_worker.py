import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from anylabeling.services.auto_labeling.worker import GenericWorker
from anylabeling.services.auto_labeling.model_manager import ModelManager


def _bare_manager(model):
    manager = ModelManager.__new__(ModelManager)
    super(ModelManager, manager).__init__()
    manager.loaded_model_config_lock = threading.RLock()
    manager.model_execution_thread_lock = threading.Lock()
    manager.model_execution_thread = None
    manager.model_execution_worker = None
    manager.loaded_model_config = {"model": model, "type": "test"}
    return manager


def test_worker_contains_exception_and_always_finishes():
    failures = []
    finished = []

    def fail():
        raise RuntimeError("model load failed")

    worker = GenericWorker(fail)
    worker.failed.connect(failures.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert finished == [True]


def test_model_session_is_not_unloaded_during_native_inference():
    entered = threading.Event()
    release = threading.Event()
    unloaded = threading.Event()

    class Model:
        def predict_shapes(self, _image, _filename):
            entered.set()
            assert release.wait(2)
            return object()

        def unload(self):
            unloaded.set()

    manager = _bare_manager(Model())
    prediction = threading.Thread(
        target=manager.predict_shapes,
        args=(object(), "image.png"),
        kwargs={"batch": True},
    )
    prediction.start()
    assert entered.wait(1)

    unload = threading.Thread(target=manager.unload_model)
    unload.start()
    time.sleep(0.05)
    assert not unloaded.is_set()

    release.set()
    prediction.join(2)
    unload.join(2)
    assert not prediction.is_alive()
    assert not unload.is_alive()
    assert unloaded.is_set()


def test_unload_exception_is_reported_without_escaping():
    class Model:
        def unload(self):
            raise OSError("provider DLL unavailable")

    manager = _bare_manager(Model())
    statuses = []
    manager.new_model_status.connect(statuses.append)

    assert manager.unload_model() is False
    assert manager.loaded_model_config is None
    assert any("provider DLL unavailable" in message for message in statuses)
