from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from anylabeling.views.labeling.logger import logger


class GenericWorker(QObject):
    finished = pyqtSignal()
    failed = pyqtSignal(object)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        try:
            self.func(*self.args, **self.kwargs)
        except Exception as error:  # never let an exception escape a Qt slot
            logger.exception("Unhandled background worker error: %s", error)
            self.failed.emit(error)
        finally:
            # An escaped Python exception from a PyQt slot can terminate a
            # frozen application and otherwise leaves its QThread running.
            self.finished.emit()
