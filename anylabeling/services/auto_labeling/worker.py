from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class GenericWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        try:
            self.func(*self.args, **self.kwargs)
        finally:
            self.finished.emit()
