from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class GenericWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        self.func(*self.args, **self.kwargs)
        self.finished.emit()
