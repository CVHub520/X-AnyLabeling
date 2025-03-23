from PyQt5.QtCore import QObject, pyqtSignal


class StreamingHandler(QObject):
    """Handler for streaming text updates"""

    text_update = pyqtSignal(str)
    finished = pyqtSignal(bool)
    loading = pyqtSignal(bool)
    typing = pyqtSignal(bool)  # Signal for typing animation
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.current_message = ""
        self.stop_requested = False

    def reset(self):
        """Reset the current message buffer"""
        self.current_message = ""

    def append_text(self, text):
        """Append text to current message and emit update"""
        self.current_message += text
        self.text_update.emit(text)

    def get_current_message(self):
        """Get the complete current message"""
        return self.current_message

    def start_loading(self):
        """Indicate loading state has started"""
        self.loading.emit(True)

    def stop_loading(self):
        """Indicate loading state has stopped"""
        self.loading.emit(False)
        self.stop_requested = False

    def start_typing(self):
        """Indicate typing animation should start"""
        self.typing.emit(True)

    def stop_typing(self):
        """Indicate typing animation should stop"""
        self.typing.emit(False)

    def report_error(self, error_message):
        """Report an error that occurred during streaming"""
        self.error_occurred.emit(error_message)
