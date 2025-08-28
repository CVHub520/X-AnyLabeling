from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLineEdit, QTextEdit


class AutoResizeTextEdit(QTextEdit):
    """
    A QTextEdit widget that automatically adjusts its height based on content.

    This widget dynamically resizes its height within the specified limits
    (initial and maximum height) as the user types, improving UX by reducing
    the need for scrollbars when unnecessary.

    Args:
        initial_height (int, optional): The starting height of the widget. Defaults to 60.
        max_height (int, optional): The maximum height limit. Defaults to 200.
        parent (QWidget, optional): The parent widget. Defaults to None.
    """

    def __init__(self, initial_height=60, max_height=200, parent=None):
        super().__init__(parent)
        self.initial_height = initial_height
        self.max_height = max_height

        self.setFixedHeight(self.initial_height)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setLineWrapMode(QTextEdit.WidgetWidth)

        self.textChanged.connect(self.adjust_height)

    def adjust_height(self):
        """
        Adjusts the height of the text edit based on the content size.
        """
        doc = self.document()
        doc_height = doc.size().height()

        content_height = int(doc_height) + 16
        new_height = max(content_height, self.initial_height)

        if new_height > self.max_height:
            new_height = self.max_height
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        else:
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        if self.height() != new_height:
            self.setFixedHeight(new_height)


class PageInputLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vqa_dialog = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            text = self.text().strip()

            if not text:
                if self.vqa_dialog:
                    self.vqa_dialog.restore_current_page_number()
                return

            if self.vqa_dialog:
                self.vqa_dialog.switch_image("jump")
            return

        super().keyPressEvent(event)


def create_truncated_widget(text, widget_class, max_width=500):
    """
    Creates a widget with truncated text if it exceeds the specified maximum width.

    This function instantiates a given widget class (e.g., QLabel or QPushButton),
    sets its text, and truncates the text with an ellipsis if it exceeds the
    specified `max_width`. In such cases, the full text is also added as a tooltip.

    Args:
        text (str): The full text to display in the widget.
        widget_class (Type[QWidget]): The widget class to instantiate. Must have
            setText(), setToolTip(), and fontMetrics() methods (e.g., QLabel).
        max_width (int, optional): The maximum allowed width for the text in pixels.
            Defaults to 500.

    Returns:
        QWidget: An instance of the specified widget class with appropriate text and width.

    Examples:
        >>> label = create_truncated_widget("A very long string...", QLabel, max_width=100)
        >>> label.toolTip()
        'A very long string...'
    """
    widget = widget_class()
    font_metrics = widget.fontMetrics()
    text_width = font_metrics.horizontalAdvance(text)

    if text_width > max_width:
        elided_text = font_metrics.elidedText(text, Qt.ElideRight, max_width)
        widget.setText(elided_text)
        widget.setToolTip(text)
        widget.setMaximumWidth(max_width + 20)
    else:
        widget.setText(text)

    return widget
