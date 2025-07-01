from PyQt5.QtCore import Qt, QSize, QRect, QPoint
from PyQt5.QtWidgets import QLayout, QSizePolicy


class FlowLayout(QLayout):
    """
    A custom layout that arranges child widgets in a flowing manner, similar to text wrapping.

    This layout places widgets horizontally until the row is full, then wraps to a new line.
    It's particularly useful for dynamically-sized UIs where widget dimensions may vary or
    where a fixed number of columns is not desired.

    Attributes:
        itemList (List[QLayoutItem]): List of layout items managed by the layout.

    Args:
        parent (QWidget, optional): The parent widget of the layout. Defaults to None.
        margin (int, optional): Margin around the layout. Defaults to 0.
        spacing (int, optional): Spacing between widgets. Defaults to -1 (use style defaults).
    """

    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self.itemList = []

    def __del__(self):
        """
        Destructor that ensures all layout items are removed and cleaned up.
        """
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        """
        Adds a new QLayoutItem to the layout.

        Args:
            item (QLayoutItem): The layout item to add.
        """
        self.itemList.append(item)

    def count(self):
        """
        Returns the number of items in the layout.

        Returns:
            int: The number of layout items.
        """
        return len(self.itemList)

    def itemAt(self, index):
        """
        Retrieves the item at the specified index.

        Args:
            index (int): The index of the item.

        Returns:
            QLayoutItem or None: The layout item at the given index, or None if out of bounds.
        """
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        """
        Removes and returns the item at the specified index.

        Args:
            index (int): The index of the item to remove.

        Returns:
            QLayoutItem or None: The removed layout item, or None if index is invalid.
        """
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        """
        Specifies that this layout does not expand in any direction.

        Returns:
            Qt.Orientations: No orientation expansion.
        """
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        """
        Indicates the layout prefers height based on its width.

        Returns:
            bool: True, since height is dynamically calculated from width.
        """
        return True

    def heightForWidth(self, width):
        """
        Calculates the height required for a given width.

        Args:
            width (int): The available width.

        Returns:
            int: The corresponding height required for layout.
        """
        return self.doLayout(QRect(0, 0, width, 0), testOnly=True)

    def setGeometry(self, rect):
        """
        Applies the layout geometry within the given rectangle.

        Args:
            rect (QRect): The rectangle within which to lay out widgets.
        """
        super().setGeometry(rect)
        self.doLayout(rect, testOnly=False)

    def sizeHint(self):
        """
        Provides a recommended size for the layout.

        Returns:
            QSize: The suggested size.
        """
        return self.minimumSize()

    def minimumSize(self):
        """
        Calculates the minimum size needed to fit all items.

        Returns:
            QSize: The minimum layout size.
        """
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        return size

    def doLayout(self, rect, testOnly):
        """
        Performs the actual layout of items within the specified rectangle.

        Items are arranged left to right, wrapping to the next line as needed.

        Args:
            rect (QRect): The bounding rectangle for layout.
            testOnly (bool): If True, performs calculations only without setting geometry.

        Returns:
            int: The total height used by the layout.
        """
        x = rect.x()
        y = rect.y()
        lineHeight = 0
        spacing = self.spacing()

        for item in self.itemList:
            wid = item.widget()
            spaceX = spacing + wid.style().layoutSpacing(
                QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Horizontal
            )
            spaceY = spacing + wid.style().layoutSpacing(
                QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Vertical
            )

            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()

    def clear(self):
        """
        Removes all items from the layout and deletes their associated widgets.
        """
        while self.itemList:
            item = self.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()
