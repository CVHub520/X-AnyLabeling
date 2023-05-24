"""Defines Toaster widget"""

from PyQt5 import QtCore, QtGui, QtWidgets


class QToaster(QtWidgets.QFrame):
    """Toaster widget
    For displaying a short notification which can be hide after a duration
    """

    closed = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        QtWidgets.QHBoxLayout(self)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )

        self.setStyleSheet(
            """
            QToaster {
                border: 1px solid black;
                border-radius: 0px;
                color: rgb(30, 30, 30);
                background-color: rgb(255, 255, 255);
            }
        """
        )
        # alternatively:
        self.setAutoFillBackground(True)
        self.setFrameShape(self.Box)

        self.timer = QtCore.QTimer(singleShot=True, timeout=self.hide)

        if self.parent():
            self.opacity_effect = QtWidgets.QGraphicsOpacityEffect(opacity=0)
            self.setGraphicsEffect(self.opacity_effect)
            self.opacity_ani = QtCore.QPropertyAnimation(
                self.opacity_effect, b"opacity"
            )
            # we have a parent, install an eventFilter so that when it's resized
            # the notification will be correctly moved to the right corner
            self.parent().installEventFilter(self)
        else:
            # there's no parent, use the window opacity property, assuming that
            # the window manager supports it; if it doesn't, this won'd do
            # anything (besides making the hiding a bit longer by half a
            # second)
            self.opacity_ani = QtCore.QPropertyAnimation(
                self, b"windowOpacity"
            )
        self.opacity_ani.setStartValue(0.0)
        self.opacity_ani.setEndValue(1.0)
        self.opacity_ani.setDuration(100)
        self.opacity_ani.finished.connect(self.check_closed)

        self.corner = QtCore.Qt.TopLeftCorner
        self.margin = 10

    def check_closed(self):
        """Close the toaster after fading out"""
        # if we have been fading out, we're closing the notification
        if self.opacity_ani.direction() == self.opacity_ani.Backward:
            self.close()

    def restore(self):
        """Restore toaster (timer + opacity)"""
        # this is a "helper function", that can be called from mouseEnterEvent
        # and when the parent widget is resized. We will not close the
        # notification if the mouse is in or the parent is resized
        self.timer.stop()
        # also, stop the animation if it's fading out...
        self.opacity_ani.stop()
        # ...and restore the opacity
        if self.parent():
            self.opacity_effect.setOpacity(1)
        else:
            self.setWindowOpacity(1)

    def hide(self):
        """Hide toaster by opacity effect"""
        self.opacity_ani.setDirection(self.opacity_ani.Backward)
        self.opacity_ani.setDuration(500)
        self.opacity_ani.start()

    def eventFilter(self, source, event):
        """Event filter"""
        if source == self.parent() and event.type() == QtCore.QEvent.Resize:
            self.opacity_ani.stop()
            parent_rect = self.parent().rect()
            geo = self.geometry()
            if self.corner == QtCore.Qt.TopLeftCorner:
                geo.moveTopLeft(
                    parent_rect.topLeft()
                    + QtCore.QPoint(self.margin, self.margin)
                )
            elif self.corner == QtCore.Qt.TopRightCorner:
                geo.moveTopRight(
                    parent_rect.topRight()
                    + QtCore.QPoint(-self.margin, self.margin)
                )
            elif self.corner == QtCore.Qt.BottomRightCorner:
                geo.moveBottomRight(
                    parent_rect.bottomRight()
                    + QtCore.QPoint(-self.margin, -self.margin)
                )
            else:
                geo.moveBottomLeft(
                    parent_rect.bottomLeft()
                    + QtCore.QPoint(self.margin, -self.margin)
                )
            self.setGeometry(geo)
            self.restore()
            self.timer.start()
        return super().eventFilter(source, event)

    def enterEvent(self, _):
        """
        Restore toaster (opacity) when move mouse into it
        Keep it open as long as the mouse does not leave
        """
        self.restore()

    def leaveEvent(self, _):
        """
        When mouse leaves the toaster, start the timer again to
        count down to close event
        """
        self.timer.start()

    def closeEvent(self, _):
        """Handle close event"""
        # we don't need the notification anymore, delete it!
        self.deleteLater()

    def resizeEvent(self, event):
        """Handles request event"""
        super().resizeEvent(event)
        # if you don't set a stylesheet, you don't need any of the following!
        if not self.parent():
            # there's no parent, so we need to update the mask
            path = QtGui.QPainterPath()
            path.addRoundedRect(
                QtCore.QRectF(self.rect()).translated(-0.5, -0.5), 4, 4
            )
            self.setMask(
                QtGui.QRegion(
                    path.toFillPolygon(QtGui.QTransform()).toPolygon()
                )
            )
        else:
            self.clearMask()

    @staticmethod
    def show_message(
        parent,
        message,
        corner=QtCore.Qt.TopLeftCorner,
        margin=10,
        closable=True,
        timeout=5000,
        desktop=False,
        parent_window=True,
    ):  # pylint: disable=too-many-statements,too-many-locals,too-many-arguments
        """Show message as a toaster"""

        if parent and parent_window:
            parent = parent.window()

        if not parent or desktop:
            self = QToaster(None)
            self.setWindowFlags(
                self.windowFlags()
                | QtCore.Qt.FramelessWindowHint
                | QtCore.Qt.BypassWindowManagerHint
            )
            # This is a dirty hack!
            # parentless objects are garbage collected, so the widget will be
            # deleted as soon as the function that calls it returns, but if an
            # object is referenced to *any* other object it will not, at least
            # for PyQt (I didn't test it to a deeper level)
            self.__self = self

            current_screen = QtWidgets.QApplication.primaryScreen()
            if parent and parent.window().geometry().size().isValid():
                # the notification is to be shown on the desktop, but there is a
                # parent that is (theoretically) visible and mapped, we'll try to
                # use its geometry as a reference to guess which desktop shows
                # most of its area; if the parent is not a top level window, use
                # that as a reference
                reference = parent.window().geometry()
            else:
                # the parent has not been mapped yet, let's use the cursor as a
                # reference for the screen
                reference = QtCore.QRect(
                    QtGui.QCursor.pos() - QtCore.QPoint(1, 1),
                    QtCore.QSize(3, 3),
                )
            max_area = 0
            for screen in QtWidgets.QApplication.screens():
                intersected = screen.geometry().intersected(reference)
                area = intersected.width() * intersected.height()
                if area > max_area:
                    max_area = area
                    current_screen = screen
            parent_rect = current_screen.availableGeometry()
        else:
            self = QToaster(parent)
            parent_rect = parent.rect()

        self.timer.setInterval(timeout)

        label = QtWidgets.QLabel(message)
        label.setStyleSheet("color: rgb(33, 33, 33);")
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setWeight(100)
        label.setFont(font)
        self.layout().addWidget(label)

        if closable:
            close_button = QtWidgets.QToolButton()
            self.layout().addWidget(close_button)
            close_icon = self.style().standardIcon(
                QtWidgets.QStyle.SP_TitleBarCloseButton
            )
            close_button.setIcon(close_icon)
            close_button.setAutoRaise(True)
            close_button.clicked.connect(self.close)

        self.timer.start()

        # raise the widget and adjust its size to the minimum
        self.raise_()
        self.adjustSize()

        self.corner = corner
        self.margin = margin

        geo = self.geometry()
        # now the widget should have the correct size hints, let's move it to the
        # right place
        if corner == QtCore.Qt.TopLeftCorner:
            geo.moveTopLeft(
                parent_rect.topLeft() + QtCore.QPoint(margin, margin)
            )
        elif corner == QtCore.Qt.TopRightCorner:
            geo.moveTopRight(
                parent_rect.topRight() + QtCore.QPoint(-margin, margin)
            )
        elif corner == QtCore.Qt.BottomRightCorner:
            geo.moveBottomRight(
                parent_rect.bottomRight() + QtCore.QPoint(-margin, -margin)
            )
        else:
            geo.moveBottomLeft(
                parent_rect.bottomLeft() + QtCore.QPoint(margin, -margin)
            )

        self.setGeometry(geo)
        self.show()
        self.opacity_ani.start()
