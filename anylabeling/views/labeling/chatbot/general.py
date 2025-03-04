from PyQt5.QtCore import Qt, QEasingCurve, QTimer, QPropertyAnimation
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QFrame, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton
)

from anylabeling.views.labeling.chatbot.config import *
from anylabeling.views.labeling.chatbot.style import *
from anylabeling.views.labeling.chatbot.utils import *


class ChatMessage(QFrame):
    """Custom widget for a single chat message"""

    def __init__(self, role, content, parent=None, is_error=False):
        super().__init__(parent)
        self.role = role
        self.content = content
        self.is_error = is_error

        # Create message container with appropriate styling
        is_user = role == "user"

        # Set up layout
        layout = QVBoxLayout(self)
        # Set the spacing between bubble
        layout.setContentsMargins(0, 0, 0, 12)  # (left, top, right, bottom)

        # Create a horizontal layout to position the bubble
        h_container = QHBoxLayout()

        # Create bubble with smooth corners
        self.bubble = QWidget(self)
        self.bubble.setObjectName("messageBubble")
        self.bubble.setStyleSheet(ChatMessageStyle.get_bubble_style(is_user))

        # Set bubble width to 70% for user messages
        if is_user:
            h_container.addStretch(30)  # 30% empty space on the left
            h_container.addWidget(self.bubble, 70)  # 70% width for the bubble
        else:
            h_container.addWidget(self.bubble)  # Assistant bubble takes full width

        bubble_layout = QVBoxLayout(self.bubble)
        # Set the spacing between elements inside the bubble
        bubble_layout.setContentsMargins(12, 12, 12, 12)  # (left, top, right, bottom)

        # Add header with role
        header_layout = QHBoxLayout()
        role_label = QLabel(self.tr("User") if is_user else self.tr("Assistant"))
        role_label.setStyleSheet(ChatMessageStyle.get_role_label_style())

        # Add copy button to header
        copy_btn = QPushButton()
        copy_btn.setIcon(QIcon(set_icon_path("copy")))
        copy_btn.setFixedSize(*ICON_SIZE_SMALL)
        copy_btn.setStyleSheet(ChatMessageStyle.get_button_style())
        copy_btn.setToolTip(self.tr("Copy message"))
        copy_btn.setCursor(Qt.PointingHandCursor)
        copy_btn.clicked.connect(lambda: self.copy_content_to_clipboard(copy_btn))

        if is_user:
            header_layout.addStretch()
            header_layout.addWidget(role_label)
            header_layout.addWidget(copy_btn)
        else:
            header_layout.addWidget(role_label)
            header_layout.addWidget(copy_btn)
            header_layout.addStretch()

        bubble_layout.addLayout(header_layout)

        # Set maximum width for proper wrapping
        if is_user:
            content_label.setMinimumWidth(100)
            content_label.setMaximumWidth(400)
        # Add message content
        content_label = QLabel(content)
        content_label.setWordWrap(True)
        content_label.setTextFormat(Qt.PlainText)
        content_label.setStyleSheet(ChatMessageStyle.get_content_label_style(self.is_error))
        bubble_layout.addWidget(content_label)

        if is_user:
            layout.setAlignment(Qt.AlignRight)
        else:
            layout.setAlignment(Qt.AlignLeft)
        self.bubble.setMaximumWidth(2000)
        layout.addLayout(h_container)

        # Add animation when first appearing
        self.setMaximumHeight(0)
        self.animation = QPropertyAnimation(self, b"maximumHeight")
        self.animation.setDuration(int(ANIMATION_DURATION[:-2]))  # Convert "200ms" to 200
        self.animation.setStartValue(0)
        self.animation.setEndValue(self.sizeHint().height())
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()

    def copy_content_to_clipboard(self, button):
        """Copy message content to clipboard with visual feedback"""
        # Copy the content to clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(self.content)

        # Change the button icon to a checkmark
        button.setIcon(QIcon(set_icon_path("check")))

        # Reset the button after a delay
        QTimer.singleShot(1000, lambda: self.reset_copy_button(button))

    def reset_copy_button(self, button):
        """Reset the copy button to its original state"""
        button.setIcon(QIcon(set_icon_path("copy")))
        button.setToolTip(self.tr("Copy message"))
        button.setStyleSheet(ChatMessageStyle.get_button_style())
