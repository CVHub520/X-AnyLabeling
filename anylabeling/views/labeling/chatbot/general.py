from PyQt5.QtCore import Qt, QEasingCurve, QTimer, QPropertyAnimation
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QFrame, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QSizePolicy
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
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(0)

        # Create a horizontal layout to position the bubble
        h_container = QHBoxLayout()
        h_container.setContentsMargins(0, 0, 0, 0)
        h_container.setSpacing(0)

        # Create bubble with smooth corners
        self.bubble = QWidget(self)
        self.bubble.setObjectName("messageBubble")
        self.bubble.setStyleSheet(ChatMessageStyle.get_bubble_style(is_user))

        if is_user:
            h_container.addStretch(100 - MAX_USER_MSG_WIDTH)
            h_container.addWidget(self.bubble, MAX_USER_MSG_WIDTH)
        else:
            h_container.addWidget(self.bubble, 100)

        bubble_layout = QVBoxLayout(self.bubble)
        bubble_layout.setContentsMargins(10, 8, 10, 8)
        bubble_layout.setSpacing(4)

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

        # Add message content
        processed_content = ""
        if len(content) > 50 and " " not in content:
            chunk_size = 50
            for i in range(0, len(content), chunk_size):
                processed_content += content[i:i+chunk_size]
                if i + chunk_size < len(content):
                    processed_content += "\u200B"  # Zero-width space, allows line breaks but is invisible
        else:
            processed_content = content

        # Create label with processed content
        content_label = QLabel(processed_content)
        content_label.setWordWrap(True)
        
        # Force minimum width to ensure correct wrapping
        content_label.setMinimumWidth(200)

        # To ensure long content displays correctly, we use RichText format
        if "\u200B" in processed_content or is_error:
            content_label.setTextFormat(Qt.RichText)
        else:
            content_label.setTextFormat(Qt.PlainText)

        content_label.setStyleSheet(ChatMessageStyle.get_content_label_style(self.is_error))

        # Add text copy selection combination flag
        content_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)

        # Use a more appropriate size policy to avoid excessive vertical space
        content_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        # Ensure consistent font display across all platforms
        default_font = content_label.font()
        content_label.setFont(default_font)
        
        # Set alignment
        content_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        # Set minimum height to avoid excessive height
        content_label.setMinimumHeight(10)

        self.content_label = content_label
        
        bubble_layout.addWidget(content_label)

        self.bubble.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        
        if parent:
            parent_width = parent.width()
            if parent_width > 0:
                # Different width constraints for user vs assistant
                if self.role == "user":
                    max_width = int(parent_width * MAX_USER_MSG_WIDTH / 100)
                else:
                    max_width = int(parent_width)
                
                if max_width > 0:
                    if hasattr(self, 'content_label'):
                        self.content_label.setMaximumWidth(max_width - 20)

                    self.bubble.setMaximumWidth(max_width)

                    if hasattr(self, 'content_label') and hasattr(self, 'adjust_height_after_animation'):
                        self.adjust_height_after_animation()

                    self.updateGeometry()
                    self.bubble.updateGeometry()

        layout.addLayout(h_container)

        # Add animation when first appearing
        self.setMaximumHeight(0)
        self.animation = QPropertyAnimation(self, b"maximumHeight")
        self.animation.setDuration(int(ANIMATION_DURATION[:-2]))  # Convert "200ms" to 200
        self.animation.setStartValue(0)

        content_height = self.content_label.sizeHint().height()
        bubble_height = content_height + (bubble_layout.contentsMargins().top() + 
                                         bubble_layout.contentsMargins().bottom() + 
                                         bubble_layout.spacing() + 
                                         30)
        
        # Set minimum height to avoid excessive height
        min_height = 40
        anim_end_height = max(min_height, bubble_height)

        # Set end value and easing curve
        self.animation.setEndValue(anim_end_height)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # After animation, adjust height to a reasonable value
        self.animation.finished.connect(self.adjust_height_after_animation)

        self.animation.start()

    def copy_content_to_clipboard(self, button):
        """Copy message content to clipboard with visual feedback"""
        # Copy the content to clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(self.content)

        # Change the button icon to a checkmark
        button.setIcon(QIcon(set_icon_path("check")))

        # Start a timer to reset the button after a delay
        QTimer.singleShot(2000, lambda: self.reset_copy_button(button))

    def update_width_constraint(self):
        """Update width constraint based on parent width"""
        if self.parent():
            parent_width = self.parent().width()
            if parent_width > 0:
                if self.role == "user":
                    max_width = int(parent_width * MAX_USER_MSG_WIDTH / 100)
                else:
                    max_width = int(parent_width)

                if max_width > 0:
                    if hasattr(self, 'content_label'):
                        self.content_label.setMaximumWidth(max_width - 20)

                    self.bubble.setMaximumWidth(max_width)

                    if hasattr(self, 'content_label') and hasattr(self, 'adjust_height_after_animation'):
                        self.adjust_height_after_animation()

                    self.updateGeometry()
                    self.bubble.updateGeometry()

    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        self.update_width_constraint()

    def reset_copy_button(self, button):
        """Reset the copy button to its original state"""
        button.setIcon(QIcon(set_icon_path("copy")))
        button.setToolTip(self.tr("Copy message"))
        button.setStyleSheet(ChatMessageStyle.get_button_style())

    def adjust_height_after_animation(self):
        """Adjust height after animation"""
        # Get the actual height needed for the content
        content_height = self.content_label.heightForWidth(self.content_label.width())
        # If the height cannot be obtained correctly, use sizeHint
        if content_height <= 0:
            content_height = self.content_label.sizeHint().height()
        
        # Add extra space for the header and padding
        total_height = content_height + 40
        
        # Set a reasonable maximum height, slightly larger than the calculated height
        self.setMaximumHeight(total_height + 10)
