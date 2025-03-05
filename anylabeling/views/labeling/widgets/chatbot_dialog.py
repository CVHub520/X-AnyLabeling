import base64
import threading
from openai import OpenAI

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QButtonGroup,
    QApplication,
    QScrollArea,
    QFrame,
    QSplitter,
)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QSize, Qt

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.general import open_url
from anylabeling.views.labeling.chatbot import *


class ChatbotDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.resize(*DEFAULT_WINDOW_SIZE)  # Wider to accommodate three columns
        self.setWindowIcon(QIcon(set_icon_path("chat")))

        # Apply global styles
        self.setStyleSheet(ChatbotDialogStyle.get_dialog_style())

        # Initialize cache for storing image-text mappings and chat history
        self.chat_history = []  # Store chat history

        # Streaming handler setup
        self.stream_handler = StreamingHandler()
        self.stream_handler.text_update.connect(self.update_output)
        self.stream_handler.finished.connect(self.on_stream_finished)
        self.stream_handler.loading.connect(self.handle_loading_state)
        self.stream_handler.error_occurred.connect(self.handle_error)

        # Main horizontal layout with splitters
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create main splitter for three columns
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(1)
        self.main_splitter.setStyleSheet(ChatbotDialogStyle.get_main_splitter_style())

        ################################
        # Left panel - Model Providers #
        ################################
        self.left_widget = QWidget()
        left_panel = QVBoxLayout(self.left_widget)
        left_panel.setContentsMargins(20, 24, 20, 24)
        left_panel.setSpacing(12)

        # Create button group for providers
        provider_group = QButtonGroup(self)

        # Set provider buttons
        providers = PROVIDER_CONFIGS.keys()
        for provider in providers:
            btn = QPushButton(self.tr(provider.capitalize()))
            btn.setIcon(QIcon(PROVIDER_CONFIGS[provider]["icon"]))
            btn.setCheckable(True)
            btn.setFixedHeight(40)
            # Set icon size and text alignment
            btn.setIconSize(QSize(20, 20))
            btn.setStyleSheet(ChatbotDialogStyle.get_provider_button_style())
            # Connect button to switch provider using a default argument
            btn.clicked.connect(lambda checked, p=provider: self.switch_provider(p))
            provider_group.addButton(btn)  # Add button to group
            setattr(self, f"{provider}_btn", btn)
            left_panel.addWidget(btn)

        # Set default fields
        getattr(self, f"{DEFAULT_PROVIDER}_btn").setChecked(True)

        # Add stretch to push everything to the top
        left_panel.addStretch()

        # Styling for the left panel
        self.left_widget.setStyleSheet(ChatbotDialogStyle.get_left_widget_style())
        self.left_widget.setMinimumWidth(200)
        self.left_widget.setMaximumWidth(250)

        #################################
        # Middle panel - Chat interface #
        #   - chat_layout (85%)         #
        #   - input_layout (15%)        #
        #################################
        self.middle_widget = QWidget()
        self.middle_widget.setStyleSheet(ChatbotDialogStyle.get_middle_widget_style())
        middle_panel = QVBoxLayout(self.middle_widget)
        middle_panel.setContentsMargins(0, 0, 0, 0)
        middle_panel.setSpacing(0)

        # Chat area - takes 90% of the vertical space
        chat_container = QWidget()
        chat_container.setStyleSheet(ChatbotDialogStyle.get_chat_container_style())
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(24, 20, 24, 20)
        chat_layout.setSpacing(16)

        # Scroll area for chat messages
        self.chat_scroll_area = QScrollArea()
        self.chat_scroll_area.setWidgetResizable(True)
        self.chat_scroll_area.setFrameShape(QFrame.NoFrame)
        self.chat_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_scroll_area.setStyleSheet(ChatbotDialogStyle.get_chat_scroll_area_style())

        # Widget to contain all chat messages
        self.chat_container = QWidget()
        self.chat_container.setStyleSheet(ChatbotDialogStyle.get_chat_container_style())
        self.chat_messages_layout = QVBoxLayout(self.chat_container)
        self.chat_messages_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_messages_layout.setSpacing(16)
        self.chat_messages_layout.addStretch()

        self.chat_scroll_area.setWidget(self.chat_container)
        chat_layout.addWidget(self.chat_scroll_area)

        # Input area with simplified design
        input_container = QWidget()
        input_container.setStyleSheet(ChatbotDialogStyle.get_input_container_style())
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(24, 12, 24, 12)
        input_layout.setSpacing(0)

        # Create a container for the input field with embedded send button
        input_frame = QFrame()
        input_frame.setObjectName("inputFrame")
        input_frame.setStyleSheet(ChatbotDialogStyle.get_input_frame_style())

        # Use a relative layout for the input frame
        input_frame_layout = QVBoxLayout(input_frame)
        input_frame_layout.setContentsMargins(12, 8, 12, 8)
        input_frame_layout.setSpacing(0)

        # Create the message input
        self.message_input = QTextEdit()
        self.message_input.setPlaceholderText(self.tr("Type your message here..."))
        self.message_input.setStyleSheet(ChatbotDialogStyle.get_message_input_style())
        self.message_input.setAcceptRichText(False)
        self.message_input.setMinimumHeight(MIN_MSG_INPUT_HEIGHT)
        self.message_input.setMaximumHeight(MAX_MSG_INPUT_HEIGHT)
        self.message_input.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.message_input.setFrameShape(QFrame.NoFrame)
        self.message_input.setFrameShadow(QFrame.Plain)
        self.message_input.setLineWrapMode(QTextEdit.WidgetWidth)

        # Completely remove any internal frame or border
        document = self.message_input.document()
        document.setDocumentMargin(0)
        self.message_input.setDocument(document)

        # Set text format to remove any block margins
        cursor = self.message_input.textCursor()
        format = cursor.blockFormat()
        format.setBottomMargin(0)
        format.setTopMargin(0)
        cursor.setBlockFormat(format)
        self.message_input.setTextCursor(cursor)

        # Connect textChanged to dynamically resize input
        self.message_input.textChanged.connect(self.resize_input)
        self.message_input.installEventFilter(self)  # For Enter key handling

        # Initialize the input size
        self.resize_input()

        # Create a container for the input and send button
        input_with_button = QWidget()
        input_with_button_layout = QHBoxLayout(input_with_button)
        input_with_button_layout.setContentsMargins(0, 0, 0, 0)
        input_with_button_layout.setSpacing(0)
        input_with_button_layout.setAlignment(Qt.AlignVCenter)

        # Add the message input to the layout
        input_with_button_layout.addWidget(self.message_input, 1, Qt.AlignVCenter)

        # Create the send button
        self.send_btn = QPushButton()
        self.send_btn.setIcon(QIcon(set_icon_path("send")))
        self.send_btn.setIconSize(QSize(20, 20))
        self.send_btn.setStyleSheet(ChatbotDialogStyle.get_send_button_style())
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setFixedSize(24, 24)
        self.send_btn.clicked.connect(self.start_generation)

        # Add the send button to the layout
        input_with_button_layout.addWidget(self.send_btn, 0, Qt.AlignRight | Qt.AlignBottom)
        input_frame_layout.addWidget(input_with_button)
        input_layout.addWidget(input_frame)

        # Add the chat container and input container to the middle panel
        middle_panel.addWidget(chat_container, 88)
        middle_panel.addWidget(input_container, 12)

        ############################################
        # Right panel - Image preview and settings #
        ############################################
        self.right_widget = QWidget()
        right_panel = QVBoxLayout(self.right_widget)
        right_panel.setContentsMargins(0, 0, 0, 0)
        right_panel.setSpacing(0)
        
        # Image preview panel
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(24, 24, 24, 16)

        # Image preview
        self.image_preview = QLabel()
        self.image_preview.setStyleSheet(ChatbotDialogStyle.get_image_preview_style())
        self.image_preview.setMinimumHeight(200)
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setScaledContents(False)
        image_layout.addWidget(self.image_preview)

        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.prev_image_btn = QPushButton()
        self.prev_image_btn.setIcon(QIcon(set_icon_path("arrow-left")))
        self.prev_image_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.prev_image_btn.setStyleSheet(ChatbotDialogStyle.get_navigation_btn_style())
        self.prev_image_btn.setToolTip(self.tr("Previous Image"))
        self.prev_image_btn.setCursor(Qt.PointingHandCursor)
        self.prev_image_btn.clicked.connect(self.link_previous_image)

        self.next_image_btn = QPushButton()
        self.next_image_btn.setIcon(QIcon(set_icon_path("arrow-right")))
        self.next_image_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.next_image_btn.setStyleSheet(ChatbotDialogStyle.get_navigation_btn_style())
        self.next_image_btn.setToolTip(self.tr("Next Image"))
        self.next_image_btn.setCursor(Qt.PointingHandCursor)
        self.next_image_btn.clicked.connect(self.link_next_image)
        nav_layout.addWidget(self.prev_image_btn)
        nav_layout.addStretch()

        # Add image and video buttons for importing media
        self.open_image_btn = QPushButton()
        self.open_image_btn.setIcon(QIcon(set_icon_path("image")))
        self.open_image_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.open_image_btn.setStyleSheet(ChatbotDialogStyle.get_navigation_btn_style())
        self.open_image_btn.setToolTip(self.tr("Open Image Folder"))
        self.open_image_btn.setCursor(Qt.PointingHandCursor)
        self.open_image_btn.clicked.connect(self.open_image_folder)
        nav_layout.addWidget(self.open_image_btn)

        self.open_video_btn = QPushButton()
        self.open_video_btn.setIcon(QIcon(set_icon_path("video")))
        self.open_video_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.open_video_btn.setStyleSheet(ChatbotDialogStyle.get_navigation_btn_style())
        self.open_video_btn.setToolTip(self.tr("Open Video File"))
        self.open_video_btn.setCursor(Qt.PointingHandCursor)
        self.open_video_btn.clicked.connect(self.open_video_file)
        nav_layout.addWidget(self.open_video_btn)
        nav_layout.addStretch()

        nav_layout.addWidget(self.next_image_btn)
        image_layout.addLayout(nav_layout)

        # Settings panel
        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(24, 24, 24, 24)
        settings_layout.setSpacing(12)

        # API Address with help icon
        api_address_container = QHBoxLayout()
        api_address_label = QLabel(self.tr("API Address"))
        api_address_label.setStyleSheet(ChatbotDialogStyle.get_settings_label_style())

        # Create a container for label and help button
        label_with_help = QWidget()
        label_help_layout = QHBoxLayout(label_with_help)
        label_help_layout.setContentsMargins(0, 0, 0, 0)

        api_help_btn = QPushButton()
        api_help_btn.setObjectName("api_help_btn")
        api_help_btn.setIcon(QIcon(set_icon_path("help-circle")))
        api_help_btn.setFixedSize(*ICON_SIZE_SMALL)
        api_help_btn.setStyleSheet(ChatbotDialogStyle.get_help_btn_style())
        api_help_btn.setToolTip(self.tr("View API documentation"))
        api_help_btn.setCursor(Qt.PointingHandCursor)
        api_help_btn.clicked.connect(lambda: open_url(PROVIDER_CONFIGS[DEFAULT_PROVIDER]["api_docs_url"]))

        label_help_layout.addWidget(api_address_label)
        label_help_layout.addWidget(api_help_btn)
        label_help_layout.addStretch()

        api_address_container.addWidget(label_with_help)
        api_address_container.addStretch()
        settings_layout.addLayout(api_address_container)

        self.api_address = QLineEdit(PROVIDER_CONFIGS[DEFAULT_PROVIDER]["api_address"])
        self.api_address.setStyleSheet(ChatbotDialogStyle.get_settings_edit_style())
        self.api_address.installEventFilter(self)
        settings_layout.addWidget(self.api_address)
        
        # Model Name with help icon
        model_name_container = QHBoxLayout()
        model_name_label = QLabel(self.tr("Model Name"))
        model_name_label.setStyleSheet(ChatbotDialogStyle.get_settings_label_style())

        # Create a container for label and help button
        model_label_with_help = QWidget()
        model_label_help_layout = QHBoxLayout(model_label_with_help)
        model_label_help_layout.setContentsMargins(0, 0, 0, 0)

        model_help_btn = QPushButton()
        model_help_btn.setObjectName("model_help_btn")
        model_help_btn.setIcon(QIcon(set_icon_path("help-circle")))
        model_help_btn.setFixedSize(*ICON_SIZE_SMALL)
        model_help_btn.setStyleSheet(ChatbotDialogStyle.get_help_btn_style())
        model_help_btn.setToolTip(self.tr("View model details"))
        model_help_btn.setCursor(Qt.PointingHandCursor)
        model_help_btn.clicked.connect(lambda: open_url(PROVIDER_CONFIGS[DEFAULT_PROVIDER]["model_docs_url"]))

        model_label_help_layout.addWidget(model_name_label)
        model_label_help_layout.addWidget(model_help_btn)
        model_label_help_layout.addStretch()

        model_name_container.addWidget(model_label_with_help)
        model_name_container.addStretch()
        settings_layout.addLayout(model_name_container)
        
        self.model_name = QLineEdit(PROVIDER_CONFIGS[DEFAULT_PROVIDER]["model_name"])
        self.model_name.setStyleSheet(ChatbotDialogStyle.get_settings_edit_style())
        self.model_name.installEventFilter(self)
        settings_layout.addWidget(self.model_name)

        # API Key with help icon
        api_key_container = QHBoxLayout()
        api_key_label = QLabel(self.tr("API Key"))
        api_key_label.setStyleSheet(ChatbotDialogStyle.get_settings_label_style())

        # Create a container for label and help button
        key_label_with_help = QWidget()
        key_label_help_layout = QHBoxLayout(key_label_with_help)
        key_label_help_layout.setContentsMargins(0, 0, 0, 0)
        
        api_key_help_btn = QPushButton()
        api_key_help_btn.setObjectName("api_key_help_btn")
        api_key_help_btn.setIcon(QIcon(set_icon_path("help-circle")))
        api_key_help_btn.setFixedSize(*ICON_SIZE_SMALL)
        api_key_help_btn.setStyleSheet(ChatbotDialogStyle.get_help_btn_style())
        api_key_help_btn.setToolTip(self.tr("Get API key"))
        api_key_help_btn.setCursor(Qt.PointingHandCursor)
        api_key_help_btn.clicked.connect(lambda: open_url(PROVIDER_CONFIGS[DEFAULT_PROVIDER]["api_key_url"]))

        key_label_help_layout.addWidget(api_key_label)
        key_label_help_layout.addWidget(api_key_help_btn)
        key_label_help_layout.addStretch()

        api_key_container.addWidget(key_label_with_help)
        api_key_container.addStretch()
        settings_layout.addLayout(api_key_container)

        # API key input with toggle visibility
        api_key_container = QHBoxLayout()

        self.api_key = QLineEdit(PROVIDER_CONFIGS[DEFAULT_PROVIDER]["api_key"])
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText(self.tr("Enter API key"))
        self.api_key.setStyleSheet(ChatbotDialogStyle.get_settings_edit_style())
        self.api_key.installEventFilter(self)

        self.toggle_visibility_btn = QPushButton()
        self.toggle_visibility_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.toggle_visibility_btn.setIcon(QIcon(set_icon_path("eye-off")))
        self.toggle_visibility_btn.setCheckable(True)
        self.toggle_visibility_btn.setStyleSheet(ChatbotDialogStyle.get_toggle_visibility_btn_style())
        self.toggle_visibility_btn.clicked.connect(self.toggle_api_key_visibility)

        api_key_container.addWidget(self.api_key)
        api_key_container.addWidget(self.toggle_visibility_btn)
        settings_layout.addLayout(api_key_container)
        settings_layout.addStretch()

        # Create a splitter for the right panel to separate image and settings
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setHandleWidth(1)
        right_splitter.setStyleSheet(ChatbotDialogStyle.get_right_splitter_style())

        right_splitter.addWidget(image_panel)
        right_splitter.addWidget(settings_panel)

        # Set initial sizes for right splitter (40% image, 60% settings)
        right_splitter.setSizes([300, 400])
        right_panel.addWidget(right_splitter)

        # Styling for the right panel
        self.right_widget.setStyleSheet(ChatbotDialogStyle.get_right_widget_style())
        self.right_widget.setMinimumWidth(300)
        self.right_widget.setMaximumWidth(400)

        # Add panels to main splitter
        self.main_splitter.addWidget(self.left_widget)
        self.main_splitter.addWidget(self.middle_widget)
        self.main_splitter.addWidget(self.right_widget)
        self.main_splitter.setSizes([200, 700, 300])

        # Set stretch factors to ensure middle panel gets priority when resizing
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)

        main_layout.addWidget(self.main_splitter)

        # Streaming state
        self.streaming = False
        self.stream_thread = None

        # Add loading indicators
        self.loading_timer = None
        self.loading_dots = 0
        self.loading_message = None

        # Current assistant message for streaming updates
        self.current_assistant_message = None

        # After initializing UI components, load initial data if available
        self.load_initial_data()

    def switch_provider(self, provider):
        """Switch between different model providers"""
        if provider in PROVIDER_CONFIGS:
            self.api_address.setText(PROVIDER_CONFIGS[provider]["api_address"])
            api_docs_url = PROVIDER_CONFIGS[provider]["api_docs_url"]
            api_help_btn = self.findChild(QPushButton, "api_help_btn")
            if api_help_btn:
                if api_docs_url:
                    # Show the help button and update its click handler
                    api_help_btn.setVisible(True)
                    api_help_btn.clicked.disconnect()
                    api_help_btn.clicked.connect(lambda: open_url(api_docs_url))
                else:
                    # Hide the help button if there's no API docs URL
                    api_help_btn.setVisible(False)

            self.model_name.setText(PROVIDER_CONFIGS[provider]["model_name"])
            model_docs_url = PROVIDER_CONFIGS[provider]["model_docs_url"]
            model_help_btn = self.findChild(QPushButton, "model_help_btn")
            if model_help_btn:
                if model_docs_url:
                    model_help_btn.setVisible(True)
                    model_help_btn.clicked.disconnect()
                    model_help_btn.clicked.connect(lambda: open_url(model_docs_url))
                else:
                    model_help_btn.setVisible(False)

            api_key_url = PROVIDER_CONFIGS[provider]["api_key_url"]
            api_key_help_btn = self.findChild(QPushButton, "api_key_help_btn")
            if api_key_help_btn:
                if api_key_url:
                    api_key_help_btn.setVisible(True)
                    api_key_help_btn.clicked.disconnect()
                    api_key_help_btn.clicked.connect(lambda: open_url(api_key_url))
                else:
                    api_key_help_btn.setVisible(False)

    def resize_input(self):
        """Dynamically resize input based on content"""
        # Calculate required height
        document = self.message_input.document()
        doc_height = document.size().height()
        margins = self.message_input.contentsMargins()

        # Calculate total height needed
        total_height = doc_height + margins.top() + margins.bottom() + 4

        # Set a maximum height limit
        max_height = MAX_MSG_INPUT_HEIGHT

        # Determine if we need scrollbars
        needs_scrollbar = total_height > max_height

        # Set the appropriate height
        if needs_scrollbar:
            # Use maximum height and enable scrollbar
            self.message_input.setMinimumHeight(max_height)
            self.message_input.setMaximumHeight(max_height)
            self.message_input.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

            # Ensure cursor is visible by scrolling to it
            cursor = self.message_input.textCursor()
            self.message_input.ensureCursorVisible()
        else:
            # Use calculated height and disable scrollbar
            actual_height = max(total_height, MIN_MSG_INPUT_HEIGHT)
            self.message_input.setMinimumHeight(actual_height)
            self.message_input.setMaximumHeight(actual_height)
            self.message_input.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Force update to ensure changes take effect immediately
        self.message_input.updateGeometry()
        QApplication.processEvents()
    
    def update_image_preview(self):
        """Update the image preview when switching images"""
        if self.parent().filename:
            pixmap = QPixmap(self.parent().filename)

            # If pixmap is valid
            if not pixmap.isNull():
                # Calculate scaled size while maintaining aspect ratio
                preview_size = self.image_preview.size()

                # Only scale if the preview has a valid size (width and height > 0)
                if preview_size.width() > 0 and preview_size.height() > 0:
                    scaled_pixmap = pixmap.scaled(
                        preview_size.width(),
                        preview_size.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )

                    self.image_preview.setPixmap(scaled_pixmap)
                    self.image_preview.setAlignment(Qt.AlignCenter)
                else:
                    # If the preview size isn't valid yet, schedule another update
                    QTimer.singleShot(200, self.update_image_preview)
            else:
                self.image_preview.setText(self.tr("Image not available"))
    
    def load_initial_data(self):
        """Load initial data from parent's other_data if available"""
        if self.parent().filename:
            other_data = self.parent().other_data

            # If we have chat history
            if "chat_history" in other_data and isinstance(other_data["chat_history"], list):
                # Load chat history and display messages
                for message in other_data["chat_history"]:
                    if "role" in message and "content" in message:
                        self.add_message(message["role"], message["content"])

            # Update image preview
            self.update_image_preview()

            # Add a slight delay to ensure the widget is fully rendered before scaling the image
            QTimer.singleShot(100, self.update_image_preview)

        # Update visibility of import buttons based on whether files are loaded
        self.update_import_buttons_visibility()
    
    def update_import_buttons_visibility(self):
        """Update visibility of import buttons based on whether files are loaded"""
        has_images = bool(self.parent().image_list)
        # Show navigation buttons only when images are loaded
        self.prev_image_btn.setVisible(has_images)
        self.next_image_btn.setVisible(has_images)
    
    def open_image_folder(self):
        """Open an image folder"""
        if hasattr(self.parent(), 'open_folder_dialog'):
            self.parent().open_folder_dialog()
            # Check if images were successfully loaded
            if self.parent().image_list:
                # Update image preview
                self.update_image_preview()
                # Load initial data
                self.load_initial_data()
                # Update button visibility
                self.update_import_buttons_visibility()
    
    def open_video_file(self):
        """Open a video file"""
        if hasattr(self.parent(), 'open_video_file'):
            self.parent().open_video_file()
            # Check if frames were successfully extracted
            if self.parent().image_list:
                # Update image preview
                self.update_image_preview()
                # Load initial data
                self.load_initial_data()
                # Update button visibility
                self.update_import_buttons_visibility()
    
    def add_message(self, role, content, delete_last_message=False):
        """Add a new message to the chat area"""
        # Remove the stretch item if it exists
        while self.chat_messages_layout.count() > 0:
            item = self.chat_messages_layout.itemAt(self.chat_messages_layout.count()-1)
            if item and item.spacerItem():
                self.chat_messages_layout.removeItem(item)
                break

        # Create and add the message widget
        is_error = True if delete_last_message else False
        message_widget = ChatMessage(role, content, self.chat_container, is_error=is_error)
        self.chat_messages_layout.addWidget(message_widget)
        
        # Add the stretch back
        self.chat_messages_layout.addStretch()
        
        # Store message in chat history
        if delete_last_message:
            self.chat_history = self.chat_history[:-1]  # roll back the last message
        else:
            self.chat_history.append({"role": role, "content": content})

        # Scroll to bottom - use a longer delay to ensure layout is updated
        QTimer.singleShot(200, self.scroll_to_bottom)
        
        # Update parent data
        if role == "assistant" and self.parent().filename and not delete_last_message:
            self.parent().other_data["chat_history"] = self.chat_history
    
    def scroll_to_bottom(self):
        """Scroll chat area to the bottom"""
        scrollbar = self.chat_scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Force update to ensure scrolling takes effect
        QApplication.processEvents()
    
    def start_generation(self):
        """Start generating a response"""
        if self.streaming:
            return

        user_message = self.message_input.toPlainText().strip()
        if not user_message:
            return

        # Add user message to chat history first
        self.add_message("user", user_message)
        self.message_input.clear()

        # Reset input box size to initial height
        self.message_input.setMinimumHeight(24)
        self.message_input.setMaximumHeight(24)

        # Start generation process
        self.streaming = True
        self.set_components_enabled(False)

        # Create loading message
        self.add_loading_message()

        # Reset stream handler
        self.stream_handler.reset()

        # Start streaming in a separate thread
        self.stream_thread = threading.Thread(
            target=self.stream_generation,
            args=(self.api_address.text(), self.api_key.text())
        )
        self.stream_thread.start()

    def add_loading_message(self):
        """Add a loading message that will be replaced with the actual response"""
        # Remove stretch
        while self.chat_messages_layout.count() > 0:
            item = self.chat_messages_layout.itemAt(self.chat_messages_layout.count()-1)
            if item and item.spacerItem():
                self.chat_messages_layout.removeItem(item)
                break
                
        # Create a loading message widget
        self.loading_message = QFrame(self.chat_container)
        loading_layout = QVBoxLayout(self.loading_message)
        loading_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create bubble with smooth corners
        bubble = QWidget(self.loading_message)
        bubble.setObjectName("messageBubble")
        bubble.setStyleSheet(ChatMessageStyle.get_bubble_style(is_user=False))

        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 12, 12, 12)
        
        # Add header
        header_layout = QHBoxLayout()
        role_label = QLabel(self.tr("Assistant"))
        role_label.setStyleSheet(ChatMessageStyle.get_role_label_style())

        # Add copy button (disabled during loading)
        copy_btn = QPushButton()
        copy_btn.setIcon(QIcon(set_icon_path("copy")))
        copy_btn.setFixedSize(*ICON_SIZE_SMALL)
        copy_btn.setEnabled(False)
        copy_btn.setStyleSheet(ChatMessageStyle.get_button_style())

        header_layout.addWidget(role_label)
        header_layout.addWidget(copy_btn)
        header_layout.addStretch()

        bubble_layout.addLayout(header_layout)
        
        # Add loading text
        self.loading_text = QLabel(self.tr("Thinking..."))
        self.loading_text.setStyleSheet(ChatMessageStyle.get_content_label_style(is_error=False))
        bubble_layout.addWidget(self.loading_text)

        # Set maximum width for the bubble
        bubble.setMaximumWidth(2000)
        loading_layout.addWidget(bubble)
        loading_layout.setAlignment(Qt.AlignLeft)

        # Store bubble reference for later updates
        self.loading_message.bubble = bubble
        
        # Add to chat layout
        self.chat_messages_layout.addWidget(self.loading_message)
        
        # Add stretch back
        self.chat_messages_layout.addStretch()
        
        # Start loading animation
        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self.update_loading_animation)
        self.loading_timer.start(int(ANIMATION_DURATION[:-2]))
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def update_loading_animation(self):
        """Update the loading animation dots"""
        if hasattr(self, 'loading_text') and self.loading_text:
            try:
                self.loading_dots = (self.loading_dots + 1) % 4
                dots = "." * self.loading_dots
                self.loading_text.setText(f"Thinking{dots}")
            except RuntimeError:
                # The QLabel has been deleted, stop the timer
                if self.loading_timer and self.loading_timer.isActive():
                    self.loading_timer.stop()
                self.loading_text = None
    
    def update_output(self, text):
        """Update the output text with streaming content"""
        if self.loading_message:
            # Stop the loading animation
            if self.loading_timer and self.loading_timer.isActive():
                self.loading_timer.stop()
            
            # If this is the first update, replace the loading text with a QLabel for content
            if not hasattr(self.loading_message, 'content_label'):
                # Remove the loading text
                if hasattr(self, 'loading_text') and self.loading_text:
                    self.loading_text.setParent(None)
                    self.loading_text.deleteLater()
                    self.loading_text = None
                
                # Create content label
                self.loading_message.content_label = QLabel("")
                self.loading_message.content_label.setWordWrap(True)
                self.loading_message.content_label.setTextFormat(Qt.PlainText)
                self.loading_message.content_label.setStyleSheet(ChatMessageStyle.get_content_label_style())
                
                # Set minimum and maximum width for proper wrapping
                self.loading_message.content_label.setMinimumWidth(100)
                self.loading_message.content_label.setMaximumWidth(1999)

                # Add to bubble layout
                self.loading_message.bubble.layout().addWidget(self.loading_message.content_label)
            
            # Update the content
            current_text = self.loading_message.content_label.text()
            self.loading_message.content_label.setText(current_text + text)
            
            # Scroll chat area to bottom
            self.scroll_to_bottom()
    
    def on_stream_finished(self, success):
        """Handle completion of streaming"""
        # Stop the loading timer if it's still active
        if self.loading_timer and self.loading_timer.isActive():
            self.loading_timer.stop()
        
        if success and self.loading_message:
            # Get the final text
            final_text = ""
            if hasattr(self.loading_message, 'content_label'):
                final_text = self.loading_message.content_label.text()
            
            # Remove the loading message
            self.loading_message.setParent(None)
            self.loading_message.deleteLater()
            self.loading_message = None
            
            # Add the final message
            self.add_message("assistant", final_text)

            # Set dirty flag
            if self.parent().image_list:
                self.parent().set_dirty()
        
        # Reset streaming state
        self.streaming = False
        self.set_components_enabled(True)
    
    def handle_loading_state(self, is_loading):
        """Handle loading state changes"""
        if is_loading:
            # Disable UI components during loading
            self.set_components_enabled(False)
        else:
            # Re-enable UI components after loading
            self.set_components_enabled(True)
    
    def toggle_api_key_visibility(self):
        """Toggle visibility of API key"""
        if self.api_key.echoMode() == QLineEdit.Password:
            self.api_key.setEchoMode(QLineEdit.Normal)
            self.toggle_visibility_btn.setIcon(QIcon(set_icon_path("eye")))
        else:
            self.api_key.setEchoMode(QLineEdit.Password)
            self.toggle_visibility_btn.setIcon(QIcon(set_icon_path("eye-off")))
    
    def link_previous_image(self):
        """Navigate to previous image and load its chat history"""
        try:
            if self.parent().image_list:
                current_index = self.parent().image_list.index(self.parent().filename)
                if current_index > 0:
                    prev_index = current_index - 1
                    prev_file = self.parent().image_list[prev_index]
                    self.parent().load_file(prev_file)
                    
                    # Force UI update
                    QApplication.processEvents()
                    
                    # Update image and chat
                    self.update_image_preview()
                    self.load_chat_for_current_image()
                    
                    # Update import buttons visibility
                    self.update_import_buttons_visibility()
        except Exception as e:
            logger.error(f"Error navigating to previous image: {e}")
    
    def link_next_image(self):
        """Navigate to next image and load its chat history"""
        try:
            if self.parent().image_list:
                current_index = self.parent().image_list.index(self.parent().filename)
                if current_index < len(self.parent().image_list) - 1:
                    next_index = current_index + 1
                    next_file = self.parent().image_list[next_index]
                    self.parent().load_file(next_file)
                    
                    # Force UI update
                    QApplication.processEvents()
                    
                    # Update image and chat
                    self.update_image_preview()
                    self.load_chat_for_current_image()
                    
                    # Update import buttons visibility
                    self.update_import_buttons_visibility()
        except Exception as e:
            logger.error(f"Error navigating to next image: {e}")
    
    def load_chat_for_current_image(self):
        """Load chat history for the current image"""
        try:
            # Clear current chat messages
            while self.chat_messages_layout.count() > 0:
                item = self.chat_messages_layout.takeAt(0)
                if item and item.widget():
                    widget = item.widget()
                    widget.setParent(None)
                    widget.deleteLater()
                elif item:
                    self.chat_messages_layout.removeItem(item)

            # Reset chat history
            self.chat_history = []

            # Add stretch
            self.chat_messages_layout.addStretch()

            # Process the UI events to prevent freezing
            QApplication.processEvents()

            # Load data for current image
            self.load_initial_data()

            if "chat_history" in self.parent().other_data:
                logger.debug(f"Loaded chat history with {len(self.parent().other_data['chat_history'])} messages")
        except Exception as e:
            logger.error(f"Error loading chat for current image: {e}")

    def eventFilter(self, obj, event):
        """Event filter for handling Enter key in message input and preventing it in settings fields"""
        if obj == self.message_input and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Return and event.modifiers() & Qt.ControlModifier:
                # Ctrl+Enter sends the message
                self.start_generation()
                return True
            elif event.key() == Qt.Key_Return and not event.modifiers() & Qt.ControlModifier:
                # Enter without Ctrl adds a new line
                return False
        # Prevent Enter key from triggering buttons when in settings fields
        elif obj in [self.api_address, self.model_name, self.api_key] and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                # Consume the Enter key event to prevent it from triggering buttons
                return True
        return super().eventFilter(obj, event)
    
    def stream_generation(self, api_address, api_key):
        """Generate streaming response from the API"""
        try:
            # Signal loading state
            self.stream_handler.start_loading()
            
            model_name = self.model_name.text()
            
            # Prepare messages
            messages = []
            for msg in self.chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Create client
            client = OpenAI(
                base_url=api_address,
                api_key=api_key
            )
            
            # Get image data if available
            image_data = None
            if self.parent().filename:
                try:
                    with open(self.parent().filename, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                except Exception as e:
                    logger.error(f"Error reading image file: {e}")

            # Add image to the message if available
            if image_data:
                # Find the last user message
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i]["role"] == "user":
                        # Add image to content
                        messages[i]["content"] = [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": messages[i]["content"]
                            }
                        ]
                        break
            
            # Make API call with streaming
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True
            )

            # Process streaming response
            for chunk in response:
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    self.stream_handler.append_text(content)
            
            # Signal completion
            self.stream_handler.finished.emit(True)
            
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            # Signal error through the handler instead of directly updating UI
            self.stream_handler.report_error(str(e))
            
            # Signal completion with failure
            self.stream_handler.finished.emit(False)
        
        finally:
            # Signal loading state ended
            self.stream_handler.stop_loading()
    
    def set_components_enabled(self, enabled):
        """Enable or disable UI components during streaming"""
        self.message_input.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        self.prev_image_btn.setEnabled(enabled)
        self.next_image_btn.setEnabled(enabled)
        
        # Also disable provider switching during streaming
        for provider in PROVIDER_CONFIGS.keys():
            if hasattr(self, f"{provider}_btn"):
                getattr(self, f"{provider}_btn").setEnabled(enabled)
        
        # Update cursor for input
        if enabled:
            self.message_input.setCursor(Qt.IBeamCursor)
        else:
            self.message_input.setCursor(Qt.ForbiddenCursor)

    def handle_error(self, error_message):
        """Handle error messages from the streaming thread"""
        # Stop the loading timer if it's still active
        if self.loading_timer and self.loading_timer.isActive():
            self.loading_timer.stop()
        
        if self.loading_message:
            # Remove the loading message
            self.loading_message.setParent(None)
            self.loading_message.deleteLater()
            self.loading_message = None

            # Add error message but don't add it to chat history
            self.add_message("assistant", f"Error: {error_message}", delete_last_message=True)

    def resizeEvent(self, event):
        """Handle window resize event, update message layout constraints"""
        super().resizeEvent(event)

        for i in range(self.chat_messages_layout.count()):
            item = self.chat_messages_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if hasattr(widget, 'update_width_constraint'):
                    widget.update_width_constraint()

        self.chat_container.updateGeometry()
        QApplication.processEvents()
        QTimer.singleShot(100, self.scroll_to_bottom)
