import base64
import threading
from openai import OpenAI

from PyQt5.QtCore import QTimer, Qt, QSize, QPoint, QEvent
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
    QComboBox,
    QTabWidget,
    QSlider,
    QSpinBox,
    QMessageBox,
)
from PyQt5.QtGui import QCursor, QIcon, QPixmap, QColor, QTextCursor, QTextCharFormat

from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.general import open_url
from anylabeling.views.labeling.chatbot import *


class ChatbotDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.resize(*DEFAULT_WINDOW_SIZE)
        self.setWindowIcon(QIcon(set_icon_path("chat")))

        dialog_style = ChatbotDialogStyle.get_dialog_style()
        menu_style = ChatbotDialogStyle.get_menu_style()
        combined_style = dialog_style + menu_style
        self.setStyleSheet(combined_style)

        # Initialize
        self.chat_history = []
        self.attach_image_to_chat = False

        # Create all tooltips first to ensure they exist before any event filtering
        self.temperature_tooltip = CustomTooltip(
            title="Recommended values:",
            value_pairs=[
                ("Coding / Math", "0"),
                ("Data Cleaning / Data Analysis", "1"),
                ("General Conversation", "1.3"),
                ("Translation", "1.3"),
                ("Creative Writing / Poetry", "1.5")
            ]
        )
        self.refresh_models_tooltip = CustomTooltip(
            title="Refresh available models",
        )

        pixmap = QPixmap(set_icon_path("click"))
        scaled_pixmap = pixmap.scaled(*ICON_SIZE_SMALL, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.click_cursor = QCursor(scaled_pixmap)

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
        for provider in PROVIDER_CONFIGS.keys():
            btn = QPushButton(self.tr(provider.capitalize()))
            btn.setIcon(QIcon(set_icon_path(provider)))
            btn.setCheckable(True)
            btn.setFixedHeight(40)
            btn.setIconSize(QSize(*ICON_SIZE_SMALL))
            btn.setStyleSheet(ChatbotDialogStyle.get_provider_button_style())
            btn.clicked.connect(lambda checked, p=provider: self.switch_provider(p) if checked else None)
            provider_group.addButton(btn)
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
        #################################
        self.middle_widget = QWidget()
        self.middle_widget.setStyleSheet(ChatbotDialogStyle.get_middle_widget_style())
        middle_panel = QVBoxLayout(self.middle_widget)
        middle_panel.setContentsMargins(0, 0, 0, 0)
        middle_panel.setSpacing(0)

        # Chat area
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
        self.message_input.setPlaceholderText(self.tr("Type something, add @image to include an image"))
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

        # Connect textChanged to dynamically resize input and highlight @image tag
        self.message_input.textChanged.connect(self.on_text_changed)
        self.message_input.installEventFilter(self)  # For Enter key handling

        # Initialize the input size
        self.resize_input()

        # Create a container for the input and send button
        input_with_button = QWidget()
        input_with_button_layout = QVBoxLayout(input_with_button)
        input_with_button_layout.setContentsMargins(0, 0, 0, 0)
        input_with_button_layout.setSpacing(0)

        # Add the message input to the layout (at the top)
        input_with_button_layout.addWidget(self.message_input, 1)

        # Create a button bar container for the bottom
        button_bar = QWidget()
        button_bar_layout = QHBoxLayout(button_bar)
        button_bar_layout.setContentsMargins(0, 4, 0, 0)  # Add a small top margin
        button_bar_layout.setSpacing(8)  # Spacing between buttons

        # Add clear context button (left side)
        self.clear_context_btn = QPushButton()
        self.clear_context_btn.setIcon(QIcon(set_icon_path("trash")))
        self.clear_context_btn.setIconSize(QSize(*ICON_SIZE_SMALL))
        self.clear_context_btn.setStyleSheet(ChatbotDialogStyle.get_send_button_style())
        self.clear_context_btn.setCursor(Qt.PointingHandCursor)
        self.clear_context_btn.setFixedSize(*ICON_SIZE_SMALL)
        self.clear_context_btn.setToolTip(self.tr("Clear Conversation"))
        self.clear_context_btn.clicked.connect(self.clear_conversation)

        # Add buttons to layout
        button_bar_layout.addWidget(self.clear_context_btn, 0, Qt.AlignBottom)
        button_bar_layout.addStretch(1)  # Push buttons to left and right edges

        # Create the send button (right side)
        self.send_btn = QPushButton()
        self.send_btn.setIcon(QIcon(set_icon_path("send")))
        self.send_btn.setIconSize(QSize(*ICON_SIZE_SMALL))
        self.send_btn.setStyleSheet(ChatbotDialogStyle.get_send_button_style())
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setFixedSize(*ICON_SIZE_SMALL)
        self.send_btn.clicked.connect(self.start_generation)
        self.send_btn.setEnabled(False)

        # Add send button to layout
        button_bar_layout.addWidget(self.send_btn, 0, Qt.AlignBottom)
        input_with_button_layout.addWidget(button_bar, 0, Qt.AlignBottom)
        input_frame_layout.addWidget(input_with_button)
        input_layout.addWidget(input_frame)

        # Add the chat container and input container to the middle panel
        middle_panel.addWidget(chat_container, CHAT_PANEL_PERCENTAGE)
        middle_panel.addWidget(input_container, INPUT_PANEL_PERCENTAGE)

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
        self.prev_image_btn.clicked.connect(lambda: self.navigate_image(direction="prev"))

        self.next_image_btn = QPushButton()
        self.next_image_btn.setIcon(QIcon(set_icon_path("arrow-right")))
        self.next_image_btn.setFixedSize(*ICON_SIZE_NORMAL)
        self.next_image_btn.setStyleSheet(ChatbotDialogStyle.get_navigation_btn_style())
        self.next_image_btn.setToolTip(self.tr("Next Image"))
        self.next_image_btn.setCursor(Qt.PointingHandCursor)
        self.next_image_btn.clicked.connect(lambda: self.navigate_image(direction="next"))

        nav_layout.addWidget(self.prev_image_btn)
        nav_layout.addStretch()

        # Add image and video buttons for importing media
        import_media_btn_modes = ["image", "folder", "video"]
        import_media_btn_names = ["open_image_file_btn", "open_image_folder_btn", "open_video_btn"]
        import_media_btn_tooltips = ["Open Image File", "Open Image Folder", "Open Video File"]
        for btn_mode, btn_name, btn_tooltip in zip(
            import_media_btn_modes, import_media_btn_names, import_media_btn_tooltips
        ):
            btn = QPushButton()
            btn.setIcon(QIcon(set_icon_path(btn_mode)))
            btn.setFixedSize(*ICON_SIZE_NORMAL)
            btn.setStyleSheet(ChatbotDialogStyle.get_navigation_btn_style())
            btn.setToolTip(self.tr(btn_tooltip))
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, mode=btn_mode: 
                               self.open_image_folder_or_video_file(mode=mode))
            nav_layout.addWidget(btn)
            setattr(self, btn_name, btn)  # Store reference as class attribute
        nav_layout.addStretch()

        nav_layout.addWidget(self.next_image_btn)
        image_layout.addLayout(nav_layout)

        # Settings panel with tabs
        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(0)
        
        # Create tab widget
        self.settings_tabs = QTabWidget()
        self.settings_tabs.setStyleSheet(ChatbotDialogStyle.get_tab_widget_style())
        self.settings_tabs.setUsesScrollButtons(False)
        self.settings_tabs.setDocumentMode(True)
        self.settings_tabs.setElideMode(Qt.ElideNone)
        
        # First tab - API Settings
        api_settings_tab = QWidget()
        api_settings_layout = QVBoxLayout(api_settings_tab)
        api_settings_layout.setContentsMargins(24, 24, 24, 24)
        api_settings_layout.setSpacing(12)
        
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
        api_help_btn.setCursor(self.click_cursor)
        api_help_btn.clicked.connect(lambda: open_url(PROVIDER_CONFIGS[DEFAULT_PROVIDER]["api_docs_url"]))

        label_help_layout.addWidget(api_address_label)
        label_help_layout.addWidget(api_help_btn)
        label_help_layout.addStretch()

        api_address_container.addWidget(label_with_help)
        api_address_container.addStretch()
        api_settings_layout.addLayout(api_address_container)

        self.api_address = QLineEdit(PROVIDER_CONFIGS[DEFAULT_PROVIDER]["api_address"])
        self.api_address.setStyleSheet(ChatbotDialogStyle.get_settings_edit_style())
        self.api_address.installEventFilter(self)
        api_settings_layout.addWidget(self.api_address)

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
        api_key_help_btn.setCursor(self.click_cursor)
        api_key_help_btn.clicked.connect(lambda: open_url(PROVIDER_CONFIGS[DEFAULT_PROVIDER]["api_key_url"]))

        key_label_help_layout.addWidget(api_key_label)
        key_label_help_layout.addWidget(api_key_help_btn)
        key_label_help_layout.addStretch()

        api_key_container.addWidget(key_label_with_help)
        api_key_container.addStretch()
        api_settings_layout.addLayout(api_key_container)

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
        api_settings_layout.addLayout(api_key_container)

        # Model Name with help icon
        model_name_container = QHBoxLayout()
        model_name_label = QLabel(self.tr("Model Name"))
        model_name_label.setStyleSheet(ChatbotDialogStyle.get_settings_label_style())

        # Create a container for label and buttons
        model_label_with_help = QWidget()
        model_label_help_layout = QHBoxLayout(model_label_with_help)
        model_label_help_layout.setContentsMargins(0, 0, 0, 0)
        model_label_help_layout.setSpacing(4)

        model_help_btn = QPushButton()
        model_help_btn.setObjectName("model_help_btn")
        model_help_btn.setIcon(QIcon(set_icon_path("help-circle")))
        model_help_btn.setFixedSize(*ICON_SIZE_SMALL)
        model_help_btn.setStyleSheet(ChatbotDialogStyle.get_help_btn_style())
        model_help_btn.setCursor(self.click_cursor)
        model_help_btn.clicked.connect(lambda: open_url(
            PROVIDER_CONFIGS[DEFAULT_PROVIDER]["model_docs_url"]))

        self.refresh_models_btn = QPushButton()
        self.refresh_models_btn.setIcon(QIcon(set_icon_path("refresh")))
        self.refresh_models_btn.setFixedSize(*ICON_SIZE_SMALL)
        self.refresh_models_btn.setStyleSheet(ChatbotDialogStyle.get_help_btn_style())
        self.refresh_models_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_models_btn.clicked.connect(lambda: self.fetch_models(log_errors=True))
        self.refresh_models_btn.installEventFilter(self)
        self.refresh_models_btn.setObjectName("refresh_btn")

        model_label_help_layout.addWidget(model_name_label)
        model_label_help_layout.addWidget(model_help_btn)
        model_label_help_layout.addWidget(self.refresh_models_btn)
        model_name_container.addWidget(model_label_with_help)
        model_name_container.addStretch()
        api_settings_layout.addLayout(model_name_container)

        # Create ComboBox for model selection
        self.model_name = QComboBox()
        self.model_name.setStyleSheet(ChatbotDialogStyle.get_combobox_style(set_icon_path("chevron-down_blue")))
        self.model_name.setMinimumHeight(40)
        api_settings_layout.addWidget(self.model_name)
        api_settings_layout.addStretch()
        
        # Second tab - Model Parameters
        model_params_tab = QWidget()
        model_params_layout = QVBoxLayout(model_params_tab)
        model_params_layout.setContentsMargins(24, 24, 24, 24)
        model_params_layout.setSpacing(16)
        
        # System prompt section
        system_prompt_label = QLabel(self.tr("System instruction"))
        system_prompt_label.setStyleSheet(ChatbotDialogStyle.get_settings_label_style())
        model_params_layout.addWidget(system_prompt_label)
        
        # System prompt input
        system_prompt_container = QHBoxLayout()
        self.system_prompt_input = QLineEdit()
        self.system_prompt_input.setStyleSheet(ChatbotDialogStyle.get_settings_edit_style())
        system_prompt_container.addWidget(self.system_prompt_input)
        model_params_layout.addLayout(system_prompt_container)
        
        # Temperature parameter with info icon
        temp_header = QHBoxLayout()
        temp_label = QLabel(self.tr("Temperature"))
        temp_label.setStyleSheet(ChatbotDialogStyle.get_settings_label_style())
        
        temp_info_btn = QPushButton()
        temp_info_btn.setIcon(QIcon(set_icon_path("help-circle")))
        temp_info_btn.setFixedSize(*ICON_SIZE_SMALL)
        temp_info_btn.setStyleSheet(ChatbotDialogStyle.get_help_btn_style())
        temp_info_btn.setCursor(Qt.PointingHandCursor)
        temp_info_btn.installEventFilter(self)
        temp_info_btn.setObjectName("temperature_btn")

        self.temp_value = QLabel(f"{DEFAULT_TEMPERATURE_VALUE/10:.1f}")
        self.temp_value.setStyleSheet(ChatbotDialogStyle.get_settings_label_style())
        self.temp_value.setAlignment(Qt.AlignRight)

        temp_header.addWidget(temp_label)
        temp_header.addWidget(temp_info_btn)
        temp_header.addStretch()
        temp_header.addWidget(self.temp_value)
        model_params_layout.addLayout(temp_header)

        # Temperature slider
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(0)
        self.temp_slider.setMaximum(20)  # 0.0 to 2.0 with step of 0.1
        self.temp_slider.setValue(DEFAULT_TEMPERATURE_VALUE)
        self.temp_slider.setStyleSheet(ChatbotDialogStyle.get_slider_style())
        self.temp_slider.valueChanged.connect(lambda v: self.temp_value.setText(f"{v/10:.1f}"))
        model_params_layout.addWidget(self.temp_slider)

        # Temperature labels
        temp_labels_layout = QHBoxLayout()

        precise_label = QLabel(self.tr("Precise"))
        precise_label.setStyleSheet(ChatbotDialogStyle.get_temperature_label_style())
        precise_label.setAlignment(Qt.AlignLeft)

        neutral_label = QLabel(self.tr("Neutral"))
        neutral_label.setStyleSheet(ChatbotDialogStyle.get_temperature_label_style())
        neutral_label.setAlignment(Qt.AlignCenter)

        creative_label = QLabel(self.tr("Creative"))
        creative_label.setStyleSheet(ChatbotDialogStyle.get_temperature_label_style())
        creative_label.setAlignment(Qt.AlignRight)

        temp_labels_layout.addWidget(precise_label)
        temp_labels_layout.addWidget(neutral_label)
        temp_labels_layout.addWidget(creative_label)
        
        model_params_layout.addLayout(temp_labels_layout)
        model_params_layout.addSpacing(16)

        # Maximum output length
        max_length_label = QLabel(self.tr("Max output tokens"))
        max_length_label.setStyleSheet(ChatbotDialogStyle.get_settings_label_style())
        model_params_layout.addWidget(max_length_label)

        self.max_length_input = QSpinBox()
        self.max_length_input.setMinimum(0)
        self.max_length_input.setMaximum(9999999)
        self.max_length_input.setButtonSymbols(QSpinBox.UpDownArrows)
        self.max_length_input.setStyleSheet(ChatbotDialogStyle.get_spinbox_style())
        self.max_length_input.setFixedHeight(40)
        model_params_layout.addWidget(self.max_length_input)

        # Add stretch to push everything to the top
        model_params_layout.addStretch()

        # Add tabs to tab widget
        self.settings_tabs.addTab(api_settings_tab, self.tr("Backend"))
        self.settings_tabs.addTab(model_params_tab, self.tr("Generation"))
        self.settings_tabs.tabBar().setExpanding(True)
        self.settings_tabs.tabBar().setStyleSheet(ChatbotDialogStyle.get_settings_tabs_style())

        # Add tab widget to settings layout
        settings_layout.addWidget(self.settings_tabs)

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
        
        # Fetch available models
        self.fetch_models()

        self.message_input.setFocus()

    def switch_provider(self, provider):
        """Switch between different model providers"""
        if provider in PROVIDER_CONFIGS:
            # set api address and key
            self.api_address.setText(PROVIDER_CONFIGS[provider]["api_address"])
            self.api_key.setText(PROVIDER_CONFIGS[provider]["api_key"])

            # clear model dropdown and fetch models
            self.model_name.clear()
            self.fetch_models()

            # update help button urls
            button_url_mapping = [
                {"button_name": "api_help_btn", "url_key": "api_docs_url"},
                {"button_name": "model_help_btn", "url_key": "model_docs_url"},
                {"button_name": "api_key_help_btn", "url_key": "api_key_url"}
            ]
            for mapping in button_url_mapping:
                button_name, url_key = mapping["button_name"], mapping["url_key"]
                button = self.findChild(QPushButton, button_name)
                url = PROVIDER_CONFIGS[provider][url_key]
                if button:
                    if url:
                        button.setVisible(True)
                        button.clicked.disconnect()
                        button.clicked.connect(lambda checked=False, u=url: open_url(u))
                    else:
                        button.setVisible(False)

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
                    QTimer.singleShot(int(ANIMATION_DURATION[:-2]), self.update_image_preview)
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
            QTimer.singleShot(int(ANIMATION_DURATION[:-2]), self.update_image_preview)

        # Update visibility of import buttons based on whether files are loaded
        self.update_import_buttons_visibility()
    
    def update_import_buttons_visibility(self):
        """Update visibility of import buttons based on whether files are loaded"""
        has_images = bool(self.parent().image_list)
        # Show navigation buttons only when images are loaded
        self.prev_image_btn.setVisible(has_images)
        self.next_image_btn.setVisible(has_images)
    
    def open_image_folder_or_video_file(self, mode="image"):
        """Open an image file or image folder or a video file"""
        if mode == "image":
            self.parent().open_file()
            if self.parent().filename:
                self.load_chat_for_current_image()
                self.update_import_buttons_visibility()
        else:
            if mode == "video":
                self.parent().open_video_file()
            else:
                self.parent().open_folder_dialog()
            if self.parent().image_list:
                self.load_chat_for_current_image()
                self.update_import_buttons_visibility()

    def add_message(self, role, content, delete_last_message=False):
        """Add a new message to the chat area"""
        # Remove the stretch item if it exists
        while self.chat_messages_layout.count() > 0:
            item = self.chat_messages_layout.itemAt(self.chat_messages_layout.count()-1)
            if item and item.spacerItem():
                self.chat_messages_layout.removeItem(item)
                break

        # Check for special token in user messages
        if role == "user":
            if "@image" in content:
                self.attach_image_to_chat = True
            content = content.replace("@image", "", 1).strip()

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
        QTimer.singleShot(int(ANIMATION_DURATION[:-2]), self.scroll_to_bottom)
        
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

        # Clear input and reset to plain text mode
        self.message_input.clear()
        self.message_input.setPlainText("")  # Ensure we're back to plain text mode

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

        header_layout.addWidget(role_label)
        header_layout.addStretch()

        bubble_layout.addLayout(header_layout)
        
        # Add loading text
        self.loading_text = QLabel(self.tr("Generating..."))
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
                self.loading_text.setText(f"Generating{dots}")
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
                self.loading_message.content_label.setStyleSheet(
                    ChatMessageStyle.get_content_label_style(is_error=False)
                )
                
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
            
            # Store reference to loading message for removal
            loading_message_to_remove = self.loading_message
            self.loading_message = None
            
            # Add the final message first
            self.add_message("assistant", final_text)
            
            # Then remove the loading message after adding the final message
            loading_message_to_remove.setParent(None)
            loading_message_to_remove.deleteLater()

            # Set dirty flag
            if self.parent().image_list:
                self.parent().set_dirty()
        
        # Reset streaming state
        self.streaming = False
        self.set_components_enabled(True)
        
        # Auto focus on message input after generation
        self.message_input.setFocus()
    
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
    
    def navigate_image(self, direction="next"):
        """Navigate to previous or next image and load its chat history

        Args:
            direction (str): Direction to navigate, either "next" or "prev"
        """
        try:
            if self.parent().image_list:
                current_index = self.parent().image_list.index(self.parent().filename)

                if direction == "prev" and current_index > 0:
                    new_index = current_index - 1
                elif direction == "next" and current_index < len(self.parent().image_list) - 1:
                    new_index = current_index + 1
                else:
                    return

                new_file = self.parent().image_list[new_index]
                self.parent().load_file(new_file)

                # Force UI and data updates
                QApplication.processEvents()
                self.update_image_preview()
                self.load_chat_for_current_image()
                self.update_import_buttons_visibility()
        except Exception as e:
            logger.error(f"Error navigating to {direction} image: {e}")

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

        except Exception as e:
            logger.error(f"Error loading chat for current image: {e}")

    def eventFilter(self, obj, event):
        """Event filter for handling events"""
        # Tooltip handler for multiple buttons
        tooltip_buttons = {
            "temperature_btn": self.temperature_tooltip,
            "refresh_btn": self.refresh_models_tooltip,
        }
        for btn_name, tooltip in tooltip_buttons.items():
            if obj.objectName() == btn_name:
                if event.type() == QEvent.Enter:
                    button_pos = obj.mapToGlobal(QPoint(0, 0))
                    tooltip.move(button_pos)
                    tooltip.adjustSize()
                    tooltip_width = tooltip.width()
                    tooltip_height = tooltip.height()
                    target_x = button_pos.x() - tooltip_width + 5
                    target_y = button_pos.y() - tooltip_height - 5
                    tooltip.move(target_x, target_y)
                    tooltip.show()
                    return True
                elif event.type() == QEvent.Leave:
                    tooltip.hide()
                    return True

        if obj == self.message_input and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Return and event.modifiers() & Qt.ControlModifier:
                self.start_generation()
                return True
            elif event.key() == Qt.Key_Return and not event.modifiers() & Qt.ControlModifier:
                # Enter without Ctrl adds a new line
                return False
        # Prevent Enter key from triggering buttons when in settings fields
        elif hasattr(self, 'api_address') and hasattr(self, 'model_name') and hasattr(self, 'api_key'):
            if obj in [self.api_address, self.model_name, self.api_key] and event.type() == event.KeyPress:
                if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
                    return True
        return super().eventFilter(obj, event)

    def stream_generation(self, api_address, api_key):
        """Generate streaming response from the API"""
        try:
            # Signal loading state
            self.stream_handler.start_loading()

            if self.model_name.currentText():
                model_name = self.model_name.currentText()
            else:
                raise ValueError("No model selected. Check API address/key, then refresh the models list.")

            # Get temperature value from slider
            temperature = self.temp_slider.value() / 10.0
            
            # Get system prompt if provided
            system_prompt = self.system_prompt_input.text().strip()
            
            # Get max tokens if provided
            max_tokens = None
            if hasattr(self, 'max_length_input') and self.max_length_input.text().strip():
                try:
                    max_tokens = int(self.max_length_input.text().strip())
                except ValueError:
                    pass

            # Prepare messages
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            # Add conversation history
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

            # Add image to the message if available and requested
            if image_data and self.attach_image_to_chat:
                # Find the last user message
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i]["role"] == "user":
                        user_content = messages[i]["content"]
                        messages[i]["content"] = [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                            },
                            {
                                "type": "text",
                                "text": user_content
                            }
                        ]
                        break

            # Prepare API call parameters
            api_params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "stream": True  # Don't change this to False, it will break the streaming
            }

            # Add max_tokens if provided
            if max_tokens:
                api_params["max_tokens"] = max_tokens
            
            # Make API call with streaming
            response = client.chat.completions.create(**api_params)

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
        self.send_btn.setEnabled(False)
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

    def resubmit_edited_message(self, message_widget, new_content):
        """Handle resubmission of an edited message"""
        # Find the index of the message in the chat history
        message_widgets = []
        for i in range(self.chat_messages_layout.count()):
            item = self.chat_messages_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), ChatMessage):
                message_widgets.append(item.widget())
        
        if message_widget in message_widgets:
            message_index = message_widgets.index(message_widget)
            
            # Clear all messages after and including this one
            self.clear_messages_after(message_index)
            
            # Set the edited text to the message input
            self.message_input.setPlainText(new_content)
            
            # Trigger generation with the new content
            self.start_generation()
    
    def clear_messages_after(self, index):
        """Clear all messages after and including the specified index"""
        # Find all message widgets
        message_widgets = []
        for i in range(self.chat_messages_layout.count()):
            item = self.chat_messages_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), ChatMessage):
                message_widgets.append((i, item.widget()))
        
        # Identify which items to remove (in reverse order to avoid index issues)
        to_remove = []
        for i, (layout_index, widget) in enumerate(message_widgets):
            if i >= index:
                to_remove.append((layout_index, widget))
        
        # Remove widgets in reverse order
        for layout_index, widget in reversed(to_remove):
            # Remove from layout
            self.chat_messages_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        
        # Update chat history to match
        if len(to_remove) > 0:
            self.chat_history = self.chat_history[:index]
        
        # Make sure we still have a stretch at the end
        # First, check if there's already a stretch
        has_stretch = False
        for i in range(self.chat_messages_layout.count()):
            item = self.chat_messages_layout.itemAt(i)
            if item and item.spacerItem():
                has_stretch = True
                break
        
        # If no stretch, add one
        if not has_stretch:
            self.chat_messages_layout.addStretch()
        
        # Force update layout
        QApplication.processEvents()

    def fetch_models(self, log_errors=False):
        """Fetch available models from the API and populate the dropdown

        Args:
            log_errors: Whether to log errors that occur during model fetching
        """
        try:
            self.refresh_models_btn.setEnabled(False)

            # Start in a separate thread to avoid blocking UI
            threading.Thread(target=self._fetch_models_thread, kwargs={"log_errors": log_errors}).start()

        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            self.refresh_models_btn.setEnabled(True)

    def _fetch_models_thread(self, log_errors=False):
        """Thread function to fetch models"""
        try:
            models_list = get_models_list(self.api_address.text(), self.api_key.text())

            # Update UI in the main thread
            QApplication.instance().postEvent(self, UpdateModelsEvent(models_list))

        except Exception as e:
            if log_errors:
                logger.error(f"Error in model fetch thread: {e}")
            # Re-enable refresh button in main thread
            QApplication.instance().postEvent(self, EnableRefreshButtonEvent())
    
    def event(self, event):
        """Handle custom events"""
        if isinstance(event, UpdateModelsEvent):
            # Update the dropdown with fetched models
            self.model_name.clear()
            
            for model_id in event.models:
                self.model_name.addItem(model_id)
            
            # Select first item if available
            if self.model_name.count() > 0:
                self.model_name.setCurrentIndex(0)
            
            # Re-enable refresh button
            self.refresh_models_btn.setEnabled(True)
            return True
            
        elif isinstance(event, EnableRefreshButtonEvent):
            self.refresh_models_btn.setEnabled(True)
            return True
            
        return super().event(event)

    def regenerate_response(self, message_widget):
        """Regenerate the assistant's response"""
        if self.streaming:
            return

        # Find all message widgets
        message_widgets = []
        for i in range(self.chat_messages_layout.count()):
            item = self.chat_messages_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), ChatMessage):
                message_widgets.append((i, item.widget()))

        # Find the index of the message in the list
        message_index = None
        for i, (_, widget) in enumerate(message_widgets):
            if widget == message_widget:
                message_index = i
                break

        if message_index is not None:
            # Remove this assistant message and any messages after it
            self.clear_messages_after(message_index)

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

    def on_text_changed(self):
        """Handle text changes in the message input, resize and highlight @image tag"""
        self.resize_input()
        
        # Update send button state based on whether input is empty
        current_text = self.message_input.toPlainText().strip()
        self.send_btn.setEnabled(bool(current_text))

        # Highlight @image tag
        cursor = self.message_input.textCursor()
        current_position = cursor.position()
        document = self.message_input.document()
        text = document.toPlainText()
        
        # Block signals temporarily to prevent recursive calls
        self.message_input.blockSignals(True)

        # Reset formatting
        cursor.select(QTextCursor.Document)
        format = QTextCharFormat()
        cursor.setCharFormat(format)

        # Find and highlight @image
        tag = "@image"
        start_index = text.find(tag)
        if start_index != -1:
            # Set highlight format
            highlight_format = QTextCharFormat()
            highlight_format.setBackground(QColor("#E3F2FD"))  # Light blue background
            highlight_format.setForeground(QColor("#1976D2"))  # Darker blue text

            # Select and format the tag
            cursor.setPosition(start_index)
            cursor.setPosition(start_index + len(tag), QTextCursor.KeepAnchor)
            cursor.setCharFormat(highlight_format)

            # Restore cursor position
            cursor.setPosition(current_position)
            self.message_input.setTextCursor(cursor)

        # Unblock signals
        self.message_input.blockSignals(False)

    def clear_conversation(self):
        """Clear all chat messages and reset history"""
        # Show confirmation dialog
        confirm_dialog = QMessageBox(self)
        confirm_dialog.setText(self.tr("Are you sure you want to clear the entire conversation?"))
        confirm_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        confirm_dialog.setDefaultButton(QMessageBox.No)
        confirm_dialog.setIcon(QMessageBox.Warning)

        # Show dialog and handle response
        response = confirm_dialog.exec_()
        if response == QMessageBox.Yes:
            # Clear chat messages from the layout
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
            
            # Update parent data if applicable
            if self.parent().filename:
                self.parent().other_data["chat_history"] = self.chat_history
                self.parent().set_dirty()  # Mark as dirty/modified
