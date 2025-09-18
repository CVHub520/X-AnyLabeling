from difflib import SequenceMatcher
from pathlib import Path

from PyQt5.QtWidgets import (
    QLabel,
    QLineEdit,
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon

from anylabeling.views.labeling.chatbot.config import *
from anylabeling.views.labeling.chatbot.utils import load_json, save_json
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.qt import new_icon, new_icon_path


class SearchBar(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Search models")
        self.setFixedHeight(DEFAULT_FIXED_HEIGHT)
        self.setStyleSheet(
            f"""
            QLineEdit {{
                background-color: #d4d4d8;
                border-radius: {BORDER_RADIUS};
                padding: 5px 5px 5px 32px;
                font-size: {FONT_SIZE_SMALL};
            }}
            QLineEdit:focus {{
                border: 3px solid #60A5FA;
            }}
        """
        )

        self.search_icon = QLabel(self)
        self.search_icon.setPixmap(
            QIcon(new_icon("search", "svg")).pixmap(QSize(*ICON_SIZE_SMALL))
        )
        self.search_icon.setFixedSize(self.search_icon.pixmap().size())
        self.search_icon.setStyleSheet("background-color: transparent;")
        self.resizeEvent = self.on_resize

    def on_resize(self, event):
        icon_height = self.search_icon.height()
        y_position = (self.height() - icon_height) // 2
        self.search_icon.move(10, y_position)

        super().resizeEvent(event)


class ModelItem(QFrame):
    clicked = pyqtSignal(str)
    favoriteToggled = pyqtSignal(str, bool)

    def __init__(
        self, model_name, model_data, parent=None, in_favorites_section=False
    ):
        super().__init__(parent)
        self.model_name = model_name
        self.model_data = model_data
        self.is_selected = model_data.get("selected", False)
        self.is_favorite = model_data.get("favorite", False)
        self.in_favorites_section = in_favorites_section

        self.setFixedHeight(DEFAULT_FIXED_HEIGHT)
        self.setFrameShape(QFrame.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)

        self.name_label = QLabel(model_name)
        self.name_label.setStyleSheet(
            f"""
            font-size: {FONT_SIZE_SMALL};
        """
        )
        layout.addWidget(self.name_label)
        layout.addStretch()

        # Checkmark for selected item
        self.check_icon = QLabel()
        if self.is_selected:
            self.check_icon.setPixmap(
                QIcon(new_icon("check", "svg")).pixmap(QSize(*ICON_SIZE_SMALL))
            )
        layout.addWidget(self.check_icon)

        # Favorite star (initially hidden, shows on hover)
        self.star_icon = QPushButton()
        self.star_icon.setFixedSize(*ICON_SIZE_SMALL)
        self.star_icon.setStyleSheet(
            """
            QPushButton {
                border: none;
                background-color: transparent;
            }
        """
        )
        if self.is_favorite:
            self.star_icon.setIcon(QIcon(new_icon("starred", "svg")))
            if self.in_favorites_section:
                self.star_icon.setVisible(False)
        else:
            self.star_icon.setIcon(QIcon(new_icon("star", "svg")))
            self.star_icon.setVisible(False)

        self.star_icon.clicked.connect(self.toggle_favorite)
        layout.addWidget(self.star_icon)

        # Vision icon if applicable
        if model_data.get("vision", False):
            vision_icon = QLabel()
            vision_icon.setPixmap(
                QIcon(new_icon("vision", "svg")).pixmap(
                    QSize(*ICON_SIZE_SMALL)
                )
            )
            layout.addWidget(vision_icon)

        self.setStyleSheet(
            f"""
            ModelItem {{
                background-color: transparent;
                border-radius: 4px;
            }}
            ModelItem:hover {{
                background-color: #d1d0d4;
            }}
        """
        )

    def enterEvent(self, event):
        self.star_icon.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self.in_favorites_section or not self.is_favorite:
            self.star_icon.setVisible(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self.clicked.emit(self.model_name)
        super().mousePressEvent(event)

    def toggle_favorite(self):
        self.is_favorite = not self.is_favorite
        if self.is_favorite:
            self.star_icon.setIcon(QIcon(new_icon("starred", "svg")))
        else:
            self.star_icon.setIcon(QIcon(new_icon("star", "svg")))
        self.favoriteToggled.emit(self.model_name, self.is_favorite)

    def update_selection(self, is_selected):
        self.is_selected = is_selected
        if is_selected:
            self.check_icon.setPixmap(
                QIcon(new_icon("check", "svg")).pixmap(QSize(*ICON_SIZE_SMALL))
            )
            self.setStyleSheet(
                """
                background-color: #d1d0d4;
                border-radius: 4px;
            """
            )
        else:
            self.check_icon.clear()
            self.setStyleSheet(
                """
                background-color: transparent;
                border-radius: 4px;
            """
            )

    def update_favorite(self, is_favorite):
        self.is_favorite = is_favorite
        if is_favorite:
            self.star_icon.setIcon(QIcon(new_icon("starred", "svg")))
        else:
            self.star_icon.setIcon(QIcon(new_icon("star", "svg")))
        self.star_icon.setVisible(is_favorite or self.underMouse())


class ProviderSection(QFrame):
    def __init__(self, provider_name, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        # Provider header
        header = QHBoxLayout()
        icon = QLabel()
        if provider_name == "Favorites":
            icon_name, ext = "star-black", "svg"
        else:
            icon_name, ext = provider_name.lower(), "png"
        icon.setPixmap(
            QIcon(new_icon(icon_name, ext)).pixmap(QSize(*ICON_SIZE_SMALL))
        )
        header.addWidget(icon)

        label = QLabel(provider_name)
        label.setStyleSheet(
            """
            font-family: "sans-serif";
            font-weight: 700;
            font-size: 13px;
            color: black;
        """
        )
        header.addWidget(label)
        header.addStretch()
        layout.addLayout(header)

        # Container for model items
        self.models_container = QVBoxLayout()
        self.models_container.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.models_container)

    def add_model_item(self, model_item):
        self.models_container.addWidget(model_item)


class ModelDropdown(QWidget):
    modelSelected = pyqtSignal(str)
    providerSelected = pyqtSignal(str)

    def __init__(
        self, models_data: dict = {}, current_provider: str = None, parent=None
    ):
        super().__init__(parent)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.resize(360, 500)
        self.current_provider = current_provider

        self.setStyleSheet(
            f"""
            ModelDropdown {{
                background-color: #e3e2e6;
                border-radius: 8px;
            }}
            QScrollBar:vertical {{
                background-color: #fcfcfc;
                width: 10px;
                margin: 16px 0 16px 0;
            }}
            QScrollBar::handle:vertical {{
                background-color: #636363;
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::add-line:vertical {{
                border: none;
                background: #fcfcfc;
                height: 16px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
                image: url({new_icon_path("caret-down", "svg")});
            }}
            QScrollBar::sub-line:vertical {{
                border: none;
                background: #fcfcfc;
                height: 16px;
                subcontrol-position: top;
                subcontrol-origin: margin;
                image: url({new_icon_path("caret-up", "svg")});
            }}
            QFrame[frameShape="4"] {{
                color: #e5e5e8;
                max-height: 1px;
            }}
        """
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # Search bar
        self.search_bar = SearchBar()
        main_layout.addWidget(self.search_bar)

        # Scroll area for models
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        self.container_layout = QVBoxLayout(container)
        self.container_layout.setContentsMargins(0, 8, 0, 8)
        self.container_layout.setSpacing(8)

        scroll_area.setWidget(container)
        main_layout.addWidget(scroll_area)

        self.model_items = {}
        self.models_data = models_data
        self.setup_model_list()

        # Connect search bar to filter
        self.search_bar.textChanged.connect(self.filter_models)

    def update_models_data(
        self, models_data: dict, current_provider: str = None
    ):
        self.models_data = models_data
        if current_provider:
            self.current_provider = current_provider
        self.setup_model_list()

    def save_models_data(self, provider: str = None, model_id: str = None):
        """Save models data to the config file"""
        try:
            model_config = load_json(MODELS_CONFIG_PATH)
            model_config["models_data"] = self.models_data

            if provider and model_id:
                model_config["settings"]["provider"] = provider
                model_config["settings"]["model_id"] = model_id

            save_json(model_config, MODELS_CONFIG_PATH)

        except Exception as e:
            logger.error(f"Failed to save models data: {e}")

    def setup_model_list(self):
        # Clear existing widgets
        for i in reversed(range(self.container_layout.count())):
            item = self.container_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                elif item.spacerItem():
                    self.container_layout.removeItem(item)

        self.model_items = {}

        if not self.current_provider:
            self._setup_all_providers()
            return

        current_provider_models = self.models_data.get(
            self.current_provider, {}
        )

        favorites = []
        for model_name, model_data in current_provider_models.items():
            if model_data.get("favorite", False):
                favorites.append((model_name, model_data))

        if favorites:
            fav_section = ProviderSection("Favorites")
            self.container_layout.addWidget(fav_section)

            for model_name, model_data in favorites:
                model_item = ModelItem(
                    model_name, model_data, in_favorites_section=True
                )
                model_item.clicked.connect(self.select_model)
                model_item.favoriteToggled.connect(self.toggle_favorite)
                fav_section.add_model_item(model_item)
                self.model_items[model_name] = model_item

            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Plain)
            self.container_layout.addWidget(separator)

        if current_provider_models:
            provider_section = ProviderSection(
                self.current_provider.capitalize()
            )
            self.container_layout.addWidget(provider_section)

            for model_name, model_data in current_provider_models.items():
                model_item = ModelItem(model_name, model_data)
                model_item.clicked.connect(self.select_model)
                model_item.favoriteToggled.connect(self.toggle_favorite)
                provider_section.add_model_item(model_item)
                self.model_items[model_name] = model_item

        self.container_layout.addStretch()
        self.container_layout.update()
        self.container_layout.parentWidget().adjustSize()

    def _setup_all_providers(self):
        # Add favorites section if there are favorites
        favorites = []
        for provider, models in self.models_data.items():
            for model_name, model_data in models.items():
                if model_data.get("favorite", False):
                    favorites.append((provider, model_name, model_data))

        if favorites:
            fav_section = ProviderSection("Favorites")
            self.container_layout.addWidget(fav_section)

            for provider, model_name, model_data in favorites:
                model_item = ModelItem(
                    model_name, model_data, in_favorites_section=True
                )
                model_item.clicked.connect(self.select_model)
                model_item.favoriteToggled.connect(self.toggle_favorite)
                fav_section.add_model_item(model_item)
                self.model_items[model_name] = model_item

            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Plain)
            self.container_layout.addWidget(separator)

        # Add provider sections
        for provider, models in self.models_data.items():
            if not models:
                continue

            provider_section = ProviderSection(provider)
            self.container_layout.addWidget(provider_section)

            for model_name, model_data in models.items():
                model_item = ModelItem(model_name, model_data)
                model_item.clicked.connect(self.select_model)
                model_item.favoriteToggled.connect(self.toggle_favorite)
                provider_section.add_model_item(model_item)
                self.model_items[model_name] = model_item

            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Plain)
            self.container_layout.addWidget(separator)

        # Add stretch at the end to push content to the top
        self.container_layout.addStretch()

        # Force layout update
        self.container_layout.update()
        self.container_layout.parentWidget().adjustSize()

    def select_model(self, model_name):
        # Unselect all models
        for provider, models in self.models_data.items():
            for name, data in models.items():
                data["selected"] = False
                if name in self.model_items:
                    self.model_items[name].update_selection(False)
                    self.models_data[provider][name]["selected"] = False

        # Select the clicked model
        for provider, models in self.models_data.items():
            if model_name in models:
                models[model_name]["selected"] = True
                self.model_items[model_name].update_selection(True)
                self.models_data[provider][model_name]["selected"] = True
                self.save_models_data(provider, model_name)
                self.providerSelected.emit(provider)
                self.modelSelected.emit(model_name)
                break

        self.close()

    def toggle_favorite(self, model_name, is_favorite):
        for provider, models in self.models_data.items():
            if model_name in models:
                self.models_data[provider][model_name][
                    "favorite"
                ] = is_favorite
                break

        # Rebuild the entire list to reflect changes
        self.save_models_data()
        self.setup_model_list()

    def filter_models(self, search_text, match_threshold=0.4):
        empty_text = "No models found."

        for i in reversed(range(self.container_layout.count())):
            widget = self.container_layout.itemAt(i).widget()
            if isinstance(widget, QLabel) and widget.text() == empty_text:
                widget.deleteLater()

        for i in range(self.container_layout.count()):
            widget = self.container_layout.itemAt(i).widget()
            if widget and not isinstance(widget, QLabel):
                widget.setVisible(True)

        for name, item in self.model_items.items():
            item.setVisible(True)

        if not search_text:
            return

        search_text = search_text.lower()
        found_any = False

        # Check if any models match the search
        for name, item in self.model_items.items():
            similarity = SequenceMatcher(
                None, search_text, name.lower()
            ).ratio()
            if search_text in name.lower() or similarity >= match_threshold:
                item.setVisible(True)
                found_any = True
            else:
                item.setVisible(False)

        # Handle UI update based on search results
        if not found_any:
            for i in range(self.container_layout.count()):
                widget = self.container_layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(False)

            no_results = QLabel(empty_text)
            no_results.setAlignment(Qt.AlignCenter)
            no_results.setStyleSheet(
                """
                color: #09090b;
                font-size: 14px;
                padding: 20px;
            """
            )

            self.container_layout.addWidget(no_results)
        else:
            for i in range(self.container_layout.count()):
                widget = self.container_layout.itemAt(i).widget()
                if isinstance(widget, ProviderSection):
                    has_visible_models = False
                    for j in range(widget.models_container.count()):
                        model_widget = widget.models_container.itemAt(
                            j
                        ).widget()
                        if model_widget and model_widget.isVisible():
                            has_visible_models = True
                            break
                    widget.setVisible(has_visible_models)
                elif (
                    isinstance(widget, QFrame)
                    and widget.frameShape() == QFrame.HLine
                ):
                    prev_widget = (
                        self.container_layout.itemAt(i - 1).widget()
                        if i > 0
                        else None
                    )
                    next_widget = (
                        self.container_layout.itemAt(i + 1).widget()
                        if i < self.container_layout.count() - 1
                        else None
                    )
                    should_be_visible = False
                    if prev_widget is not None and next_widget is not None:
                        should_be_visible = (
                            prev_widget.isVisible() and next_widget.isVisible()
                        )
                    widget.setVisible(should_be_visible)
