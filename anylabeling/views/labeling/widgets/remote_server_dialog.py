from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QSizePolicy,
)

from anylabeling.views.labeling.utils.qt import new_icon
from anylabeling.views.labeling.utils.style import (
    get_lineedit_style,
    get_normal_button_style,
    get_highlight_button_style,
)


class RemoteServerDialog(QDialog):
    def __init__(self, parent=None, default_url="", default_api_key=""):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Remote Server Settings"))
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint
        )
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        url_label = QLabel(self.tr("Server URL:"))
        layout.addWidget(url_label)

        self.url_input = QLineEdit(default_url)
        self.url_input.setStyleSheet(get_lineedit_style())
        self.url_input.setPlaceholderText(self.tr("Enter remote server URL"))
        self.url_input.setToolTip(
            self.tr("Set the remote server URL for model inference")
        )
        layout.addWidget(self.url_input)

        api_key_label = QLabel(self.tr("API Key (Optional):"))
        layout.addWidget(api_key_label)

        api_key_container = QHBoxLayout()
        self.api_key_input = QLineEdit(default_api_key)
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText(self.tr("Enter API key"))
        self.api_key_input.setStyleSheet(get_lineedit_style())
        self.api_key_input.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        button_size = self.api_key_input.sizeHint().height()

        self.toggle_visibility_btn = QPushButton()
        self.toggle_visibility_btn.setCheckable(True)
        self.toggle_visibility_btn.setFixedSize(button_size, button_size)
        self.toggle_visibility_btn.setStyleSheet(get_normal_button_style())
        self.toggle_visibility_btn.clicked.connect(
            self.toggle_api_key_visibility
        )

        self._update_visibility_button(False)

        api_key_container.addWidget(self.api_key_input)
        api_key_container.addWidget(self.toggle_visibility_btn)
        layout.addLayout(api_key_container)

        button_container = QHBoxLayout()
        button_container.addStretch()

        cancel_button = QPushButton(self.tr("Cancel"))
        cancel_button.setStyleSheet(get_normal_button_style())
        cancel_button.clicked.connect(self.reject)
        button_container.addWidget(cancel_button)

        ok_button = QPushButton(self.tr("OK"))
        ok_button.setStyleSheet(get_highlight_button_style())
        ok_button.clicked.connect(self.accept)
        button_container.addWidget(ok_button)

        layout.addLayout(button_container)

    def _update_visibility_button(self, checked):
        """Updates the icon or text of the visibility button."""
        show_text = self.tr("Show")
        hide_text = self.tr("Hide")
        if checked:
            try:
                self.toggle_visibility_btn.setIcon(
                    QIcon(new_icon("eye", "svg"))
                )
                self.toggle_visibility_btn.setText("")
                self.toggle_visibility_btn.setToolTip(hide_text)
            except:  # noqa
                self.toggle_visibility_btn.setIcon(QIcon())
                self.toggle_visibility_btn.setText(hide_text)
                self.toggle_visibility_btn.setToolTip("")
        else:
            try:
                self.toggle_visibility_btn.setIcon(
                    QIcon(new_icon("eye-off", "svg"))
                )
                self.toggle_visibility_btn.setText("")
                self.toggle_visibility_btn.setToolTip(show_text)
            except:  # noqa
                self.toggle_visibility_btn.setIcon(QIcon())
                self.toggle_visibility_btn.setText(show_text)
                self.toggle_visibility_btn.setToolTip("")

    def toggle_api_key_visibility(self, checked):
        """Toggles the visibility of the API key input."""
        if checked:
            self.api_key_input.setEchoMode(QLineEdit.Normal)
        else:
            self.api_key_input.setEchoMode(QLineEdit.Password)
        self._update_visibility_button(checked)

    def get_server_url(self):
        """Gets the server URL currently entered in the input field."""
        return self.url_input.text().strip()

    def get_api_key(self):
        """Gets the API key currently entered in the input field."""
        return self.api_key_input.text().strip()
