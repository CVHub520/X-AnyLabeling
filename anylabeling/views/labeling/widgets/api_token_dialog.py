import os

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QSizePolicy,
    QDialogButtonBox,
)

from anylabeling.views.labeling.utils.qt import new_icon
from anylabeling.views.labeling.utils.style import (
    get_lineedit_style,
    get_normal_button_style,
    get_ok_btn_style,
    get_cancel_btn_style,
)

_cached_api_token = os.getenv("GROUNDING_DINO_API_TOKEN", "")


class ApiTokenDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Set API Token"))
        self.setMinimumWidth(500)

        self.layout = QVBoxLayout(self)

        self.label = QLabel(self.tr("Enter your API Token:"))
        self.layout.addWidget(self.label)

        # API key input with toggle visibility
        api_key_container = QHBoxLayout()
        self.api_key_input = QLineEdit(_cached_api_token)
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText(self.tr("Enter API key"))
        self.api_key_input.setStyleSheet(get_lineedit_style())
        self.api_key_input.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        button_size = (
            self.api_key_input.sizeHint().height()
        )  # Match height of line edit

        self.toggle_visibility_btn = QPushButton()
        self.toggle_visibility_btn.setCheckable(True)
        self.toggle_visibility_btn.setFixedSize(button_size, button_size)
        self.toggle_visibility_btn.setStyleSheet(get_normal_button_style())
        self.toggle_visibility_btn.clicked.connect(
            self.toggle_api_key_visibility
        )

        # Set initial icon/text
        self._update_visibility_button(False)

        api_key_container.addWidget(self.api_key_input)
        api_key_container.addWidget(self.toggle_visibility_btn)
        self.layout.addLayout(api_key_container)

        # OK and Cancel buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        ok_button = self.button_box.button(QDialogButtonBox.Ok)
        cancel_button = self.button_box.button(QDialogButtonBox.Cancel)
        if ok_button:
            ok_button.setStyleSheet(get_ok_btn_style())
            ok_button.setIcon(QIcon())
        if cancel_button:
            cancel_button.setStyleSheet(get_cancel_btn_style())
            cancel_button.setIcon(QIcon())

        self.button_box.accepted.connect(self.accept_and_update_cache)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def _update_visibility_button(self, checked):
        """Updates the icon or text of the visibility button."""
        # Use shorter text labels as fallback if icons fail or aren't available
        show_text = self.tr("Show")
        hide_text = self.tr("Hide")
        if checked:  # Visible state
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
        else:  # Hidden state (Password mode)
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
        if checked:
            self.api_key_input.setEchoMode(QLineEdit.Normal)
        else:
            self.api_key_input.setEchoMode(QLineEdit.Password)
        self._update_visibility_button(checked)

    def get_token(self):
        """Gets the token currently entered in the input field."""
        return self.api_key_input.text()

    def accept_and_update_cache(self):
        """Updates the module-level cache before accepting the dialog."""
        global _cached_api_token
        _cached_api_token = self.get_token()
        self.accept()
