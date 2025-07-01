import threading

from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
except ImportError:
    QWebEngineView = None

from anylabeling.app_info import (
    __appname__,
    __version__,
    __preferred_device__,
)
from anylabeling.views.labeling.utils.general import (
    collect_system_info,
    open_url,
)
from anylabeling.views.labeling.utils.qt import new_icon, new_icon_path
from anylabeling.views.labeling.utils.update_checker import (
    check_for_updates_sync,
)
from anylabeling.views.labeling.widgets.popup import Popup
from anylabeling.views.labeling.chatbot.render import convert_markdown_to_html


class AboutDialog(QDialog):
    update_available = pyqtSignal(dict)
    no_update = pyqtSignal()
    error = pyqtSignal(str)

    email_address = "cv_hub@163.com"
    website_url = "https://github.com/CVHub520/X-AnyLabeling"
    discord_url = (
        "https://discord.com/channels/1350265627142651994/1350265628832829514"
    )
    twitter_url = "https://x.com/xanylabeling"
    github_url = "https://github.com/CVHub520/X-AnyLabeling"
    github_issues_url = "https://github.com/CVHub520/X-AnyLabeling/issues"
    changelog_url = (
        "https://github.com/CVHub520/X-AnyLabeling/tree/main/CHANGELOG.md"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        self._cached_update_info = None

        self.setWindowTitle(" ")
        self.setFixedSize(350, 250)

        self.setStyleSheet(
            """
            QDialog {
                background-color: #FFFFFF;
                border-radius: 10px;
            }
            QLabel {
                color: #1d1d1f;
            }
            QPushButton {
                border: none;
                background: transparent;
                color: #0066FF;
                text-align: center;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: #F0F0F0;
                border-radius: 4px;
            }
            QPushButton#link-btn {
                color: #0066FF;
            }
            QPushButton#social-btn {
                padding: 8px;
            }
            QPushButton#social-btn:hover {
                background-color: #F0F0F0;
                border-radius: 4px;
            }
            QPushButton#close-btn {
                color: #86868b;
                font-size: 16px;
                padding: 8px;
            }
            QPushButton#close-btn:hover {
                background-color: #F0F0F0;
                border-radius: 4px;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # App name and version
        title_label = QLabel(f"<b>X-AnyLabeling</b> v{__version__}")
        title_label.setStyleSheet("font-size: 16px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Links row - centered
        links_layout = QHBoxLayout()
        links_layout.setSpacing(8)
        links_layout.setAlignment(Qt.AlignCenter)

        website_btn = QPushButton(self.tr("Website"))
        website_btn.setObjectName("link-btn")
        website_btn.clicked.connect(lambda: open_url(self.website_url))

        copy_btn = QPushButton(self.tr("Copy App Info"))
        copy_btn.setObjectName("link-btn")
        copy_btn.clicked.connect(self.copy_app_info)

        report_btn = QPushButton(self.tr("Report Issue"))
        report_btn.setObjectName("link-btn")
        report_btn.clicked.connect(lambda: open_url(self.github_issues_url))

        links_layout.addWidget(website_btn)
        links_layout.addWidget(QLabel("·"))
        links_layout.addWidget(copy_btn)
        links_layout.addWidget(QLabel("·"))
        links_layout.addWidget(report_btn)
        layout.addLayout(links_layout)

        # Social links - centered
        social_layout = QHBoxLayout()
        social_layout.setSpacing(4)
        social_layout.setAlignment(Qt.AlignCenter)

        # Email
        email_btn = QPushButton()
        email_btn.setObjectName("social-btn")
        email_btn.setIcon(QIcon(new_icon("email")))
        email_btn.setIconSize(QSize(20, 20))
        email_btn.setToolTip(self.email_address)
        email_btn.clicked.connect(
            lambda: self.copy_to_clipboard(self.email_address)
        )

        # Discord
        discord_btn = QPushButton()
        discord_btn.setObjectName("social-btn")
        discord_btn.setIcon(QIcon(new_icon("discord")))
        discord_btn.setIconSize(QSize(20, 20))
        discord_btn.clicked.connect(lambda: open_url(self.discord_url))

        # Twitter
        twitter_btn = QPushButton()
        twitter_btn.setObjectName("social-btn")
        twitter_btn.setIcon(QIcon(new_icon("twitter")))
        twitter_btn.setIconSize(QSize(20, 20))
        twitter_btn.clicked.connect(lambda: open_url(self.twitter_url))

        # GitHub
        github_btn = QPushButton()
        github_btn.setObjectName("social-btn")
        github_btn.setIcon(QIcon(new_icon("github")))
        github_btn.setIconSize(QSize(20, 20))
        github_btn.clicked.connect(lambda: open_url(self.github_url))

        social_layout.addWidget(email_btn)
        social_layout.addWidget(discord_btn)
        social_layout.addWidget(twitter_btn)
        social_layout.addWidget(github_btn)
        layout.addLayout(social_layout)

        # Changelog and update - centered
        update_layout = QHBoxLayout()
        update_layout.setSpacing(8)
        update_layout.setAlignment(Qt.AlignCenter)

        changelog_btn = QPushButton(self.tr("Changelog"))
        changelog_btn.setObjectName("link-btn")
        changelog_btn.clicked.connect(lambda: open_url(self.changelog_url))

        check_update_btn = QPushButton(self.tr("Check for Updates"))
        check_update_btn.setObjectName("link-btn")
        check_update_btn.clicked.connect(self.check_for_updates)

        update_layout.addWidget(changelog_btn)
        update_layout.addWidget(QLabel("·"))
        update_layout.addWidget(check_update_btn)
        layout.addLayout(update_layout)

        # Add stretch to push everything to the top
        layout.addStretch()

        # Copyright
        copyright_label = QLabel(
            "Copyright © 2023 CVHub. All rights reserved."
        )
        copyright_label.setStyleSheet("color: #86868b; font-size: 12px;")
        copyright_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(copyright_label)

        self.move_to_center()

        QTimer.singleShot(0, self.check_updates_in_background)

    def move_to_center(self):
        """Move dialog to center of the screen"""
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def copy_app_info(self):
        """Copy app info to clipboard"""
        app_info = (
            f"App name: {__appname__}\n"
            f"App version: {__version__}\n"
            f"Device: {__preferred_device__}\n"
        )
        system_info, pkg_info = collect_system_info()
        system_info_str = "\n".join(
            [f"{key}: {value}" for key, value in system_info.items()]
        )
        pkg_info_str = "\n".join(
            [f"{key}: {value}" for key, value in pkg_info.items()]
        )
        msg = f"{app_info}\n{system_info_str}\n\n{pkg_info_str}"

        popup = Popup(
            self.tr("Copied!"),
            self.parent,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self.parent, copy_msg=msg)

    def check_for_updates(self):
        """Handle user-initiated update check"""
        if self._cached_update_info:
            if self._cached_update_info["has_update"]:
                self.show_update_dialog(self._cached_update_info)
            else:
                popup = Popup(
                    self.tr("No Updates Available"),
                    self.parent,
                    icon=new_icon_path("copy-green", "svg"),
                )
                popup.show_popup(self.parent)
            return

        # If no cache, perform new check with error reporting
        update_info = check_for_updates_sync(timeout=10)
        if update_info:
            self._cached_update_info = update_info
            if update_info["has_update"]:
                self.show_update_dialog(update_info)
            else:
                popup = Popup(
                    self.tr("No Updates Available"),
                    self.parent,
                    icon=new_icon_path("copy-green", "svg"),
                )
                popup.show_popup(self.parent)
        else:
            popup = Popup(
                self.tr("Check update failed"),
                self.parent,
                icon=new_icon_path("error", "svg"),
            )
            popup.show_popup(self.parent)

    def check_updates_in_background(self):
        """Check for updates in background when dialog opens"""

        def update_check_thread():
            update_info = check_for_updates_sync(timeout=5)
            if update_info and update_info["has_update"]:
                self._cached_update_info = update_info
                QTimer.singleShot(
                    0, lambda: self.show_update_dialog(update_info)
                )

        thread = threading.Thread(target=update_check_thread, daemon=True)
        thread.start()

    def show_update_dialog(self, update_info):
        """Show update available dialog with Markdown support"""
        dialog = QDialog(self)
        dialog.setWindowTitle(self.tr("Update Available"))
        dialog.setMinimumSize(500, 400)

        layout = QVBoxLayout(dialog)

        template = "A new version {version} is available!"
        translated_template = self.tr(template)
        display_text = translated_template.format(
            version=update_info["latest_version"]
        )
        title_label = QLabel(display_text)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            """
            font-size: 16px;
            font-weight: bold;
            color: #0066FF;
            margin: 15px 0;
            padding: 10px;
        """
        )
        layout.addWidget(title_label)

        web_view = QWebEngineView()
        web_view.setMinimumHeight(350)
        web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        web_view.setHtml(
            convert_markdown_to_html(update_info["release_notes"])
        )

        layout.addWidget(web_view)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.clicked.connect(dialog.reject)

        ok_btn = QPushButton(self.tr("Download"))
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(
            lambda: self._handle_update_ok(dialog, update_info["download_url"])
        )

        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)

        dialog.exec_()

    def _handle_update_ok(self, dialog, url):
        """Handle OK button click in update dialog"""
        dialog.accept()
        open_url(url)

    def copy_to_clipboard(self, text):
        """Copy text to clipboard and show popup"""
        popup = Popup(
            self.tr("Copied!"),
            self.parent,
            icon=new_icon_path("copy-green", "svg"),
        )
        popup.show_popup(self.parent, copy_msg=text)
