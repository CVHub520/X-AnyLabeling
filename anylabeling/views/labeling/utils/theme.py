import os
import subprocess
from typing import Dict

try:
    import darkdetect as _darkdetect
except ImportError:
    _darkdetect = None

LIGHT: Dict[str, str] = {
    "primary": "#0071e3",
    "primary_hover": "#0077ED",
    "primary_pressed": "#0068D0",
    "background": "#ffffff",
    "background_secondary": "#F9F9F9",
    "background_hover": "#DBDBDB",
    "surface": "#f5f5f7",
    "surface_hover": "#e5e5e5",
    "surface_pressed": "#d5d5d5",
    "border": "#E5E5E5",
    "border_light": "#d2d2d7",
    "text": "#1d1d1f",
    "text_secondary": "#86868b",
    "text_placeholder": "#718096",
    "highlight": "#60A5FA",
    "highlight_text": "#2196F3",
    "success": "#30D158",
    "warning": "#FF9F0A",
    "error": "#FF453A",
    "scrollbar": "#c1c1c1",
    "scrollbar_hover": "#a8a8a8",
    "selection": "#0071e3",
    "selection_text": "#ffffff",
    "tooltip_bg": "#1d1d1f",
    "tooltip_text": "#f5f5f7",
    "spinbox_button": "#f0f0f0",
    "spinbox_button_hover": "#e0e0e0",
}

DARK: Dict[str, str] = {
    "primary": "#0A84FF",
    "primary_hover": "#409CFF",
    "primary_pressed": "#0071e3",
    "background": "#1c1c1e",
    "background_secondary": "#2c2c2e",
    "background_hover": "#3a3a3c",
    "surface": "#2c2c2e",
    "surface_hover": "#3a3a3c",
    "surface_pressed": "#48484a",
    "border": "#3a3a3c",
    "border_light": "#48484a",
    "text": "#f5f5f7",
    "text_secondary": "#aeaeb2",
    "text_placeholder": "#8e8e93",
    "highlight": "#409CFF",
    "highlight_text": "#409CFF",
    "success": "#30D158",
    "warning": "#FF9F0A",
    "error": "#FF453A",
    "scrollbar": "#48484a",
    "scrollbar_hover": "#636366",
    "selection": "#0A84FF",
    "selection_text": "#ffffff",
    "tooltip_bg": "#3a3a3c",
    "tooltip_text": "#f5f5f7",
    "spinbox_button": "#3a3a3c",
    "spinbox_button_hover": "#48484a",
}

_active_mode: str = "light"
_active_theme: Dict[str, str] = LIGHT


def _is_wsl() -> bool:
    """Returns True when the process is running inside WSL (1 or 2)."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", encoding="utf-8") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def _wsl_windows_theme() -> str:
    """
    Reads the Windows host dark/light preference from inside WSL via
    WSL interop (reg.exe is always available in WSL2).

    reg.exe output example:
      HKEY_CURRENT_USER\\...\\Personalize
          AppsUseLightTheme    REG_DWORD    0x0   ← dark
          AppsUseLightTheme    REG_DWORD    0x1   ← light
    """
    try:
        result = subprocess.run(
            [
                "reg.exe",
                "query",
                r"HKCU\Software\Microsoft\Windows\CurrentVersion"
                r"\Themes\Personalize",
                "/v",
                "AppsUseLightTheme",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "AppsUseLightTheme" in line:
                    return "light" if "0x1" in line else "dark"
    except Exception:
        pass
    return "light"


def _detect_system_theme() -> str:
    """
    Returns the OS dark/light preference as 'dark' or 'light'.

    WSL2 is a special case: the process appears as Linux, so darkdetect
    probes GTK/gsettings which do not exist — instead we call reg.exe via
    WSL interop to read the Windows host registry directly.
    For all other platforms (Windows, macOS, native Linux) darkdetect handles
    detection natively.
    """
    if _is_wsl():
        return _wsl_windows_theme()

    if _darkdetect is not None:
        try:
            result = _darkdetect.theme()
            if isinstance(result, str):
                return "dark" if result.lower() == "dark" else "light"
        except Exception:
            pass

    return "light"


def init_theme(mode: str) -> None:
    """
    Initializes the global theme state at application startup.

    Args:
        mode: 'auto' (default), 'light', or 'dark'.
              'auto' resolves the OS preference via _detect_system_theme().
    """
    global _active_mode, _active_theme
    if mode == "auto":
        resolved = _detect_system_theme()
    elif mode in ("light", "dark"):
        resolved = mode
    else:
        resolved = "light"
    _active_mode = resolved
    _active_theme = DARK if resolved == "dark" else LIGHT


def get_theme() -> Dict[str, str]:
    """
    Returns the currently active theme color dictionary.

    Returns:
        Dict[str, str]: Mapping of color role names to hex color strings.
    """
    return _active_theme


def get_mode() -> str:
    """
    Returns the currently active theme mode.

    Returns:
        str: Either 'light' or 'dark'.
    """
    return _active_mode


def _checkbox_indicator_qss() -> str:
    """
    Returns QSS for QCheckBox indicator sub-controls.

    Dark mode: primary accent fill + white checkmark (professional style).
    Light mode: white fill + blue checkmark.

    Returns:
        str: QSS fragment for QCheckBox::indicator rules.
    """
    t = _active_theme
    if _active_mode == "dark":
        checked_bg = t["primary"]
        checked_border = t["primary"]
        checkmark = ":/images/images/checkmark-white.svg"
    else:
        checked_bg = "#ffffff"
        checked_border = t["border_light"]
        checkmark = ":/images/images/checkmark.svg"
    return f"""
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid {t["border_light"]};
            background-color: {t["background_secondary"]};
        }}
        QCheckBox::indicator:hover {{
            border-color: {t["primary"]};
        }}
        QCheckBox::indicator:checked {{
            background-color: {checked_bg};
            border: 1px solid {checked_border};
            image: url({checkmark});
        }}
    """


def get_app_stylesheet() -> str:
    """
    Returns a comprehensive QSS stylesheet for the active theme, suitable
    for application-level application via QApplication.setStyleSheet().
    Returns an empty string for light mode to preserve original behavior.

    Returns:
        str: QSS stylesheet string, or empty string when mode is 'light'.
    """
    if _active_mode == "light":
        return ""
    t = _active_theme
    return f"""
        QMainWindow, QDialog {{
            background-color: {t["background"]};
            color: {t["text"]};
        }}

        QMainWindow::separator {{
            background-color: {t["border"]};
            width: 1px;
            height: 1px;
        }}

        QMenuBar {{
            background-color: {t["background"]};
            color: {t["text"]};
            border-bottom: 1px solid {t["border"]};
        }}
        QMenuBar::item {{
            background-color: transparent;
            padding: 4px 8px;
        }}
        QMenuBar::item:selected {{
            background-color: {t["surface_hover"]};
            border-radius: 4px;
        }}
        QMenuBar::item:pressed {{
            background-color: {t["surface_pressed"]};
        }}

        QMenu {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            padding: 4px 0 4px 8px;
        }}
        QMenu::item {{
            padding: 6px 24px 6px 8px;
        }}
        QMenu::item:selected {{
            background-color: {t["surface_hover"]};
        }}
        QMenu::item:disabled {{
            color: {t["text_secondary"]};
        }}
        QMenu::separator {{
            height: 1px;
            background-color: {t["border"]};
            margin: 4px 0;
        }}

        QToolBar {{
            background-color: {t["background"]};
            border-bottom: 1px solid {t["border"]};
            spacing: 2px;
            padding: 2px;
        }}
        QToolBar::separator {{
            background-color: {t["border"]};
            width: 1px;
            margin: 4px 2px;
        }}
        QToolButton {{
            background-color: transparent;
            border: none;
            border-radius: 4px;
        }}
        QToolButton:hover {{
            background-color: {t["surface_hover"]};
        }}
        QToolButton:pressed, QToolButton:checked {{
            background-color: {t["surface_pressed"]};
        }}

        QDockWidget {{
            color: {t["text"]};
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }}
        QDockWidget::title {{
            background-color: {t["surface"]};
            border-bottom: 1px solid {t["border"]};
            padding: 4px;
            text-align: center;
        }}
        QDockWidget::close-button, QDockWidget::float-button {{
            border: none;
            background-color: transparent;
        }}

        QStatusBar {{
            background-color: {t["background"]};
            color: {t["text_secondary"]};
            border-top: 1px solid {t["border"]};
        }}
        QStatusBar::item {{
            border: none;
        }}

        QScrollBar:vertical {{
            background-color: {t["background_secondary"]};
            width: 8px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background-color: {t["scrollbar"]};
            border-radius: 4px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {t["scrollbar_hover"]};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar:horizontal {{
            background-color: {t["background_secondary"]};
            height: 8px;
            margin: 0;
        }}
        QScrollBar::handle:horizontal {{
            background-color: {t["scrollbar"]};
            border-radius: 4px;
            min-width: 20px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background-color: {t["scrollbar_hover"]};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0;
        }}

        QListWidget, QTreeWidget, QTableWidget,
        QListView, QTreeView {{
            background-color: {t["background"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            alternate-background-color: {t["background_secondary"]};
            outline: none;
        }}
        QListWidget::item, QTreeWidget::item, QTableWidget::item,
        QListView::item, QTreeView::item {{
            padding: 2px 4px;
        }}
        QListWidget::item:selected, QTreeWidget::item:selected,
        QTableWidget::item:selected,
        QListView::item:selected, QTreeView::item:selected {{
            background-color: {t["selection"]};
            color: {t["selection_text"]};
        }}
        QListWidget::item:hover, QTreeWidget::item:hover,
        QTableWidget::item:hover,
        QListView::item:hover, QTreeView::item:hover {{
            background-color: {t["surface_hover"]};
        }}

        QHeaderView {{
            background-color: {t["surface"]};
            color: {t["text"]};
            border: none;
        }}
        QHeaderView::section {{
            background-color: {t["surface"]};
            color: {t["text"]};
            border: none;
            border-right: 1px solid {t["border"]};
            border-bottom: 1px solid {t["border"]};
            padding: 4px 8px;
        }}
        QTableCornerButton::section {{
            background-color: {t["surface"]};
            border: none;
            border-right: 1px solid {t["border"]};
            border-bottom: 1px solid {t["border"]};
        }}

        QLineEdit {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            padding: 4px 8px;
            selection-background-color: {t["selection"]};
            selection-color: {t["selection_text"]};
        }}
        QLineEdit:hover {{
            border-color: {t["border_light"]};
        }}
        QLineEdit:focus {{
            border: 2px solid {t["highlight"]};
        }}
        QLineEdit:disabled {{
            background-color: {t["surface"]};
            color: {t["text_secondary"]};
        }}

        QTextEdit, QPlainTextEdit {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            selection-background-color: {t["selection"]};
            selection-color: {t["selection_text"]};
        }}
        QTextEdit:focus, QPlainTextEdit:focus {{
            border: 2px solid {t["highlight"]};
        }}

        QComboBox {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            padding: 4px 8px;
            min-height: 24px;
        }}
        QComboBox:hover {{
            border-color: {t["border_light"]};
        }}
        QComboBox:focus {{
            border: 2px solid {t["highlight"]};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox::down-arrow {{
            image: url(:/images/images/caret-down.svg);
            width: 12px;
            height: 12px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 0;
            padding: 0;
            margin: 0;
            selection-background-color: {t["selection"]};
            selection-color: {t["selection_text"]};
            outline: none;
        }}
        QComboBox QAbstractItemView::item {{
            background-color: {t["background_secondary"]};
            padding: 4px 8px;
            min-height: 22px;
        }}
        QComboBox QAbstractItemView::item:selected {{
            background-color: {t["selection"]};
            color: {t["selection_text"]};
        }}

        QCheckBox {{
            color: {t["text"]};
            spacing: 6px;
        }}
        {_checkbox_indicator_qss()}

        QRadioButton {{
            color: {t["text"]};
            spacing: 6px;
        }}
        QRadioButton::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 8px;
            border: 1px solid {t["border_light"]};
            background-color: {t["background_secondary"]};
        }}
        QRadioButton::indicator:checked {{
            background-color: {t["primary"]};
            border-color: {t["primary"]};
        }}

        QPushButton {{
            background-color: {t["surface"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 8px;
            padding: 5px 12px;
            min-height: 24px;
        }}
        QPushButton:hover {{
            background-color: {t["surface_hover"]};
        }}
        QPushButton:pressed {{
            background-color: {t["surface_pressed"]};
        }}
        QPushButton:disabled {{
            background-color: {t["surface"]};
            color: {t["text_secondary"]};
            border-color: {t["border"]};
        }}

        QTabWidget::pane {{
            background-color: {t["background"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
        }}
        QTabBar::tab {{
            background-color: {t["surface"]};
            color: {t["text_secondary"]};
            border: 1px solid {t["border"]};
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 6px 12px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{
            background-color: {t["background"]};
            color: {t["text"]};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: {t["surface_hover"]};
        }}

        QGroupBox {{
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 4px;
            color: {t["text_secondary"]};
        }}

        QSplitter::handle {{
            background-color: {t["border"]};
        }}
        QSplitter::handle:horizontal {{
            width: 1px;
        }}
        QSplitter::handle:vertical {{
            height: 1px;
        }}

        QToolTip {{
            background-color: {t["tooltip_bg"]};
            color: {t["tooltip_text"]};
            border: 1px solid {t["border"]};
            border-radius: 4px;
            padding: 4px 8px;
        }}

        QSpinBox, QDoubleSpinBox {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 6px;
            padding: 5px 8px;
            min-height: 24px;
            selection-background-color: {t["selection"]};
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            width: 20px;
            border: none;
            background-color: {t["spinbox_button"]};
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: {t["spinbox_button_hover"]};
        }}

        QProgressBar {{
            background-color: {t["surface"]};
            border: none;
            border-radius: 4px;
            text-align: center;
            color: transparent;
        }}
        QProgressBar::chunk {{
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #0066FF,
                stop:0.5 #00A6FF,
                stop:1 #0066FF
            );
            border-radius: 3px;
        }}

        QLabel {{
            background-color: transparent;
            color: {t["text"]};
        }}

        QScrollArea {{
            background-color: transparent;
            border: none;
        }}

        QSlider::groove:horizontal {{
            height: 4px;
            background-color: {t["border"]};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background-color: {t["primary"]};
            border: none;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}
        QSlider::sub-page:horizontal {{
            background-color: {t["primary"]};
            border-radius: 2px;
        }}
    """
