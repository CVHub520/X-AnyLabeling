from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.theme import (
    get_theme,
    get_mode,
    _checkbox_indicator_qss,
)
from PyQt5.QtGui import QColor


def get_table_item_bg_color() -> "QColor":
    """
    Returns the appropriate QColor for editable table cell backgrounds.

    Returns:
        QColor: Background color that fits the active theme.
    """
    t = get_theme()
    return QColor(t["background_secondary"])


def get_table_item_disabled_bg_color() -> "QColor":
    """
    Returns the QColor for disabled/read-only table cell backgrounds.

    Returns:
        QColor: Disabled background color that fits the active theme.
    """
    t = get_theme()
    return QColor(t["surface"])


def get_progress_dialog_style(color=None, height=None):
    t = get_theme()
    return f"""
        QProgressDialog {{
            background-color: {t["background"]};
            border-radius: 12px;
            min-width: 280px;
            min-height: 120px;
            padding: 20px;
        }}
        QProgressBar {{
            border: none;
            border-radius: 4px;
            background-color: {t["surface"]};
            text-align: center;
            color: {"transparent" if color is None else color};
            font-size: 13px;
            min-height: {6 if height is None else height}px;
            max-height: {6 if height is None else height}px;
            margin: 16px 0;
        }}
        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #0066FF,
                stop:0.5 #00A6FF,
                stop:1 #0066FF);
            border-radius: 3px;
        }}
        QLabel#progressLabel {{
            color: {t["text"]};
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 8px;
        }}
        QLabel#detailLabel {{
            color: {t["text_secondary"]};
            font-size: 11px;
            margin-top: 4px;
        }}
        QPushButton {{
            background-color: {t["background_secondary"]};
            border: 0.5px solid {t["border_light"]};
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
            color: {t["primary"]};
            min-width: 82px;
            height: 36px;
            padding: 0 16px;
            margin-top: 16px;
        }}
        QPushButton:hover {{
            background-color: {t["surface_hover"]};
        }}
        QPushButton:pressed {{
            background-color: {t["surface_pressed"]};
        }}
    """


def get_msg_box_style():
    t = get_theme()
    return f"""
        QMessageBox {{
            background-color: {t["background"]};
            border-radius: 10px;
        }}
        QPushButton {{
            background-color: {t["surface"]};
            border: 1px solid {t["border_light"]};
            border-radius: 8px;
            font-weight: 500;
            min-width: 100px;
            height: 36px;
        }}
        QPushButton:hover {{
            background-color: {t["surface_hover"]};
        }}
    """


def get_ok_btn_style():
    t = get_theme()
    return f"""
        QPushButton {{
            background-color: {t["primary"]};
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            min-width: 100px;
            height: 36px;
        }}
        QPushButton:hover {{
            background-color: {t["primary_hover"]};
        }}
        QPushButton:pressed {{
            background-color: {t["primary_pressed"]};
        }}
    """


def get_cancel_btn_style():
    t = get_theme()
    return f"""
        QPushButton {{
            background-color: {t["surface"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 8px;
            font-weight: 500;
            min-width: 100px;
            height: 36px;
        }}
        QPushButton:hover {{
            background-color: {t["surface_hover"]};
        }}
        QPushButton:pressed {{
            background-color: {t["surface_pressed"]};
        }}
    """


def get_export_option_style():
    t = get_theme()
    return f"""
        QDialog {{
            background-color: {t["background"]};
            border-radius: 8px;
        }}

        QLabel {{
            font-size: 13px;
            background-color: transparent;
            border-left: none;
        }}

        QLineEdit {{
            border: 1px solid {t["border"]};
            border-radius: 8;
            background-color: {t["background_secondary"]};
            font-size: 13px;
            height: 36px;
            padding-left: 4px;
        }}
        QLineEdit:hover {{
            background-color: {t["background_hover"]};
            border-radius: 8px;
        }}
        QLineEdit:focus {{
            border: 3px solid "{t["highlight"]}";
            background-color: "{t["background_secondary"]}";
        }}

        {get_checkbox_indicator_style()}
    """


def get_normal_button_style():
    t = get_theme()
    return f"""
        QPushButton {{
            height: 24px;
            min-width: 80px;
            padding: 5px 8px;
            border-radius: 8px;
            background-color: {t["surface"]};
            border: 1px solid {t["border_light"]};
        }}
        QPushButton:hover {{
            background-color: {t["surface_hover"]};
        }}
        QPushButton:pressed {{
            background-color: {t["surface_pressed"]};
        }}
    """


def get_toggle_button_style(button_color: str):
    t = get_theme()
    return f"""
        QPushButton {{
            height: 24px;
            min-width: 80px;
            padding: 5px 8px;
            border-radius: 8px;
            background-color: {button_color};
            border: 1px solid {t["border_light"]};
        }}
    """


def get_highlight_button_style():
    t = get_theme()
    return f"""
        QPushButton {{
            height: 24px;
            color: white;
            border: none;
            min-width: 80px;
            padding: 5px 8px;
            border-radius: 8px;
            background-color: {t["primary"]};
        }}
        QPushButton:hover {{
            background-color: {t["primary_hover"]};
        }}
        QPushButton:pressed {{
            background-color: {t["primary_pressed"]};
        }}
    """


def get_spinbox_style():
    t = get_theme()
    return f"""
        QSpinBox {{
            padding: 5px 8px;
            background: {t["background_secondary"]};
            border: 1px solid {t["border_light"]};
            border-radius: 6px;
            min-height: 24px;
            selection-background-color: {t["primary"]};
        }}
        QSpinBox::up-button, QSpinBox::down-button {{
            width: 20px;
            border: none;
            background: {t["spinbox_button"]};
        }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
            background: {t["spinbox_button_hover"]};
        }}
        QSpinBox::up-arrow {{
            image: url({new_icon_path("caret-up", "svg")});
            width: 12px;
            height: 12px;
        }}
        QSpinBox::down-arrow {{
            image: url({new_icon_path("caret-down", "svg")});
            width: 12px;
            height: 12px;
        }}
    """


def get_double_spinbox_style():
    """
    Returns the CSS stylesheet for a QDoubleSpinBox, suitable for decimals.
    """
    t = get_theme()
    return f"""
        QDoubleSpinBox {{
            padding: 5px 8px;
            background: {t["background_secondary"]};
            border: 1px solid {t["border_light"]};
            border-radius: 6px;
            min-height: 24px;
            selection-background-color: {t["primary"]};
        }}
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            width: 20px;
            border: none;
            background: {t["spinbox_button"]};
        }}
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
            background: {t["spinbox_button_hover"]};
        }}
        QDoubleSpinBox::up-arrow {{
            image: url({new_icon_path("caret-up", "svg")});
            width: 12px;
            height: 12px;
        }}
        QDoubleSpinBox::down-arrow {{
            image: url({new_icon_path("caret-down", "svg")});
            width: 12px;
            height: 12px;
        }}
    """


def get_dock_style() -> str:
    """
    Returns a theme-aware stylesheet for QDockWidget and its child widgets.
    Must be comprehensive because widget.setStyleSheet() blocks app-level QSS
    from propagating to the widget's descendants.

    Returns:
        str: QSS stylesheet string covering dock widget and common children.
    """
    t = get_theme()
    return f"""
        QDockWidget {{
            color: {t["text"]};
        }}
        QDockWidget::title {{
            text-align: center;
            padding: 0px;
            background-color: {t["surface"]};
            color: {t["text"]};
        }}
        QWidget {{
            background-color: {t["background"]};
            color: {t["text"]};
        }}
        QListWidget {{
            background-color: {t["background"]};
            color: {t["text"]};
            border: none;
            outline: none;
        }}
        QListWidget::item {{
            padding: 2px 0;
        }}
        QListWidget::item:selected {{
            background-color: {t["selection"]};
            color: {t["selection_text"]};
        }}
        QListWidget::item:hover {{
            background-color: {t["surface_hover"]};
        }}
        QListWidget::indicator {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
            border: 1px solid {t["border_light"]};
            background-color: {t["background_secondary"]};
            margin-left: 2px;
            margin-right: 4px;
        }}
        QListWidget::indicator:checked {{
            background-color: {t["primary"]};
            border-color: {t["primary"]};
            image: url(:/images/images/checkmark-white.svg);
        }}
        QLineEdit {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            padding: 4px 8px;
            selection-background-color: {t["selection"]};
        }}
        QLineEdit:focus {{
            border: 2px solid {t["highlight"]};
        }}
        QLabel {{
            background-color: transparent;
            color: {t["text"]};
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
        QCheckBox {{
            color: {t["text"]};
        }}
        {get_checkbox_indicator_style()}
        QPushButton {{
            background-color: {t["surface"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 6px;
            padding: 3px 8px;
        }}
        QPushButton:hover {{
            background-color: {t["surface_hover"]};
        }}
        QPushButton:pressed {{
            background-color: {t["surface_pressed"]};
        }}
        QComboBox {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            padding: 3px 8px;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox::down-arrow {{
            image: url({new_icon_path("caret-down", "svg")});
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
    """


def get_panel_style() -> str:
    """
    Returns a stylesheet for a sidebar panel QFrame container.

    Provides a subtle card appearance with a rounded border so each sidebar
    section is visually distinct while remaining visually cohesive.
    Uses object-name selector so child QFrames are not affected.

    Returns:
        str: QSS stylesheet string targeting QFrame#sidebarPanel.
    """
    t = get_theme()
    return (
        f"QFrame#sidebarPanel {{"
        f" background-color: {t['surface']};"
        f" border: 1px solid {t['border']};"
        f" border-radius: 6px;"
        f" }}"
    )


def get_plain_text_edit_style() -> str:
    """
    Returns a borderless, theme-aware stylesheet for QPlainTextEdit.

    Removes the default Qt frame so the widget blends into the sidebar
    without an outer border that the panel container already provides.

    Returns:
        str: QSS stylesheet string for QPlainTextEdit.
    """
    t = get_theme()
    return f"""
        QPlainTextEdit {{
            background-color: {t["background"]};
            color: {t["text"]};
            border: none;
            selection-background-color: {t["selection"]};
            selection-color: {t["selection_text"]};
        }}
        QPlainTextEdit:focus {{
            border: 2px solid {t["highlight"]};
        }}
    """


def get_lineedit_style():
    t = get_theme()
    return f"""
        QLineEdit {{
            border: 1px solid {t["border"]};
            border-radius: 8;
            background-color: {t["background_secondary"]};
            font-size: 13px;
            height: 24px;
            padding: 5px 8px;
        }}
        QLineEdit:hover {{
            background-color: {t["background_hover"]};
            border-radius: 8px;
        }}
        QLineEdit:focus {{
            border: 3px solid "{t["highlight"]}";
            background-color: "{t["background_secondary"]}";
        }}
    """


def get_checkbox_indicator_style() -> str:
    """
    Returns QSS for QCheckBox indicator sub-controls.

    Delegates to the central theme module so the logic is defined in one place.

    Returns:
        str: QSS fragment for QCheckBox::indicator rules.
    """
    return _checkbox_indicator_qss()


def get_dialog_style() -> str:
    """
    Returns a comprehensive base stylesheet for QDialog subclasses.

    Single unified entry point for all dialog theming. Covers the most
    common widget types so each dialog only needs one setStyleSheet call.
    Dialog-specific overrides can append additional QSS as needed.

    Returns:
        str: Complete QSS stylesheet for dialogs.
    """
    t = get_theme()
    return f"""
        QDialog {{
            background-color: {t["background"]};
            color: {t["text"]};
        }}
        QLabel {{
            background-color: transparent;
            color: {t["text"]};
        }}
        QGroupBox {{
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 8px;
        }}
        QGroupBox::title {{
            color: {t["text_secondary"]};
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 4px;
        }}
        QLineEdit {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            padding: 4px 8px;
            height: 28px;
            selection-background-color: {t["selection"]};
        }}
        QLineEdit:hover {{
            border-color: {t["border_light"]};
        }}
        QLineEdit:focus {{
            border: 2px solid {t["highlight"]};
        }}
        QSpinBox, QDoubleSpinBox {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 6px;
            padding: 5px 8px;
            min-height: 24px;
            selection-background-color: {t["primary"]};
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            width: 20px;
            border: none;
            background: {t["spinbox_button"]};
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background: {t["spinbox_button_hover"]};
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: url({new_icon_path("caret-up", "svg")});
            width: 12px;
            height: 12px;
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: url({new_icon_path("caret-down", "svg")});
            width: 12px;
            height: 12px;
        }}
        QComboBox {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 6px;
            padding: 4px 8px;
            min-height: 24px;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox::down-arrow {{
            image: url({new_icon_path("caret-down", "svg")});
            width: 12px;
            height: 12px;
        }}
        QComboBox:focus {{
            border: 2px solid {t["highlight"]};
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
        QSlider {{
            height: 28px;
        }}
        QSlider::groove:horizontal {{
            height: 4px;
            background: {t["border"]};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {t["primary"]};
            border: none;
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }}
        QSlider::sub-page:horizontal {{
            background: {t["primary"]};
            border-radius: 2px;
        }}
        QCheckBox {{
            color: {t["text"]};
            spacing: 6px;
        }}
        {get_checkbox_indicator_style()}
        QPushButton {{
            background-color: {t["surface"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 8px;
            font-weight: 500;
            min-width: 100px;
            height: 36px;
            padding: 0 12px;
        }}
        QPushButton:hover {{
            background-color: {t["surface_hover"]};
        }}
        QPushButton:pressed {{
            background-color: {t["surface_pressed"]};
        }}
        QPushButton:disabled {{
            color: {t["text_secondary"]};
            border-color: {t["border"]};
        }}
        QTableWidget {{
            background-color: {t["background"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            gridline-color: {t["border"]};
            outline: none;
        }}
        QTableWidget::item {{
            padding: 4px 8px;
            border: none;
        }}
        QTableWidget::item:selected {{
            background-color: {t["selection"]};
            color: {t["selection_text"]};
        }}
        QTableWidget::item:hover {{
            background-color: {t["surface_hover"]};
        }}
        QHeaderView::section {{
            background-color: {t["surface"]};
            color: {t["text_secondary"]};
            border: none;
            border-right: 1px solid {t["border"]};
            border-bottom: 1px solid {t["border"]};
            padding: 4px 8px;
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
    """
