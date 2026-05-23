from anylabeling.views.labeling.utils.theme import (
    _checkbox_indicator_qss,
    get_theme,
)

from .config import BORDER_RADIUS, FONT_SIZE_NORMAL, FONT_SIZE_SMALL


def get_toolbar_button_style(compact=False):
    t = get_theme()
    height = "30px" if compact else "34px"
    min_width = "72px" if compact else "94px"
    padding = "0 12px" if compact else "0 14px"
    return f"""
        QPushButton {{
            background-color: {t["primary"]};
            color: white;
            font-size: {FONT_SIZE_NORMAL};
            font-weight: 600;
            border: 1px solid {t["primary"]};
            border-radius: 6px;
            padding: {padding};
            height: {height};
            min-height: {height};
            min-width: {min_width};
        }}
        QPushButton:hover {{ background-color: {t["primary_hover"]}; }}
        QPushButton:pressed {{ background-color: {t["primary_pressed"]}; }}
        QPushButton:focus {{ border: 1px solid {t["highlight"]}; }}
        QPushButton:disabled {{
            background-color: {t["surface"]};
            color: {t["text_secondary"]};
            border-color: {t["border"]};
        }}
    """


def get_secondary_button_style(compact=False):
    t = get_theme()
    height = "30px" if compact else "34px"
    min_width = "64px" if compact else "82px"
    padding = "0 12px" if compact else "0 14px"
    return f"""
        QPushButton {{
            background-color: {t["surface"]};
            color: {t["text"]};
            font-size: {FONT_SIZE_NORMAL};
            border: 1px solid {t["border_light"]};
            border-radius: 6px;
            padding: {padding};
            height: {height};
            min-height: {height};
            min-width: {min_width};
        }}
        QPushButton:hover {{ background-color: {t["surface_hover"]}; }}
        QPushButton:pressed {{ background-color: {t["surface_pressed"]}; }}
        QPushButton:focus {{ border: 1px solid {t["highlight"]}; }}
        QPushButton:disabled {{
            background-color: {t["surface"]};
            color: {t["text_secondary"]};
            border-color: {t["border"]};
        }}
    """


def get_icon_button_style():
    t = get_theme()
    return f"""
        QPushButton {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 5px;
            padding: 0px;
            color: {t["text"]};
            min-width: 24px;
            max-width: 24px;
            min-height: 24px;
            max-height: 24px;
        }}
        QPushButton:hover {{
            background-color: {t["surface"]};
            border-color: {t["border_light"]};
        }}
        QPushButton:pressed {{ background-color: {t["surface_hover"]}; }}
        QPushButton:checked {{
            background-color: {t["surface_hover"]};
            border-color: {t["primary"]};
        }}
        QPushButton:disabled {{
            color: {t["text_secondary"]};
            border-color: transparent;
            background: transparent;
        }}
    """


def get_drop_zone_style():
    t = get_theme()
    return f"""
        QFrame#XvaDropZone {{
            background-color: {t["background_secondary"]};
            border: 2px dashed {t["border"]};
            border-radius: {BORDER_RADIUS};
        }}
        QFrame#XvaDropZone[hover="true"] {{
            border-color: {t["primary"]};
            background-color: {t["surface"]};
        }}
        QLabel#XvaDropZoneTitle {{
            color: {t["text"]};
            font-size: 16px;
            font-weight: 600;
            background: transparent;
        }}
        QLabel#XvaDropZoneHint {{
            color: {t["text_secondary"]};
            font-size: 12px;
            background: transparent;
        }}
        QLabel#XvaDropZoneIcon {{ background: transparent; }}
    """


def get_video_frame_style():
    t = get_theme()
    return f"""
        QFrame#XvaVideoFrame {{
            background-color: black;
            border: 1px solid {t["border"]};
            border-radius: {BORDER_RADIUS};
        }}
    """


def get_panel_frame_style():
    t = get_theme()
    return f"""
        QFrame#XvaPanel {{
            background-color: {t["background"]};
            border: 1px solid {t["border"]};
            border-radius: {BORDER_RADIUS};
        }}
        QLabel#XvaPanelTitle {{
            color: {t["text"]};
            font-size: 13px;
            font-weight: 600;
            background: transparent;
        }}
    """


def get_timeline_frame_style():
    t = get_theme()
    return f"""
        QFrame#XvaTimelineFrame,
        QFrame#XvaTimelinePanel {{
            background-color: {t["background"]};
            border: 1px solid {t["border"]};
            border-radius: {BORDER_RADIUS};
        }}
        QScrollBar#XvaTimelineScrollBar:horizontal {{
            border: none;
            background-color: {t["background_secondary"]};
            height: 8px;
            margin: 0;
        }}
        QScrollBar#XvaTimelineScrollBar::handle:horizontal {{
            background-color: {t["scrollbar"]};
            min-width: 20px;
            border-radius: 4px;
        }}
        QScrollBar#XvaTimelineScrollBar::handle:horizontal:hover {{
            background-color: {t["scrollbar_hover"]};
        }}
        QScrollBar#XvaTimelineScrollBar::handle:horizontal:disabled {{
            background: transparent;
        }}
        QScrollBar#XvaTimelineScrollBar::add-line:horizontal,
        QScrollBar#XvaTimelineScrollBar::sub-line:horizontal {{
            width: 0;
            border: none;
            background: transparent;
        }}
        QScrollBar#XvaTimelineScrollBar::add-page:horizontal,
        QScrollBar#XvaTimelineScrollBar::sub-page:horizontal {{
            background: transparent;
        }}
    """


def get_label_button_style(color):
    t = get_theme()
    return f"""
        QPushButton {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            font-size: {FONT_SIZE_NORMAL};
            border: 1px solid {t["border_light"]};
            border-left: 6px solid {color};
            border-radius: 8px;
            padding: 6px 10px;
            text-align: left;
        }}
        QPushButton:hover {{ background-color: {t["surface_hover"]}; }}
        QPushButton:checked {{
            background-color: {t["surface_hover"]};
            border-color: {t["primary"]};
            border-left: 6px solid {color};
            color: {t["text"]};
        }}
    """


def get_combobox_style():
    t = get_theme()
    return f"""
        QComboBox {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 6px;
            padding: 3px 24px 3px 8px;
            min-width: 60px;
            font-size: {FONT_SIZE_SMALL};
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
            selection-background-color: {t["primary"]};
            selection-color: white;
        }}
    """


def get_segment_list_style():
    t = get_theme()
    return f"""
        QListWidget {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 8px;
            padding: 3px;
            outline: none;
        }}
        QListWidget::item {{
            padding: 7px 8px;
            border-radius: 6px;
        }}
        QListWidget::item:hover {{
            background-color: {t["surface_hover"]};
        }}
        QListWidget::item:selected {{
            background-color: {t["primary"]};
            color: white;
        }}
        QListWidget QScrollBar:vertical {{
            background: {t["background_secondary"]};
            width: 12px;
            margin: 12px 0px 12px 0px;
            border: none;
        }}
        QListWidget QScrollBar:vertical:disabled {{
            background: transparent;
        }}
        QListWidget QScrollBar::handle:vertical {{
            background: {t["scrollbar"]};
            min-height: 34px;
            border-radius: 5px;
        }}
        QListWidget QScrollBar::handle:vertical:hover {{
            background: {t["scrollbar_hover"]};
        }}
        QListWidget QScrollBar::handle:vertical:disabled {{
            background: transparent;
        }}
        QListWidget QScrollBar::add-line:vertical {{
            background: {t["background_secondary"]};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: bottom;
            height: 12px;
            image: url(:/images/images/caret-down.svg);
        }}
        QListWidget QScrollBar::sub-line:vertical {{
            background: {t["background_secondary"]};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: top;
            height: 12px;
            image: url(:/images/images/caret-up.svg);
        }}
        QListWidget QScrollBar::add-line:vertical:disabled,
        QListWidget QScrollBar::sub-line:vertical:disabled {{
            background: transparent;
            image: none;
        }}
        QListWidget QScrollBar::add-page:vertical,
        QListWidget QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
    """


def get_label_settings_dialog_style():
    t = get_theme()
    return get_dialog_style() + _checkbox_indicator_qss() + f"""
        QLabel#XvaSettingsTitle {{
            color: {t["text"]};
            font-size: 15px;
            font-weight: 600;
            background: transparent;
        }}
        QTableWidget#XvaLabelSettingsTable {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 8px;
            gridline-color: transparent;
            outline: none;
        }}
        QTableWidget#XvaLabelSettingsTable::item {{
            padding: 4px 8px;
            border: none;
        }}
        QTableWidget#XvaLabelSettingsTable::item:selected {{
            background: transparent;
            color: {t["text"]};
        }}
        QHeaderView::section {{
            background-color: {t["surface"]};
            color: {t["text_secondary"]};
            border: none;
            border-bottom: 1px solid {t["border"]};
            padding: 6px 8px;
            font-size: {FONT_SIZE_SMALL};
            font-weight: 600;
        }}
        QLineEdit {{
            background-color: {t["background"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 6px;
            padding: 0 8px;
            selection-background-color: {t["selection"]};
            selection-color: {t["selection_text"]};
        }}
        QLineEdit:focus {{
            border-color: {t["primary"]};
        }}
        QLineEdit:disabled {{
            background-color: {t["surface"]};
            color: {t["text_secondary"]};
        }}
        QPushButton#XvaColorButton {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 0px;
        }}
        QPushButton#XvaColorButton:hover {{
            background-color: {t["surface_hover"]};
            border-color: {t["border_light"]};
        }}
        QTableWidget#XvaLabelSettingsTable QScrollBar:vertical {{
            background: {t["surface"]};
            width: 12px;
            margin: 34px 0px 0px 0px;
            border: none;
        }}
        QTableWidget#XvaLabelSettingsTable QScrollBar::handle:vertical {{
            background: {t["scrollbar"]};
            min-height: 34px;
            border-radius: 5px;
        }}
        QTableWidget#XvaLabelSettingsTable QScrollBar::handle:vertical:hover {{
            background: {t["scrollbar_hover"]};
        }}
        QTableWidget#XvaLabelSettingsTable QScrollBar::add-line:vertical {{
            background: transparent;
            border: none;
            height: 0px;
            image: none;
        }}
        QTableWidget#XvaLabelSettingsTable QScrollBar::sub-line:vertical {{
            background: transparent;
            border: none;
            height: 0px;
            image: none;
        }}
        QTableWidget#XvaLabelSettingsTable QScrollBar::add-page:vertical,
        QTableWidget#XvaLabelSettingsTable QScrollBar::sub-page:vertical {{
            background: {t["background_secondary"]};
        }}
    """


def get_dialog_style():
    t = get_theme()
    return f"""
        QDialog {{
            background-color: {t["background"]};
            color: {t["text"]};
        }}
        QFrame#XvaToolbar {{
            background-color: {t["background_secondary"]};
            border: 1px solid {t["border"]};
            border-radius: {BORDER_RADIUS};
        }}
        QFrame#XvaPreviewPanel,
        QFrame#XvaTimelinePanel {{
            background-color: {t["background"]};
            border: 1px solid {t["border"]};
            border-radius: {BORDER_RADIUS};
        }}
        QFrame#XvaPreviewHeader,
        QFrame#XvaPreviewFooter,
        QFrame#XvaTimelineToolbar {{
            background-color: {t["background_secondary"]};
            border: none;
        }}
        QFrame#XvaZoomPanel {{
            background-color: {t["background_secondary"]};
            border-top: 1px solid {t["border"]};
        }}
        QScrollBar#XvaTimelineScrollBar:horizontal {{
            border: none;
            background-color: {t["background_secondary"]};
            height: 8px;
            margin: 0;
        }}
        QScrollBar#XvaTimelineScrollBar::handle:horizontal {{
            background-color: {t["scrollbar"]};
            min-width: 20px;
            border-radius: 4px;
        }}
        QScrollBar#XvaTimelineScrollBar::handle:horizontal:hover {{
            background-color: {t["scrollbar_hover"]};
        }}
        QScrollBar#XvaTimelineScrollBar::handle:horizontal:disabled {{
            background: transparent;
        }}
        QScrollBar#XvaTimelineScrollBar::add-line:horizontal,
        QScrollBar#XvaTimelineScrollBar::sub-line:horizontal {{
            width: 0;
            border: none;
            background: transparent;
        }}
        QScrollBar#XvaTimelineScrollBar::add-page:horizontal,
        QScrollBar#XvaTimelineScrollBar::sub-page:horizontal {{
            background: transparent;
        }}
        QSplitter::handle {{
            background-color: transparent;
        }}
        QSplitter::handle:horizontal {{
            width: 8px;
        }}
        QLabel {{
            color: {t["text"]};
            background: transparent;
        }}
        QLabel#XvaFileTitle {{
            color: {t["text"]};
            font-size: 14px;
            font-weight: 600;
        }}
        QLabel#XvaStatusLabel {{
            color: {t["text_secondary"]};
            font-family: monospace;
            font-size: 12px;
        }}
        QLabel#XvaCurrentTimeLabel {{
            color: {t["primary"]};
            font-family: monospace;
            font-size: 13px;
            font-weight: 700;
        }}
        QLabel#XvaDurationLabel {{
            color: {t["text"]};
            font-family: monospace;
            font-size: 13px;
        }}
        QLabel#XvaMetaLabel {{
            color: {t["text_secondary"]};
            font-size: 12px;
        }}
        QLabel#XvaTimelineHint {{
            color: {t["text_secondary"]};
            font-size: 12px;
        }}
        QPlainTextEdit#XvaSegmentDescription {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 8px;
            padding: 7px 8px;
            selection-background-color: {t["selection"]};
            selection-color: {t["selection_text"]};
        }}
        QPlainTextEdit#XvaSegmentDescription:focus {{
            border-color: {t["primary"]};
        }}
        QPlainTextEdit#XvaSegmentDescription:disabled {{
            background-color: {t["surface"]};
            color: {t["text_secondary"]};
        }}
        QPlainTextEdit#XvaSegmentDescription QScrollBar:vertical {{
            background: {t["background_secondary"]};
            width: 12px;
            margin: 12px 0px 12px 0px;
            border: none;
        }}
        QPlainTextEdit#XvaSegmentDescription QScrollBar:vertical:disabled {{
            background: transparent;
        }}
        QPlainTextEdit#XvaSegmentDescription QScrollBar::handle:vertical {{
            background: {t["scrollbar"]};
            min-height: 34px;
            border-radius: 5px;
        }}
        QPlainTextEdit#XvaSegmentDescription QScrollBar::handle:vertical:hover {{
            background: {t["scrollbar_hover"]};
        }}
        QPlainTextEdit#XvaSegmentDescription QScrollBar::handle:vertical:disabled {{
            background: transparent;
        }}
        QPlainTextEdit#XvaSegmentDescription QScrollBar::add-line:vertical {{
            background: {t["background_secondary"]};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: bottom;
            height: 12px;
            image: url(:/images/images/caret-down.svg);
        }}
        QPlainTextEdit#XvaSegmentDescription QScrollBar::sub-line:vertical {{
            background: {t["background_secondary"]};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: top;
            height: 12px;
            image: url(:/images/images/caret-up.svg);
        }}
        QPlainTextEdit#XvaSegmentDescription QScrollBar::add-line:vertical:disabled,
        QPlainTextEdit#XvaSegmentDescription QScrollBar::sub-line:vertical:disabled {{
            background: transparent;
            image: none;
        }}
        QPlainTextEdit#XvaSegmentDescription QScrollBar::add-page:vertical,
        QPlainTextEdit#XvaSegmentDescription QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
        QMenu {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            padding: 6px 0;
        }}
        QMenu::item {{
            padding: 7px 28px 7px 32px;
            min-height: 20px;
        }}
        QMenu::item:selected {{
            background-color: {t["surface_hover"]};
        }}
        QMenu::separator {{
            height: 1px;
            background-color: {t["border"]};
            margin: 4px 0;
        }}
        QMenu#XvaSegmentMenu {{
            border-radius: 0px;
            padding: 0px;
        }}
        QMenu#XvaShortcutMenu {{
            border-radius: 6px;
            padding: 0px;
        }}
        QWidget#XvaShortcutPanel {{
            background-color: {t["background_secondary"]};
        }}
        QLabel#XvaShortcutTitle {{
            color: {t["text"]};
            font-size: 13px;
            font-weight: 600;
            background: transparent;
        }}
        QLabel#XvaShortcutHeader {{
            color: {t["text_secondary"]};
            font-size: 11px;
            font-weight: 600;
            background: transparent;
        }}
        QLabel#XvaShortcutKey {{
            color: {t["text"]};
            font-family: monospace;
            font-size: 12px;
            background: transparent;
        }}
        QLabel#XvaShortcutAction {{
            color: {t["text_secondary"]};
            font-size: 12px;
            background: transparent;
        }}
        QMenu#XvaSegmentMenu QWidget#XvaSegmentMenuItem {{
            background: transparent;
            border: none;
            border-radius: 0px;
            color: {t["text"]};
            min-height: 28px;
            max-height: 28px;
            padding: 0px;
        }}
        QMenu#XvaSegmentMenu QWidget#XvaSegmentMenuItem QLabel {{
            color: {t["text"]};
            background: transparent;
        }}
        QMenu#XvaSegmentMenu QWidget#XvaSegmentMenuItem:hover {{
            background-color: {t["surface_hover"]};
        }}
        QMenu#XvaSegmentMenu::separator {{
            height: 1px;
            background-color: {t["border"]};
            margin: 0px;
        }}
    """


def get_slider_style():
    t = get_theme()
    return f"""
        QSlider::groove:horizontal {{
            height: 4px;
            border-radius: 2px;
            background: {t["border_light"]};
        }}
        QSlider::sub-page:horizontal {{
            height: 4px;
            border-radius: 2px;
            background: {t["primary"]};
        }}
        QSlider::handle:horizontal {{
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
            background: {t["background"]};
            border: 1px solid {t["border_light"]};
        }}
        QSlider::handle:horizontal:hover {{
            border-color: {t["primary"]};
        }}
    """


def get_export_dialog_style():
    t = get_theme()
    return f"""
        QDialog {{
            background-color: {t["background"]};
            color: {t["text"]};
        }}
        QLabel, QCheckBox, QRadioButton {{
            color: {t["text"]};
            background: transparent;
            font-size: {FONT_SIZE_NORMAL};
        }}
        QLabel#XvaExportSectionTitle {{
            color: {t["text"]};
            font-size: {FONT_SIZE_NORMAL};
            font-weight: 600;
        }}
        QLabel#XvaExportHint {{
            color: {t["text_secondary"]};
            font-size: {FONT_SIZE_SMALL};
        }}
        QLabel#XvaExportFieldLabel {{
            color: {t["text_secondary"]};
            font-size: {FONT_SIZE_SMALL};
        }}
        QFrame#XvaExportSection {{
            background-color: {t["background_secondary"]};
            border: 1px solid {t["border"]};
            border-radius: 8px;
        }}
        QLineEdit, QSpinBox {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 4px;
            min-height: 30px;
            padding: 0 9px;
            selection-background-color: {t["selection"]};
            selection-color: {t["selection_text"]};
        }}
        QLineEdit:focus, QSpinBox:focus {{
            border: 1px solid {t["highlight"]};
        }}
        QSpinBox::up-button,
        QSpinBox::down-button {{
            width: 0px;
            height: 0px;
            border: none;
        }}
        QSpinBox::up-arrow,
        QSpinBox::down-arrow {{
            image: none;
            width: 0px;
            height: 0px;
        }}
        {_checkbox_indicator_qss()}
        QCheckBox, QRadioButton {{
            spacing: 8px;
        }}
        QRadioButton::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 8px;
            border: 1px solid {t["border_light"]};
            background-color: {t["background"]};
        }}
        QRadioButton::indicator:checked {{
            background-color: {t["primary"]};
            border-color: {t["primary"]};
        }}
    """
