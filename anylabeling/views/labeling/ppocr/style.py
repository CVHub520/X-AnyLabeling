from __future__ import annotations

from anylabeling.views.labeling.utils.theme import get_mode, get_theme

from .config import (
    PPOCR_COLOR_EDITED,
    PPOCR_COLOR_TEXT,
    PPOCR_COLOR_OVERLAY,
)


def _is_dark_mode() -> bool:
    return get_mode() == "dark"


def _panel_background() -> str:
    return (
        get_theme()["background_secondary"]
        if _is_dark_mode()
        else "rgb(247, 249, 255)"
    )


def _header_background() -> str:
    return get_theme()["surface"] if _is_dark_mode() else "rgb(228, 236, 255)"


def _selected_background() -> str:
    return (
        get_theme()["surface_pressed"]
        if _is_dark_mode()
        else "rgb(241, 245, 255)"
    )


def _card_background() -> str:
    return get_theme()["surface"] if _is_dark_mode() else "rgb(255, 255, 255)"


def get_dialog_style() -> str:
    t = get_theme()
    return f"""
        QDialog {{
            background-color: {t["background_secondary"]};
            color: {t["text"]};
        }}
        QSplitter::handle {{
            background: {t["border"]};
        }}
        QListWidget {{
            background: {t["background"]};
            border: none;
            outline: none;
        }}
        QListWidget::item {{
            background: transparent;
            padding: 0px;
            border: none;
        }}
        QListWidget::item:hover,
        QListWidget::item:selected,
        QListWidget::item:selected:active,
        QListWidget::item:selected:!active {{
            background: transparent;
            border: none;
        }}
        QTabWidget::pane {{
            border: 1px solid {t["border"]};
            background: {t["background"]};
            border-radius: 8px;
        }}
        QTabBar::tab {{
            background: transparent;
            color: {t["text_secondary"]};
            padding: 8px 16px;
            margin-right: 8px;
        }}
        QTabBar::tab:selected {{
            color: {PPOCR_COLOR_TEXT};
            border-bottom: 2px solid {PPOCR_COLOR_TEXT};
        }}
        QScrollArea {{
            border: none;
            background: transparent;
        }}
        QTextEdit, QPlainTextEdit, QLineEdit, QComboBox {{
            background: {t["background"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            padding: 6px 10px;
            selection-background-color: {PPOCR_COLOR_TEXT};
        }}
        QComboBox::drop-down {{
            width: 28px;
            border: none;
        }}
    """


def get_primary_button_style() -> str:
    return """
        QPushButton {
            background: rgb(70, 88, 255);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: rgb(53, 71, 235);
        }
        QPushButton:disabled {
            background: rgb(180, 186, 220);
        }
    """


def get_secondary_button_style() -> str:
    t = get_theme()
    return f"""
        QPushButton {{
            background: {t["background"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            color: {PPOCR_COLOR_TEXT};
            border-color: {PPOCR_COLOR_TEXT};
        }}
        QPushButton:disabled {{
            color: {t["text_secondary"]};
        }}
    """


def get_danger_button_style() -> str:
    return """
        QPushButton {
            background: rgb(255, 69, 58);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: rgb(230, 44, 33);
        }
    """


def get_recents_item_style(selected: bool) -> str:
    t = get_theme()
    bg = _selected_background() if selected else "transparent"
    border = t["primary"] if selected else "transparent"
    return f"""
        QWidget {{
            background: {bg};
            border: 1px solid {border};
            border-radius: 10px;
        }}
        QLabel, QPushButton {{
            background: transparent;
            border: none;
        }}
    """


def get_icon_button_style() -> str:
    t = get_theme()
    return f"""
        QPushButton {{
            background: transparent;
            border: none;
            border-radius: 8px;
            color: {t["text"]};
            padding: 0px;
            min-width: 0px;
            min-height: 0px;
            text-align: center;
        }}
        QPushButton:hover {{
            background: {_selected_background()};
            color: {PPOCR_COLOR_TEXT};
        }}
    """


def get_sidebar_icon_button_style() -> str:
    t = get_theme()
    return f"""
        QPushButton {{
            background: transparent;
            border: none;
            border-radius: 8px;
            padding: 0px;
            color: {t["text"]};
            min-width: 0px;
            min-height: 0px;
            text-align: center;
        }}
        QPushButton:hover {{
            background: {_selected_background()};
            color: {PPOCR_COLOR_TEXT};
        }}
        QPushButton:pressed {{
            background: {_selected_background()};
        }}
    """


def get_sidebar_panel_style() -> str:
    t = get_theme()
    return f"""
        QWidget#PPOCRSidebar {{
            background: {t["background"]};
            border-right: 1px solid {t["border"]};
        }}
        QFrame#PPOCRSidebarDivider {{
            background: {t["border"]};
            min-height: 1px;
            max-height: 1px;
            border: none;
        }}
    """


def get_sidebar_tab_style(active: bool) -> str:
    color = PPOCR_COLOR_TEXT if active else get_theme()["text_secondary"]
    weight = 700 if active else 500
    return f"""
        QPushButton {{
            background: transparent;
            border: none;
            color: {color};
            font-size: 13px;
            font-weight: {weight};
            padding: 0px;
        }}
        QPushButton:hover {{
            color: {PPOCR_COLOR_TEXT};
        }}
    """


def get_sidebar_search_style() -> str:
    t = get_theme()
    return f"""
        QLineEdit {{
            background: {t["background"]};
            color: {t["text"]};
            border: 1px solid {t["border_light"]};
            border-radius: 10px;
            padding: 8px 12px;
        }}
    """


def get_chip_button_style() -> str:
    t = get_theme()
    return f"""
        QPushButton {{
            background: {t["background"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 8px;
            padding: 6px 14px;
        }}
        QPushButton:hover {{
            color: {PPOCR_COLOR_TEXT};
        }}
    """


def get_card_style(
    border_color: str,
    edited: bool,
    active: bool = False,
) -> str:
    border = PPOCR_COLOR_EDITED if edited else border_color
    resolved_border = border if (active or edited) else "transparent"
    return f"""
        QFrame#PPOCRBlockCardContentFrame {{
            background: {_card_background()};
            border: 1px solid {resolved_border};
            border-radius: 0px;
        }}
    """


def get_card_label_style(category_color: str) -> str:
    return f"""
        QLabel {{
            background: {category_color};
            color: white;
            border: none;
            border-radius: 0px;
            padding: 6px 9px;
            font-size: 10px;
            font-weight: 600;
        }}
    """


def get_floating_action_bar_style() -> str:
    t = get_theme()
    return f"""
        QFrame {{
            background: {_card_background()};
            border: 1px solid {t["border"]};
            border-radius: 18px;
        }}
    """


def get_floating_action_button_style() -> str:
    t = get_theme()
    return f"""
        QPushButton {{
            background: transparent;
            color: {t["text"]};
            border: none;
            border-radius: 14px;
            padding: 7px 10px;
            font-size: 13px;
            font-weight: 500;
            text-align: center;
        }}
        QPushButton:hover {{
            background: transparent;
            color: {PPOCR_COLOR_TEXT};
        }}
    """


def get_section_panel_style() -> str:
    t = get_theme()
    return f"""
        QWidget {{
            background: {t["background"]};
            border: 1px solid {t["border"]};
            border-radius: 8px;
        }}
    """


def get_preview_panel_style() -> str:
    t = get_theme()
    panel_background = _panel_background()
    return f"""
        QWidget#PPOCRPreviewPanel {{
            background: {panel_background};
        }}
        QFrame#PPOCRPreviewFrame {{
            background: {t["background"]};
            border: none;
            border-radius: 0px;
        }}
        QFrame#PPOCRPreviewContentFrame {{
            background: transparent;
            border: none;
            border-radius: 0px;
        }}
        QFrame#PPOCRSourceFileDivider {{
            background: {t["border"]};
            min-height: 1px;
            max-height: 1px;
            border: none;
        }}
        QFrame#PPOCRPageControlDivider {{
            background: {t["border"]};
            min-width: 1px;
            max-width: 1px;
            border: none;
        }}
        QScrollArea#PPOCRPreviewScrollArea {{
            background: transparent;
            border: none;
        }}
        QScrollArea#PPOCRPreviewScrollArea QScrollBar:vertical {{
            background: {panel_background};
            width: 12px;
            margin: 12px 0px 12px 0px;
            border: none;
        }}
        QScrollArea#PPOCRPreviewScrollArea QScrollBar::handle:vertical {{
            background: {t["scrollbar"]};
            min-height: 34px;
            border-radius: 5px;
        }}
        QScrollArea#PPOCRPreviewScrollArea QScrollBar::add-line:vertical {{
            background: {panel_background};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: bottom;
            height: 12px;
            image: url(:/images/images/caret-down.svg);
        }}
        QScrollArea#PPOCRPreviewScrollArea QScrollBar::sub-line:vertical {{
            background: {panel_background};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: top;
            height: 12px;
            image: url(:/images/images/caret-up.svg);
        }}
        QScrollArea#PPOCRPreviewScrollArea QScrollBar::add-page:vertical,
        QScrollArea#PPOCRPreviewScrollArea QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
        QScrollArea#PPOCRPreviewScrollArea QScrollBar:horizontal {{
            background: {panel_background};
            height: 12px;
            margin: 0px 12px 0px 12px;
            border: none;
        }}
        QScrollArea#PPOCRPreviewScrollArea QScrollBar::handle:horizontal {{
            background: {t["scrollbar"]};
            min-width: 34px;
            border-radius: 5px;
        }}
        QScrollArea#PPOCRPreviewScrollArea QScrollBar::add-line:horizontal {{
            background: {panel_background};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: right;
            width: 12px;
            image: url(:/images/images/caret-right.svg);
        }}
        QScrollArea#PPOCRPreviewScrollArea QScrollBar::sub-line:horizontal {{
            background: {panel_background};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: left;
            width: 12px;
            image: url(:/images/images/caret-left.svg);
        }}
        QScrollArea#PPOCRPreviewScrollArea QScrollBar::add-page:horizontal,
        QScrollArea#PPOCRPreviewScrollArea QScrollBar::sub-page:horizontal {{
            background: transparent;
        }}
        QScrollArea#PPOCRPreviewScrollArea::corner {{
            background: {panel_background};
            border: none;
            width: 12px;
            height: 12px;
        }}
    """


def get_source_file_info_style() -> str:
    t = get_theme()
    return f"""
        QFrame#PPOCRSourceFileInfoFrame {{
            background: {t["background"]};
            border: none;
        }}
        QLabel#PPOCRSourceFileTitle {{
            background: {_header_background()};
            color: {PPOCR_COLOR_TEXT};
            border: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            border-bottom-left-radius: 0px;
            border-bottom-right-radius: 0px;
            padding: 11px;
            font-size: 12px;
            font-weight: 600;
        }}
        QLabel#PPOCRSourceFileName {{
            color: {t["text"]};
            border: none;
            font-size: 13px;
            font-weight: 500;
        }}
        QLabel#PPOCRSourceFileSize {{
            color: {t["text_secondary"]};
            border: none;
            font-size: 11px;
            font-weight: 400;
        }}
    """


def get_model_combo_style() -> str:
    t = get_theme()
    selected_background = _selected_background()
    return f"""
        QComboBox {{
            combobox-popup: 0;
            background: transparent;
            color: {PPOCR_COLOR_TEXT};
            border: none;
            font-size: 13px;
            font-weight: 500;
            min-height: 28px;
            padding: 0px 8px 0px 0px;
        }}
        QComboBox:hover {{
            background: transparent;
        }}
        QComboBox:disabled {{
            color: {t["text_secondary"]};
        }}
        QComboBox::drop-down {{
            background: transparent;
            border: none;
            width: 10px;
            subcontrol-origin: padding;
            subcontrol-position: center right;
        }}
        QComboBox::down-arrow {{
            image: url(:/images/images/caret-down.svg);
            width: 8px;
            height: 8px;
        }}
        QComboBox QAbstractItemView {{
            background: {t["background"]};
            color: {t["text"]};
            border: 1px solid {t["border"]};
            border-radius: 0px;
            margin: 0px;
            padding: 0px;
            outline: none;
            selection-background-color: {selected_background};
            selection-color: {PPOCR_COLOR_TEXT};
        }}
        QComboBox QAbstractItemView::item {{
            background: {t["background"]};
            min-height: 24px;
            padding: 2px 8px;
            border-radius: 0px;
            border-bottom: 1px solid {t["border"]};
        }}
        QComboBox QAbstractItemView::item:hover {{
            background: {t["surface_hover"]};
        }}
        QComboBox QAbstractItemView::item:selected {{
            background: {selected_background};
            color: {PPOCR_COLOR_TEXT};
        }}
    """


def get_result_header_style() -> str:
    t = get_theme()
    panel_background = _panel_background()
    selected_background = _selected_background()
    return f"""
        QWidget#PPOCRResultPanel {{
            background: {panel_background};
        }}
        QFrame#PPOCRParsingModelHeader {{
            background: {_header_background()};
            border: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            border-bottom-left-radius: 0px;
            border-bottom-right-radius: 0px;
        }}
        QLabel#PPOCRParsingModelTitle {{
            background: transparent;
            color: {PPOCR_COLOR_TEXT};
            border: none;
            font-size: 12px;
            font-weight: 600;
        }}
        QComboBox#PPOCRParsingModelCombo {{
            combobox-popup: 0;
            background: transparent;
            color: {PPOCR_COLOR_TEXT};
            border: none;
            font-size: 10pt;
            font-weight: 500;
            min-height: 28px;
            padding: 0px 8px 0px 0px;
        }}
        QComboBox#PPOCRParsingModelCombo:hover {{
            background: transparent;
        }}
        QComboBox#PPOCRParsingModelCombo:disabled {{
            color: {t["text_secondary"]};
        }}
        QComboBox#PPOCRParsingModelCombo::drop-down {{
            background: transparent;
            subcontrol-origin: padding;
            subcontrol-position: center right;
            width: 10px;
            border: none;
        }}
        QComboBox#PPOCRParsingModelCombo::down-arrow {{
            image: url(:/images/images/caret-down.svg);
            width: 8px;
            height: 8px;
        }}
        QComboBox#PPOCRParsingModelCombo QAbstractItemView {{
            background: {t["background"]};
            color: {t["text"]};
            font-size: 10pt;
            border: 1px solid {t["border"]};
            border-radius: 0px;
            margin: 0px;
            padding: 0px;
            outline: none;
            selection-background-color: {selected_background};
            selection-color: {PPOCR_COLOR_TEXT};
        }}
        QComboBox#PPOCRParsingModelCombo QAbstractItemView::item {{
            background: {t["background"]};
            min-height: 24px;
            padding: 2px 8px;
            border-radius: 0px;
            border-bottom: 1px solid {t["border"]};
        }}
        QComboBox#PPOCRParsingModelCombo QAbstractItemView::item:hover {{
            background: {t["surface_hover"]};
        }}
        QComboBox#PPOCRParsingModelCombo QAbstractItemView::item:selected {{
            background: {selected_background};
            color: {PPOCR_COLOR_TEXT};
        }}
        QFrame#PPOCRResultInfoFrame {{
            background: {t["background"]};
            border: none;
        }}
        QPushButton#PPOCRResultModeButton {{
            background: transparent;
            border: none;
            border-radius: 6px;
            color: {t["text_secondary"]};
            font-size: 13px;
            font-weight: 500;
            padding: 6px 10px;
            text-align: left;
        }}
        QPushButton#PPOCRResultModeButton:hover {{
            color: {PPOCR_COLOR_TEXT};
        }}
        QPushButton#PPOCRResultModeButton:checked {{
            background: {selected_background};
            color: {PPOCR_COLOR_TEXT};
        }}
        QFrame#PPOCRResultActionsFrame {{
            background: transparent;
            border: none;
        }}
        QFrame#PPOCRResultActionsDivider {{
            background: {t["border"]};
            min-width: 1px;
            max-width: 1px;
            border: none;
        }}
        QPushButton#PPOCRResultActionButton {{
            background: transparent;
            border: none;
            border-radius: 8px;
            padding: 0px;
            color: {t["text"]};
        }}
        QPushButton#PPOCRResultActionButton:hover {{
            background: {selected_background};
            color: {PPOCR_COLOR_TEXT};
        }}
        QFrame#PPOCRResultDivider {{
            background: {t["border"]};
            min-height: 1px;
            max-height: 1px;
            border: none;
        }}
        QStackedWidget#PPOCRResultContentStack {{
            background: {t["background"]};
            border: none;
        }}
        QWidget#PPOCRResultDocumentTab {{
            background: {t["background"]};
            border: none;
        }}
        QScrollArea#PPOCRResultCardsScrollArea {{
            background: {t["background"]};
            border: none;
        }}
        QScrollArea#PPOCRResultCardsScrollArea QScrollBar:vertical,
        QPlainTextEdit#PPOCRResultJsonViewer QScrollBar:vertical {{
            background: {panel_background};
            width: 12px;
            margin: 12px 0px 12px 0px;
            border: none;
        }}
        QScrollArea#PPOCRResultCardsScrollArea QScrollBar::handle:vertical,
        QPlainTextEdit#PPOCRResultJsonViewer QScrollBar::handle:vertical {{
            background: {t["scrollbar"]};
            min-height: 34px;
            border-radius: 5px;
        }}
        QScrollArea#PPOCRResultCardsScrollArea QScrollBar::add-line:vertical,
        QPlainTextEdit#PPOCRResultJsonViewer QScrollBar::add-line:vertical {{
            background: {panel_background};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: bottom;
            height: 12px;
            image: url(:/images/images/caret-down.svg);
        }}
        QScrollArea#PPOCRResultCardsScrollArea QScrollBar::sub-line:vertical,
        QPlainTextEdit#PPOCRResultJsonViewer QScrollBar::sub-line:vertical {{
            background: {panel_background};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: top;
            height: 12px;
            image: url(:/images/images/caret-up.svg);
        }}
        QScrollArea#PPOCRResultCardsScrollArea QScrollBar::add-page:vertical,
        QScrollArea#PPOCRResultCardsScrollArea QScrollBar::sub-page:vertical,
        QPlainTextEdit#PPOCRResultJsonViewer QScrollBar::add-page:vertical,
        QPlainTextEdit#PPOCRResultJsonViewer QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
        QScrollArea#PPOCRResultCardsScrollArea QScrollBar:horizontal,
        QPlainTextEdit#PPOCRResultJsonViewer QScrollBar:horizontal {{
            background: {panel_background};
            height: 12px;
            margin: 0px 12px 0px 12px;
            border: none;
        }}
        QScrollArea#PPOCRResultCardsScrollArea QScrollBar::handle:horizontal,
        QPlainTextEdit#PPOCRResultJsonViewer QScrollBar::handle:horizontal {{
            background: {t["scrollbar"]};
            min-width: 34px;
            border-radius: 5px;
        }}
        QScrollArea#PPOCRResultCardsScrollArea QScrollBar::add-line:horizontal,
        QPlainTextEdit#PPOCRResultJsonViewer QScrollBar::add-line:horizontal {{
            background: {panel_background};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: right;
            width: 12px;
            image: url(:/images/images/caret-right.svg);
        }}
        QScrollArea#PPOCRResultCardsScrollArea QScrollBar::sub-line:horizontal,
        QPlainTextEdit#PPOCRResultJsonViewer QScrollBar::sub-line:horizontal {{
            background: {panel_background};
            border: none;
            subcontrol-origin: margin;
            subcontrol-position: left;
            width: 12px;
            image: url(:/images/images/caret-left.svg);
        }}
        QScrollArea#PPOCRResultCardsScrollArea QScrollBar::add-page:horizontal,
        QScrollArea#PPOCRResultCardsScrollArea QScrollBar::sub-page:horizontal,
        QPlainTextEdit#PPOCRResultJsonViewer QScrollBar::add-page:horizontal,
        QPlainTextEdit#PPOCRResultJsonViewer QScrollBar::sub-page:horizontal {{
            background: transparent;
        }}
        QScrollArea#PPOCRResultCardsScrollArea::corner,
        QPlainTextEdit#PPOCRResultJsonViewer::corner {{
            background: {panel_background};
            border: none;
            width: 12px;
            height: 12px;
        }}
        QWidget#PPOCRResultCardsContainer {{
            background: {t["background"]};
            border: none;
        }}
        QWidget#PPOCRPageDivider {{
            background: transparent;
            border: none;
        }}
        QLabel#PPOCRPageDividerText {{
            background: transparent;
            border: none;
            color: {t["text_secondary"]};
            font-size: 12px;
            font-weight: 500;
            padding: 0px 4px;
        }}
        QFrame#PPOCRPageDividerLineLeft {{
            min-height: 1px;
            max-height: 1px;
            border: none;
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(220, 228, 244, 0),
                stop:1 {t["border"]}
            );
        }}
        QFrame#PPOCRPageDividerLineRight {{
            min-height: 1px;
            max-height: 1px;
            border: none;
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 {t["border"]},
                stop:1 rgba(220, 228, 244, 0)
            );
        }}
        QPlainTextEdit#PPOCRResultJsonViewer {{
            background: {t["background"]};
            color: {t["text"]};
            border: none;
            padding: 0px;
        }}
    """


def get_page_control_style() -> str:
    t = get_theme()
    return f"""
        QWidget#PageControl {{
            background: {t["background"]};
            border: 1px solid {t["border"]};
            border-radius: 16px;
        }}
    """


def get_overlay_label_style() -> str:
    return f"""
        QLabel {{
            color: {PPOCR_COLOR_OVERLAY};
            font-size: 13px;
            font-weight: 600;
            background: transparent;
        }}
    """
