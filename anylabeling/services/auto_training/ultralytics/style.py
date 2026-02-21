from anylabeling.views.labeling.utils.theme import get_theme


def get_ultralytics_dialog_style():
    t = get_theme()
    return f"""
        QWidget {{
            background-color: {t["background"]};
            color: {t["text"]};
        }}
    """


def get_advanced_toggle_btn_style():
    t = get_theme()
    return f"""
        QPushButton {{
            border: none;
            text-align: center;
            font-weight: bold;
            font-size: 10px;
            margin-left: 3px;
            color: {t["text"]};
        }}
        QPushButton:hover {{
            background-color: {t["surface_hover"]};
            border-radius: 3px;
        }}
    """


def get_custom_table_style():
    t = get_theme()
    return f"""
        QTableWidget {{
            border: 1px solid {t["border"]};
            border-radius: 8px;
            background-color: {t["background"]};
            gridline-color: transparent;
            outline: none;
        }}
        QTableWidget::item {{
            padding: 12px 16px;
            border: none;
            border-bottom: 1px solid {t["border"]};
            color: {t["text"]};
            font-size: 13px;
            outline: none;
        }}
        QTableWidget::item:hover {{
            background-color: {t["surface_hover"]};
        }}
        QHeaderView::section {{
            background-color: {t["surface"]};
            color: {t["text_secondary"]};
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 16px;
            border: none;
            border-bottom: 2px solid {t["border"]};
            border-right: 1px solid {t["border"]};
            outline: none;
        }}
        QHeaderView::section:first {{
            border-top-left-radius: 8px;
        }}
        QHeaderView::section:last {{
            border-top-right-radius: 8px;
            border-right: none;
        }}
        QTableWidget::item:alternate {{
            background-color: {t["background_secondary"]};
        }}
        QScrollBar:vertical {{
            background: {t["background_secondary"]};
            width: 8px;
            border-radius: 4px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: {t["scrollbar"]};
            border-radius: 4px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {t["scrollbar_hover"]};
        }}
        QScrollBar::add-line:vertical {{
            height: 0px;
            subcontrol-position: bottom;
            subcontrol-origin: margin;
        }}
        QScrollBar::sub-line:vertical {{
            height: 0px;
            subcontrol-position: top;
            subcontrol-origin: margin;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
    """


def get_image_row_label_style():
    t = get_theme()
    return f"font-weight: bold; margin-bottom: 5px; color: {t['text']};"


def get_image_label_style():
    t = get_theme()
    return f"""
        QLabel {{
            border: 1px solid {t["border_light"]};
            background-color: {t["background_secondary"]};
        }}
    """


def get_log_display_style():
    t = get_theme()
    return f"""
        QTextEdit {{
            background-color: {t["surface"]};
            color: {t["text"]};
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 11px;
            border: 1px solid {t["border"]};
            padding: 8px;
        }}
    """


def get_progress_bar_style():
    t = get_theme()
    return f"""
        QProgressBar {{
            border: 1px solid {t["border_light"]};
            text-align: center;
            height: 20px;
            background-color: {t["surface"]};
            color: {t["text"]};
        }}
        QProgressBar::chunk {{
            background-color: {t["primary"]};
        }}
    """


def get_status_label_style(color=None):
    if color is None:
        color = get_theme()["text_secondary"]
    return f"""
        font-size: 14px;
        font-weight: bold;
        color: {color};
    """
