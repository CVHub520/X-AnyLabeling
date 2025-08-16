def get_advanced_toggle_btn_style():
    return """
        QPushButton {
            border: none;
            text-align: center;
            font-weight: bold;
            font-size: 10px;
            margin-left: 3px;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
            border-radius: 3px;
        }
    """


def get_custom_table_style():
    return """
        QTableWidget {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background-color: white;
            gridline-color: transparent;
            outline: none;
        }
        QTableWidget::item {
            padding: 12px 16px;
            border: none;
            border-bottom: 1px solid #f3f4f6;
            color: #374151;
            font-size: 13px;
            outline: none;
        }
        QTableWidget::item:hover {
            background-color: #f9fafb;
        }
        QHeaderView::section {
            background-color: #f8fafc;
            color: #6b7280;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 16px;
            border: none;
            border-bottom: 2px solid #e5e7eb;
            border-right: 1px solid #f3f4f6;
            outline: none;
        }
        QHeaderView::section:first {
            border-top-left-radius: 8px;
        }
        QHeaderView::section:last {
            border-top-right-radius: 8px;
            border-right: none;
        }
        QTableWidget::item:alternate {
            background-color: #fafafa;
        }
        QScrollBar:vertical {
            background: #f3f4f6;
            width: 8px;
            border-radius: 4px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #d1d5db;
            border-radius: 4px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: #9ca3af;
        }
        QScrollBar::add-line:vertical {
            height: 0px;
            subcontrol-position: bottom;
            subcontrol-origin: margin;
        }
        QScrollBar::sub-line:vertical {
            height: 0px;
            subcontrol-position: top;
            subcontrol-origin: margin;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: transparent;
        }
    """


def get_image_row_label_style():
    return "font-weight: bold; margin-bottom: 5px;"


def get_image_label_style():
    return """
        QLabel {
            border: 1px solid #d2d2d7;
            background-color: #f8f9fa;
        }
    """


def get_log_display_style():
    return """
        QTextEdit {
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 11px;
            border: 1px solid #3c3c3c;
            padding: 8px;
        }
    """


def get_progress_bar_style():
    return """
        QProgressBar {
            border: 1px solid #d2d2d7;
            text-align: center;
            height: 20px;
        }
        QProgressBar::chunk {
            background-color: #0071e3;
        }
    """


def get_status_label_style(color="#6c757d"):
    return f"""
        font-size: 14px;
        font-weight: bold;
        color: {color};
        /* qproperty-alignment: AlignCenter; */
    """
