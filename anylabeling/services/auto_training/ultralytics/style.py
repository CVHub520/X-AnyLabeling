
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
