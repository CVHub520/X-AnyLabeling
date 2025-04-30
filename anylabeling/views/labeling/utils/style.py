from anylabeling.views.labeling.utils.qt import new_icon_path


def get_progress_dialog_style(color=None, height=None):
    return f"""
        QProgressDialog {{
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            min-width: 280px;
            min-height: 120px;
            padding: 20px;
            backdrop-filter: blur(20px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08),
                        0 2px 6px rgba(0, 0, 0, 0.04);
        }}
        QProgressBar {{
            border: none;
            border-radius: 4px;
            background-color: rgba(0, 0, 0, 0.05);
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
            color: #1d1d1f;
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 8px;
        }}
        QLabel#detailLabel {{
            color: #86868b;
            font-size: 11px;
            margin-top: 4px;
        }}
        QPushButton {{
            background-color: rgba(255, 255, 255, 0.8);
            border: 0.5px solid rgba(0, 0, 0, 0.1);
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
            color: #0066FF;
            min-width: 82px;
            height: 36px;
            padding: 0 16px;
            margin-top: 16px;
        }}
        QPushButton:hover {{
            background-color: rgba(0, 0, 0, 0.05);
        }}
        QPushButton:pressed {{
            background-color: rgba(0, 0, 0, 0.08);
        }}
    """


def get_msg_box_style():
    return """
        QMessageBox {
            background-color: #ffffff;
            border-radius: 10px;
        }
        QPushButton {
            background-color: #f5f5f7;
            border: 1px solid #d2d2d7;
            border-radius: 8px;
            font-weight: 500;
            min-width: 100px;
            height: 36px;
        }
        QPushButton:hover {
            background-color: #e5e5e5;
        }
    """


def get_ok_btn_style():
    return """
        QPushButton {
            background-color: #0071e3;
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            min-width: 100px;
            height: 36px;
        }
        QPushButton:hover {
            background-color: #0077ED;
        }
        QPushButton:pressed {
            background-color: #0068D0;
        }
    """


def get_cancel_btn_style():
    return """
        QPushButton {
            background-color: #f5f5f7;
            color: #1d1d1f;
            border: 1px solid #d2d2d7;
            border-radius: 8px;
            font-weight: 500;
            min-width: 100px;
            height: 36px;
        }
        QPushButton:hover {
            background-color: #e5e5e5;
        }
        QPushButton:pressed {
            background-color: #d5d5d5;
        }
    """


def get_export_option_style():
    return f"""
        QDialog {{
            background-color: #ffffff;
            border-radius: 8px;
        }}

        QLabel {{
            font-size: 13px;
            background-color: transparent;
            border-left: none;
        }}

        QLineEdit {{
            border: 1px solid #E5E5E5;
            border-radius: 8;
            background-color: #F9F9F9;
            font-size: 13px;
            height: 36px;
            padding-left: 4px;
        }}
        QLineEdit:hover {{
            background-color: #DBDBDB;
            border-radius: 8px;
        }}
        QLineEdit:focus {{
            border: 3px solid "#60A5FA";
            background-color: "#F9F9F9";
        }}

        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 1px solid #d2d2d7;
            background-color: white;
        }}
        QCheckBox::indicator:checked {{
            background-color: white;
            border: 1px solid #d2d2d7;
            image: url({new_icon_path("checkmark", "svg")});
        }}
    """


def get_normal_button_style():
    return """
        QPushButton {
            height: 24px;
            min-width: 80px;
            padding: 5px 8px;
            border-radius: 8px;
            background-color: #f5f5f7;
            border: 1px solid #d2d2d7;
        }
        QPushButton:hover {
            background-color: #e5e5e5;
        }
        QPushButton:pressed {
            background-color: #d5d5d5;
        }
    """


def get_toggle_button_style(button_color: str):
    return f"""
        QPushButton {{
            height: 24px;
            min-width: 80px;
            padding: 5px 8px;
            border-radius: 8px;
            background-color: {button_color};
            border: 1px solid #d2d2d7;
        }}
    """


def get_highlight_button_style():
    return """
        QPushButton {
            height: 24px;
            color: white;
            border: none;
            min-width: 80px;
            padding: 5px 8px;
            border-radius: 8px;
            background-color: #0071e3;
        }
        QPushButton:hover {
            background-color: #0077ED;
        }
        QPushButton:pressed {
            background-color: #0068D0;
        }
    """


def get_spinbox_style():
    return f"""
        QSpinBox {{
            padding: 5px 8px;
            background: white;
            border: 1px solid #d2d2d7;
            border-radius: 6px;
            min-height: 24px;
            selection-background-color: #0071e3;
        }}
        QSpinBox::up-button, QSpinBox::down-button {{
            width: 20px;
            border: none;
            background: #f0f0f0;
        }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
            background: #e0e0e0;
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
    return f"""
        QDoubleSpinBox {{
            padding: 5px 8px;
            background: white;
            border: 1px solid #d2d2d7;
            border-radius: 6px;
            min-height: 24px;
            selection-background-color: #0071e3;
        }}
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            width: 20px;
            border: none;
            background: #f0f0f0;
        }}
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
            background: #e0e0e0;
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


def get_lineedit_style():
    return """
        QLineEdit {
            border: 1px solid #E5E5E5;
            border-radius: 8;
            background-color: #F9F9F9;
            font-size: 13px;
            height: 24px;
            padding: 5px 8px;
        }
        QLineEdit:hover {
            background-color: #DBDBDB;
            border-radius: 8px;
        }
        QLineEdit:focus {
            border: 3px solid "#60A5FA";
            background-color: "#F9F9F9";
        }
    """
