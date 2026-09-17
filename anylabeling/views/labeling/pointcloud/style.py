from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import (
    get_checkbox_indicator_style,
    get_dialog_style,
)
from anylabeling.views.labeling.utils.theme import get_theme


def get_pointcloud_style():
    theme = get_theme()
    indicators = get_checkbox_indicator_style().replace(
        "QCheckBox", "QListWidget#pointcloudList"
    )
    return (
        get_dialog_style()
        + f"""
        QMainWindow#pointcloudWorkspace {{
            background-color: {theme["background"]};
            color: {theme["text"]};
        }}
        QFrame#pointcloudToolSeparator {{
            background-color: {theme["border"]};
            border: none;
        }}
        QScrollArea#pointcloudToolScroll QScrollBar:horizontal {{
            background-color: {theme["background"]};
            height: 16px;
            margin: 4px 8px;
        }}
        QScrollArea#pointcloudToolScroll QScrollBar::handle:horizontal {{
            background-color: {theme["scrollbar"]};
            border-radius: 4px;
            min-width: 24px;
        }}
        QScrollArea#pointcloudToolScroll QScrollBar::handle:horizontal:hover {{
            background-color: {theme["scrollbar_hover"]};
        }}
        QScrollArea#pointcloudToolScroll QScrollBar::add-line:horizontal,
        QScrollArea#pointcloudToolScroll QScrollBar::sub-line:horizontal {{
            width: 0;
        }}
        QScrollArea#pointcloudToolScroll QScrollBar::add-page:horizontal,
        QScrollArea#pointcloudToolScroll QScrollBar::sub-page:horizontal {{
            background: transparent;
        }}
        QFrame#pointcloudPanel {{
            background-color: {theme["background"]};
            border: none;
        }}
        QWidget#pointcloudPage {{
            background-color: {theme["background"]};
        }}
        QFrame#pointcloudListPanel {{
            background-color: {theme["surface"]};
            border: 1px solid {theme["border"]};
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            border-bottom-left-radius: 0;
            border-bottom-right-radius: 0;
        }}
        QListWidget#pointcloudList[integratedPanel="true"] {{
            background-color: {theme["background"]};
            border: none;
            border-radius: 0;
        }}
        QMainWindow#pointcloudWorkspace QPushButton {{
            min-width: 0;
        }}
        QLabel#pointcloudSectionTitle {{
            color: {theme["text"]};
            font-weight: 600;
            padding: 2px 0;
        }}
        QLabel#pointcloudMuted {{
            color: {theme["text_secondary"]};
        }}
        QLabel#pointcloudTarget {{
            color: {theme["text"]};
            background-color: {theme["surface"]};
            border: 1px solid {theme["border"]};
            border-radius: 6px;
            padding: 8px;
        }}
        QListWidget#pointcloudList {{
            background-color: {theme["background_secondary"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
            border-radius: 6px;
            outline: none;
            selection-background-color: {theme["selection"]};
            selection-color: {theme["selection_text"]};
        }}
        QListWidget#pointcloudList::item {{
            min-height: 22px;
            padding: 3px 6px;
        }}
        QListWidget#pointcloudList::item:selected {{
            background-color: {theme["selection"]};
            color: {theme["selection_text"]};
        }}
        QListWidget#pointcloudList::item:hover:!selected {{
            background-color: {theme["surface_hover"]};
        }}
        QTabWidget#pointcloudTabs::pane {{
            background-color: {theme["background"]};
            border: none;
            border-top: 1px solid {theme["border"]};
        }}
        QTabWidget#pointcloudTabs QTabBar::tab {{
            background-color: {theme["background"]};
            color: {theme["text_secondary"]};
            border: none;
            border-bottom: 2px solid transparent;
            height: 19px;
            padding: 10px 12px;
        }}
        QTabWidget#pointcloudTabs QTabBar::tab:selected {{
            color: {theme["text"]};
            font-weight: 600;
        }}
        QTabWidget#pointcloudTabs QTabBar::tab:hover:!selected {{
            background-color: {theme["surface_hover"]};
            color: {theme["text"]};
        }}
        QToolBar#pointcloudFileTools,
        QFrame#pointcloudPanelHeader,
        QFrame#pointcloudViewTools {{
            background-color: {theme["background"]};
            border: none;
            border-bottom: 1px solid {theme["border"]};
            spacing: 4px;
            padding: 4px 8px;
        }}
        QFrame#pointcloudViewTools,
        QFrame#pointcloudPanelHeader {{
            padding: 0;
        }}
        QToolBar#pointcloudFileTools {{
            spacing: 0;
            padding: 4px 6px;
        }}
        QMainWindow#pointcloudWorkspace QToolBar#pointcloudFileTools QToolButton {{
            padding: 4px 0;
            margin: 0;
            border: none;
        }}
        QMainWindow#pointcloudWorkspace QToolBar#pointcloudFileTools QToolButton:disabled {{
            background-color: transparent;
            border: none;
            color: {theme["text_secondary"]};
        }}
        QMainWindow#pointcloudWorkspace QToolBar#pointcloudFileTools QToolButton#pointcloudHelpButton {{
            min-width: 18px;
            min-height: 18px;
            padding: 3px;
            border: 1px solid transparent;
            border-radius: 4px;
        }}
        QMainWindow#pointcloudWorkspace QToolBar#pointcloudFileTools QToolButton#pointcloudHelpButton:hover {{
            background-color: {theme["surface_hover"]};
            border-color: {theme["border"]};
        }}
        QMainWindow#pointcloudWorkspace QToolBar#pointcloudFileTools QToolButton#pointcloudHelpButton:pressed {{
            background-color: {theme["surface_pressed"]};
        }}
        QToolBar#pointcloudFileTools::separator,
        QToolBar#pointcloudViewTools::separator {{
            background-color: {theme["border"]};
            width: 1px;
            margin: 6px 4px;
        }}
        QMainWindow#pointcloudWorkspace QToolButton {{
            background-color: transparent;
            color: {theme["text"]};
            border: 1px solid transparent;
            border-radius: 6px;
            min-height: 22px;
            padding: 4px 8px;
        }}
        QMainWindow#pointcloudWorkspace QToolButton[compact="true"] {{
            min-height: 18px;
            padding: 2px 6px;
        }}
        QMainWindow#pointcloudWorkspace QToolButton#pointcloudIconButton {{
            min-height: 18px;
            min-width: 18px;
            padding: 5px;
        }}
        QSpinBox#pointcloudFrameNumber {{
            min-width: 18px;
            min-height: 18px;
            padding: 3px 2px;
        }}
        QMainWindow#pointcloudWorkspace QToolButton#pointcloudIconButton[navigation="true"] {{
            min-height: 16px;
            min-width: 16px;
            padding: 3px;
        }}
        QMainWindow#pointcloudWorkspace QToolButton#pointcloudIconButton:hover:!checked,
        QMainWindow#pointcloudWorkspace QToolButton#pointcloudIconButton:disabled {{
            background-color: transparent;
            border-color: transparent;
        }}
        QMainWindow#pointcloudWorkspace QToolButton#pointcloudMenuButton {{
            padding-right: 22px;
        }}
        QMainWindow#pointcloudWorkspace QToolButton#pointcloudIconButton[panelHeader="true"] {{
            padding: 3px;
            border-radius: 4px;
        }}
        QMainWindow#pointcloudWorkspace QToolButton#pointcloudIconButton[panelHeader="true"]:hover:enabled {{
            background-color: {theme["surface_hover"]};
            border-color: {theme["border"]};
        }}
        QMainWindow#pointcloudWorkspace QToolButton#pointcloudIconButton[panelHeader="true"]:pressed:enabled {{
            background-color: {theme["surface_pressed"]};
            border-color: {theme["border"]};
        }}
        QMainWindow#pointcloudWorkspace QToolButton#pointcloudMenuButton::menu-indicator {{
            image: url({new_icon_path("caret-down", "svg")});
            subcontrol-origin: padding;
            subcontrol-position: right center;
            width: 12px;
            height: 12px;
            right: 6px;
        }}
        QMainWindow#pointcloudWorkspace QToolButton:hover {{
            background-color: {theme["surface_hover"]};
        }}
        QMainWindow#pointcloudWorkspace QToolButton:pressed {{
            background-color: {theme["surface_pressed"]};
        }}
        QMainWindow#pointcloudWorkspace QToolButton:checked {{
            background-color: {theme["surface"]};
            border-color: {theme["primary"]};
            color: {theme["primary"]};
        }}
        QMainWindow#pointcloudWorkspace QToolButton:focus {{
            border-color: {theme["highlight"]};
        }}
        QMainWindow#pointcloudWorkspace QToolButton[variant="primary"] {{
            background-color: {theme["primary"]};
            border-color: {theme["primary"]};
            color: white;
            padding: 4px 14px;
        }}
        QMainWindow#pointcloudWorkspace QToolButton[variant="primary"]:hover {{
            background-color: {theme["primary_hover"]};
            border-color: {theme["primary_hover"]};
        }}
        QMainWindow#pointcloudWorkspace QToolButton[variant="primary"]:pressed {{
            background-color: {theme["primary_pressed"]};
            border-color: {theme["primary_pressed"]};
        }}
        QMainWindow#pointcloudWorkspace QToolButton[variant="primary"]:focus {{
            border-color: {theme["highlight"]};
        }}
        QMainWindow#pointcloudWorkspace QToolButton:disabled,
        QMainWindow#pointcloudWorkspace QToolButton[variant="primary"]:disabled {{
            background-color: {theme["surface"]};
            border-color: {theme["border"]};
            color: {theme["text_secondary"]};
        }}
        QMainWindow#pointcloudWorkspace QScrollArea {{
            background-color: {theme["background"]};
            border: none;
        }}
        QMainWindow#pointcloudWorkspace QSplitter::handle {{
            background-color: {theme["border"]};
        }}
        QMainWindow#pointcloudWorkspace QSplitter::handle:horizontal {{
            width: 1px;
        }}
        QMainWindow#pointcloudWorkspace QSplitter::handle:vertical {{
            height: 1px;
        }}
        QMainWindow#pointcloudWorkspace QStatusBar {{
            background-color: {theme["background"]};
            color: {theme["text_secondary"]};
            border-top: 1px solid {theme["border"]};
            padding: 3px 8px;
        }}
        QMainWindow#pointcloudWorkspace QStatusBar::item {{
            border: none;
        }}
        QMenu {{
            background-color: {theme["background"]};
            color: {theme["text"]};
            border: 1px solid {theme["border"]};
            border-radius: 0;
            padding: 0;
        }}
        QMenu::item {{
            padding: 6px 24px 6px 8px;
            margin: 0;
            border-radius: 0;
        }}
        QMenu::item:selected {{
            background-color: {theme["surface_hover"]};
            color: {theme["text"]};
        }}
        QMenu::item:disabled {{
            color: {theme["text_secondary"]};
        }}
        QMenu::icon {{
            padding: 2px;
        }}
        QMenu::indicator {{
            width: 14px;
            height: 14px;
        }}
        QMenu::indicator:non-exclusive:checked,
        QMenu::indicator:exclusive:checked {{
            image: url(:/images/images/checkmark.svg);
        }}
        QMenu::separator {{
            height: 1px;
            background-color: {theme["border"]};
            margin: 4px 0;
        }}
        """
        + indicators
        + """
        QListWidget#pointcloudList::indicator {
            width: 14px;
            height: 14px;
            margin-right: 4px;
        }
        """
    )
