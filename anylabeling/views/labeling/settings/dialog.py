from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.utils.qt import new_icon_path
from anylabeling.views.labeling.utils.style import (
    get_checkbox_indicator_style,
    get_double_spinbox_style,
    get_settings_combo_style,
    get_spinbox_style,
)
from anylabeling.views.labeling.utils.theme import get_mode, get_theme

from .controller import SettingsController, SettingsValidationError
from .editors import (
    ColorRgbaEditor,
    HexColorPickerEditor,
    ShortcutLineEditor,
    Vector2Editor,
)
from .schema import (
    SettingField,
    SETTINGS_PRIMARY_ORDER,
    fields_for_primary,
)


@dataclass
class EditorBinding:
    field: SettingField
    setter: Callable[[Any], None]
    error_setter: Callable[[bool], None]


class ElidedLabel(QtWidgets.QLabel):
    def __init__(self, text: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._full_text = text
        self.setWordWrap(False)
        self._update_elided_text()

    def set_full_text(self, text: str) -> None:
        self._full_text = text
        self._update_elided_text()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_elided_text()

    def _update_elided_text(self) -> None:
        metrics = QtGui.QFontMetrics(self.font())
        available_width = max(10, self.width() - 2)
        elided = metrics.elidedText(
            self._full_text,
            QtCore.Qt.TextElideMode.ElideRight,
            available_width,
        )
        super().setText(elided)


class SettingsDialog(QtWidgets.QDialog):
    _nav_icon_size = 20

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        controller: SettingsController,
    ):
        super().__init__(parent)
        self._controller = controller
        self._bindings: dict[str, EditorBinding] = {}
        self._nav_item_widgets: dict[
            str, tuple[QtWidgets.QLabel, QtWidgets.QLabel]
        ] = {}
        self._nav_items: list[str] = list(SETTINGS_PRIMARY_ORDER)
        self._palette = self._build_palette()
        self._drag_offset: QtCore.QPoint | None = None
        self._content_height_hint = 0
        self._dirty_primaries: set[str] = set()
        self._active_primary = ""
        self._shortcut_fields_by_group: dict[str, list[SettingField]] = {}
        self._shortcut_group_list: QtWidgets.QListWidget | None = None
        self._shortcut_rows_layout: QtWidgets.QVBoxLayout | None = None
        self._shortcut_rows_parent: QtWidgets.QWidget | None = None
        self._shortcut_rows_scroll: QtWidgets.QScrollArea | None = None
        self._content_area_layout: QtWidgets.QVBoxLayout | None = None
        self._content_bottom_spacer: QtWidgets.QWidget | None = None
        self._shortcut_editor_roots: list[QtWidgets.QWidget] = []
        self._wheel_block_widgets: set[QtCore.QObject] = set()
        self._single_editor_width = 124
        self._vector_component_width = 124
        self._editor_height = 36
        self._section_gap = 8
        self._status_level = "info"
        self._did_show_once = False
        self._combo_animation_effect: Any | None = None
        self._combo_animation_prev_enabled: bool | None = None

        self.setModal(False)
        self.setWindowTitle(self.tr("Settings"))
        self.resize(920, 600)
        self.setMinimumSize(920, 600)
        self.setWindowFlags(
            QtCore.Qt.WindowType.Dialog
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True
        )
        self.setStyleSheet("background: transparent;")

        self._setup_ui()
        self._connect_signals()
        self._disable_combo_animation_if_needed()
        self.nav_list.setCurrentRow(0)
        self._set_status(self.tr("Ready to edit settings"), "info")

    def _build_palette(self) -> dict[str, tuple[int, int, int]]:
        theme = get_theme()
        primary_color = QtGui.QColor(theme["primary"])
        primary_rgb = (
            primary_color.red(),
            primary_color.green(),
            primary_color.blue(),
        )
        if get_mode() == "dark":
            return {
                "left_bg": (44, 44, 46),
                "left_text": (174, 174, 178),
                "left_hover": (58, 58, 60),
                "left_selected": (72, 72, 74),
                "left_active_text": primary_rgb,
                "right_bg": (24, 24, 24),
                "card_bg": (44, 44, 46),
                "title_text": (245, 245, 247),
                "desc_text": (174, 174, 178),
                "line": (72, 72, 74),
                "shortcut_middle_bg": (26, 26, 28),
                "shortcut_group_hover": (34, 34, 36),
                "outer_border": (183, 183, 183),
                "close_hover": (58, 58, 60),
            }
        return {
            "left_bg": (225, 225, 225),
            "left_text": (144, 144, 144),
            "left_hover": (217, 217, 220),
            "left_selected": (212, 212, 216),
            "left_active_text": primary_rgb,
            "right_bg": (239, 238, 239),
            "card_bg": (234, 234, 235),
            "title_text": (0, 0, 0),
            "desc_text": (144, 144, 144),
            "line": (228, 228, 231),
            "shortcut_middle_bg": (255, 255, 255),
            "shortcut_group_hover": (246, 246, 248),
            "outer_border": (183, 183, 183),
            "close_hover": (217, 217, 220),
        }

    def _rgb(self, key: str) -> str:
        r, g, b = self._palette[key]
        return f"rgb({r}, {g}, {b})"

    def _icon_pixmap(self, name: str, color: QtGui.QColor) -> QtGui.QPixmap:
        size = self._nav_icon_size
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(color)
        pen.setWidthF(1.7)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        scale = size / 20.0

        def point(x: float, y: float) -> QtCore.QPointF:
            return QtCore.QPointF(x * scale, y * scale)

        if name == "General":
            knob_radius = 1.6
            slider_rows = (
                (4.8, 15.2, 7.0, 5.8),
                (4.8, 15.2, 12.8, 10.0),
                (4.8, 15.2, 9.4, 14.2),
            )
            for x1, x2, knob_x, y in slider_rows:
                painter.drawLine(
                    point(x1, y),
                    point(knob_x - knob_radius - 0.8, y),
                )
                painter.drawLine(
                    point(knob_x + knob_radius + 0.8, y),
                    point(x2, y),
                )
                painter.drawEllipse(
                    point(knob_x, y),
                    knob_radius * scale,
                    knob_radius * scale,
                )
        elif name == "Shortcuts":
            petal_radius = 2.8
            petals = (
                point(7.2, 7.2),
                point(12.8, 7.2),
                point(7.2, 12.8),
                point(12.8, 12.8),
            )
            for petal in petals:
                painter.drawEllipse(
                    petal,
                    petal_radius * scale,
                    petal_radius * scale,
                )
            painter.drawEllipse(point(10.0, 10.0), 0.9 * scale, 0.9 * scale)
        elif name == "Canvas":
            rect = QtCore.QRectF(
                4.4 * scale, 4.4 * scale, 11.2 * scale, 11.2 * scale
            )
            painter.drawRoundedRect(rect, 2.0 * scale, 2.0 * scale)
            painter.drawLine(point(10.0, 5.8), point(10.0, 14.2))
            painter.drawLine(point(5.8, 10.0), point(14.2, 10.0))
        elif name == "Shape":
            points = (
                point(10.0, 4.9),
                point(14.5, 7.5),
                point(14.5, 12.5),
                point(10.0, 15.1),
                point(5.5, 12.5),
                point(5.5, 7.5),
            )
            painter.drawPolygon(QtGui.QPolygonF(points))
            for point in points:
                painter.drawEllipse(point, 1.05 * scale, 1.05 * scale)
        else:
            cx = size / 2
            cy = size / 2
            painter.drawEllipse(QtCore.QPointF(cx, cy), 6.4, 6.4)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QBrush(color))
            for dx in (-3.0, 0.0, 3.0):
                painter.drawEllipse(
                    QtCore.QPointF(cx + dx, cy),
                    1.2,
                    1.2,
                )
        painter.end()
        return pixmap

    def _brand_logo_pixmap(self) -> QtGui.QPixmap:
        pixmap = QtGui.QPixmap(new_icon_path("icon", "png"))
        if pixmap.isNull():
            return self._icon_pixmap(
                "Brand", QtGui.QColor(*self._palette["left_active_text"])
            )
        return pixmap.scaled(
            self._nav_icon_size,
            self._nav_icon_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

    def _combo_style(self) -> str:
        return get_settings_combo_style()

    def _scrollbar_style(self) -> str:
        up_arrow = new_icon_path("caret-up", "svg")
        down_arrow = new_icon_path("caret-down", "svg")
        return f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: {self._rgb('right_bg')};
                width: 10px;
                margin: 16px 0 16px 0;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background-color: {self._rgb('left_text')};
                min-height: 20px;
                border-radius: 5px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {self._rgb('left_text')};
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                border: none;
                background: {self._rgb('right_bg')};
                height: 16px;
            }}
            QScrollBar::sub-line:vertical {{
                subcontrol-position: top;
                subcontrol-origin: margin;
                image: url({up_arrow});
            }}
            QScrollBar::add-line:vertical {{
                subcontrol-position: bottom;
                subcontrol-origin: margin;
                image: url({down_arrow});
            }}
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
        """

    def _radio_style(self) -> str:
        accent = self._rgb("left_active_text")
        text = self._rgb("title_text")
        border = self._rgb("left_text")
        radio_size = 16
        radio_radius = radio_size // 2
        dot_stop = 0.50  # 16px outer ring with ~8px inner dot.
        return f"""
            QRadioButton {{
                color: {text};
                spacing: 6px;
                background: transparent;
            }}
            QRadioButton::indicator {{
                width: {radio_size}px;
                height: {radio_size}px;
                min-width: {radio_size}px;
                min-height: {radio_size}px;
                max-width: {radio_size}px;
                max-height: {radio_size}px;
                border-radius: {radio_radius}px;
            }}
            QRadioButton::indicator:unchecked {{
                border: 1px solid {border};
                background: transparent;
            }}
            QRadioButton::indicator:hover {{
                border: 1px solid {accent};
            }}
            QRadioButton::indicator:checked {{
                border: 1px solid {accent};
                background: qradialgradient(
                    cx: 0.5, cy: 0.5,
                    fx: 0.5, fy: 0.5,
                    radius: 0.5,
                    stop: 0 {accent},
                    stop: {dot_stop} {accent},
                    stop: {dot_stop + 0.01} transparent,
                    stop: 1 transparent
                );
            }}
            QRadioButton::indicator:unchecked:disabled {{
                border: 1px solid {border};
                background: transparent;
            }}
            QRadioButton::indicator:checked:disabled {{
                border: 1px solid {accent};
                background: qradialgradient(
                    cx: 0.5, cy: 0.5,
                    fx: 0.5, fy: 0.5,
                    radius: 0.5,
                    stop: 0 {accent},
                    stop: {dot_stop} {accent},
                    stop: {dot_stop + 0.01} transparent,
                    stop: 1 transparent
                );
            }}
        """

    def _message_box_style(self) -> str:
        return f"""
            QMessageBox {{
                background: {self._rgb('card_bg')};
                color: {self._rgb('title_text')};
            }}
            QMessageBox QLabel {{
                background: transparent;
                color: {self._rgb('title_text')};
            }}
            QMessageBox QPushButton {{
                min-width: 84px;
                min-height: 30px;
                border: 1px solid {self._rgb('line')};
                border-radius: 6px;
                background: {self._rgb('right_bg')};
                color: {self._rgb('title_text')};
                padding: 0 12px;
            }}
            QMessageBox QPushButton:hover {{
                background: {self._rgb('card_bg')};
            }}
        """

    def _setup_ui(self) -> None:
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        shell = QtWidgets.QFrame(self)
        shell.setObjectName("settingsShell")
        shell.setStyleSheet(f"""
            QFrame#settingsShell {{
                background: transparent;
            }}
            """)
        shell_layout = QtWidgets.QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        left_panel = QtWidgets.QWidget(shell)
        left_panel.setObjectName("settingsLeftPanel")
        left_panel.setFixedWidth(160)
        left_panel.setStyleSheet(f"""
            QWidget#settingsLeftPanel {{
                background: {self._rgb('left_bg')};
                border-top-left-radius: 11px;
                border-bottom-left-radius: 11px;
            }}
            """)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 12, 0, 12)
        left_layout.setSpacing(10)

        brand_widget = QtWidgets.QWidget(left_panel)
        brand_layout = QtWidgets.QHBoxLayout(brand_widget)
        brand_layout.setContentsMargins(16, 0, 8, 0)
        brand_layout.setSpacing(8)

        brand_icon = QtWidgets.QLabel(brand_widget)
        brand_icon.setFixedSize(self._nav_icon_size, self._nav_icon_size)
        brand_icon.setPixmap(self._brand_logo_pixmap())
        brand_title = QtWidgets.QLabel(self.tr("Settings"), brand_widget)
        brand_title.setStyleSheet(
            f"color: {self._rgb('left_text')}; font-weight: 600; font-size: 14px;"
        )
        brand_layout.addWidget(brand_icon)
        brand_layout.addWidget(brand_title, 1)

        self.nav_list = QtWidgets.QListWidget(left_panel)
        self.nav_list.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.nav_list.setSpacing(6)
        self.nav_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.nav_list.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.nav_list.setStyleSheet(f"""
            QListWidget {{
                background: transparent;
                border: none;
                outline: none;
            }}
            QListWidget::item {{
                border-radius: 8px;
                height: 28px;
            }}
            QListWidget::item:hover {{
                background: {self._rgb('left_hover')};
            }}
            QListWidget::item:selected {{
                background: {self._rgb('left_selected')};
            }}
            """)

        for name in self._nav_items:
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(144, 28))
            self.nav_list.addItem(item)

            row_widget = QtWidgets.QWidget(self.nav_list)
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(16, 0, 8, 0)
            row_layout.setSpacing(8)

            icon_label = QtWidgets.QLabel(row_widget)
            icon_label.setFixedSize(self._nav_icon_size, self._nav_icon_size)
            text_label = QtWidgets.QLabel(
                self._display_primary_text(name), row_widget
            )
            text_label.setStyleSheet(f"color: {self._rgb('left_text')};")

            row_layout.addWidget(icon_label)
            row_layout.addWidget(text_label)
            row_layout.addStretch(1)

            row_widget.setStyleSheet("background: transparent;")
            self.nav_list.setItemWidget(item, row_widget)
            self._nav_item_widgets[name] = (icon_label, text_label)

        left_layout.addWidget(brand_widget)
        left_layout.addWidget(self.nav_list)
        left_layout.addStretch(1)

        right_panel = QtWidgets.QWidget(shell)
        right_panel.setObjectName("settingsRightPanel")
        right_panel.setStyleSheet(f"""
            QWidget#settingsRightPanel {{
                background: {self._rgb('right_bg')};
                border-left: 1px solid {self._rgb('line')};
                border-top-right-radius: 11px;
                border-bottom-right-radius: 11px;
            }}
            """)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.header = QtWidgets.QWidget(right_panel)
        self.header.setFixedHeight(48)
        header_layout = QtWidgets.QHBoxLayout(self.header)
        header_layout.setContentsMargins(16, 0, 0, 0)
        header_layout.setSpacing(10)

        self.header_title = QtWidgets.QLabel(self.header)
        self.header_title.setStyleSheet(
            f"color: {self._rgb('title_text')}; font-weight: 700; font-size: 15px;"
        )

        self.close_button = QtWidgets.QPushButton("\u00d7", self.header)
        self.close_button.setFixedSize(48, 48)
        self.close_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.close_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.close_button.setStyleSheet(f"""
            QPushButton {{
                border: none;
                background: transparent;
                color: {self._rgb('title_text')};
                font-size: 16px;
                font-weight: 500;
                border-radius: 0px;
                outline: none;
            }}
            QPushButton:focus {{
                outline: none;
            }}
            QPushButton:hover {{
                background: rgb(239, 68, 68);
                color: white;
                border-radius: 0px;
                border-top-right-radius: 11px;
            }}
            """)

        header_layout.addWidget(self.header_title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.close_button, 0)

        self.header_line = QtWidgets.QFrame(right_panel)
        self.header_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.header_line.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.header_line.setStyleSheet(
            f"background: {self._rgb('line')}; max-height: 1px; min-height: 1px; border: none;"
        )

        content_area = QtWidgets.QWidget(right_panel)
        content_area_layout = QtWidgets.QVBoxLayout(content_area)
        content_area_layout.setContentsMargins(16, 16, 16, 12)
        content_area_layout.setSpacing(8)
        self._content_area_layout = content_area_layout

        self.content_card = QtWidgets.QFrame(content_area)
        self.content_card.setStyleSheet(
            f"background: {self._rgb('card_bg')}; border-radius: 12px;"
        )
        self.content_card_layout = QtWidgets.QVBoxLayout(self.content_card)
        self.content_card_layout.setContentsMargins(0, 0, 0, 0)
        self.content_card_layout.setSpacing(0)

        self.content_scroll = QtWidgets.QScrollArea(self.content_card)
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.content_scroll.setStyleSheet(self._scrollbar_style())

        self.content_body = QtWidgets.QWidget()
        self.content_body_layout = QtWidgets.QVBoxLayout(self.content_body)
        self.content_body_layout.setContentsMargins(0, 0, 0, 0)
        self.content_body_layout.setSpacing(0)
        self.content_scroll.setWidget(self.content_body)
        self.content_card_layout.addWidget(self.content_scroll)

        self.status_label = QtWidgets.QLabel(content_area)
        self.status_label.setStyleSheet(f"color: {self._rgb('desc_text')};")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        self.shortcuts_footer = QtWidgets.QWidget(content_area)
        self.shortcuts_footer_layout = QtWidgets.QHBoxLayout(
            self.shortcuts_footer
        )
        self.shortcuts_footer_layout.setContentsMargins(0, 1, 0, 1)
        self.shortcuts_footer_layout.setSpacing(8)
        self.shortcuts_footer_layout.addStretch(1)
        self.shortcuts_footer.setMinimumHeight(34)

        self.shortcuts_reset_button = QtWidgets.QPushButton(
            self.tr("Reset"), self.shortcuts_footer
        )
        self.shortcuts_reset_button.setFixedHeight(32)
        self.shortcuts_reset_button.setStyleSheet(f"""
            QPushButton {{
                min-width: 84px;
                border: 1px solid {self._rgb('line')};
                border-radius: 8px;
                color: {self._rgb('title_text')};
                background: {self._rgb('right_bg')};
            }}
            QPushButton:hover {{
                background: {self._rgb('card_bg')};
            }}
            """)
        self.shortcuts_save_button = QtWidgets.QPushButton(
            self.tr("Save"), self.shortcuts_footer
        )
        self.shortcuts_save_button.setFixedHeight(32)
        self.shortcuts_save_button.setEnabled(False)
        theme = get_theme()
        self.shortcuts_save_button.setStyleSheet(f"""
            QPushButton {{
                min-width: 84px;
                border: none;
                border-radius: 8px;
                color: white;
                background: {theme["primary"]};
            }}
            QPushButton:hover {{
                background: {theme["primary_hover"]};
            }}
            QPushButton:disabled {{
                background: rgb(180, 180, 180);
                color: rgb(245, 245, 245);
            }}
            """)
        self.shortcuts_footer_layout.addWidget(self.shortcuts_reset_button)
        self.shortcuts_footer_layout.addWidget(self.shortcuts_save_button)
        self.shortcuts_footer.setVisible(False)

        self.shortcuts_split_line = QtWidgets.QFrame(content_area)
        self.shortcuts_split_line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.shortcuts_split_line.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.shortcuts_split_line.setStyleSheet(
            f"background: {self._rgb('line')}; min-height: 1px; max-height: 1px; border: none;"
        )
        self.shortcuts_split_line.setVisible(False)

        self.shortcuts_bottom_panel = QtWidgets.QWidget(content_area)
        self.shortcuts_bottom_panel.setObjectName("shortcutsBottomPanel")
        self.shortcuts_bottom_panel.setFixedHeight(48)
        self.shortcuts_bottom_panel.setStyleSheet(
            "QWidget#shortcutsBottomPanel { border: none; background: transparent; }"
        )
        self.shortcuts_bottom_layout = QtWidgets.QHBoxLayout(
            self.shortcuts_bottom_panel
        )
        self.shortcuts_bottom_layout.setContentsMargins(12, 8, 12, 8)
        self.shortcuts_bottom_layout.setSpacing(12)
        self.status_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.status_label.setMinimumHeight(32)
        self.status_label.setMaximumHeight(40)
        self.status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.shortcuts_bottom_layout.addWidget(self.status_label, 1)
        self.shortcuts_bottom_layout.addWidget(
            self.shortcuts_footer,
            0,
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter,
        )
        self.shortcuts_bottom_panel.setVisible(False)

        self.content_card.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._content_bottom_spacer = QtWidgets.QWidget(content_area)
        self._content_bottom_spacer.setFixedHeight(0)
        self._content_bottom_spacer.setStyleSheet(
            "background: transparent; border: none;"
        )
        content_area_layout.addWidget(self.content_card, 1)
        content_area_layout.addWidget(self._content_bottom_spacer, 0)
        content_area_layout.addWidget(self.shortcuts_split_line, 0)
        content_area_layout.addWidget(self.shortcuts_bottom_panel, 0)

        right_layout.addWidget(self.header)
        right_layout.addWidget(self.header_line)
        right_layout.addWidget(content_area, 1)

        shell_layout.addWidget(left_panel)
        shell_layout.addWidget(right_panel, 1)

        root_layout.addWidget(shell)

    def _connect_signals(self) -> None:
        self.header.installEventFilter(self)
        self.content_scroll.viewport().installEventFilter(self)
        self.nav_list.installEventFilter(self)
        self.nav_list.viewport().installEventFilter(self)
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        self.close_button.clicked.connect(self.close)
        self.shortcuts_reset_button.clicked.connect(self._on_reset_clicked)
        self.shortcuts_save_button.clicked.connect(self._on_save_clicked)
        self._controller.save_succeeded.connect(self._on_save_succeeded)
        self._controller.save_failed.connect(self._on_save_failed)

    def _on_nav_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._nav_items):
            return
        primary = self._nav_items[row]
        self._update_nav_visuals(primary)
        self._render_primary(primary)
        self._reset_nav_scroll()
        QtCore.QTimer.singleShot(0, self._reset_nav_scroll)

    def _update_nav_visuals(self, active_primary: str) -> None:
        for primary, labels in self._nav_item_widgets.items():
            icon_label, text_label = labels
            is_active = primary == active_primary
            color_key = "left_active_text" if is_active else "left_text"
            text_label.setStyleSheet(f"color: {self._rgb(color_key)};")
            icon_label.setPixmap(
                self._icon_pixmap(
                    primary, QtGui.QColor(*self._palette[color_key])
                )
            )

    def _is_primary_dirty(self, primary: str) -> bool:
        return primary in self._dirty_primaries

    def _set_primary_dirty(self, primary: str, dirty: bool) -> None:
        if dirty:
            self._dirty_primaries.add(primary)
            return
        self._dirty_primaries.discard(primary)

    def _set_bottom_controls_visible(self, visible: bool) -> None:
        self.shortcuts_split_line.setVisible(visible)
        self.shortcuts_bottom_panel.setVisible(visible)
        self.status_label.setVisible(visible)
        self.shortcuts_footer.setVisible(visible)

    def _ready_status_text(self, primary: str) -> str:
        if primary == "Shortcuts":
            return self.tr("Ready to edit shortcuts")
        return self.tr("Ready to edit {page} settings").format(
            page=self._display_primary_text(primary)
        )

    def _pending_status_text(self, primary: str) -> str:
        if primary == "Shortcuts":
            return self.tr(
                "Shortcut changes are pending. Click Save to persist."
            )
        return self.tr(
            "{page} changes are pending. Click Save to persist."
        ).format(page=self._display_primary_text(primary))

    def _display_primary_text(self, primary: str) -> str:
        return self.tr(primary)

    def _display_shortcut_group_text(self, group_name: str) -> str:
        return self.tr(group_name)

    def _show_primary_status(self, primary: str) -> None:
        if self._is_primary_dirty(primary):
            self._set_status(self._pending_status_text(primary), "info")
            return
        if self._dirty_primaries:
            self._set_status(
                self.tr(
                    "Pending changes exist in other pages. Click Save to persist."
                ),
                "info",
            )
            return
        self._set_status(self._ready_status_text(primary), "info")

    def _clear_layout(self, layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _render_primary(self, primary: str) -> None:
        self._active_primary = primary
        self._bindings.clear()
        self._shortcut_editor_roots = []
        self.header_title.setText(self._display_primary_text(primary))
        self._clear_layout(self.content_body_layout)
        self._content_height_hint = 0
        self._set_bottom_controls_visible(False)
        is_middle_primary = primary in {"General", "Shape", "Canvas"}
        is_shortcuts = primary == "Shortcuts"
        if is_shortcuts:
            self.content_scroll.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
        else:
            self.content_scroll.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
        if self._content_area_layout is not None:
            if is_shortcuts:
                self._content_area_layout.setContentsMargins(0, 0, 0, 0)
                self._content_area_layout.setSpacing(0)
            else:
                self._content_area_layout.setContentsMargins(16, 16, 16, 0)
                self._content_area_layout.setSpacing(0)
        self.content_card_layout.setContentsMargins(0, 0, 0, 0)
        if is_middle_primary:
            self.content_scroll.setViewportMargins(
                0, self._section_gap, 0, self._section_gap
            )
        else:
            self.content_scroll.setViewportMargins(0, 0, 0, 0)
        if self._content_bottom_spacer is not None:
            self._content_bottom_spacer.setFixedHeight(
                16 if is_middle_primary else 0
            )
        if is_middle_primary:
            self.shortcuts_bottom_layout.setContentsMargins(0, 8, 0, 8)
        elif is_shortcuts:
            self.shortcuts_bottom_layout.setContentsMargins(16, 8, 16, 8)
        else:
            self.shortcuts_bottom_layout.setContentsMargins(12, 8, 12, 8)
        self.content_body_layout.setContentsMargins(0, 0, 0, 0)
        if is_shortcuts:
            self.content_card.setStyleSheet(
                f"background: {self._rgb('card_bg')}; border-radius: 0px;"
            )
        else:
            self.content_card.setStyleSheet(
                f"background: {self._rgb('card_bg')}; border-radius: 12px;"
            )

        fields = fields_for_primary(primary)
        if primary in {"General", "Shape", "Canvas"}:
            strip_prefix = "canvas." if primary == "Canvas" else None
            self._render_form_fields(fields, strip_prefix=strip_prefix)
            self._set_bottom_controls_visible(True)
            self.shortcuts_save_button.setEnabled(bool(self._dirty_primaries))
            self._show_primary_status(primary)
            rows = len(fields)
            self._content_height_hint = rows * 45 + max(0, rows - 1)
            self._update_card_max_height()
            self.content_scroll.verticalScrollBar().setValue(0)
            return
        self.shortcuts_save_button.setEnabled(bool(self._dirty_primaries))
        self._set_bottom_controls_visible(True)
        self._show_primary_status(primary)
        self._render_shortcut_fields(fields)
        self._content_height_hint = max(
            340, self.content_body.sizeHint().height()
        )
        self._update_card_max_height()
        self.content_scroll.verticalScrollBar().setValue(0)

    def _update_card_max_height(self) -> None:
        card_parent = self.content_card.parentWidget()
        if card_parent is None:
            return
        available_height = card_parent.height()
        if available_height <= 0:
            return
        self.content_card.setMaximumHeight(max(120, available_height))

    def _set_error_style(
        self,
        widget: QtWidgets.QWidget,
        enabled: bool,
        default_style: str = "",
    ) -> None:
        if enabled:
            widget.setStyleSheet(
                "border: 1px solid #FF453A; border-radius: 8px;"
            )
            return
        widget.setStyleSheet(default_style)

    def _register_wheel_block(self, widget: QtWidgets.QWidget) -> None:
        candidates = [widget]
        candidates.extend(widget.findChildren(QtWidgets.QAbstractSpinBox))
        for candidate in candidates:
            candidate.installEventFilter(self)
            self._wheel_block_widgets.add(candidate)

    def _prepare_combo_popup(self, combo: QtWidgets.QComboBox) -> None:
        popup_view = QtWidgets.QListView(combo)
        combo.setView(popup_view)
        combo.setMaxVisibleItems(min(12, max(1, combo.count())))
        if isinstance(popup_view, QtWidgets.QListView):
            popup_view.setUniformItemSizes(True)
        popup_view.setMouseTracking(True)
        popup_view.viewport().setMouseTracking(True)
        popup_view.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        popup_view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        popup_view.doItemsLayout()

    def _disable_combo_animation_if_needed(self) -> None:
        if (
            QtCore.QOperatingSystemVersion.currentType()
            != QtCore.QOperatingSystemVersion.OSType.Windows
        ):
            return
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        ui_effect = getattr(QtCore.Qt, "UIEffect", None)
        if ui_effect is None:
            return
        animate_combo_effect = getattr(ui_effect, "UI_AnimateCombo", None)
        if animate_combo_effect is None:
            return
        is_effect_enabled = getattr(app, "isEffectEnabled", None)
        set_effect_enabled = getattr(app, "setEffectEnabled", None)
        if not callable(is_effect_enabled) or not callable(set_effect_enabled):
            return
        try:
            self._combo_animation_prev_enabled = bool(
                is_effect_enabled(animate_combo_effect)
            )
            self._combo_animation_effect = animate_combo_effect
            set_effect_enabled(animate_combo_effect, False)
        except Exception:
            self._combo_animation_effect = None
            self._combo_animation_prev_enabled = None

    def _restore_combo_animation_if_needed(self) -> None:
        if (
            self._combo_animation_effect is None
            or self._combo_animation_prev_enabled is None
        ):
            return
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        set_effect_enabled = getattr(app, "setEffectEnabled", None)
        if not callable(set_effect_enabled):
            return
        try:
            set_effect_enabled(
                self._combo_animation_effect,
                self._combo_animation_prev_enabled,
            )
        except Exception:
            pass
        self._combo_animation_effect = None
        self._combo_animation_prev_enabled = None

    def _add_row_separator(
        self,
        layout: QtWidgets.QVBoxLayout | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        target_layout = layout or self.content_body_layout
        target_parent = parent or self.content_body
        line = QtWidgets.QFrame(target_parent)
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        line.setStyleSheet(
            f"background: {self._rgb('line')}; min-height: 1px; max-height: 1px; border: none;"
        )
        target_layout.addWidget(line)

    def _build_bool_editor(
        self, field: SettingField
    ) -> tuple[
        QtWidgets.QWidget, Callable[[Any], None], Callable[[bool], None]
    ]:
        editor = QtWidgets.QCheckBox(self.content_body)
        editor.setText("")
        editor.setStyleSheet(
            "QCheckBox { background: transparent; spacing: 0; }"
            + get_checkbox_indicator_style()
        )
        editor.setFixedWidth(22)
        editor.toggled.connect(
            lambda checked, f=field: self._on_editor_value_changed(
                f,
                bool(checked),
            )
        )
        return (
            editor,
            lambda value, w=editor: self._set_checkbox_value(w, value),
            lambda _enabled: None,
        )

    def _build_model_hub_editor(
        self, field: SettingField
    ) -> tuple[
        QtWidgets.QWidget, Callable[[Any], None], Callable[[bool], None]
    ]:
        container = QtWidgets.QWidget(self.content_body)
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        group = QtWidgets.QButtonGroup(container)
        radio_buttons: dict[str, QtWidgets.QRadioButton] = {}
        for option in field.options:
            radio = QtWidgets.QRadioButton(str(option), container)
            radio.setStyleSheet(self._radio_style())
            radio.toggled.connect(
                lambda checked, value=option, f=field: self._on_model_hub_toggled(
                    f,
                    checked,
                    value,
                )
            )
            group.addButton(radio)
            layout.addWidget(radio)
            radio_buttons[str(option)] = radio

        return (
            container,
            lambda value, radios=radio_buttons: self._set_radio_value(
                radios, value
            ),
            lambda enabled, w=container: self._set_error_style(w, enabled),
        )

    def _build_logger_editor(
        self, field: SettingField
    ) -> tuple[
        QtWidgets.QWidget, Callable[[Any], None], Callable[[bool], None]
    ]:
        editor = QtWidgets.QComboBox(self.content_body)
        editor.setFixedWidth(self._single_editor_width)
        editor.setStyleSheet(self._combo_style())
        self._register_wheel_block(editor)
        for option in field.options:
            editor.addItem(str(option), option)
        self._prepare_combo_popup(editor)
        editor.currentIndexChanged.connect(
            lambda _index, f=field, w=editor: self._on_editor_value_changed(
                f,
                w.currentData(),
            )
        )
        return (
            editor,
            lambda value, w=editor: self._set_combo_value(w, value),
            lambda enabled, w=editor: self._set_error_style(
                w,
                enabled,
                self._combo_style(),
            ),
        )

    def _build_enum_editor(
        self, field: SettingField
    ) -> tuple[
        QtWidgets.QWidget, Callable[[Any], None], Callable[[bool], None]
    ]:
        if field.key == "model_hub":
            return self._build_model_hub_editor(field)
        if field.key == "logger_level":
            return self._build_logger_editor(field)
        if field.key == "canvas.double_click":
            container = QtWidgets.QWidget(self.content_body)
            layout = QtWidgets.QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(14)
            radio_buttons: dict[str, QtWidgets.QRadioButton] = {}
            for option in field.options:
                text = self.tr("None") if option is None else str(option)
                radio = QtWidgets.QRadioButton(text, container)
                radio.setStyleSheet(self._radio_style())
                radio.toggled.connect(
                    lambda checked, value=option, f=field: self._on_model_hub_toggled(
                        f,
                        checked,
                        value,
                    )
                )
                layout.addWidget(radio)
                radio_buttons[str(option)] = radio
            layout.addStretch(1)
            return (
                container,
                lambda value, radios=radio_buttons: self._set_radio_value(
                    radios, value
                ),
                lambda enabled, w=container: self._set_error_style(w, enabled),
            )

        editor = QtWidgets.QComboBox(self.content_body)
        editor.setFixedWidth(self._single_editor_width)
        editor.setStyleSheet(self._combo_style())
        self._register_wheel_block(editor)
        for option in field.options:
            text = self.tr("None") if option is None else str(option)
            editor.addItem(text, option)
        self._prepare_combo_popup(editor)
        editor.currentIndexChanged.connect(
            lambda _index, f=field, w=editor: self._on_editor_value_changed(
                f,
                w.currentData(),
            )
        )
        return (
            editor,
            lambda value, w=editor: self._set_combo_value(w, value),
            lambda enabled, w=editor: self._set_error_style(
                w,
                enabled,
                self._combo_style(),
            ),
        )

    def _build_int_editor(
        self, field: SettingField
    ) -> tuple[
        QtWidgets.QWidget, Callable[[Any], None], Callable[[bool], None]
    ]:
        editor = QtWidgets.QSpinBox(self.content_body)
        minimum = int(field.minimum) if field.minimum is not None else -1000000
        maximum = int(field.maximum) if field.maximum is not None else 1000000
        editor.setRange(minimum, maximum)
        editor.setFixedHeight(self._editor_height)
        editor.setFixedWidth(self._single_editor_width)
        editor.setStyleSheet(get_spinbox_style())
        self._register_wheel_block(editor)
        editor.valueChanged.connect(
            lambda value, f=field: self._on_editor_value_changed(f, int(value))
        )
        return (
            editor,
            lambda value, w=editor, f=field: self._set_int_spinbox_value(
                w, value, f
            ),
            lambda enabled, w=editor: self._set_error_style(
                w, enabled, get_spinbox_style()
            ),
        )

    def _build_float_editor(
        self, field: SettingField
    ) -> tuple[
        QtWidgets.QWidget, Callable[[Any], None], Callable[[bool], None]
    ]:
        editor = QtWidgets.QDoubleSpinBox(self.content_body)
        minimum = (
            float(field.minimum) if field.minimum is not None else -1000000.0
        )
        maximum = (
            float(field.maximum) if field.maximum is not None else 1000000.0
        )
        editor.setRange(minimum, maximum)
        editor.setDecimals(max(0, field.decimals))
        editor.setSingleStep(10 ** (-max(0, field.decimals)))
        editor.setFixedHeight(self._editor_height)
        editor.setFixedWidth(self._single_editor_width)
        editor.setStyleSheet(get_double_spinbox_style())
        self._register_wheel_block(editor)
        editor.valueChanged.connect(
            lambda value, f=field: self._on_editor_value_changed(
                f, float(value)
            )
        )
        return (
            editor,
            lambda value, w=editor: self._set_float_spinbox_value(w, value),
            lambda enabled, w=editor: self._set_error_style(
                w, enabled, get_double_spinbox_style()
            ),
        )

    def _line_edit_style(self) -> str:
        return (
            "QLineEdit {"
            f"background-color: {self._rgb('right_bg')};"
            f"color: {self._rgb('title_text')};"
            f"border: 1px solid {self._rgb('line')};"
            "border-radius: 6px;"
            "padding: 0 8px;"
            "min-height: 30px;"
            "}"
        )

    def _build_str_editor(
        self, field: SettingField
    ) -> tuple[
        QtWidgets.QWidget, Callable[[Any], None], Callable[[bool], None]
    ]:
        if field.key == "canvas.crosshair.color":
            editor = HexColorPickerEditor(parent=self.content_body)
            editor.setFixedWidth(self._single_editor_width)
            editor.value_changed.connect(
                lambda value, f=field: self._on_editor_value_changed(f, value)
            )
            return (
                editor,
                lambda value, w=editor: w.set_value(value),
                lambda _enabled: None,
            )

        editor = QtWidgets.QLineEdit(self.content_body)
        editor.setFixedWidth(self._single_editor_width)
        editor.setFixedHeight(self._editor_height)
        editor.setStyleSheet(self._line_edit_style())
        editor.editingFinished.connect(
            lambda f=field, w=editor: self._on_editor_value_changed(
                f,
                self._line_edit_payload(f, w.text()),
            )
        )
        return (
            editor,
            lambda value, w=editor: self._set_line_edit_value(w, value),
            lambda enabled, w=editor: self._set_error_style(
                w, enabled, self._line_edit_style()
            ),
        )

    def _build_color_editor(
        self, field: SettingField
    ) -> tuple[
        QtWidgets.QWidget, Callable[[Any], None], Callable[[bool], None]
    ]:
        if field.primary == "Shape":
            output_mode = "rgba255" if field.channels >= 4 else "rgb255"
            fixed_alpha = None if output_mode == "rgba255" else 255
            editor = HexColorPickerEditor(
                output_mode=output_mode,
                fixed_alpha=fixed_alpha,
                parent=self.content_body,
            )
            editor.setFixedWidth(self._single_editor_width)
            editor.value_changed.connect(
                lambda value, f=field: self._on_editor_value_changed(f, value)
            )
            return (
                editor,
                lambda value, w=editor: w.set_value(value),
                lambda _enabled: None,
            )

        if field.key.startswith("canvas.attributes."):
            editor = HexColorPickerEditor(
                output_mode="rgba255",
                fixed_alpha=255,
                parent=self.content_body,
            )
            editor.setFixedWidth(self._single_editor_width)
            editor.value_changed.connect(
                lambda value, f=field: self._on_editor_value_changed(f, value)
            )
            return (
                editor,
                lambda value, w=editor: w.set_value(value),
                lambda _enabled: None,
            )

        editor = ColorRgbaEditor(
            channels=max(3, field.channels), parent=self.content_body
        )
        self._register_wheel_block(editor)
        editor.value_changed.connect(
            lambda value, f=field: self._on_editor_value_changed(f, value)
        )
        return (
            editor,
            lambda value, w=editor: w.set_value(value),
            lambda _enabled: None,
        )

    def _build_vector2_editor(
        self, field: SettingField
    ) -> tuple[
        QtWidgets.QWidget, Callable[[Any], None], Callable[[bool], None]
    ]:
        editor = Vector2Editor(
            minimum=field.minimum,
            maximum=field.maximum,
            decimals=max(0, field.decimals),
            parent=self.content_body,
        )
        for spinbox in editor.findChildren(QtWidgets.QDoubleSpinBox):
            spinbox.setFixedWidth(self._vector_component_width)
            spinbox.setFixedHeight(self._editor_height)
        editor.setFixedWidth(self._vector_component_width * 2 + 4)
        editor.setFixedHeight(self._editor_height)
        self._register_wheel_block(editor)
        editor.value_changed.connect(
            lambda value, f=field: self._on_editor_value_changed(f, value)
        )
        return (
            editor,
            lambda value, w=editor: w.set_value(value),
            lambda _enabled: None,
        )

    def _build_shortcut_editor(
        self, field: SettingField
    ) -> tuple[
        QtWidgets.QWidget, Callable[[Any], None], Callable[[bool], None]
    ]:
        editor = ShortcutLineEditor(
            allow_none=field.allow_none,
            allow_multiple=field.key == "shortcuts.zoom_in",
            parent=self.content_body,
        )
        editor.setFixedWidth(244)
        editor.value_changed.connect(
            lambda value, f=field: self._on_editor_value_changed(f, value)
        )
        return (
            editor,
            lambda value, w=editor: w.set_value(value),
            lambda enabled, w=editor: w.set_error_state(enabled),
        )

    def _build_editor_for_field(
        self, field: SettingField
    ) -> tuple[
        QtWidgets.QWidget, Callable[[Any], None], Callable[[bool], None]
    ]:
        if field.control == "bool":
            return self._build_bool_editor(field)
        if field.control == "enum":
            return self._build_enum_editor(field)
        if field.control == "int":
            return self._build_int_editor(field)
        if field.control == "float":
            return self._build_float_editor(field)
        if field.control == "str":
            return self._build_str_editor(field)
        if field.control == "color":
            return self._build_color_editor(field)
        if field.control == "vector2":
            return self._build_vector2_editor(field)
        return self._build_str_editor(field)

    def _display_field_title(
        self, field: SettingField, strip_prefix: str | None
    ) -> str:
        return field.label

    def _shortcut_usage_hint(self, key: str) -> str | None:
        # Concise usage hints for AI shortcut bindings.
        hints = {
            "shortcuts.auto_run": self.tr(
                "Usually used to enable batch labeling mode."
            ),
            "shortcuts.auto_label": self.tr("Open the auto-labeling panel."),
            "shortcuts.auto_labeling_add_point": self.tr(
                "Add a positive prompt point."
            ),
            "shortcuts.auto_labeling_clear": self.tr(
                "Clear all prompt points added to the current image."
            ),
            "shortcuts.auto_labeling_finish_object": self.tr(
                "Finish the current object annotation."
            ),
            "shortcuts.auto_labeling_remove_point": self.tr(
                "Add a negative prompt point."
            ),
            "shortcuts.auto_labeling_run": self.tr(
                "Run auto-labeling on the current image."
            ),
            "shortcuts.close": self.tr("Close the current file."),
            "shortcuts.delete_file": self.tr("Delete the current label file."),
            "shortcuts.delete_image_file": self.tr(
                "Delete the current image file."
            ),
            "shortcuts.open": self.tr("Open an image or label file."),
            "shortcuts.open_dir": self.tr("Open an image directory."),
            "shortcuts.open_video": self.tr("Open a video file."),
            "shortcuts.quit": self.tr("Quit the application."),
            "shortcuts.save": self.tr("Save labels to file."),
            "shortcuts.save_as": self.tr("Save labels to another file."),
            "shortcuts.save_to": self.tr("Change the output directory."),
            "shortcuts.open_next": self.tr("Open the next image."),
            "shortcuts.open_next_unchecked": self.tr(
                "Open the next unchecked image."
            ),
            "shortcuts.open_prev": self.tr("Open the previous image."),
            "shortcuts.open_prev_unchecked": self.tr(
                "Open the previous unchecked image."
            ),
        }
        return hints.get(key)

    def _render_form_fields(
        self, fields: list[SettingField], strip_prefix: str | None = None
    ) -> None:
        for index, field in enumerate(fields):
            row = QtWidgets.QWidget(self.content_body)
            row.setFixedHeight(45)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(16, 4, 16, 4)
            row_layout.setSpacing(12)

            left = QtWidgets.QWidget(row)
            left_layout = QtWidgets.QVBoxLayout(left)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(0)

            title = QtWidgets.QLabel(
                self._display_field_title(field, strip_prefix), left
            )
            title.setStyleSheet(
                f"color: {self._rgb('title_text')}; font-size: 13px;"
            )
            title.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            left_layout.addWidget(title)

            if field.description:
                desc = QtWidgets.QLabel(self.tr(field.description), left)
                desc.setWordWrap(False)
                desc.setStyleSheet(
                    f"color: {self._rgb('desc_text')}; font-size: 10px;"
                )
                left_layout.addWidget(desc)

            row_layout.addWidget(left, 1)

            editor_widget, setter, error_setter = self._build_editor_for_field(
                field
            )

            row_layout.addWidget(
                editor_widget,
                0,
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter,
            )
            self.content_body_layout.addWidget(row)
            self._bindings[field.key] = EditorBinding(
                field, setter, error_setter
            )
            setter(self._controller.get_value(field.key))

            if index < len(fields) - 1:
                self._add_row_separator()

        self.content_body_layout.addStretch(1)

    def _render_shortcut_fields(self, fields: list[SettingField]) -> None:
        self._shortcut_fields_by_group = {}
        for field in fields:
            self._shortcut_fields_by_group.setdefault(
                field.secondary, []
            ).append(field)

        container = QtWidgets.QWidget(self.content_body)
        container_layout = QtWidgets.QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        middle_panel = QtWidgets.QWidget(container)
        middle_panel.setFixedWidth(208)
        middle_panel.setStyleSheet(f"""
            background: {self._rgb('shortcut_middle_bg')};
            border-radius: 0px;
            """)
        middle_layout = QtWidgets.QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(8, 8, 8, 8)
        middle_layout.setSpacing(8)

        self._shortcut_group_list = QtWidgets.QListWidget(middle_panel)
        self._shortcut_group_list.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._shortcut_group_list.setSpacing(5)
        self._shortcut_group_list.setStyleSheet(f"""
            QListWidget {{
                background: transparent;
                border: none;
                outline: none;
            }}
            QListWidget::item {{
                min-height: 30px;
                border-radius: 6px;
                color: {self._rgb('title_text')};
                padding: 0 8px;
            }}
            QListWidget::item:hover {{
                background: {self._rgb('shortcut_group_hover')};
            }}
            QListWidget::item:selected {{
                background: {self._rgb('left_selected')};
                color: {self._rgb('left_active_text')};
            }}
            """)

        for group_name, group_fields in self._shortcut_fields_by_group.items():
            item = QtWidgets.QListWidgetItem(
                f"{self._display_shortcut_group_text(group_name)} ({len(group_fields)})"
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, group_name)
            self._shortcut_group_list.addItem(item)

        middle_layout.addWidget(self._shortcut_group_list, 1)

        center_line = QtWidgets.QFrame(container)
        center_line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        center_line.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        center_line.setStyleSheet(
            f"background: {self._rgb('line')}; min-width: 1px; max-width: 1px; border: none;"
        )

        right_panel = QtWidgets.QWidget(container)
        right_panel_layout = QtWidgets.QVBoxLayout(right_panel)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.setSpacing(0)

        top_panel = QtWidgets.QWidget(right_panel)
        top_panel.setObjectName("shortcutsTopPanel")
        top_panel.setStyleSheet(f"""
            QWidget#shortcutsTopPanel {{
                border: none;
                border-radius: 0px;
                background: {self._rgb('card_bg')};
            }}
            """)
        top_panel_layout = QtWidgets.QVBoxLayout(top_panel)
        top_panel_layout.setContentsMargins(0, 0, 0, 0)
        top_panel_layout.setSpacing(0)

        self._shortcut_rows_scroll = QtWidgets.QScrollArea(top_panel)
        self._shortcut_rows_scroll.setWidgetResizable(True)
        self._shortcut_rows_scroll.setFrameShape(
            QtWidgets.QFrame.Shape.NoFrame
        )
        self._shortcut_rows_scroll.setStyleSheet(self._scrollbar_style())

        self._shortcut_rows_parent = QtWidgets.QWidget(
            self._shortcut_rows_scroll
        )
        self._shortcut_rows_layout = QtWidgets.QVBoxLayout(
            self._shortcut_rows_parent
        )
        self._shortcut_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._shortcut_rows_layout.setSpacing(0)
        self._shortcut_rows_scroll.setWidget(self._shortcut_rows_parent)
        top_panel_layout.addWidget(self._shortcut_rows_scroll, 1)

        container_layout.addWidget(middle_panel, 0)
        container_layout.addWidget(center_line, 0)
        container_layout.addWidget(right_panel, 1)
        self.content_body_layout.addWidget(container, 1)
        right_panel_layout.addWidget(top_panel, 1)

        self._shortcut_group_list.currentRowChanged.connect(
            self._on_shortcut_group_changed
        )
        if self._shortcut_group_list.count() > 0:
            self._shortcut_group_list.setCurrentRow(0)
        self._refresh_shortcut_rows()

    def _on_shortcut_group_changed(self, _row: int) -> None:
        self._refresh_shortcut_rows()

    def _current_shortcut_group(self) -> str | None:
        if self._shortcut_group_list is None:
            return None
        item = self._shortcut_group_list.currentItem()
        if item is None:
            return None
        return item.data(QtCore.Qt.ItemDataRole.UserRole)

    def _refresh_shortcut_rows(self) -> None:
        if (
            self._shortcut_rows_layout is None
            or self._shortcut_rows_parent is None
        ):
            return
        self._bindings.clear()
        self._shortcut_editor_roots = []
        self._clear_layout(self._shortcut_rows_layout)
        group_name = self._current_shortcut_group()
        visible_fields = (
            self._shortcut_fields_by_group.get(group_name, [])
            if group_name is not None
            else []
        )
        if not visible_fields:
            empty_label = QtWidgets.QLabel(
                self.tr("No shortcuts in the current category."),
                self._shortcut_rows_parent,
            )
            empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            empty_label.setStyleSheet(
                f"color: {self._rgb('desc_text')}; font-size: 12px;"
            )
            self._shortcut_rows_layout.addStretch(1)
            self._shortcut_rows_layout.addWidget(empty_label)
            self._shortcut_rows_layout.addStretch(1)
            return

        for index, field in enumerate(visible_fields):
            hint_text = self._shortcut_usage_hint(field.key)
            row = QtWidgets.QWidget(self._shortcut_rows_parent)
            row.setFixedHeight(56 if hint_text else 45)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(16, 4, 16, 4)
            row_layout.setSpacing(12)

            left = QtWidgets.QWidget(row)
            left_layout = QtWidgets.QVBoxLayout(left)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(0)

            title = ElidedLabel(self.tr(field.label), left)
            title.setStyleSheet(
                f"color: {self._rgb('title_text')}; font-size: 12px;"
            )
            left_layout.addWidget(title)
            if hint_text:
                hint = ElidedLabel(hint_text, left)
                hint.setStyleSheet(
                    f"color: {self._rgb('desc_text')}; font-size: 10px;"
                )
                left_layout.addWidget(hint)
            row_layout.addWidget(left, 1)

            editor_widget, setter, error_setter = self._build_shortcut_editor(
                field
            )
            self._register_shortcut_editor(editor_widget)
            row_layout.addWidget(
                editor_widget,
                0,
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter,
            )

            self._shortcut_rows_layout.addWidget(row)
            self._bindings[field.key] = EditorBinding(
                field, setter, error_setter
            )
            setter(self._controller.get_value(field.key))

            if index < len(visible_fields) - 1:
                self._add_row_separator(
                    layout=self._shortcut_rows_layout,
                    parent=self._shortcut_rows_parent,
                )

        self._shortcut_rows_layout.addStretch(1)
        if self._shortcut_rows_scroll is not None:
            self._shortcut_rows_scroll.verticalScrollBar().setValue(0)

    def _register_shortcut_editor(self, editor: QtWidgets.QWidget) -> None:
        self._shortcut_editor_roots.append(editor)
        editor.installEventFilter(self)
        for child in editor.findChildren(QtWidgets.QWidget):
            child.installEventFilter(self)

    def _is_shortcut_editor_focus(
        self, widget: QtWidgets.QWidget | None
    ) -> bool:
        if widget is None:
            return False
        found = False
        valid_editors: list[QtWidgets.QWidget] = []
        for editor in self._shortcut_editor_roots:
            try:
                valid_editors.append(editor)
                if widget is editor or editor.isAncestorOf(widget):
                    found = True
            except RuntimeError:
                continue
        self._shortcut_editor_roots = valid_editors
        return found

    def _clear_shortcut_error_status_if_needed(self) -> None:
        if (
            self._active_primary != "Shortcuts"
            or self._status_level != "error"
        ):
            return
        if self._is_shortcut_editor_focus(
            QtWidgets.QApplication.focusWidget()
        ):
            return
        if self._is_primary_dirty("Shortcuts"):
            self._set_status(self._pending_status_text("Shortcuts"), "info")
            return
        self._set_status(self._ready_status_text("Shortcuts"), "info")

    def _set_checkbox_value(
        self, checkbox: QtWidgets.QCheckBox, value: Any
    ) -> None:
        checkbox.blockSignals(True)
        checkbox.setChecked(bool(value))
        checkbox.blockSignals(False)

    def _set_radio_value(
        self, radio_buttons: dict[str, QtWidgets.QRadioButton], value: Any
    ) -> None:
        for key, radio in radio_buttons.items():
            radio.blockSignals(True)
            radio.setChecked(key == str(value))
            radio.blockSignals(False)

    def _set_combo_value(self, combo: QtWidgets.QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index < 0:
            index = 0
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _set_int_spinbox_value(
        self,
        spinbox: QtWidgets.QSpinBox,
        value: Any,
        field: SettingField | None = None,
    ) -> None:
        spinbox.blockSignals(True)
        if value is None and field is not None:
            spinbox.setValue(self._default_int_display_value(field))
        else:
            spinbox.setValue(int(value))
        spinbox.blockSignals(False)

    def _default_int_display_value(self, field: SettingField) -> int:
        if field.key == "qt_image_allocation_limit":
            try:
                return int(QtGui.QImageReader.allocationLimit())
            except Exception:
                return 256
        if field.minimum is not None:
            return int(field.minimum)
        return 0

    def _set_float_spinbox_value(
        self, spinbox: QtWidgets.QDoubleSpinBox, value: Any
    ) -> None:
        spinbox.blockSignals(True)
        spinbox.setValue(float(value))
        spinbox.blockSignals(False)

    def _set_line_edit_value(
        self, line_edit: QtWidgets.QLineEdit, value: Any
    ) -> None:
        line_edit.blockSignals(True)
        line_edit.setText("" if value is None else str(value))
        line_edit.blockSignals(False)

    def _line_edit_payload(self, field: SettingField, text: str) -> str | None:
        trimmed = text.strip()
        if trimmed:
            return trimmed
        return None if field.allow_none else ""

    def _clear_errors(self) -> None:
        for binding in self._bindings.values():
            binding.error_setter(False)

    def _on_model_hub_toggled(
        self,
        field: SettingField,
        checked: bool,
        value: Any,
    ) -> None:
        if not checked:
            return
        self._on_editor_value_changed(field, value)

    def _on_editor_value_changed(
        self, field: SettingField, value: Any
    ) -> None:
        self._clear_errors()
        try:
            changed = self._controller.update_field(
                field.key,
                value,
                schedule_save=False,
            )
            if changed:
                self._set_primary_dirty(field.primary, True)
                self.shortcuts_save_button.setEnabled(
                    bool(self._dirty_primaries)
                )
                self._set_status(
                    self._pending_status_text(field.primary), "info"
                )
        except SettingsValidationError as exc:
            binding = self._bindings.get(field.key)
            if binding is not None:
                binding.error_setter(True)
                binding.setter(self._controller.get_value(field.key))
            for conflict_key in exc.conflict_keys:
                conflict_binding = self._bindings.get(conflict_key)
                if conflict_binding is not None:
                    conflict_binding.error_setter(True)
            self._set_status(str(exc), "error")
        except Exception as exc:
            binding = self._bindings.get(field.key)
            if binding is not None:
                binding.error_setter(True)
                binding.setter(self._controller.get_value(field.key))
            self._set_status(str(exc), "error")

    def _on_save_clicked(self) -> None:
        if not self._dirty_primaries:
            self._set_status(self.tr("No settings changes to save"), "info")
            return
        self._controller.save_now()

    def _confirm_reset(self, title: str, text: str) -> bool:
        confirm_dialog = QtWidgets.QMessageBox(self)
        confirm_dialog.setIcon(QtWidgets.QMessageBox.Icon.Question)
        confirm_dialog.setWindowTitle(title)
        confirm_dialog.setText(text)
        confirm_dialog.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
        )
        confirm_dialog.setDefaultButton(
            QtWidgets.QMessageBox.StandardButton.No
        )
        confirm_dialog.setEscapeButton(QtWidgets.QMessageBox.StandardButton.No)
        confirm_dialog.setStyleSheet(self._message_box_style())
        answer = confirm_dialog.exec()
        return answer == int(QtWidgets.QMessageBox.StandardButton.Yes)

    def _on_reset_clicked(self) -> None:
        if self._active_primary == "Shortcuts":
            self._on_shortcuts_reset_clicked()
            return
        if self._active_primary in {"General", "Shape", "Canvas"}:
            self._on_primary_reset_clicked(self._active_primary)
            return
        self._set_status(
            self.tr("Current page has no resettable fields"), "info"
        )

    def _on_primary_reset_clicked(self, primary: str) -> None:
        fields = fields_for_primary(primary)
        if not fields:
            self._set_status(
                self.tr("No settings found in {page}").format(
                    page=self._display_primary_text(primary)
                ),
                "info",
            )
            return
        if not self._confirm_reset(
            self.tr("Reset Settings"),
            self.tr(
                "Reset all settings in '{page}' to defaults?\n\n"
                "Changes will not be saved until you click Save."
            ).format(page=self._display_primary_text(primary)),
        ):
            return
        changed = False
        for field in fields:
            default_value = self._controller.get_default_value(field.key)
            changed = (
                self._controller.update_field(
                    field.key,
                    default_value,
                    schedule_save=False,
                    emit_signal=True,
                )
                or changed
            )
        if changed:
            self._set_primary_dirty(primary, True)
            self.shortcuts_save_button.setEnabled(bool(self._dirty_primaries))
            self._set_status(
                self.tr(
                    "{page} defaults restored. Click Save to persist."
                ).format(page=self._display_primary_text(primary)),
                "info",
            )
            for field in fields:
                binding = self._bindings.get(field.key)
                if binding is not None:
                    binding.error_setter(False)
                    binding.setter(self._controller.get_value(field.key))
            return
        self._set_status(
            self.tr("{page} values are already defaults").format(
                page=self._display_primary_text(primary)
            ),
            "info",
        )

    def _on_shortcuts_reset_clicked(self) -> None:
        group_name = self._current_shortcut_group()
        if not group_name:
            self._set_status(self.tr("No shortcut category selected"), "info")
            return
        group_fields = self._shortcut_fields_by_group.get(group_name, [])
        if not group_fields:
            self._set_status(
                self.tr("No shortcuts found in category: {group}").format(
                    group=self._display_shortcut_group_text(group_name)
                ),
                "info",
            )
            return
        if not self._confirm_reset(
            self.tr("Reset Shortcuts"),
            self.tr(
                "Reset all shortcuts in '{group}' to defaults?\n\n"
                "Changes will not be saved until you click Save."
            ).format(group=self._display_shortcut_group_text(group_name)),
        ):
            return
        needs_reset = any(
            self._controller.get_value(field.key)
            != self._controller.get_default_value(field.key)
            for field in group_fields
        )
        if not needs_reset:
            self._set_status(
                self.tr(
                    "Shortcut values in '{group}' are already defaults"
                ).format(group=self._display_shortcut_group_text(group_name)),
                "info",
            )
            return
        try:
            for field in group_fields:
                empty_value = [] if field.control == "multi_shortcut" else None
                self._controller.update_field(
                    field.key,
                    empty_value,
                    schedule_save=False,
                    emit_signal=True,
                )
            for field in group_fields:
                default_value = self._controller.get_default_value(field.key)
                self._controller.update_field(
                    field.key,
                    default_value,
                    schedule_save=False,
                    emit_signal=True,
                )
        except SettingsValidationError as exc:
            for conflict_key in exc.conflict_keys:
                conflict_binding = self._bindings.get(conflict_key)
                if conflict_binding is not None:
                    conflict_binding.error_setter(True)
            self._set_status(str(exc), "error")
            self._refresh_shortcut_rows()
            return
        except Exception as exc:
            self._set_status(str(exc), "error")
            self._refresh_shortcut_rows()
            return
        self._set_primary_dirty("Shortcuts", True)
        self.shortcuts_save_button.setEnabled(bool(self._dirty_primaries))
        self._set_status(
            self.tr(
                "Shortcut defaults restored for '{group}'. Click Save to persist."
            ).format(group=self._display_shortcut_group_text(group_name)),
            "info",
        )
        self._refresh_shortcut_rows()

    def _on_save_succeeded(self) -> None:
        self._dirty_primaries.clear()
        self.shortcuts_save_button.setEnabled(False)
        self._set_status(self.tr("Settings saved"), "success")
        self._show_restart_required_notice(self._controller.last_saved_keys)

    def _on_save_failed(self, message: str) -> None:
        self._set_status(
            self.tr("Save failed: {message}").format(message=message),
            "error",
        )

    def _show_restart_required_notice(self, changed_keys: list[str]) -> None:
        if "qt_image_allocation_limit" not in changed_keys:
            return
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg_box.setWindowTitle(self.tr("Restart Required"))
        msg_box.setText(
            self.tr(
                "Qt image allocation limit changes will take effect after restarting the application."
            )
        )
        msg_box.setStyleSheet(self._message_box_style())
        msg_box.exec()

    def _set_status(self, text: str, level: str) -> None:
        self._status_level = level
        if level == "error":
            color = "rgb(255, 69, 58)"
        elif level == "success":
            color = "rgb(48, 209, 88)"
        else:
            color = self._rgb("desc_text")
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._controller.close_session()
        self._dirty_primaries.clear()
        self.shortcuts_save_button.setEnabled(False)
        if self._active_primary:
            self._show_primary_status(self._active_primary)
        self._did_show_once = False
        self._restore_combo_animation_if_needed()
        super().closeEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(
            QtGui.QPainter.RenderHint.SmoothPixmapTransform, True
        )
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 12, 12)
        pen = QtGui.QPen(QtGui.QColor(*self._palette["outer_border"]), 1)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self._update_card_max_height()
        super().resizeEvent(event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._sync_layout_after_show)

    def _sync_layout_after_show(self) -> None:
        self._update_card_max_height()
        self._reset_nav_scroll()
        QtCore.QTimer.singleShot(0, self._reset_nav_scroll)
        if self._did_show_once or not self._active_primary:
            return
        self._did_show_once = True
        self._render_primary(self._active_primary)

    def _reset_nav_scroll(self) -> None:
        self.nav_list.verticalScrollBar().setValue(0)

    def eventFilter(
        self, watched: QtCore.QObject, event: QtCore.QEvent
    ) -> bool:
        if (
            event.type() == QtCore.QEvent.Type.Wheel
            and watched in self._wheel_block_widgets
        ):
            return True
        if event.type() == QtCore.QEvent.Type.Wheel and watched in {
            self.nav_list,
            self.nav_list.viewport(),
        }:
            return True
        if (
            event.type() == QtCore.QEvent.Type.FocusOut
            and isinstance(watched, QtWidgets.QWidget)
            and self._is_shortcut_editor_focus(watched)
        ):
            QtCore.QTimer.singleShot(
                0, self._clear_shortcut_error_status_if_needed
            )
        if watched is self.content_scroll.viewport():
            if (
                self._active_primary == "Shortcuts"
                and event.type() == QtCore.QEvent.Type.Wheel
            ):
                return True
            return super().eventFilter(watched, event)
        if watched is self.header:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                mouse_event = event
                if mouse_event.button() == QtCore.Qt.MouseButton.LeftButton:
                    self._drag_offset = (
                        mouse_event.globalPosition().toPoint()
                        - self.frameGeometry().topLeft()
                    )
                    return True
            if (
                event.type() == QtCore.QEvent.Type.MouseMove
                and self._drag_offset
            ):
                mouse_event = event
                if mouse_event.buttons() & QtCore.Qt.MouseButton.LeftButton:
                    self.move(
                        mouse_event.globalPosition().toPoint()
                        - self._drag_offset
                    )
                    return True
            if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                self._drag_offset = None
                return True
        return super().eventFilter(watched, event)
