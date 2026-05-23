from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap

from anylabeling.views.labeling.utils.qt import new_icon, new_icon_path
from anylabeling.views.labeling.utils.theme import get_theme

ICON_RENDER_SCALE = 2


def themed_icon(name, ext="png", color=None, size=18):
    if not color or ext != "svg":
        return new_icon(name, ext)

    real_size = size * ICON_RENDER_SCALE
    source = QIcon(new_icon_path(name, ext)).pixmap(
        QSize(real_size, real_size)
    )
    if source.isNull():
        return new_icon(name, ext)

    def tinted_pixmap(fill_color):
        pixmap = QPixmap(source.size())
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.drawPixmap(0, 0, source)
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_SourceIn
        )
        painter.fillRect(pixmap.rect(), QColor(fill_color))
        painter.end()
        return pixmap

    icon = QIcon()
    icon.addPixmap(tinted_pixmap(color), QIcon.Mode.Normal)
    icon.addPixmap(
        tinted_pixmap(get_theme()["text_secondary"]), QIcon.Mode.Disabled
    )
    return icon


def apply_button_icon(button, name, ext="png", size=16, color=None):
    icon = themed_icon(name, ext, color, size)
    if icon is None or icon.isNull():
        return
    button.setIcon(icon)
    button.setIconSize(QSize(size, size))


def theme_icon_color(role="text"):
    return get_theme().get(role, get_theme()["text"])
