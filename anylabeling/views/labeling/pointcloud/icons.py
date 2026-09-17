from PyQt6 import QtCore, QtGui, QtSvg


def center_pixmap(pixmap):
    bounds = QtGui.QRegion(pixmap.mask()).boundingRect()
    if bounds.isEmpty():
        return pixmap
    ratio = pixmap.devicePixelRatio()
    source = QtGui.QPixmap(pixmap)
    source.setDevicePixelRatio(1)
    centered = QtGui.QPixmap(pixmap.size())
    centered.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(centered)
    painter.drawPixmap(
        QtCore.QPointF(
            (pixmap.width() - bounds.width()) / 2 - bounds.x(),
            (pixmap.height() - bounds.height()) / 2 - bounds.y(),
        ),
        source,
    )
    painter.end()
    centered.setDevicePixelRatio(ratio)
    return centered


def _polygon(points):
    return QtGui.QPolygonF([QtCore.QPointF(*point) for point in points])


def _draw_view(painter, view, accent):
    faces = {
        "top": [(12, 3), (21, 8), (12, 13), (3, 8)],
        "front": [(3, 8), (12, 13), (12, 22), (3, 17)],
        "side": [(12, 13), (21, 8), (21, 17), (12, 22)],
    }
    fill = QtGui.QColor(accent)
    fill.setAlpha(96)
    painter.setBrush(fill)
    painter.drawPolygon(_polygon(faces[view]))
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawPolygon(
        _polygon([(12, 3), (21, 8), (21, 17), (12, 22), (3, 17), (3, 8)])
    )
    painter.drawPolyline(_polygon([(3, 8), (12, 13), (21, 8)]))
    painter.drawLine(QtCore.QPointF(12, 13), QtCore.QPointF(12, 22))


def _draw_brush(painter, accent):
    painter.save()
    painter.translate(12, 12)
    painter.rotate(35)
    painter.translate(-12, -12)
    painter.drawRoundedRect(QtCore.QRectF(11, 3, 5, 10), 2, 2)
    painter.drawLine(QtCore.QPointF(11, 10), QtCore.QPointF(16, 10))
    bristles = QtGui.QPainterPath(QtCore.QPointF(11, 13))
    bristles.cubicTo(7, 13, 10, 19, 4, 20)
    bristles.cubicTo(10, 23, 17, 19, 16, 13)
    bristles.closeSubpath()
    fill = QtGui.QColor(accent)
    fill.setAlpha(96)
    painter.setBrush(fill)
    painter.drawPath(bristles)
    painter.restore()


def _draw_polygon(painter, accent):
    vertices = [(5, 5), (19, 4), (21, 17), (10, 21), (3, 14)]
    fill = QtGui.QColor(accent)
    fill.setAlpha(48)
    painter.setBrush(fill)
    painter.drawPolygon(_polygon(vertices))
    painter.setBrush(QtGui.QColor(accent))
    for point in vertices:
        painter.drawEllipse(QtCore.QPointF(*point), 1.5, 1.5)


def _draw_merge(painter, accent):
    painter.drawPolyline(_polygon([(4, 5), (9, 5), (15, 12), (20, 12)]))
    painter.drawPolyline(_polygon([(4, 19), (9, 19), (15, 12)]))
    painter.drawPolyline(_polygon([(17, 9), (20, 12), (17, 15)]))


def _draw_fit(painter, accent):
    painter.save()
    pen = painter.pen()
    pen.setColor(QtGui.QColor(accent))
    painter.setPen(pen)
    for points in (
        [(4, 9), (4, 4), (9, 4)],
        [(15, 4), (20, 4), (20, 9)],
        [(4, 15), (4, 20), (9, 20)],
        [(15, 20), (20, 20), (20, 15)],
    ):
        painter.drawPolyline(_polygon(points))
    painter.restore()
    painter.drawEllipse(QtCore.QPointF(12, 12), 2, 2)


def _draw_semantic(painter, accent):
    painter.drawPolygon(
        _polygon([(3, 4), (13, 4), (21, 12), (12, 21), (3, 12)])
    )
    painter.drawEllipse(QtCore.QPointF(7, 8), 1, 1)


def _draw_instance(painter, accent, operation):
    painter.drawRoundedRect(QtCore.QRectF(3, 4, 13, 16), 2, 2)
    if operation == "split":
        painter.drawLine(QtCore.QPointF(9, 4), QtCore.QPointF(9, 20))
        painter.drawPolyline(_polygon([(18, 9), (21, 12), (18, 15)]))
        return
    if operation in ("add", "remove"):
        painter.drawLine(QtCore.QPointF(6, 12), QtCore.QPointF(13, 12))
        if operation == "add":
            painter.drawLine(
                QtCore.QPointF(9.5, 8.5), QtCore.QPointF(9.5, 15.5)
            )
        return
    painter.drawLine(QtCore.QPointF(13, 12), QtCore.QPointF(21, 12))
    if operation != "remove":
        painter.drawLine(QtCore.QPointF(17, 8), QtCore.QPointF(17, 16))
    if operation == "new":
        painter.drawEllipse(QtCore.QPointF(7, 8), 1, 1)


def _draw_through(painter, accent):
    painter.drawPolygon(_polygon([(3, 8), (12, 3), (21, 8), (12, 13)]))
    painter.drawPolyline(_polygon([(3, 13), (12, 18), (21, 13)]))
    painter.drawLine(QtCore.QPointF(12, 8), QtCore.QPointF(12, 22))
    painter.drawPolyline(_polygon([(9, 19), (12, 22), (15, 19)]))


def _draw_render_semantic(painter, accent):
    fill = QtGui.QColor(accent)
    fill.setAlpha(75)
    painter.setBrush(fill)
    painter.drawPolygon(
        _polygon([(3, 5), (13, 5), (21, 13), (13, 21), (3, 11)])
    )
    painter.setBrush(painter.pen().color())
    painter.drawEllipse(QtCore.QPointF(7, 9), 1.2, 1.2)
    painter.drawPolyline(_polygon([(8, 2), (15, 2), (22, 9)]))


def _draw_render_intensity(painter, accent):
    circle = QtCore.QRectF(3, 3, 18, 18)
    painter.setBrush(painter.pen().color())
    painter.drawPie(circle, 90 * 16, 180 * 16)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawEllipse(circle)


def _draw_render_rgb(painter, accent):
    for x, y, color in (
        (12, 7.5, "#E35D6A"),
        (7.5, 15, "#49AE7B"),
        (16.5, 15, "#528EE5"),
    ):
        fill = QtGui.QColor(color)
        fill.setAlpha(210)
        painter.setBrush(fill)
        painter.drawEllipse(QtCore.QPointF(x, y), 5, 5)


def _draw_render_instance(painter, accent):
    for rect, color in (
        (QtCore.QRectF(2, 3, 9, 12), accent),
        (QtCore.QRectF(14, 9, 8, 12), "#E5A84B"),
    ):
        fill = QtGui.QColor(color)
        fill.setAlpha(65)
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 1.5, 1.5)
        painter.setBrush(QtGui.QColor(color))
        painter.drawEllipse(rect.center(), 1.5, 1.5)


def _draw_point_size(painter, accent):
    painter.setBrush(QtGui.QColor(accent))
    for x, radius in ((4, 1.5), (11, 2.5), (20, 3.5)):
        painter.drawEllipse(QtCore.QPointF(x, 12), radius, radius)


def _draw_panel(painter, accent, side):
    painter.drawRoundedRect(QtCore.QRectF(3, 4, 18, 16), 3, 3)
    x = 9 if side == "left" else 15
    painter.drawLine(QtCore.QPointF(x, 4), QtCore.QPointF(x, 20))


def _draw_keyboard(painter, accent):
    painter.drawRoundedRect(QtCore.QRectF(2, 4, 20, 16), 2, 2)
    for y in (8, 12):
        for x in (6, 10, 14, 18):
            painter.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(x + 0.01, y))
    painter.drawLine(QtCore.QPointF(6, 16), QtCore.QPointF(18, 16))


def get_icon(name, color, accent):
    if name == "upload":
        source = QtCore.QFile(":/images/images/download.svg")
        if not source.open(QtCore.QIODevice.OpenModeFlag.ReadOnly):
            return None
        data = bytes(source.readAll())
        source.close()
        data = data.replace(
            b"<path ",
            b'<path transform="translate(0 824) scale(1 -1)" ',
            1,
        )
        data = data.replace(b'fill="#636363"', f'fill="{color}"'.encode())
        pixmap = QtGui.QPixmap(36, 36)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        QtSvg.QSvgRenderer(QtCore.QByteArray(data)).render(painter)
        painter.end()
        return QtGui.QIcon(center_pixmap(pixmap))
    if name == "click":
        source = QtCore.QFile(":/images/images/click.svg")
        if not source.open(QtCore.QIODevice.OpenModeFlag.ReadOnly):
            return None
        data = bytes(source.readAll())
        source.close()
        data = data.replace(b'fill="#13227a"', f'fill="{color}"'.encode(), 1)
        data = data.replace(b'fill="#13227a"', f'fill="{accent}"'.encode(), 1)
        pixmap = QtGui.QPixmap(72, 72)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        QtSvg.QSvgRenderer(QtCore.QByteArray(data)).render(painter)
        painter.end()
        return QtGui.QIcon(center_pixmap(pixmap))
    drawers = {
        "keyboard": _draw_keyboard,
        "brush": _draw_brush,
        "polygon": _draw_polygon,
        "merge": _draw_merge,
        "fit": _draw_fit,
        "semantic": _draw_semantic,
        "through": _draw_through,
        "instance-new": lambda p, a: _draw_instance(p, a, "new"),
        "instance-add": lambda p, a: _draw_instance(p, a, "add"),
        "instance-remove": lambda p, a: _draw_instance(p, a, "remove"),
        "instance-split": lambda p, a: _draw_instance(p, a, "split"),
        "point-size": _draw_point_size,
        "panel-left": lambda p, a: _draw_panel(p, a, "left"),
        "panel-right": lambda p, a: _draw_panel(p, a, "right"),
        "render-semantic": _draw_render_semantic,
        "render-intensity": _draw_render_intensity,
        "render-rgb": _draw_render_rgb,
        "render-instance": _draw_render_instance,
    }
    if name not in ("top", "front", "side") and name not in drawers:
        return None
    pixmap = QtGui.QPixmap(72, 72)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.scale(3, 3)
    painter.setPen(
        QtGui.QPen(
            QtGui.QColor(color),
            1.5,
            QtCore.Qt.PenStyle.SolidLine,
            QtCore.Qt.PenCapStyle.RoundCap,
            QtCore.Qt.PenJoinStyle.RoundJoin,
        )
    )
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    if name in ("top", "front", "side"):
        _draw_view(painter, name, accent)
    else:
        drawers[name](painter, accent)
    painter.end()
    return QtGui.QIcon(center_pixmap(pixmap))


def view_icon(view, color, accent):
    if view not in ("top", "front", "side"):
        raise ValueError("Unknown point-cloud view")
    return get_icon(view, color, accent)
