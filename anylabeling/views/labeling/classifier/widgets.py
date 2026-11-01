from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QTransform,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QPlainTextEdit,
    QRadioButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)

from anylabeling.views.labeling.classifier.style import get_overlay_text_style
from anylabeling.views.labeling.utils.style import (
    get_dock_style,
    get_settings_combo_style,
)
from anylabeling.views.labeling.utils.theme import get_theme


class ObjectComboBox(QComboBox):
    def showPopup(self):
        super().showPopup()
        self.update()

    def hidePopup(self):
        super().hidePopup()
        self.update()

    def paintEvent(self, event):
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        if self.view().isVisible():
            option.currentText = ""
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)


class ObjectItemDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(self.parent().height())
        return size


def configure_object_combo(combo, height):
    combo.setStyleSheet(
        get_settings_combo_style()
        + "QComboBox { min-height: 0; }"
        + "QComboBox QAbstractItemView::item { min-height: 0; padding: 0 8px; }"
    )
    combo.setFixedHeight(height)
    view = QListView(combo)
    view.setUniformItemSizes(True)
    view.setMouseTracking(True)
    view.viewport().setMouseTracking(True)
    view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    combo.setView(view)
    combo.setItemDelegate(ObjectItemDelegate(combo))
    combo.setMaxVisibleItems(12)


class TextResizeHandle(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.setFixedSize(16, 16)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.start_y = 0
        self.start_height = 0

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_y = event.globalPosition().y()
            self.start_height = self.parentWidget().height()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            height = (
                self.start_height + event.globalPosition().y() - self.start_y
            )
            self.parentWidget().setFixedHeight(max(48, round(height)))
            event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(get_theme()["text_secondary"]))
        painter.drawLine(5, 13, 13, 5)
        painter.drawLine(9, 13, 13, 9)


class ResizableTextEdit(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(72)
        self.handle = TextResizeHandle(self)
        theme = get_theme()
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                border: 1px solid {theme["border"]};
                border-radius: 8px;
                padding: 4px;
                background-color: {theme["background_secondary"]};
                color: {theme["text"]};
            }}
            QPlainTextEdit:focus {{ border: 2px solid {theme["primary"]}; }}
        """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.handle.move(self.width() - 17, self.height() - 17)


class ObjectPreviewMixin:
    def set_objects(self, shapes, image_size, selected=-1):
        self.object_shapes = shapes
        self.object_image_size = image_size
        self.selected_object = selected
        self.update()

    def object_transform(self):
        rect = self.image_rect()
        size = self.object_image_size
        transform = QTransform()
        if size.width() and size.height():
            transform.translate(rect.left(), rect.top())
            transform.scale(
                rect.width() / size.width(), rect.height() / size.height()
            )
        return transform

    def object_path(self, shape):
        if (
            shape.shape_type in {"rectangle", "rotation"}
            and len(shape.points) == 2
        ):
            path = QPainterPath()
            path.addRect(QRectF(shape.points[0], shape.points[1]).normalized())
            return path
        path = shape.make_path()
        if shape.shape_type in {
            "rectangle",
            "rotation",
            "polygon",
            "quadrilateral",
        }:
            path.closeSubpath()
        return path

    def object_color(self, shape, attribute="line_color"):
        color = QColor(getattr(shape, attribute))
        canvas = self.source_canvas
        if getattr(canvas, "display_mode", "label") in ("instance", "group"):
            replacement = canvas._display_color(shape)
            replacement.setAlpha(color.alpha())
            return replacement
        return color

    def object_visible(self, shape):
        canvas = self.source_canvas
        return shape.visible and (
            not hasattr(canvas, "is_visible") or canvas.is_visible(shape)
        )

    def paint_objects(self, painter):
        if not getattr(self, "object_shapes", None):
            return
        painter.save()
        painter.setClipRect(self.image_rect(), Qt.ClipOperation.IntersectClip)
        transform = self.object_transform()
        canvas = self.source_canvas
        for index, shape in enumerate(self.object_shapes):
            if not self.object_visible(shape) or (
                self.selected_object >= 0 and index != self.selected_object
            ):
                continue
            path = transform.map(self.object_path(shape))
            color = self.object_color(shape)
            pen = QPen(color, float(shape.line_width))
            if shape.difficult:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            fill = QColor(color)
            maskable = shape.shape_type in {
                "rectangle",
                "rotation",
                "polygon",
                "quadrilateral",
                "circle",
            }
            if getattr(canvas, "show_masks", True) and maskable:
                opacity = getattr(canvas, "shape_opacity", 1.0)
                fill.setAlpha(
                    round(getattr(canvas, "mask_opacity", 80) * opacity)
                )
                painter.setBrush(fill)
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            if shape.shape_type == "point" and shape.points:
                painter.setBrush(color)
                painter.drawEllipse(transform.map(shape.points[0]), 3, 3)
            if index == self.selected_object:
                points = shape.points
                if shape.shape_type == "rectangle" and len(points) == 2:
                    rect = QRectF(points[0], points[1]).normalized()
                    points = [
                        rect.topLeft(),
                        rect.topRight(),
                        rect.bottomRight(),
                        rect.bottomLeft(),
                    ]
                vertex = self.object_color(shape, "vertex_fill_color")
                vertex.setAlpha(255)
                painter.setBrush(vertex)
                painter.setPen(QPen(QColor("black"), 1))
                radius = shape.point_size / 2
                for point in points:
                    position = transform.map(point)
                    if shape.point_type == shape.P_SQUARE:
                        painter.drawRect(
                            QRectF(
                                position.x() - radius,
                                position.y() - radius,
                                radius * 2,
                                radius * 2,
                            )
                        )
                    else:
                        painter.drawEllipse(position, radius, radius)
        painter.restore()

    def paint_object_labels(self, painter):
        if self.selected_object >= 0 or not getattr(
            self, "object_shapes", None
        ):
            return
        painter.save()
        painter.setFont(QFont("Arial", self.label_font_size))
        metrics = QFontMetrics(painter.font())
        transform = self.object_transform()
        viewport = QRectF(self.rect()).adjusted(10.5, 10.5, -10.5, -10.5)
        for index, shape in enumerate(self.object_shapes):
            if (
                not self.object_visible(shape)
                or not shape.points
                or (
                    self.selected_object >= 0 and index != self.selected_object
                )
            ):
                continue
            path = transform.map(self.object_path(shape))
            bounds = path.boundingRect()
            if not bounds.intersects(viewport) and not viewport.contains(
                bounds.center()
            ):
                continue
            label = shape.label or ""
            if not label:
                continue
            width = metrics.tightBoundingRect(label).width() + 8
            height = metrics.height() + 4
            x = max(
                0,
                min(
                    int(bounds.left() - shape.line_width / 2),
                    self.width() - width,
                ),
            )
            y = max(
                0,
                min(
                    int(bounds.top() - height + shape.line_width / 2),
                    self.height() - height,
                ),
            )
            rect = QRect(x, y, width, height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.object_color(shape))
            painter.drawRoundedRect(QRectF(rect), 4.0, 4.0)
            if shape.shape_type in ("rectangle", "rotation"):
                painter.fillRect(
                    QRect(rect.left(), rect.bottom() - 3, rect.width(), 4),
                    self.object_color(shape),
                )
            painter.setPen(QPen(QColor("#000000"), 8, Qt.PenStyle.SolidLine))
            painter.drawText(
                QPoint(
                    rect.left() + 4, rect.bottom() + 1 - 2 - metrics.descent()
                ),
                label,
            )
        painter.restore()

    def mousePressEvent(self, event):
        if self.multitask_enabled and event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
        ):
            self.pan_start = event.position()
            self.pan_origin = event.position()
            self.pan_button = event.button()
            self.pan_dragging = False
            event.accept()
            return
        super().mousePressEvent(event)

    def select_at(self, position):
        if getattr(self, "object_shapes", None):
            transform = self.object_transform()
            for index in reversed(range(len(self.object_shapes))):
                if not self.object_visible(self.object_shapes[index]) or (
                    self.selected_object >= 0 and index != self.selected_object
                ):
                    continue
                path = transform.map(
                    self.object_path(self.object_shapes[index])
                )
                stroker = QPainterPathStroker()
                stroker.setWidth(10)
                if (
                    path.contains(position)
                    or stroker.createStroke(path).contains(position)
                    or (
                        self.object_shapes[index].shape_type == "point"
                        and self.object_shapes[index].points
                        and (
                            transform.map(self.object_shapes[index].points[0])
                            - position
                        ).manhattanLength()
                        <= 8
                    )
                ):
                    self.object_selected.emit(index)
                    return
        self.object_selected.emit(-1)
        self.background_clicked.emit()


class MultiTaskForm(QWidget):
    def __init__(self, parent, combo_height=26):
        super().__init__()
        self.owner = parent
        self.combo_height = combo_height
        self.layout_box = QVBoxLayout(self)
        self.layout_box.setContentsMargins(0, 0, 0, 0)
        self.layout_box.setSpacing(10)
        self.show_object(-1)

    def clear_form(self):
        while self.layout_box.count():
            item = self.layout_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def show_object(self, index):
        self.clear_form()
        owner = self.owner
        shapes = owner.canvas.shapes
        shape = shapes[index] if 0 <= index < len(shapes) else None
        properties = owner.attributes.get(shape.label) if shape else None
        if not owner.attributes or (shape and properties is None):
            message = self.tr(
                "Please load an object attributes configuration file first."
            )
            label = QLabel(message)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            self.layout_box.addStretch()
            self.layout_box.addWidget(label)
        elif shape:
            for name, options in properties.items():
                kind = owner.attribute_widget_types.get(shape.label, {}).get(
                    name, "combobox"
                )
                title = QWidget()
                title_layout = QHBoxLayout(title)
                title_layout.setContentsMargins(0, 0, 0, 0)
                title_layout.setSpacing(8)
                title_label = QLabel(name)
                title_label.setStyleSheet("font-weight: bold;")
                title_layout.addWidget(title_label)
                title_layout.addStretch()
                search = None
                if kind == "combobox":
                    search = QLineEdit()
                    search.setPlaceholderText(self.tr("Search options..."))
                    search.setClearButtonEnabled(True)
                    search.setMaximumWidth(240)
                    search.setStyleSheet(get_dock_style())
                    title_layout.addWidget(search)
                self.layout_box.addWidget(title)
                value = shape.attributes.get(name)
                if kind == "lineedit":
                    control = ResizableTextEdit()
                    control.setPlainText(
                        str(value) if value is not None else ""
                    )
                    control.textChanged.connect(
                        lambda s=shape, n=name, editor=control: self.set_value(
                            s, n, editor.toPlainText()
                        )
                    )
                elif kind == "radiobutton":
                    control = QWidget()
                    control.setSizePolicy(
                        QSizePolicy.Policy.Preferred,
                        QSizePolicy.Policy.Maximum,
                    )
                    row = QVBoxLayout(control)
                    row.setContentsMargins(0, 0, 0, 0)
                    group = QButtonGroup(control)
                    choices = list(options)
                    if value is not None and str(value) not in choices:
                        choices.append(str(value))
                    for option in choices:
                        button = QRadioButton(option)
                        theme = get_theme()
                        button.setStyleSheet(
                            "QRadioButton { spacing: 4px; }"
                            "QRadioButton::indicator { width: 12px; height: 12px;"
                            f" border: 1px solid {theme['text_secondary']};"
                            f" border-radius: 7px; background: {theme['background']}; }}"
                            "QRadioButton::indicator:checked { width: 6px; height: 6px;"
                            f" border: 4px solid {theme['primary']}; border-radius: 7px; }}"
                        )
                        group.addButton(button)
                        button.setChecked(
                            value is not None and str(value) == option
                        )
                        button.toggled.connect(
                            lambda checked, s=shape, n=name, v=option: checked
                            and self.set_value(s, n, v)
                        )
                        row.addWidget(button)
                else:
                    control = ObjectComboBox()
                    configure_object_combo(control, self.combo_height)
                    choices = (
                        sorted(
                            {
                                str(s.group_id)
                                for s in shapes
                                if s.group_id is not None
                            }
                        )
                        if kind == "group_id"
                        else list(options)
                    )
                    control.addItem("")
                    control.addItems(choices)
                    if value is not None:
                        text = str(value)
                        if control.findText(text) < 0:
                            control.addItem(text)
                        control.setCurrentText(text)
                    control.currentTextChanged.connect(
                        lambda text, s=shape, n=name: self.set_value(
                            s, n, text
                        )
                    )
                    if search is not None:
                        search.textChanged.connect(
                            lambda text, combo=control: self.filter_options(
                                combo, text
                            )
                        )
                self.layout_box.addWidget(control)
        self.layout_box.addStretch()

    def filter_options(self, combo, text):
        query = text.casefold()
        for index in range(1, combo.count()):
            combo.view().setRowHidden(
                index, query not in combo.itemText(index).casefold()
            )

    def set_value(self, shape, name, value):
        if shape.attributes.get(name) == value:
            return
        shape.attributes[name] = value
        self.owner.set_dirty()
        self.owner.canvas.update()


class ClassificationImagePreview(ObjectPreviewMixin, QLabel):
    object_selected = pyqtSignal(int)
    background_clicked = pyqtSignal()
    resized = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.multitask_enabled = False
        self.show_object_labels = True
        self.mask_background = False
        self.selected_object = -1
        self.source_canvas = None
        self.label_font_size = 8
        self.fit_mode = "window"
        self.zoom_factor = 1.0
        self.pan_offset = QPointF()
        self.pan_start = None
        self.pan_origin = None
        self.pan_button = None
        self.pan_dragging = False

    def set_fit_mode(self, mode):
        self.fit_mode = mode
        self.zoom_factor = 1.0
        self.pan_offset = QPointF()
        self.update()

    def image_rect(self):
        image_area = QRectF(self.rect()).adjusted(10.5, 10.5, -10.5, -10.5)
        pixmap = self.pixmap()
        if pixmap.isNull():
            return image_area
        image_size = pixmap.deviceIndependentSize()
        if self.multitask_enabled:
            scale = image_area.width() / image_size.width()
            if self.fit_mode == "window":
                scale = min(scale, image_area.height() / image_size.height())
            image_size *= scale * self.zoom_factor
        else:
            image_size.scale(
                image_area.size(), Qt.AspectRatioMode.KeepAspectRatio
            )
        image_rect = QRectF(image_area.topLeft(), image_size)
        image_rect.moveCenter(image_area.center())
        if self.multitask_enabled:
            image_rect.translate(self.pan_offset)
        return image_rect

    def paintEvent(self, event):
        theme = get_theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        frame = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        background = QLinearGradient(frame.topLeft(), frame.bottomRight())
        top_color = QColor(theme["background_secondary"])
        top_color.setAlpha(220)
        bottom_color = QColor(theme["surface"])
        bottom_color.setAlpha(180)
        background.setColorAt(0, top_color)
        background.setColorAt(1, bottom_color)
        painter.setBrush(background)
        painter.setPen(QPen(QColor(theme["border"]), 1))
        painter.drawRoundedRect(frame, 24, 24)

        pixmap = self.pixmap()
        if pixmap.isNull():
            return

        image_rect = self.image_rect()
        clip = QPainterPath()
        if self.multitask_enabled:
            viewport = QRectF(self.rect()).adjusted(10.5, 10.5, -10.5, -10.5)
            clip.addRoundedRect(viewport, 18, 18)
        else:
            clip.addRoundedRect(image_rect, 18, 18)
        painter.save()
        painter.setClipPath(clip)
        shapes = getattr(self, "object_shapes", [])
        if (
            self.multitask_enabled
            and self.mask_background
            and 0 <= self.selected_object < len(shapes)
            and self.object_visible(shapes[self.selected_object])
        ):
            shape = shapes[self.selected_object]
            path = self.object_transform().map(self.object_path(shape))
            if shape.shape_type in ("line", "linestrip"):
                stroker = QPainterPathStroker()
                stroker.setWidth(max(shape.line_width, 1))
                path = stroker.createStroke(path)
            elif shape.shape_type == "point" and shape.points:
                path.addEllipse(
                    self.object_transform().map(shape.points[0]),
                    shape.point_size / 2,
                    shape.point_size / 2,
                )
            painter.fillRect(image_rect, Qt.GlobalColor.black)
            painter.save()
            painter.setClipPath(path, Qt.ClipOperation.IntersectClip)
            painter.drawPixmap(image_rect, pixmap, QRectF(pixmap.rect()))
            painter.restore()
        else:
            painter.drawPixmap(image_rect, pixmap, QRectF(pixmap.rect()))
        self.paint_objects(painter)
        painter.restore()
        if self.multitask_enabled and self.show_object_labels:
            self.paint_object_labels(painter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()

    def focus_selected_object(self):
        shapes = getattr(self, "object_shapes", [])
        if not self.multitask_enabled or not 0 <= self.selected_object < len(
            shapes
        ):
            return
        shape = shapes[self.selected_object]
        if not shape.points or self.pixmap().isNull():
            return
        bounds = (
            self.object_transform().map(self.object_path(shape)).boundingRect()
        )
        if shape.shape_type == "point":
            center = self.object_transform().map(shape.points[0])
            bounds = QRectF(center, center)
        viewport = QRectF(self.rect()).adjusted(10.5, 10.5, -10.5, -10.5)
        scale = min(
            viewport.width() * 0.75 / max(bounds.width(), 20),
            viewport.height() * 0.75 / max(bounds.height(), 20),
        )
        self.zoom_factor = max(1.0, min(20.0, self.zoom_factor * scale))
        self.pan_offset = QPointF()
        center = (
            self.object_transform()
            .map(self.object_path(shape))
            .boundingRect()
            .center()
        )
        if shape.shape_type == "point":
            center = self.object_transform().map(shape.points[0])
        self.pan_offset = viewport.center() - center
        self.update()

    def mouseDoubleClickEvent(self, event):
        if (
            self.multitask_enabled
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.pan_start = None
            self.pan_button = None
            self.select_at(event.position())
            self.focus_selected_object()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if not self.multitask_enabled or self.pixmap().isNull():
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if not delta:
            return
        old_rect = self.image_rect()
        old_zoom = self.zoom_factor
        self.zoom_factor = max(
            0.1, min(20.0, old_zoom * (1.15 ** (delta / 120)))
        )
        ratio = self.zoom_factor / old_zoom
        self.pan_offset += (event.position() - old_rect.center()) * (1 - ratio)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event):
        if self.multitask_enabled and self.pan_start is not None:
            if not self.pan_dragging:
                distance = (
                    event.position() - self.pan_origin
                ).manhattanLength()
                if distance < QApplication.startDragDistance():
                    event.accept()
                    return
                self.pan_dragging = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.pan_offset += event.position() - self.pan_start
            self.pan_start = event.position()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.pan_start is not None and event.button() == self.pan_button:
            if (
                self.pan_button == Qt.MouseButton.LeftButton
                and not self.pan_dragging
            ):
                self.select_at(event.position())
            self.pan_start = None
            self.pan_origin = None
            self.pan_button = None
            self.pan_dragging = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ClassificationOverlay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(get_overlay_text_style())
        self.hide()

    def update_text(self, text: str):
        if text:
            self.setText(text)
            self.adjustSize()
            self.show()
        else:
            self.hide()

    def position_overlay(self, parent_widget):
        if self.isVisible():
            image_rect = parent_widget.image_rect()
            self.move(
                round(image_rect.right() - self.width()),
                round(image_rect.top()),
            )
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        image_rect = (
            self.parentWidget().image_rect().translated(-self.x(), -self.y())
        )
        image_clip = QPainterPath()
        image_clip.addRoundedRect(image_rect, 18, 18)
        background = QPainterPath()
        background.addRect(QRectF(self.rect()))
        painter.fillPath(background.intersected(image_clip), QColor(0, 0, 0))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            self.rect().adjusted(4, 4, -4, -4),
            Qt.AlignmentFlag.AlignCenter,
            self.text(),
        )


class ClassificationCheckBoxGroup(QWidget):
    def __init__(self, labels, is_multiclass=True, parent=None):
        super().__init__(parent)
        self.labels = labels
        self.is_multiclass = is_multiclass
        self.checkboxes = {}
        self.button_group = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        for i, label in enumerate(self.labels):
            checkbox = QCheckBox(f"{label}({i})")
            checkbox.setObjectName(label)
            self.checkboxes[label] = checkbox

            if self.is_multiclass:
                checkbox.toggled.connect(self._handle_multiclass_toggle)

            layout.addWidget(checkbox)

    def _handle_multiclass_toggle(self, checked):
        if not self.is_multiclass:
            return

        sender = self.sender()
        if checked:
            for _, checkbox in self.checkboxes.items():
                if checkbox != sender and checkbox.isChecked():
                    checkbox.blockSignals(True)
                    checkbox.setChecked(False)
                    checkbox.blockSignals(False)

    def get_selected_flags(self):
        flags = {}
        for label, checkbox in self.checkboxes.items():
            flags[label] = checkbox.isChecked()
        return flags

    def set_flags(self, flags):
        for label, checkbox in self.checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(flags.get(label, False))
            checkbox.blockSignals(False)

    def clear_selection(self):
        for checkbox in self.checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)


class PageInputLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.classifier_dialog = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            text = self.text().strip()
            if not text:
                if self.classifier_dialog:
                    self.classifier_dialog.restore_current_page_number()
                return
            if self.classifier_dialog:
                self.classifier_dialog.jump_to_page(int(text))
            return
        super().keyPressEvent(event)
