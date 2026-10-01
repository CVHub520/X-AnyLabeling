import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.utils.style import get_dialog_style
from anylabeling.views.labeling.utils.theme import get_theme
from anylabeling.views.labeling.utils.qt import new_icon

from .cuboid import EDGES
from .icons import center_pixmap, get_icon


def load_calibration(path):
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("Calibration must be a JSON file.")
    data = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "image_size",
        "camera_model",
        "camera_matrix",
        "T_pointcloud_to_camera",
        "distortion_model",
        "distortion_coefficients",
    }
    if not isinstance(data, dict):
        raise ValueError("Calibration must be a JSON object.")
    missing = fields - data.keys()
    if missing:
        raise ValueError(
            f"Missing calibration fields: {', '.join(sorted(missing))}"
        )
    unknown = data.keys() - fields
    if unknown:
        raise ValueError(
            f"Unknown calibration fields: {', '.join(sorted(unknown))}"
        )
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("schema_version must be 1.")
    if data["camera_model"] != "pinhole":
        raise ValueError("camera_model must be pinhole.")
    size = data["image_size"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(value) is not int or value <= 0 for value in size)
    ):
        raise ValueError(
            "image_size must contain positive integer width and height."
        )
    data["image_size"] = tuple(size)
    model = data["distortion_model"]
    if model not in ("none", "opencv5"):
        raise ValueError("distortion_model must be none or opencv5.")
    for name, shape in (
        ("camera_matrix", (3, 3)),
        ("T_pointcloud_to_camera", (4, 4)),
        ("distortion_coefficients", (0,) if model == "none" else (5,)),
    ):
        value = np.asarray(data[name])
        if (
            value.shape != shape
            or value.dtype.kind not in "iuf"
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                f"{name} must contain finite numbers with shape {shape}."
            )
        data[name] = value.astype(np.float64)
    intrinsic = data["camera_matrix"]
    if (
        intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
        or intrinsic[0, 1] != 0
        or intrinsic[1, 0] != 0
        or not np.array_equal(intrinsic[2], [0, 0, 1])
    ):
        raise ValueError(
            "camera_matrix must be [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] "
            "with positive focal lengths."
        )
    transform = data["T_pointcloud_to_camera"]
    rotation = transform[:3, :3]
    if (
        not np.array_equal(transform[3], [0, 0, 0, 1])
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-5)
    ):
        raise ValueError(
            "T_pointcloud_to_camera must be a rigid transform with a proper "
            "rotation and last row [0, 0, 0, 1]."
        )
    return data


def _project_camera_points(camera, calibration):
    intrinsic = calibration["camera_matrix"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pixels = camera[:, :2] / camera[:, 2, None]
        if calibration["distortion_model"] == "opencv5":
            k1, k2, p1, p2, k3 = calibration["distortion_coefficients"]
            x, y = pixels[:, 0].copy(), pixels[:, 1].copy()
            radius2 = x * x + y * y
            radial = 1 + radius2 * (k1 + radius2 * (k2 + radius2 * k3))
            pixels[:, 0] = (
                x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x)
            )
            pixels[:, 1] = (
                y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y
            )
        pixels *= [intrinsic[0, 0], intrinsic[1, 1]]
        pixels += intrinsic[:2, 2]
    return pixels


def project_points(points, calibration, width, height):
    if (width, height) != calibration["image_size"]:
        raise ValueError(
            "Image dimensions do not match calibration image_size."
        )
    transform = calibration["T_pointcloud_to_camera"]
    with np.errstate(over="ignore", invalid="ignore"):
        camera = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
    indices = np.flatnonzero(valid)
    pixels = _project_camera_points(camera[indices], calibration)
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    indices = indices[inside]
    pixels = pixels[inside].astype(np.int32)
    order = np.argsort(-camera[indices, 2], kind="stable")
    return indices[order], pixels[order]


def _clip_image_segments(segments, width, height):
    segments = segments[np.isfinite(segments).all(axis=(1, 2))]
    start = segments[:, 0]
    delta = segments[:, 1] - start
    lower = np.zeros(len(segments))
    upper = np.ones(len(segments))
    valid = np.ones(len(segments), dtype=bool)
    for direction, distance in (
        (-delta[:, 0], start[:, 0]),
        (delta[:, 0], width - 1 - start[:, 0]),
        (-delta[:, 1], start[:, 1]),
        (delta[:, 1], height - 1 - start[:, 1]),
    ):
        parallel = direction == 0
        valid &= ~(parallel & (distance < 0))
        ratio = np.divide(
            distance, direction, out=np.zeros_like(distance), where=~parallel
        )
        lower = np.maximum(lower, np.where(direction < 0, ratio, 0))
        upper = np.minimum(upper, np.where(direction > 0, ratio, 1))
    valid &= lower <= upper
    return (
        start[valid, None]
        + np.stack((lower[valid], upper[valid]), axis=1)[..., None]
        * delta[valid, None]
    )


def project_cuboid(cuboid, calibration, width, height):
    if (width, height) != calibration["image_size"]:
        raise ValueError(
            "Image dimensions do not match calibration image_size."
        )
    transform = calibration["T_pointcloud_to_camera"]
    camera = cuboid.corners() @ transform[:3, :3].T + transform[:3, 3]
    edges = camera[np.asarray(EDGES)]
    near = 1e-6
    visible = edges[:, :, 2] >= near
    edges, visible = edges[visible.any(axis=1)], visible[visible.any(axis=1)]
    for end in (0, 1):
        behind = ~visible[:, end]
        start, other = edges[behind, end], edges[behind, 1 - end]
        fraction = (near - start[:, 2]) / (other[:, 2] - start[:, 2])
        edges[behind, end] = start + fraction[:, None] * (other - start)
    samples = 33 if calibration["distortion_model"] == "opencv5" else 2
    fraction = np.linspace(0, 1, samples)[None, :, None]
    camera = edges[:, :1] * (1 - fraction) + edges[:, 1:] * fraction
    pixels = _project_camera_points(
        camera.reshape(-1, 3), calibration
    ).reshape(-1, samples, 2)
    segments = np.stack((pixels[:, :-1], pixels[:, 1:]), axis=2).reshape(
        -1, 2, 2
    )
    return _clip_image_segments(segments, width, height)


def image_files(directory):
    formats = {
        bytes(value).decode().lower()
        for value in QtGui.QImageReader.supportedImageFormats()
    }
    images = {}
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix[1:].lower() not in formats:
            continue
        if path.stem in images:
            raise ValueError(
                f"Multiple images have the same basename: {path.stem}"
            )
        images[path.stem] = path
    if not images:
        raise ValueError("No supported images in this directory.")
    return images


@dataclass(frozen=True)
class CameraSource:
    directory: Path
    files: dict
    calibration: dict | None = None
    calibration_path: Path | None = None

    @property
    def name(self):
        return camera_name(self.directory)


def camera_name(directory):
    path = Path(directory)
    return path.parent.name if path.name == "data" else path.name or str(path)


class _CameraSourceFields(QtWidgets.QWidget):
    directory_changed = QtCore.pyqtSignal()

    def __init__(self, title, source, parent):
        super().__init__(parent)
        self.empty_title = title
        form = QtWidgets.QGridLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setColumnStretch(0, 1)
        self.title = QtWidgets.QLabel(title)
        self.title.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.title.setObjectName("pointcloudCameraTitle")
        self.title.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        form.addWidget(self.title, 0, 0)
        self.remove_button = QtWidgets.QToolButton()
        self.remove_button.setObjectName("pointcloudIconButton")
        self.remove_button.setProperty("panelHeader", True)
        pixmap = center_pixmap(new_icon("trash", "svg").pixmap(32, 32))
        painter = QtGui.QPainter(pixmap)
        painter.setCompositionMode(
            QtGui.QPainter.CompositionMode.CompositionMode_SourceIn
        )
        painter.fillRect(pixmap.rect(), QtGui.QColor(get_theme()["text"]))
        painter.end()
        self.remove_button.setIcon(QtGui.QIcon(pixmap))
        self.remove_button.setIconSize(QtCore.QSize(16, 16))
        self.remove_button.setFixedSize(24, 24)
        self.remove_button.setToolTip(parent.tr("Remove camera"))
        self.remove_button.setAccessibleName(parent.tr("Remove camera"))
        self.remove_button.clicked.connect(lambda: parent.remove_camera(self))
        form.addWidget(
            self.remove_button, 0, 1, QtCore.Qt.AlignmentFlag.AlignRight
        )
        self.directory_input = QtWidgets.QLineEdit(
            str(source.directory) if source else ""
        )
        self.calibration_input = QtWidgets.QLineEdit(
            str(source.calibration_path or "") if source else ""
        )
        for row, (field, title, callback) in enumerate(
            (
                (
                    self.directory_input,
                    parent.tr("Image directory"),
                    self._browse_directory,
                ),
                (
                    self.calibration_input,
                    parent.tr("Calibration JSON (optional)"),
                    self._browse_calibration,
                ),
            ),
            1,
        ):
            field.setPlaceholderText(title)
            field.setAccessibleName(title)
            field.setMinimumHeight(32)
            button = QtWidgets.QPushButton(parent.tr("Browse…"))
            button.setAutoDefault(False)
            button.setMinimumHeight(32)
            button.clicked.connect(callback)
            form.addWidget(field, row, 0)
            form.addWidget(button, row, 1)
        self.directory_input.textChanged.connect(self._directory_changed)
        self._directory_changed()

    def _directory_changed(self):
        directory = self.directory_input.text().strip()
        self.title.setText(
            camera_name(directory) + ":" if directory else self.empty_title
        )
        self.title.setToolTip(directory)
        self.directory_changed.emit()

    def _browse_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.directory_input.placeholderText(),
            self.directory_input.text(),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if directory:
            self.directory_input.setText(directory)

    def _browse_calibration(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.calibration_input.placeholderText(),
            self.calibration_input.text(),
            QtCore.QCoreApplication.translate(
                "CameraConfigurationDialog", "JSON files (*.json)"
            ),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self.calibration_input.setText(path)


class CameraConfigurationDialog(QtWidgets.QDialog):
    def __init__(self, panel, parent=None, sources=None, embedded=False):
        super().__init__(parent)
        if embedded:
            self.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.setWindowTitle(self.tr("Camera image"))
        self.setStyleSheet(get_dialog_style() + """
            QLabel#pointcloudCameraTitle { font-weight: 600; }
            QScrollArea#pointcloudCameraSources { border: none; background: transparent; }
        """)
        self.setMinimumWidth(540)
        self.configuration = []
        self.entries = []
        self._previews = []
        self._preview_cache = OrderedDict()
        self._preview_index = 0
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self._refresh_previews)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(12)
        layout.setColumnStretch(0, 1)
        if not embedded:
            layout.setRowStretch(3, 1)
        self.sources_scroll = QtWidgets.QScrollArea()
        self.sources_scroll.setObjectName("pointcloudCameraSources")
        self.sources_scroll.setWidgetResizable(True)
        self.sources_scroll.setMinimumHeight(96)
        self.sources_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.sources_widget = QtWidgets.QWidget()
        self.sources_layout = QtWidgets.QVBoxLayout(self.sources_widget)
        self.sources_layout.setContentsMargins(0, 0, 8, 0)
        self.sources_layout.setSpacing(16)
        self.sources_layout.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetMinimumSize
        )
        self.sources_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.sources_scroll.setWidget(self.sources_widget)
        self.sources_widget.setAutoFillBackground(False)
        layout.addWidget(self.sources_scroll, 0, 0, 1, 2)
        self.sources_scroll.ensurePolished()
        self._scrollbar_gutter = (
            self.sources_scroll.verticalScrollBar().sizeHint().width() + 4
        )
        layout.setColumnMinimumWidth(1, self._scrollbar_gutter)
        layout.setContentsMargins(
            20, 20, max(0, 20 - self._scrollbar_gutter), 16
        )
        self.sources_scroll.viewport().installEventFilter(self)
        self.error_label = QtWidgets.QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.error_label.hide()
        layout.addWidget(self.error_label, 1, 0)
        self.add_button = QtWidgets.QPushButton(self.tr("Add camera"), self)
        self.add_button.setAutoDefault(False)
        self.add_button.clicked.connect(lambda: self.add_camera())
        self.cancel_button = QtWidgets.QPushButton(self.tr("Cancel"), self)
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button = QtWidgets.QPushButton(self.tr("OK"), self)
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)
        if embedded:
            self.cancel_button.hide()
            self.ok_button.hide()
        else:
            buttons = QtWidgets.QHBoxLayout()
            buttons.setSpacing(10)
            buttons.addWidget(self.add_button)
            buttons.addStretch()
            buttons.addWidget(self.cancel_button)
            buttons.addWidget(self.ok_button)
            layout.addLayout(buttons, 2, 0)
        self.preview = CameraView()
        self.preview.allow_zoom = False
        self.preview.setMinimumHeight(180)
        if embedded:
            self.preview.setFixedHeight(180)
        self.preview.setStyleSheet(
            "QGraphicsView { border: none; background: #101114; }"
        )
        self.preview.camera_step.connect(self._step_preview)
        self.preview_caption = QtWidgets.QLabel()
        self.preview_caption.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.preview_caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview, 3, 0)
        layout.addWidget(self.preview_caption, 4, 0)
        for source in (panel.sources if sources is None else sources) or [
            None
        ]:
            self.add_camera(source)
        self.directory_input = self.entries[0].directory_input
        self.calibration_input = self.entries[0].calibration_input
        self._refresh_previews()

    def eventFilter(self, watched, event):
        if (
            watched is self.sources_scroll.viewport()
            and event.type() == QtCore.QEvent.Type.Resize
        ):
            occupied = self.sources_scroll.width() - watched.width()
            self.sources_layout.setContentsMargins(
                0, 0, max(0, self._scrollbar_gutter - occupied), 0
            )
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        if self.isWindow():
            super().keyPressEvent(event)
        else:
            event.ignore()

    def add_camera(self, source=None):
        entry = _CameraSourceFields(
            self.tr("Camera {number}").format(number=len(self.entries) + 1),
            source,
            self,
        )
        self.entries.append(entry)
        self.directory_input = self.entries[0].directory_input
        self.calibration_input = self.entries[0].calibration_input
        self.sources_layout.addWidget(entry)
        entry.setFixedHeight(entry.sizeHint().height())
        entry.show()
        entry.directory_changed.connect(self._preview_timer.start)
        entry.directory_input.textChanged.connect(self.error_label.hide)
        entry.calibration_input.textChanged.connect(self.error_label.hide)
        self._update_sources_height()
        if self.isWindow():
            self.adjustSize()
        QtCore.QTimer.singleShot(
            0,
            lambda: (
                self.sources_scroll.ensureWidgetVisible(entry)
                if entry in self.entries
                else None
            ),
        )

    def _update_sources_height(self):
        visible = self.entries[: 3 if self.isWindow() else 1]
        if visible:
            self.sources_scroll.setFixedHeight(
                sum(member.sizeHint().height() for member in visible)
                + (len(visible) - 1) * self.sources_layout.spacing()
            )

    def remove_camera(self, entry):
        self.entries.remove(entry)
        self.sources_layout.removeWidget(entry)
        entry.deleteLater()
        for number, member in enumerate(self.entries, 1):
            member.empty_title = self.tr("Camera {number}").format(
                number=number
            )
            member._directory_changed()
        self.directory_input = (
            self.entries[0].directory_input if self.entries else None
        )
        self.calibration_input = (
            self.entries[0].calibration_input if self.entries else None
        )
        self._update_sources_height()
        self.error_label.hide()
        self._refresh_previews()

    def _refresh_previews(self):
        current = (
            self._previews[self._preview_index][0] if self._previews else None
        )
        previews = []
        for entry in self.entries:
            directory = entry.directory_input.text().strip()
            if not directory:
                continue
            preview = self._preview_cache.pop(directory, None)
            if preview is None:
                try:
                    files = image_files(Path(directory).expanduser())
                    for path in files.values():
                        reader = QtGui.QImageReader(str(path))
                        size = reader.size()
                        if size.isValid():
                            reader.setScaledSize(
                                size.scaled(
                                    960,
                                    540,
                                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                )
                            )
                        image = reader.read()
                        if not image.isNull():
                            preview = (directory, image)
                            break
                except (OSError, ValueError):
                    pass
            if preview is not None:
                self._preview_cache[directory] = preview
                previews.append(preview)
        while len(self._preview_cache) > 8:
            self._preview_cache.popitem(last=False)
        self._previews = previews
        self._preview_index = next(
            (
                index
                for index, item in enumerate(previews)
                if item[0] == current
            ),
            0,
        )
        self.preview.set_camera_count(len(previews))
        self._show_preview()

    def _show_preview(self):
        if not self._previews:
            self.preview.image_item.setPixmap(QtGui.QPixmap())
            self.preview.setSceneRect(QtCore.QRectF())
            self.preview_caption.clear()
            return
        directory, image = self._previews[self._preview_index]
        self.preview.image_item.setPixmap(QtGui.QPixmap.fromImage(image))
        self.preview.setSceneRect(QtCore.QRectF(image.rect()))
        self.preview.fit_image()
        self.preview_caption.setText(camera_name(directory))

    def _step_preview(self, delta):
        if len(self._previews) < 2:
            return
        self._preview_index = (self._preview_index + delta) % len(
            self._previews
        )
        self._show_preview()

    def read_configuration(self):
        configuration = []
        for entry in self.entries:
            directory = entry.directory_input.text().strip()
            calibration = entry.calibration_input.text().strip()
            if not directory:
                continue
            try:
                path = Path(directory).expanduser()
                calibration_path = (
                    Path(calibration).expanduser() if calibration else None
                )
                files = image_files(path)
                parameters = (
                    load_calibration(calibration_path)
                    if calibration_path
                    else None
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.error_label.setText(f"{entry.title.text()} {error}")
                self.error_label.show()
                self.sources_scroll.ensureWidgetVisible(entry)
                return
            configuration.append(
                CameraSource(path, files, parameters, calibration_path)
            )
        return configuration

    def accept(self):
        configuration = self.read_configuration()
        if configuration is not None:
            self.configuration = configuration
            super().accept()


class _CameraCuboidsItem(QtWidgets.QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.paths = []
        self._rect = QtCore.QRectF()
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)

    def boundingRect(self):
        return self._rect

    def set_cuboids(
        self, boxes, selected_id, colors, calibration, width, height, preview
    ):
        self.prepareGeometryChange()
        self._rect = QtCore.QRectF(0, 0, width, height)
        self.paths = []
        ordered = sorted(boxes, key=lambda box: box.id == selected_id)
        if preview is not None:
            ordered.append(preview)
        for box in ordered:
            lines = project_cuboid(box, calibration, width, height)
            if not len(lines):
                continue
            path = QtGui.QPainterPath()
            for start, end in lines:
                path.moveTo(float(start[0]), float(start[1]))
                path.lineTo(float(end[0]), float(end[1]))
            color = (
                "#ffd54f"
                if box is preview
                else colors.get(box.class_id, "#64b5f6")
            )
            selected = box is preview or box.id == selected_id
            pen = QtGui.QPen(QtGui.QColor(color), 2.5 if selected else 1.3)
            pen.setCosmetic(True)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            if box.occluded:
                pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            self.paths.append((box.id, path, pen))
        self.update()

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setClipRect(self._rect)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        for _, path, pen in self.paths:
            painter.setPen(pen)
            painter.drawPath(path)


class CameraView(QtWidgets.QGraphicsView):
    camera_step = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
        self.viewport().setAutoFillBackground(False)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.image_item = self.scene().addPixmap(QtGui.QPixmap())
        self.overlay_item = self.scene().addPixmap(QtGui.QPixmap())
        self.cuboid_item = _CameraCuboidsItem()
        self.cuboid_item.setZValue(1)
        self.scene().addItem(self.cuboid_item)
        self.fitted = True
        self.allow_zoom = True
        self.camera_count = 0
        self._hovered = False
        self._wheel_distance = 0
        self.message = QtWidgets.QLabel(self.viewport())
        self.message.setWordWrap(True)
        self.message.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.message.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.message.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.message.setStyleSheet(
            "background: transparent; color: #cbd2df; border: none;"
        )
        self.message.hide()
        self.navigation = []
        for step, title, icon in (
            (-1, self.tr("Previous camera"), "camera-left"),
            (1, self.tr("Next camera"), "camera-right"),
        ):
            button = QtWidgets.QToolButton(self.viewport())
            button.setObjectName("pointcloudCameraNavigation")
            button.setIcon(get_icon(icon, "#e1e5ec", "#e1e5ec"))
            button.setIconSize(QtCore.QSize(20, 20))
            button.setFixedSize(32, 32)
            button.setToolTip(title)
            button.setAccessibleName(title)
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet("""
                QToolButton#pointcloudCameraNavigation {
                    background: rgba(32, 35, 42, 90); border: none;
                    border-radius: 16px; padding: 0;
                }
                QToolButton#pointcloudCameraNavigation:hover {
                    background: rgba(90, 97, 110, 160);
                }
            """)
            button.clicked.connect(
                lambda checked=False, step=step: self.camera_step.emit(step)
            )
            button.hide()
            self.navigation.append(button)

    def set_camera_count(self, count):
        self.camera_count = count
        self._wheel_distance = 0
        self.setToolTip(
            self.tr("Wheel: zoom · Arrows: switch camera")
            if count > 1 and self.allow_zoom
            else ""
        )
        self._update_navigation()

    def _update_navigation(self):
        self.message.setGeometry(
            self.viewport().rect().adjusted(42, 4, -42, -4)
        )
        for index, button in enumerate(self.navigation):
            button.move(
                (
                    8
                    if index == 0
                    else self.viewport().width() - button.width() - 8
                ),
                max(0, (self.viewport().height() - button.height()) // 2),
            )
            button.setVisible(self._hovered and self.camera_count > 1)
            button.raise_()

    def enterEvent(self, event):
        self._hovered = True
        self._update_navigation()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._update_navigation()
        super().leaveEvent(event)

    def fit_image(self):
        self.fitted = True
        if not self.sceneRect().isEmpty():
            self.fitInView(
                self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fitted:
            self.fit_image()
        self._update_navigation()

    def wheelEvent(self, event):
        if self.camera_count > 1 and not self.allow_zoom:
            delta = event.angleDelta().y()
            self._wheel_distance += delta or event.pixelDelta().y() * 3
            if abs(self._wheel_distance) >= 120:
                self.camera_step.emit(-1 if self._wheel_distance > 0 else 1)
                self._wheel_distance = 0
            event.accept()
            return
        if not self.allow_zoom or not event.angleDelta().y():
            event.accept()
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        scale = self.transform().m11() * factor
        if 0.02 <= scale <= 32:
            self.fitted = False
            self.scale(factor, factor)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.fit_image()
        event.accept()


class _ProjectionSettingsMenu(QtWidgets.QMenu):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(
            get_dialog_style()
            + "QMenu { background: transparent; border: none; padding: 4px; }"
        )

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        theme = get_theme()
        painter.setPen(QtGui.QPen(QtGui.QColor(theme["border"]), 1))
        painter.setBrush(QtGui.QColor(theme["surface"]))
        painter.drawRoundedRect(
            QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5),
            12,
            12,
        )


class _CameraResizeHandle(QtWidgets.QWidget):
    def __init__(self, panel):
        super().__init__(panel)
        self._start = None
        self.setFixedSize(16, 16)
        self.setCursor(QtCore.Qt.CursorShape.SizeBDiagCursor)
        self.setToolTip(panel.tr("Drag to resize camera image"))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor(203, 210, 223, 150), 1.2))
        for offset in (0, 4):
            painter.drawLine(3, 6 + offset, 10 - offset, 13)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._start = (event.globalPosition(), self.parentWidget().size())
            event.accept()

    def mouseMoveEvent(self, event):
        if self._start is None:
            return
        if not event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._start = None
            return
        origin, size = self._start
        delta = event.globalPosition() - origin
        self.parentWidget().resize_image_region(
            QtCore.QSize(
                size.width() - round(delta.x()),
                size.height() + round(delta.y()),
            )
        )
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._start = None
            event.accept()


class CameraPanel(QtWidgets.QFrame):
    def __init__(self, workspace):
        super().__init__(workspace.viewport)
        self.setObjectName("pointcloudCameraPanel")
        self.setStyleSheet("""
            QFrame#pointcloudCameraPanel, QFrame#pointcloudCameraFooter,
            QGraphicsView, QGraphicsView > QWidget {
                background: transparent; border: none; padding: 0;
            }
            QLabel {
                background: transparent; color: #cbd2df;
                border: none; padding: 0; font-size: 10px;
            }
            QToolButton#pointcloudCameraTool {
                background: transparent; border: none; padding: 0;
                min-width: 0; min-height: 0; border-radius: 3px;
            }
            QToolButton#pointcloudCameraTool:hover,
            QToolButton#pointcloudCameraTool:checked {
                background: rgba(203, 210, 223, 30);
            }
        """)
        self.directory = None
        self.sources = []
        self.active_index = 0
        self._user_size = None
        self.files = {}
        self.calibration = None
        self.calibration_path = None
        self.workspace = workspace
        self._key = None
        self._images = OrderedDict()
        self._indices = np.empty(0, dtype=np.int64)
        self._pixels = np.empty((0, 2), dtype=np.int32)
        self._depth_colors = np.empty((0, 3), dtype=np.uint8)
        self._image = QtGui.QImage()
        self._cuboid_signature = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QtWidgets.QFrame()
        header.setObjectName("pointcloudCameraFooter")
        header.setFixedHeight(24)
        row = QtWidgets.QHBoxLayout(header)
        row.setContentsMargins(18, 2, 2, 2)
        row.setSpacing(4)
        self.filename = QtWidgets.QLabel()
        self.filename.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.filename.setMinimumWidth(0)
        self.filename.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        row.addWidget(self.filename, 1)
        self.overlay_action = workspace._action(
            self.tr("Projected points"), self.refresh, self, icon="point-size"
        )
        self.overlay_action.setCheckable(True)
        self.overlay_action.setChecked(True)
        self.overlay_action.setEnabled(False)
        self.overlay_action.setToolTip(
            self.tr("Load calibration to overlay points.")
        )
        self.view = CameraView()
        self.view.camera_step.connect(self.step_camera)
        self.cuboid_action = workspace._action(
            self.tr("Projected 3D boxes"),
            self.refresh_cuboids,
            self,
            icon="front",
        )
        self.cuboid_action.setCheckable(True)
        self.cuboid_action.setChecked(True)
        self.cuboid_action.setEnabled(False)
        self._build_overlay_settings()
        self.overlay_settings_action = workspace._action(
            self.tr("Projection settings"),
            self._show_overlay_settings,
            self,
            icon="settings",
        )
        for action, icon in (
            (self.overlay_action, "point-size"),
            (self.cuboid_action, "front"),
            (self.overlay_settings_action, "settings"),
            (
                workspace._action(
                    self.tr("Fit image"), self.view.fit_image, self, icon="fit"
                ),
                "fit",
            ),
        ):
            button = workspace._tool_button(action, row)
            button.setObjectName("pointcloudCameraTool")
            button.setIconSize(QtCore.QSize(13, 13))
            button.setFixedSize(20, 20)
            action.setIcon(get_icon(icon, "#cbd2df", "#cbd2df"))
            if action is self.overlay_settings_action:
                self.overlay_settings_button = button
        self.message = self.view.message
        layout.addWidget(self.view, 1)
        layout.addWidget(header)
        self.resize_handle = _CameraResizeHandle(self)
        workspace.viewport.installEventFilter(self)
        workspace.viewport.cuboids_changed.connect(self.refresh_cuboids)

    def _build_overlay_settings(self):
        self.overlay_settings_menu = _ProjectionSettingsMenu(self.workspace)
        theme = get_theme()
        content = QtWidgets.QWidget()
        content.setStyleSheet(
            f"QRadioButton {{ color: {theme['text']}; "
            "background: transparent; spacing: 6px; }"
            "QRadioButton::indicator { width: 14px; height: 14px; "
            f"border: 1px solid {theme['text_secondary']}; "
            f"border-radius: 8px; background: {theme['surface']}; }}"
            "QRadioButton::indicator:checked { "
            f"border-color: {theme['primary']}; "
            "background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, "
            f"fx:0.5, fy:0.5, stop:0 {theme['primary']}, "
            f"stop:0.5 {theme['primary']}, stop:0.6 {theme['surface']}, "
            f"stop:1 {theme['surface']}); }}"
            "QRadioButton::indicator:hover { "
            f"border-color: {theme['primary']}; }}"
            "QToolButton { background: transparent; border: none; "
            "padding: 0; border-radius: 5px; }"
            f"QToolButton:hover {{ background: {theme['surface_hover']}; }}"
            f"QToolButton:pressed {{ background: {theme['selection']}; }}"
        )
        form = QtWidgets.QGridLayout(content)
        form.setContentsMargins(14, 12, 14, 12)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        self.overlay_color = QtWidgets.QButtonGroup(content)
        self.overlay_color_buttons = {}
        choices = QtWidgets.QHBoxLayout()
        choices.setSpacing(16)
        for title, mode in (
            (self.tr("Depth"), "depth"),
            (self.tr("Point cloud colors"), "cloud"),
        ):
            button = QtWidgets.QRadioButton(title)
            button.setProperty("color_mode", mode)
            self.overlay_color.addButton(button)
            self.overlay_color_buttons[mode] = button
            choices.addWidget(button)
        choices.addStretch()
        mode = self.workspace.settings.value("camera_overlay/color", "depth")
        self.overlay_color_buttons.get(
            mode, self.overlay_color_buttons["depth"]
        ).setChecked(True)
        label = QtWidgets.QLabel(self.tr("Color"))
        label.setBuddy(self.overlay_color_buttons["depth"])
        form.addWidget(label, 0, 0)
        form.addLayout(choices, 0, 1, 1, 3)
        self.overlay_size = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.overlay_opacity = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.overlay_reset_buttons = {}
        for row, (
            slider,
            title,
            key,
            maximum,
            default,
            suffix,
            reset_title,
        ) in enumerate(
            (
                (
                    self.overlay_size,
                    self.tr("Point size"),
                    "size",
                    7,
                    1,
                    " px",
                    self.tr("Reset point size (1 px)"),
                ),
                (
                    self.overlay_opacity,
                    self.tr("Opacity"),
                    "opacity",
                    100,
                    65,
                    "%",
                    self.tr("Reset opacity (65%)"),
                ),
            ),
            1,
        ):
            slider.setRange(1 if key == "size" else 0, maximum)
            slider.setValue(
                int(
                    self.workspace.settings.value(
                        "camera_overlay/" + key, default
                    )
                )
            )
            slider.setMinimumWidth(120)
            slider.setAccessibleName(title)
            label = QtWidgets.QLabel(title)
            label.setBuddy(slider)
            value = QtWidgets.QLabel(str(slider.value()) + suffix)
            value.setMinimumWidth(
                value.fontMetrics().horizontalAdvance("100 px")
            )
            value.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            slider.valueChanged.connect(
                lambda number, target=value, unit=suffix: target.setText(
                    str(number) + unit
                )
            )
            form.addWidget(label, row, 0)
            form.addWidget(slider, row, 1)
            form.addWidget(value, row, 2)
            reset = QtWidgets.QToolButton()
            reset.setIcon(
                get_icon(
                    "rotate-cw",
                    theme["text_secondary"],
                    theme["text_secondary"],
                )
            )
            reset.setIconSize(QtCore.QSize(14, 14))
            reset.setFixedSize(24, 24)
            reset.setToolTip(reset_title)
            reset.setAccessibleName(reset_title)
            reset.clicked.connect(
                lambda checked=False, target=slider, initial=default: target.setValue(
                    initial
                )
            )
            self.overlay_reset_buttons[key] = reset
            form.addWidget(reset, row, 3)
        self.depth_legend = QtWidgets.QLabel(self.tr("Near → Far"))
        self.depth_legend.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.depth_legend.setStyleSheet(
            "color: #101828; padding: 3px 8px; border-radius: 3px; "
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #ff6441, stop:0.25 #ffd740, stop:0.5 #67dc78, "
            "stop:0.75 #32bee6, stop:1 #5a64eb);"
        )
        form.addWidget(self.depth_legend, 3, 0, 1, 4)
        action = QtWidgets.QWidgetAction(self.overlay_settings_menu)
        action.setDefaultWidget(content)
        self.overlay_settings_menu.addAction(action)
        self.view.overlay_item.setOpacity(self.overlay_opacity.value() / 100)
        self.depth_legend.setEnabled(
            self.overlay_color_buttons["depth"].isChecked()
        )
        self.overlay_color.buttonToggled.connect(
            lambda button, checked: (
                self._overlay_settings_changed() if checked else None
            )
        )
        self.overlay_size.valueChanged.connect(self._overlay_settings_changed)
        self.overlay_opacity.valueChanged.connect(
            self._overlay_opacity_changed
        )

    def _show_overlay_settings(self):
        button = self.overlay_settings_button
        self.overlay_settings_menu.popup(
            button.mapToGlobal(
                QtCore.QPoint(
                    button.width()
                    - self.overlay_settings_menu.sizeHint().width(),
                    button.height(),
                )
            )
        )

    def _overlay_settings_changed(self):
        self.workspace.settings.setValue(
            "camera_overlay/color",
            self.overlay_color.checkedButton().property("color_mode"),
        )
        self.workspace.settings.setValue(
            "camera_overlay/size", self.overlay_size.value()
        )
        self.depth_legend.setEnabled(
            self.overlay_color_buttons["depth"].isChecked()
        )
        self.refresh()

    def _overlay_opacity_changed(self, value):
        self.workspace.settings.setValue("camera_overlay/opacity", value)
        self.view.overlay_item.setOpacity(value / 100)

    def _update_depth_colors(self, points):
        transform = self.calibration["T_pointcloud_to_camera"]
        depth = points[self._indices, :3] @ transform[2, :3] + transform[2, 3]
        if not len(depth):
            self._depth_colors = np.empty((0, 3), dtype=np.uint8)
            self.depth_legend.setText(self.tr("Near → Far"))
            return
        near, far = np.percentile(depth, [2, 98])
        normalized = np.clip((depth - near) / max(far - near, 1e-6), 0, 1)
        stops = np.array(
            [
                [255, 100, 65],
                [255, 215, 64],
                [103, 220, 120],
                [50, 190, 230],
                [90, 100, 235],
            ]
        )
        self._depth_colors = np.column_stack(
            [
                np.interp(normalized, np.linspace(0, 1, len(stops)), channel)
                for channel in stops.T
            ]
        ).astype(np.uint8)
        self.depth_legend.setText(
            self.tr("Near {near:.3g} → Far {far:.3g}").format(
                near=near, far=far
            )
        )
        self.depth_legend.setToolTip(
            self.tr(
                "Camera depth in point-cloud units. Colors use the 2nd–98th percentile of this frame; values outside the range are clipped."
            )
        )

    def eventFilter(self, watched, event):
        if (
            event.type() == QtCore.QEvent.Type.Resize
            and watched is self.parentWidget()
        ):
            self.position_panel()
        return super().eventFilter(watched, event)

    def position_panel(self):
        area = self.workspace.viewport
        minimum = self.minimum_image_size()
        requested = self._user_size or minimum
        width = min(
            max(minimum.width(), requested.width()), max(1, area.width() - 16)
        )
        height = min(
            max(minimum.height(), requested.height()),
            max(1, area.height() - 16),
        )
        self.setGeometry(area.width() - width - 8, 8, width, height)
        self.raise_()

    def minimum_image_size(self):
        area = self.workspace.viewport
        width = min(420, max(180, int(area.width() * 0.42)), area.width() - 16)
        ratio = (
            self._image.height() / self._image.width()
            if not self._image.isNull()
            else 0.6
        )
        height = min(int(width * ratio) + 24, max(100, area.height() // 2))
        return QtCore.QSize(max(1, width), max(1, height))

    def resize_image_region(self, size):
        self._user_size = size
        self.position_panel()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_handle.move(
            0, max(0, self.height() - self.resize_handle.height())
        )
        self.resize_handle.raise_()

    def set_images_visible(self, visible):
        visible = bool(visible and self.sources)
        action = self.workspace.camera_panel_action
        action.setEnabled(bool(self.sources))
        action.setChecked(visible)
        title = (
            self.workspace.tr("Hide camera images")
            if visible
            else self.workspace.tr("Show camera images")
        )
        action.setText(title)
        action.setToolTip(title)
        self.workspace.camera_panel_button.setAccessibleName(title)
        self.setVisible(visible)
        if visible:
            self.position_panel()
            self.refresh()
        else:
            self.overlay_settings_menu.hide()

    def configure(self, directory, files, calibration, calibration_path):
        self.configure_sources(
            [
                CameraSource(
                    Path(directory), files, calibration, calibration_path
                )
            ]
        )

    def configure_sources(self, sources):
        previous = self.directory
        self.sources = list(sources)
        self.active_index = next(
            (
                index
                for index, source in enumerate(self.sources)
                if source.directory == previous
            ),
            0,
        )
        self._images.clear()
        self.view.set_camera_count(len(self.sources))
        self._activate_source()
        self.set_images_visible(bool(self.sources))

    def _activate_source(self):
        source = self.sources[self.active_index] if self.sources else None
        self.directory = source.directory if source else None
        self.files = source.files if source else {}
        self.calibration = source.calibration if source else None
        self.calibration_path = source.calibration_path if source else None
        self.filename.setText(source.name if source else "")
        self.overlay_action.setEnabled(self.calibration is not None)
        self.overlay_action.setToolTip(
            self.tr("Projected points")
            if self.calibration is not None
            else self.tr("Load calibration to overlay points.")
        )
        self._key = None
        self._cuboid_signature = None
        self._image = QtGui.QImage()
        self.view.image_item.setPixmap(QtGui.QPixmap())
        self.view.overlay_item.setPixmap(QtGui.QPixmap())
        self.cuboid_action.setEnabled(False)
        self.view.cuboid_item.setVisible(False)

    def step_camera(self, delta):
        if len(self.sources) < 2:
            return
        self.active_index = (self.active_index + delta) % len(self.sources)
        self._activate_source()
        self.refresh()

    def refresh(self, *args):
        if self.isHidden():
            return
        document = self.workspace.document
        if document is None:
            self.refresh_cuboids()
            self.overlay_action.setEnabled(False)
            self.message.setText(
                self.tr("Open a point cloud to view its camera image.")
            )
            self.message.show()
            self.view.image_item.setPixmap(QtGui.QPixmap())
            self.view.overlay_item.setPixmap(QtGui.QPixmap())
            self.view.show()
            return
        path = document.frame.path
        key = document
        if key != self._key:
            self._cuboid_signature = None
            self.view.cuboid_item.paths = []
            self.view.cuboid_item.setVisible(False)
            self.cuboid_action.setEnabled(False)
            self._key = key
            self._image = QtGui.QImage()
            self._indices = np.empty(0, dtype=np.int64)
            self._pixels = np.empty((0, 2), dtype=np.int32)
            self._depth_colors = np.empty((0, 3), dtype=np.uint8)
            self.depth_legend.setText(self.tr("Near → Far"))
            self.depth_legend.setToolTip("")
            self.overlay_action.setEnabled(False)
            self.view.overlay_item.setPixmap(QtGui.QPixmap())
            self.view.image_item.setPixmap(QtGui.QPixmap())
            image_path = self.files.get(path.stem)
            self.filename.setToolTip(
                str(image_path) if image_path else path.stem
            )
            image = self._images.pop(image_path, None)
            if image is None:
                image = (
                    QtGui.QImage(str(image_path))
                    if image_path
                    else QtGui.QImage()
                )
            if image.isNull():
                self.message.setText(
                    self.tr("No matching camera image for this frame.")
                )
                self.message.show()
                self.view.show()
                self.position_panel()
                return
            self._images[image_path] = image
            while len(self._images) > 4:
                self._images.popitem(last=False)
            self._image = image
            self.view.image_item.setPixmap(QtGui.QPixmap.fromImage(image))
            self.view.overlay_item.setPixmap(QtGui.QPixmap())
            self.view.setSceneRect(QtCore.QRectF(image.rect()))
            self.view.show()
            self.message.hide()
            self.overlay_action.setEnabled(self.calibration is not None)
            if self.calibration is not None:
                width, height = self.calibration["image_size"]
                if (image.width(), image.height()) != (width, height):
                    warning = self.tr(
                        "Image size %1 x %2 does not match calibration "
                        "%3 x %4. Projection disabled."
                    )
                    for index, value in enumerate(
                        (image.width(), image.height(), width, height), 1
                    ):
                        warning = warning.replace(f"%{index}", str(value))
                    self.message.setText(warning)
                    self.message.show()
                    self.overlay_action.setEnabled(False)
                    self.overlay_action.setToolTip(warning)
                else:
                    self.overlay_action.setToolTip(self.tr("Projected points"))
                    self._indices, self._pixels = project_points(
                        document.frame.points,
                        self.calibration,
                        image.width(),
                        image.height(),
                    )
                    self._update_depth_colors(document.frame.points)
            self.position_panel()
            self.view.fit_image()
        if self._image.isNull():
            return
        self.refresh_cuboids()
        show_overlay = (
            self.overlay_action.isEnabled() and self.overlay_action.isChecked()
        )
        self.view.overlay_item.setVisible(show_overlay)
        if not show_overlay:
            return
        mask = self.workspace._visible[self._indices]
        pixels = self._pixels[mask]
        indices = self._indices[mask]
        rgba = np.zeros(
            (self._image.height(), self._image.width(), 4), dtype=np.uint8
        )
        colors = (
            self._depth_colors[mask]
            if self.overlay_color_buttons["depth"].isChecked()
            else (self.workspace.viewport._colors[indices, :3] * 255).astype(
                np.uint8
            )
        )
        ranks = np.full(rgba.shape[:2], -1, dtype=np.int32)
        order = np.arange(len(pixels), dtype=np.int32)
        size = self.overlay_size.value()
        offsets = range(-(size // 2), size - size // 2)
        for dy in offsets:
            for dx in offsets:
                x, y = pixels[:, 0] + dx, pixels[:, 1] + dy
                inside = (
                    (x >= 0)
                    & (x < self._image.width())
                    & (y >= 0)
                    & (y < self._image.height())
                )
                np.maximum.at(ranks, (y[inside], x[inside]), order[inside])
        occupied = ranks >= 0
        rgba[occupied, :3] = colors[ranks[occupied]]
        rgba[occupied, 3] = 255
        overlay = QtGui.QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.strides[0],
            QtGui.QImage.Format.Format_RGBA8888,
        )
        self.view.overlay_item.setPixmap(QtGui.QPixmap.fromImage(overlay))

    def refresh_cuboids(self):
        source = self.workspace.viewport
        available = (
            self.workspace.document is not None
            and self._key is self.workspace.document
            and not self._image.isNull()
            and self.calibration is not None
            and (self._image.width(), self._image.height())
            == self.calibration["image_size"]
            and source.detection_enabled
        )
        self.cuboid_action.setEnabled(available)
        visible = available and self.cuboid_action.isChecked()
        self.view.cuboid_item.setVisible(visible)
        if not visible or self.isHidden():
            return
        signature = (
            source.cuboids,
            source.selected_id,
            tuple(source.class_colors.items()),
            source._creation_preview,
        )
        if signature != self._cuboid_signature:
            self.view.cuboid_item.set_cuboids(
                source.cuboids,
                source.selected_id,
                source.class_colors,
                self.calibration,
                self._image.width(),
                self._image.height(),
                source._creation_preview,
            )
            self._cuboid_signature = signature
