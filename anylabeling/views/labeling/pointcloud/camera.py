import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.utils.style import get_dialog_style

from .icons import get_icon


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


def project_points(points, calibration, width, height):
    if (width, height) != calibration["image_size"]:
        raise ValueError(
            "Image dimensions do not match calibration image_size."
        )
    transform = calibration["T_pointcloud_to_camera"]
    intrinsic = calibration["camera_matrix"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        camera = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
        indices = np.flatnonzero(valid)
        pixels = camera[indices, :2] / camera[indices, 2, None]
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


class CameraConfigurationDialog(QtWidgets.QDialog):
    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Camera image"))
        self.setStyleSheet(get_dialog_style())
        self.setMinimumWidth(540)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)
        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(12)
        form.setColumnStretch(0, 1)
        self.directory_input = QtWidgets.QLineEdit(str(panel.directory or ""))
        self.directory_input.setPlaceholderText(self.tr("Image directory"))
        self.directory_input.setAccessibleName(self.tr("Image directory"))
        self.calibration_input = QtWidgets.QLineEdit(
            str(panel.calibration_path or "")
        )
        self.calibration_input.setPlaceholderText(
            self.tr("Calibration JSON (optional)")
        )
        self.calibration_input.setAccessibleName(
            self.tr("Calibration JSON (optional)")
        )
        for row, (field, callback) in enumerate(
            (
                (self.directory_input, self._browse_directory),
                (self.calibration_input, self._browse_calibration),
            )
        ):
            button = QtWidgets.QPushButton(self.tr("Browse…"))
            button.setAutoDefault(False)
            button.clicked.connect(callback)
            field.setMinimumHeight(32)
            button.setMinimumHeight(32)
            form.addWidget(field, row, 0)
            form.addWidget(button, row, 1)
        layout.addLayout(form)
        self.error_label = QtWidgets.QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.error_label.hide()
        layout.addWidget(self.error_label)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch()
        cancel = QtWidgets.QPushButton(self.tr("Cancel"))
        cancel.clicked.connect(self.reject)
        confirm = QtWidgets.QPushButton(self.tr("OK"))
        confirm.setDefault(True)
        confirm.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        layout.addLayout(buttons)

    def _browse_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Image directory"),
            self.directory_input.text(),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if directory:
            self.directory_input.setText(directory)

    def _browse_calibration(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Calibration JSON"),
            self.calibration_input.text(),
            self.tr("JSON files (*.json)"),
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self.calibration_input.setText(path)

    def accept(self):
        directory = self.directory_input.text().strip()
        calibration = self.calibration_input.text().strip()
        if not directory:
            self.error_label.setText(self.tr("Choose an image directory."))
            self.error_label.show()
            return
        try:
            files = image_files(directory)
            parameters = load_calibration(calibration) if calibration else None
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            return
        self.configuration = (
            Path(directory),
            files,
            parameters,
            Path(calibration) if calibration else None,
        )
        super().accept()


class CameraView(QtWidgets.QGraphicsView):
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
        self.fitted = True

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

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        scale = self.transform().m11() * factor
        if 0.02 <= scale <= 32:
            self.fitted = False
            self.scale(factor, factor)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.fit_image()
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
        self.files = {}
        self.calibration = None
        self.calibration_path = None
        self.workspace = workspace
        self._key = None
        self._images = OrderedDict()
        self._indices = np.empty(0, dtype=np.int64)
        self._pixels = np.empty((0, 2), dtype=np.int32)
        self._image = QtGui.QImage()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QtWidgets.QFrame()
        header.setObjectName("pointcloudCameraFooter")
        header.setFixedHeight(24)
        row = QtWidgets.QHBoxLayout(header)
        row.setContentsMargins(2, 2, 2, 2)
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
        for action in (
            self.overlay_action,
            workspace._action(
                self.tr("Fit image"), self.view.fit_image, self, icon="fit"
            ),
        ):
            button = workspace._tool_button(action, row)
            button.setObjectName("pointcloudCameraTool")
            button.setIconSize(QtCore.QSize(13, 13))
            button.setFixedSize(20, 20)
            action.setIcon(
                get_icon(
                    "point-size" if action is self.overlay_action else "fit",
                    "#cbd2df",
                    "#cbd2df",
                )
            )
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        self.message.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.message.setObjectName("pointcloudMuted")
        layout.addWidget(self.message, 1)
        layout.addWidget(self.view, 1)
        layout.addWidget(header)
        workspace.viewport.installEventFilter(self)

    def eventFilter(self, watched, event):
        if (
            event.type() == QtCore.QEvent.Type.Resize
            and watched is self.parentWidget()
        ):
            self.position_panel()
        return super().eventFilter(watched, event)

    def position_panel(self):
        area = self.workspace.viewport
        width = min(420, max(180, int(area.width() * 0.42)), area.width() - 16)
        ratio = (
            self._image.height() / self._image.width()
            if not self._image.isNull()
            else 0.6
        )
        height = min(int(width * ratio) + 24, max(100, area.height() // 2))
        self.setGeometry(area.width() - width - 8, 8, width, height)
        self.raise_()

    def configure(self, directory, files, calibration, calibration_path):
        self.directory = directory
        self.files = files
        self.calibration = calibration
        self.calibration_path = calibration_path
        self.overlay_action.setEnabled(calibration is not None)
        self.overlay_action.setToolTip(
            self.tr("Projected points")
            if calibration is not None
            else self.tr("Load calibration to overlay points.")
        )
        self._images.clear()
        self._key = None
        self.show()
        self.position_panel()
        self.refresh()

    def refresh(self, *args):
        if self.isHidden():
            return
        document = self.workspace.document
        if document is None:
            self.overlay_action.setEnabled(False)
            self.message.setText(
                self.tr("Open a point cloud to view its camera image.")
            )
            self.message.show()
            self.view.hide()
            return
        path = document.frame.path
        key = document
        if key != self._key:
            self._key = key
            self._image = QtGui.QImage()
            self._indices = np.empty(0, dtype=np.int64)
            self._pixels = np.empty((0, 2), dtype=np.int32)
            self.overlay_action.setEnabled(False)
            self.view.overlay_item.setPixmap(QtGui.QPixmap())
            image_path = self.files.get(path.stem)
            self.filename.setText(image_path.name if image_path else path.stem)
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
                self.view.hide()
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
            self.position_panel()
            self.view.fit_image()
        if self._image.isNull():
            return
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
        colors = (self.workspace.viewport._colors[indices, :3] * 255).astype(
            np.uint8
        )
        for dx, dy in ((0, 0), (1, 0), (0, 1)):
            x = np.minimum(pixels[:, 0] + dx, self._image.width() - 1)
            y = np.minimum(pixels[:, 1] + dy, self._image.height() - 1)
            rgba[y, x, :3] = colors
            rgba[y, x, 3] = 230
        overlay = QtGui.QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.strides[0],
            QtGui.QImage.Format.Format_RGBA8888,
        )
        self.view.overlay_item.setPixmap(QtGui.QPixmap.fromImage(overlay))
