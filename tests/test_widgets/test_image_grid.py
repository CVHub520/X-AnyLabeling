import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.resources import resources  # noqa: F401
from anylabeling.views.labeling.utils.theme import get_mode, init_theme
from anylabeling.views.labeling.widgets.canvas import Canvas
from anylabeling.views.labeling.widgets.canvas_adjustment import (
    CanvasAdjustmentWidget,
)


class TestImageGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_controls_and_saved_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "settings.ini")
            settings = QtCore.QSettings(path, QtCore.QSettings.Format.IniFormat)
            with patch(
                "anylabeling.views.labeling.widgets.canvas_adjustment.QtCore.QSettings",
                return_value=settings,
            ):
                panel = CanvasAdjustmentWidget()
                self.assertEqual(panel.grid_size.value(), 100)
                self.assertEqual(panel.grid_color.currentData(), "white")
                self.assertEqual(panel.grid_opacity.value(), 0)
                self.assertFalse(panel.grid_labels.isChecked())
                self.assertFalse(panel.grid_popup.isVisible())
                changes = []
                panel.grid_changed.connect(lambda *values: changes.append(values))
                panel.grid_size.setValue(75)
                panel.grid_color.setCurrentIndex(2)
                panel.grid_opacity.setValue(50)
                panel.grid_labels.setChecked(True)
                self.assertEqual(changes[-1], (True, 75, "red", 50, True))
                panel.move(40, 150)
                panel.show()
                self.app.processEvents()
                width = panel.width()
                panel.grid_settings_button.click()
                self.app.processEvents()
                self.assertTrue(panel.grid_popup.isVisible())
                self.assertEqual(panel.width(), width)
                self.assertEqual(panel.grid_labels.x(), panel.grid_size.x())
                self.assertEqual(panel.grid_labels.x(), panel.grid_color.x())
                self.assertEqual(
                    panel.grid_popup.arrow_y,
                    panel.grid_color.geometry().center().y(),
                )
                self.assertAlmostEqual(
                    panel.grid_popup.mapToGlobal(QtCore.QPoint()).x() + 1,
                    panel.mapToGlobal(QtCore.QPoint(panel.width(), 0)).x(),
                    delta=2,
                )
                anchor = panel.grid_settings_button.mapToGlobal(
                    panel.grid_settings_button.rect().center()
                )
                self.assertEqual(
                    panel.grid_popup.mapToGlobal(QtCore.QPoint()).y()
                    + panel.grid_popup.arrow_y,
                    anchor.y() + 1,
                )
                panel.grid_popup.hide()
                panel.close()
                restored = CanvasAdjustmentWidget()
                self.assertEqual(restored.grid_size.value(), 100)
                self.assertEqual(restored.grid_color.currentData(), "red")
                self.assertEqual(restored.grid_opacity.value(), 50)
                self.assertFalse(restored.grid_labels.isChecked())
                restored.grid_size.setValue(75)
                restored.grid_labels.setChecked(True)
                grid_row = restored.content_widget.layout().itemAt(3).layout()
                self.assertIs(grid_row.itemAt(1).widget(), restored.grid_opacity)
                reset = grid_row.itemAt(3).widget()
                reset.click()
                self.assertEqual(restored.grid_opacity.value(), 0)
                self.assertEqual(restored.grid_size.value(), 100)
                self.assertEqual(restored.grid_color.currentData(), "white")
                self.assertFalse(restored.grid_labels.isChecked())
                self.assertEqual(settings.value("grid/size", type=int), 100)
                self.assertEqual(settings.value("grid/color"), "white")
                self.assertFalse(settings.value("grid/labels", type=bool))
                restored.close()

    def test_popup_uses_theme_colors_and_transparent_corners(self):
        original_mode = get_mode()
        try:
            with tempfile.TemporaryDirectory() as directory:
                settings = QtCore.QSettings(
                    os.path.join(directory, "settings.ini"),
                    QtCore.QSettings.Format.IniFormat,
                )
                with patch(
                    "anylabeling.views.labeling.widgets.canvas_adjustment.QtCore.QSettings",
                    return_value=settings,
                ):
                    styles = []
                    for mode in ("light", "dark"):
                        init_theme(mode)
                        panel = CanvasAdjustmentWidget()
                        panel.show()
                        panel.grid_settings_button.click()
                        self.app.processEvents()
                        image = panel.grid_popup.grab().toImage()
                        self.assertEqual(image.pixelColor(0, 0).alpha(), 0)
                        self.assertEqual(image.pixelColor(12, 20).name(), "#ffffff")
                        styles.append(
                            (
                                panel.grid_labels.styleSheet(),
                                panel.grid_size.styleSheet(),
                                panel.grid_color.styleSheet(),
                            )
                        )
                        panel.grid_popup.hide()
                        panel.close()
                    self.assertEqual(styles[0], styles[1])
        finally:
            init_theme(original_mode)

    def test_grid_spacing_and_opacity(self):
        canvas = Canvas(parent=None)
        canvas.resize(200, 200)
        canvas.pixmap = QtGui.QPixmap(200, 200)
        canvas.pixmap.fill(QtGui.QColor("black"))
        canvas.grid_enabled = True
        canvas.grid_size = 100
        canvas.grid_color = "white"
        canvas.grid_opacity = 100

        def render():
            image = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_RGB32)
            canvas.render(image)
            return image

        image = render()
        self.assertGreater(image.pixelColor(100, 50).red(), 200)
        self.assertEqual(image.pixelColor(50, 50), QtGui.QColor("black"))
        canvas.grid_opacity = 0
        self.assertEqual(render().pixelColor(100, 50), QtGui.QColor("black"))
        canvas.grid_opacity = 100
        canvas.scale = 0.5
        image = render()
        self.assertGreater(image.pixelColor(100, 75).red(), 200)
        canvas.close()

    def test_optional_labels_are_rendered_inside_cells(self):
        canvas = Canvas(parent=None)
        canvas.resize(200, 200)
        canvas.pixmap = QtGui.QPixmap(200, 200)
        canvas.pixmap.fill(QtGui.QColor("black"))
        canvas.grid_enabled = True
        image = QtGui.QImage(200, 200, QtGui.QImage.Format.Format_RGB32)
        canvas.render(image)
        without_labels = image.copy()
        canvas.grid_labels = True
        canvas.render(image)
        for x, y in ((0, 0), (100, 0), (0, 100)):
            self.assertTrue(
                any(
                    image.pixel(i, j) != without_labels.pixel(i, j)
                    for i in range(x + 3, x + 35)
                    for j in range(y + 3, y + 25)
                )
            )
        canvas.close()

    def test_partial_cells_are_completed_outside_the_image(self):
        canvas = Canvas(parent=None)
        canvas.resize(300, 300)
        canvas.pixmap = QtGui.QPixmap(150, 150)
        canvas.pixmap.fill(QtGui.QColor("black"))
        canvas.grid_enabled = True
        canvas.grid_size = 100
        canvas.grid_color = "red"
        image = QtGui.QImage(300, 300, QtGui.QImage.Format.Format_RGB32)
        canvas.render(image)
        for x, y in ((175, 250), (250, 175), (275, 175), (175, 275)):
            self.assertGreater(image.pixelColor(x, y).red(), 200)
            self.assertLess(image.pixelColor(x, y).green(), 50)
        canvas.close()


if __name__ == "__main__":
    unittest.main()
