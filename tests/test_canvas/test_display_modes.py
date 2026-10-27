import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.widgets.canvas import Canvas


class TestDisplayModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.canvas = Canvas(parent=None)
        self.canvas.resize(100, 100)
        self.canvas.pixmap = QtGui.QPixmap(100, 100)
        self.canvas.pixmap.fill(QtGui.QColor("blue"))
        self.canvas.show_labels = False
        self.canvas.show_scores = False
        self.canvas.show_linking = False
        self.canvas.mask_opacity = 255
        self.canvas.shapes = [self.rectangle(10), self.rectangle(50)]

    def tearDown(self):
        self.canvas.close()
        self.canvas.deleteLater()
        self.app.processEvents()

    @staticmethod
    def rectangle(x, group_id=None):
        shape = Shape(label="object", shape_type="rectangle", group_id=group_id)
        shape.points = [
            QtCore.QPointF(x, 10), QtCore.QPointF(x + 20, 10),
            QtCore.QPointF(x + 20, 30), QtCore.QPointF(x, 30),
        ]
        shape.line_color = QtGui.QColor("red")
        shape.close()
        return shape

    def render(self, mode):
        self.canvas.set_display_mode(mode)
        image = QtGui.QImage(self.canvas.size(), QtGui.QImage.Format.Format_RGB32)
        self.canvas.render(image)
        return image

    def test_instance_colors_and_label_restoration(self):
        before = [shape.to_dict() for shape in self.canvas.shapes]
        labels = self.render("label")
        self.assertEqual(labels.pixelColor(20, 20), labels.pixelColor(60, 20))
        instances = self.render("instance")
        self.assertNotEqual(instances.pixelColor(20, 20), instances.pixelColor(60, 20))
        self.assertEqual(before, [shape.to_dict() for shape in self.canvas.shapes])
        self.assertEqual(self.canvas.shapes[0].line_color, QtGui.QColor("red"))
        self.assertEqual(labels, self.render("label"))

    def test_group_colors_follow_group_changes(self):
        first, second = self.canvas.shapes
        first.group_id = second.group_id = 0
        grouped = self.render("group")
        self.assertEqual(grouped.pixelColor(20, 20), grouped.pixelColor(60, 20))
        second.group_id = 1
        split = self.render("group")
        self.assertNotEqual(split.pixelColor(20, 20), split.pixelColor(60, 20))
        first.group_id = second.group_id = None
        ungrouped = self.render("group")
        self.assertNotEqual(ungrouped.pixelColor(20, 20), ungrouped.pixelColor(60, 20))

    def test_instance_color_survives_reordering_and_deletion(self):
        initial = self.render("instance").pixelColor(60, 20)
        self.canvas.shapes.reverse()
        self.assertEqual(initial, self.render("instance").pixelColor(60, 20))
        self.canvas.shapes.pop()
        self.assertEqual(initial, self.render("instance").pixelColor(60, 20))

    def test_bitmap_is_binary_despite_overlays_and_opacity(self):
        self.canvas.shape_opacity = 0.1
        self.canvas.show_masks = False
        self.canvas.show_labels = True
        self.canvas.show_groups = True
        self.canvas.shapes[0].selected = True
        self.canvas.shapes[0].group_id = 7
        self.canvas.compare_pixmap = self.canvas.pixmap
        image = self.render("bitmap")
        self.assertEqual(image.pixelColor(20, 20), QtGui.QColor("white"))
        self.assertEqual(image.pixelColor(90, 90), QtGui.QColor("black"))
        colors = {image.pixel(x, y) for x in range(100) for y in range(100)}
        self.assertEqual(colors, {QtGui.qRgb(0, 0, 0), QtGui.qRgb(255, 255, 255)})

    def test_bitmap_respects_both_visibility_filters(self):
        self.canvas.shapes[0].visible = False
        self.canvas.visible[self.canvas.shapes[1]] = False
        image = self.render("bitmap")
        self.assertEqual(image.pixelColor(20, 20), QtGui.QColor("black"))
        self.assertEqual(image.pixelColor(60, 20), QtGui.QColor("black"))

    def test_bitmap_brush_mask_preserves_holes_and_other_shapes(self):
        brush = self.rectangle(10)
        brush._brush_using_mask = True
        brush.mask = np.zeros((100, 100), dtype=np.uint8)
        brush.mask[40:80, 10:40] = 255
        brush.mask[50:60, 20:30] = 0
        self.canvas.shapes.append(brush)
        image = self.render("bitmap")
        self.assertEqual(image.pixelColor(15, 45), QtGui.QColor("white"))
        self.assertEqual(image.pixelColor(25, 55), QtGui.QColor("black"))
        self.assertEqual(image.pixelColor(60, 20), QtGui.QColor("white"))

    def test_bitmap_two_point_rectangle_and_circle(self):
        rectangle = self.canvas.shapes[0]
        rectangle.points = [rectangle.points[0], rectangle.points[2]]
        circle = self.canvas.shapes[1]
        circle.shape_type = "circle"
        circle.points = [QtCore.QPointF(60, 60), QtCore.QPointF(70, 60)]
        image = self.render("bitmap")
        self.assertEqual(image.pixelColor(20, 20), QtGui.QColor("white"))
        self.assertEqual(image.pixelColor(60, 60), QtGui.QColor("white"))
        self.assertEqual(image.pixelColor(72, 60), QtGui.QColor("black"))
