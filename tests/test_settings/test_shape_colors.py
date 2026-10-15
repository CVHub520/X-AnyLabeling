import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.views.labeling.label_widget import LabelingWidget
from anylabeling.views.labeling.settings.runtime_applier import (
    SettingsRuntimeApplier,
)
from anylabeling.views.labeling.settings.schema import load_template_config
from anylabeling.views.labeling.shape import Shape


class TestShapeColors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
            []
        )

    def setUp(self):
        self.config = copy.deepcopy(load_template_config())
        self.rgb = (31, 97, 173)
        self.widget = SimpleNamespace(
            _config=self.config,
            _get_rgb_by_label=Mock(return_value=self.rgb),
            canvas=SimpleNamespace(shapes=[], update=Mock()),
        )
        self.widget._update_shape_color = (
            lambda shape: LabelingWidget._update_shape_color(
                self.widget, shape
            )
        )
        self.original = {
            key: getattr(Shape, key)
            for key in (*self.expected(), "line_width", "point_size")
        }
        self.addCleanup(self.restore_shape_defaults)

    def restore_shape_defaults(self):
        for key, value in self.original.items():
            setattr(Shape, key, value)

    def expected(self):
        return {
            "line_color": (*self.rgb, 255),
            "vertex_fill_color": (*self.rgb, 255),
            "hvertex_fill_color": (255, 255, 255, 255),
            "fill_color": (*self.rgb, 128),
            "select_line_color": (255, 255, 255, 255),
            "select_fill_color": (*self.rgb, 155),
        }

    def assert_colors(self, shape, expected):
        for key, value in expected.items():
            self.assertEqual(getattr(shape, key).getRgb(), value, key)

    def test_default_label_palette_is_preserved(self):
        shape = Shape(label="object")
        self.widget._update_shape_color(shape)
        self.assert_colors(shape, self.expected())

    def test_each_override_survives_recolor_and_new_shape_and_reset(self):
        for key in self.expected():
            with self.subTest(key=key):
                shape = Shape(label="object")
                self.widget.canvas.shapes = [shape]
                self.config["shape"][key] = [201, 42, 83, 67]
                applier = SettingsRuntimeApplier(self.widget)
                applier.apply_change("shape." + key, self.config["shape"][key])
                expected = self.expected()
                expected[key] = (201, 42, 83, 67)
                self.assert_colors(shape, expected)
                self.widget._update_shape_color(shape)
                self.assert_colors(shape, expected)
                new_shape = Shape(label="object")
                self.widget._update_shape_color(new_shape)
                self.assert_colors(new_shape, expected)
                self.config["shape"][key] = copy.deepcopy(
                    load_template_config()["shape"][key]
                )
                applier.apply_change("shape." + key, self.config["shape"][key])
                self.assert_colors(shape, self.expected())

    def test_strategy_changes_preserve_overrides_and_label_list_colors(self):
        self.config["shape"]["fill_color"] = [201, 42, 83, 67]
        shape = Shape(label="object")
        item = Mock()
        item.shape.return_value = shape
        self.widget.canvas.shapes = [shape]
        self.widget.label_info = {"object": {"color": list(self.rgb)}}
        self.widget.label_list = [item]
        self.widget.unique_label_list = Mock()
        self.widget.image_tags_widget = Mock()
        for strategy in ("auto", "manual", None):
            with self.subTest(strategy=strategy):
                self.config["shape_color"] = strategy
                SettingsRuntimeApplier(self.widget).apply_change(
                    "shape_color", strategy
                )
                self.assertEqual(shape.fill_color.getRgb(), (201, 42, 83, 67))
                self.assertEqual(shape.line_color.getRgb(), (*self.rgb, 255))
                self.assertEqual(
                    item.setBackground.call_args.args[0].getRgb(),
                    (*self.rgb, 128),
                )

    def test_normal_and_selected_fill_render_independently(self):
        self.config["shape"]["fill_color"] = [201, 42, 83, 255]
        self.config["shape"]["select_fill_color"] = [12, 164, 75, 255]
        shape = Shape(label="object", shape_type="rectangle")
        shape.points = [
            QtCore.QPointF(10, 10),
            QtCore.QPointF(90, 10),
            QtCore.QPointF(90, 90),
            QtCore.QPointF(10, 90),
        ]
        shape.close()
        shape.fill = True
        self.widget._update_shape_color(shape)
        for selected, color in (
            (False, (201, 42, 83, 255)),
            (True, (12, 164, 75, 255)),
        ):
            shape.selected = selected
            image = QtGui.QImage(100, 100, QtGui.QImage.Format.Format_ARGB32)
            image.fill(QtGui.QColor("white"))
            painter = QtGui.QPainter(image)
            try:
                shape.paint(painter)
            finally:
                painter.end()
            self.assertEqual(image.pixelColor(50, 50).getRgb(), color)
