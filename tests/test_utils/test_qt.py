import os
import unittest
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore

    from anylabeling.views.labeling import utils
    from anylabeling.views.labeling.shape import Shape

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for Qt utility tests")
class TestQtUtils(unittest.TestCase):

    def test_distance_to_line_handles_2d_points_without_numpy_warning(self):
        point = QtCore.QPointF(5.0, 5.0)
        line = [QtCore.QPointF(0.0, 0.0), QtCore.QPointF(10.0, 0.0)]

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            distance = utils.distance_to_line(point, line)

        self.assertEqual(distance, 5.0)

    def test_nearest_edge_handles_line_shapes_without_numpy_warning(self):
        shape = Shape(label="line", shape_type="line")
        shape.points = [
            QtCore.QPointF(0.0, 0.0),
            QtCore.QPointF(10.0, 0.0),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            edge = shape.nearest_edge(QtCore.QPointF(5.0, 1.0), 2.0)

        self.assertIn(edge, (0, 1))
