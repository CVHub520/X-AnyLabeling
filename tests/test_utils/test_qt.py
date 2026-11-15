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

    def test_distance_to_segment_endpoints_projection_and_degenerate_edges(
        self,
    ):
        for start, end, point, expected in (
            ((0, 0), (10, 0), (-3, 4), 5),
            ((0, 0), (10, 0), (13, 4), 5),
            ((0, 0), (10, 0), (5, 4), 4),
            ((0, 0), (10, 0), (0, 4), 4),
            ((0, 0), (10, 0), (10, 4), 4),
            ((0, 0), (10, 0), (5, 0), 0),
            ((1, 2), (1, 2), (4, 6), 5),
            ((1, 2), (1, 2), (1, 2), 0),
            ((0, 0), (6, 8), (-1, 7), 5),
            ((-2, -4), (-2, 8), (1, 2), 3),
        ):
            with self.subTest(start=start, end=end, point=point):
                line = [QtCore.QPointF(*start), QtCore.QPointF(*end)]
                query = QtCore.QPointF(*point)
                self.assertAlmostEqual(
                    utils.distance_to_line(query, line), expected
                )
                self.assertAlmostEqual(
                    utils.distance_to_line(query, line[::-1]), expected
                )

    def test_nearest_vertex_preserves_ties_tolerance_and_cuboid_controls(self):
        shape = Shape(shape_type="polygon")
        shape.points = [QtCore.QPointF(-3, 4), QtCore.QPointF(3, 4)]
        query = QtCore.QPointF(0, 0)
        self.assertEqual(shape.nearest_vertex(query, 5), 0)
        self.assertIsNone(shape.nearest_vertex(query, 4.99))
        shape.points = []
        self.assertIsNone(shape.nearest_vertex(query, 5))
        shape.shape_type = "cuboid"
        shape.points = [QtCore.QPointF(20, 20)] * 4 + [query] * 4
        self.assertIsNone(shape.nearest_vertex(query, 5))

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
