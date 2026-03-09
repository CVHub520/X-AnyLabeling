import math
import unittest

try:
    from PyQt6 import QtCore, QtGui

    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.widgets.canvas import (
        CUBOID_FACE_BACK,
        CUBOID_FACE_LEFT,
        CUBOID_FACE_RIGHT,
        CUBOID_FACE_TOP,
        Canvas,
    )

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for cuboid interaction tests")
class TestCuboidInteraction(unittest.TestCase):

    def setUp(self):
        self.canvas = Canvas.__new__(Canvas)
        self.canvas.cuboid_default_depth_vector = [24.0, -24.0]
        self.canvas.cuboid_min_depth = 5.0

    def create_cuboid_shape(self, depth_x=24.0, depth_y=-16.0):
        shape = Shape(label="Car", shape_type="cuboid")
        front_points = [
            QtCore.QPointF(20.0, 30.0),
            QtCore.QPointF(120.0, 30.0),
            QtCore.QPointF(120.0, 140.0),
            QtCore.QPointF(20.0, 140.0),
        ]
        depth_vector = QtCore.QPointF(depth_x, depth_y)
        self.canvas.set_cuboid_points(shape, front_points, depth_vector)
        return shape

    def assert_depth_parallel(self, shape):
        base_depth = shape.points[4] - shape.points[0]
        for i in range(4):
            depth = shape.points[i + 4] - shape.points[i]
            self.assertAlmostEqual(depth.x(), base_depth.x(), places=5)
            self.assertAlmostEqual(depth.y(), base_depth.y(), places=5)

    def test_visible_controls_follow_cvat_11_node_model(self):
        shape = self.create_cuboid_shape(depth_x=20.0, depth_y=-12.0)
        visible_controls = shape.get_cuboid_visible_control_indices()
        self.assertEqual(len(visible_controls), 11)
        self.assertEqual(len(set(visible_controls)), 11)
        self.assertIn(5, visible_controls)
        self.assertIn(6, visible_controls)
        self.assertNotIn(4, visible_controls)
        self.assertNotIn(7, visible_controls)

    def test_hovered_cuboid_draws_visible_controls(self):
        shape = self.create_cuboid_shape(depth_x=24.0, depth_y=-12.0)
        shape.selected = False
        shape.hovered = True
        drawn_indices = []
        original_draw_vertex = shape.draw_vertex

        def draw_vertex_spy(path, index, show_difficult=False):
            drawn_indices.append(index)
            original_draw_vertex(path, index, show_difficult)

        shape.draw_vertex = draw_vertex_spy
        image = QtGui.QImage(256, 256, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(image)
        shape.paint(painter)
        painter.end()

        for index in shape.get_cuboid_visible_control_indices():
            self.assertIn(index, drawn_indices)

    def test_cuboid_control_drag_keeps_depth_parallel(self):
        shape = self.create_cuboid_shape()
        self.canvas.move_cuboid_control(shape, 0, QtCore.QPointF(10.0, 20.0))
        self.canvas.move_cuboid_control(
            shape,
            Shape.CUBOID_BACK_RIGHT_EDGE_CENTER,
            QtCore.QPointF(180.0, 90.0),
        )
        self.assert_depth_parallel(shape)
        self.assertGreater(
            math.hypot(
                shape.get_cuboid_depth_vector()[0],
                shape.get_cuboid_depth_vector()[1],
            ),
            0.0,
        )

    def test_non_front_face_drag_updates_geometry(self):
        shape = self.create_cuboid_shape(depth_x=30.0, depth_y=-10.0)
        old_width = abs(shape.points[1].x() - shape.points[0].x())
        old_depth = shape.get_cuboid_depth_vector()

        self.canvas.move_cuboid_face_by(
            shape, CUBOID_FACE_RIGHT, QtCore.QPointF(8.0, 0.0)
        )
        self.assertGreater(abs(shape.points[1].x() - shape.points[0].x()), old_width)
        self.assertAlmostEqual(
            shape.get_cuboid_depth_vector()[0], old_depth[0], places=5
        )
        self.assertAlmostEqual(
            shape.get_cuboid_depth_vector()[1], old_depth[1], places=5
        )

        self.canvas.move_cuboid_face_by(
            shape, CUBOID_FACE_BACK, QtCore.QPointF(6.0, -3.0)
        )
        new_depth = shape.get_cuboid_depth_vector()
        self.assertAlmostEqual(new_depth[0], old_depth[0] + 6.0, places=5)
        self.assertAlmostEqual(new_depth[1], old_depth[1] - 3.0, places=5)
        self.assert_depth_parallel(shape)

    def test_left_face_drag_moves_in_xy_and_respects_back_right_constraint(self):
        shape = self.create_cuboid_shape(depth_x=30.0, depth_y=-10.0)
        margin = self.canvas.cuboid_constraint_margin()
        left_indices = [0, 3, 4, 7]
        right_indices = [1, 2, 5, 6]
        left_before = [QtCore.QPointF(shape.points[i]) for i in left_indices]
        right_before = [QtCore.QPointF(shape.points[i]) for i in right_indices]
        front_right_mid = (shape.points[1].x() + shape.points[2].x()) / 2.0
        back_right_mid = (shape.points[5].x() + shape.points[6].x()) / 2.0
        right_top, right_bottom = (1, 2)
        if back_right_mid > front_right_mid:
            right_top, right_bottom = (5, 6)
        right_limit = (
            min(
                shape.points[right_top].x(),
                (shape.points[right_top].x() + shape.points[right_bottom].x()) / 2.0,
                shape.points[right_bottom].x(),
            )
            - margin
        )

        self.canvas.move_cuboid_face_by(
            shape, CUBOID_FACE_LEFT, QtCore.QPointF(300.0, 18.0)
        )

        dx = shape.points[0].x() - left_before[0].x()
        dy = shape.points[0].y() - left_before[0].y()
        self.assertAlmostEqual(dy, 18.0, places=5)
        for i, before in zip(left_indices, left_before):
            self.assertAlmostEqual(shape.points[i].x(), before.x() + dx, places=5)
            self.assertAlmostEqual(shape.points[i].y(), before.y() + dy, places=5)
        for i, before in zip(right_indices, right_before):
            self.assertAlmostEqual(shape.points[i].x(), before.x(), places=5)
            self.assertAlmostEqual(shape.points[i].y(), before.y(), places=5)
        self.assertLessEqual(
            max(shape.points[i].x() for i in left_indices), right_limit
        )

    def test_right_face_drag_moves_in_xy_and_respects_back_left_constraint(self):
        shape = self.create_cuboid_shape(depth_x=30.0, depth_y=-10.0)
        margin = self.canvas.cuboid_constraint_margin()
        left_indices = [0, 3, 4, 7]
        right_indices = [1, 2, 5, 6]
        left_before = [QtCore.QPointF(shape.points[i]) for i in left_indices]
        right_before = [QtCore.QPointF(shape.points[i]) for i in right_indices]
        front_left_mid = (shape.points[0].x() + shape.points[3].x()) / 2.0
        back_left_mid = (shape.points[4].x() + shape.points[7].x()) / 2.0
        left_top, left_bottom = (0, 3)
        if back_left_mid < front_left_mid:
            left_top, left_bottom = (4, 7)
        left_limit = (
            max(
                shape.points[left_top].x(),
                (shape.points[left_top].x() + shape.points[left_bottom].x()) / 2.0,
                shape.points[left_bottom].x(),
            )
            + margin
        )

        self.canvas.move_cuboid_face_by(
            shape, CUBOID_FACE_RIGHT, QtCore.QPointF(-300.0, -14.0)
        )

        dx = shape.points[1].x() - right_before[0].x()
        dy = shape.points[1].y() - right_before[0].y()
        self.assertAlmostEqual(dy, -14.0, places=5)
        for i, before in zip(right_indices, right_before):
            self.assertAlmostEqual(shape.points[i].x(), before.x() + dx, places=5)
            self.assertAlmostEqual(shape.points[i].y(), before.y() + dy, places=5)
        for i, before in zip(left_indices, left_before):
            self.assertAlmostEqual(shape.points[i].x(), before.x(), places=5)
            self.assertAlmostEqual(shape.points[i].y(), before.y(), places=5)
        self.assertGreaterEqual(
            min(shape.points[i].x() for i in right_indices), left_limit
        )

    def test_top_face_drag_is_invalid(self):
        shape = self.create_cuboid_shape(depth_x=30.0, depth_y=-10.0)
        before_points = [QtCore.QPointF(p) for p in shape.points]
        self.canvas.move_cuboid_face_by(
            shape, CUBOID_FACE_TOP, QtCore.QPointF(0.0, -20.0)
        )
        for p1, p2 in zip(before_points, shape.points):
            self.assertAlmostEqual(p1.x(), p2.x(), places=5)
            self.assertAlmostEqual(p1.y(), p2.y(), places=5)

    def test_back_center_has_front_margin_constraint(self):
        shape = self.create_cuboid_shape(depth_x=24.0, depth_y=-12.0)
        margin = self.canvas.cuboid_constraint_margin()
        front_right = max(shape.points[1].x(), shape.points[2].x())
        self.canvas.move_cuboid_control(
            shape,
            Shape.CUBOID_BACK_RIGHT_EDGE_CENTER,
            QtCore.QPointF(front_right - 200.0, 0.0),
        )
        center = shape.get_cuboid_control_point(
            Shape.CUBOID_BACK_RIGHT_EDGE_CENTER
        )
        self.assertGreaterEqual(center.x(), front_right + margin)

    def test_back_center_moves_with_depth_slope(self):
        shape = self.create_cuboid_shape(depth_x=24.0, depth_y=-12.0)
        front_center = shape.get_cuboid_control_point(
            Shape.CUBOID_FRONT_RIGHT_EDGE_CENTER
        )
        before_center = shape.get_cuboid_control_point(
            Shape.CUBOID_BACK_RIGHT_EDGE_CENTER
        )
        self.canvas.move_cuboid_control(
            shape,
            Shape.CUBOID_BACK_RIGHT_EDGE_CENTER,
            QtCore.QPointF(before_center.x() + 20.0, before_center.y() + 120.0),
        )
        after_center = shape.get_cuboid_control_point(
            Shape.CUBOID_BACK_RIGHT_EDGE_CENTER
        )

        self.assertGreater(after_center.x(), before_center.x())
        self.assertLess(after_center.y(), before_center.y())
        slope = (before_center.y() - front_center.y()) / (
            before_center.x() - front_center.x()
        )
        expected_dy = (after_center.x() - before_center.x()) * slope
        self.assertAlmostEqual(
            after_center.y() - before_center.y(), expected_dy, places=5
        )

    def test_back_face_drag_crosses_orientation_and_switches_controls(self):
        shape = self.create_cuboid_shape(depth_x=30.0, depth_y=-10.0)
        front_before = [QtCore.QPointF(shape.points[i]) for i in range(4)]
        back_before = [QtCore.QPointF(shape.points[i]) for i in range(4, 8)]

        self.assertIn(5, shape.get_cuboid_visible_control_indices())
        self.assertIn(Shape.CUBOID_BACK_RIGHT_EDGE_CENTER, shape.get_cuboid_visible_control_indices())

        self.canvas.move_cuboid_face_by(
            shape, CUBOID_FACE_BACK, QtCore.QPointF(-140.0, 30.0)
        )

        for i in range(4):
            self.assertAlmostEqual(shape.points[i].x(), front_before[i].x(), places=5)
            self.assertAlmostEqual(shape.points[i].y(), front_before[i].y(), places=5)
        for i in range(4):
            self.assertAlmostEqual(
                shape.points[i + 4].x(), back_before[i].x() - 140.0, places=5
            )
            self.assertAlmostEqual(
                shape.points[i + 4].y(), back_before[i].y() + 30.0, places=5
            )

        self.assertLess(shape.get_cuboid_depth_vector()[0], 0.0)
        visible_controls = shape.get_cuboid_visible_control_indices()
        self.assertIn(4, visible_controls)
        self.assertIn(7, visible_controls)
        self.assertIn(Shape.CUBOID_BACK_LEFT_EDGE_CENTER, visible_controls)
        self.assertNotIn(5, visible_controls)
        self.assertNotIn(6, visible_controls)
        self.assertNotIn(Shape.CUBOID_BACK_RIGHT_EDGE_CENTER, visible_controls)

    def test_visible_back_top_vertex_moves_only_vertically(self):
        shape = self.create_cuboid_shape(depth_x=24.0, depth_y=-12.0)
        top_index, bottom_index = shape.get_cuboid_visible_rear_edge_indices()
        margin = self.canvas.cuboid_constraint_margin()
        top_indices = [0, 1, 4, 5]
        bottom_indices = [2, 3, 6, 7]
        top_before = [QtCore.QPointF(shape.points[i]) for i in top_indices]
        bottom_before = [QtCore.QPointF(shape.points[i]) for i in bottom_indices]
        self.canvas.move_cuboid_control(
            shape,
            top_index,
            QtCore.QPointF(
                shape.points[top_index].x() + 30.0,
                shape.points[top_index].y() + 8.0,
            ),
        )
        dy = shape.points[top_index].y() - top_before[top_indices.index(top_index)].y()
        for i, before in zip(top_indices, top_before):
            self.assertAlmostEqual(shape.points[i].x(), before.x(), places=5)
            self.assertAlmostEqual(shape.points[i].y(), before.y() + dy, places=5)
        for i, before in zip(bottom_indices, bottom_before):
            self.assertAlmostEqual(shape.points[i].x(), before.x(), places=5)
            self.assertAlmostEqual(shape.points[i].y(), before.y(), places=5)
        self.assertLessEqual(shape.points[top_index].y(), shape.points[bottom_index].y() - margin)

    def test_front_edit_preserves_per_vertex_back_offsets(self):
        shape = self.create_cuboid_shape(depth_x=24.0, depth_y=-12.0)
        top_index, _ = shape.get_cuboid_visible_rear_edge_indices()
        self.canvas.move_cuboid_control(
            shape,
            top_index,
            QtCore.QPointF(shape.points[top_index].x(), shape.points[top_index].y() + 8.0),
        )
        before_offsets = [shape.points[i + 4] - shape.points[i] for i in range(4)]
        self.canvas.move_cuboid_control(
            shape,
            Shape.CUBOID_FRONT_RIGHT_EDGE_CENTER,
            QtCore.QPointF(shape.get_cuboid_control_point(Shape.CUBOID_FRONT_RIGHT_EDGE_CENTER).x() + 12.0, 0.0),
        )
        after_offsets = [shape.points[i + 4] - shape.points[i] for i in range(4)]
        for before, after in zip(before_offsets, after_offsets):
            self.assertAlmostEqual(before.x(), after.x(), places=5)
            self.assertAlmostEqual(before.y(), after.y(), places=5)

    def test_left_orientation_back_controls_mirror_constraints(self):
        shape = self.create_cuboid_shape(depth_x=-24.0, depth_y=-12.0)
        margin = self.canvas.cuboid_constraint_margin()
        visible_controls = shape.get_cuboid_visible_control_indices()
        self.assertIn(4, visible_controls)
        self.assertIn(7, visible_controls)
        self.assertIn(Shape.CUBOID_BACK_LEFT_EDGE_CENTER, visible_controls)
        self.assertNotIn(Shape.CUBOID_BACK_RIGHT_EDGE_CENTER, visible_controls)

        front_left = min(shape.points[0].x(), shape.points[3].x())
        self.canvas.move_cuboid_control(
            shape,
            Shape.CUBOID_BACK_LEFT_EDGE_CENTER,
            QtCore.QPointF(front_left + 200.0, 0.0),
        )
        center = shape.get_cuboid_control_point(Shape.CUBOID_BACK_LEFT_EDGE_CENTER)
        self.assertLessEqual(center.x(), front_left - margin)

    def test_front_vertex_constraint_keeps_margin(self):
        shape = self.create_cuboid_shape(depth_x=24.0, depth_y=-12.0)
        margin = self.canvas.cuboid_constraint_margin()
        self.canvas.move_cuboid_control(
            shape,
            0,
            QtCore.QPointF(shape.points[2].x() + 200.0, shape.points[2].y() + 200.0),
        )
        self.assertLessEqual(shape.points[0].x(), shape.points[2].x() - margin)
        self.assertLessEqual(shape.points[0].y(), shape.points[2].y() - margin)

    def test_cuboid_save_reload_keeps_order_and_depth_vector(self):
        shape = self.create_cuboid_shape(depth_x=26.0, depth_y=-14.0)
        data = shape.to_dict()
        loaded_shape = Shape().load_from_dict(data)

        self.assertEqual(len(loaded_shape.points), 8)
        for p1, p2 in zip(shape.points, loaded_shape.points):
            self.assertAlmostEqual(p1.x(), p2.x(), places=5)
            self.assertAlmostEqual(p1.y(), p2.y(), places=5)

        depth_vector = loaded_shape.get_cuboid_depth_vector()
        expected_depth = loaded_shape.points[4] - loaded_shape.points[0]
        self.assertAlmostEqual(depth_vector[0], expected_depth.x(), places=5)
        self.assertAlmostEqual(depth_vector[1], expected_depth.y(), places=5)


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is required for shape regression tests")
class TestShapeRegression(unittest.TestCase):

    def test_non_cuboid_nearest_vertex_unchanged(self):
        rectangle = Shape(shape_type="rectangle")
        rectangle.points = [
            QtCore.QPointF(10.0, 10.0),
            QtCore.QPointF(50.0, 10.0),
            QtCore.QPointF(50.0, 40.0),
            QtCore.QPointF(10.0, 40.0),
        ]
        self.assertEqual(
            rectangle.nearest_vertex(QtCore.QPointF(11.0, 9.0), 5.0), 0
        )

        polygon = Shape(shape_type="polygon")
        polygon.points = [
            QtCore.QPointF(0.0, 0.0),
            QtCore.QPointF(8.0, 0.0),
            QtCore.QPointF(8.0, 8.0),
        ]
        self.assertEqual(
            polygon.nearest_vertex(QtCore.QPointF(8.0, 7.5), 2.0), 2
        )

        rotation = Shape(shape_type="rotation")
        rotation.points = [
            QtCore.QPointF(20.0, 20.0),
            QtCore.QPointF(40.0, 20.0),
            QtCore.QPointF(40.0, 40.0),
            QtCore.QPointF(20.0, 40.0),
        ]
        self.assertEqual(
            rotation.nearest_vertex(QtCore.QPointF(21.0, 21.0), 3.0), 0
        )
