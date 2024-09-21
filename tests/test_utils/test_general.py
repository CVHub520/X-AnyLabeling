import unittest

from anylabeling.views.labeling.utils.general import is_possible_rectangle


class TestIsRectangle(unittest.TestCase):

    def test_normal_rectangle(self):
        points = [[0, 0], [1000, 0], [1000, 1], [0, 1]]
        self.assertEqual(is_possible_rectangle(points), True)

    def test_irregular_shape(self):
        points = [[0, 0], [2, 3], [4, 5], [6, 7]]
        self.assertEqual(is_possible_rectangle(points), False)

    def test_rectangle_with_square_shape(self):
        points = [[0, 0], [0, 1], [1, 1], [1, 0]]
        self.assertEqual(is_possible_rectangle(points), True)

    def test_rectangle_with_diagonal_points(self):
        points = [[1, 1], [1, 2], [2, 1], [2, 2]]
        self.assertEqual(is_possible_rectangle(points), True)

    def test_lese_than_four_points(self):
        points = [[0, 0], [1, 1], [1, 0]]
        self.assertEqual(is_possible_rectangle(points), False)

    def test_more_than_four_points(self):
        points = [[0, 0], [1, 1], [1, 0], [2, 0], [1, 2]]
        self.assertEqual(is_possible_rectangle(points), False)
