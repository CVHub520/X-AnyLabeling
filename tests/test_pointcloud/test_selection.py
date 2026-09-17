import unittest
from dataclasses import replace

import numpy as np

from anylabeling.views.labeling.pointcloud.selection import (
    DEPTH_MAX,
    SelectionSnapshot,
    project_points,
    select_brush,
    select_polygon,
    validate_polygon,
)


class TestPointCloudSelection(unittest.TestCase):
    def snapshot(self, mode="surface", visible=None, size=2):
        screen = np.array(
            [
                [10, 10],
                [10, 10],
                [10, 10],
                [11, 10],
                [25, 10],
                [-1, 10],
                [10, 10],
            ],
            dtype=float,
        )
        depths = np.array([0.2, 0.7, 0.2, 0.8, 0.1, 0.1, 1.1])
        if visible is None:
            visible = np.ones(7, dtype=bool)
        return SelectionSnapshot.create(
            screen, depths, visible, 32, 24, size, mode
        )

    def test_surface_selects_ties_and_partially_visible_footprints(self):
        selected = select_brush(self.snapshot(), (10, 10), (10, 10), 3)
        np.testing.assert_array_equal(selected, [0, 2, 3])

    def test_occluded_fragment_of_partially_visible_point_is_not_hit(self):
        selected = select_brush(
            self.snapshot(), (10.5, 10.5), (10.5, 10.5), 0.1
        )
        np.testing.assert_array_equal(selected, [0, 2])
        selected = select_brush(
            self.snapshot(), (11.5, 10.5), (11.5, 10.5), 0.1
        )
        np.testing.assert_array_equal(selected, [3])

    def test_hidden_front_points_reveal_back_surface(self):
        selected = select_brush(
            self.snapshot(visible=np.array([0, 1, 0, 1, 1, 1, 1], dtype=bool)),
            (10, 10),
            (10, 10),
            3,
        )
        np.testing.assert_array_equal(selected, [1, 3])

    def test_through_still_excludes_hidden_and_clipped_points(self):
        visible = np.array([1, 1, 0, 1, 1, 1, 1], dtype=bool)
        selected = select_brush(
            self.snapshot("through", visible), (10, 10), (10, 10), 20
        )
        np.testing.assert_array_equal(selected, [0, 1, 3, 4])

    def test_complete_brush_segment_includes_points_between_events(self):
        selected = select_brush(self.snapshot("through"), (2, 10), (29, 10), 1)
        np.testing.assert_array_equal(selected, [0, 1, 2, 3, 4])

    def test_point_size_controls_edge_hit(self):
        point = np.array([[10.0, 10.0]])
        small = SelectionSnapshot.create(
            point, np.array([0.5]), np.array([True]), 32, 24, 1, "surface"
        )
        large = SelectionSnapshot.create(
            point, np.array([0.5]), np.array([True]), 32, 24, 4, "surface"
        )
        self.assertEqual(
            len(select_brush(small, (8.5, 8.5), (8.5, 8.5), 0.1)), 0
        )
        np.testing.assert_array_equal(
            select_brush(large, (8.5, 8.5), (8.5, 8.5), 0.1), [0]
        )

    def test_visibility_is_frozen_for_stroke(self):
        visible = np.ones(7, dtype=bool)
        snapshot = self.snapshot(visible=visible)
        visible[:] = False
        np.testing.assert_array_equal(
            select_brush(snapshot, (10, 10), (10, 10), 3), [0, 2, 3]
        )

    def test_depth_tolerance_is_limited_to_one_depth_quantization_bin(self):
        depth = 0.25
        snapshot = SelectionSnapshot.create(
            np.full((3, 2), 10.0),
            np.array([depth, depth + 0.1 / DEPTH_MAX, depth + 2 / DEPTH_MAX]),
            np.ones(3, dtype=bool),
            32,
            24,
            2,
            "surface",
        )
        np.testing.assert_array_equal(
            select_brush(snapshot, (10, 10), (10, 10), 3), [0, 1]
        )

    def test_polygon_obeys_surface_and_includes_pixel_center_boundary(self):
        polygon = [(9.5, 9.5), (11.5, 9.5), (11.5, 11.5), (9.5, 11.5)]
        np.testing.assert_array_equal(
            select_polygon(self.snapshot(), polygon), [0, 2, 3]
        )
        np.testing.assert_array_equal(
            select_polygon(self.snapshot("through"), polygon), [0, 1, 2, 3]
        )

    def test_invalid_polygons_are_rejected(self):
        for vertices in (
            [],
            [(1, 1), (2, 2)],
            [(1, 1), (2, 2), (3, 3)],
            [(0, 0), (4, 4), (0, 4), (3, 0)],
            [(0, 0), (4, 0), (4, 4), (4, 0), (0, 4)],
        ):
            with (
                self.subTest(vertices=vertices),
                self.assertRaises(ValueError),
            ):
                validate_polygon(vertices)

    def test_projection_rejects_behind_camera_and_outside_frustum(self):
        points = np.array([[0, 0, 0], [2, 0, 0], [0, 0, 2]], dtype=np.float32)
        screen, depths, visible = project_points(points, np.eye(4), 100, 80)
        np.testing.assert_array_equal(visible, [True, False, False])
        np.testing.assert_allclose(screen[0], [50, 40])
        self.assertEqual(depths[0], 0.5)
        matrix = np.eye(4)
        matrix[3, 3] = -1
        self.assertFalse(project_points(points, matrix, 100, 80)[2].any())

    def test_empty_filter_produces_no_selection(self):
        snapshot = self.snapshot(visible=np.zeros(7, dtype=bool))
        self.assertEqual(
            len(select_brush(snapshot, (10, 10), (10, 10), 30)), 0
        )

    def test_indexed_queries_match_complete_footprint_scan(self):
        rng = np.random.default_rng(31270)
        screen = rng.uniform((-3, -3), (259, 197), (6000, 2))
        depths = rng.random(len(screen))
        visible = rng.random(len(screen)) > 0.05
        screen[5500:] = screen[:500]
        depths[5500:] = depths[:500]
        for size in (1, 4, 11):
            for mode in ("surface", "through"):
                snapshot = SelectionSnapshot.create(
                    screen, depths, visible, 256, 194, size, mode
                )
                self.assertIsNotNone(snapshot._tile_order)
                reference = replace(snapshot)
                for start, end, radius in (
                    ((31.5, 31.5), (64.5, 32.5), 0.1),
                    ((-5, -5), (4, 4), 5),
                    ((254, 193), (259, 197), 4),
                    ((-100, -100), (-90, -90), 1),
                    ((400, 300), (450, 320), 1),
                    ((0, 0), (256, 194), 500),
                ):
                    with self.subTest(size=size, mode=mode, start=start):
                        np.testing.assert_array_equal(
                            select_brush(snapshot, start, end, radius),
                            select_brush(reference, start, end, radius),
                        )
                polygon = [(31.5, 31.5), (64.5, 32.5), (41.5, 100.5)]
                np.testing.assert_array_equal(
                    select_polygon(snapshot, polygon),
                    select_polygon(reference, polygon),
                )

    def test_surface_owner_query_expands_exact_footprint_depth_ties(self):
        screen = np.array([[10, 10], [10, 10], [10, 10], [11, 10]])
        depths = np.array([0.2, 0.2 + 0.1 / DEPTH_MAX, 0.2, 0.8])
        owners = np.zeros((24, 32), dtype=np.uint32)
        owners[9:11, 9:11] = 1
        owners[9:11, 11] = 4
        snapshot = SelectionSnapshot.create(
            screen,
            depths,
            np.ones(4, dtype=bool),
            32,
            24,
            2,
            "surface",
            surface_owners=owners,
        )
        np.testing.assert_array_equal(
            select_brush(snapshot, (10, 10), (10, 10), 3), [0, 2, 3]
        )
        np.testing.assert_array_equal(
            select_brush(snapshot, (10.5, 10.5), (10.5, 10.5), 0.1), [0, 2]
        )
        np.testing.assert_array_equal(
            select_brush(snapshot, (11.5, 10.5), (11.5, 10.5), 0.1), [3]
        )

    def test_surface_owners_with_mixed_depths_at_one_origin(self):
        owners = np.zeros((24, 32), dtype=np.uint32)
        owners[9:11, 9] = 1
        owners[9:11, 10] = 2
        snapshot = SelectionSnapshot.create(
            np.full((3, 2), 10.0),
            np.array([0.2, 0.7, 0.2]),
            np.ones(3, dtype=bool),
            32,
            24,
            2,
            "surface",
            surface_owners=owners,
        )
        np.testing.assert_array_equal(
            select_brush(snapshot, (10, 10), (10, 10), 3), [0, 1, 2]
        )

    def test_large_surface_region_preserves_footprints_and_ties(self):
        owners = np.zeros((600, 800), dtype=np.uint32)
        owners[9:11, 9:11] = 1
        owners[9:11, 11] = 4
        snapshot = SelectionSnapshot.create(
            np.array([[10, 10], [10, 10], [10, 10], [11, 10]]),
            np.array([0.2, 0.7, 0.2, 0.8]),
            np.ones(4, dtype=bool),
            800,
            600,
            2,
            "surface",
            surface_owners=owners,
        )
        np.testing.assert_array_equal(
            select_polygon(snapshot, [(0, 0), (800, 0), (800, 600), (0, 600)]),
            [0, 2, 3],
        )

    def test_packed_surface_index_preserves_ties_and_clipped_corners(self):
        screen = np.full((5000, 2), 100.0)
        screen[:8] = [
            [10, 10],
            [10, 10],
            [10, 10],
            [11, 10],
            [0, 0],
            [0, 0],
            [255, 193],
            [255, 193],
        ]
        depths = np.full(5000, 0.9, dtype=np.float32)
        depths[:8] = [0.2, 0.7, 0.2, 0.8, -0.0, 0.0, 0.4, 0.4]
        owners = np.zeros((194, 256), dtype=np.uint32)
        owners[9:11, 9:11] = 1
        owners[9:11, 11] = 4
        owners[0, 0] = 5
        owners[192:194, 254:256] = 7
        snapshot = SelectionSnapshot.create(
            screen,
            depths,
            np.ones(5000, dtype=bool),
            256,
            194,
            2,
            "surface",
            surface_owners=owners,
        )
        self.assertIsNotNone(snapshot._surface_keys)
        reference = replace(snapshot)
        for center, expected in (
            ((10, 10), [0, 2, 3]),
            ((0, 0), [4, 5]),
            ((255, 193), [6, 7]),
            ((100, 100), []),
        ):
            np.testing.assert_array_equal(
                select_brush(snapshot, center, center, 3), expected
            )
            np.testing.assert_array_equal(
                select_brush(snapshot, center, center, 3),
                select_brush(reference, center, center, 3),
            )
        polygon = [(0, 0), (256, 0), (256, 194), (0, 194)]
        np.testing.assert_array_equal(
            select_polygon(snapshot, polygon), [0, 2, 3, 4, 5, 6, 7]
        )

    def test_derived_indexes_do_not_change_snapshot_equality(self):
        screen = np.full((5000, 2), 10.0)
        depths = np.full(5000, 0.5, dtype=np.float32)
        owners = np.zeros((24, 32), dtype=np.uint32)
        owners[10, 10] = 1
        for mode in ("through", "surface"):
            snapshot = SelectionSnapshot.create(
                screen,
                depths,
                np.ones(5000, dtype=bool),
                32,
                24,
                1,
                mode,
                surface_owners=owners,
            )
            self.assertEqual(snapshot, replace(snapshot))

    def test_indexed_brush_preserves_nonfinite_radius_behavior(self):
        screen = np.full((5000, 2), 10.0)
        depths = np.full(5000, 0.5, dtype=np.float32)
        owners = np.zeros((24, 32), dtype=np.uint32)
        owners[10, 10] = 1
        for mode in ("through", "surface"):
            snapshot = SelectionSnapshot.create(
                screen,
                depths,
                np.ones(5000, dtype=bool),
                32,
                24,
                1,
                mode,
                surface_owners=owners,
            )
            np.testing.assert_array_equal(
                select_brush(snapshot, (10, 10), (10, 10), np.inf),
                np.arange(5000),
            )
            self.assertEqual(
                len(select_brush(snapshot, (10, 10), (10, 10), np.nan)), 0
            )

    def test_orthographic_fast_projection_preserves_pixel_boundaries(self):
        rng = np.random.default_rng(873)
        points = rng.uniform(-1, 1, (8000, 4)).astype(np.float32)
        points[:6, :3] = np.array(
            [
                [-1, -1, -1],
                [1, 1, 1],
                [0, 0, 0],
                [0.25, 0.5, 0.75],
                [0, 0, 1.00001],
                [0, -1.00001, 0],
            ]
        )
        for matrix in (
            np.eye(4, dtype=np.float32),
            np.array(
                [
                    [0.25, -0.125, 0.5, -0.25],
                    [-0.5, 0.25, 0.125, 0.5],
                    [0.125, 0.5, -0.25, 0.125],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            ),
        ):
            clip = points[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
            w = points[:, :3] @ matrix[3, :3] + matrix[3, 3]
            ndc = clip / w[:, None]
            expected_screen = np.column_stack(
                ((ndc[:, 0] + 1) * 1024 / 2, (1 - ndc[:, 1]) * 768 / 2)
            )
            screen, depths, valid = project_points(points, matrix, 1024, 768)
            np.testing.assert_array_equal(screen, expected_screen)
            np.testing.assert_array_equal(depths, (ndc[:, 2] + 1) / 2)
            np.testing.assert_array_equal(
                valid, (np.abs(ndc) <= 1).all(axis=1)
            )


if __name__ == "__main__":
    unittest.main()
