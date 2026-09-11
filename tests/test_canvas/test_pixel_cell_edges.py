import numpy as np
from shapely.geometry import Polygon

from anylabeling.views.labeling.utils.pixel_cell_edges import (
    _repair_radii,
    _trace_cells,
    fit_rectangle_to_edges,
    refine_polygon_to_edge,
    segment_box_to_edge_polygon,
    validate_polygon_edge_fit,
)


def assert_grid(points):
    np.testing.assert_allclose(points - 0.5, np.round(points - 0.5))
    delta = np.roll(points, -1, axis=0) - points
    assert np.all((delta[:, 0] == 0) | (delta[:, 1] == 0))


def test_bright_gray_boundary_uses_outer_gray_edge_not_gray_black():
    image = np.full((50, 50), 220, np.uint8)
    image[10:40, 10:40] = 160
    image[15:35, 15:35] = 20
    result = segment_box_to_edge_polygon(image, [[10, 10], [39, 39]])
    assert result.succeeded, result.reason
    np.testing.assert_allclose(result.points.min(axis=0), [9.5, 9.5])
    np.testing.assert_allclose(result.points.max(axis=0), [39.5, 39.5])
    assert_grid(result.points)


def test_bright_core_inside_gray_halo_uses_core_boundary():
    image = np.full((50, 50), 20, np.uint8)
    image[10:40, 10:40] = 160
    image[15:35, 15:35] = 220
    result = segment_box_to_edge_polygon(image, [[14, 14], [35, 35]])
    assert result.succeeded
    np.testing.assert_allclose(result.points.min(axis=0), [14.5, 14.5])
    np.testing.assert_allclose(result.points.max(axis=0), [34.5, 34.5])


def test_corner_contacts_never_make_diagonal_edges():
    mask = np.zeros((8, 8), np.uint8)
    mask[2:4, 2:4] = 1
    mask[4:6, 4:6] = 1
    loops = _trace_cells(mask, 100)
    assert len(loops) == 2
    for points in loops:
        assert_grid(points)
        assert len(points) == 4


def test_single_pixel_object_has_four_exact_cell_corners():
    image = np.zeros((9, 9), np.uint8)
    image[4, 4] = 255
    result = segment_box_to_edge_polygon(image, [[3, 3], [5, 5]])
    assert result.succeeded
    assert len(result.points) == 4
    assert_grid(result.points)


def test_sparse_old_annotation_reconstructs_staircase_without_mutation():
    image = np.zeros((30, 30), np.uint8)
    image[8:20, 8:20] = 220
    image[5:8, 10:17] = 220
    original = np.array([[7.0, 4.0], [20.0, 4.0], [20.0, 20.0], [7.0, 20.0]])
    snapshot = original.copy()
    result = refine_polygon_to_edge(
        image, original, {"search_radius": 5, "point_spacing": 100}
    )
    assert result.succeeded, result.reason
    assert len(result.points) > 4
    assert_grid(result.points)
    np.testing.assert_array_equal(original, snapshot)


def test_no_silent_opposite_boundary_fallback():
    image = np.full((50, 50), 220, np.uint8)
    image[5:45, 5:45] = 160
    image[20:30, 20:30] = 20
    original = [[19.5, 19.5], [29.5, 19.5], [29.5, 29.5], [19.5, 29.5]]
    result = refine_polygon_to_edge(
        image,
        original,
        {
            "boundary_side": "bright",
            "search_radius": 20,
            "max_area_change": 0.6,
        },
    )
    assert not result.succeeded


def test_diagonal_and_self_intersection_cannot_be_confirmed():
    image = np.zeros((20, 20), np.uint8)
    for points in (
        [[1.5, 1.5], [4.5, 4.5], [4.5, 7.5], [1.5, 7.5]],
        [
            [1.5, 1.5],
            [7.5, 1.5],
            [7.5, 7.5],
            [4.5, 7.5],
            [4.5, -1.5],
            [1.5, -1.5],
        ],
    ):
        assert not validate_polygon_edge_fit(image, points).succeeded


def test_small_split_is_bridged_only_within_configured_change_budget():
    image = np.zeros((60, 60), np.uint8)
    image[10:50, 10:50] = 220
    image[10:50, 29:31] = 0
    original = np.array([[9.5, 9.5], [49.5, 9.5], [49.5, 49.5], [9.5, 49.5]])
    result = refine_polygon_to_edge(
        image,
        original,
        {
            "search_radius": 4,
            "max_area_change": 0.2,
            "gap_repair": True,
            "gap_bridge_max": 3,
            "gap_bridge_ratio": 0.08,
        },
    )
    assert result.succeeded, result.reason
    assert result.review_required
    assert "1px" in result.reason
    assert_grid(result.points)


def test_box_prefers_one_repaired_loop_over_two_unrepaired_halves():
    image = np.zeros((60, 60), np.uint8)
    image[10:50, 10:50] = 220
    image[10:50, 29:31] = 0
    result = segment_box_to_edge_polygon(
        image,
        [[8, 8], [51, 51]],
        {
            "gap_repair": True,
            "gap_bridge_max": 3,
            "gap_bridge_ratio": 0.08,
        },
    )
    assert result.succeeded, result.reason
    assert result.review_required
    assert "1px" in result.reason
    assert len(result.regions) == 1
    assert abs(Polygon(result.points).area - 1600) < 1e-6
    assert_grid(result.points)


def test_box_returns_each_spatial_object_without_background_or_duplicates():
    image = np.zeros((80, 120), np.uint8)
    image[10:30, 10:35] = 220
    image[40:70, 70:105] = 180
    result = segment_box_to_edge_polygon(
        image, [[5, 5], [110, 75]], {"boundary_side": "bright"}
    )
    assert result.succeeded, result.reason
    assert len(result.regions) == 2
    areas = sorted(
        abs(
            np.dot(points[:, 0], np.roll(points[:, 1], -1))
            - np.dot(points[:, 1], np.roll(points[:, 0], -1))
        )
        / 2
        for points in result.regions
    )
    np.testing.assert_allclose(areas, [500, 1050])


def test_large_split_can_be_bridged_for_manual_review():
    image = np.zeros((80, 80), np.uint8)
    image[10:70, 10:70] = 220
    image[10:70, 36:44] = 0
    result = segment_box_to_edge_polygon(
        image,
        [[8, 8], [72, 72]],
        {
            "gap_repair": True,
            "gap_bridge_max": 12,
            "gap_bridge_ratio": 0.25,
        },
    )
    assert result.succeeded, result.reason
    assert result.review_required
    assert "4px" in result.reason
    assert abs(Polygon(result.points).area - 3600) < 1e-6
    assert_grid(result.points)


def test_expanded_search_never_jumps_to_nearby_object_outside_box():
    image = np.zeros((70, 100), np.uint8)
    image[15:55, 10:39] = 220
    image[15:55, 40:82] = 220
    result = segment_box_to_edge_polygon(
        image,
        [[8, 12], [39, 58]],
        {
            "search_radius": 10,
            "gap_repair": True,
            "gap_bridge_max": 12,
            "gap_bridge_ratio": 0.5,
        },
    )
    assert result.succeeded, result.reason
    assert result.points[:, 0].max() <= 39.5
    assert Polygon(result.points).area < 1500


def test_annotate_all_keeps_close_objects_separate():
    image = np.zeros((70, 100), np.uint8)
    image[15:55, 10:39] = 220
    image[15:55, 41:82] = 220
    result = segment_box_to_edge_polygon(
        image,
        [[8, 12], [84, 58]],
        {
            "annotate_all_in_box": True,
            "gap_repair": True,
            "gap_bridge_max": 12,
            "gap_bridge_ratio": 0.5,
        },
    )
    assert result.succeeded, result.reason
    assert len(result.regions) == 2
    bounds = sorted((r[:, 0].min(), r[:, 0].max()) for r in result.regions)
    assert bounds[0][1] < bounds[1][0]


def test_large_gap_search_uses_bounded_number_of_attempts():
    radii = _repair_radii(100)
    assert radii[-1] == 100
    assert len(radii) <= 14


def test_rectangle_four_sides_snap_independently_to_cell_edges():
    image = np.zeros((100, 160), np.uint8)
    image[20:80, 25:140] = 200
    result = fit_rectangle_to_edges(
        image,
        [[20, 15], [145, 85]],
        {"search_radius": 8, "min_contrast": 8},
    )
    assert result.succeeded, result.reason
    np.testing.assert_allclose(
        result.points,
        [[24.5, 19.5], [139.5, 19.5], [139.5, 79.5], [24.5, 79.5]],
    )
    assert_grid(result.points)
