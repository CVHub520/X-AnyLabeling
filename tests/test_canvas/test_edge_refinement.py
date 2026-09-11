import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtGui

    from anylabeling.views.labeling.utils.edge_refinement import (
        EdgeRefinementOptions,
        qimage_to_rgb,
        refine_model_polygons_to_edges,
        refine_polygon_to_edge,
        segment_box_to_edge_polygon,
        validate_polygon_edge_fit,
    )

    DEPENDENCIES_AVAILABLE = True
except Exception:
    DEPENDENCIES_AVAILABLE = False


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "OpenCV and PyQt6 are required")
class TestEdgeRefinement(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((80, 100), dtype=np.uint8)
        self.image[20:60, 30:70] = 255
        self.settings = {
            "threshold_mode": "manual",
            "threshold": 128,
            "polarity": "bright",
            "point_spacing": 4.0,
            "search_radius": 4.0,
            "min_contrast": 8.0,
            "max_area_change": 0.9,
        }

    def test_box_segmentation_follows_half_pixel_transition(self):
        result = segment_box_to_edge_polygon(
            self.image, [[20, 10], [80, 70]], self.settings
        )

        self.assertTrue(result.succeeded, result.reason)
        self.assertLess(len(result.points), 100)
        self.assertAlmostEqual(result.points[:, 0].min(), 29.5, places=2)
        self.assertAlmostEqual(result.points[:, 0].max(), 69.5, places=2)
        self.assertAlmostEqual(result.points[:, 1].min(), 19.5, places=2)
        self.assertAlmostEqual(result.points[:, 1].max(), 59.5, places=2)

    def test_box_automatically_calculates_threshold_from_selected_roi(self):
        image = np.full((80, 100), 250, dtype=np.uint8)
        image[10:71, 20:81] = 35
        image[20:60, 30:70] = 191
        settings = dict(self.settings)
        settings.update(
            threshold_mode="auto",
            threshold=250,
            polarity="auto",
        )

        result = segment_box_to_edge_polygon(
            image, [[20, 10], [80, 70]], settings
        )

        self.assertTrue(result.succeeded, result.reason)
        self.assertGreater(result.threshold_used, 0)
        self.assertLess(result.threshold_used, 255)
        self.assertAlmostEqual(result.points[:, 0].min(), 19.5, places=1)

    def test_curved_subpixel_boundary_stays_within_half_pixel(self):
        height, width = 140, 160
        y, x = np.mgrid[:height, :width]
        center_x, center_y, radius = 81.3, 67.7, 34.4
        signed_distance = (
            np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) - radius
        )
        image = np.clip(127.5 - 90.0 * signed_distance, 20, 235).astype(
            np.uint8
        )
        settings = {
            **self.settings,
            "threshold_mode": "auto",
            "point_spacing": 2.0,
            "search_radius": 3.0,
        }

        result = segment_box_to_edge_polygon(
            image, [[35, 20], [130, 118]], settings
        )

        self.assertTrue(result.succeeded, result.reason)
        point_errors = np.abs(
            np.sqrt(
                (result.points[:, 0] - center_x) ** 2
                + (result.points[:, 1] - center_y) ** 2
            )
            - radius
        )
        # Bright-side cell boundaries intentionally differ from the old
        # continuous gradient midpoint. Validate discrete geometry instead.
        np.testing.assert_allclose(
            result.points - 0.5, np.round(result.points - 0.5)
        )
        delta = np.roll(result.points, -1, axis=0) - result.points
        self.assertTrue(np.all((delta[:, 0] == 0) | (delta[:, 1] == 0)))
        self.assertLessEqual(result.fit_error, 0.5)

    def test_fit_tolerance_cannot_be_looser_than_half_pixel(self):
        options = EdgeRefinementOptions.from_mapping(
            {"max_fit_error": 8.0, "min_fit_fraction": 0.5}
        )

        self.assertEqual(options.max_fit_error, 0.5)
        self.assertEqual(options.min_fit_fraction, 1.0)

    def test_manual_preview_validation_requires_every_point_within_half_pixel(
        self,
    ):
        fitted = np.asarray(
            [
                [29.5, 19.5],
                [29.5, 59.5],
                [69.5, 59.5],
                [69.5, 19.5],
            ]
        )

        valid = validate_polygon_edge_fit(self.image, fitted, self.settings)
        edited_off_edge = fitted.copy()
        edited_off_edge[0] = [32.0, 30.0]
        invalid = validate_polygon_edge_fit(
            self.image, edited_off_edge, self.settings
        )

        self.assertTrue(valid.succeeded, valid.reason)
        self.assertLessEqual(valid.fit_error, 0.5)
        self.assertFalse(invalid.succeeded)
        self.assertIsNone(invalid.points)

    def test_auto_threshold_adjustment_is_applied_after_local_estimate(self):
        settings = dict(self.settings)
        settings.update(
            threshold_mode="auto",
            threshold_adjustment=12,
            polarity="auto",
        )

        baseline = segment_box_to_edge_polygon(
            self.image,
            [[20, 10], [80, 70]],
            {**settings, "threshold_adjustment": 0},
        )
        result = segment_box_to_edge_polygon(
            self.image, [[20, 10], [80, 70]], settings
        )

        self.assertTrue(result.succeeded, result.reason)
        self.assertTrue(baseline.succeeded, baseline.reason)
        self.assertEqual(result.threshold_used, baseline.threshold_used + 12)

    def test_existing_polygon_keeps_vertex_count_and_moves_to_edge(self):
        points = np.array(
            [
                [28, 28],
                [50, 18],
                [72, 28],
                [72, 50],
                [62, 62],
                [38, 62],
                [28, 50],
            ],
            dtype=np.float64,
        )

        result = refine_polygon_to_edge(self.image, points, self.settings)

        self.assertTrue(result.succeeded, result.reason)
        self.assertGreater(len(result.points), len(points))
        self.assertEqual(result.moved_points, len(result.points))
        self.assertTrue(np.any(np.abs(result.points % 1.0 - 0.5) < 0.02))

    def test_low_contrast_image_returns_failure_without_candidate(self):
        image = np.full((40, 40), 100, dtype=np.uint8)

        result = refine_polygon_to_edge(
            image,
            [[10, 10], [30, 10], [30, 30], [10, 30]],
            self.settings,
        )

        self.assertFalse(result.succeeded)
        self.assertIsNone(result.points)

    def test_model_polygon_batch_returns_preview_without_mutating_input(self):
        polygon = np.array(
            [[28, 18], [72, 18], [72, 62], [28, 62]], dtype=np.float64
        )
        original = polygon.copy()

        candidates = refine_model_polygons_to_edges(
            self.image, [polygon], self.settings
        )

        np.testing.assert_array_equal(polygon, original)
        self.assertEqual(len(candidates), 1)
        self.assertIsNotNone(candidates[0])
        self.assertGreater(len(candidates[0]), len(polygon))
        self.assertTrue(np.any(np.abs(candidates[0] % 1.0 - 0.5) < 0.02))

    def test_model_postprocess_uses_same_half_pixel_curve_precision(self):
        height, width = 140, 160
        y, x = np.mgrid[:height, :width]
        center_x, center_y, radius = 81.3, 67.7, 34.4
        signed_distance = (
            np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) - radius
        )
        image = np.clip(127.5 - 90.0 * signed_distance, 20, 235).astype(
            np.uint8
        )
        angles = np.linspace(0.0, np.pi * 2.0, 48, endpoint=False)
        model_polygon = np.column_stack(
            [
                center_x + (radius + 1.5) * np.cos(angles),
                center_y + (radius + 1.5) * np.sin(angles),
            ]
        )

        candidates = refine_model_polygons_to_edges(
            image,
            [model_polygon],
            {
                **self.settings,
                "threshold_mode": "auto",
                "point_spacing": 2.0,
                "search_radius": 3.0,
            },
        )

        self.assertIsNotNone(candidates[0])
        errors = np.abs(
            np.sqrt(
                (candidates[0][:, 0] - center_x) ** 2
                + (candidates[0][:, 1] - center_y) ** 2
            )
            - radius
        )
        np.testing.assert_allclose(
            candidates[0] - 0.5, np.round(candidates[0] - 0.5)
        )
        delta = np.roll(candidates[0], -1, axis=0) - candidates[0]
        self.assertTrue(np.all((delta[:, 0] == 0) | (delta[:, 1] == 0)))

    def test_model_polygon_batch_keeps_unreliable_result_as_none(self):
        image = np.full((40, 40), 100, dtype=np.uint8)

        candidates = refine_model_polygons_to_edges(
            image,
            [[[10, 10], [30, 10], [30, 30], [10, 30]]],
            self.settings,
        )

        self.assertEqual(candidates, [None])

    def test_box_without_enclosed_edge_uses_failure_rollback(self):
        image = np.full((40, 40), 100, dtype=np.uint8)

        result = segment_box_to_edge_polygon(
            image, [[5, 5], [35, 35]], self.settings
        )

        self.assertFalse(result.succeeded)
        self.assertIsNone(result.points)
        self.assertTrue(result.reason)

    def test_qimage_conversion_preserves_rgb_channels(self):
        image = QtGui.QImage(2, 1, QtGui.QImage.Format.Format_RGB32)
        image.setPixelColor(0, 0, QtGui.QColor(12, 34, 56))
        image.setPixelColor(1, 0, QtGui.QColor(210, 180, 140))

        array = qimage_to_rgb(image)

        np.testing.assert_array_equal(
            array,
            np.array([[[12, 34, 56], [210, 180, 140]]], dtype=np.uint8),
        )
