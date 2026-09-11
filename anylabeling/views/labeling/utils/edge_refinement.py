"""Pixel/subpixel edge extraction helpers for precision annotations.

The routines in this module are deliberately independent from Canvas state.
They calculate a candidate geometry first and leave committing, undo history,
and user feedback to the caller.  This makes every edge operation safe to
fall back to the original annotation when the image evidence is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import cv2
import numpy as np
from PyQt6 import QtGui


@dataclass(frozen=True)
class EdgeRefinementOptions:
    """Validated settings shared by manual and model edge refinement.

    ``threshold`` is an edge-strength threshold, not a foreground intensity
    split.  ``polarity`` remains accepted only for config compatibility.
    """

    threshold_mode: str = "auto"
    threshold: int = 128
    threshold_adjustment: int = 0
    polarity: str = "auto"
    blur_radius: int = 0
    morph_kernel: int = 0
    point_spacing: float = 2.0
    search_radius: float = 3.0
    min_contrast: float = 8.0
    min_area: float = 4.0
    max_area_change: float = 0.6
    subpixel_iterations: int = 3
    max_fit_error: float = 0.5
    min_fit_fraction: float = 1.0

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, object] | None
    ) -> "EdgeRefinementOptions":
        values = values or {}
        threshold_mode = str(values.get("threshold_mode", "auto")).lower()
        if threshold_mode not in {"auto", "manual"}:
            threshold_mode = "auto"
        polarity = str(values.get("polarity", "auto")).lower()
        if polarity not in {"auto", "bright", "dark"}:
            polarity = "auto"
        return cls(
            threshold_mode=threshold_mode,
            threshold=max(0, min(255, int(values.get("threshold", 128)))),
            threshold_adjustment=max(
                -127,
                min(127, int(values.get("threshold_adjustment", 0))),
            ),
            polarity=polarity,
            blur_radius=max(0, min(15, int(values.get("blur_radius", 0)))),
            morph_kernel=max(0, min(15, int(values.get("morph_kernel", 0)))),
            point_spacing=max(0.25, float(values.get("point_spacing", 2.0))),
            search_radius=max(0.25, float(values.get("search_radius", 3.0))),
            min_contrast=max(0.0, float(values.get("min_contrast", 8.0))),
            min_area=max(1.0, float(values.get("min_area", 4.0))),
            max_area_change=max(
                0.0, min(0.95, float(values.get("max_area_change", 0.6)))
            ),
            subpixel_iterations=max(
                1, min(5, int(values.get("subpixel_iterations", 3)))
            ),
            # Pixel-edge candidates must never be accepted with a configured
            # convergence tolerance looser than half an original-image pixel.
            max_fit_error=max(
                0.05, min(0.5, float(values.get("max_fit_error", 0.5)))
            ),
            # Every vertex must pass validation.  Accepting a fraction smaller
            # than one would leave some points with an unbounded edge error,
            # which contradicts the advertised <= 0.5 original-pixel limit.
            min_fit_fraction=1.0,
        )


@dataclass(frozen=True)
class EdgeRefinementResult:
    """Candidate edge geometry and diagnostic information."""

    points: np.ndarray | None
    reason: str = ""
    moved_points: int = 0
    threshold_used: float | None = None
    polarity_used: str = ""
    fit_error: float | None = None
    fit_fraction: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.points is not None and len(self.points) >= 3


def qimage_to_rgb(image: QtGui.QImage) -> np.ndarray:
    """Copy a QImage into a tightly packed RGB numpy array."""
    if image is None or image.isNull():
        raise ValueError("No image is loaded")
    converted = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    pointer = converted.constBits()
    pointer.setsize(converted.sizeInBytes())
    rows = np.frombuffer(pointer, dtype=np.uint8).reshape(
        converted.height(), converted.bytesPerLine()
    )
    rgba = rows[:, : converted.width() * 4].reshape(
        converted.height(), converted.width(), 4
    )
    return np.ascontiguousarray(rgba[:, :, :3])


def _as_gray(image: np.ndarray | QtGui.QImage) -> np.ndarray:
    if isinstance(image, QtGui.QImage):
        image = qimage_to_rgb(image)
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] >= 3:
        gray = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Expected a grayscale, RGB, or QImage input")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(gray)


def _polygon_area(points: np.ndarray) -> float:
    if points is None or len(points) < 3:
        return 0.0
    contour = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    return abs(float(cv2.contourArea(contour)))


def _resolve_threshold(
    edge_strength: np.ndarray, options: EdgeRefinementOptions
) -> int:
    """Resolve a manual or ROI-local edge-strength threshold."""
    if options.threshold_mode == "manual":
        return options.threshold
    if edge_strength.size == 0:
        return options.threshold
    otsu_threshold, _ = cv2.threshold(
        edge_strength, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    low_values = edge_strength[edge_strength <= otsu_threshold]
    high_values = edge_strength[edge_strength > otsu_threshold]
    if low_values.size and high_values.size:
        threshold = (
            float(low_values.mean()) + float(high_values.mean())
        ) / 2.0
    else:
        threshold = float(otsu_threshold)
    return int(
        np.clip(
            round(threshold) + options.threshold_adjustment,
            0,
            255,
        )
    )


def _edge_strength(gray: np.ndarray) -> np.ndarray:
    """Return a normalized Scharr gradient magnitude for one image ROI."""
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    nonzero = magnitude[magnitude > 1e-6]
    if not nonzero.size:
        return np.zeros(gray.shape, dtype=np.uint8)
    scale = float(np.percentile(nonzero, 99.5))
    if scale <= 1e-6:
        return np.zeros(gray.shape, dtype=np.uint8)
    return np.clip(magnitude * (255.0 / scale), 0, 255).astype(np.uint8)


def _select_component(mask: np.ndarray, min_area: float) -> np.ndarray | None:
    """Select an enclosed component, preferring one under the box centre."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None
    height, width = mask.shape
    center_label = int(labels[height // 2, width // 2])
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if touches_border:
            continue
        candidates.append((label == center_label, int(area), label))
    if not candidates:
        return None
    selected = max(candidates)[2]
    return np.where(labels == selected, 255, 0).astype(np.uint8)


def _resample_closed_contour(points: np.ndarray, spacing: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    if len(points) < 3:
        return points
    closed = np.vstack([points, points[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(segment_lengths.sum())
    if perimeter <= 1e-6:
        return points
    sample_count = max(3, int(np.ceil(perimeter / max(spacing, 0.25))))
    distances = np.linspace(0.0, perimeter, sample_count, endpoint=False)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    segment_indices = np.searchsorted(cumulative, distances, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(points) - 1)
    local = distances - cumulative[segment_indices]
    denom = np.maximum(segment_lengths[segment_indices], 1e-6)
    ratios = (local / denom)[:, None]
    starts = closed[segment_indices]
    ends = closed[segment_indices + 1]
    return starts + ratios * (ends - starts)


def _edge_samples(
    gray: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    # One-eighth-pixel sampling gives the midpoint/quadratic refinement enough
    # local information to converge well inside the 0.5 px acceptance bound.
    sample_offsets = np.arange(
        -radius, radius + 0.0625, 0.125, dtype=np.float32
    )
    sample_x = points[:, 0:1] + normals[:, 0:1] * sample_offsets[None, :]
    sample_y = points[:, 1:2] + normals[:, 1:2] * sample_offsets[None, :]
    values = cv2.remap(
        gray,
        sample_x.astype(np.float32),
        sample_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sample_offsets, values.astype(np.float32)


def _refine_polygon_pass(
    gray: np.ndarray,
    points: np.ndarray,
    options: EdgeRefinementOptions,
    radius: float,
    gradient_fields: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform one normal-direction edge fit pass."""
    previous_points = np.roll(points, 1, axis=0)
    next_points = np.roll(points, -1, axis=0)
    tangents = next_points - previous_points
    lengths = np.linalg.norm(tangents, axis=1)
    valid_tangents = lengths > 1e-6
    normals = np.zeros_like(tangents)
    normals[valid_tangents, 0] = (
        -tangents[valid_tangents, 1] / lengths[valid_tangents]
    )
    normals[valid_tangents, 1] = (
        tangents[valid_tangents, 0] / lengths[valid_tangents]
    )
    if gradient_fields is not None:
        gradient_x, gradient_y = gradient_fields
        map_x = points[:, 0].reshape((-1, 1)).astype(np.float32)
        map_y = points[:, 1].reshape((-1, 1)).astype(np.float32)
        sampled_x = cv2.remap(
            gradient_x,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1)
        sampled_y = cv2.remap(
            gradient_y,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1)
        magnitudes = np.hypot(sampled_x, sampled_y)
        has_image_normal = magnitudes > 1e-6
        normals[has_image_normal, 0] = (
            sampled_x[has_image_normal] / magnitudes[has_image_normal]
        )
        normals[has_image_normal, 1] = (
            sampled_y[has_image_normal] / magnitudes[has_image_normal]
        )
        valid_tangents |= has_image_normal
    offsets, values = _edge_samples(gray, points, normals, radius)
    refined = points.copy()
    fitted = np.zeros(len(points), dtype=bool)
    displacements = np.full(len(points), np.inf, dtype=np.float32)
    centers = (offsets[:-1] + offsets[1:]) / 2.0

    for index in range(len(points)):
        if not valid_tangents[index]:
            continue
        row = values[index]
        differences = np.abs(np.diff(row))
        if not len(differences):
            continue
        strongest = float(differences.max())
        if strongest < options.min_contrast:
            continue
        minimum = max(options.min_contrast, strongest * 0.35)
        candidate_indices = np.flatnonzero(differences >= minimum)
        if not len(candidate_indices):
            continue

        # Prefer the strongest transition while using proximity only as a
        # small tie-breaker.  The previous nearest-only rule could lock onto
        # weak texture or one side of a thick gradient band.
        proximity_penalty = np.abs(centers[candidate_indices]) / max(
            radius, 0.125
        )
        scores = (
            differences[candidate_indices]
            - strongest * 0.08 * proximity_penalty
        )
        candidate_index = int(candidate_indices[np.argmax(scores)])
        displacement = float(centers[candidate_index])

        left_level = float(np.median(row[: candidate_index + 1]))
        right_level = float(np.median(row[candidate_index + 1 :]))
        if abs(right_level - left_level) >= options.min_contrast:
            local_threshold = (left_level + right_level) / 2.0
            crossings = np.flatnonzero(
                (row[:-1] - local_threshold) * (row[1:] - local_threshold) <= 0
            )
            if len(crossings):
                crossing_index = int(
                    crossings[np.argmin(np.abs(crossings - candidate_index))]
                )
                value_a = float(row[crossing_index])
                value_b = float(row[crossing_index + 1])
                if abs(value_b - value_a) > 1e-6:
                    fraction = np.clip(
                        (local_threshold - value_a) / (value_b - value_a),
                        0.0,
                        1.0,
                    )
                    displacement = float(
                        offsets[crossing_index]
                        + fraction
                        * (
                            offsets[crossing_index + 1]
                            - offsets[crossing_index]
                        )
                    )
        elif 0 < candidate_index < len(differences) - 1:
            left = float(differences[candidate_index - 1])
            center = float(differences[candidate_index])
            right = float(differences[candidate_index + 1])
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-6:
                sub_index = np.clip(
                    0.5 * (left - right) / denominator, -0.5, 0.5
                )
                displacement += float(sub_index) * float(
                    offsets[1] - offsets[0]
                )

        if abs(displacement) > radius + 1e-6:
            continue
        candidate = points[index] + normals[index] * displacement
        candidate[0] = np.clip(candidate[0], 0.0, gray.shape[1] - 1.0)
        candidate[1] = np.clip(candidate[1], 0.0, gray.shape[0] - 1.0)
        refined[index] = candidate
        fitted[index] = True
        displacements[index] = float(np.linalg.norm(candidate - points[index]))

    return refined, fitted, displacements


def _refine_gradient_ridge_pass(
    points: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    gradient_magnitude: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move points onto the subpixel maximum of the local gradient ridge."""
    map_x = points[:, 0:1].astype(np.float32)
    map_y = points[:, 1:2].astype(np.float32)
    sampled_x = cv2.remap(
        gradient_x,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1)
    sampled_y = cv2.remap(
        gradient_y,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1)
    magnitudes = np.hypot(sampled_x, sampled_y)
    fitted = magnitudes > 1e-6
    normals = np.zeros_like(points)
    normals[fitted, 0] = sampled_x[fitted] / magnitudes[fitted]
    normals[fitted, 1] = sampled_y[fitted] / magnitudes[fitted]

    offsets = np.arange(-radius, radius + 0.0625, 0.125, dtype=np.float32)
    sample_x = points[:, 0:1] + normals[:, 0:1] * offsets[None, :]
    sample_y = points[:, 1:2] + normals[:, 1:2] * offsets[None, :]
    profiles = cv2.remap(
        gradient_magnitude,
        sample_x.astype(np.float32),
        sample_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    refined = points.copy()
    displacements = np.full(len(points), np.inf, dtype=np.float32)

    for index in np.flatnonzero(fitted):
        profile = profiles[index]
        peak = float(profile.max())
        if peak <= 1e-6:
            fitted[index] = False
            continue
        tied = np.flatnonzero(profile >= peak - 1e-5)
        peak_index = int(tied[np.argmin(np.abs(offsets[tied]))])
        displacement = float(offsets[peak_index])
        if 0 < peak_index < len(offsets) - 1:
            left = float(profile[peak_index - 1])
            center = float(profile[peak_index])
            right = float(profile[peak_index + 1])
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-6:
                sub_index = np.clip(
                    0.5 * (left - right) / denominator, -0.5, 0.5
                )
                displacement += float(sub_index) * float(
                    offsets[1] - offsets[0]
                )
        candidate = points[index] + normals[index] * displacement
        candidate[0] = np.clip(candidate[0], 0.0, gradient_x.shape[1] - 1.0)
        candidate[1] = np.clip(candidate[1], 0.0, gradient_x.shape[0] - 1.0)
        refined[index] = candidate
        displacements[index] = abs(displacement)

    return refined, fitted, displacements


def _edge_centering_errors(
    gray: np.ndarray,
    points: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    min_contrast: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate point-to-local-midpoint error in original-image pixels."""
    map_x = points[:, 0:1].astype(np.float32)
    map_y = points[:, 1:2].astype(np.float32)
    sampled_x = cv2.remap(
        gradient_x,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1)
    sampled_y = cv2.remap(
        gradient_y,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1)
    magnitudes = np.hypot(sampled_x, sampled_y)
    valid = magnitudes > 1e-6
    normals = np.zeros_like(points)
    normals[valid, 0] = sampled_x[valid] / magnitudes[valid]
    normals[valid, 1] = sampled_y[valid] / magnitudes[valid]

    probe_distance = 1.0
    offsets = np.asarray(
        [-probe_distance, 0.0, probe_distance], dtype=np.float32
    )
    sample_x = points[:, 0:1] + normals[:, 0:1] * offsets[None, :]
    sample_y = points[:, 1:2] + normals[:, 1:2] * offsets[None, :]
    samples = cv2.remap(
        gray,
        sample_x.astype(np.float32),
        sample_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)
    contrast = np.abs(samples[:, 2] - samples[:, 0])
    valid &= contrast >= min_contrast
    local_midpoint = (samples[:, 0] + samples[:, 2]) * 0.5
    gradient_per_pixel = contrast / (2.0 * probe_distance)
    errors = np.full(len(points), np.inf, dtype=np.float32)
    errors[valid] = np.abs(
        samples[valid, 1] - local_midpoint[valid]
    ) / np.maximum(gradient_per_pixel[valid], 1e-6)
    return errors, valid


def _legacy_refine_polygon_to_edge(
    image: np.ndarray | QtGui.QImage,
    points: Sequence[Sequence[float]] | np.ndarray,
    settings: Mapping[str, object] | EdgeRefinementOptions | None = None,
) -> EdgeRefinementResult:
    """Move polygon vertices to subpixel edges and enforce a 0.5 px bound.

    Coordinates and all distances are measured in original-image pixels.  The
    normal-direction fit is repeated so curved and oblique contours converge;
    a candidate is rejected when the last pass still moves any fitted point by
    more than ``max_fit_error`` or too few vertices have reliable contrast.
    """
    options = (
        settings
        if isinstance(settings, EdgeRefinementOptions)
        else EdgeRefinementOptions.from_mapping(settings)
    )
    gray = _as_gray(image)
    original = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    if len(original) < 3 or not np.isfinite(original).all():
        return EdgeRefinementResult(
            None, "Polygon has fewer than three valid points"
        )

    refined = original.copy()
    fit_error = float("inf")
    fit_fraction = 0.0
    gradient_fields = (
        cv2.Scharr(gray, cv2.CV_32F, 1, 0),
        cv2.Scharr(gray, cv2.CV_32F, 0, 1),
    )
    for iteration in range(options.subpixel_iterations):
        radius = (
            options.search_radius
            if iteration == 0
            else min(options.search_radius, 1.5)
        )
        refined, fitted, displacements = _refine_polygon_pass(
            gray,
            refined,
            options,
            radius,
            gradient_fields if iteration else None,
        )
        fit_fraction = float(np.mean(fitted))
        if fit_fraction < options.min_fit_fraction:
            return EdgeRefinementResult(
                None,
                "Too few vertices have a reliable nearby edge",
                fit_fraction=fit_fraction,
            )
        fit_error = float(np.max(displacements[fitted]))

    midpoint_refined = refined.copy()

    # The intensity-midpoint passes get close to the transition.  Also build
    # the two-dimensional gradient ridge so corners and oblique boundaries do
    # not retain the normal-direction bias of the initial polygon contour.
    gradient_magnitude = cv2.magnitude(*gradient_fields)
    ridge_radius = min(options.search_radius, 1.0)
    for _iteration in range(3):
        refined, fitted, displacements = _refine_gradient_ridge_pass(
            refined,
            gradient_fields[0],
            gradient_fields[1],
            gradient_magnitude,
            ridge_radius,
        )
        fit_fraction = float(np.mean(fitted))
        if fit_fraction < options.min_fit_fraction:
            return EdgeRefinementResult(
                None,
                "Too few vertices have a reliable gradient ridge",
                fit_fraction=fit_fraction,
            )
        fit_error = float(np.max(displacements[fitted]))

    # A Scharr ridge can be about one eighth pixel off on a perfectly sharp,
    # axis-aligned step, while the intensity midpoint is exact there.  Select
    # the better of the two candidates point by point using a spatial error
    # estimate from the local two-sided intensity profile.
    midpoint_errors, midpoint_valid = _edge_centering_errors(
        gray,
        midpoint_refined,
        gradient_fields[0],
        gradient_fields[1],
        options.min_contrast,
    )
    ridge_errors, ridge_valid = _edge_centering_errors(
        gray,
        refined,
        gradient_fields[0],
        gradient_fields[1],
        options.min_contrast,
    )
    choose_ridge = ridge_valid & (
        ~midpoint_valid | (ridge_errors < midpoint_errors)
    )
    selected = midpoint_refined.copy()
    selected[choose_ridge] = refined[choose_ridge]
    selected_errors = np.where(choose_ridge, ridge_errors, midpoint_errors)
    selected_valid = np.where(choose_ridge, ridge_valid, midpoint_valid)
    fit_fraction = float(np.mean(selected_valid))
    if fit_fraction < options.min_fit_fraction:
        return EdgeRefinementResult(
            None,
            "Too few vertices pass the 0.5 px edge validation",
            fit_fraction=fit_fraction,
        )
    fit_error = float(np.max(selected_errors[selected_valid]))
    refined = selected

    if fit_error > options.max_fit_error + 1e-6:
        return EdgeRefinementResult(
            None,
            "Subpixel fit did not converge within 0.5 px",
            fit_error=fit_error,
            fit_fraction=fit_fraction,
        )

    original_area = _polygon_area(original)
    refined_area = _polygon_area(refined)
    if original_area < options.min_area or refined_area < options.min_area:
        return EdgeRefinementResult(None, "Refined polygon area is too small")
    area_change = abs(refined_area - original_area) / max(original_area, 1e-6)
    if area_change > options.max_area_change:
        return EdgeRefinementResult(
            None, "Refined polygon changed area too much"
        )
    moved = int(
        np.count_nonzero(np.linalg.norm(refined - original, axis=1) >= 0.01)
    )
    return EdgeRefinementResult(
        refined.astype(np.float64),
        moved_points=moved,
        fit_error=fit_error,
        fit_fraction=fit_fraction,
    )


def _legacy_validate_polygon_edge_fit(
    image: np.ndarray | QtGui.QImage,
    points: Sequence[Sequence[float]] | np.ndarray,
    settings: Mapping[str, object] | EdgeRefinementOptions | None = None,
) -> EdgeRefinementResult:
    """Validate edited polygon points against the same strict edge metric.

    This does not move any point.  It is used immediately before committing an
    editable existing-annotation preview, so manual vertex edits cannot bypass
    the half-original-pixel acceptance bound.
    """
    options = (
        settings
        if isinstance(settings, EdgeRefinementOptions)
        else EdgeRefinementOptions.from_mapping(settings)
    )
    gray = _as_gray(image)
    candidate = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    if len(candidate) < 3 or not np.isfinite(candidate).all():
        return EdgeRefinementResult(
            None, "Polygon has fewer than three valid points"
        )
    gradient_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    errors, valid = _edge_centering_errors(
        gray,
        candidate,
        gradient_x,
        gradient_y,
        options.min_contrast,
    )
    fit_fraction = float(np.mean(valid))
    if fit_fraction < 1.0:
        return EdgeRefinementResult(
            None,
            "Every vertex must have a reliable nearby edge",
            fit_fraction=fit_fraction,
        )
    fit_error = float(np.max(errors))
    if fit_error > options.max_fit_error + 1e-6:
        return EdgeRefinementResult(
            None,
            "One or more vertices are farther than 0.5 px from the edge",
            fit_error=fit_error,
            fit_fraction=fit_fraction,
        )
    return EdgeRefinementResult(
        candidate.astype(np.float64),
        fit_error=fit_error,
        fit_fraction=fit_fraction,
    )


def _legacy_segment_box_to_edge_polygon(
    image: np.ndarray | QtGui.QImage,
    box: Sequence[Sequence[float]] | np.ndarray,
    settings: Mapping[str, object] | EdgeRefinementOptions | None = None,
) -> EdgeRefinementResult:
    """Threshold a box-local component and return a sampled subpixel contour."""
    options = (
        settings
        if isinstance(settings, EdgeRefinementOptions)
        else EdgeRefinementOptions.from_mapping(settings)
    )
    gray = _as_gray(image)
    box_points = np.asarray(box, dtype=np.float32).reshape((-1, 2))
    if len(box_points) < 2 or not np.isfinite(box_points).all():
        return EdgeRefinementResult(None, "Invalid rectangle")
    x0 = max(0, int(np.floor(box_points[:, 0].min())))
    y0 = max(0, int(np.floor(box_points[:, 1].min())))
    x1 = min(gray.shape[1], int(np.ceil(box_points[:, 0].max())) + 1)
    y1 = min(gray.shape[0], int(np.ceil(box_points[:, 1].max())) + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return EdgeRefinementResult(None, "Rectangle is too small")
    roi = gray[y0:y1, x0:x1]
    working = roi
    if options.blur_radius:
        size = options.blur_radius * 2 + 1
        working = cv2.GaussianBlur(working, (size, size), 0)
    strength = _edge_strength(working)
    threshold = _resolve_threshold(strength, options)
    mask = np.where(strength >= threshold, 255, 0).astype(np.uint8)
    close_radius = max(1, options.morph_kernel)
    size = close_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return EdgeRefinementResult(
            None,
            "No continuous edge found in rectangle",
            threshold_used=float(threshold),
        )
    candidates = []
    height, width = working.shape
    for contour in contours:
        area = abs(float(cv2.contourArea(contour)))
        if area < options.min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        contour_points = contour.reshape((-1, 2))
        mean_strength = float(
            strength[contour_points[:, 1], contour_points[:, 0]].mean()
        )
        center = np.asarray([width / 2.0, height / 2.0])
        contour_center = contour_points.mean(axis=0)
        center_distance = float(np.linalg.norm(contour_center - center))
        score = area + mean_strength * perimeter * 0.05 - center_distance
        candidates.append((score, area, contour))
    if not candidates:
        return EdgeRefinementResult(
            None,
            "No enclosed edge contour found in rectangle",
            threshold_used=float(threshold),
        )
    _, _, contour = max(candidates, key=lambda item: (item[0], item[1]))
    local_points = contour.reshape((-1, 2)).astype(np.float32)
    global_points = local_points + np.array([x0, y0], dtype=np.float32)
    sampled = _resample_closed_contour(global_points, options.point_spacing)
    if len(sampled) < 3:
        return EdgeRefinementResult(
            None, "Contour sampling produced too few points"
        )
    refine_options = replace(
        options,
        threshold_mode="manual",
        threshold=threshold,
    )
    refined = refine_polygon_to_edge(gray, sampled, refine_options)
    if refined.succeeded:
        return replace(
            refined,
            threshold_used=float(threshold),
        )
    return EdgeRefinementResult(
        None,
        refined.reason,
        threshold_used=float(threshold),
    )


def _legacy_refine_model_polygons_to_edges(
    image: np.ndarray | QtGui.QImage,
    polygons: Sequence[Sequence[Sequence[float]] | np.ndarray],
    settings: Mapping[str, object] | EdgeRefinementOptions | None = None,
) -> list[np.ndarray | None]:
    """Build edge candidates for model polygons without mutating inputs.

    The image is converted to grayscale once, then each polygon is processed
    in a small local crop.  This function is suitable for a worker thread and
    returns ``None`` for any polygon whose nearby edge is not trustworthy.
    """
    options = (
        settings
        if isinstance(settings, EdgeRefinementOptions)
        else EdgeRefinementOptions.from_mapping(settings)
    )
    gray = _as_gray(image)
    image_height, image_width = gray.shape
    margin = max(1.0, options.search_radius) + 2.0
    candidates: list[np.ndarray | None] = []

    for polygon in polygons:
        original = np.asarray(polygon, dtype=np.float64).reshape((-1, 2))
        if len(original) < 3 or not np.isfinite(original).all():
            candidates.append(None)
            continue

        x0 = max(0, int(np.floor(original[:, 0].min() - margin)))
        y0 = max(0, int(np.floor(original[:, 1].min() - margin)))
        x1 = min(
            image_width,
            int(np.ceil(original[:, 0].max() + margin)) + 1,
        )
        y1 = min(
            image_height,
            int(np.ceil(original[:, 1].max() + margin)) + 1,
        )
        if x1 - x0 < 3 or y1 - y0 < 3:
            candidates.append(None)
            continue

        offset = np.asarray([x0, y0], dtype=np.float64)
        local_original = original - offset
        roi = gray[y0:y1, x0:x1]
        local_box = np.asarray(
            [[0.0, 0.0], [float(roi.shape[1] - 1), float(roi.shape[0] - 1)]],
            dtype=np.float64,
        )

        segmented = segment_box_to_edge_polygon(roi, local_box, options)
        candidate = None
        if segmented.succeeded:
            proposed = np.asarray(segmented.points, dtype=np.float64) + offset
            original_area = _polygon_area(original)
            proposed_area = _polygon_area(proposed)
            area_change = abs(proposed_area - original_area) / max(
                original_area, 1e-6
            )
            original_contour = original.astype(np.float32).reshape((-1, 1, 2))
            distances = [
                abs(
                    cv2.pointPolygonTest(
                        original_contour,
                        (float(point[0]), float(point[1])),
                        True,
                    )
                )
                for point in proposed
            ]
            if (
                area_change <= options.max_area_change
                and distances
                and float(np.median(distances)) <= margin * 1.5
            ):
                candidate = proposed

        if candidate is None:
            refined = refine_polygon_to_edge(roi, local_original, options)
            if refined.succeeded:
                candidate = (
                    np.asarray(refined.points, dtype=np.float64) + offset
                )

        candidates.append(candidate)

    return candidates


# Public pixel-edge operations use discrete cell boundaries. Keep the previous
# continuous fitter available internally for regression comparisons only.
from .pixel_cell_edges import (  # noqa: E402,F401
    fit_rectangle_to_edges,
    refine_model_polygons_to_edges,
    refine_polygons_to_edges,
    refine_polygon_to_edge,
    segment_box_to_edge_polygon,
    validate_polygon_edge_fit,
)
