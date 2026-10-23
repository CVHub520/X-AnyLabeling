"""Native r4 compatibility and independent ellipse precision checks."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from shapely.geometry import Point, Polygon

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.ellipse import (
    clip_polygon,
    ellipse_direction,
    ellipse_frame,
    ellipse_frame_points,
    ellipse_polygon,
    segmentation_points,
    validate_polygon_points,
)


@pytest.mark.parametrize("count", [12, 64, 128, 512])
@pytest.mark.parametrize("ratio", [1, 2, 100, 1000])
@pytest.mark.parametrize(
    "angle", [0, 0.37, math.radians(89.99), math.pi, 6.28]
)
def test_sampling_area_and_distance(
    count: int, ratio: int, angle: float
) -> None:
    """Check affine sampling against analytic area and a distance bound."""
    a, b = float(ratio), 1.0
    points = ellipse_frame_points(0, 0, 2 * a, 2 * b, angle)
    contour = ellipse_polygon(points, count)
    polygon = Polygon(contour)
    expected = a * b * count * math.sin(math.tau / count) / 2
    assert polygon.area == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert polygon.is_valid and len(contour) == count
    for t in np.linspace(0, math.tau, 129):
        x, y = a * math.cos(t), b * math.sin(t)
        p = Point(
            x * math.cos(angle) - y * math.sin(angle),
            x * math.sin(angle) + y * math.cos(angle),
        )
        assert (
            p.distance(polygon) <= a * (1 - math.cos(math.pi / count)) + 1e-8
        )


@pytest.mark.parametrize("count", [12, 64, 128, 512])
@pytest.mark.parametrize("ratio", [1, 2, 100, 1000])
@pytest.mark.parametrize("center", [(0, 0), (0, 5), (5, 5)])
def test_clipped_area_against_independent_integral(
    count: int, ratio: int, center: tuple[int, int]
) -> None:
    """Bound clipped area error by the original inscribed polygon loss."""
    a, b = float(ratio), 1.0
    cx, cy = center
    angle = 0.43
    cosine, sine = math.cos(angle), math.sin(angle)
    aa = cosine**2 / a**2 + sine**2 / b**2
    bb = 2 * cosine * sine * (1 / a**2 - 1 / b**2)
    cc = sine**2 / a**2 + cosine**2 / b**2

    def vertical_length(x: float) -> float:
        """Integrate an analytic ellipse cross section within the image."""
        x -= cx
        discriminant = (bb * x) ** 2 - 4 * cc * (aa * x * x - 1)
        if discriminant <= 0:
            return 0.0
        root = math.sqrt(discriminant)
        low = cy + (-bb * x - root) / (2 * cc)
        high = cy + (-bb * x + root) / (2 * cc)
        return max(0.0, min(10.0, high) - max(0.0, low))

    reference, error = quad(
        vertical_length,
        0,
        10,
        epsabs=1e-8,
        points=np.linspace(0, 10, 101),
        limit=300,
    )
    contour = clip_polygon(
        ellipse_polygon(
            ellipse_frame_points(cx, cy, 2 * a, 2 * b, angle), count
        ),
        10,
        10,
    )
    actual = Polygon(contour).area if contour else 0.0
    loss = (
        math.pi * a * b * (1 - count * math.sin(math.tau / count) / math.tau)
    )
    assert -1e-7 <= reference - actual <= loss + error + 1e-6
    if contour:
        assert Polygon(contour).is_valid
        assert all(0 <= x <= 10 and 0 <= y <= 10 for x, y in contour)


@pytest.mark.parametrize("filename", ["r4.json", "r4_gui_saved.json"])
def test_r4_native_roundtrip(filename: str) -> None:
    """Read the untouched archived r4 sample and retain geometry/metadata."""
    path = Path(__file__).parents[1] / "fixtures/ellipse" / filename
    raw = path.read_bytes()
    data = json.loads(raw)
    original = next(s for s in data["shapes"] if s["shape_type"] == "ellipse")
    loaded = Shape().load_from_dict(copy.deepcopy(original), source=str(path))
    stored = loaded.to_dict()
    assert stored["points"] == [tuple(p) for p in original["points"]]
    assert stored["direction"] == pytest.approx(original["direction"])
    assert stored["attributes"] == original["attributes"]
    assert "direction" not in loaded.other_data
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    "direction", [True, None, "0", [], {}, float("nan"), float("inf")]
)
def test_bad_direction(direction: object) -> None:
    """Reject present invalid direction fields even when corners are valid."""
    with pytest.raises(ValueError, match="direction"):
        ellipse_direction(
            {
                "points": ellipse_frame_points(5, 5, 4, 2, 0),
                "direction": direction,
            },
            "sample.json: object 2",
        )


def test_direction_derived_without_mutation() -> None:
    """Corners win over finite stale direction without changing input data."""
    data = {"points": ellipse_frame_points(5, 5, 4, 2, 0.37), "direction": 0}
    before = copy.deepcopy(data)
    assert ellipse_direction(data) == pytest.approx(0.37)
    assert data == before


@pytest.mark.parametrize("count", [11, 513, 12.0, True, "128", None])
def test_invalid_count(count: object) -> None:
    """Reject invalid configuration rather than coercing it."""
    with pytest.raises(ValueError):
        validate_polygon_points(count)


@pytest.mark.parametrize(
    "points",
    [
        [(0, 0), (4, 0), (5, 2), (0, 2)],
        [(0, 0), (4, 0), (5, 2), (1, 2)],
        [(0, 0), (0, 2), (4, 2), (4, 0)],
        [(0, 0), (0.5, 0), (0.5, 2), (0, 2)],
        [(0, 0), (4, 0), (4, 2), (False, 2)],
        [(0, 0), (4, 0), (4, 2), (float("nan"), 2)],
    ],
)
def test_invalid_frame(points: list) -> None:
    """Reject unordered, skewed, degenerate or nonnumeric control frames."""
    with pytest.raises(ValueError):
        ellipse_frame(points)


def test_continuous_clipping_and_legacy_quad() -> None:
    """Keep continuous ellipse edges distinct from legacy vertex clamping."""
    enclosing = ellipse_frame_points(5, 5, 100, 100, 0.3)
    contour = segmentation_points(
        {"shape_type": "ellipse", "points": enclosing}, 10, 10
    )
    assert set(contour) == {(0, 0), (10, 0), (10, 10), (0, 10)}
    quad_shape = {
        "shape_type": "quadrilateral",
        "points": [(-1, -1), (12, 0), (12, 12), (0, 12)],
    }
    assert segmentation_points(quad_shape, 10, 10) == [
        (0, 0),
        (9, 0),
        (9, 9),
        (0, 9),
    ]
    with pytest.raises(ValueError, match="outside"):
        segmentation_points(
            {
                "shape_type": "ellipse",
                "points": ellipse_frame_points(-100, -100, 4, 2, 0),
            },
            10,
            10,
        )


@pytest.mark.parametrize("ratio", [1, 2, 100, 1000])
@pytest.mark.parametrize("center", [(0, 0), (0, 5), (5, 5)])
def test_nested_clipped_samples_converge(
    ratio: int, center: tuple[int, int]
) -> None:
    """Nested inscribed contours increase toward the clipped analytic area."""
    frame = ellipse_frame_points(*center, 2 * ratio, 2, 0.43)
    previous = Polygon()
    for count in (64, 128, 256, 512):
        contour = clip_polygon(ellipse_polygon(frame, count), 10, 10)
        current = Polygon(contour) if contour else Polygon()
        assert current.area + 1e-9 >= previous.area
        assert previous.difference(current).area <= 1e-9
        previous = current


def test_frame_tolerance_inside_and_outside() -> None:
    """Test positional and orthogonality tolerances on both sides."""
    frame = [(0, 0), (100, 0), (100, 50), (0, 50)]
    tolerance = 1e-6 + 1e-6 * 100
    for factor, valid in ((0.5, True), (2, False)):
        diagonal = list(frame)
        diagonal[2] = (100 + factor * tolerance, 50)
        skew = [(0, 0), (100, 0), (100 + factor * 50e-6, 50), (factor * 50e-6, 50)]
        for candidate in (diagonal, skew):
            if valid:
                ellipse_frame(candidate)
            else:
                with pytest.raises(ValueError):
                    ellipse_frame(candidate)


def test_tangent_ellipse_is_not_an_exportable_polygon() -> None:
    """An ellipse merely touching the image has no exportable area."""
    shape = {"shape_type": "ellipse", "points": ellipse_frame_points(-2, 5, 4, 2, 0)}
    with pytest.raises(ValueError, match="zero area"):
        segmentation_points(shape, 10, 10)
