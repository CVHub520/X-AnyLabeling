"""Geometry shared by the canvas preview and segmentation exporters."""

import math
from numbers import Real
from typing import Any, Mapping, Sequence

Point = tuple[float, float]
Frame = tuple[float, float, float, float, float]
DEFAULT_POLYGON_POINTS = 128


def validate_polygon_points(count: int) -> int:
    """Validate an ellipse approximation size without coercion.

    Args:
        count: Requested number of sampled vertices.

    Returns:
        The validated integer.

    Raises:
        ValueError: If count is not an integer between 12 and 512.
    """
    if type(count) is not int or not 12 <= count <= 512:
        raise ValueError(
            "Ellipse polygon points must be an integer from 12 to 512"
        )
    return count


def _finite_number(value: Any) -> bool:
    """Return whether a coordinate is a finite real number, excluding bools.

    Args:
        value: A JSON coordinate or numeric parameter.

    Returns:
        Whether the value can safely participate in ellipse geometry.
    """
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def ellipse_frame(points: Sequence[Sequence[float]]) -> Frame:
    """Validate an ordered rectangular frame and derive ellipse parameters.

    Args:
        points: Local top-left, top-right, bottom-right, bottom-left corners.

    Returns:
        Center x/y, full width/height, and clockwise radians in [0, 2*pi).

    Raises:
        ValueError: If corners are nonfinite, unordered, nonrectangular or small.
    """
    if len(points) != 4:
        raise ValueError("Ellipse requires four control points")
    if any(
        len(p) != 2 or not all(_finite_number(v) for v in p) for p in points
    ):
        raise ValueError("Ellipse requires finite numeric coordinate pairs")
    p = [(float(x), float(y)) for x, y in points]
    width = math.dist(p[0], p[1])
    height = math.dist(p[0], p[3])
    if width < 1 or height < 1:
        raise ValueError("Ellipse axes are too short")
    ux, uy = p[1][0] - p[0][0], p[1][1] - p[0][1]
    vx, vy = p[3][0] - p[0][0], p[3][1] - p[0][1]
    tolerance = 1e-6 + 1e-6 * max(width, height)
    if ux * vy - uy * vx <= 0:
        raise ValueError(
            "Ellipse corners must follow positive local frame order"
        )
    if abs((ux * vx + uy * vy) / (width * height)) > 1e-6:
        raise ValueError("Ellipse control frame axes are not orthogonal")
    expected = (p[1][0] + p[3][0] - p[0][0], p[1][1] + p[3][1] - p[0][1])
    if math.dist(expected, p[2]) > tolerance:
        raise ValueError(
            "Ellipse opposite edges or diagonal midpoints disagree"
        )
    cx = (p[0][0] + p[2][0]) / 2
    cy = (p[0][1] + p[2][1]) / 2
    angle = math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0])
    return cx, cy, width, height, angle % math.tau


def ellipse_direction(
    shape: Mapping[str, Any], source: str = "ellipse"
) -> float:
    """Validate stored direction and derive its authoritative frame angle.

    Args:
        shape: A native ellipse dictionary, optionally containing direction.
        source: File and object context for errors and mismatch warnings.

    Returns:
        The width-axis angle derived from the four corners.

    Raises:
        ValueError: If the frame or an explicitly stored direction is invalid.
    """
    angle = ellipse_frame(shape["points"])[4]
    if "direction" in shape:
        direction = shape["direction"]
        if not _finite_number(direction):
            raise ValueError(
                f"{source}: ellipse direction must be a finite number"
            )
        difference = math.remainder(float(direction) - angle, math.tau)
        if abs(difference) > 1e-6:
            from anylabeling.views.labeling.logger import logger

            logger.warning(
                f"{source}: ellipse direction disagrees with corners; using {angle}"
            )
    return angle


def ellipse_frame_points(
    cx: float, cy: float, width: float, height: float, angle: float
) -> list[Point]:
    """Build ordered frame corners.

    Args:
        cx: Center x in image coordinates.
        cy: Center y in image coordinates.
        width: Full width-axis length, at least one pixel.
        height: Full height-axis length, at least one pixel.
        angle: Clockwise angle in radians.

    Returns:
        Four local top-left, top-right, bottom-right, bottom-left corners.

    Raises:
        ValueError: If parameters are nonfinite or axes are too small.
    """
    values = (cx, cy, width, height, angle)
    if not all(_finite_number(value) for value in values):
        raise ValueError("Ellipse has non-finite parameters")
    if width < 1 or height < 1:
        raise ValueError("Ellipse axes must be at least one image pixel")
    ux, uy = math.cos(angle) * width / 2, math.sin(angle) * width / 2
    vx, vy = -math.sin(angle) * height / 2, math.cos(angle) * height / 2
    return [
        (cx - ux - vx, cy - uy - vy),
        (cx + ux - vx, cy + uy - vy),
        (cx + ux + vx, cy + uy + vy),
        (cx - ux + vx, cy - uy + vy),
    ]


def ellipse_handle_positions(points: Sequence[Sequence[float]]) -> list[Point]:
    """Get resize handles in local order.

    Args:
        points: Four ordered control corners.

    Returns:
        Corners followed by top, right, bottom and left midpoints.
    """
    if len(points) != 4:
        return []
    p = [(float(x), float(y)) for x, y in points]
    return p + [
        ((p[i][0] + p[(i + 1) % 4][0]) / 2, (p[i][1] + p[(i + 1) % 4][1]) / 2)
        for i in range(4)
    ]


def resize_ellipse(
    points: Sequence[Sequence[float]],
    handle: int,
    target: Sequence[float],
    keep_ratio: bool = False,
) -> list[Point]:
    """Resize on local axes, retaining the opposite handle as anchor.

    Args:
        points: Four original control corners.
        handle: Corner index 0-3 or edge midpoint index 4-7.
        target: Desired handle position in image coordinates.
        keep_ratio: Preserve the original ratio when resizing a corner.

    Returns:
        Resized control corners with axes of at least one pixel.

    Raises:
        ValueError: If the frame, handle or target is invalid.
    """
    cx, cy, width, height, angle = ellipse_frame(points)
    if handle not in range(8):
        raise ValueError("Invalid ellipse handle")
    tx, ty = float(target[0]), float(target[1])
    if not math.isfinite(tx) or not math.isfinite(ty):
        raise ValueError("Invalid ellipse resize target")
    ux, uy = math.cos(angle), math.sin(angle)
    vx, vy = -uy, ux
    handles = ellipse_handle_positions(points)
    opposite = (handle + 2) % 4 if handle < 4 else 4 + (handle - 4 + 2) % 4
    anchor_x, anchor_y = handles[opposite]
    local_x = (tx - anchor_x) * ux + (ty - anchor_y) * uy
    local_y = (tx - anchor_x) * vx + (ty - anchor_y) * vy
    sign_x = 1 if handle in (1, 2, 5) else -1
    sign_y = 1 if handle in (2, 3, 6) else -1
    new_width = (
        max(1.0, sign_x * local_x) if handle < 4 or handle in (5, 7) else width
    )
    new_height = (
        max(1.0, sign_y * local_y)
        if handle < 4 or handle in (4, 6)
        else height
    )
    if keep_ratio and handle < 4:
        ratio = height / width
        new_width = max(
            1.0,
            1.0 / ratio,
            (new_width + ratio * new_height) / (1 + ratio * ratio),
        )
        new_height = ratio * new_width
    if handle < 4:
        cx = (
            anchor_x
            + sign_x * ux * new_width / 2
            + sign_y * vx * new_height / 2
        )
        cy = (
            anchor_y
            + sign_x * uy * new_width / 2
            + sign_y * vy * new_height / 2
        )
    elif handle in (5, 7):
        cx = anchor_x + sign_x * ux * new_width / 2
        cy = anchor_y + sign_x * uy * new_width / 2
    else:
        cx = anchor_x + sign_y * vx * new_height / 2
        cy = anchor_y + sign_y * vy * new_height / 2
    return ellipse_frame_points(cx, cy, new_width, new_height, angle)


def ellipse_polygon(
    points: Sequence[Sequence[float]], count: int = DEFAULT_POLYGON_POINTS
) -> list[Point]:
    """Uniformly sample an ellipse without a duplicate closing vertex.

    Args:
        points: Four ordered control corners.
        count: Number of vertices, an integer from 12 to 512.

    Returns:
        Ordered image-space contour vertices.

    Raises:
        ValueError: If the frame or vertex count is invalid.
    """
    validate_polygon_points(count)
    cx, cy, _, _, _ = ellipse_frame(points)
    p = [(float(x), float(y)) for x, y in points]
    if not all(math.isfinite(c) for point in p for c in point):
        raise ValueError("Ellipse has non-finite coordinates")
    cx = (p[0][0] + p[2][0]) / 2
    cy = (p[0][1] + p[2][1]) / 2
    ux = (p[1][0] - p[0][0]) / 2
    uy = (p[1][1] - p[0][1]) / 2
    vx = (p[3][0] - p[0][0]) / 2
    vy = (p[3][1] - p[0][1]) / 2
    if math.hypot(ux, uy) < 0.5 or math.hypot(vx, vy) < 0.5:
        raise ValueError("Ellipse axes are too short")
    if abs(ux * vy - uy * vx) < 0.25:
        raise ValueError("Ellipse axes are degenerate")
    return [
        (
            cx
            + ux * math.cos(2 * math.pi * i / count)
            + vx * math.sin(2 * math.pi * i / count),
            cy
            + uy * math.cos(2 * math.pi * i / count)
            + vy * math.sin(2 * math.pi * i / count),
        )
        for i in range(count)
    ]


def clip_polygon(
    points: Sequence[Point], width: float, height: float
) -> list[Point]:
    """Intersect a convex contour with continuous image bounds.

    Args:
        points: Ordered vertices of the sampled ellipse.
        width: Positive continuous image width.
        height: Positive continuous image height.

    Returns:
        Cleaned clipped vertices, or an empty list for a degenerate result.

    Raises:
        ValueError: If image dimensions are invalid.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size")
    output = list(points)
    for axis, edge, keep_greater in (
        (0, 0.0, True),
        (0, float(width), False),
        (1, 0.0, True),
        (1, float(height), False),
    ):
        source, output = output, []
        if not source:
            break

        def inside(point: Point) -> bool:
            """Return whether a point lies inside this clipping half-plane."""
            return point[axis] >= edge if keep_greater else point[axis] <= edge

        def intersection(a: Point, b: Point) -> Point:
            """Return an edge crossing on this clipping boundary."""
            ratio = (edge - a[axis]) / (b[axis] - a[axis])
            x = a[0] + ratio * (b[0] - a[0])
            y = a[1] + ratio * (b[1] - a[1])
            return (edge, y) if axis == 0 else (x, edge)

        previous = source[-1]
        for current in source:
            a, b = inside(previous), inside(current)
            if a != b:
                output.append(intersection(previous, current))
            if b:
                output.append(current)
            previous = current
    clean = []
    for point in output:
        point = tuple(
            (
                0.0
                if abs(value) < 1e-9
                else float(limit) if abs(value - limit) < 1e-9 else value
            )
            for value, limit in zip(point, (width, height))
        )
        if not clean or math.dist(point, clean[-1]) > 1e-7:
            clean.append(point)
    if len(clean) > 1 and math.dist(clean[0], clean[-1]) <= 1e-7:
        clean.pop()
    # Preserve all N samples when no boundary clipping occurred.
    if any(not (0 <= x <= width and 0 <= y <= height) for x, y in points):
        changed = True
        while changed and len(clean) > 3:
            changed = False
            for i, b in enumerate(clean):
                a, c = clean[i - 1], clean[(i + 1) % len(clean)]
                cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (
                    c[0] - b[0]
                )
                if abs(cross) <= 1e-10 * max(
                    1.0, math.dist(a, b) * math.dist(b, c)
                ):
                    clean.pop(i)
                    changed = True
                    break
    area = (
        abs(
            sum(
                a[0] * b[1] - b[0] * a[1]
                for a, b in zip(clean, clean[1:] + clean[:1])
            )
        )
        / 2
    )
    if len(clean) < 3 or area <= 1e-6:
        return []
    return clean


def segmentation_points(
    shape: Mapping[str, Any],
    width: int,
    height: int,
    ellipse_points: int = DEFAULT_POLYGON_POINTS,
) -> list[Point] | None:
    """Adapt a supported shape while retaining legacy polygon clipping rules.

    Args:
        shape: Native shape dictionary.
        width: Image width.
        height: Image height.
        ellipse_points: Validated ellipse sampling count.

    Returns:
        Export vertices, or None for an unrelated shape type.

    Raises:
        ValueError: If newly supported geometry cannot be exported.
    """
    kind = shape["shape_type"]
    if kind not in {"polygon", "quadrilateral", "ellipse"}:
        return None
    if kind == "ellipse":
        ellipse_direction(shape)
        points = ellipse_polygon(shape["points"], ellipse_points)
        result = clip_polygon(points, width, height)
        if not result:
            raise ValueError(
                "Ellipse is outside image bounds or clips to zero area"
            )
        return result
    points = [(float(x), float(y)) for x, y in shape["points"]]
    if kind == "quadrilateral" and len(points) != 4:
        raise ValueError("Quadrilateral requires four vertices")
    if len(points) < 3 or not all(math.isfinite(c) for p in points for c in p):
        raise ValueError("Invalid polygon vertices")
    return [
        (max(0, min(x, width - 1)), max(0, min(y, height - 1)))
        for x, y in points
    ]
