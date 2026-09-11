"""Local intensity segmentation with exact, orthogonal pixel-cell boundaries."""

from dataclasses import dataclass, field

import cv2
import numpy as np
from shapely.geometry import Polygon


@dataclass
class CellEdgeResult:
    points: object = None
    reason: str = ""
    moved_points: int = 0
    threshold_used: object = None
    fit_error: object = None
    fit_fraction: float = 0.0
    score: float = float("-inf")
    candidates: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    review_required: bool = False

    @property
    def succeeded(self):
        return self.points is not None and len(self.points) >= 4


def _gray(image):
    from .edge_refinement import _as_gray

    return _as_gray(image)


def _settings(settings):
    if settings is None:
        return {}
    return dict(settings) if isinstance(settings, dict) else vars(settings)


def _repair_radii(max_gap):
    """Bound repair work while still offering useful large-gap distances."""
    max_gap = max(0, min(100, int(max_gap)))
    if not max_gap:
        return []
    radii = list(range(1, min(4, max_gap) + 1))
    radii.extend((6, 8, 12, 16, 24, 32, 48, 64, 100, max_gap))
    return sorted({value for value in radii if value <= max_gap})


def _levels(gray):
    """Deterministic weighted one-dimensional clustering of raw intensities."""
    values, counts = np.unique(gray, return_counts=True)
    if len(values) < 2:
        return []
    centers = np.linspace(
        float(values[0]), float(values[-1]), min(3, len(values))
    )
    for _ in range(24):
        labels = np.abs(values[:, None] - centers).argmin(axis=1)
        updated = np.array(
            [
                (
                    np.average(
                        values[labels == i], weights=counts[labels == i]
                    )
                    if np.any(labels == i)
                    else centers[i]
                )
                for i in range(len(centers))
            ]
        )
        if np.max(np.abs(updated - centers)) < 0.01:
            break
        centers = updated
    return sorted(set(float(v) for v in updated))


def _trace_cells(mask, spacing):
    """Trace directed exposed unit edges; never interpolate across a corner."""
    padded = np.pad(mask.astype(bool), 1)
    center = padded[1:-1, 1:-1]
    edges = {}
    for exposed, start, end in (
        (center & ~padded[:-2, 1:-1], (0, 0), (1, 0)),
        (center & ~padded[1:-1, 2:], (1, 0), (1, 1)),
        (center & ~padded[2:, 1:-1], (1, 1), (0, 1)),
        (center & ~padded[1:-1, :-2], (0, 1), (0, 0)),
    ):
        ys, xs = np.nonzero(exposed)
        for x, y in zip(xs.tolist(), ys.tolist()):
            a, b = (x + start[0], y + start[1]), (x + end[0], y + end[1])
            edges.setdefault(a, []).append(b)
    loops = []
    while edges:
        first = next(iter(edges))
        a = first
        points = []
        previous = None
        while a in edges:
            points.append(a)
            choices = edges[a]
            if previous is None or len(choices) == 1:
                b = choices[0]
            else:
                dx, dy = a[0] - previous[0], a[1] - previous[1]
                # Right turn keeps corner-touching components separate.
                b = max(
                    choices,
                    key=lambda q: dx * (q[1] - a[1]) - dy * (q[0] - a[0]),
                )
            choices.remove(b)
            if not choices:
                del edges[a]
            previous, a = a, b
            if a == first:
                break
        if a != first or len(points) < 4:
            continue
        pts = np.asarray(points, dtype=np.float64) - 0.5
        incoming = pts - np.roll(pts, 1, axis=0)
        outgoing = np.roll(pts, -1, axis=0) - pts
        corners = np.flatnonzero(np.any(incoming != outgoing, axis=1))
        if len(corners) < 4:
            continue
        result = []
        stride = max(1, int(np.floor(spacing)))
        for index, corner in enumerate(corners):
            stop = int(corners[(index + 1) % len(corners)])
            length = (stop - corner) % len(pts)
            result.extend(
                pts[(corner + step) % len(pts)]
                for step in range(0, length, stride)
            )
        contour = np.asarray(result)
        polygon = Polygon(contour)
        if polygon.is_valid and polygon.area > 0:
            loops.append(contour)
    return loops


def _search(image, box, settings, original=None):
    gray = _gray(image)
    options = _settings(settings)
    box = np.asarray(box, dtype=float).reshape(-1, 2)
    if len(box) < 2 or not np.isfinite(box).all():
        return CellEdgeResult(reason="框选坐标无效，原数据保持不变")
    radius = max(1.0, float(options.get("search_radius", 3)))
    limit = (
        min(100.0, radius * 2)
        if options.get("adaptive_search", True)
        else radius
    )
    margin = int(np.ceil(limit)) + 3
    lo = np.maximum(0, np.floor(box.min(axis=0)).astype(int) - margin)
    hi = np.minimum(
        gray.shape[::-1], np.ceil(box.max(axis=0)).astype(int) + margin + 1
    )
    x0, y0 = lo
    x1, y1 = hi
    roi = gray[y0:y1, x0:x1]
    if min(roi.shape, default=0) < 2 or np.ptp(roi) == 0:
        return CellEdgeResult(
            reason="当前区域没有可区分的灰度边界，框选和原数据已保留"
        )
    # Estimate levels inside the requested region; the expanded crop supplies
    # boundary context without letting unrelated bright objects set thresholds.
    a = np.maximum(0, np.floor(box.min(axis=0) - lo).astype(int))
    b = np.minimum(
        roi.shape[::-1], np.ceil(box.max(axis=0) - lo).astype(int) + 1
    )
    focus = roi[a[1] : b[1], a[0] : b[0]]
    blur = max(0, min(15, int(options.get("blur_radius", 0))))
    if blur and focus.size:
        focus = cv2.GaussianBlur(focus, (blur * 2 + 1, blur * 2 + 1), 0)
    levels = _levels(focus)
    if (
        len(levels) < 2
        or np.ptp(focus) < np.ptp(roi) * 0.5
        or int(roi.max()) - int(focus.max()) > 8
        or int(focus.min()) - int(roi.min()) > 8
    ):
        levels = _levels(roi)
    side = options.get("boundary_side", "bright")
    thresholds = [(a + b) / 2 for a, b in zip(levels, levels[1:])]
    if side != "dark":
        thresholds.reverse()
    mode = options.get(
        "level_threshold_mode", options.get("threshold_mode", "auto")
    )
    adjustment = options.get(
        "level_threshold_adjustment", options.get("threshold_adjustment", 0)
    )
    if mode == "manual":
        thresholds = [
            float(
                options.get("level_threshold", options.get("threshold", 128))
            )
        ]
    else:
        thresholds = [
            float(np.clip(t + adjustment, 0, 254)) for t in thresholds
        ]
        if side != "auto":
            # Never silently substitute the opposite boundary for a white/gray request.
            thresholds = thresholds[:1]
        thresholds = list(
            dict.fromkeys(
                t
                for base in thresholds
                for t in (base, max(0, base - 4), min(254, base + 4))
            )
        )
    guide = Polygon(original) if original is not None else None
    if guide is not None and (not guide.is_valid or guide.area <= 0):
        return CellEdgeResult(reason="原轮廓存在自相交，请先人工修正")
    center = box.mean(axis=0)
    candidates = []
    seen = set()
    # The requested *side* chooses the local intensity split, while either
    # side of that split may be the bounded object (for example a gray object
    # inside a white background). Crop-touch rejection removes the unbounded
    # complement without silently switching to another gray-level boundary.
    polarities = (True, False)
    repair_radius = int(options.get("_repair_radius", 0))
    for threshold in thresholds:
        for bright in polarities:
            mask = (roi > threshold) if bright else (roi <= threshold)
            if repair_radius:
                original_mask = mask
                mask = cv2.morphologyEx(
                    mask.astype(np.uint8),
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(
                        cv2.MORPH_RECT,
                        (repair_radius * 2 + 1, repair_radius * 2 + 1),
                    ),
                ).astype(bool)
                changed = np.count_nonzero(mask != original_mask)
                change_limit = float(options.get("gap_bridge_ratio", 0.05))
                if (
                    changed / max(1, np.count_nonzero(original_mask))
                    > change_limit
                ):
                    continue
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=4
            )
            ranked = sorted(
                range(1, count),
                key=lambda i: stats[i, cv2.CC_STAT_AREA],
                reverse=True,
            )
            for label in ranked[:64]:
                x, y, w, h, area = stats[label]
                if area < float(options.get("min_area", 1)):
                    continue
                touch_count = sum(
                    (
                        x == 0,
                        y == 0,
                        x + w >= roi.shape[1],
                        y + h >= roi.shape[0],
                    )
                )
                touches_crop = (
                    (x == 0 and x0 > 0)
                    or (y == 0 and y0 > 0)
                    or (x + w >= roi.shape[1] and x1 < gray.shape[1])
                    or (y + h >= roi.shape[0] and y1 < gray.shape[0])
                )
                if (
                    touches_crop
                    or area > roi.size * 0.9
                    or (guide is None and touch_count >= 3)
                ):
                    continue
                component = labels[y : y + h, x : x + w] == label
                for local in _trace_cells(
                    component, float(options.get("point_spacing", 2))
                ):
                    points = local + lo + [x, y]
                    polygon = Polygon(points)
                    key = polygon.wkb
                    if key in seen:
                        continue
                    seen.add(key)
                    if guide is not None:
                        distance = polygon.boundary.hausdorff_distance(
                            guide.boundary
                        )
                        change = abs(polygon.area - guide.area) / guide.area
                        if distance > limit or change > float(
                            options.get("max_area_change", 0.6)
                        ):
                            continue
                        overlap = (
                            polygon.intersection(guide).area
                            / polygon.union(guide).area
                        )
                        score = overlap * 100 - distance - change * 10
                    else:
                        from shapely.geometry import box as rect

                        selection = rect(*box.min(axis=0), *box.max(axis=0))
                        intersection_area = polygon.intersection(
                            selection
                        ).area
                        containment = intersection_area / max(polygon.area, 1)
                        selection_coverage = intersection_area / max(
                            selection.area, 1
                        )
                        centroid = np.asarray(polygon.centroid.coords[0])
                        # A nearby component may be present inside the expanded
                        # search crop, but it must never become the result when
                        # its center lies outside the rectangle or most of its
                        # area crosses the requested boundary.
                        if (
                            np.any(centroid < box.min(axis=0))
                            or np.any(centroid > box.max(axis=0))
                            or containment
                            < float(options.get("min_box_containment", 0.7))
                        ):
                            continue
                        score = (
                            containment * 30
                            + selection_coverage * 10
                            + np.log1p(polygon.area)
                            - np.linalg.norm(centroid - center) * 0.1
                        )
                    candidates.append((score, points, threshold))
    if not candidates:
        max_gap = max(0, min(100, int(options.get("gap_bridge_max", 24))))
        if (
            options.get("gap_repair", True)
            and not options.get("_repair_search_complete", False)
            and repair_radius < max_gap
        ):
            repairs = []
            for radius_value in _repair_radii(max_gap):
                if radius_value <= repair_radius:
                    continue
                repaired = _search(
                    image,
                    box,
                    {
                        **options,
                        "_repair_radius": radius_value,
                        "_repair_search_complete": True,
                    },
                    original,
                )
                if repaired.succeeded:
                    repairs.append(repaired)
            if repairs:
                return max(repairs, key=lambda item: item.score)
        return CellEdgeResult(
            reason="附近未找到匹配的闭合像素边界，可调整阈值或搜索半径；原数据已保留"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    if (
        options.get("gap_repair", True)
        and repair_radius == 0
        and not options.get("_repair_search_complete", False)
        and not options.get("annotate_all_in_box", False)
    ):
        repaired_results = []
        for radius_value in _repair_radii(options.get("gap_bridge_max", 24)):
            repaired = _search(
                image,
                box,
                {
                    **options,
                    "_repair_radius": radius_value,
                    "_repair_search_complete": True,
                },
                original,
            )
            if repaired.succeeded:
                repaired_results.append(repaired)
        if repaired_results:
            repaired = max(repaired_results, key=lambda item: item.score)
            if repaired.score > best[0] + 0.2:
                # Rebuild arrays after byte-key de-duplication while preserving
                # the repaired candidate first for preview/confirmation.
                unique = []
                seen_points = set()
                for points in [repaired.points] + [
                    candidate[1] for candidate in candidates[:8]
                ]:
                    key = np.asarray(points).tobytes()
                    if key not in seen_points:
                        unique.append(np.asarray(points))
                        seen_points.add(key)
                repaired.candidates = unique
                return repaired
    # Keep one best contour for each spatial object. Threshold perturbations
    # often generate nearly identical loops, which must not become duplicate
    # annotations when the user requests every object in the selected region.
    regions = []
    region_polygons = []
    max_objects = max(1, min(500, int(options.get("max_box_objects", 100))))
    for _score, points, _threshold in candidates:
        polygon = Polygon(points)
        if guide is None:
            centroid = np.asarray(polygon.centroid.coords[0])
            if np.any(centroid < box.min(axis=0)) or np.any(
                centroid > box.max(axis=0)
            ):
                continue
        duplicate = False
        for accepted in region_polygons:
            union = polygon.union(accepted).area
            overlap = polygon.intersection(accepted).area / max(union, 1e-9)
            if overlap >= 0.6:
                duplicate = True
                break
        if duplicate:
            continue
        regions.append(points)
        region_polygons.append(polygon)
        if len(regions) >= max_objects:
            break
    review = bool(
        repair_radius
        or min(np.diff(levels), default=0)
        < float(options.get("min_contrast", 8))
    )
    reason = (
        f"已用最短像素路径桥接不超过 {repair_radius}px 的小缺口，请人工检查"
        if repair_radius
        else (
            "灰度层级较弱，请人工检查"
            if review
            else "已找到像素格边界，请确认白灰分界"
        )
    )
    if guide is not None:
        distance = Polygon(best[1]).boundary.hausdorff_distance(guide.boundary)
        if distance > radius:
            reason += f"；已扩展搜索，轮廓最大偏移 {distance:.1f}px"
    return CellEdgeResult(
        best[1],
        reason=reason,
        moved_points=len(best[1]),
        threshold_used=best[2],
        fit_error=0.0,
        fit_fraction=1.0,
        score=float(best[0]),
        candidates=[c[1] for c in candidates[:8]],
        regions=regions,
        review_required=review,
    )


def segment_box_to_edge_polygon(image, box, settings=None):
    return _search(image, box, settings)


def refine_polygon_to_edge(image, points, settings=None):
    original = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(original) < 3 or not np.isfinite(original).all():
        return CellEdgeResult(reason="原标注没有足够的有效顶点")
    return _search(
        image, [original.min(axis=0), original.max(axis=0)], settings, original
    )


def refine_polygons_to_edges(image, polygons, settings=None):
    """Return independent preview results for multiple existing polygons."""
    gray = _gray(image)
    results = []
    for points in polygons:
        try:
            results.append(refine_polygon_to_edge(gray, points, settings))
        except (
            Exception
        ) as error:  # defensive: one bad shape must not abort all
            results.append(CellEdgeResult(reason=str(error)))
    return results


def fit_rectangle_to_edges(image, box, settings=None):
    """Snap the four sides of a loose axis-aligned rectangle independently."""
    gray = _gray(image).astype(np.float32)
    options = _settings(settings)
    box = np.asarray(box, dtype=float).reshape(-1, 2)
    if len(box) < 2 or not np.isfinite(box).all():
        return CellEdgeResult(reason="矩形引导坐标无效")
    lo = box.min(axis=0)
    hi = box.max(axis=0)
    if np.any(hi - lo < 2):
        return CellEdgeResult(reason="矩形引导区域太小")
    radius = max(1, int(np.ceil(float(options.get("search_radius", 3)))))
    if options.get("adaptive_search", True):
        radius = min(100, radius * 2)

    height, width = gray.shape
    x_start = max(0, int(np.floor(lo[0])) - radius - 1)
    x_stop = min(width - 1, int(np.ceil(hi[0])) + radius)
    y_start = max(0, int(np.floor(lo[1])) - radius - 1)
    y_stop = min(height - 1, int(np.ceil(hi[1])) + radius)
    if x_stop <= x_start or y_stop <= y_start:
        return CellEdgeResult(reason="矩形超出图像可检测范围")

    vertical = np.abs(
        gray[y_start : y_stop + 1, 1:] - gray[y_start : y_stop + 1, :-1]
    )
    horizontal = np.abs(
        gray[1:, x_start : x_stop + 1] - gray[:-1, x_start : x_stop + 1]
    )

    def strongest(profile, target, lower, upper):
        start = max(lower, int(np.floor(target)) - radius)
        stop = min(upper, int(np.ceil(target)) + radius)
        if stop < start:
            return None, 0.0
        indices = np.arange(start, stop + 1, dtype=int)
        values = profile[indices]
        if not len(values):
            return None, 0.0
        # Prefer a nearby line when several edges have comparable contrast.
        distance_penalty = np.abs(indices + 0.5 - target) * 0.02
        best_index = int(np.argmax(values - distance_penalty))
        return float(indices[best_index] + 0.5), float(values[best_index])

    vertical_profile = np.percentile(vertical, 70, axis=0)
    horizontal_profile = np.percentile(horizontal, 70, axis=1)
    left, left_strength = strongest(vertical_profile, lo[0], 0, width - 2)
    right, right_strength = strongest(vertical_profile, hi[0], 0, width - 2)
    top, top_strength = strongest(horizontal_profile, lo[1], 0, height - 2)
    bottom, bottom_strength = strongest(
        horizontal_profile, hi[1], 0, height - 2
    )
    strengths = [left_strength, right_strength, top_strength, bottom_strength]
    if None in (left, right, top, bottom) or left >= right or top >= bottom:
        return CellEdgeResult(
            reason="四条边中有边界未能可靠定位，已保留引导矩形"
        )
    minimum = float(options.get("min_contrast", 8))
    weak = [index for index, value in enumerate(strengths) if value < minimum]
    if weak:
        names = "左右上下"
        return CellEdgeResult(
            reason="、".join(names[index] for index in weak)
            + "边局部对比度不足，请放宽搜索或调整引导框"
        )
    points = np.asarray(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float64,
    )
    return CellEdgeResult(
        points=points,
        reason="四条矩形边已分别贴合到最近的像素分界",
        moved_points=4,
        fit_error=0.0,
        fit_fraction=1.0,
        candidates=[points],
        regions=[points],
    )


def validate_polygon_edge_fit(image, points, settings=None):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 4 or not np.isfinite(points).all():
        return CellEdgeResult(reason="轮廓没有足够的有效拐点")
    delta = np.roll(points, -1, axis=0) - points
    if not np.allclose(points - 0.5, np.round(points - 0.5), atol=1e-7):
        return CellEdgeResult(reason="请将顶点放在像素格交点上")
    if np.any((np.abs(delta[:, 0]) > 1e-7) & (np.abs(delta[:, 1]) > 1e-7)):
        return CellEdgeResult(reason="轮廓含斜线，请沿像素边界直角连接")
    if np.any(np.linalg.norm(delta, axis=1) <= 1e-7):
        return CellEdgeResult(reason="轮廓含重复顶点，请先合并后再确认")
    polygon = Polygon(points)
    if not polygon.is_valid or polygon.area <= 0:
        return CellEdgeResult(reason="轮廓自相交或面积为零，请修正")
    gray = _gray(image)
    if np.any(points.min(axis=0) < -0.5) or np.any(
        points.max(axis=0) > np.asarray(gray.shape[::-1]) - 0.5
    ):
        return CellEdgeResult(reason="轮廓超出图像边界，请修正")
    return CellEdgeResult(points.copy(), fit_error=0.0, fit_fraction=1.0)


def refine_model_polygons_to_edges(image, polygons, settings=None):
    gray = _gray(image)
    results = []
    for points in polygons:
        try:
            result = refine_polygon_to_edge(gray, points, settings)
            results.append(
                result.points
                if result.succeeded and not result.review_required
                else None
            )
        except (ValueError, cv2.error):
            results.append(None)
    return results
