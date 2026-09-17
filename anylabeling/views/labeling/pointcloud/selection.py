"""Screen-space selection using the renderer's square pixel footprints."""

from dataclasses import dataclass, field

import numpy as np

DEPTH_MAX = (1 << 24) - 1
_TILE_SIZE = 16


def project_points(points, matrix, width, height):
    coordinates = np.asarray(points[:, :3], dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    clip = coordinates @ matrix[:3, :3].T + matrix[:3, 3]
    finite = (
        np.isfinite(clip[:, 0])
        & np.isfinite(clip[:, 1])
        & np.isfinite(clip[:, 2])
    )
    if matrix[3, 3] == 1 and not matrix[3, :3].any():
        clip[~finite] = np.nan
        valid = (
            (np.abs(clip[:, 0]) <= 1)
            & (np.abs(clip[:, 1]) <= 1)
            & (np.abs(clip[:, 2]) <= 1)
        )
        screen = np.column_stack(
            ((clip[:, 0] + 1) * width / 2, (1 - clip[:, 1]) * height / 2)
        )
        return screen, (clip[:, 2] + 1) / 2, valid
    w = coordinates @ matrix[3, :3] + matrix[3, 3]
    valid = (w > 0) & finite
    ndc = np.full(clip.shape, np.nan, dtype=np.float32)
    np.divide(clip, w[:, None], out=ndc, where=valid[:, None])
    valid &= (
        (np.abs(ndc[:, 0]) <= 1)
        & (np.abs(ndc[:, 1]) <= 1)
        & (np.abs(ndc[:, 2]) <= 1)
    )
    screen = np.column_stack(
        ((ndc[:, 0] + 1) * width / 2, (1 - ndc[:, 1]) * height / 2)
    )
    return screen, (ndc[:, 2] + 1) / 2, valid


@dataclass
class SelectionSnapshot:
    indices: np.ndarray
    origins: np.ndarray
    depths: np.ndarray
    width: int
    height: int
    point_size: int
    surface_depth: object = None
    surface_owners: object = None
    exact_depths: object = None
    _tile_order: object = field(
        default=None, init=False, repr=False, compare=False
    )
    _tile_offsets: object = field(
        default=None, init=False, repr=False, compare=False
    )
    _surface_keys: object = field(
        default=None, init=False, repr=False, compare=False
    )
    _surface_order: object = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def create(
        cls,
        screen,
        depths,
        visible,
        width,
        height,
        point_size,
        mode,
        surface_owners=None,
    ):
        if mode not in ("surface", "through"):
            raise ValueError("Unknown depth selection mode")
        size = max(1, int(round(point_size)))
        valid = np.asarray(visible, dtype=bool).copy()
        valid &= (
            np.isfinite(screen[:, 0])
            & np.isfinite(screen[:, 1])
            & np.isfinite(depths)
        )
        valid &= (screen[:, 0] >= 0) & (screen[:, 0] < width)
        valid &= (screen[:, 1] >= 0) & (screen[:, 1] < height)
        valid &= (depths >= 0) & (depths <= 1)
        indices = np.flatnonzero(valid)
        origins = np.floor(screen[indices]).astype(np.int32) - size // 2
        quantized = np.floor(
            np.asarray(depths[indices], dtype=np.float64) * DEPTH_MAX + 0.5
        ).astype(np.uint32)
        snapshot = cls(indices, origins, quantized, width, height, size)
        indexed_surface = (
            len(indices) >= 4096
            and mode == "surface"
            and surface_owners is not None
            and depths.dtype == np.float32
            and width * height <= np.iinfo(np.uint32).max
        )
        if len(indices) >= 4096 and not indexed_surface:
            columns = (width + _TILE_SIZE - 1) // _TILE_SIZE
            centers = origins + size // 2
            tiles = (
                centers[:, 1] // _TILE_SIZE * columns
                + centers[:, 0] // _TILE_SIZE
            )
            snapshot._tile_order = np.argsort(tiles)
            counts = np.bincount(
                tiles,
                minlength=columns * ((height + _TILE_SIZE - 1) // _TILE_SIZE),
            )
            snapshot._tile_offsets = np.empty(len(counts) + 1, dtype=np.int64)
            snapshot._tile_offsets[0] = 0
            np.cumsum(counts, out=snapshot._tile_offsets[1:])
        if mode == "surface" and surface_owners is not None:
            lookup = np.full(len(visible) + 1, -1, dtype=np.int64)
            lookup[indices + 1] = np.arange(len(indices))
            snapshot.surface_owners = lookup[surface_owners.ravel()]
            snapshot.exact_depths = depths[indices].copy()
            if indexed_surface:
                keys = snapshot._footprint_depth_keys(slice(None))
                snapshot._surface_order = np.argsort(keys)
                snapshot._surface_keys = keys[snapshot._surface_order]
        elif mode == "surface":
            snapshot.surface_depth = np.full(
                width * height, DEPTH_MAX + 1, dtype=np.uint32
            )
            for dx in range(size):
                for dy in range(size):
                    x = origins[:, 0] + dx
                    y = origins[:, 1] + dy
                    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                    np.minimum.at(
                        snapshot.surface_depth,
                        y[inside] * width + x[inside],
                        quantized[inside],
                    )
        return snapshot

    def _footprint_depth_keys(self, indices):
        origins = self.origins[indices]
        half = self.point_size // 2
        pixels = (origins[:, 1] + half).astype(np.uint64)
        pixels *= self.width
        pixels += (origins[:, 0] + half).astype(np.uint64)
        return (pixels << 32) | (
            self.exact_depths[indices].view(np.uint32) & 0x7FFFFFFF
        )

    def _candidates(self, bounds):
        left, top, right, bottom = bounds
        origins = self.origins
        candidate = None
        if self._tile_order is not None and np.isfinite(bounds).all():
            half = self.point_size // 2
            columns = (self.width + _TILE_SIZE - 1) // _TILE_SIZE
            rows = (self.height + _TILE_SIZE - 1) // _TILE_SIZE
            x0 = max(
                0, int(np.floor((left - self.point_size + half) / _TILE_SIZE))
            )
            y0 = max(
                0, int(np.floor((top - self.point_size + half) / _TILE_SIZE))
            )
            x1 = min(columns - 1, int(np.floor((right + half) / _TILE_SIZE)))
            y1 = min(rows - 1, int(np.floor((bottom + half) / _TILE_SIZE)))
            if x0 > x1 or y0 > y1:
                return np.empty(0, dtype=np.int64)
            if (x1 - x0 + 1) * (y1 - y0 + 1) < columns * rows // 2:
                candidate = np.concatenate(
                    [
                        self._tile_order[
                            self._tile_offsets[
                                y * columns + x0
                            ] : self._tile_offsets[y * columns + x1 + 1]
                        ]
                        for y in range(y0, y1 + 1)
                    ]
                )
                origins = origins[candidate]
        inside = (
            (origins[:, 0] + self.point_size >= left)
            & (origins[:, 0] <= right)
            & (origins[:, 1] + self.point_size >= top)
            & (origins[:, 1] <= bottom)
        )
        return (
            np.flatnonzero(inside) if candidate is None else candidate[inside]
        )

    def _select_surface(self, bounds, contains, candidate=None):
        if not np.isfinite(bounds).all():
            return None
        left, top, right, bottom = bounds
        x0, y0 = max(0, int(np.ceil(left - 0.5))), max(
            0, int(np.ceil(top - 0.5))
        )
        x1, y1 = min(self.width, int(np.floor(right - 0.5)) + 1), min(
            self.height, int(np.floor(bottom - 0.5)) + 1
        )
        if x0 >= x1 or y0 >= y1:
            return np.empty(0, dtype=np.int64)
        area = (x1 - x0) * (y1 - y0)
        count = len(self.indices) if candidate is None else len(candidate)
        if area > min(262144, max(4096, count * self.point_size**2 * 4)):
            return None
        region = self.surface_owners.reshape(self.height, self.width)[
            y0:y1, x0:x1
        ]
        y, x = np.nonzero(region >= 0)
        owners = region[y, x]
        x, y = x + x0, y + y0
        origins = self.origins[owners]
        inside = (
            (x >= origins[:, 0])
            & (x < origins[:, 0] + self.point_size)
            & (y >= origins[:, 1])
            & (y < origins[:, 1] + self.point_size)
            & contains(x + 0.5, y + 0.5)
        )
        owners = np.unique(owners[inside])
        if not len(owners):
            return np.empty(0, dtype=np.int64)
        if self._surface_keys is not None:
            keys = np.unique(self._footprint_depth_keys(owners))
            starts = np.searchsorted(self._surface_keys, keys, side="left")
            ends = np.searchsorted(self._surface_keys, keys, side="right")
            if np.all(ends - starts == 1):
                selected = self._surface_order[starts]
            else:
                selected = np.concatenate(
                    [
                        self._surface_order[start:end]
                        for start, end in zip(starts, ends)
                    ]
                )
            return np.sort(self.indices[selected])
        origins = self.origins[owners]
        pixels = origins[:, 1].astype(np.int64) * self.width + origins[:, 0]
        pixels, first, inverse = np.unique(
            pixels, return_index=True, return_inverse=True
        )
        if np.any(
            self.exact_depths[owners]
            != self.exact_depths[owners[first[inverse]]]
        ):
            return None
        owners = owners[first]
        origins = self.origins[candidate]
        candidate_pixels = (
            origins[:, 1].astype(np.int64) * self.width + origins[:, 0]
        )
        positions = np.searchsorted(pixels, candidate_pixels)
        np.minimum(positions, len(pixels) - 1, out=positions)
        selected = (pixels[positions] == candidate_pixels) & (
            self.exact_depths[candidate]
            == self.exact_depths[owners[positions]]
        )
        return np.sort(self.indices[candidate[selected]])

    def select(self, bounds, contains):
        origins = self.origins
        if self._surface_keys is not None:
            result = self._select_surface(bounds, contains)
            if result is not None:
                return result
        candidate = self._candidates(bounds)
        if not len(candidate):
            return np.empty(0, dtype=np.int64)
        if self.surface_owners is not None:
            result = self._select_surface(bounds, contains, candidate)
            if result is not None:
                return result
        selected = np.zeros(len(candidate), dtype=bool)
        for dx in range(self.point_size):
            for dy in range(self.point_size):
                remaining = np.flatnonzero(~selected)
                if not len(remaining):
                    break
                local = candidate[remaining]
                x = origins[local, 0] + dx
                y = origins[local, 1] + dy
                inside = (
                    (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
                )
                inside &= contains(x + 0.5, y + 0.5)
                if self.surface_depth is not None:
                    valid = np.flatnonzero(inside)
                    inside[valid] &= (
                        self.depths[local[valid]]
                        == self.surface_depth[y[valid] * self.width + x[valid]]
                    )
                elif self.surface_owners is not None:
                    valid = np.flatnonzero(inside)
                    owners = self.surface_owners[
                        y[valid] * self.width + x[valid]
                    ]
                    same_footprint = np.all(
                        origins[local[valid]] == origins[owners], axis=1
                    )
                    same_depth = (
                        self.exact_depths[local[valid]]
                        == self.exact_depths[owners]
                    )
                    inside[valid] &= (owners >= 0) & (
                        (owners == local[valid])
                        | (same_footprint & same_depth)
                    )
                selected[remaining[inside]] = True
        result = self.indices[candidate[selected]]
        return np.sort(result) if self._tile_order is not None else result


def select_brush(snapshot, start, end, radius):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length_squared = float(direction @ direction)

    def contains(x, y):
        fraction = np.zeros(len(x))
        if length_squared:
            fraction = np.clip(
                ((x - start[0]) * direction[0] + (y - start[1]) * direction[1])
                / length_squared,
                0,
                1,
            )
        dx = x - start[0] - fraction * direction[0]
        dy = y - start[1] - fraction * direction[1]
        return dx * dx + dy * dy <= radius * radius

    return snapshot.select(
        (
            min(start[0], end[0]) - radius,
            min(start[1], end[1]) - radius,
            max(start[0], end[0]) + radius,
            max(start[1], end[1]) + radius,
        ),
        contains,
    )


def validate_polygon(vertices):
    polygon = np.asarray(vertices, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("A polygon needs at least three vertices.")
    if not np.isfinite(polygon).all():
        raise ValueError("Polygon vertices must be finite.")
    if len(np.unique(polygon, axis=0)) != len(polygon):
        raise ValueError("Polygon vertices must be distinct.")
    following = np.roll(polygon, -1, axis=0)
    area = np.sum(
        polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    )
    if abs(area) < 1e-7:
        raise ValueError("The polygon must have a nonzero area.")

    def orientation(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for i, (a, b) in enumerate(zip(polygon, following)):
        for j in range(i + 2, len(polygon)):
            if i == 0 and j == len(polygon) - 1:
                continue
            c, d = polygon[j], following[j]
            boxes_overlap = max(min(a[0], b[0]), min(c[0], d[0])) <= min(
                max(a[0], b[0]), max(c[0], d[0])
            ) and max(min(a[1], b[1]), min(c[1], d[1])) <= min(
                max(a[1], b[1]), max(c[1], d[1])
            )
            if (
                boxes_overlap
                and orientation(a, b, c) * orientation(a, b, d) <= 0
                and orientation(c, d, a) * orientation(c, d, b) <= 0
            ):
                raise ValueError(
                    "Self-intersecting polygons are not supported."
                )
    return polygon


def select_polygon(snapshot, vertices):
    polygon = validate_polygon(vertices)

    def contains(x, y):
        inside = np.zeros(len(x), dtype=bool)
        boundary = np.zeros(len(x), dtype=bool)
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            dx, dy = b - a
            cross = (x - a[0]) * dy - (y - a[1]) * dx
            boundary |= (
                (np.abs(cross) <= 1e-7)
                & (x >= min(a[0], b[0]))
                & (x <= max(a[0], b[0]))
                & (y >= min(a[1], b[1]))
                & (y <= max(a[1], b[1]))
            )
            if dy:
                inside ^= ((a[1] > y) != (b[1] > y)) & (
                    x < a[0] + (y - a[1]) * dx / dy
                )
        return inside | boundary

    return snapshot.select(
        (*polygon.min(axis=0), *polygon.max(axis=0)), contains
    )
