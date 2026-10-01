from dataclasses import dataclass, replace
import math

import numpy as np

MIN_SIZE = 0.01
CORNER_SIGNS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)
EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 3, 7, 4),
    (1, 2, 6, 5),
)
VIEW_AXES = {"top": (0, 1), "front": (1, 2), "side": (0, 2)}


def rotation_matrix(angles):
    x, y, z = angles
    a, b = math.cos(x), math.sin(x)
    c, d = math.cos(y), math.sin(y)
    e, f = math.cos(z), math.sin(z)
    return np.array(
        [
            [c * e, -c * f, d],
            [a * f + b * e * d, a * e - b * f * d, -b * c],
            [b * f - a * e * d, b * e + a * f * d, a * c],
        ]
    )


def rotation_angles(matrix):
    y = math.asin(np.clip(matrix[0, 2], -1, 1))
    if abs(matrix[0, 2]) < 0.9999999:
        x = math.atan2(-matrix[1, 2], matrix[2, 2])
        z = math.atan2(-matrix[0, 1], matrix[0, 0])
    else:
        x = math.atan2(matrix[2, 1], matrix[1, 1])
        z = 0.0
    return x, y, z


@dataclass(frozen=True)
class Cuboid:
    id: int
    class_id: int
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    occluded: bool = False
    locked: bool = False

    def __post_init__(self):
        for name in ("id", "class_id"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 65535:
                raise ValueError("Cuboid IDs and classes must be in 1-65535.")
        for name in ("center", "size", "rotation"):
            value = np.asarray(getattr(self, name))
            if (
                value.shape != (3,)
                or value.dtype.kind not in "fiu"
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    f"Cuboid {name} requires three finite numbers."
                )
            object.__setattr__(self, name, tuple(float(v) for v in value))
        if min(self.size) < MIN_SIZE:
            raise ValueError(f"Cuboid dimensions must be at least {MIN_SIZE}.")
        if type(self.occluded) is not bool or type(self.locked) is not bool:
            raise ValueError("Cuboid flags must be boolean.")

    @property
    def matrix(self):
        return rotation_matrix(self.rotation)

    def corners(self):
        return (
            CORNER_SIGNS * (np.asarray(self.size) / 2)
        ) @ self.matrix.T + self.center

    def contains(self, points):
        local = (np.asarray(points)[:, :3] - self.center) @ self.matrix
        return np.all(
            np.abs(local) <= np.asarray(self.size) / 2 + 1e-6, axis=1
        )

    def resized(self, axes, signs, delta):
        size = np.array(self.size)
        shift = np.zeros(3)
        for axis, sign in zip(axes, signs):
            if sign:
                size[axis] = max(MIN_SIZE, size[axis] + sign * delta[axis])
                shift[axis] = sign * (size[axis] - self.size[axis]) / 2
        return replace(
            self,
            size=tuple(size),
            center=tuple(self.center + self.matrix @ shift),
        )

    def rotated(self, axis, angle):
        angles = np.zeros(3)
        angles[axis] = angle
        return replace(
            self,
            rotation=rotation_angles(self.matrix @ rotation_matrix(angles)),
        )

    def fitted(self, points):
        points = np.asarray(points)
        if not len(points):
            raise ValueError("There are no points inside this cuboid to fit.")
        local = (points[:, :3] - self.center) @ self.matrix
        low, high = local.min(axis=0), local.max(axis=0)
        return replace(
            self,
            center=tuple(self.center + self.matrix @ ((low + high) / 2)),
            size=tuple(np.maximum(high - low, MIN_SIZE)),
        )
