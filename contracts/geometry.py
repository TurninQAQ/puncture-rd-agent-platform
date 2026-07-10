"""Geometry primitives.

Conventions are deliberately explicit:

* ``size_ijk`` is image index order (i, j, k), never NumPy array order.
* world coordinates are millimetres in the declared LPS/RAS system.
* ``direction_cosines`` contains a 3x3 matrix in row-major order.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from .enums import CoordinateSystem


def _require_triplet(name: str, values: tuple[float, float, float]) -> None:
    if len(values) != 3 or not all(isfinite(float(value)) for value in values):
        raise ValueError(f"{name} must contain exactly three finite numbers")


@dataclass(frozen=True, slots=True)
class WorldPoint:
    x_mm: float
    y_mm: float
    z_mm: float

    def __post_init__(self) -> None:
        _require_triplet("world point", (self.x_mm, self.y_mm, self.z_mm))

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x_mm, self.y_mm, self.z_mm)


@dataclass(frozen=True, slots=True)
class VoxelPoint:
    i: float
    j: float
    k: float

    def __post_init__(self) -> None:
        _require_triplet("voxel point", (self.i, self.j, self.k))

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.i, self.j, self.k)


@dataclass(frozen=True, slots=True)
class VolumeGeometry:
    size_ijk: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float]
    direction_cosines: tuple[float, float, float, float, float, float, float, float, float]
    coordinate_system: CoordinateSystem
    geometry_fingerprint: str = ""

    def __post_init__(self) -> None:
        if len(self.size_ijk) != 3 or any(value <= 0 for value in self.size_ijk):
            raise ValueError("size_ijk must contain three positive integers")
        _require_triplet("spacing_mm", self.spacing_mm)
        if any(value <= 0 for value in self.spacing_mm):
            raise ValueError("spacing_mm values must be positive")
        _require_triplet("origin_mm", self.origin_mm)
        if len(self.direction_cosines) != 9 or not all(
            isfinite(float(value)) for value in self.direction_cosines
        ):
            raise ValueError("direction_cosines must contain nine finite numbers")
        expected = self.compute_fingerprint()
        if self.geometry_fingerprint and self.geometry_fingerprint != expected:
            raise ValueError("geometry_fingerprint does not match geometry fields")
        if not self.geometry_fingerprint:
            object.__setattr__(self, "geometry_fingerprint", expected)

    def compute_fingerprint(self) -> str:
        payload = "|".join(
            (
                ",".join(str(value) for value in self.size_ijk),
                ",".join(f"{value:.9g}" for value in self.spacing_mm),
                ",".join(f"{value:.9g}" for value in self.origin_mm),
                ",".join(f"{value:.9g}" for value in self.direction_cosines),
                self.coordinate_system.value,
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def is_compatible_with(
        self,
        other: "VolumeGeometry",
        *,
        spacing_tolerance_mm: float = 1e-4,
        origin_tolerance_mm: float = 1e-3,
        direction_tolerance: float = 1e-5,
    ) -> bool:
        return (
            self.size_ijk == other.size_ijk
            and self.coordinate_system == other.coordinate_system
            and all(abs(a - b) <= spacing_tolerance_mm for a, b in zip(self.spacing_mm, other.spacing_mm))
            and all(abs(a - b) <= origin_tolerance_mm for a, b in zip(self.origin_mm, other.origin_mm))
            and all(
                abs(a - b) <= direction_tolerance
                for a, b in zip(self.direction_cosines, other.direction_cosines)
            )
        )
