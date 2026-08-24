"""Field geometry: Cropmix treats a field as a set of plant coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Iterable, Sequence

import numpy as np

from .errors import ValidationError


def _as_coordinates(values: Iterable[Sequence[float]]) -> np.ndarray:
    coordinates = np.asarray(list(values), dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValidationError("coordinates must have shape (n_sites, 2).")
    if coordinates.shape[0] == 0:
        raise ValidationError("A field must contain at least one planting site.")
    if not np.isfinite(coordinates).all():
        raise ValidationError("Field coordinates must be finite.")
    if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
        raise ValidationError("Field coordinates must be unique.")
    coordinates.setflags(write=False)
    return coordinates


def _point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> bool:
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > 1e-10:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= 1e-10


def _point_in_polygon(point: Sequence[float], polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test that includes the boundary."""
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_cross:
                inside = not inside
    return inside


@dataclass(frozen=True)
class Field:
    """Planting-site geometry.

    The canonical geometry is an arbitrary set of two-dimensional plant
    coordinates.  Rectangles, masks, polygons, GPS-like point sets and other
    shapes are convenience constructors around that representation.
    """

    coordinates: np.ndarray
    site_ids: tuple[str, ...] | None = None
    boundary: np.ndarray | None = None
    grid_shape: tuple[int, int] | None = None
    metadata: dict[str, object] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = _as_coordinates(self.coordinates)
        object.__setattr__(self, "coordinates", coords)

        if self.site_ids is None:
            ids = tuple(str(i) for i in range(coords.shape[0]))
        else:
            ids = tuple(str(value) for value in self.site_ids)
            if len(ids) != coords.shape[0]:
                raise ValidationError("site_ids must have one entry per coordinate.")
            if len(set(ids)) != len(ids):
                raise ValidationError("site_ids must be unique.")
        object.__setattr__(self, "site_ids", ids)

        if self.boundary is not None:
            boundary = np.asarray(self.boundary, dtype=float)
            if boundary.ndim != 2 or boundary.shape[1] != 2 or len(boundary) < 3:
                raise ValidationError("boundary must have shape (n_vertices, 2), n_vertices >= 3.")
            if not np.isfinite(boundary).all():
                raise ValidationError("boundary coordinates must be finite.")
            boundary.setflags(write=False)
            object.__setattr__(self, "boundary", boundary)

        if self.grid_shape is not None:
            rows, cols = self.grid_shape
            if rows <= 0 or cols <= 0 or rows * cols != self.n_sites:
                raise ValidationError("grid_shape must match the number of sites.")

    @property
    def n_sites(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def x(self) -> np.ndarray:
        return self.coordinates[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.coordinates[:, 1]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (
            float(self.x.min()),
            float(self.x.max()),
            float(self.y.min()),
            float(self.y.max()),
        )

    def distance_matrix(self) -> np.ndarray:
        delta = self.coordinates[:, None, :] - self.coordinates[None, :, :]
        return np.sqrt(np.sum(delta * delta, axis=2))

    def site_index(self, site_id: str) -> int:
        try:
            return self.site_ids.index(str(site_id))
        except ValueError as exc:
            raise KeyError(f"Unknown site_id: {site_id!r}") from exc

    @classmethod
    def from_coordinates(
        cls,
        coordinates: Iterable[Sequence[float]],
        *,
        site_ids: Sequence[str] | None = None,
        boundary: Iterable[Sequence[float]] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "Field":
        return cls(
            coordinates=np.asarray(list(coordinates), dtype=float),
            site_ids=None if site_ids is None else tuple(site_ids),
            boundary=None if boundary is None else np.asarray(list(boundary), dtype=float),
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def rectangular(
        cls,
        rows: int,
        columns: int,
        *,
        spacing: float | tuple[float, float] = 1.0,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> "Field":
        if rows <= 0 or columns <= 0:
            raise ValidationError("rows and columns must be positive.")
        if isinstance(spacing, tuple):
            sx, sy = float(spacing[0]), float(spacing[1])
        else:
            sx = sy = float(spacing)
        if sx <= 0 or sy <= 0:
            raise ValidationError("spacing must be positive.")

        ox, oy = map(float, origin)
        coords = np.array(
            [(ox + c * sx, oy + r * sy) for r in range(rows) for c in range(columns)],
            dtype=float,
        )
        boundary = np.array(
            [
                (ox, oy),
                (ox + (columns - 1) * sx, oy),
                (ox + (columns - 1) * sx, oy + (rows - 1) * sy),
                (ox, oy + (rows - 1) * sy),
            ],
            dtype=float,
        )
        return cls(
            coordinates=coords,
            boundary=boundary,
            grid_shape=(rows, columns),
            metadata={"constructor": "rectangular", "spacing": (sx, sy)},
        )

    @classmethod
    def from_mask(
        cls,
        mask: np.ndarray,
        *,
        spacing: float | tuple[float, float] = 1.0,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> "Field":
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValidationError("mask must be two-dimensional.")
        if not mask.any():
            raise ValidationError("mask must contain at least one True cell.")
        if isinstance(spacing, tuple):
            sx, sy = map(float, spacing)
        else:
            sx = sy = float(spacing)
        if sx <= 0 or sy <= 0:
            raise ValidationError("spacing must be positive.")
        ox, oy = map(float, origin)
        coords = []
        ids = []
        for r, c in zip(*np.where(mask)):
            coords.append((ox + c * sx, oy + r * sy))
            ids.append(f"r{r}c{c}")
        return cls(
            coordinates=np.asarray(coords),
            site_ids=tuple(ids),
            metadata={"constructor": "mask", "mask_shape": tuple(mask.shape), "spacing": (sx, sy)},
        )

    @classmethod
    def from_polygon(
        cls,
        boundary: Iterable[Sequence[float]],
        *,
        spacing: float | tuple[float, float] = 1.0,
        origin: tuple[float, float] | None = None,
    ) -> "Field":
        """Generate a regular planting lattice clipped to an arbitrary polygon.

        For already surveyed planting positions, prefer :meth:`from_coordinates`.
        """
        polygon = np.asarray(list(boundary), dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValidationError("boundary must contain at least three 2D vertices.")
        if isinstance(spacing, tuple):
            sx, sy = map(float, spacing)
        else:
            sx = sy = float(spacing)
        if sx <= 0 or sy <= 0:
            raise ValidationError("spacing must be positive.")

        xmin, ymin = polygon.min(axis=0)
        xmax, ymax = polygon.max(axis=0)
        if origin is None:
            ox, oy = float(xmin), float(ymin)
        else:
            ox, oy = map(float, origin)

        xs = np.arange(ox, xmax + sx * 0.5, sx)
        ys = np.arange(oy, ymax + sy * 0.5, sy)
        coords = [(x, y) for y in ys for x in xs if _point_in_polygon((x, y), polygon)]
        if not coords:
            raise ValidationError("No planting sites fall inside the polygon at the requested spacing.")
        return cls(
            coordinates=np.asarray(coords, dtype=float),
            boundary=polygon,
            metadata={"constructor": "polygon", "spacing": (sx, sy)},
        )
