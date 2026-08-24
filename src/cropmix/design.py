"""Spatial assignment of varieties to field planting sites."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .errors import ValidationError
from .geometry import Field


@dataclass(frozen=True)
class MixtureDesign:
    """Assign exactly one named variety to every planting site."""

    field: Field
    assignment: tuple[str, ...]

    def __post_init__(self) -> None:
        assignment = tuple(str(x) for x in self.assignment)
        if len(assignment) != self.field.n_sites:
            raise ValidationError(
                f"assignment has {len(assignment)} labels but field has {self.field.n_sites} sites."
            )
        if any(not name.strip() for name in assignment):
            raise ValidationError("Variety labels cannot be empty.")
        object.__setattr__(self, "assignment", assignment)

    @property
    def n_sites(self) -> int:
        return self.field.n_sites

    @property
    def varieties(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.assignment)))

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(self.assignment))

    @property
    def proportions(self) -> dict[str, float]:
        return {name: count / self.n_sites for name, count in self.counts.items()}

    def variety_indices(self, variety: str) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.assignment, dtype=object) == variety)

    def as_grid(self) -> np.ndarray:
        if self.field.grid_shape is None:
            raise ValidationError("This field has no complete rectangular grid_shape.")
        return np.asarray(self.assignment, dtype=object).reshape(self.field.grid_shape)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "site_id": self.field.site_ids,
                "x": self.field.x,
                "y": self.field.y,
                "variety": self.assignment,
            }
        )

    def swapped(self, first: int, second: int) -> "MixtureDesign":
        if not (0 <= first < self.n_sites and 0 <= second < self.n_sites):
            raise IndexError("Swap indices are outside the field.")
        values = list(self.assignment)
        values[first], values[second] = values[second], values[first]
        return MixtureDesign(self.field, tuple(values))

    def plot(self, ax=None, *, marker_size: float = 80, legend: bool = True):
        """Plot the planting assignment. Requires the optional `viz` extra."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install plotting support with `pip install cropmix[viz]`.") from exc

        if ax is None:
            _, ax = plt.subplots()
        for variety in self.varieties:
            idx = self.variety_indices(variety)
            ax.scatter(self.field.x[idx], self.field.y[idx], s=marker_size, label=variety)
        if self.field.boundary is not None:
            boundary = np.vstack([self.field.boundary, self.field.boundary[0]])
            ax.plot(boundary[:, 0], boundary[:, 1], linewidth=1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if legend:
            ax.legend()
        return ax

    @classmethod
    def monoculture(cls, field: Field, variety: str) -> "MixtureDesign":
        return cls(field=field, assignment=tuple([variety] * field.n_sites))

    @classmethod
    def random(
        cls,
        field: Field,
        counts: Mapping[str, int],
        *,
        seed: int | None = None,
    ) -> "MixtureDesign":
        counts = {str(name): int(count) for name, count in counts.items()}
        if any(count < 0 for count in counts.values()):
            raise ValidationError("Variety counts cannot be negative.")
        if sum(counts.values()) != field.n_sites:
            raise ValidationError(
                f"Counts sum to {sum(counts.values())}, but field has {field.n_sites} sites."
            )
        values: list[str] = []
        for name, count in counts.items():
            values.extend([name] * count)
        rng = np.random.default_rng(seed)
        rng.shuffle(values)
        return cls(field=field, assignment=tuple(values))

    @classmethod
    def from_grid(cls, grid: Sequence[Sequence[str]], *, spacing: float = 1.0) -> "MixtureDesign":
        array = np.asarray(grid, dtype=object)
        if array.ndim != 2:
            raise ValidationError("grid must be two-dimensional.")
        field = Field.rectangular(array.shape[0], array.shape[1], spacing=spacing)
        return cls(field=field, assignment=tuple(str(x) for x in array.ravel()))

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        *,
        x: str = "x",
        y: str = "y",
        variety: str = "variety",
        site_id: str | None = "site_id",
    ) -> "MixtureDesign":
        required = [x, y, variety]
        missing = [column for column in required if column not in dataframe.columns]
        if missing:
            raise ValidationError(f"Missing columns: {missing}")
        ids = None
        if site_id is not None and site_id in dataframe.columns:
            ids = tuple(dataframe[site_id].astype(str))
        field = Field.from_coordinates(
            dataframe[[x, y]].to_numpy(float),
            site_ids=ids,
        )
        return cls(field=field, assignment=tuple(dataframe[variety].astype(str)))
