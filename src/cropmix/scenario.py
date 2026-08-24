"""Epidemic initial conditions and simulation horizon."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ValidationError


@dataclass(frozen=True)
class Inoculum:
    """Initially infectious plants.

    ``count`` may be zero. If ``sites`` is ``None``, ``count`` sites are chosen
    uniformly without replacement independently for each stochastic replicate.
    Explicit sites make the plant introduction fixed.
    """

    count: int = 1
    sites: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValidationError("Inoculum count must be non-negative.")
        if self.sites is not None:
            sites = tuple(int(x) for x in self.sites)
            if len(sites) != self.count:
                raise ValidationError("len(sites) must equal count.")
            if len(set(sites)) != len(sites):
                raise ValidationError("Explicit inoculum sites must be unique.")
            if any(index < 0 for index in sites):
                raise ValidationError("Inoculum site indices must be non-negative.")
            object.__setattr__(self, "sites", sites)

    @classmethod
    def none(cls) -> "Inoculum":
        return cls(count=0, sites=None)

    @classmethod
    def random(cls, count: int = 1) -> "Inoculum":
        return cls(count=count)

    @classmethod
    def fixed(cls, sites: tuple[int, ...] | list[int]) -> "Inoculum":
        values = tuple(int(x) for x in sites)
        return cls(count=len(values), sites=values)


@dataclass(frozen=True)
class VectorInoculum:
    """Initial infection pressure carried by vectors.

    Parameters
    ----------
    infectious_fraction:
        Probability that an individual vector is virus-bearing at time zero.
        Must lie in [0, 1]. A value of zero represents a virus-free vector
        population.
    origin_weights:
        Optional weights across the system varieties for assigning provenance to
        exogenously infected vectors. If omitted, provenance is assigned in
        proportion to the planted variety frequencies. Provenance affects
        bookkeeping only in the current SPT model; inoculation depends on the
        destination host.
    """

    infectious_fraction: float = 0.0
    origin_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        value = float(self.infectious_fraction)
        if not 0.0 <= value <= 1.0:
            raise ValidationError("infectious_fraction must lie in [0, 1].")
        object.__setattr__(self, "infectious_fraction", value)
        if self.origin_weights is not None:
            weights = tuple(float(x) for x in self.origin_weights)
            if not weights or any(x < 0 for x in weights) or sum(weights) <= 0:
                raise ValidationError("origin_weights must be non-negative and sum to > 0.")
            object.__setattr__(self, "origin_weights", weights)

    @classmethod
    def none(cls) -> "VectorInoculum":
        return cls(0.0)


@dataclass(frozen=True)
class Scenario:
    """Field-level conditions for one epidemic experiment.

    Plant and vector introductions can be used independently or together. The
    default represents the cassava case-study convention of one randomly located
    infectious plant and virus-free vectors.
    """

    duration: float = 300.0
    vectors_per_plant: int = 10
    inoculum: Inoculum = Inoculum()
    vector_inoculum: VectorInoculum = VectorInoculum()

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValidationError("duration must be positive.")
        if self.vectors_per_plant <= 0:
            raise ValidationError("vectors_per_plant must be positive.")
        if self.inoculum.count == 0 and self.vector_inoculum.infectious_fraction == 0:
            # A disease-free control is valid; no error.
            pass
