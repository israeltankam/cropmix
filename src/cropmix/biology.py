"""Biological parameter objects used throughout Cropmix."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .errors import ValidationError

TransmissionMode = Literal["SPT", "PT"]


def _nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValidationError(f"{name} must be non-negative; got {value!r}.")


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValidationError(f"{name} must be positive; got {value!r}.")


@dataclass(frozen=True)
class HostTransmission:
    """Host-dependent virus transmission rates.

    Parameters
    ----------
    acquisition_rate:
        Rate at which a virus-free vector acquires virus while feeding on an
        infectious plant of this variety, in day^-1.
    inoculation_rate:
        Per-virus-bearing-vector inoculation rate for a susceptible plant of
        this variety, in vector^-1 day^-1.
    """

    acquisition_rate: float
    inoculation_rate: float

    def __post_init__(self) -> None:
        _nonnegative("acquisition_rate", self.acquisition_rate)
        _nonnegative("inoculation_rate", self.inoculation_rate)


@dataclass(frozen=True)
class PlantParameters:
    """Plant-side epidemic rates, in day^-1."""

    latent_progression_rate: float
    roguing_rate: float = 0.0

    def __post_init__(self) -> None:
        _nonnegative("latent_progression_rate", self.latent_progression_rate)
        _nonnegative("roguing_rate", self.roguing_rate)


@dataclass(frozen=True)
class YieldParameters:
    """Healthy and infectious yields in a common user-chosen unit."""

    healthy: float
    infected: float
    unit: str = "t/ha"

    def __post_init__(self) -> None:
        _nonnegative("healthy yield", self.healthy)
        _nonnegative("infected yield", self.infected)
        if not self.unit:
            raise ValidationError("Yield unit cannot be empty.")


@dataclass(frozen=True)
class Variety:
    """One crop variety used in a mixture."""

    name: str
    transmission: HostTransmission
    plant: PlantParameters
    yield_model: YieldParameters

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValidationError("Variety name cannot be empty.")

    def with_transmission(self, transmission: HostTransmission) -> "Variety":
        """Return a copy with updated transmission rates."""
        return replace(self, transmission=transmission)


@dataclass(frozen=True)
class VectorParameters:
    """Vector demographic and movement rates, in day^-1."""

    mortality_rate: float
    dispersal_rate: float

    def __post_init__(self) -> None:
        _nonnegative("mortality_rate", self.mortality_rate)
        _nonnegative("dispersal_rate", self.dispersal_rate)


@dataclass(frozen=True)
class PathogenParameters:
    """Virus-vector parameters shared across host varieties.

    `vector_latent_progression_rate` is required only for PT transmission.
    Cropmix 0.1 can infer PT parameters through the EpiPvr bridge, but the
    spatial simulation engine currently implements SPT dynamics only.
    """

    vector_clearance_rate: float
    transmission_mode: TransmissionMode = "SPT"
    vector_latent_progression_rate: float | None = None

    def __post_init__(self) -> None:
        _nonnegative("vector_clearance_rate", self.vector_clearance_rate)
        if self.transmission_mode not in ("SPT", "PT"):
            raise ValidationError("transmission_mode must be 'SPT' or 'PT'.")
        if self.transmission_mode == "PT":
            if self.vector_latent_progression_rate is None:
                raise ValidationError(
                    "PT transmission requires vector_latent_progression_rate."
                )
            _positive(
                "vector_latent_progression_rate",
                self.vector_latent_progression_rate,
            )
        elif self.vector_latent_progression_rate is not None:
            _nonnegative(
                "vector_latent_progression_rate",
                self.vector_latent_progression_rate,
            )


@dataclass(frozen=True)
class TransmissionDraw:
    """One coherent multi-variety transmission draw for uncertainty propagation.

    Rates are stored by variety name.  The draw can originate from EpiPvr or
    another inferential source.  Keeping acquisition and inoculation together
    in one draw prevents accidental independent resampling of posterior
    marginals.
    """

    acquisition_rates: dict[str, float]
    inoculation_rates: dict[str, float]
    vector_clearance_rate: float
    vector_latent_progression_rate: float | None = None

    def __post_init__(self) -> None:
        if set(self.acquisition_rates) != set(self.inoculation_rates):
            raise ValidationError(
                "Acquisition and inoculation dictionaries must have identical variety names."
            )
        for name, value in self.acquisition_rates.items():
            _nonnegative(f"acquisition rate for {name}", value)
        for name, value in self.inoculation_rates.items():
            _nonnegative(f"inoculation rate for {name}", value)
        _nonnegative("vector_clearance_rate", self.vector_clearance_rate)
        if self.vector_latent_progression_rate is not None:
            _nonnegative(
                "vector_latent_progression_rate",
                self.vector_latent_progression_rate,
            )
