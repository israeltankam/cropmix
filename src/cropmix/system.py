"""A complete Cropmix biological and movement system."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .biology import PathogenParameters, TransmissionDraw, Variety, VectorParameters
from .design import MixtureDesign
from .errors import UnsupportedModelError, ValidationError
from .kernels import ExponentialKernel


@dataclass(frozen=True)
class CropMixSystem:
    """Biological parameters plus the vector movement kernel."""

    varieties: tuple[Variety, ...]
    vector: VectorParameters
    pathogen: PathogenParameters
    kernel: ExponentialKernel = ExponentialKernel()

    def __post_init__(self) -> None:
        varieties = tuple(self.varieties)
        if not varieties:
            raise ValidationError("At least one variety is required.")
        names = [variety.name for variety in varieties]
        if len(set(names)) != len(names):
            raise ValidationError("Variety names must be unique within a system.")
        yield_units = {variety.yield_model.unit for variety in varieties}
        if len(yield_units) != 1:
            raise ValidationError("All varieties must use the same yield unit.")
        object.__setattr__(self, "varieties", varieties)

    @property
    def variety_names(self) -> tuple[str, ...]:
        return tuple(variety.name for variety in self.varieties)

    @property
    def variety_map(self) -> dict[str, Variety]:
        return {variety.name: variety for variety in self.varieties}

    @property
    def yield_unit(self) -> str:
        return self.varieties[0].yield_model.unit

    def variety(self, name: str) -> Variety:
        try:
            return self.variety_map[name]
        except KeyError as exc:
            raise KeyError(f"Unknown variety {name!r}. Available: {self.variety_names}") from exc

    def validate_design(self, design: MixtureDesign) -> None:
        missing = set(design.varieties) - set(self.variety_names)
        if missing:
            raise ValidationError(f"Design uses varieties not present in the system: {sorted(missing)}")

    def ensure_spatial_supported(self) -> None:
        if self.pathogen.transmission_mode != "SPT":
            raise UnsupportedModelError(
                "Cropmix 0.1 spatial simulation implements SPT dynamics only. "
                "PT inference is supported by the EpiPvr bridge, but PT spatial simulation "
                "requires an explicit exposed-vector compartment and is intentionally not guessed."
            )

    def with_kernel_scale(self, scale: float) -> "CropMixSystem":
        return replace(self, kernel=self.kernel.with_scale(scale))

    def point_transmission_draw(self) -> TransmissionDraw:
        return TransmissionDraw(
            acquisition_rates={v.name: v.transmission.acquisition_rate for v in self.varieties},
            inoculation_rates={v.name: v.transmission.inoculation_rate for v in self.varieties},
            vector_clearance_rate=self.pathogen.vector_clearance_rate,
            vector_latent_progression_rate=self.pathogen.vector_latent_progression_rate,
        )

    @classmethod
    def from_iterable(
        cls,
        varieties: Iterable[Variety],
        *,
        vector: VectorParameters,
        pathogen: PathogenParameters,
        kernel: ExponentialKernel | None = None,
    ) -> "CropMixSystem":
        return cls(
            varieties=tuple(varieties),
            vector=vector,
            pathogen=pathogen,
            kernel=ExponentialKernel() if kernel is None else kernel,
        )
