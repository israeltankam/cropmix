"""Python-native representations of EpiPvr access-period inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..biology import HostTransmission, PathogenParameters
from ..errors import EpiPvrError, ValidationError

EpiPvrMode = Literal["SPT", "PT"]


@dataclass(frozen=True)
class AccessPeriodAssay:
    """One varying-access-period sub-assay."""

    duration: tuple[float, ...]
    tested: tuple[int, ...]
    infected: tuple[int, ...]

    def __post_init__(self) -> None:
        duration = tuple(float(x) for x in self.duration)
        tested = tuple(int(x) for x in self.tested)
        infected = tuple(int(x) for x in self.infected)
        if not (len(duration) == len(tested) == len(infected)) or not duration:
            raise ValidationError("duration, tested and infected must have equal non-zero lengths.")
        if any(x <= 0 for x in duration):
            raise ValidationError("Access durations must be positive.")
        if any(x <= 0 for x in tested):
            raise ValidationError("Numbers tested must be positive.")
        if any(x < 0 for x in infected):
            raise ValidationError("Numbers infected cannot be negative.")
        if any(i > n for i, n in zip(infected, tested)):
            raise ValidationError("infected cannot exceed tested.")
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "tested", tested)
        object.__setattr__(self, "infected", infected)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"T_vec": self.duration, "R_vec": self.tested, "I_vec": self.infected}
        )


@dataclass(frozen=True)
class AccessPeriodExperiment:
    """A complete EpiPvr SPT or PT access-period experiment."""

    transmission_type: EpiPvrMode
    acquisition: AccessPeriodAssay
    inoculation: AccessPeriodAssay
    fixed_durations: np.ndarray
    vectors_per_plant: int
    latent: AccessPeriodAssay | None = None

    def __post_init__(self) -> None:
        mode = self.transmission_type.upper()
        if mode not in ("SPT", "PT"):
            raise ValidationError("transmission_type must be SPT or PT.")
        object.__setattr__(self, "transmission_type", mode)
        if self.vectors_per_plant <= 0:
            raise ValidationError("vectors_per_plant must be positive.")
        matrix = np.asarray(self.fixed_durations, dtype=float)
        expected_shape = (2, 2) if mode == "SPT" else (3, 3)
        if matrix.shape != expected_shape:
            raise ValidationError(
                f"fixed_durations must have shape {expected_shape} for {mode}."
            )
        if mode == "PT" and self.latent is None:
            raise ValidationError("PT experiments require a latent-period sub-assay.")
        if mode == "SPT" and self.latent is not None:
            raise ValidationError("SPT experiments must not contain a latent-period sub-assay.")
        object.__setattr__(self, "fixed_durations", matrix)

    @classmethod
    def spt(
        cls,
        *,
        acquisition: AccessPeriodAssay,
        inoculation: AccessPeriodAssay,
        fixed_inoculation_for_acquisition: float,
        fixed_acquisition_for_inoculation: float,
        vectors_per_plant: int,
    ) -> "AccessPeriodExperiment":
        matrix = np.array(
            [
                [-1.0, float(fixed_inoculation_for_acquisition)],
                [float(fixed_acquisition_for_inoculation), -1.0],
            ]
        )
        return cls(
            transmission_type="SPT",
            acquisition=acquisition,
            inoculation=inoculation,
            fixed_durations=matrix,
            vectors_per_plant=vectors_per_plant,
        )

    @classmethod
    def pt(
        cls,
        *,
        acquisition: AccessPeriodAssay,
        latent: AccessPeriodAssay,
        inoculation: AccessPeriodAssay,
        fixed_when_acquisition_varies: tuple[float, float],
        fixed_when_latency_varies: tuple[float, float],
        fixed_when_inoculation_varies: tuple[float, float],
        vectors_per_plant: int,
    ) -> "AccessPeriodExperiment":
        # Columns are AAP, LAP, IAP; rows correspond to the varying component.
        matrix = np.array(
            [
                [-1.0, fixed_when_acquisition_varies[0], fixed_when_acquisition_varies[1]],
                [fixed_when_latency_varies[0], -1.0, fixed_when_latency_varies[1]],
                [fixed_when_inoculation_varies[0], fixed_when_inoculation_varies[1], -1.0],
            ],
            dtype=float,
        )
        return cls(
            transmission_type="PT",
            acquisition=acquisition,
            latent=latent,
            inoculation=inoculation,
            fixed_durations=matrix,
            vectors_per_plant=vectors_per_plant,
        )

    def write_bundle(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.acquisition.to_dataframe().to_csv(directory / "AAP.csv", index=False)
        self.inoculation.to_dataframe().to_csv(directory / "IAP.csv", index=False)
        if self.latent is not None:
            self.latent.to_dataframe().to_csv(directory / "LAP.csv", index=False)
        pd.DataFrame(self.fixed_durations).to_csv(directory / "durations.csv", index=False, header=False)
        pd.DataFrame(
            [
                {
                    "transmission_type": self.transmission_type,
                    "vectors_per_plant": self.vectors_per_plant,
                }
            ]
        ).to_csv(directory / "metadata.csv", index=False)


@dataclass(frozen=True)
class EpiPvrFitOptions:
    """Arguments passed to EpiPvr's parameter-estimation functions."""

    survival_upper_days: float = 40.0
    d_num_pts_pd: int = 1
    warmup: int = 1000
    iterations: int = 2000
    chains: int = 4
    parallel: int = 0
    seed: int = 123

    def __post_init__(self) -> None:
        if self.survival_upper_days <= 0:
            raise ValidationError("survival_upper_days must be positive.")
        if self.d_num_pts_pd <= 0:
            raise ValidationError("d_num_pts_pd must be positive.")
        if not 0 <= self.warmup < self.iterations:
            raise ValidationError("Require 0 <= warmup < iterations.")
        if self.chains <= 0:
            raise ValidationError("chains must be positive.")
        if self.parallel < 0:
            raise ValidationError("parallel cannot be negative.")


@dataclass
class EpiPvrFit:
    """Python-side EpiPvr posterior and diagnostics."""

    transmission_type: EpiPvrMode
    posterior_hourly: pd.DataFrame
    summary_table: pd.DataFrame
    diagnostics: dict[str, str]
    bayes_r2: pd.DataFrame
    stdout: str = ""
    stderr: str = ""

    def posterior(self, *, unit: Literal["per_hour", "per_day"] = "per_day") -> pd.DataFrame:
        frame = self.posterior_hourly.copy()
        rate_columns = [
            column
            for column in (
                "acquisition_rate",
                "inoculation_rate",
                "vector_clearance_rate",
                "vector_latent_progression_rate",
            )
            if column in frame.columns
        ]
        if unit == "per_day":
            frame[rate_columns] = frame[rate_columns] * 24.0
        elif unit != "per_hour":
            raise ValidationError("unit must be 'per_hour' or 'per_day'.")
        return frame

    def parameter_summary(self, *, unit: Literal["per_hour", "per_day"] = "per_day") -> pd.DataFrame:
        posterior = self.posterior(unit=unit)
        rate_columns = [c for c in posterior.columns if c.endswith("_rate")]
        rows = []
        for column in rate_columns:
            values = posterior[column].to_numpy(float)
            rows.append(
                {
                    "parameter": column,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "q05": float(np.quantile(values, 0.05)),
                    "q95": float(np.quantile(values, 0.95)),
                    "unit": unit,
                }
            )
        return pd.DataFrame(rows)

    def median_host_transmission(self) -> HostTransmission:
        posterior = self.posterior(unit="per_day")
        return HostTransmission(
            acquisition_rate=float(posterior["acquisition_rate"].median()),
            inoculation_rate=float(posterior["inoculation_rate"].median()),
        )

    def median_pathogen_parameters(self) -> PathogenParameters:
        posterior = self.posterior(unit="per_day")
        latent = None
        if "vector_latent_progression_rate" in posterior:
            latent = float(posterior["vector_latent_progression_rate"].median())
        return PathogenParameters(
            vector_clearance_rate=float(posterior["vector_clearance_rate"].median()),
            transmission_mode=self.transmission_type,
            vector_latent_progression_rate=latent,
        )

    def convergence_report(
        self,
        *,
        max_rhat: float = 1.01,
        min_ess_per_chain: float = 100.0,
    ) -> dict[str, object]:
        virus_names = {"al[1]", "be[1]", "mu[1]", "lat[1]"}
        summary = self.summary_table
        if "variable" in summary.columns:
            virus_summary = summary[summary["variable"].astype(str).isin(virus_names)]
        else:
            virus_summary = summary
        rhat_ok = True
        ess_ok = True
        max_seen_rhat = np.nan
        min_seen_ess = np.nan
        if "rhat" in virus_summary.columns and len(virus_summary):
            values = pd.to_numeric(virus_summary["rhat"], errors="coerce").dropna()
            if len(values):
                max_seen_rhat = float(values.max())
                rhat_ok = max_seen_rhat <= max_rhat
        if "ess_bulk" in virus_summary.columns and len(virus_summary):
            values = pd.to_numeric(virus_summary["ess_bulk"], errors="coerce").dropna()
            if len(values):
                min_seen_ess = float(values.min())
                # Chain count can be inferred from posterior metadata when exported.
                chains = int(self.posterior_hourly.get("chain", pd.Series([1])).nunique())
                ess_ok = min_seen_ess >= min_ess_per_chain * max(chains, 1)

        divergent = int(float(self.diagnostics.get("divergent_transitions", "0")))
        treedepth_value = self.diagnostics.get("max_treedepth_exceeded", "FALSE").upper()
        treedepth_ok = treedepth_value not in {"TRUE", "T", "1"}
        return {
            "usable": bool(rhat_ok and ess_ok and divergent == 0 and treedepth_ok),
            "rhat_ok": rhat_ok,
            "ess_ok": ess_ok,
            "divergent_transitions": divergent,
            "treedepth_ok": treedepth_ok,
            "max_rhat": max_seen_rhat,
            "min_ess_bulk": min_seen_ess,
        }

    def require_usable(self, **kwargs) -> None:
        report = self.convergence_report(**kwargs)
        if not report["usable"]:
            raise EpiPvrError(f"EpiPvr diagnostics are not satisfactory: {report}")


@dataclass(frozen=True)
class LocalEpidemicParameters:
    """Semantic wrapper for EpiPvr's local field parameters, in day^-1."""

    dispersal_rate: float
    roguing_rate: float
    harvest_rate: float
    vector_mortality_rate: float
    plant_latent_progression_rate: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValidationError(f"{name} must be non-negative.")


@dataclass
class EpidemicProbabilityResult:
    probabilities: np.ndarray
    vectors_per_plant: int

    @property
    def from_single_infectious_plant(self) -> float:
        return float(self.probabilities[0])

    @property
    def from_single_infectious_vector(self) -> float:
        n_vars = 3 * (self.vectors_per_plant + 1) - 1
        # R is 1-indexed and the vignette uses numVars-(numInsects-1).
        r_index = n_vars - (self.vectors_per_plant - 1)
        return float(self.probabilities[r_index - 1])
