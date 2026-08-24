"""Structured simulation results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .design import MixtureDesign
from .scenario import Scenario


@dataclass
class SimulationResult:
    """Monte Carlo output for one supplied planting design."""

    design: MixtureDesign
    scenario: Scenario
    time: np.ndarray
    yield_runs: np.ndarray
    final_incidence_runs: np.ndarray
    incidence_runs: np.ndarray
    vector_prevalence_runs: np.ndarray
    incidence_by_variety_runs: dict[str, np.ndarray]
    final_states: np.ndarray | None
    infection_probability: np.ndarray | None
    kernel_scale: float
    mean_step_distance: float
    seed: int
    yield_unit: str

    @property
    def mean_yield(self) -> float:
        return float(np.mean(self.yield_runs))

    @property
    def yield_sd(self) -> float:
        return float(np.std(self.yield_runs, ddof=1)) if len(self.yield_runs) > 1 else 0.0

    @property
    def yield_se(self) -> float:
        return self.yield_sd / np.sqrt(len(self.yield_runs)) if len(self.yield_runs) else float("nan")

    @property
    def mean_final_incidence(self) -> float:
        return float(np.mean(self.final_incidence_runs))

    @property
    def final_incidence_sd(self) -> float:
        return (
            float(np.std(self.final_incidence_runs, ddof=1))
            if len(self.final_incidence_runs) > 1
            else 0.0
        )

    @property
    def mean_incidence(self) -> np.ndarray:
        return np.mean(self.incidence_runs, axis=0)

    @property
    def incidence_sd(self) -> np.ndarray:
        """Pointwise standard deviation of stochastic incidence trajectories."""
        if self.incidence_runs.shape[0] <= 1:
            return np.zeros_like(self.mean_incidence)
        return np.std(self.incidence_runs, axis=0, ddof=1)

    @property
    def mean_vector_prevalence(self) -> np.ndarray:
        return np.mean(self.vector_prevalence_runs, axis=0)

    @property
    def vector_prevalence_sd(self) -> np.ndarray:
        """Pointwise standard deviation of stochastic vector-prevalence trajectories."""
        if self.vector_prevalence_runs.shape[0] <= 1:
            return np.zeros_like(self.mean_vector_prevalence)
        return np.std(self.vector_prevalence_runs, axis=0, ddof=1)

    @property
    def incidence_by_variety_sd(self) -> dict[str, np.ndarray]:
        """Pointwise standard deviations for each cultivar-specific incidence trajectory."""
        output: dict[str, np.ndarray] = {}
        for name, values in self.incidence_by_variety_runs.items():
            if np.isnan(values).all():
                output[name] = np.full(values.shape[1], np.nan, dtype=float)
            elif values.shape[0] <= 1:
                output[name] = np.zeros(values.shape[1], dtype=float)
            else:
                output[name] = np.nanstd(values, axis=0, ddof=1)
        return output

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "n_runs": len(self.yield_runs),
                    "mean_yield": self.mean_yield,
                    "yield_sd": self.yield_sd,
                    "yield_se": self.yield_se,
                    "yield_unit": self.yield_unit,
                    "mean_final_incidence": self.mean_final_incidence,
                    "final_incidence_sd": self.final_incidence_sd,
                    "kernel_scale": self.kernel_scale,
                    "mean_step_distance": self.mean_step_distance,
                }
            ]
        )

    def trajectory_dataframe(self) -> pd.DataFrame:
        incidence_mean = self.mean_incidence
        incidence_sd = self.incidence_sd
        vector_mean = self.mean_vector_prevalence
        vector_sd = self.vector_prevalence_sd
        data: dict[str, object] = {
            "time": self.time,
            "incidence": incidence_mean,
            "incidence_sd": incidence_sd,
            "incidence_lower_1sd": np.clip(incidence_mean - incidence_sd, 0.0, 1.0),
            "incidence_upper_1sd": np.clip(incidence_mean + incidence_sd, 0.0, 1.0),
            "vector_prevalence": vector_mean,
            "vector_prevalence_sd": vector_sd,
            "vector_prevalence_lower_1sd": np.clip(vector_mean - vector_sd, 0.0, 1.0),
            "vector_prevalence_upper_1sd": np.clip(vector_mean + vector_sd, 0.0, 1.0),
        }
        by_variety_sd = self.incidence_by_variety_sd
        for name, values in self.incidence_by_variety_runs.items():
            mean = (
                np.full(values.shape[1], np.nan, dtype=float)
                if np.isnan(values).all()
                else np.nanmean(values, axis=0)
            )
            sd = by_variety_sd[name]
            data[f"incidence_{name}"] = mean
            data[f"incidence_{name}_sd"] = sd
            data[f"incidence_{name}_lower_1sd"] = np.clip(mean - sd, 0.0, 1.0)
            data[f"incidence_{name}_upper_1sd"] = np.clip(mean + sd, 0.0, 1.0)
        return pd.DataFrame(data)

    def plot_incidence(
        self,
        ax=None,
        *,
        by_variety: bool = False,
        show_sd: bool = True,
        sd_multiplier: float = 1.0,
        envelope_alpha: float = 0.2,
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install plotting support with `pip install cropmix[viz]`.") from exc
        if ax is None:
            _, ax = plt.subplots()
        field_mean = self.mean_incidence
        field_sd = self.incidence_sd
        field_line = ax.plot(self.time, field_mean, label="field")[0]
        if show_sd:
            ax.fill_between(
                self.time,
                np.clip(field_mean - sd_multiplier * field_sd, 0.0, 1.0),
                np.clip(field_mean + sd_multiplier * field_sd, 0.0, 1.0),
                alpha=envelope_alpha,
                color=field_line.get_color(),
                linewidth=0,
            )
        if by_variety:
            by_variety_sd = self.incidence_by_variety_sd
            for name, values in self.incidence_by_variety_runs.items():
                mean = (
                    np.full(values.shape[1], np.nan, dtype=float)
                    if np.isnan(values).all()
                    else np.nanmean(values, axis=0)
                )
                sd = by_variety_sd[name]
                line = ax.plot(self.time, mean, label=name)[0]
                if show_sd:
                    ax.fill_between(
                        self.time,
                        np.clip(mean - sd_multiplier * sd, 0.0, 1.0),
                        np.clip(mean + sd_multiplier * sd, 0.0, 1.0),
                        alpha=envelope_alpha,
                        color=line.get_color(),
                        linewidth=0,
                    )
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Infectious plant fraction")
        ax.legend()
        return ax

    def plot_vector_prevalence(
        self,
        ax=None,
        *,
        show_sd: bool = True,
        sd_multiplier: float = 1.0,
        envelope_alpha: float = 0.2,
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install plotting support with `pip install cropmix[viz]`.") from exc
        if ax is None:
            _, ax = plt.subplots()
        mean = self.mean_vector_prevalence
        sd = self.vector_prevalence_sd
        line = ax.plot(self.time, mean)[0]
        if show_sd:
            ax.fill_between(
                self.time,
                np.clip(mean - sd_multiplier * sd, 0.0, 1.0),
                np.clip(mean + sd_multiplier * sd, 0.0, 1.0),
                alpha=envelope_alpha,
                color=line.get_color(),
                linewidth=0,
            )
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Virus-bearing vector prevalence")
        return ax

    def plot_final_infection_probability(self, ax=None, *, marker_size: float = 100):
        if self.infection_probability is None:
            raise ValueError("Final states were not stored; infection probability is unavailable.")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install plotting support with `pip install cropmix[viz]`.") from exc
        if ax is None:
            _, ax = plt.subplots()
        scatter = ax.scatter(
            self.design.field.x,
            self.design.field.y,
            c=self.infection_probability,
            s=marker_size,
            vmin=0,
            vmax=1,
        )
        plt.colorbar(scatter, ax=ax, label="P(infectious at T)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return ax
