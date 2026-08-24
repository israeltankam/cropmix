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
    def mean_vector_prevalence(self) -> np.ndarray:
        return np.mean(self.vector_prevalence_runs, axis=0)

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
        data: dict[str, object] = {
            "time": self.time,
            "incidence": self.mean_incidence,
            "vector_prevalence": self.mean_vector_prevalence,
        }
        for name, values in self.incidence_by_variety_runs.items():
            data[f"incidence_{name}"] = np.mean(values, axis=0)
        return pd.DataFrame(data)

    def plot_incidence(self, ax=None, *, by_variety: bool = False):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install plotting support with `pip install cropmix[viz]`.") from exc
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.time, self.mean_incidence, label="field")
        if by_variety:
            for name, values in self.incidence_by_variety_runs.items():
                ax.plot(self.time, np.mean(values, axis=0), label=name)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Infectious plant fraction")
        ax.legend()
        return ax

    def plot_vector_prevalence(self, ax=None):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install plotting support with `pip install cropmix[viz]`.") from exc
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.time, self.mean_vector_prevalence)
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
