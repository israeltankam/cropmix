"""Generalized PLOS-style deterministic mean-field model for SPT mixtures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .design import MixtureDesign
from .errors import UnsupportedModelError, ValidationError
from .scenario import Scenario
from .system import CropMixSystem


@dataclass
class MeanFieldResult:
    time: np.ndarray
    latent: np.ndarray
    infectious: np.ndarray
    viruliferous_by_origin: np.ndarray
    variety_names: tuple[str, ...]
    proportions: np.ndarray
    vector_burden: int
    total_vectors: float
    final_yield: float
    yield_unit: str

    @property
    def incidence(self) -> np.ndarray:
        return self.infectious.sum(axis=0)

    @property
    def vector_prevalence(self) -> np.ndarray:
        return self.viruliferous_by_origin.sum(axis=0) / self.total_vectors

    def incidence_within_variety(self, name: str) -> np.ndarray:
        index = self.variety_names.index(name)
        theta = self.proportions[index]
        if theta <= 0:
            return np.full_like(self.time, np.nan, dtype=float)
        return self.infectious[index] / theta

    def trajectory_dataframe(self) -> pd.DataFrame:
        data: dict[str, object] = {
            "time": self.time,
            "incidence": self.incidence,
            "vector_prevalence": self.vector_prevalence,
        }
        for index, name in enumerate(self.variety_names):
            data[f"infectious_fraction_field_{name}"] = self.infectious[index]
            data[f"incidence_within_{name}"] = self.incidence_within_variety(name)
        return pd.DataFrame(data)


def solve_mean_field(
    design: MixtureDesign,
    system: CropMixSystem,
    scenario: Scenario,
    *,
    observation_times: np.ndarray | None = None,
) -> MeanFieldResult:
    """Solve the n-variety PLOS-style SPT mean-field model.

    The deterministic model preserves the PLOS convention in which plant
    latent/infectious states are fractions of the entire field and vector
    provenance states are counts.  It requires a common vector-clearance rate,
    which is represented at pathogen level in `CropMixSystem`.
    """

    if system.pathogen.transmission_mode != "SPT":
        raise UnsupportedModelError("The PLOS-style mean-field solver currently implements SPT only.")
    system.validate_design(design)

    if observation_times is None:
        observation_times = np.linspace(0.0, scenario.duration, 101)
    else:
        observation_times = np.asarray(observation_times, dtype=float)
    if np.any(np.diff(observation_times) < 0):
        raise ValidationError("observation_times must be sorted.")
    if observation_times[0] < 0 or observation_times[-1] > scenario.duration:
        raise ValidationError("observation_times must lie inside the simulation horizon.")

    names = system.variety_names
    n = len(names)
    name_to_index = {name: i for i, name in enumerate(names)}
    K = design.n_sites
    m = scenario.vectors_per_plant
    theta = np.zeros(n, dtype=float)
    for name, count in design.counts.items():
        theta[name_to_index[name]] = count / K

    alpha = np.asarray([v.transmission.acquisition_rate for v in system.varieties], dtype=float)
    beta = np.asarray([v.transmission.inoculation_rate for v in system.varieties], dtype=float)
    gamma = np.asarray([v.plant.latent_progression_rate for v in system.varieties], dtype=float)
    rho = np.asarray([v.plant.roguing_rate for v in system.varieties], dtype=float)

    sigma = system.vector.dispersal_rate
    omega = system.vector.mortality_rate
    clearance = system.pathogen.vector_clearance_rate
    q = omega + clearance
    psi = 1.0 / (sigma + q)
    F = float(m * K)

    initial_infectious = np.zeros(n, dtype=float)
    if scenario.inoculum.sites is not None:
        for site in scenario.inoculum.sites:
            if site >= K:
                raise ValidationError("Explicit inoculum site is outside the design.")
            initial_infectious[name_to_index[design.assignment[site]]] += 1.0 / K
    else:
        initial_infectious = theta * (scenario.inoculum.count / K)

    vector_inoculum = scenario.vector_inoculum
    if vector_inoculum.origin_weights is None:
        origin_weights = theta.copy()
    else:
        if len(vector_inoculum.origin_weights) != n:
            raise ValidationError(
                "vector_inoculum.origin_weights must have one value per system variety."
            )
        origin_weights = np.asarray(vector_inoculum.origin_weights, dtype=float)
        origin_weights = origin_weights / origin_weights.sum()
    initial_vectors = F * vector_inoculum.infectious_fraction * origin_weights

    y0 = np.concatenate(
        [
            np.zeros(n, dtype=float),
            initial_infectious,
            initial_vectors,
        ]
    )

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        latent = state[:n]
        infectious = state[n : 2 * n]
        vector_origin = state[2 * n :]
        vector_total = float(vector_origin.sum())
        susceptible = np.maximum(theta - latent - infectious, 0.0)

        d_latent = (
            (sigma * psi / K) * beta * susceptible * vector_total
            - gamma * latent
        )
        d_infectious = gamma * latent - rho * infectious
        d_vector = alpha * (
            F * infectious
            - psi * (sigma * infectious * vector_total + q * vector_origin)
        ) - q * vector_origin
        return np.concatenate([d_latent, d_infectious, d_vector])

    solution = solve_ivp(
        rhs,
        (0.0, scenario.duration),
        y0,
        t_eval=observation_times,
        method="LSODA",
        rtol=1e-9,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    latent = solution.y[:n]
    infectious = solution.y[n : 2 * n]
    vector_origin = solution.y[2 * n :]

    final_yield = 0.0
    for index, variety in enumerate(system.varieties):
        final_yield += (
            variety.yield_model.healthy * (theta[index] - infectious[index, -1])
            + variety.yield_model.infected * infectious[index, -1]
        )

    result = MeanFieldResult(
        time=solution.t,
        latent=latent,
        infectious=infectious,
        viruliferous_by_origin=vector_origin,
        variety_names=names,
        proportions=theta,
        vector_burden=m,
        total_vectors=F,
        final_yield=float(final_yield),
        yield_unit=system.yield_unit,
    )
    return result
