"""Planting-design optimization with count-preserving swap proposals."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .biology import TransmissionDraw
from .design import MixtureDesign
from .errors import ValidationError
from .geometry import Field
from .results import SimulationResult
from .scenario import Scenario
from .simulation import simulate_mixture
from .system import CropMixSystem

Objective = str | Callable[[SimulationResult], float]


@dataclass(frozen=True)
class OptimizationConfig:
    iterations: int = 500
    n_runs_per_candidate: int = 20
    final_runs: int = 500
    initial_temperature: float = 0.5
    cooling_rate: float = 0.99
    seed: int = 12345
    final_seed_offset: int = 50_000_000

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValidationError("iterations cannot be negative.")
        if self.n_runs_per_candidate <= 0 or self.final_runs <= 0:
            raise ValidationError("Simulation replication counts must be positive.")
        if self.initial_temperature <= 0:
            raise ValidationError("initial_temperature must be positive.")
        if not 0 < self.cooling_rate <= 1:
            raise ValidationError("cooling_rate must lie in (0, 1].")


@dataclass
class OptimizationResult:
    best_design: MixtureDesign
    best_score: float
    final_result: SimulationResult
    history: pd.DataFrame
    objective_name: str
    variety_counts: dict[str, int]
    evaluations: int

    def summary(self) -> pd.DataFrame:
        frame = self.final_result.summary().copy()
        frame.insert(0, "objective", self.objective_name)
        frame.insert(1, "optimization_score", self.best_score)
        frame.insert(2, "evaluations", self.evaluations)
        return frame


def _objective_value(result: SimulationResult, objective: Objective) -> tuple[float, str]:
    if callable(objective):
        return float(objective(result)), getattr(objective, "__name__", "custom")
    if objective == "expected_yield":
        return result.mean_yield, objective
    if objective == "min_final_incidence":
        return -result.mean_final_incidence, objective
    if objective == "yield_stability":
        return result.mean_yield - result.yield_sd, objective
    raise ValidationError(
        "Unknown objective. Use 'expected_yield', 'min_final_incidence', "
        "'yield_stability', or provide a callable."
    )


def _design_key(design: MixtureDesign) -> tuple[str, ...]:
    return design.assignment


def _random_heterotypic_pair(rng: np.random.Generator, assignment: tuple[str, ...]) -> tuple[int, int] | None:
    if len(set(assignment)) < 2:
        return None
    n = len(assignment)
    first = int(rng.integers(0, n))
    candidates = [index for index, value in enumerate(assignment) if value != assignment[first]]
    second = int(candidates[int(rng.integers(0, len(candidates)))])
    return first, second


def optimize_mixture(
    field: Field,
    variety_counts: Mapping[str, int],
    system: CropMixSystem,
    scenario: Scenario,
    *,
    objective: Objective = "expected_yield",
    config: OptimizationConfig | None = None,
    initial_design: MixtureDesign | None = None,
    transmission_draw: TransmissionDraw | None = None,
) -> OptimizationResult:
    """Search for a high-performing planting design with fixed variety counts.

    Cropmix 0.1 uses swap-based simulated annealing.  A swap proposal preserves
    the requested counts exactly. Candidate designs are evaluated with the same
    Monte Carlo seed block (common random numbers), reducing simulation noise in
    pairwise design comparisons.

    The optimizer is heuristic: combinatorial design spaces become enormous
    even for modest fields, so global optimality is not claimed.
    """

    config = OptimizationConfig() if config is None else config
    counts = {str(name): int(value) for name, value in variety_counts.items()}
    if any(value < 0 for value in counts.values()):
        raise ValidationError("Variety counts cannot be negative.")
    if sum(counts.values()) != field.n_sites:
        raise ValidationError("Variety counts must sum to the number of field sites.")
    unknown = set(counts) - set(system.variety_names)
    if unknown:
        raise ValidationError(f"Counts contain varieties absent from system: {sorted(unknown)}")

    if initial_design is None:
        current = MixtureDesign.random(field, counts, seed=config.seed)
    else:
        if initial_design.field is not field and not np.array_equal(
            initial_design.field.coordinates, field.coordinates
        ):
            raise ValidationError("initial_design does not use the supplied field geometry.")
        if initial_design.counts != counts:
            raise ValidationError("initial_design counts do not match variety_counts.")
        current = initial_design

    rng = np.random.default_rng(config.seed)
    cache: dict[tuple[str, ...], tuple[float, SimulationResult]] = {}
    history_rows: list[dict[str, object]] = []

    def evaluate(design: MixtureDesign) -> tuple[float, SimulationResult, str]:
        key = _design_key(design)
        if key not in cache:
            result = simulate_mixture(
                design,
                system,
                scenario,
                n_runs=config.n_runs_per_candidate,
                seed=config.seed + 1_000_000,
                transmission_draw=transmission_draw,
                store_final_states=False,
            )
            score, objective_name = _objective_value(result, objective)
            cache[key] = (score, result)
        else:
            score, result = cache[key]
            objective_name = objective if isinstance(objective, str) else getattr(objective, "__name__", "custom")
        return score, result, str(objective_name)

    current_score, _, objective_name = evaluate(current)
    best = current
    best_score = current_score

    history_rows.append(
        {
            "iteration": 0,
            "current_score": current_score,
            "best_score": best_score,
            "accepted": True,
            "temperature": config.initial_temperature,
        }
    )

    for iteration in range(1, config.iterations + 1):
        pair = _random_heterotypic_pair(rng, current.assignment)
        if pair is None:
            break
        candidate = current.swapped(*pair)
        candidate_score, _, _ = evaluate(candidate)
        temperature = config.initial_temperature * (config.cooling_rate ** (iteration - 1))
        delta = candidate_score - current_score
        accept = delta >= 0 or rng.random() < exp(delta / max(temperature, 1e-15))

        if accept:
            current = candidate
            current_score = candidate_score

        if current_score > best_score:
            best = current
            best_score = current_score

        history_rows.append(
            {
                "iteration": iteration,
                "current_score": current_score,
                "best_score": best_score,
                "accepted": bool(accept),
                "temperature": temperature,
            }
        )

    final_result = simulate_mixture(
        best,
        system,
        scenario,
        n_runs=config.final_runs,
        seed=config.seed + config.final_seed_offset,
        transmission_draw=transmission_draw,
        store_final_states=True,
    )

    return OptimizationResult(
        best_design=best,
        best_score=float(best_score),
        final_result=final_result,
        history=pd.DataFrame(history_rows),
        objective_name=objective_name,
        variety_counts=counts,
        evaluations=len(cache),
    )
