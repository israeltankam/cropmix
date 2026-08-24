"""Mean-field consistency diagnostics across spatial movement scales.

This module does not estimate a biological dispersal scale. It identifies the
movement regime in which a finite spatial process becomes practically
indistinguishable from the corresponding non-spatial mean-field reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .design import MixtureDesign
from .errors import ValidationError
from .geometry import Field
from .meanfield import solve_mean_field
from .scenario import Scenario
from .simulation import simulate_mixture
from .system import CropMixSystem


@dataclass
class MeanFieldConsistencyResult:
    """Result of a multi-context mean-field consistency assessment."""

    selected_scale: float
    literal_scale: float
    minimax_scale: float
    acceptable_min: float
    acceptable_max: float
    mean_step_distance: float
    context_table: pd.DataFrame
    joint_table: pd.DataFrame
    context_optima: pd.DataFrame
    leave_one_context_out: pd.DataFrame
    reference_variety: str
    score_weights: tuple[float, float]

    @property
    def acceptable_log10_width(self) -> float:
        if self.acceptable_min <= 0 or self.acceptable_max <= 0:
            return float("nan")
        return float(np.log10(self.acceptable_max / self.acceptable_min))

    @property
    def boundary_warning(self) -> bool:
        values = self.joint_table["scale"].to_numpy(float)
        return bool(np.isclose(self.literal_scale, values.min()) or np.isclose(self.literal_scale, values.max()))

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "reference_variety": self.reference_variety,
                    "literal_scale": self.literal_scale,
                    "selected_scale": self.selected_scale,
                    "minimax_scale": self.minimax_scale,
                    "acceptable_min": self.acceptable_min,
                    "acceptable_max": self.acceptable_max,
                    "acceptable_log10_width": self.acceptable_log10_width,
                    "mean_step_distance": self.mean_step_distance,
                    "boundary_warning": self.boundary_warning,
                }
            ]
        )

    def plot(self, ax=None):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install plotting support with `pip install cropmix[viz]`.") from exc
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.joint_table["scale"], self.joint_table["mean_score"], label="mean context score")
        ax.plot(self.joint_table["scale"], self.joint_table["max_score"], label="worst context score")
        ax.axvspan(self.acceptable_min, self.acceptable_max, alpha=0.15, label="one-SE region")
        ax.axvline(self.selected_scale, linestyle=":", label=f"selected={self.selected_scale:.3g}")
        ax.set_xscale("log")
        ax.set_xlabel("Exponential kernel scale")
        ax.set_ylabel("Dynamic mean-field discrepancy")
        ax.legend()
        return ax


def _trajectory_score(
    spatial_incidence: np.ndarray,
    spatial_vectors: np.ndarray,
    meanfield_incidence: np.ndarray,
    meanfield_vectors: np.ndarray,
    weights: tuple[float, float],
) -> tuple[float, float, float]:
    incidence_rmse = float(np.sqrt(np.mean((spatial_incidence - meanfield_incidence) ** 2)))
    vector_rmse = float(np.sqrt(np.mean((spatial_vectors - meanfield_vectors) ** 2)))
    score = weights[0] * incidence_rmse + weights[1] * vector_rmse
    return score, incidence_rmse, vector_rmse


def _bootstrap_score_se(
    incidence_runs: np.ndarray,
    vector_runs: np.ndarray,
    meanfield_incidence: np.ndarray,
    meanfield_vectors: np.ndarray,
    weights: tuple[float, float],
    *,
    n_bootstrap: int,
    seed: int,
) -> float:
    if incidence_runs.shape[0] < 2 or n_bootstrap <= 1:
        return 0.0
    rng = np.random.default_rng(seed)
    n_runs = incidence_runs.shape[0]
    values = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sample = rng.integers(0, n_runs, size=n_runs)
        values[b] = _trajectory_score(
            incidence_runs[sample].mean(axis=0),
            vector_runs[sample].mean(axis=0),
            meanfield_incidence,
            meanfield_vectors,
            weights,
        )[0]
    return float(np.std(values, ddof=1))


def _aggregate_joint(context_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale, group in context_table.groupby("scale", sort=True):
        ses = group["score_se"].to_numpy(float)
        joint_se = float(np.sqrt(np.sum(ses**2)) / len(group))
        rows.append(
            {
                "scale": float(scale),
                "mean_step_distance": float(group["mean_step_distance"].iloc[0]),
                "mean_score": float(group["score"].mean()),
                "max_score": float(group["score"].max()),
                "mean_score_se": joint_se,
                "mean_relative_yield_error": float(group["relative_yield_error"].mean()),
                "max_relative_yield_error": float(group["relative_yield_error"].max()),
                "n_contexts": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("scale").reset_index(drop=True)


def _select_one_se(joint: pd.DataFrame) -> tuple[float, float, float, float]:
    best_row = joint.loc[joint["mean_score"].idxmin()]
    literal = float(best_row["scale"])
    threshold = float(best_row["mean_score"] + best_row["mean_score_se"])
    acceptable = joint[joint["mean_score"] <= threshold]
    selected = float(acceptable.iloc[0]["scale"])
    return selected, literal, float(acceptable["scale"].min()), float(acceptable["scale"].max())


def assess_mean_field_consistency(
    field: Field,
    system: CropMixSystem,
    scenarios: Sequence[Scenario],
    *,
    reference_variety: str,
    scales: np.ndarray | None = None,
    n_runs: int = 50,
    observation_times: np.ndarray | None = None,
    incidence_weight: float = 0.5,
    vector_weight: float = 0.5,
    bootstrap_reps: int = 100,
    seed: int = 20260810,
) -> MeanFieldConsistencyResult:
    """Assess when the spatial model approaches the mean-field regime.

    This function deliberately does *not* infer a biological dispersal scale.
    It compares complete plant-incidence and virus-bearing-vector prevalence
    trajectories with the finite-size PLOS-style mean-field reference across a
    supplied scale grid. The returned one-SE scale is a practical threshold for
    mean-field adequacy under the tested contexts. Biological analyses should
    use an independently justified movement scale (literature, tracking data or
    spatial epidemic data). Final yield is recorded only as an external
    diagnostic and is not part of the discrepancy score.
    """

    if not scenarios:
        raise ValidationError("At least one calibration scenario is required.")
    if reference_variety not in system.variety_names:
        raise ValidationError(f"Unknown reference_variety {reference_variety!r}.")
    if scales is None:
        scales = np.geomspace(0.05, 100.0, 100)
    scales = np.asarray(scales, dtype=float)
    if scales.ndim != 1 or len(scales) < 2 or np.any(scales <= 0):
        raise ValidationError("scales must be a 1D array of at least two positive values.")
    if n_runs <= 0:
        raise ValidationError("n_runs must be positive.")
    if incidence_weight < 0 or vector_weight < 0 or incidence_weight + vector_weight <= 0:
        raise ValidationError("Calibration weights must be non-negative and not both zero.")
    total_weight = incidence_weight + vector_weight
    weights = (incidence_weight / total_weight, vector_weight / total_weight)

    design = MixtureDesign.monoculture(field, reference_variety)
    records: list[dict[str, object]] = []

    for context_index, scenario in enumerate(scenarios):
        if observation_times is None:
            times = np.linspace(0.0, scenario.duration, 101)
        else:
            times = np.asarray(observation_times, dtype=float)
            if times[-1] > scenario.duration:
                raise ValidationError("observation_times exceed a calibration scenario duration.")

        meanfield = solve_mean_field(design, system, scenario, observation_times=times)
        context_seed = seed + context_index * 1_000_000

        for scale_index, scale in enumerate(scales):
            candidate_system = system.with_kernel_scale(float(scale))
            spatial = simulate_mixture(
                design,
                candidate_system,
                scenario,
                n_runs=n_runs,
                seed=context_seed,
                observation_times=times,
                store_final_states=False,
            )
            score, incidence_rmse, vector_rmse = _trajectory_score(
                spatial.mean_incidence,
                spatial.mean_vector_prevalence,
                meanfield.incidence,
                meanfield.vector_prevalence,
                weights,
            )
            score_se = _bootstrap_score_se(
                spatial.incidence_runs,
                spatial.vector_prevalence_runs,
                meanfield.incidence,
                meanfield.vector_prevalence,
                weights,
                n_bootstrap=bootstrap_reps,
                seed=seed + 50_000_000 + context_index * 100_000 + scale_index,
            )
            yield_error = abs(spatial.mean_yield - meanfield.final_yield)
            records.append(
                {
                    "context": context_index,
                    "vectors_per_plant": scenario.vectors_per_plant,
                    "initial_infectious": scenario.inoculum.count,
                    "duration": scenario.duration,
                    "scale": float(scale),
                    "mean_step_distance": candidate_system.kernel.prepare(field).mean_step_distance,
                    "score": score,
                    "score_se": score_se,
                    "incidence_rmse": incidence_rmse,
                    "vector_prevalence_rmse": vector_rmse,
                    "spatial_final_yield": spatial.mean_yield,
                    "meanfield_final_yield": meanfield.final_yield,
                    "absolute_yield_error": yield_error,
                    "relative_yield_error": yield_error / max(abs(meanfield.final_yield), 1e-12),
                    "spatial_final_incidence": spatial.mean_final_incidence,
                    "meanfield_final_incidence": float(meanfield.incidence[-1]),
                }
            )

    context_table = pd.DataFrame(records)
    joint = _aggregate_joint(context_table)
    selected, literal, acceptable_min, acceptable_max = _select_one_se(joint)
    minimax = float(joint.loc[joint["max_score"].idxmin(), "scale"])
    selected_distance = float(
        joint.loc[np.isclose(joint["scale"], selected), "mean_step_distance"].iloc[0]
    )

    context_optima = (
        context_table.loc[context_table.groupby("context")["score"].idxmin()]
        [["context", "vectors_per_plant", "initial_infectious", "scale", "score"]]
        .rename(columns={"scale": "context_literal_scale", "score": "context_min_score"})
        .reset_index(drop=True)
    )

    loco_rows = []
    context_ids = sorted(context_table["context"].unique())
    if len(context_ids) > 1:
        for omitted in context_ids:
            subset = context_table[context_table["context"] != omitted]
            loco_joint = _aggregate_joint(subset)
            loco_selected, loco_literal, loco_min, loco_max = _select_one_se(loco_joint)
            loco_rows.append(
                {
                    "omitted_context": omitted,
                    "selected_scale": loco_selected,
                    "literal_scale": loco_literal,
                    "acceptable_min": loco_min,
                    "acceptable_max": loco_max,
                }
            )
    leave_one_out = pd.DataFrame(loco_rows)

    return MeanFieldConsistencyResult(
        selected_scale=selected,
        literal_scale=literal,
        minimax_scale=minimax,
        acceptable_min=acceptable_min,
        acceptable_max=acceptable_max,
        mean_step_distance=selected_distance,
        context_table=context_table,
        joint_table=joint,
        context_optima=context_optima,
        leave_one_context_out=leave_one_out,
        reference_variety=reference_variety,
        score_weights=weights,
    )


# Backward compatibility ----------------------------------------------------
KernelCalibrationResult = MeanFieldConsistencyResult

def calibrate_kernel(*args, **kwargs):
    """Deprecated compatibility alias for :func:`assess_mean_field_consistency`.

    The historical name ``calibrate_kernel`` suggested biological parameter
    estimation. Cropmix 0.2 retains it only so older scripts keep working.
    """
    import warnings
    warnings.warn(
        "calibrate_kernel() is deprecated: mean-field matching does not estimate "
        "a biological movement scale. Use assess_mean_field_consistency().",
        DeprecationWarning,
        stacklevel=2,
    )
    return assess_mean_field_consistency(*args, **kwargs)
