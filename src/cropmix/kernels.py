"""Distance kernels and finite-field balancing."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .errors import ValidationError
from .geometry import Field


@dataclass(frozen=True)
class PreparedKernel:
    """A finite-field movement matrix and its diagnostics."""

    scale: float
    probabilities: np.ndarray
    distances: np.ndarray
    cdf: np.ndarray
    mean_step_distance: float
    max_row_error: float
    max_symmetry_error: float

    def __post_init__(self) -> None:
        for array_name in ("probabilities", "distances", "cdf"):
            array = np.asarray(getattr(self, array_name), dtype=float)
            array.setflags(write=False)
            object.__setattr__(self, array_name, array)


@dataclass(frozen=True)
class ExponentialKernel:
    """Exponential spatial kernel ``K(d) = exp(-d / scale)``.

    The raw distance weights are symmetrically balanced to produce a matrix
    that is both row-stochastic and reciprocal on a finite open field.  This is
    required by Cropmix's conservative pair-exchange movement generator.
    """

    scale: float = 1.0
    balance_tolerance: float = 1e-11
    balance_max_iter: int = 20_000

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValidationError("Kernel scale must be positive.")
        if self.balance_tolerance <= 0:
            raise ValidationError("balance_tolerance must be positive.")
        if self.balance_max_iter <= 0:
            raise ValidationError("balance_max_iter must be positive.")

    def with_scale(self, scale: float) -> "ExponentialKernel":
        return replace(self, scale=float(scale))

    def prepare(self, field: Field) -> PreparedKernel:
        distances = field.distance_matrix()
        weights = np.exp(-distances / self.scale)
        np.fill_diagonal(weights, 0.0)
        probabilities = symmetric_sinkhorn(
            weights,
            tol=self.balance_tolerance,
            max_iter=self.balance_max_iter,
        )
        row_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
        symmetry_error = float(np.max(np.abs(probabilities - probabilities.T)))
        mean_distance = float(np.mean(np.sum(probabilities * distances, axis=1)))
        cdf = np.cumsum(probabilities, axis=1)
        cdf[:, -1] = 1.0
        return PreparedKernel(
            scale=float(self.scale),
            probabilities=probabilities,
            distances=distances,
            cdf=cdf,
            mean_step_distance=mean_distance,
            max_row_error=row_error,
            max_symmetry_error=symmetry_error,
        )


def symmetric_sinkhorn(
    weights: np.ndarray,
    *,
    tol: float = 1e-11,
    max_iter: int = 20_000,
) -> np.ndarray:
    """Balance a symmetric non-negative matrix to a symmetric doubly stochastic matrix.

    Alternating row/column Sinkhorn scaling is used.  The exponential complete
    graph used by Cropmix has total support, so convergence is expected.  The
    final matrix is averaged with its transpose only at floating-point scale.
    """

    matrix = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError("weights must be square.")
    if np.any(matrix < 0) or not np.isfinite(matrix).all():
        raise ValidationError("weights must be finite and non-negative.")
    if np.any(matrix.sum(axis=1) <= 0):
        raise ValidationError("Every site must have at least one movement destination.")

    n = matrix.shape[0]
    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    for iteration in range(max_iter):
        row_scale = 1.0 / (matrix @ col_scale)
        col_scale = 1.0 / (matrix.T @ row_scale)

        if iteration % 10 == 0:
            balanced = (row_scale[:, None] * matrix) * col_scale[None, :]
            error = max(
                float(np.max(np.abs(balanced.sum(axis=1) - 1.0))),
                float(np.max(np.abs(balanced.sum(axis=0) - 1.0))),
            )
            if error < tol:
                break
    else:
        raise RuntimeError("Symmetric Sinkhorn balancing did not converge.")

    balanced = (row_scale[:, None] * matrix) * col_scale[None, :]
    symmetry_error = float(np.max(np.abs(balanced - balanced.T)))
    if symmetry_error > 1e-8:
        raise RuntimeError(
            f"Balanced kernel is unexpectedly asymmetric ({symmetry_error:.3e})."
        )
    balanced = 0.5 * (balanced + balanced.T)
    np.fill_diagonal(balanced, 0.0)

    # The averaging step should only change machine precision. If it changed
    # row sums more materially, rebalance once more rather than silently
    # returning a matrix that violates the movement derivation.
    if np.max(np.abs(balanced.sum(axis=1) - 1.0)) > 10 * tol:
        return symmetric_sinkhorn(balanced, tol=tol, max_iter=max_iter)
    return balanced
