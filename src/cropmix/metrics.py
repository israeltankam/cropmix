"""Geometry-agnostic arrangement metrics with explicit definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .design import MixtureDesign


def pairwise_mixing_metrics(
    design: MixtureDesign,
    *,
    radius: float,
) -> pd.Series:
    """Simple local mixing descriptors for arbitrary coordinate fields.

    Two sites are neighbours when their Euclidean separation is at most
    `radius`.  This explicit distance definition avoids assuming a square-grid
    Moore or rook neighbourhood on irregular fields.
    """

    if radius <= 0:
        raise ValueError("radius must be positive.")
    distances = design.field.distance_matrix()
    labels = np.asarray(design.assignment, dtype=object)
    n = design.n_sites

    neighbour_mask = (distances > 0) & (distances <= radius)
    heterotypic = labels[:, None] != labels[None, :]
    pair_mask = np.triu(neighbour_mask, k=1)
    n_pairs = int(pair_mask.sum())
    heterotypic_pairs = int(np.sum(pair_mask & heterotypic))

    local_fractions = []
    for i in range(n):
        neighbours = np.flatnonzero(neighbour_mask[i])
        if len(neighbours):
            local_fractions.append(float(np.mean(labels[neighbours] != labels[i])))

    return pd.Series(
        {
            "neighbour_radius": radius,
            "n_neighbour_pairs": n_pairs,
            "heterotypic_pair_fraction": (
                heterotypic_pairs / n_pairs if n_pairs else np.nan
            ),
            "mean_local_heterospecific_fraction": (
                float(np.mean(local_fractions)) if local_fractions else np.nan
            ),
        }
    )
