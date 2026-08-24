# EpiPvr bridge

Cropmix keeps the user-facing workflow in Python. R and EpiPvr remain behind a subprocess bridge.

## External installation

Install R, then once in R:

```r
install.packages("EpiPvr")
```

Check from the shell:

```bash
cropmix doctor
```

## SPT access-period experiment

```python
from cropmix.epipvr import (
    AccessPeriodAssay,
    AccessPeriodExperiment,
    EpiPvrBackend,
)

aap = AccessPeriodAssay(
    duration=(2, 3, 4, 5),
    tested=(30, 30, 30, 30),
    infected=(10, 15, 18, 20),
)

iap = AccessPeriodAssay(
    duration=(0.25, 0.5, 0.75, 1.0),
    tested=(30, 30, 30, 30),
    infected=(5, 10, 15, 20),
)

experiment = AccessPeriodExperiment.spt(
    acquisition=aap,
    inoculation=iap,
    fixed_inoculation_for_acquisition=6,
    fixed_acquisition_for_inoculation=4,
    vectors_per_plant=20,
)

fit = EpiPvrBackend().fit(experiment)
```

The bridge maps the Python experiment to EpiPvr's `d_AAP`, `d_IAP`, `d_durations`, `d_vectorspp`, and `d_virusType` structure.

For PT, it additionally supplies `d_LAP` and a 3x3 duration matrix.

## Output

`EpiPvrFit` exposes:

```python
fit.posterior(unit="per_day")
fit.parameter_summary(unit="per_day")
fit.convergence_report()
fit.median_host_transmission()
fit.median_pathogen_parameters()
```

Posterior variables are kept jointly by draw. Cropmix does not independently resample marginal `alpha`, `beta`, and `mu` distributions.

The bridge also exposes EpiPvr's branching-process epidemic-probability function through `EpiPvrBackend.epidemic_probability()`.

## Diagnostics

EpiPvr itself emphasizes assessing model fit before reporting or propagating parameter estimates. Cropmix exports the EpiPvr summary table, Bayesian R2 values, divergent-transition count, and tree-depth diagnostic.

Use `fit.require_usable()` to enforce a conservative diagnostic gate before parameter propagation.

## Important model boundary

PT parameter inference does not imply PT spatial simulation. Cropmix 0.1 raises an explicit error if a PT `CropMixSystem` is passed to the spatial engine.
