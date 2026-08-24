# Cropmix

**Cropmix** is a Python package for designing, simulating, calibrating, and optimizing spatial crop varietal mixtures under vector-borne disease pressure.

It is built around a simple principle: **field geometry, planting design, host biology, vector biology, pathogen transmission, and epidemic scenario are separate objects**. This keeps the public API manageable even when fields are irregular, mixtures contain more than two varieties, or transmission parameters come from laboratory inference.

## Scientific hierarchy

Cropmix is designed to connect three modelling layers without conflating them:

1. **EpiPvr Bayesian inference** estimates transmission parameters from access-period experiments.
2. **EpiPvr branching processes** use those parameters to infer early epidemic establishment risk in a local field model.
3. **Cropmix Gillespie simulation** uses biological parameters in a finite, spatially explicit continuous-time Markov chain to simulate full epidemic trajectories and planting-design effects.

The deterministic PLOS mixture model is retained as a **mean-field reference** for consistency checks and kernel-scale calibration.

The full long-term predictive target is 

$$ p(\mathcal O\mid D,z) =
\int p_{\mathrm{CTMC}}(\mathcal O\mid z,\Theta)
       p_{\mathrm{EpiPvr}}(\Theta\mid D)\,d\Theta,
$$

where `D` is access-period data, `Theta` is a coherent set of transmission parameters, and `z` is a planting design.

## Current alpha scope

Cropmix 0.2 provides:

- arbitrary 2D fields defined by planting coordinates;
- rectangles, masks, polygon-clipped planting lattices, or directly supplied coordinates;
- arbitrary numbers of crop varieties in the **SPT spatial engine**;
- conservative distance-weighted vector exchange with exactly fixed vector burden per plant;
- exponential movement kernels with finite-field symmetric balancing;
- Numba-accelerated stochastic Gillespie simulation;
- generalized PLOS-style SPT mean-field dynamics;
- multi-context kernel-scale calibration against mean-field trajectories;
- count-preserving simulated-annealing optimization of planting arrangements;
- a Python-only user interface to **EpiPvr through an internal Rscript bridge**;
- EpiPvr SPT and PT parameter inference through that bridge;
- EpiPvr branching-process epidemic-probability calls through the same bridge.

**Important:** PT parameters can be inferred through EpiPvr, but Cropmix 0.2 does **not** silently simulate PT spatial dynamics. A PT spatial engine needs an explicit exposed-vector compartment and will be added as a separately validated model.

## Installation

Once published to PyPI:

```bash
python -m pip install cropmix
```

With plotting and development extras:

```bash
python -m pip install "cropmix[viz]"
python -m pip install "cropmix[dev,viz,docs]"
```

From a cloned repository during development:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/cropmix.git
cd cropmix
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev,viz,docs]"
pytest
```

## 60-second example

```python
import cropmix as cm

field = cm.Field.rectangular(rows=10, columns=10, spacing=1.0)

susc = cm.Variety(
    name="SUSC",
    transmission=cm.HostTransmission(
        acquisition_rate=15.31,
        inoculation_rate=1.34,
    ),
    plant=cm.PlantParameters(
        latent_progression_rate=1 / 30,
        roguing_rate=0.0,
    ),
    yield_model=cm.YieldParameters(
        healthy=31.0,
        infected=3.1,
        unit="t/ha",
    ),
)

res = cm.Variety(
    name="RES",
    transmission=cm.HostTransmission(4.84, 0.42),
    plant=cm.PlantParameters(1 / 30, 0.0),
    yield_model=cm.YieldParameters(25.0, 2.1, unit="t/ha"),
)

system = cm.CropMixSystem(
    varieties=(susc, res),
    vector=cm.VectorParameters(
        mortality_rate=0.19,
        dispersal_rate=0.45,
    ),
    pathogen=cm.PathogenParameters(
        vector_clearance_rate=19.37,
        transmission_mode="SPT",
    ),
    kernel=cm.ExponentialKernel(scale=1.0),
)

design = cm.MixtureDesign.random(
    field,
    counts={"SUSC": 50, "RES": 50},
    seed=1,
)

scenario = cm.Scenario(
    duration=360,
    vectors_per_plant=10,
    inoculum=cm.Inoculum.random(count=1),
)

result = cm.simulate_mixture(
    design,
    system,
    scenario,
    n_runs=20,
    seed=42,
)

print(result.summary())
```

The two main high-level operations are intentionally simple:

```python
cm.simulate_mixture(...)
cm.optimize_mixture(...)
```

The complexity lives in typed model objects rather than giant function signatures.

## Arbitrary field geometry

A field is fundamentally a set of plant coordinates, not a square matrix.

```python
field = cm.Field.from_coordinates([
    (0.0, 0.0),
    (1.0, 0.0),
    (2.0, 0.0),
    (0.5, 1.0),
    (1.5, 1.0),
    (1.0, 2.0),
])
```

Or generate a planting lattice clipped to a polygon:

```python
field = cm.Field.from_polygon(
    [(0, 0), (12, 0), (10, 8), (3, 11), (0, 6)],
    spacing=1.0,
)
```

The spatial model only needs plant coordinates and pairwise distances.

## More than two varieties

The spatial SPT engine is not hard-coded to two varieties:

```python
system = cm.CropMixSystem(
    varieties=(variety_a, variety_b, variety_c, variety_d),
    vector=vector,
    pathogen=pathogen,
    kernel=cm.ExponentialKernel(scale=2.0),
)

design = cm.MixtureDesign.random(
    field,
    counts={"A": 25, "B": 25, "C": 30, "D": 20},
    seed=10,
)
```

Internally, virus-bearing vectors can retain the variety on which acquisition occurred, while movement remains independent of host attractiveness unless a future model explicitly adds such biology.

## Kernel-scale calibration

If the biological exponential-kernel scale is not known, Cropmix can compare spatial dynamics with the corresponding PLOS-style finite-size mean-field model over multiple epidemic contexts:

```python
import numpy as np

calibration = cm.calibrate_kernel(
    field,
    system,
    scenarios=[
        cm.Scenario(vectors_per_plant=1, inoculum=cm.Inoculum.random(1)),
        cm.Scenario(vectors_per_plant=5, inoculum=cm.Inoculum.random(1)),
        cm.Scenario(vectors_per_plant=10, inoculum=cm.Inoculum.random(1)),
    ],
    reference_variety="SUSC",
    scales=np.geomspace(0.05, 100.0, 100),
    n_runs=100,
)

print(calibration.summary())
```

Cropmix compares **complete plant-incidence and virus-bearing-vector prevalence trajectories**. It reports context-specific minima as diagnostics, one common operational scale, a minimax scale, a one-standard-error acceptable region, and leave-one-context-out sensitivity.

This procedure is a **mean-field consistency calibration**, not a claim that terminal yield uniquely measures biological insect dispersal.

## Optimize a mixture while preserving counts

```python
config = cm.OptimizationConfig(
    iterations=1000,
    n_runs_per_candidate=30,
    final_runs=1000,
    initial_temperature=0.5,
    cooling_rate=0.995,
    seed=123,
)

optimum = cm.optimize_mixture(
    field=field,
    variety_counts={"SUSC": 50, "RES": 50},
    system=system,
    scenario=scenario,
    objective="expected_yield",
    config=config,
)

print(optimum.summary())
optimum.best_design.plot()
```

The optimizer uses **heterotypic swaps**, so variety counts are preserved exactly. Candidate designs are evaluated using common random numbers to reduce avoidable Monte Carlo noise. Because the design space is combinatorial, the alpha optimizer is heuristic and does not claim global optimality.

## Use EpiPvr without opening R

The public workflow remains Python-only. R is an implementation dependency behind the bridge.

First install R and EpiPvr once:

```r
install.packages("EpiPvr")
```

Then check the bridge from a terminal:

```bash
cropmix doctor
```

In Python:

```python
from cropmix.epipvr import (
    AccessPeriodAssay,
    AccessPeriodExperiment,
    EpiPvrBackend,
    EpiPvrFitOptions,
)

aap = AccessPeriodAssay(
    duration=(2, 3, 4, 5, 6, 8),
    tested=(30, 30, 30, 30, 30, 30),
    infected=(16, 19, 15, 15, 16, 16),
)

iap = AccessPeriodAssay(
    duration=(1/12, 1/6, 1/4, 1/3, 1/2, 1),
    tested=(30, 30, 30, 30, 30, 30),
    infected=(4, 6, 9, 13, 8, 22),
)

experiment = AccessPeriodExperiment.spt(
    acquisition=aap,
    inoculation=iap,
    fixed_inoculation_for_acquisition=6,
    fixed_acquisition_for_inoculation=4,
    vectors_per_plant=20,
)

backend = EpiPvrBackend()
fit = backend.fit(
    experiment,
    options=EpiPvrFitOptions(
        survival_upper_days=40,
        warmup=4500,
        iterations=6000,
        chains=4,
    ),
)

print(fit.parameter_summary(unit="per_day"))
print(fit.convergence_report())
```

Cropmix converts EpiPvr's posterior variables to semantic column names and provides explicit hour-to-day conversion. The joint posterior is retained; it is not reconstructed from independently sampled marginal distributions.

The EpiPvr branching-process epidemic probability is also available through Python:

```python
from cropmix.epipvr import LocalEpidemicParameters

bp = backend.epidemic_probability(
    vectors_per_plant=3,
    virus_parameters_per_day=(2.4, 24.0, 24.0),
    local_parameters=LocalEpidemicParameters(
        dispersal_rate=0.45,
        roguing_rate=1/28,
        harvest_rate=1/365,
        vector_mortality_rate=1/14,
        plant_latent_progression_rate=1/14,
    ),
)

print(bp.from_single_infectious_plant)
```

This branching-process quantity is an **early-invasion probability**, not the same output as a complete Cropmix field trajectory.

## Conservative vector exchange

For an unordered plant pair $\{i,j\}$, Cropmix builds a symmetric row-stochastic distance matrix \(P\) and uses the physical pair-exchange rate

$$
\lambda_{ij}=m\sigma P_{ij}.
$$

A swap chooses one vector at each plant and exchanges them. Therefore each plant retains exactly \(m\) vectors. Because

$$
\sum_j P_{ij}=1,
$$

a tagged vector has total movement rate

$$
\sum_j \lambda_{ij}/m = \sigma.
$$

The finite-field matrix is obtained by symmetrically balancing the raw exponential weights

$$
K(d)=e^{-d/a}.
$$

This avoids the reciprocity problem created by ordinary row normalization near open field boundaries.

## Reproducibility

```bash
pytest
ruff check .
python -m build
python -m twine check dist/*
```

Every scientific change should include a mathematical statement of the changed process and an invariant/limiting-case test.

## Scientific reference notebook

The repository includes `notebooks/cassava_spatial_model_hierarchical_scientific_reference.ipynb`, which documents the modelling lineage from EpiPvr Bayesian inference and branching-process invasion analysis to the PLOS mean field and the spatial Gillespie CTMC. It is a scientific reference, not part of the installed runtime.

## Documentation and publishing

- Full docs live in `docs/` and can be served locally with `mkdocs serve`.
- `HOSTING.md` gives a complete GitHub → CI → documentation → TestPyPI → PyPI → Zenodo workflow.
- `.github/workflows/release.yml` is configured for PyPI **Trusted Publishing**; replace the placeholder GitHub URLs before first release.

## Scientific references

- Tankam Chedjou, I., Donnelly, R. & Gilligan, C. A. (2025). *Optimizing crop varietal mixtures for viral disease management: A case study on cassava virus epidemics.* PLOS Computational Biology 21(9): e1012842. https://doi.org/10.1371/journal.pcbi.1012842
- Donnelly, R., Tankam Chedjou, I. & Gilligan, C. A. (2026). *Plant pathogen profiling with the EpiPvr package.* Methods in Ecology and Evolution 17: 837–849. https://doi.org/10.1111/2041-210x.70219

## License

Cropmix is distributed under the GNU General Public License v3.0 or later. The EpiPvr integration calls an independently installed GPL-3 R package through `Rscript`; Cropmix does not vendor EpiPvr's source code.


## Important modelling convention in 0.2

A movement scale should be estimated independently of the non-spatial model.
Mean-field matching is provided only to identify the regime in which spatial
structure ceases to matter. The CBSD example uses a literature-informed primary
scale and evaluates harvest-time sensitivity.
