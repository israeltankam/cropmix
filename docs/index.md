# Cropmix

Cropmix is a scientific Python package for **spatial crop varietal mixtures under vector-borne disease pressure**.

Its two primary user operations are:

```python
cropmix.simulate_mixture(...)
cropmix.optimize_mixture(...)
```

The API remains manageable because geometry, planting design, varieties, vector biology, pathogen biology, movement, and epidemic scenario are separate typed objects.

## What Cropmix connects

Cropmix is designed around a modelling hierarchy:

1. EpiPvr infers transmission parameters from access-period experiments using Bayesian models.
2. EpiPvr can propagate those parameters into branching-process estimates of early epidemic establishment probability.
3. The PLOS mixture ODE provides a deterministic mean-field reference.
4. Cropmix uses a finite spatial Gillespie CTMC to simulate complete epidemics and compare planting arrangements.

The current spatial engine implements **semi-persistent transmission (SPT)**. PT inference is available through the EpiPvr bridge, but PT spatial dynamics are intentionally not guessed.

## Installation

After the package has been released to PyPI:

```bash
python -m pip install cropmix
```

For plots:

```bash
python -m pip install "cropmix[viz]"
```

For development:

```bash
python -m pip install -e ".[dev,viz,docs]"
```

See the [quick start](quickstart.md) next.
