# Spatial simulation

## Plant states

For each site `i`:

\[
X_i(t)\in\{S,L,I\}.
\]

## Vector state

Each site retains exactly `m` vectors. Virus-bearing vectors are stratified by the variety on which acquisition occurred. Healthy vector count is implicit.

## Conservative exchange

Raw exponential weights are

\[
W_{ij}=\exp(-d_{ij}/a),\qquad i\ne j.
\]

Cropmix symmetrically balances these weights into a reciprocal row-stochastic matrix `P` and assigns each unordered plant pair the physical exchange rate

\[
\lambda_{ij}=m\sigma P_{ij}.
\]

This gives every tagged vector total dispersal rate `sigma` and preserves exactly `m` vectors at every plant.

## Events

The SPT engine includes:

- plant inoculation `S -> L`;
- plant latent progression `L -> I`;
- roguing `I -> S`;
- vector acquisition on infectious plants;
- vector exchange;
- virus-bearing vector mortality with healthy replacement;
- vector loss of infectivity.

The current engine does not include an explicit PT vector latent stage.

## Reproducible comparisons

Using the same `seed` for competing designs creates common random-number blocks. This is recommended for arrangement comparisons and is used internally by the optimizer.


## Initial infection

Plant and vector introductions are independent and can be combined.

```python
# one random infectious plant, virus-free vectors
scenario = cropmix.Scenario(
    duration=300,
    inoculum=cropmix.Inoculum.random(1),
    vector_inoculum=cropmix.VectorInoculum.none(),
)

# vector infection pressure only
scenario = cropmix.Scenario(
    duration=300,
    inoculum=cropmix.Inoculum.none(),
    vector_inoculum=cropmix.VectorInoculum(0.02),
)
```

For random plant inoculum, locations are drawn independently for every stochastic
replicate while remaining reproducible under the master seed.

## Stochastic trajectory uncertainty

`SimulationResult` retains every Monte Carlo trajectory. Pointwise stochastic
variability is available through `incidence_sd`, `vector_prevalence_sd`, and
`incidence_by_variety_sd`. `trajectory_dataframe()` includes the corresponding
SD and clipped mean ± 1 SD envelope columns. By default, `plot_incidence()` and
`plot_vector_prevalence()` draw these ±1 SD envelopes; pass `show_sd=False` to
recover a mean-only plot.
