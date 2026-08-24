# Mixture optimization

The number of possible arrangements is combinatorial. With 100 plants and 50/50 composition there are roughly `binom(100,50)` possible assignments, so exhaustive search is not realistic.

Cropmix 0.1 uses **swap-based simulated annealing**:

1. start from a valid assignment;
2. select two sites with different varieties;
3. swap their labels, preserving variety counts exactly;
4. estimate the objective with stochastic simulation;
5. accept improvements and sometimes accept worse proposals according to temperature;
6. retain the best design found;
7. re-evaluate it with a larger final Monte Carlo ensemble.

```python
optimum = cm.optimize_mixture(
    field,
    {"A": 30, "B": 40, "C": 30},
    system,
    scenario,
    objective="expected_yield",
    config=cm.OptimizationConfig(
        iterations=2000,
        n_runs_per_candidate=30,
        final_runs=2000,
        cooling_rate=0.995,
        seed=42,
    ),
)
```

Available built-in objectives are:

- `expected_yield`;
- `min_final_incidence`;
- `yield_stability` = mean yield minus yield SD.

A callable returning a scalar can also be supplied.

The optimizer is heuristic. `best_design` means best design found under the stated algorithm, stochastic replication level and objective; it is not a proof of global optimality.
