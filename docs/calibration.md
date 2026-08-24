# Mean-field consistency, not biological calibration

Cropmix 0.2 deliberately separates **biological dispersal information** from
**mean-field consistency**. The function `assess_mean_field_consistency()` scans
a movement-scale grid and asks at which scales the finite spatial model becomes
practically indistinguishable from the PLOS-style deterministic mean-field
reference.

The returned one-standard-error scale is therefore a *mean-field adequacy
threshold*, not an estimate of insect dispersal. Biological analyses should use a
scale supported independently by the literature, tracking data, or spatial
epidemic observations.

The deprecated name `calibrate_kernel()` remains as a compatibility alias.

```python
result = cropmix.assess_mean_field_consistency(
    field, system, scenarios, reference_variety="SUSC"
)
```

For the CBSD manuscript case study the primary spatial scale is literature based.
McQuaid et al. (2016) use an exponential radial kernel with mean dispersal
distance `2/alpha = 5.25 m` (range `1-11.5 m`). With Cropmix's kernel
`K(d)=exp(-d/a)`, this corresponds to `a = 2.625 m` and a sensitivity interval
`a = 0.5-5.75 m` before finite-field balancing.
