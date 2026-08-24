# Changelog

## 0.2.0
- Reframed mean-field matching as a consistency/adequacy diagnostic rather than biological kernel calibration.
- Added `assess_mean_field_consistency()` and retained `calibrate_kernel()` as a deprecated alias.
- Default simulation horizon is 300 d.
- Plant inoculum may be zero, random, or fixed.
- Added independent initial vector infection pressure through `VectorInoculum`; plant and vector introductions may be combined.
- Preserved non-local reciprocal vector movement and exact local vector-number conservation.

## 0.1.0
- Initial release.

### Bundle revision 2026-08-24
- `SimulationResult` now exposes pointwise stochastic trajectory standard deviations.
- `trajectory_dataframe()` now includes mean, SD, and clipped mean ± 1 SD bounds.
- Incidence and vector-prevalence plotting helpers can display ±1 SD envelopes while remaining backward compatible.
- Publication-backend ensemble summaries now retain trajectory SD arrays in addition to full replicate matrices.
