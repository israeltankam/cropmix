# Architecture

Cropmix separates scientific concepts rather than exposing a single function with dozens of parameters.

```text
Field + MixtureDesign
        |
        v
CropMixSystem ---- Scenario
  |                  |
  |                  v
  |             initial pressure
  v
Varieties + vector + pathogen + kernel
        |
        +------------------+
        |                  |
        v                  v
simulate_mixture     optimize_mixture
        |                  |
        v                  v
SimulationResult     OptimizationResult
```

## Parameter inference is an upstream layer

```text
AccessPeriodExperiment
        |
        v
EpiPvrBackend (Python)
        |
        v
internal Rscript bridge -> EpiPvr / Stan
        |
        v
EpiPvrFit / joint posterior draws
        |
        v
TransmissionDraw -> Cropmix spatial CTMC
```

The R bridge is an implementation detail. A Cropmix user need not construct R lists or open an R session manually.

## Core invariants

For the current SPT spatial engine:

- one plant occupies every field site;
- each site contains exactly `m` vectors;
- vector exchanges preserve that local burden;
- movement does not depend on cultivar attractiveness;
- a `MixtureDesign` is simply one variety label per site;
- arbitrary numbers of varieties are supported;
- a PT system is rejected by the spatial engine until an exposed-vector compartment is implemented and validated.

## Why the top-level API stays small

The two principal scientific operations are:

```python
cropmix.simulate_mixture(...)
cropmix.optimize_mixture(...)
```

Kernel calibration is a separate scientific operation because it produces identifiability diagnostics rather than merely configuring a simulator:

```python
cropmix.calibrate_kernel(...)
```
