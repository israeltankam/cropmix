# Quick start

## 1. Define a field

```python
import cropmix as cm

field = cm.Field.rectangular(10, 10, spacing=1.0)
```

The field does not have to be rectangular; arbitrary coordinates are the canonical representation.

## 2. Define varieties

```python
susc = cm.Variety(
    name="SUSC",
    transmission=cm.HostTransmission(15.31, 1.34),
    plant=cm.PlantParameters(1/30),
    yield_model=cm.YieldParameters(31, 3.1),
)

res = cm.Variety(
    name="RES",
    transmission=cm.HostTransmission(4.84, 0.42),
    plant=cm.PlantParameters(1/30),
    yield_model=cm.YieldParameters(25, 2.1),
)
```

## 3. Define vector and pathogen biology

```python
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
```

## 4. Give Cropmix a planting design

```python
design = cm.MixtureDesign.random(
    field,
    {"SUSC": 50, "RES": 50},
    seed=1,
)
```

A `MixtureDesign` is simply one variety label per planting coordinate. The same object is used for user-supplied and optimizer-generated designs.

## 5. Define the epidemic scenario

```python
scenario = cm.Scenario(
    duration=360,
    vectors_per_plant=10,
    inoculum=cm.Inoculum.random(1),
)
```

## 6. Simulate

```python
result = cm.simulate_mixture(
    design,
    system,
    scenario,
    n_runs=100,
    seed=42,
)

print(result.summary())
```

With `cropmix[viz]` installed:

```python
result.plot_incidence(by_variety=True)
result.plot_final_infection_probability()
```
