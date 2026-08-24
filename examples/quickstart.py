import cropmix as cm

field = cm.Field.rectangular(10, 10)

susc = cm.Variety(
    "SUSC",
    cm.HostTransmission(15.31, 1.34),
    cm.PlantParameters(1 / 30),
    cm.YieldParameters(31.0, 3.1),
)
res = cm.Variety(
    "RES",
    cm.HostTransmission(4.84, 0.42),
    cm.PlantParameters(1 / 30),
    cm.YieldParameters(25.0, 2.1),
)

system = cm.CropMixSystem(
    varieties=(susc, res),
    vector=cm.VectorParameters(0.19, 0.45),
    pathogen=cm.PathogenParameters(19.37, "SPT"),
    kernel=cm.ExponentialKernel(scale=1.0),
)

design = cm.MixtureDesign.random(field, {"SUSC": 50, "RES": 50}, seed=1)
scenario = cm.Scenario(duration=360, vectors_per_plant=10)

result = cm.simulate_mixture(design, system, scenario, n_runs=20, seed=42)
print(result.summary().to_string(index=False))
