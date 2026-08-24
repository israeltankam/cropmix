import cropmix as cm

field = cm.Field.rectangular(4, 4)
susc = cm.Variety("SUSC", cm.HostTransmission(2.0, 0.8), cm.PlantParameters(0.1), cm.YieldParameters(30, 4))
res = cm.Variety("RES", cm.HostTransmission(0.6, 0.25), cm.PlantParameters(0.1), cm.YieldParameters(24, 3))
system = cm.CropMixSystem(
    (susc, res),
    cm.VectorParameters(0.2, 0.45),
    cm.PathogenParameters(1.0),
)
scenario = cm.Scenario(duration=30, vectors_per_plant=3)

result = cm.optimize_mixture(
    field,
    {"SUSC": 8, "RES": 8},
    system,
    scenario,
    config=cm.OptimizationConfig(iterations=50, n_runs_per_candidate=5, final_runs=20),
)
print(result.summary().to_string(index=False))
