import numpy as np

import cropmix as cm


def test_simulation_is_reproducible(cb_system):
    field = cm.Field.rectangular(3, 4)
    design = cm.MixtureDesign.random(field, {"SUSC": 6, "RES": 6}, seed=5)
    scenario = cm.Scenario(duration=10, vectors_per_plant=2, inoculum=cm.Inoculum.random(1))
    first = cm.simulate_mixture(design, cb_system, scenario, n_runs=3, seed=10)
    second = cm.simulate_mixture(design, cb_system, scenario, n_runs=3, seed=10)
    assert np.array_equal(first.yield_runs, second.yield_runs)
    assert np.array_equal(first.incidence_runs, second.incidence_runs)
    assert np.all((first.incidence_runs >= 0) & (first.incidence_runs <= 1))
    assert np.all((first.vector_prevalence_runs >= 0) & (first.vector_prevalence_runs <= 1))


def test_more_than_two_varieties():
    field = cm.Field.rectangular(2, 3)
    varieties = []
    for index, name in enumerate(("A", "B", "C")):
        varieties.append(
            cm.Variety(
                name=name,
                transmission=cm.HostTransmission(0.5 + index * 0.1, 0.2 + index * 0.05),
                plant=cm.PlantParameters(0.1),
                yield_model=cm.YieldParameters(10 + index, 2 + index),
            )
        )
    system = cm.CropMixSystem(
        varieties=tuple(varieties),
        vector=cm.VectorParameters(0.1, 0.4),
        pathogen=cm.PathogenParameters(0.5),
    )
    design = cm.MixtureDesign(field, ("A", "B", "C", "A", "B", "C"))
    result = cm.simulate_mixture(
        design,
        system,
        cm.Scenario(duration=5, vectors_per_plant=2),
        n_runs=2,
        seed=2,
    )
    assert set(result.incidence_by_variety_runs) == {"A", "B", "C"}
