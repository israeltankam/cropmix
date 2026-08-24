import numpy as np

import cropmix as cm


def test_meanfield_shapes(cb_system):
    field = cm.Field.rectangular(4, 4)
    design = cm.MixtureDesign.random(field, {"SUSC": 8, "RES": 8}, seed=1)
    scenario = cm.Scenario(duration=20, vectors_per_plant=2)
    times = np.linspace(0, 20, 11)
    result = cm.solve_mean_field(design, cb_system, scenario, observation_times=times)
    assert result.infectious.shape == (2, 11)
    assert result.viruliferous_by_origin.shape == (2, 11)
    assert np.all(np.isfinite(result.vector_prevalence))
    assert result.final_yield >= 0
