import numpy as np
import cropmix as cm


def _system():
    v = cm.Variety(
        "A", cm.HostTransmission(1.0, 1.0),
        cm.PlantParameters(1/10), cm.YieldParameters(10, 2)
    )
    return cm.CropMixSystem(
        (v,), cm.VectorParameters(0.1, 0.2),
        cm.PathogenParameters(1.0), cm.ExponentialKernel(1.0)
    )


def test_zero_plant_inoculum_is_valid_control():
    field = cm.Field.rectangular(3, 3)
    design = cm.MixtureDesign.monoculture(field, "A")
    scenario = cm.Scenario(duration=20, vectors_per_plant=2, inoculum=cm.Inoculum.none())
    result = cm.simulate_mixture(design, _system(), scenario, n_runs=3, seed=7)
    assert np.allclose(result.final_incidence_runs, 0)


def test_vector_inoculum_can_seed_epidemic_without_infected_plants():
    field = cm.Field.rectangular(3, 3)
    design = cm.MixtureDesign.monoculture(field, "A")
    scenario = cm.Scenario(
        duration=20, vectors_per_plant=5, inoculum=cm.Inoculum.none(),
        vector_inoculum=cm.VectorInoculum(1.0)
    )
    result = cm.simulate_mixture(design, _system(), scenario, n_runs=4, seed=9)
    assert np.any(result.incidence_runs[:, -1] > 0)


def test_random_plant_inoculum_changes_location_between_runs():
    from cropmix.simulation import _initial_sites_table
    scenario = cm.Scenario(inoculum=cm.Inoculum.random(1))
    table = _initial_sites_table(scenario, 100, 20, 123)
    assert len(set(table[:, 0].tolist())) > 1
