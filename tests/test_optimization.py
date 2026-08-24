import cropmix as cm


def test_optimizer_preserves_counts(cb_system):
    field = cm.Field.rectangular(2, 4)
    counts = {"SUSC": 4, "RES": 4}
    scenario = cm.Scenario(duration=3, vectors_per_plant=1)
    config = cm.OptimizationConfig(
        iterations=3,
        n_runs_per_candidate=1,
        final_runs=2,
        seed=4,
    )
    result = cm.optimize_mixture(field, counts, cb_system, scenario, config=config)
    assert result.best_design.counts == counts
    assert result.final_result.design.counts == counts
