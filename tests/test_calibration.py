import numpy as np

import cropmix as cm


def test_calibration_smoke(cb_system):
    field = cm.Field.rectangular(2, 3)
    scenarios = [cm.Scenario(duration=5, vectors_per_plant=1)]
    result = cm.calibrate_kernel(
        field,
        cb_system,
        scenarios,
        reference_variety="SUSC",
        scales=np.array([0.5, 1.0, 2.0]),
        n_runs=2,
        bootstrap_reps=2,
        seed=7,
    )
    assert result.selected_scale in {0.5, 1.0, 2.0}
    assert len(result.joint_table) == 3
