from pathlib import Path

import numpy as np

from cropmix.epipvr import AccessPeriodAssay, AccessPeriodExperiment


def test_spt_experiment_serialization(tmp_path: Path):
    aap = AccessPeriodAssay((2, 4), (10, 10), (2, 5))
    iap = AccessPeriodAssay((0.5, 1), (10, 10), (1, 4))
    experiment = AccessPeriodExperiment.spt(
        acquisition=aap,
        inoculation=iap,
        fixed_inoculation_for_acquisition=6,
        fixed_acquisition_for_inoculation=4,
        vectors_per_plant=20,
    )
    experiment.write_bundle(tmp_path)
    assert (tmp_path / "AAP.csv").exists()
    assert (tmp_path / "IAP.csv").exists()
    assert np.array_equal(experiment.fixed_durations, np.array([[-1, 6], [4, -1]]))


def test_pt_duration_matrix():
    assay = AccessPeriodAssay((1, 2), (10, 10), (2, 4))
    experiment = AccessPeriodExperiment.pt(
        acquisition=assay,
        latent=assay,
        inoculation=assay,
        fixed_when_acquisition_varies=(0.5, 1),
        fixed_when_latency_varies=(2, 1),
        fixed_when_inoculation_varies=(2, 0.5),
        vectors_per_plant=20,
    )
    assert experiment.fixed_durations.shape == (3, 3)


def test_r_bridge_uses_correct_bayesian_r2_slot_for_pt():
    from importlib.resources import files

    bridge = files("cropmix.epipvr.resources").joinpath("fit_bridge.R").read_text()
    assert 'if (mode == "SPT") fit$array5 else fit$array6' in bridge
