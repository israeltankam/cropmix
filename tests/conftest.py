import pytest

import cropmix as cm


@pytest.fixture
def cb_system():
    susc = cm.Variety(
        name="SUSC",
        transmission=cm.HostTransmission(15.31, 1.34),
        plant=cm.PlantParameters(1 / 30, 0.0),
        yield_model=cm.YieldParameters(31.0, 3.1),
    )
    res = cm.Variety(
        name="RES",
        transmission=cm.HostTransmission(4.84, 0.42),
        plant=cm.PlantParameters(1 / 30, 0.0),
        yield_model=cm.YieldParameters(25.0, 2.1),
    )
    return cm.CropMixSystem(
        varieties=(susc, res),
        vector=cm.VectorParameters(mortality_rate=0.19, dispersal_rate=0.45),
        pathogen=cm.PathogenParameters(vector_clearance_rate=19.37, transmission_mode="SPT"),
        kernel=cm.ExponentialKernel(scale=1.0),
    )
