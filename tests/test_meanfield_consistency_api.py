import cropmix as cm

def test_new_api_exported():
    assert callable(cm.assess_mean_field_consistency)
    assert cm.MeanFieldConsistencyResult is cm.KernelCalibrationResult
