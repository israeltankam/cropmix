import numpy as np

import cropmix as cm


def test_balanced_kernel_invariants():
    field = cm.Field.from_coordinates([(0, 0), (1, 0), (3, 0), (0.5, 2)])
    kernel = cm.ExponentialKernel(scale=1.7).prepare(field)
    p = kernel.probabilities
    assert np.allclose(np.diag(p), 0.0)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(p, p.T, atol=1e-9)
    assert np.isclose(np.triu(p, 1).sum(), field.n_sites / 2, atol=1e-8)
