import numpy as np

import cropmix as cm


def test_irregular_field_and_design():
    field = cm.Field.from_coordinates([(0, 0), (1, 0), (2, 0), (0.5, 1), (1.5, 1)])
    design = cm.MixtureDesign(field, ("A", "B", "A", "B", "A"))
    assert field.n_sites == 5
    assert design.counts == {"A": 3, "B": 2}
    assert design.proportions["A"] == 0.6


def test_polygon_constructor():
    field = cm.Field.from_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], spacing=1.0)
    assert field.n_sites == 9


def test_random_design_preserves_counts():
    field = cm.Field.rectangular(4, 5)
    design = cm.MixtureDesign.random(field, {"A": 7, "B": 8, "C": 5}, seed=1)
    assert design.counts == {"A": 7, "B": 8, "C": 5}
    assert np.isclose(sum(design.proportions.values()), 1.0)
