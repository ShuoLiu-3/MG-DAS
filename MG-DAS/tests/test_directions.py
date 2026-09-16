import numpy as np
import pytest

from mgdas_v2.directions import aligned_pgd_geometry, repaired_pc_direction


def test_sign_alignment_prevents_cancellation() -> None:
    perturbations = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
    log_odds = np.asarray([2.0, -2.0])
    direction, kappa, concentration, _components = aligned_pgd_geometry(
        perturbations, log_odds, min_valid=2
    )
    assert direction == pytest.approx([1.0, 0.0])
    assert kappa == pytest.approx(1.0)
    assert concentration == pytest.approx(1.0)


def test_repaired_components_are_oriented_to_reference() -> None:
    components = np.asarray([[-1.0, 0.0], [0.0, 1.0]])
    reference = np.asarray([1.0, 1.0])
    direction = repaired_pc_direction(components, reference, components=2)
    assert direction[0] > 0
    assert direction[1] > 0
    assert np.linalg.norm(direction) == pytest.approx(1.0)

