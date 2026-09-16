import numpy as np
import pytest
import torch

from mgdas_v2.intervention import apply_relative_shift, layer_shifts, reference_scale
from mgdas_v2.types import InterventionAtom, InterventionConfig


def atom(identifier: str, layer: int, direction: list[float]) -> InterventionAtom:
    return InterventionAtom(identifier, "Age", "centroid", 1, layer, np.asarray(direction))


def test_same_layer_atoms_are_rejected() -> None:
    with pytest.raises(ValueError, match="same-layer"):
        InterventionConfig(
            atoms=(atom("a", 3, [1, 0]), atom("b", 3, [0, 1])),
            weights=(0.5, 0.5),
            strength=0.1,
        )


def test_weights_must_share_one_budget() -> None:
    config = InterventionConfig(
        atoms=(atom("a", 3, [1, 0]), atom("b", 4, [0, 1])),
        weights=(0.25, 0.75),
        strength=0.1,
    )
    shifts = layer_shifts(config, torch.device("cpu"), torch.float32)
    assert torch.linalg.vector_norm(shifts[3]).item() == pytest.approx(0.25)
    assert torch.linalg.vector_norm(shifts[4]).item() == pytest.approx(0.75)


def test_relative_shift_uses_hidden_norm() -> None:
    output = torch.tensor([[[3.0, 4.0]]])
    scale = torch.tensor([[[5.0]]])
    changed = apply_relative_shift(output, torch.tensor([1.0, 0.0]), 0.1, "all", scale)
    assert changed[0, 0, 0].item() == pytest.approx(3.5)
    assert changed[0, 0, 1].item() == pytest.approx(4.0)


def test_reference_scale_is_mean_unperturbed_token_norm() -> None:
    output = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
    scale = reference_scale(output, "all")
    assert scale.shape == (1, 1, 1)
    assert scale.item() == pytest.approx(3.5)
