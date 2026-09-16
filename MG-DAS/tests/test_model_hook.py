import numpy as np
import pytest
import torch

from mgdas_v2.intervention import intervention_hooks, reference_scale_hooks
from mgdas_v2.types import InterventionAtom, InterventionConfig


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values)
        return values


def test_hooks_apply_multiple_layers_and_are_removed() -> None:
    model = DummyModel()
    atoms = (
        InterventionAtom("a", "Age", "centroid", 1, 0, np.asarray([1.0, 0.0])),
        InterventionAtom("b", "Age", "repaired_pc3", -1, 1, np.asarray([0.0, 1.0])),
    )
    config = InterventionConfig(atoms=atoms, weights=(0.5, 0.5), strength=0.2)
    values = torch.tensor([[[3.0, 4.0]]])

    with reference_scale_hooks(model, (0, 1), "layers.{layer}", "all") as scales:
        model(values)
    with intervention_hooks(model, config, "layers.{layer}", "all", scales):
        changed = model(values)

    assert changed[0, 0, 0].item() == pytest.approx(3.5)
    assert changed[0, 0, 1].item() == pytest.approx(3.5)
    assert torch.equal(model(values), values)
