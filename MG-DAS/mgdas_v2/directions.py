from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class DirectionFamily:
    family: str
    layer: int
    direction: np.ndarray
    kappa_direction: float | None
    intervention_concentration: float | None


class LinearProbe(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden_states)


def train_linear_probe(
    train_hidden: np.ndarray,
    train_labels: np.ndarray,
    validation_hidden: np.ndarray,
    validation_labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str = "cpu",
) -> tuple[LinearProbe, float]:
    torch.manual_seed(seed)
    features = torch.as_tensor(train_hidden, dtype=torch.float32)
    labels = torch.as_tensor(train_labels, dtype=torch.long)
    probe = LinearProbe(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    probe.train()
    for _epoch in range(epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = probe(batch_features.to(device))
            loss = nn.functional.cross_entropy(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()
    probe.eval()
    with torch.no_grad():
        validation_logits = probe(torch.as_tensor(validation_hidden, dtype=torch.float32, device=device))
        predictions = validation_logits.argmax(dim=-1).cpu().numpy()
    accuracy = float(np.mean(predictions == np.asarray(validation_labels)))
    return probe, accuracy


def neutralizing_pgd(
    probe: nn.Module,
    hidden_states: np.ndarray,
    radius_fraction: float,
    step_fraction: float,
    steps: int,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    probe.eval()
    base = torch.as_tensor(hidden_states, dtype=torch.float32, device=device)
    radius = radius_fraction * torch.linalg.vector_norm(base, dim=1, keepdim=True).clamp_min(1e-8)
    delta = torch.zeros_like(base, requires_grad=True)
    best_delta = delta.detach().clone()
    with torch.no_grad():
        base_logits = probe(base)
        base_log_probabilities = nn.functional.log_softmax(base_logits, dim=-1)
        log_odds = base_log_probabilities[:, 1] - base_log_probabilities[:, 0]
    best_loss = torch.full((base.shape[0],), float("inf"), device=device)
    for _step in range(steps):
        log_probabilities = nn.functional.log_softmax(probe(base + delta), dim=-1)
        per_sample_loss = -0.5 * (log_probabilities[:, 0] + log_probabilities[:, 1])
        gradient = torch.autograd.grad(per_sample_loss.sum(), delta)[0]
        gradient_norm = torch.linalg.vector_norm(gradient, dim=1, keepdim=True).clamp_min(1e-8)
        updated = delta - step_fraction * radius * gradient / gradient_norm
        updated_norm = torch.linalg.vector_norm(updated, dim=1, keepdim=True).clamp_min(1e-8)
        projected = updated * torch.minimum(torch.ones_like(updated_norm), radius / updated_norm)
        with torch.no_grad():
            improved = per_sample_loss < best_loss
            best_delta[improved] = delta.detach()[improved]
            best_loss[improved] = per_sample_loss.detach()[improved]
        delta = projected.detach().requires_grad_(True)
    with torch.no_grad():
        final_log_probabilities = nn.functional.log_softmax(probe(base + delta), dim=-1)
        final_loss = -0.5 * (final_log_probabilities[:, 0] + final_log_probabilities[:, 1])
        improved = final_loss < best_loss
        best_delta[improved] = delta.detach()[improved]
    return best_delta.cpu().numpy(), log_odds.cpu().numpy()


def _unit_rows(matrix: np.ndarray, epsilon: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(matrix, axis=1)
    valid = np.isfinite(norms) & (norms > epsilon)
    return matrix[valid] / norms[valid, None], valid


def aligned_pgd_geometry(
    perturbations: np.ndarray,
    probe_log_odds: np.ndarray,
    min_valid: int,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    unit_perturbations, valid_mask = _unit_rows(np.asarray(perturbations, dtype=np.float64))
    valid_log_odds = np.asarray(probe_log_odds, dtype=np.float64)[valid_mask]
    if len(unit_perturbations) < min_valid:
        raise ValueError(f"only {len(unit_perturbations)} valid perturbations; require {min_valid}")
    signs = np.where(valid_log_odds >= 0.0, 1.0, -1.0)
    aligned = unit_perturbations * signs[:, None]
    mean_vector = aligned.mean(axis=0)
    kappa_direction = float(np.linalg.norm(mean_vector))
    if kappa_direction <= 1e-12:
        raise ValueError("aligned perturbations have zero mean direction")
    mean_direction = (mean_vector / kappa_direction).astype(np.float32)
    centered = aligned - mean_vector
    _left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values**2 / len(centered)
    eigenvalue_sum = float(eigenvalues.sum())
    intervention_concentration = float(eigenvalues[0] / eigenvalue_sum) if eigenvalue_sum > 0 else 1.0
    return mean_direction, kappa_direction, intervention_concentration, right.astype(np.float32)


def centroid_direction(attribute_hidden: np.ndarray, neutral_hidden: np.ndarray) -> np.ndarray:
    if len(attribute_hidden) == 0 or len(neutral_hidden) == 0:
        raise ValueError("centroid direction requires attribute and neutral examples")
    difference = np.asarray(attribute_hidden).mean(axis=0) - np.asarray(neutral_hidden).mean(axis=0)
    norm = float(np.linalg.norm(difference))
    if norm <= 1e-12:
        raise ValueError("centroid direction has zero norm")
    return (difference / norm).astype(np.float32)


def repaired_pc_direction(
    principal_components: np.ndarray, reference_direction: np.ndarray, components: int
) -> np.ndarray:
    if principal_components.shape[0] < components:
        raise ValueError(f"need {components} principal components")
    selected = np.asarray(principal_components[:components], dtype=np.float64).copy()
    reference = np.asarray(reference_direction, dtype=np.float64)
    dots = selected @ reference
    signs = np.where(dots >= 0.0, 1.0, -1.0)
    repaired = (selected * signs[:, None]).mean(axis=0)
    norm = float(np.linalg.norm(repaired))
    if norm <= 1e-12:
        raise ValueError("repaired PC direction has zero norm")
    return (repaired / norm).astype(np.float32)


def build_direction_families(
    layer: int,
    attribute_hidden: np.ndarray,
    neutral_hidden: np.ndarray,
    perturbations: np.ndarray,
    probe_log_odds: np.ndarray,
    min_valid: int,
    gamma_direction: float,
    gamma_pc3: float,
    gamma_pc5: float,
) -> list[DirectionFamily]:
    centroid = centroid_direction(attribute_hidden, neutral_hidden)
    families = [DirectionFamily("centroid", layer, centroid, None, None)]
    try:
        mean_direction, kappa_direction, concentration, components = aligned_pgd_geometry(
            perturbations, probe_log_odds, min_valid
        )
    except ValueError:
        return families
    if kappa_direction < gamma_direction:
        return families
    families.append(
        DirectionFamily("a_pgd_mean", layer, mean_direction, kappa_direction, concentration)
    )
    if concentration < gamma_pc3 and components.shape[0] >= 3:
        families.append(
            DirectionFamily(
                "repaired_pc3",
                layer,
                repaired_pc_direction(components, mean_direction, 3),
                kappa_direction,
                concentration,
            )
        )
    if concentration < gamma_pc5 and components.shape[0] >= 5:
        families.append(
            DirectionFamily(
                "repaired_pc5",
                layer,
                repaired_pc_direction(components, mean_direction, 5),
                kappa_direction,
                concentration,
            )
        )
    return families

