from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

from .types import InterventionConfig


def layer_shifts(config: InterventionConfig, device: torch.device, dtype: torch.dtype) -> dict[int, torch.Tensor]:
    shifts: dict[int, torch.Tensor] = {}
    for atom, weight in zip(config.atoms, config.weights, strict=True):
        direction = torch.as_tensor(atom.direction, device=device, dtype=dtype)
        shifts[atom.layer] = atom.polarity * weight * direction
    return shifts


def apply_relative_shift(
    output: torch.Tensor,
    unit_shift: torch.Tensor,
    total_strength: float,
    token_scope: str,
    reference_scale: torch.Tensor,
) -> torch.Tensor:
    if output.shape[-1] != unit_shift.shape[-1]:
        raise ValueError("direction dimension does not match layer output")
    scale = reference_scale.to(device=output.device, dtype=output.dtype)
    if scale.ndim != 3 or scale.shape[0] != output.shape[0] or scale.shape[1:] != (1, 1):
        raise ValueError("reference_scale must have shape [batch, 1, 1]")
    modified = output.clone()
    if token_scope == "all":
        modified = output + total_strength * scale * unit_shift.view(1, 1, -1)
    elif token_scope == "last":
        last = output[:, -1:, :]
        modified[:, -1:, :] = last + total_strength * scale * unit_shift.view(1, 1, -1)
    else:
        raise ValueError("token_scope must be all or last")
    return modified


def reference_scale(output: torch.Tensor, token_scope: str) -> torch.Tensor:
    if token_scope == "all":
        return torch.linalg.vector_norm(output, dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)
    if token_scope == "last":
        return torch.linalg.vector_norm(output[:, -1:, :], dim=-1, keepdim=True)
    raise ValueError("token_scope must be all or last")


def resolve_module(root: torch.nn.Module, dotted_path: str) -> torch.nn.Module:
    current: Any = root
    for part in dotted_path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    if not isinstance(current, torch.nn.Module):
        raise TypeError(f"{dotted_path} does not resolve to a torch module")
    return current


@contextmanager
def intervention_hooks(
    model: torch.nn.Module,
    config: InterventionConfig,
    layer_module_template: str,
    token_scope: str,
    reference_scales: dict[int, torch.Tensor],
) -> Iterator[None]:
    if config.cardinality == 0:
        yield
        return
    handles = []
    try:
        for atom, weight in zip(config.atoms, config.weights, strict=True):
            module = resolve_module(model, layer_module_template.format(layer=atom.layer))
            signed_direction = atom.polarity * weight * atom.direction
            if atom.layer not in reference_scales:
                raise ValueError(f"missing unperturbed reference scale for layer {atom.layer}")
            fixed_scale = reference_scales[atom.layer]

            def hook(_module, _inputs, output, direction=signed_direction, scale=fixed_scale):
                tensor = output[0] if isinstance(output, tuple) else output
                shift = torch.as_tensor(direction, device=tensor.device, dtype=tensor.dtype)
                if isinstance(output, tuple):
                    changed = apply_relative_shift(
                        tensor, shift, config.strength, token_scope, scale
                    )
                    return (changed, *output[1:])
                return apply_relative_shift(tensor, shift, config.strength, token_scope, scale)

            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def reference_scale_hooks(
    model: torch.nn.Module,
    layers: tuple[int, ...],
    layer_module_template: str,
    token_scope: str,
) -> Iterator[dict[int, torch.Tensor]]:
    scales: dict[int, torch.Tensor] = {}
    handles = []
    try:
        for layer in layers:
            module = resolve_module(model, layer_module_template.format(layer=layer))

            def hook(_module, _inputs, output, layer_index=layer):
                tensor = output[0] if isinstance(output, tuple) else output
                scales[layer_index] = reference_scale(tensor, token_scope).detach()

            handles.append(module.register_forward_hook(hook))
        yield scales
    finally:
        for handle in handles:
            handle.remove()
