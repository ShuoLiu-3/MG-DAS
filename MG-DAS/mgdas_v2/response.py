from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .directions import DirectionFamily
from .metrics import signed_preference
from .types import BBQExample, InterventionAtom, InterventionConfig


ScoreFunction = Callable[[list[BBQExample], InterventionConfig], np.ndarray]


@dataclass(frozen=True)
class ResponseRecord:
    family: str
    polarity: int
    layer: int
    strength_responses: tuple[float, ...]
    slope: float


def screen_behavioral_response(
    attribute: str,
    examples: list[BBQExample],
    directions: list[DirectionFamily],
    score_function: ScoreFunction,
    local_strengths: tuple[float, ...],
    epsilon_bias: float,
    near_neutral_abs_log_odds: float,
    max_layers_per_direction_polarity: int,
    excluded_layers: tuple[int, ...],
) -> tuple[list[InterventionAtom], list[ResponseRecord]]:
    baseline_scores = score_function(examples, InterventionConfig.zero())
    baseline_preference = signed_preference(baseline_scores, examples)
    near_neutral_mask = np.abs(baseline_preference) <= near_neutral_abs_log_odds
    if not near_neutral_mask.any():
        raise ValueError(f"{attribute}: no near-neutral calibration examples")
    selected_examples = [example for example, keep in zip(examples, near_neutral_mask, strict=True) if keep]
    selected_baseline = baseline_preference[near_neutral_mask]
    records = []
    for family in directions:
        if family.layer in excluded_layers:
            continue
        for polarity in (-1, 1):
            atom_id = f"{attribute}:{family.family}:p{polarity:+d}:L{family.layer}"
            atom = InterventionAtom(
                atom_id=atom_id,
                attribute=attribute,
                family=family.family,
                polarity=polarity,
                layer=family.layer,
                direction=family.direction,
            )
            mean_responses = []
            for strength in local_strengths:
                config = InterventionConfig((atom,), (1.0,), strength)
                intervened_scores = score_function(selected_examples, config)
                intervened_preference = signed_preference(intervened_scores, selected_examples)
                response = (
                    np.abs(selected_baseline) - np.abs(intervened_preference)
                ) / (np.abs(selected_baseline) + epsilon_bias)
                mean_responses.append(float(np.mean(response)))
            strengths = np.asarray(local_strengths, dtype=np.float64)
            slope = float(np.dot(mean_responses, strengths) / np.dot(strengths, strengths))
            records.append(
                ResponseRecord(
                    family=family.family,
                    polarity=polarity,
                    layer=family.layer,
                    strength_responses=tuple(mean_responses),
                    slope=slope,
                )
            )
    grouped: dict[tuple[str, int], list[ResponseRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.family, record.polarity)].append(record)
    direction_lookup = {(item.family, item.layer): item.direction for item in directions}
    nominated = []
    for (family_name, polarity), family_records in sorted(grouped.items()):
        ranked = sorted(family_records, key=lambda item: (-item.slope, item.layer))
        for record in ranked[:max_layers_per_direction_polarity]:
            nominated.append(
                InterventionAtom(
                    atom_id=f"{attribute}:{family_name}:p{polarity:+d}:L{record.layer}",
                    attribute=attribute,
                    family=family_name,
                    polarity=polarity,
                    layer=record.layer,
                    direction=direction_lookup[(family_name, record.layer)],
                    nomination_score=record.slope,
                )
            )
    return nominated, records


def nominate_by_probe_accuracy(
    attribute: str,
    directions: list[DirectionFamily],
    probe_accuracy_by_layer: dict[int, float],
    max_layers_per_direction_polarity: int,
    excluded_layers: tuple[int, ...],
) -> list[InterventionAtom]:
    layers_by_family: dict[str, list[DirectionFamily]] = defaultdict(list)
    for family in directions:
        if family.layer not in excluded_layers:
            layers_by_family[family.family].append(family)
    atoms = []
    for family_name, family_directions in sorted(layers_by_family.items()):
        ranked = sorted(
            family_directions,
            key=lambda item: (-probe_accuracy_by_layer[item.layer], item.layer),
        )[:max_layers_per_direction_polarity]
        for polarity in (-1, 1):
            for family in ranked:
                atoms.append(
                    InterventionAtom(
                        atom_id=f"{attribute}:{family_name}:p{polarity:+d}:L{family.layer}",
                        attribute=attribute,
                        family=family_name,
                        polarity=polarity,
                        layer=family.layer,
                        direction=family.direction,
                        nomination_score=probe_accuracy_by_layer[family.layer],
                    )
                )
    return atoms

