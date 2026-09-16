from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class BBQExample:
    example_id: str
    group_id: str
    attribute: str
    context: str
    question: str
    choices: tuple[str, str, str]
    label_index: int
    stereotype_index: int
    anti_stereotype_index: int
    unknown_index: int
    is_ambiguous: bool
    probe_label: int | None = None
    neutral_reference: bool = False

    def validate(self) -> None:
        indices = {self.stereotype_index, self.anti_stereotype_index, self.unknown_index}
        if indices != {0, 1, 2}:
            raise ValueError(f"{self.example_id}: stereotype/anti/unknown indices must be 0,1,2")
        if self.label_index not in {0, 1, 2}:
            raise ValueError(f"{self.example_id}: label_index must be 0,1,2")
        if self.is_ambiguous and self.label_index != self.unknown_index:
            raise ValueError(f"{self.example_id}: ambiguous example must label UNKNOWN")
        if self.probe_label not in {None, 0, 1}:
            raise ValueError(f"{self.example_id}: probe_label must be null, 0, or 1")


@dataclass(frozen=True)
class InterventionAtom:
    atom_id: str
    attribute: str
    family: str
    polarity: int
    layer: int
    direction: np.ndarray = field(compare=False, repr=False)
    nomination_score: float = 0.0

    def __post_init__(self) -> None:
        if self.polarity not in {-1, 1}:
            raise ValueError("atom polarity must be -1 or +1")
        direction = np.asarray(self.direction, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"atom {self.atom_id} has invalid direction")
        object.__setattr__(self, "direction", direction / norm)


@dataclass(frozen=True)
class InterventionConfig:
    atoms: tuple[InterventionAtom, ...]
    weights: tuple[float, ...]
    strength: float

    def __post_init__(self) -> None:
        if len(self.atoms) != len(self.weights):
            raise ValueError("atoms and weights must have equal lengths")
        if self.strength < 0:
            raise ValueError("strength must be non-negative")
        if not self.atoms:
            if self.weights or self.strength != 0:
                raise ValueError("zero intervention must have no weights and zero strength")
            return
        if any(weight < 0 for weight in self.weights):
            raise ValueError("weights must be non-negative")
        if abs(sum(self.weights) - 1.0) > 1e-8:
            raise ValueError("non-zero intervention weights must sum to one")
        layers = [atom.layer for atom in self.atoms]
        if len(layers) != len(set(layers)):
            raise ValueError("same-layer atoms are mutually exclusive")
        attributes = {atom.attribute for atom in self.atoms}
        if len(attributes) != 1:
            raise ValueError("one configuration may only target one attribute")

    @property
    def cardinality(self) -> int:
        return len(self.atoms)

    @property
    def identifier(self) -> str:
        payload = {
            "atoms": [atom.atom_id for atom in self.atoms],
            "weights": [round(weight, 12) for weight in self.weights],
            "strength": round(self.strength, 12),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def zero(cls) -> "InterventionConfig":
        return cls(atoms=(), weights=(), strength=0.0)


@dataclass(frozen=True)
class BehaviorMetrics:
    s_dis: float
    s_amb: float
    acc_dis: float
    acc_amb: float
    mmlu: float | None = None
    n_dis: int = 0
    n_amb: int = 0

    def validate(self) -> None:
        if not -100.0 <= self.s_dis <= 100.0:
            raise ValueError("s_dis must be reported on the [-100, 100] scale")
        if not -100.0 <= self.s_amb <= 100.0:
            raise ValueError("s_amb must be reported on the [-100, 100] scale")
        for name, value in (("acc_dis", self.acc_dis), ("acc_amb", self.acc_amb)):
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be a percentage in [0, 100]")
        if self.mmlu is not None and not 0.0 <= self.mmlu <= 100.0:
            raise ValueError("mmlu must be a percentage in [0, 100]")


@dataclass(frozen=True)
class EvaluatedConfig:
    config: InterventionConfig
    metrics: BehaviorMetrics
    gain: float
    delta_acc_dis: float
    direction_violation: float
    feasible: bool | None = None
    rejection_reasons: tuple[str, ...] = ()
