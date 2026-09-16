from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataio import atomic_write_json
from .directions import DirectionFamily
from .types import BehaviorMetrics, EvaluatedConfig, InterventionAtom, InterventionConfig


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for source_path in sorted(package_dir.glob("*.py"), key=lambda path: path.name):
        digest.update(source_path.name.encode("utf-8"))
        digest.update(source_path.read_bytes())
    return digest.hexdigest()


def save_direction_families(path: str | Path, families: list[DirectionFamily]) -> None:
    arrays = {}
    metadata = []
    for family_index, family in enumerate(families):
        key = f"direction_{family_index}"
        arrays[key] = family.direction
        metadata.append(
            {
                "key": key,
                "family": family.family,
                "layer": family.layer,
                "kappa_direction": family.kappa_direction,
                "intervention_concentration": family.intervention_concentration,
            }
        )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def load_direction_families(path: str | Path) -> list[DirectionFamily]:
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"]))
    return [
        DirectionFamily(
            family=item["family"],
            layer=int(item["layer"]),
            direction=archive[item["key"]].astype(np.float32),
            kappa_direction=item["kappa_direction"],
            intervention_concentration=item["intervention_concentration"],
        )
        for item in metadata
    ]


def config_to_dict(config: InterventionConfig) -> dict[str, Any]:
    return {
        "identifier": config.identifier,
        "cardinality": config.cardinality,
        "strength": config.strength,
        "weights": list(config.weights),
        "atoms": [
            {
                "atom_id": atom.atom_id,
                "attribute": atom.attribute,
                "family": atom.family,
                "polarity": atom.polarity,
                "layer": atom.layer,
                "nomination_score": atom.nomination_score,
            }
            for atom in config.atoms
        ],
    }


def metrics_to_dict(metrics: BehaviorMetrics) -> dict[str, Any]:
    return {
        "s_dis": metrics.s_dis,
        "s_amb": metrics.s_amb,
        "acc_dis": metrics.acc_dis,
        "acc_amb": metrics.acc_amb,
        "mmlu": metrics.mmlu,
        "n_dis": metrics.n_dis,
        "n_amb": metrics.n_amb,
    }


def evaluated_to_dict(evaluated: EvaluatedConfig) -> dict[str, Any]:
    return {
        "config": config_to_dict(evaluated.config),
        "metrics": metrics_to_dict(evaluated.metrics),
        "gain": evaluated.gain,
        "delta_acc_dis": evaluated.delta_acc_dis,
        "direction_violation": evaluated.direction_violation,
        "feasible": evaluated.feasible,
        "rejection_reasons": list(evaluated.rejection_reasons),
    }


def save_stage(path: str | Path, payload: Any) -> None:
    atomic_write_json(path, payload)
