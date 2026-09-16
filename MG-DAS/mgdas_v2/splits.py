from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .types import BBQExample


POOL_NAMES = ("cal", "search", "cert", "test")


def build_grouped_split_manifest(
    examples: list[BBQExample], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    if set(ratios) != set(POOL_NAMES):
        raise ValueError(f"ratios must contain exactly {POOL_NAMES}")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to one")
    groups_by_attribute: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        example.validate()
        groups_by_attribute[example.attribute].add(example.group_id)
    manifest: dict[str, str] = {}
    cumulative = []
    running = 0.0
    for pool_name in POOL_NAMES:
        running += ratios[pool_name]
        cumulative.append((pool_name, running))
    for attribute in sorted(groups_by_attribute):
        group_ids = sorted(groups_by_attribute[attribute])
        attribute_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{attribute}".encode("utf-8")).digest()[:8], "big"
        )
        random.Random(attribute_seed).shuffle(group_ids)
        for group_index, group_id in enumerate(group_ids):
            fraction = (group_index + 0.5) / len(group_ids)
            pool_name = next(name for name, threshold in cumulative if fraction <= threshold + 1e-12)
            manifest[f"{attribute}\t{group_id}"] = pool_name
    validate_manifest(examples, manifest)
    return manifest


def validate_manifest(examples: list[BBQExample], manifest: dict[str, str]) -> None:
    observed_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    missing = []
    for example in examples:
        key = f"{example.attribute}\t{example.group_id}"
        if key not in manifest:
            missing.append(key)
            continue
        pool_name = manifest[key]
        if pool_name not in POOL_NAMES:
            raise ValueError(f"invalid pool {pool_name!r} for {key}")
        observed_groups[(example.attribute, example.group_id)].add(pool_name)
    if missing:
        raise ValueError(f"manifest is missing {len(set(missing))} groups")
    expected_keys = {f"{example.attribute}\t{example.group_id}" for example in examples}
    extra = set(manifest) - expected_keys
    if extra:
        raise ValueError(f"manifest contains {len(extra)} groups absent from the dataset")
    leaked = [key for key, pools in observed_groups.items() if len(pools) != 1]
    if leaked:
        raise ValueError(f"group leakage detected for {leaked[:5]}")


def split_examples(
    examples: list[BBQExample], manifest: dict[str, str]
) -> dict[str, list[BBQExample]]:
    validate_manifest(examples, manifest)
    pools = {pool_name: [] for pool_name in POOL_NAMES}
    for example in examples:
        key = f"{example.attribute}\t{example.group_id}"
        pools[manifest[key]].append(example)
    return pools


def save_manifest(path: str | Path, manifest: dict[str, str], overwrite: bool = False) -> None:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite frozen split manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "assignments": dict(sorted(manifest.items())),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported split manifest schema")
    return {str(key): str(value) for key, value in payload["assignments"].items()}


def manifest_digest(manifest: dict[str, str]) -> str:
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
