from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .types import BBQExample


@dataclass(frozen=True)
class MultipleChoiceExample:
    example_id: str
    prompt: str
    choices: tuple[str, ...]
    label_index: int

    def validate(self) -> None:
        if len(self.choices) < 2:
            raise ValueError(f"{self.example_id}: at least two choices are required")
        if not 0 <= self.label_index < len(self.choices):
            raise ValueError(f"{self.example_id}: invalid label_index")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            rows.append(value)
    return rows


def load_normalized_bbq(path: str | Path) -> list[BBQExample]:
    examples = []
    required = {
        "example_id",
        "group_id",
        "attribute",
        "context",
        "question",
        "choices",
        "label_index",
        "stereotype_index",
        "anti_stereotype_index",
        "unknown_index",
        "is_ambiguous",
    }
    for row in _read_jsonl(path):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{row.get('example_id', '<unknown>')}: missing fields {sorted(missing)}")
        choices = tuple(str(choice) for choice in row["choices"])
        if len(choices) != 3:
            raise ValueError(f"{row['example_id']}: BBQ requires exactly three choices")
        example = BBQExample(
            example_id=str(row["example_id"]),
            group_id=str(row["group_id"]),
            attribute=str(row["attribute"]),
            context=str(row["context"]),
            question=str(row["question"]),
            choices=choices,
            label_index=int(row["label_index"]),
            stereotype_index=int(row["stereotype_index"]),
            anti_stereotype_index=int(row["anti_stereotype_index"]),
            unknown_index=int(row["unknown_index"]),
            is_ambiguous=bool(row["is_ambiguous"]),
            probe_label=None if row.get("probe_label") is None else int(row["probe_label"]),
            neutral_reference=bool(row.get("neutral_reference", False)),
        )
        example.validate()
        examples.append(example)
    identifiers = [example.example_id for example in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("BBQ example_id values must be unique")
    return examples


def load_multiple_choice(path: str | Path) -> list[MultipleChoiceExample]:
    examples = []
    for row in _read_jsonl(path):
        example = MultipleChoiceExample(
            example_id=str(row["example_id"]),
            prompt=str(row["prompt"]),
            choices=tuple(str(choice) for choice in row["choices"]),
            label_index=int(row["label_index"]),
        )
        example.validate()
        examples.append(example)
    if not examples:
        raise ValueError(f"{path}: multiple-choice dataset is empty")
    identifiers = [example.example_id for example in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{path}: example_id values must be unique")
    return examples


def atomic_write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def bbq_example_to_dict(example: BBQExample) -> dict[str, Any]:
    payload = asdict(example)
    payload["choices"] = list(example.choices)
    return payload
