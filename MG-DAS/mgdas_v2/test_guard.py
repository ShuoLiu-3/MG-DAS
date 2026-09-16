from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dataio import atomic_write_json


class FrozenTestGuard:
    def __init__(self, ledger_path: str | Path) -> None:
        self.ledger_path = Path(ledger_path)

    def _key(self, attribute: str, variant: str, config_identifier: str, split_digest: str) -> str:
        return hashlib.sha256(
            f"{attribute}\t{variant}\t{config_identifier}\t{split_digest}".encode("utf-8")
        ).hexdigest()

    def assert_available(
        self, attribute: str, variant: str, config_identifier: str, split_digest: str
    ) -> None:
        if not self.ledger_path.exists():
            return
        payload = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        key = self._key(attribute, variant, config_identifier, split_digest)
        if key in payload.get("consumed", {}):
            raise RuntimeError(
                f"frozen test evaluation already consumed for {attribute}/{variant}; "
                "refusing a second evaluation"
            )

    def record(
        self, attribute: str, variant: str, config_identifier: str, split_digest: str
    ) -> None:
        key = self._key(attribute, variant, config_identifier, split_digest)
        payload = {"schema_version": 1, "consumed": {}}
        if self.ledger_path.exists():
            payload = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        consumed = payload.setdefault("consumed", {})
        if key in consumed:
            raise RuntimeError(f"duplicate frozen test ledger entry for {attribute}/{variant}")
        consumed[key] = {
            "attribute": attribute,
            "variant": variant,
            "config_identifier": config_identifier,
            "split_digest": split_digest,
        }
        atomic_write_json(self.ledger_path, payload)
