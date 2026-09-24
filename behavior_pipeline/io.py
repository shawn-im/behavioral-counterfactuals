from __future__ import annotations

"""Small file-format helpers shared by every pipeline stage.

The pipeline deliberately uses plain JSONL split directories as the boundary
between stages. Prompt prep, generation, and training can therefore be edited
independently as long as they preserve train/val/test JSONL files.
"""

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    """Read non-empty JSONL records from path into a list of dictionaries."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write rows as UTF-8 JSONL, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_split_dir(data_dir: str | Path) -> dict[str, list[dict]]:
    """Load the required train/val/test split files."""
    data_dir = Path(data_dir)
    out = {}
    source_owner: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        out[split] = read_jsonl(path)
        for index, row in enumerate(out[split]):
            if "source_id" not in row:
                continue
            source_id = str(row["source_id"])
            previous = source_owner.get(source_id)
            if previous is not None and previous != split:
                raise ValueError(
                    f"source_id {source_id!r} appears in both "
                    f"{previous} and {split}"
                )
            source_owner[source_id] = split
    return out
